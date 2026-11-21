# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Check that every SLAM-track method sees the fixed protocol resolution, unrescaled (CPU, no model weights).

For each method this runs the image preprocessing its adapter actually uses in the full-slam suite
(``FIXED_INPUT_RESOLUTION=1``, ``RESOLUTION=IMAGE_LOAD_RESOLUTION=504``; spatialarena.slam_protocol) -- the
upstream loaders from the pinned checkouts plus the adapter's own settings -- on random frames of each protocol
size (``slam_protocol.DATASET_INPUT_SIZE``: 504x378, 504x252, 504x154), and reports the (width, height) of the
tensor the model would receive and the largest pixel difference from the input. The input must arrive unchanged:
identical size, or a center crop to a 16 px patch grid (CUT3R, Fast3R, VGGT-Omega), or the frame padded onto a
square canvas (VGGT, FastVGGT, OmniVGGT, CoME); in every case the frame's pixels must match to within one 8-bit level.
Each method runs in its own subprocess (several repositories ship packages with the same names).

    python scripts/test/probe_slam_input_sizes.py            # all methods, every protocol size
    python scripts/test/probe_slam_input_sizes.py --methods ZIPMAP R3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
# see spatialarena/paths.py
SPATIALARENA_ROOT = os.path.expanduser(os.environ.get("SPATIALARENA_ROOT", "~/spatialarena"))
CODE_ROOT = Path(os.environ.get("METHOD_CODE_ROOT",
                                f"{SPATIALARENA_ROOT}/benchmarks/_model_codes"))
TOLERANCE = 1.0 / 255.0 + 1e-6


def _write_frames(directory: Path, size: tuple[int, int], count: int = 3) -> list[str]:
    from PIL import Image  # pylint: disable=import-outside-toplevel

    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(size[1])
    paths = []
    for index in range(count):
        path = directory / f"{index:06d}.png"
        Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)).save(path)
        paths.append(str(path))
    return paths


def _to_unit(tensor) -> np.ndarray:
    """[3, H, W] (or [H, W, 3]) image as float in [0, 1]."""
    array = np.asarray(tensor.detach().float().cpu() if hasattr(tensor, "detach") else tensor, dtype=np.float64)
    if array.ndim == 3 and array.shape[-1] == 3 and array.shape[0] != 3:
        array = array.transpose(2, 0, 1)
    if array.min() < -0.01:  # [-1, 1] normalization
        array = (array + 1.0) / 2.0
    elif array.max() > 1.5:  # 0..255
        array = array / 255.0
    return array


def compare(output, source_path: str) -> dict:
    """Size of ``output`` and the max difference of the source frame's pixels inside it (exact, cropped or padded)."""
    from PIL import Image  # pylint: disable=import-outside-toplevel

    out = _to_unit(output)
    src = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.float64).transpose(2, 0, 1) / 255.0
    height, width = out.shape[-2:]
    src_h, src_w = src.shape[-2:]
    if (height, width) == (src_h, src_w):
        kind, diff = "same", np.abs(out - src).max()
    elif height <= src_h and width <= src_w:
        top, left = (src_h - height) // 2, (src_w - width) // 2
        kind, diff = "center_crop", np.abs(out - src[:, top:top + height, left:left + width]).max()
    elif height >= src_h and width >= src_w:
        top, left = (height - src_h) // 2, (width - src_w) // 2
        kind, diff = "padded", np.abs(out[:, top:top + src_h, left:left + src_w] - src).max()
    else:
        kind, diff = "rescaled", float("inf")
    return {"model_input": [int(width), int(height)], "kind": kind, "max_pixel_diff": float(diff)}


# ---------------------------------------------------------------------------------------------- per-method probes
# Each takes the frame paths and their (width, height) and returns one preprocessed image tensor (the first frame).
def probe_pi3(paths, size):
    """pi3.py: the RGB loader at new_width = 504, and the rgb_depth multimodal loader at a W x H pixel budget."""
    import importlib.util  # pylint: disable=import-outside-toplevel

    sys.path[:0] = [str(CODE_ROOT / "Pi3"), str(REPO_ROOT / "scripts" / "inference")]
    from pi3.utils.basic import load_multimodal_data  # pylint: disable=import-outside-toplevel

    # The adapter (scripts/inference/pi3.py) shares its name with the upstream pi3 package: load it by path.
    spec = importlib.util.spec_from_file_location("pi3_adapter", REPO_ROOT / "scripts" / "inference" / "pi3.py")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    rgb = adapter.load_pi3_official_eval_rgb_tensor(paths, new_width=504, device="cpu", verbose=False)[0]
    multimodal, _ = load_multimodal_data(str(Path(paths[0]).parent), None, interval=1,
                                         PIXEL_LIMIT=size[0] * size[1], verbose=False, device="cpu")
    return {"rgb": rgb, "rgb_depth": multimodal.reshape(-1, *multimodal.shape[-3:])[0]}


def probe_vggt(paths, size):
    """vggt.py (VGGT, RobustVGGT): square pad at image_load_resolution 504, then interpolate to resolution 504."""
    del size
    import torch.nn.functional as F  # pylint: disable=import-outside-toplevel

    sys.path.insert(0, str(CODE_ROOT / "RobustVGGT"))
    from vggt.utils.load_fn import load_and_preprocess_images_square  # pylint: disable=import-outside-toplevel

    images, _ = load_and_preprocess_images_square(paths, target_size=504)
    return F.interpolate(images, size=(504, 504), mode="bilinear", align_corners=False)[0]


def probe_fastvggt(paths, size):
    """fastvggt.py: materialize_resized_images (square pad, 504), then a (504, 504) resize."""
    del size
    from PIL import Image  # pylint: disable=import-outside-toplevel
    from torchvision.transforms import functional as TF  # pylint: disable=import-outside-toplevel

    sys.path.insert(0, str(REPO_ROOT / "scripts" / "inference"))
    from adapter_utils import materialize_resized_images  # pylint: disable=import-outside-toplevel

    padded = materialize_resized_images(paths, Path(paths[0]).parent.parent / "fastvggt_pad", 504)
    with Image.open(padded[0]) as image:
        return TF.to_tensor(image.convert("RGB").resize((504, 504), Image.Resampling.BICUBIC))


def probe_come(paths, size):
    """come.py: materialize_resized_images (square pad, 504); Co-Me's src/infer.py then resizes to (504, 504)."""
    del size
    from PIL import Image  # pylint: disable=import-outside-toplevel

    sys.path.insert(0, str(REPO_ROOT / "scripts" / "inference"))
    from adapter_utils import materialize_resized_images  # pylint: disable=import-outside-toplevel

    padded = materialize_resized_images(paths, Path(paths[0]).parent.parent / "come_pad", 504)
    with Image.open(padded[0]) as image:
        assert image.size == (504, 504), image.size  # so Co-Me's resize(504, 504) is the identity
        return np.asarray(image.convert("RGB"))


def probe_omnivggt(paths, size):
    """omnivggt.py: its square pad at image_load_resolution 504, then OmniVGGT's load_images_and_cameras(504)."""
    del size
    # Not scripts/inference on sys.path: its omnivggt.py would shadow the upstream (namespace) package.
    sys.path[:0] = [str(CODE_ROOT / "OmniVGGT-official"), str(REPO_ROOT)]
    from scripts.inference.adapter_utils import materialize_resized_images  # pylint: disable=import-outside-toplevel
    from visual_util import load_images_and_cameras  # pylint: disable=import-outside-toplevel

    padded = materialize_resized_images(paths, Path(paths[0]).parent.parent / "omni_pad", 504)
    images = load_images_and_cameras(str(Path(padded[0]).parent), None, None, 504)[0]
    return images.reshape(-1, *images.shape[-3:])[0]


def probe_iggt(paths, size):
    """iggt.py: IGGT's load_and_preprocess_images in "resize" mode at the frames' own (width, height)."""
    sys.path.insert(0, str(CODE_ROOT / "IGGT_official"))
    from iggt.utils.load_fn import load_and_preprocess_images  # pylint: disable=import-outside-toplevel

    return load_and_preprocess_images(paths, mode="resize", resize_target_size=size)[0]


def probe_vggt_omega(paths, size):
    """vggt_omega.py: load_patch_grid_tensor (center crop to the 16 px grid)."""
    del size
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "inference"))
    from adapter_utils import load_patch_grid_tensor  # pylint: disable=import-outside-toplevel

    return load_patch_grid_tensor(paths, 16)[0]


def probe_cut3r(paths, size):
    """cut3r.py: CUT3R's DUSt3R load_images at size 504 (long edge, then a crop to 16 px)."""
    del size
    sys.path.insert(0, str(CODE_ROOT / "CUT3R"))
    from src.dust3r.utils.image import load_images  # pylint: disable=import-outside-toplevel

    return load_images(paths, size=504, verbose=False)[0]["img"][0]


def probe_fast3r(paths, size):
    """fast3r.py: Fast3R's DUSt3R load_images at size 504."""
    del size
    sys.path.insert(0, str(CODE_ROOT / "fast3r"))
    from fast3r.dust3r.utils.image import load_images  # pylint: disable=import-outside-toplevel

    return load_images(paths, size=504, verbose=False)[0]["img"][0]


def probe_da3(paths, size):
    """da3.py: DA3's InputProcessor at process_res 504, upper_bound_resize (its default)."""
    del size
    sys.path.insert(0, str(CODE_ROOT / "depth-anything-3" / "src"))
    from depth_anything_3.utils.io.input_processor import InputProcessor  # pylint: disable=import-outside-toplevel

    tensor = InputProcessor()(paths, process_res=504, process_res_method="upper_bound_resize", num_workers=1)[0]
    image = tensor.reshape(-1, *tensor.shape[-3:])[0]
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]  # undo its ImageNet normalization
    std = np.array([0.229, 0.224, 0.225])[:, None, None]
    return image.detach().float().cpu().numpy() * std + mean


def probe_amb3r(paths, size):
    """amb3r.py: AMB3R's Demo dataset at the frames' own resolution."""
    sys.path[:0] = [str(CODE_ROOT / "amb3r"), str(CODE_ROOT / "amb3r" / "thirdparty")]  # as amb3r/model.py does
    from amb3r.datasets import Demo  # pylint: disable=import-outside-toplevel

    data = Demo(ROOT=str(Path(paths[0]).parent), resolution=size, num_seq=1, full_video=True, kf_every=1,
                disable_crop=False, max_images=len(paths))
    views, _views_all = data[0]  # (per-view dicts, the stacked batch)
    return views[0]["img"]


def probe_loger(paths, size):
    """streaming_slam.run_loger: LoGeR's _load_images_basic at the frames' own (width, height)."""
    sys.path.insert(0, str(CODE_ROOT / "LoGeR"))
    from eval.relpose.launch import _load_images_basic  # pylint: disable=import-outside-toplevel

    return _load_images_basic(paths, size=size, crop=True)[0]["img"]


def probe_zipmap(paths, size):
    """streaming_slam.run_zipmap: ZipMap's load_and_preprocess_images with target_size = the frames' width."""
    sys.path[:0] = [str(CODE_ROOT / "ZipMap"), str(CODE_ROOT / "ZipMap" / "zipmap")]
    from zipmap.utils.load_fn import load_and_preprocess_images  # pylint: disable=import-outside-toplevel

    return load_and_preprocess_images(paths, target_size=size[0])[0]


def probe_lingbot_map(paths, size):
    """streaming_slam.run_lingbot_map: fixed inputs skip the area-budget resize (the frame as loaded)."""
    del size
    from PIL import Image  # pylint: disable=import-outside-toplevel

    return np.asarray(Image.open(paths[0]).convert("RGB"))


def probe_r3(paths, size):
    """streaming_slam.run_r3: R3's load_images at --size = the long edge, 14 px patches."""
    sys.path.insert(0, str(CODE_ROOT / "R3"))
    from R3.utils.input_io import load_images  # pylint: disable=import-outside-toplevel

    return load_images(paths, size=max(size), verbose=False, patch_size=14)[0]["img"][0]


def probe_scal3r(paths, size):
    """streaming_slam.run_scal3r: Scal3R's determine_target_size + resize_to_cover at proc_max_size = width."""
    import torch  # pylint: disable=import-outside-toplevel
    from PIL import Image  # pylint: disable=import-outside-toplevel

    sys.path.insert(0, str(CODE_ROOT / "Scal3R"))
    from scal3r.utils import image_utils  # pylint: disable=import-outside-toplevel

    image = torch.from_numpy(np.asarray(Image.open(paths[0]).convert("RGB"), dtype=np.float32) / 255.0)
    target_h, target_w = image_utils.determine_target_size(
        image.shape[0], image.shape[1], image_utils.dotdict(proc_max_size=size[0], proc_align_size=14))
    mask = torch.ones(image.shape[:2])
    ixt = torch.eye(3)
    out = image_utils.resize_to_cover(image, mask, ixt, target_h, target_w, 14)[0]
    return out


def probe_abot_recon(paths, size):
    """streaming_slam.run_abot_recon: ABot-Recon's preprocess_image with width/height = the frames' size."""
    from PIL import Image  # pylint: disable=import-outside-toplevel

    sys.path.insert(0, str(CODE_ROOT / "ABot-Recon"))
    from abot_recon.preprocessing import preprocess_image  # pylint: disable=import-outside-toplevel

    with Image.open(paths[0]) as image:
        result = preprocess_image(image.convert("RGB"), height=size[1], width=size[0])
    return getattr(result, "tensor", result[0] if isinstance(result, tuple) else result)


def probe_vggt_long(paths, size):
    """long_sequence_slam (VGGT-Long, VGGT-SLAM): fixed_input_launcher.load_fixed_images."""
    del size
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "inference"))
    from fixed_input_launcher import load_fixed_images  # pylint: disable=import-outside-toplevel

    return load_fixed_images(paths)[0]


def probe_pi_long(paths, size):
    """long_sequence_slam (Pi-Long): load_images_as_tensor_pi_long at image_width = the frames' width."""
    sys.path.insert(0, str(CODE_ROOT / "Pi-Long"))
    from pi3.utils.basic import load_images_as_tensor_pi_long  # pylint: disable=import-outside-toplevel

    return load_images_as_tensor_pi_long(paths, image_width=size[0])[0]


PROBES = {
    "PI3_X": probe_pi3, "VGGT": probe_vggt, "ROBUSTVGGT": probe_vggt, "FASTVGGT": probe_fastvggt,
    "COME": probe_come, "OMNIVGGT": probe_omnivggt, "IGGT": probe_iggt, "VGGT_OMEGA": probe_vggt_omega,
    "CUT3R": probe_cut3r, "FAST3R": probe_fast3r, "DA3": probe_da3, "AMB3R": probe_amb3r, "LOGER": probe_loger,
    "ZIPMAP": probe_zipmap, "LINGBOT_MAP": probe_lingbot_map, "R3": probe_r3, "SCAL3R": probe_scal3r,
    "ABOT_RECON": probe_abot_recon, "VGGT_LONG": probe_vggt_long, "VGGT_SLAM": probe_vggt_long,
    "PI_LONG": probe_pi_long,
}


def run_one(method: str, width: int, height: int, workdir: Path) -> dict:
    """Probe one method at one size (in this process)."""
    paths = _write_frames(workdir / "images", (width, height))
    output = PROBES[method](paths, (width, height))
    outputs = output if isinstance(output, dict) else {"": output}
    return {f"{method}{'/' + key if key else ''}": compare(tensor, paths[0]) for key, tensor in outputs.items()}


def protocol_sizes() -> list[tuple[int, int]]:
    """The distinct (width, height) inputs of the SLAM datasets, largest first."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from spatialarena.slam_protocol import DATASET_INPUT_SIZE  # pylint: disable=import-outside-toplevel

    return sorted(set(DATASET_INPUT_SIZE.values()), key=lambda size: -size[1])


def main() -> None:
    """Probe every method at every protocol size, each in a subprocess; print a table and fail on a rescale."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="*", default=list(PROBES))
    parser.add_argument("--one", nargs=3, metavar=("METHOD", "WIDTH", "HEIGHT"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.one:
        with tempfile.TemporaryDirectory() as workdir:
            print(json.dumps(run_one(args.one[0], int(args.one[1]), int(args.one[2]), Path(workdir))))
        return
    failures = 0
    for method in args.methods:
        for width, height in protocol_sizes():
            result = subprocess.run([sys.executable, __file__, "--one", method, str(width), str(height)],
                                    capture_output=True, text=True, check=False)
            last = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
            if result.returncode != 0 or not last.startswith("{"):
                failures += 1
                print(f"{method:12} {width}x{height}  ERROR  {result.stderr.strip().splitlines()[-1:]}")
                continue
            for name, row in json.loads(last).items():
                ok = row["kind"] != "rescaled" and row["max_pixel_diff"] <= TOLERANCE
                failures += not ok
                print(f"{name:20} {width}x{height} -> {row['model_input'][0]}x{row['model_input'][1]} "
                      f"{row['kind']:12} max diff {row['max_pixel_diff']:.2e}  {'ok' if ok else 'FAIL'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
