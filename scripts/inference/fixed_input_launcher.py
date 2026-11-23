# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Run a method's own script with its VGGT image loader replaced by one that keeps fixed-resolution frames as is.

VGGT-Long and VGGT-SLAM call VGGT's ``load_and_preprocess_images``, which hardcodes a 518 px width, so the SLAM
track's pre-sized 504 px frames (spatialarena.slam_protocol.DATASET_INPUT_SIZE) would be upscaled. This launcher
swaps that one function in the named modules for ``load_fixed_images`` (same output: an [N, 3, H, W] tensor in
[0, 1], RGBA blended onto white) before running the script exactly as ``python script.py args...`` would; nothing
else of the method changes and its repository is not edited.

    python fixed_input_launcher.py --patch vggt.utils.load_fn --record out.json -- /code/main.py --flag ...

``--record`` writes the (width, height) the model received, for the run's ``model_input_size``.

``--dump_vggt_slam_depth DIR`` (VGGT-SLAM only, with or without ``--patch``) also saves the depth and depth
confidence VGGT predicts for every submap frame, as ``DIR/<image stem>.npz`` (``depth``, ``depth_conf``), by wrapping
``vggt_slam.solver.Solver.run_predictions``; the solver itself is unchanged (slam_depth_export.py reads the dumps).
"""

from __future__ import annotations

import argparse
import importlib
import json
import runpy
import sys
from pathlib import Path

PATCH_SIZE = 14  # VGGT's ViT patch


def load_fixed_images(image_path_list,
                      mode: str = "crop",
                      record: Path | None = None):
    """VGGT ``load_and_preprocess_images`` replacement for frames already at the protocol size: no resize, no crop."""
    import torch  # pylint: disable=import-outside-toplevel  # the method's own environment provides it
    from PIL import Image  # pylint: disable=import-outside-toplevel
    from torchvision.transforms import functional as TF  # pylint: disable=import-outside-toplevel

    del mode
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    tensors = []
    for image_path in image_path_list:
        with Image.open(image_path) as image:
            if image.mode == "RGBA":
                background = Image.new("RGBA", image.size, (255, 255, 255, 255))
                image = Image.alpha_composite(background, image)
            tensors.append(TF.to_tensor(image.convert("RGB")))
    shapes = {tuple(tensor.shape[-2:]) for tensor in tensors}
    if len(shapes) != 1:
        raise ValueError(
            f"Fixed-resolution frames must share one size, found {sorted(shapes)}"
        )
    height, width = shapes.pop()
    if height % PATCH_SIZE or width % PATCH_SIZE:
        raise ValueError(
            f"Fixed-resolution frames {width}x{height} are not on the {PATCH_SIZE} px patch grid"
        )
    if record is not None:
        record.write_text(json.dumps({"model_input_size": [width, height]}) +
                          "\n",
                          encoding="utf-8")
    return torch.stack(tensors)


def install_vggt_slam_depth_dump(directory: Path) -> None:
    """Wrap VGGT-SLAM's ``Solver.run_predictions`` to save each submap frame's depth and confidence."""
    import numpy as np  # pylint: disable=import-outside-toplevel
    import vggt_slam.solver as solver_module  # pylint: disable=import-outside-toplevel

    directory.mkdir(parents=True, exist_ok=True)
    original = solver_module.Solver.run_predictions

    def run_predictions(self, image_names, *args, **kwargs):
        predictions = original(self, image_names, *args, **kwargs)
        depth = np.asarray(
            predictions["depth"]
        )  # (S, H, W, 1): the submap's frames, in image_names order
        confidence = np.asarray(predictions["depth_conf"])
        for index, name in enumerate(image_names):
            np.savez(directory / f"{Path(name).stem}.npz",
                     depth=depth[index, ..., 0].astype(np.float32),
                     depth_conf=confidence[index].astype(np.float32))
        return predictions

    solver_module.Solver.run_predictions = run_predictions


def main() -> None:
    """Patch the loaders, then run the script as ``__main__`` with its own arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--patch",
        action="append",
        default=[],
        help="Module whose load_and_preprocess_images is replaced (repeatable)."
    )
    parser.add_argument(
        "--dump_vggt_slam_depth",
        type=Path,
        default=None,
        help="VGGT-SLAM: save each submap frame's depth and confidence here.")
    parser.add_argument("--record", type=Path, default=None)
    parser.add_argument("script", type=Path)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    # Python put this file's directory first on sys.path; its adapters (vggt.py, ...) would shadow the methods'
    # packages of the same name. Replace it with the script's directory, as `python script.py` would have.
    here = Path(__file__).resolve().parent
    sys.path[:] = [
        entry for entry in sys.path if Path(entry or ".").resolve() != here
    ]
    sys.path.insert(0, str(args.script.resolve().parent))

    def loader(image_path_list, mode="crop"):
        return load_fixed_images(image_path_list, mode, record=args.record)

    for module_name in args.patch:
        importlib.import_module(module_name).load_and_preprocess_images = loader
    if args.dump_vggt_slam_depth is not None:
        install_vggt_slam_depth_dump(args.dump_vggt_slam_depth)
    sys.argv = [str(args.script), *args.script_args]
    runpy.run_path(str(args.script), run_name="__main__")


if __name__ == "__main__":
    main()
