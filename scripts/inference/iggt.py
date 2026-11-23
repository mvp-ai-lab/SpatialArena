# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Run IGGT and export benchmark prediction artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

from adapter_utils import (
    SpatialTransform,
    canonical_image_root,
    mark_model_loaded,
    add_code_path,
    export_camera_pose_priors,
    export_spatial_depth_predictions,
    export_standard_artifacts,
    find_image_paths,
    fixed_input_size,
    fold_single_view_outputs,
    normalize_depths,
    normalize_points,
    prepare_model_input,
    print_dry_run,
    record_model_input,
    register_preprocessor,
    restore_camera_export,
    select_interval,
    single_view_mode,
    write_adapter_metrics,
)


def preprocess_iggt(image_paths: list[str],
                    *,
                    args,
                    device=None,
                    load_and_preprocess_images=None,
                    **_unused):
    """IGGT's preprocess_<method>(), registered below for prepare_model_input("IGGT", ...).

    Trusts IGGT's own vendored load_and_preprocess_images(mode="resize") directly (the same
    "trust the vendored resize" pattern as da3.py/pi3.py): confirmed by reading it that this mode
    is a pure stretch to a square canvas, ignoring aspect ratio -- no crop, no pad, so restoring
    to each image's true source resolution afterward is the plain-resize degenerate case
    restore_prediction/restore_camera_export already handle (same as pi3.py/depthvlm.py).
    Fixed-resolution inputs (full-slam) keep their own (width, height); otherwise the adapter's
    square resize. Returns (images_tensor, transforms).
    """
    from PIL import Image

    images = load_and_preprocess_images(
        image_paths,
        mode="resize",
        resize_target_size=fixed_input_size(image_paths) or
        (args.image_load_resolution, args.image_load_resolution),
    ).to(device)
    resized_wh = (images.shape[-1], images.shape[-2])
    transforms: list[SpatialTransform] = []
    for image_path in image_paths:
        with Image.open(image_path) as probe:
            source_wh = probe.size
        transforms.append(
            SpatialTransform(source_wh=source_wh, resized_wh=resized_wh))
    return images, transforms


register_preprocessor("IGGT", preprocess_iggt)

METHOD = "IGGT"
LOGGER = logging.getLogger("benchmark_iggt")

# Shared method checkouts (the engine's METHOD_CODE_ROOT); the engine also passes the *_CODE_PATH below.
# see spatialarena/paths.py
SPATIALARENA_ROOT = os.path.expanduser(
    os.environ.get("SPATIALARENA_ROOT", "~/spatialarena"))
METHOD_CODE_ROOT = Path(
    os.environ.get("METHOD_CODE_ROOT",
                   f"{SPATIALARENA_ROOT}/benchmarks/_model_codes"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--ckpt_path",
                        default=os.environ.get("IGGT_CKPT_PATH", ""))
    parser.add_argument(
        "--iggt_code_path",
        type=Path,
        default=Path(os.environ["IGGT_CODE_PATH"]) if "IGGT_CODE_PATH"
        in os.environ else METHOD_CODE_ROOT / "IGGT_official",
    )
    parser.add_argument("--track", choices=("rgb",), default="rgb")
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=504)
    parser.add_argument("--image_load_resolution", type=int, default=504)
    parser.add_argument("--max_points", type=int, default=1_000_000)
    parser.add_argument("--max_images_per_run", type=int, default=0)
    parser.add_argument("--conf_thresh", type=float, default=0.0)
    parser.add_argument("--depth_conf_thresh", type=float, default=None)
    parser.add_argument("--export_depth", action="store_true")
    parser.add_argument("--export_colmap", action="store_true")
    parser.add_argument("--depth_only", action="store_true")
    parser.add_argument("--colmap_model_subdir", default="0")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def extrinsics_3x4_to_4x4(extrinsics: np.ndarray) -> np.ndarray:
    if extrinsics.ndim == 4 and extrinsics.shape[0] == 1:
        extrinsics = extrinsics[0]
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics.astype(np.float32, copy=False)
    if extrinsics.shape[-2:] != (3, 4):
        raise ValueError(
            f"Expected IGGT extrinsics Nx3x4 or Nx4x4, got {extrinsics.shape}")
    mats = np.repeat(np.eye(4, dtype=np.float32)[None],
                     extrinsics.shape[0],
                     axis=0)
    mats[:, :3, :] = extrinsics.astype(np.float32, copy=False)
    return mats


def resized_colors(image_paths: list[str], height: int,
                   width: int) -> np.ndarray:
    from PIL import Image, ImageOps

    colors = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB").resize(
                (width, height), Image.Resampling.BICUBIC)
            colors.append(np.asarray(image, dtype=np.uint8))
    return np.stack(colors, axis=0)


def candidate_checkpoint_files(root: Path) -> Iterable[Path]:
    suffixes = (".pth", ".pt", ".bin", ".safetensors")
    for suffix in suffixes:
        yield from sorted(root.rglob(f"*{suffix}"))


def resolve_checkpoint_path(checkpoint: str) -> Path:
    """Resolve a local checkpoint path, local directory, or HF repo id."""

    path = Path(checkpoint).expanduser()
    if path.is_file():
        return path
    if path.is_dir():
        for candidate in candidate_checkpoint_files(path):
            return candidate
        raise FileNotFoundError(f"No checkpoint file found under {path}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            f"IGGT checkpoint {checkpoint!r} is not a local file/directory and "
            "huggingface_hub is unavailable for HF repo resolution.") from exc

    snapshot_dir = Path(snapshot_download(repo_id=checkpoint,
                                          repo_type="model"))
    for candidate in candidate_checkpoint_files(snapshot_dir):
        return candidate
    raise FileNotFoundError(
        f"No checkpoint file found in downloaded HF snapshot: {snapshot_dir}")


def load_checkpoint_state_dict(checkpoint_path: Path, device: str):
    import torch

    if checkpoint_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError(
                "safetensors is required to load IGGT safetensors checkpoints"
            ) from exc
        return load_file(str(checkpoint_path), device=device)
    return torch.load(str(checkpoint_path),
                      map_location=device,
                      weights_only=False)


def load_iggt_state_dict_aligner(code_path: Path):
    module_path = code_path / "utils" / "model.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            f"IGGT state-dict helper is missing: {module_path}")
    spec = importlib.util.spec_from_file_location("benchmark_iggt_utils_model",
                                                  module_path)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load IGGT state-dict helper from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.align_and_update_state_dicts


def run() -> None:
    args = parse_args()
    images_dir = canonical_image_root(args.data_path)
    image_paths = select_interval(find_image_paths(images_dir), args.interval,
                                  args.max_images_per_run)
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")
    if args.dry_run:
        print_dry_run(args, METHOD, len(image_paths))
        return
    if not args.ckpt_path:
        raise ValueError("IGGT requires --ckpt_path or IGGT_CKPT_PATH.")

    add_code_path(args.iggt_code_path, ("iggt",))
    code_path = Path(args.iggt_code_path).expanduser().resolve()
    nested = code_path / "iggt"
    for candidate in (nested, code_path):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    import torch
    from PIL import Image
    from iggt.models.vggt import IGGT
    from iggt.utils.load_fn import load_and_preprocess_images
    from iggt.utils.pose_enc import pose_encoding_to_extri_intri
    from iggt.utils.geometry import unproject_depth_map_to_point_map
    align_and_update_state_dicts = load_iggt_state_dict_aligner(code_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = IGGT()
    checkpoint_path = resolve_checkpoint_path(args.ckpt_path)
    state_dict = load_checkpoint_state_dict(checkpoint_path, device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    state_dict = {
        key.replace("module.", "", 1): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(align_and_update_state_dicts(LOGGER,
                                                       model.state_dict(),
                                                       state_dict),
                          strict=False)
    if hasattr(model, "part_adaptor"):
        model.part_adaptor = None
    if hasattr(model, "part_head"):
        model.part_head = None
    model = model.to(device).eval()

    started_at = mark_model_loaded()
    images, transforms = prepare_model_input(
        "IGGT",
        image_paths,
        args=args,
        device=device,
        load_and_preprocess_images=load_and_preprocess_images)
    record_model_input(images.shape[-1], images.shape[-2])
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability(
    )[0] >= 8 else torch.float16
    single_view = single_view_mode()
    with torch.no_grad():
        with torch.amp.autocast(
                "cuda", dtype=dtype) if device == "cuda" else torch.no_grad():
            if single_view:
                # MonoDepth: independent single-view samples [B, 1, 3, H, W], folded back to the sequence layout.
                predictions = fold_single_view_outputs(model(images[:, None]),
                                                       images.shape[0])
            else:
                predictions = model(images)
    pose_enc = predictions["pose_enc"][-1]
    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        pose_enc, images.shape[-2:])
    depths = normalize_depths(
        predictions["depth"].detach().float().cpu().numpy().squeeze(0))
    extrinsics_w2c = extrinsics_3x4_to_4x4(
        extrinsics.detach().float().cpu().numpy())
    intrinsics_np = intrinsics.detach().float().cpu().numpy().squeeze(0).astype(
        np.float32)
    confidence = predictions.get("depth_conf")
    if confidence is not None:
        confidence = confidence.detach().float().cpu().numpy().squeeze()

    if args.depth_only:
        # Restore depth/camera predictions to each image's true source resolution -- the same
        # restore_prediction/restore_camera_export machinery already proven on
        # vggt.py/fastvggt.py/da3.py/vggt_omega.py/cut3r.py/etc.
        if args.export_depth:
            export_spatial_depth_predictions(args.output_path, image_paths,
                                             depths, confidence, transforms)
        if args.export_colmap:
            restored_intrinsics, width, height = restore_camera_export(
                intrinsics_np, transforms)
            export_camera_pose_priors(
                args.output_path,
                images_dir,
                image_paths,
                extrinsics_w2c,
                restored_intrinsics,
                width=width,
                height=height,
                colmap_model_subdir=args.colmap_model_subdir,
            )
        write_adapter_metrics(
            output_path=args.output_path,
            method=METHOD,
            image_paths=image_paths,
            started_at=started_at,
            checkpoint=str(checkpoint_path),
            depth_only=True,
            metrics_extra={
                "track": args.track,
                "single_view": single_view
            },
        )
        return

    # Dense/points3D export derives pixel-indexed geometry from the model's own resized grid --
    # not restored to source resolution, same restriction every other patched adapter's "shared"
    # policy has for non-depth-only export.
    points = normalize_points(
        unproject_depth_map_to_point_map(depths[..., None], extrinsics_w2c,
                                         intrinsics_np))
    colors = resized_colors(image_paths, depths.shape[1], depths.shape[2])
    export_standard_artifacts(
        method=METHOD,
        output_path=args.output_path,
        images_dir=images_dir,
        image_paths=image_paths,
        depths=depths,
        points=points,
        colors=colors,
        confidence=confidence,
        extrinsics_w2c=extrinsics_w2c,
        intrinsics=intrinsics_np,
        export_depth=args.export_depth,
        export_colmap=args.export_colmap,
        depth_only=args.depth_only,
        colmap_model_subdir=args.colmap_model_subdir,
        max_points=args.max_points,
        conf_thresh=args.depth_conf_thresh
        if args.depth_conf_thresh is not None else args.conf_thresh,
        started_at=started_at,
        checkpoint=str(checkpoint_path),
        metrics_extra={
            "track": args.track,
            "single_view": single_view
        },
    )


if __name__ == "__main__":
    run()
