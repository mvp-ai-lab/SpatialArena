#!/usr/bin/env python
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Run VGGT and export benchmark prediction artifacts.

This is an external-method adapter used to produce test outputs for the
artifact-only benchmark evaluator. Evaluation code must read the files produced
here; it should not import or execute VGGT directly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Fully-qualified (not a bare `from adapter_utils import ...`, unlike most
# other adapters): this file is also imported as scripts.inference.vggt by
# test/test_model_adapters.py, where only ROOT_DIR (added above) is on
# sys.path, not this script's own directory -- Python only auto-adds a
# script's own directory for direct execution (`python vggt.py`), not for
# package-style imports. Imported at module load time, before
# add_vggt_code_path() (called later, inside run()) removes this script's
# own directory from sys.path to avoid this adapter
# (scripts/inference/vggt.py) shadowing the vendored `vggt` package it
# wraps -- by then adapter_utils is already cached in sys.modules, so that
# later path surgery doesn't affect this import either way.
from scripts.inference.adapter_utils import (  # noqa: E402
    save_prediction_map,
    canonical_image_root,
    export_depths_for_policy,
    find_image_paths,
    fold_single_view_outputs,
    mark_model_loaded,
    legacy_or_shared_tensor,
    prepare_model_input,
    register_preprocessor,
    record_model_input,
    restore_camera_export,
    runtime_metrics,
    save_confidence_map,
    select_interval,
    single_view_mode,
    spatial_policy_metrics,
    validate_shared_spatial_policy_depth_only,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VGGT and export benchmark prediction artifacts."
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Scene root containing an images/ directory.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Method scene output directory.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=os.environ.get("VGGT_CKPT_PATH", "facebook/VGGT-1B"),
        help="Hugging Face model id or local VGGT checkpoint path.",
    )
    parser.add_argument(
        "--vggt_code_path",
        type=Path,
        default=Path(os.environ["VGGT_CODE_DIR"]) if "VGGT_CODE_DIR" in os.environ else None,
        help="Optional checkout of facebookresearch/vggt to add to PYTHONPATH.",
    )
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=518)
    parser.add_argument("--image_load_resolution", type=int, default=1024)
    parser.add_argument(
        "--max_images_per_run",
        type=int,
        default=0,
        help="Maximum images per model call for depth-only full evaluation.",
    )
    parser.add_argument("--max_points", type=int, default=1_000_000)
    parser.add_argument("--conf_thresh", type=float, default=5.0)
    parser.add_argument("--depth_conf_thresh", type=float, default=None)
    parser.add_argument("--export_depth", action="store_true")
    parser.add_argument("--export_colmap", action="store_true")
    parser.add_argument(
        "--depth_only",
        action="store_true",
        help="Only export depth predictions; skip dense point cloud and COLMAP artifacts.",
    )
    parser.add_argument("--colmap_model_subdir", type=str, default="0")
    parser.add_argument("--allow_partial_checkpoint", action="store_true")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Report planned inputs and outputs without importing VGGT or loading weights.",
    )
    parser.add_argument(
        "--spatial_policy",
        choices=("legacy", "shared"),
        default="legacy",
        help=(
            "'legacy' (default): unchanged two-stage resize (load_and_preprocess_images_square to "
            "--image_load_resolution, then F.interpolate to --resolution), exported depth only "
            "cropped free of padding (crop_to_original_image_region), not restored to true source "
            "resolution. 'shared': a single direct pad-to-square resize straight to --resolution via "
            "adapter_utils.plan_spatial_transform(method='vggt'), with depth exported at the true "
            "original per-image resolution via restore_prediction/export_spatial_depth_predictions."
        ),
    )
    return parser.parse_args()


def planned_layout(args: argparse.Namespace, num_images: int) -> dict:
    return {
        "method": "VGGT",
        "data_path": str(args.data_path),
        "output_path": str(args.output_path),
        "num_images": num_images,
        "checkpoint": args.ckpt_path,
        "metrics": str(args.output_path / "metrics.json"),
        "dense_points": None if args.depth_only else str(args.output_path / "points3D_dense.ply"),
        "depths": str(args.output_path / "depths") if args.export_depth else None,
        "sparse": (
            str(args.output_path / "sparse" / args.colmap_model_subdir)
            if args.export_colmap and not args.depth_only
            else None
        ),
    }


def add_vggt_code_path(path: Path | None) -> None:
    script_dir = str(Path(__file__).resolve().parent)
    sys.path[:] = [
        entry
        for entry in sys.path
        if str(Path(entry or ".").resolve()) != script_dir
    ]
    shadowed_module = sys.modules.get("vggt")
    if shadowed_module is not None and not hasattr(shadowed_module, "__path__"):
        del sys.modules["vggt"]
    if path is None:
        return
    code_path = Path(path).expanduser().resolve()
    if str(code_path) not in sys.path:
        sys.path.insert(0, str(code_path))


def load_model(args: argparse.Namespace, device: str):
    import torch
    from vggt.models.vggt import VGGT

    ckpt_path = Path(args.ckpt_path).expanduser()
    if ckpt_path.is_file():
        model = VGGT()
        state = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(state, dict):
            state = state.get("model", state.get("state_dict", state))
        model.load_state_dict(
            state,
            strict=not args.allow_partial_checkpoint,
        )
    else:
        model = VGGT.from_pretrained(args.ckpt_path)

    model.eval()
    return model.to(device)


def legacy_load_vggt_images(image_paths: list[str], args: argparse.Namespace, device: str):
    """VGGT's own historical two-stage resize (unchanged by --spatial_policy): pad-to-square via
    load_and_preprocess_images_square to --image_load_resolution, then F.interpolate to
    --resolution -- not just equivalent to a single direct resize straight to --resolution (the
    "shared" policy's approach), strictly worse (two lossy interpolations instead of one), kept
    only so --spatial_policy legacy (default) stays byte-for-byte unchanged."""
    import torch.nn.functional as F
    from vggt.utils.load_fn import load_and_preprocess_images_square

    images, original_coords = load_and_preprocess_images_square(
        image_paths,
        target_size=args.image_load_resolution,
    )
    images = F.interpolate(
        images.to(device),
        size=(args.resolution, args.resolution),
        mode="bilinear",
        align_corners=False,
    )
    return images, original_coords


def preprocess_vggt(image_paths: list[str], *, args: argparse.Namespace, device: str, **_unused):
    """VGGT's preprocess_<method>(), registered below for prepare_model_input("VGGT", ...).

    --spatial_policy legacy (default): legacy_load_vggt_images's two-stage resize, unchanged.
    --spatial_policy shared: one pad-to-square resize straight to --resolution, restorable.
    Returns ((images, original_coords), transforms) -- original_coords is only used by VGGT's own
    legacy crop_to_original_image_region export, None under "shared".
    """
    images, original_coords, transforms = legacy_or_shared_tensor(
        image_paths,
        method="vggt",
        args=args,
        device=device,
        legacy_loader=lambda paths: legacy_load_vggt_images(paths, args, device),
    )
    return (images, original_coords), transforms


register_preprocessor("VGGT", preprocess_vggt)


def run_vggt(model, images: "torch.Tensor", device: str, single_view: bool = False):
    """Run VGGT on a preprocessed images tensor ([N, 3, H, W], already on ``device``) as one
    multi-view sequence, or with ``single_view`` as independent single-image samples
    ([B, 1, 3, H, W]); outputs use the sequence layout either way."""
    import torch
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    if device == "cuda":
        dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )
        autocast_context = torch.amp.autocast("cuda", dtype=dtype)
    else:
        autocast_context = nullcontext()

    with torch.no_grad():
        with autocast_context:
            batched_images = images[:, None] if single_view else images[None]
            predictions = model(batched_images)
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            if single_view:
                predictions = fold_single_view_outputs(predictions, images.shape[0])
            if not isinstance(predictions, dict):
                raise TypeError(
                    f"VGGT-compatible model must return a prediction dict or "
                    f"(prediction_dict, ...); got {type(predictions).__name__}"
                )
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                predictions["pose_enc"],
                batched_images.shape[-2:],
            )
            depth_map = predictions["depth"]
            depth_conf = predictions["depth_conf"]
            points_3d = predictions["world_points"]

    extrinsic = extrinsic.squeeze(0).detach().cpu().numpy()
    intrinsic = intrinsic.squeeze(0).detach().cpu().numpy()
    depth_map = depth_map.squeeze(0).detach().cpu().numpy()
    depth_conf = depth_conf.squeeze(0).detach().cpu().numpy()
    points_3d = points_3d.squeeze(0).detach().cpu().numpy()
    if depth_map.ndim == 4 and depth_map.shape[-1] == 1:
        depth_map = depth_map[..., 0]
    if depth_conf.ndim == 4 and depth_conf.shape[-1] == 1:
        depth_conf = depth_conf[..., 0]

    colors = (images.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    colors = colors.transpose(0, 2, 3, 1)
    return extrinsic, intrinsic, depth_map, depth_conf, points_3d, colors


def sample_points(
    points_3d: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    conf_thresh: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    from spatialarena.geometry.utils import randomly_limit_trues

    finite_mask = np.isfinite(points_3d).all(axis=-1)
    conf_mask = finite_mask & (confidence >= conf_thresh)
    if max_points > 0:
        conf_mask = randomly_limit_trues(conf_mask, max_points)
    return points_3d[conf_mask], colors[conf_mask]


def crop_to_original_image_region(
    depth: np.ndarray,
    original_coord: np.ndarray | None,
    padded_size: int,
) -> np.ndarray:
    if original_coord is None:
        return depth

    x1, y1, x2, y2, _, _ = [float(value) for value in original_coord]
    height, width = depth.shape[:2]
    scale_x = width / float(padded_size)
    scale_y = height / float(padded_size)
    left = max(0, min(int(np.floor(x1 * scale_x)), width - 1))
    top = max(0, min(int(np.floor(y1 * scale_y)), height - 1))
    right = max(left + 1, min(int(np.ceil(x2 * scale_x)), width))
    bottom = max(top + 1, min(int(np.ceil(y2 * scale_y)), height))
    return depth[top:bottom, left:right]


def export_depths(
    output_path: Path,
    image_paths: list[str],
    depths: np.ndarray,
    original_coords: np.ndarray | None = None,
    padded_size: int = 518,
    confidence: np.ndarray | None = None,
) -> None:
    """Per-frame float32 depth (padding cropped off), and the same crop of ``confidence`` (float16) when given; both
    compressed .npz."""
    depth_dir = output_path / "depths"
    depth_dir.mkdir(parents=True, exist_ok=True)
    if confidence is not None and np.shape(confidence)[:3] != np.shape(depths)[:3]:
        raise ValueError(f"confidence {np.shape(confidence)} does not match depth {np.shape(depths)}")
    for index, (image_path, depth) in enumerate(zip(image_paths, depths)):
        original_coord = None if original_coords is None else original_coords[index]
        depth = crop_to_original_image_region(depth, original_coord, padded_size)
        depth = depth.astype(np.float32, copy=False)
        depth_stem = Path(image_path).stem
        save_prediction_map(depth_dir, depth_stem, depth)
        if confidence is not None:
            save_confidence_map(output_path, image_path,
                                crop_to_original_image_region(confidence[index], original_coord, padded_size))


def run() -> None:
    args = parse_args()
    validate_shared_spatial_policy_depth_only(is_shared=args.spatial_policy == "shared", depth_only=args.depth_only)
    images_dir = canonical_image_root(args.data_path)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images directory not found: {images_dir}")

    image_paths = select_interval(find_image_paths(images_dir), args.interval, args.max_images_per_run)
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")

    if args.dry_run:
        print(json.dumps(planned_layout(args, len(image_paths)), indent=2))
        return

    add_vggt_code_path(args.vggt_code_path)

    import torch
    from spatialarena.geometry.utils import write_colmap_formats
    from spatialarena.utils.point_utils import export_ply

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    args.output_path.mkdir(parents=True, exist_ok=True)
    model = load_model(args, device)

    time_start = mark_model_loaded()
    if args.depth_only:
        if not args.export_depth:
            raise ValueError("--depth_only requires --export_depth")
        images_per_run = args.max_images_per_run if args.max_images_per_run > 0 else len(image_paths)
        single_view = single_view_mode()
        pose_extrinsics: list[np.ndarray] = []
        pose_intrinsics: list[np.ndarray] = []
        colmap_wh: tuple[int, int] | None = None
        for start in range(0, len(image_paths), images_per_run):
            batch_paths = image_paths[start:start + images_per_run]
            print(
                f"Depth-only batch {start // images_per_run + 1}: "
                f"{len(batch_paths)} images" + (" (independent single views)" if single_view else "")
            )
            (images, original_coords), transforms = prepare_model_input(
                "VGGT", batch_paths, args=args, device=device
            )
            record_model_input(images.shape[-1], images.shape[-2])
            extrinsic, intrinsic, depths, depth_conf, _, _ = run_vggt(
                model,
                images,
                device,
                single_view=single_view,
            )
            export_depths_for_policy(
                args.output_path,
                batch_paths,
                depths,
                depth_conf,
                transforms,
                legacy_export=lambda: export_depths(
                    args.output_path,
                    batch_paths,
                    depths,
                    original_coords=original_coords,
                    padded_size=args.image_load_resolution,
                    confidence=depth_conf,
                ),
            )
            if args.export_colmap:
                pose_extrinsics.append(np.asarray(extrinsic))
                if transforms is not None:
                    restored_intrinsic, width, height = restore_camera_export(intrinsic, transforms)
                    pose_intrinsics.append(restored_intrinsic)
                    colmap_wh = (width, height)
                else:
                    pose_intrinsics.append(np.asarray(intrinsic))
                    colmap_wh = (args.resolution, args.resolution)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if pose_extrinsics:
            # MVRecon evaluation fuses depth with these poses itself; skip
            # the dense point cloud (empty points3D) that a full VGGT export
            # would otherwise require.
            colmap_dir = args.output_path / "sparse" / args.colmap_model_subdir
            colmap_dir.mkdir(parents=True, exist_ok=True)
            width, height = colmap_wh
            write_colmap_formats(
                str(images_dir),
                np.concatenate(pose_extrinsics, axis=0),
                np.concatenate(pose_intrinsics, axis=0),
                image_paths,
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
                width=width,
                height=height,
                output_path=str(colmap_dir),
            )

        render_time = time.time() - time_start
        runtime = runtime_metrics(time_start)
        peak_memory = (
            torch.cuda.max_memory_allocated() / (1024.0**2)
            if torch.cuda.is_available()
            else 0.0
        )
        metrics = {
            "method": "VGGT",
            "Scene": str(args.data_path),
            "Preprocessed Images": len(image_paths),
            "inference time": render_time,
            **runtime,
            "Peak GPU Memory": peak_memory,
            "checkpoint": args.ckpt_path,
            "depth_only": True,
            **spatial_policy_metrics(args.spatial_policy, is_shared=args.spatial_policy == "shared"),
            "max_images_per_run": images_per_run,
            "single_view": single_view,
        }
        (args.output_path / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n",
            encoding="utf-8",
        )
        return

    (images, original_coords), transforms = prepare_model_input("VGGT", image_paths, args=args, device=device)
    record_model_input(images.shape[-1], images.shape[-2])
    extrinsic, intrinsic, depths, confidence, points_3d, colors = run_vggt(
        model,
        images,
        device,
    )
    render_time = time.time() - time_start
    runtime = runtime_metrics(time_start)
    peak_memory = (
        torch.cuda.max_memory_allocated() / (1024.0**2)
        if torch.cuda.is_available()
        else 0.0
    )

    conf_thresh = (
        args.depth_conf_thresh
        if args.depth_conf_thresh is not None
        else args.conf_thresh
    )
    sampled_points, sampled_colors = sample_points(
        points_3d,
        colors,
        confidence,
        conf_thresh=conf_thresh,
        max_points=args.max_points,
    )

    export_ply(str(args.output_path / "points3D_dense.ply"), sampled_points, sampled_colors)

    if args.export_depth:
        export_depths_for_policy(
            args.output_path,
            image_paths,
            depths,
            confidence,
            transforms,
            legacy_export=lambda: export_depths(
                args.output_path,
                image_paths,
                depths,
                original_coords=original_coords,
                padded_size=args.image_load_resolution,
                confidence=confidence,
            ),
        )

    if args.export_colmap:
        colmap_dir = args.output_path / "sparse" / args.colmap_model_subdir
        colmap_dir.mkdir(parents=True, exist_ok=True)
        write_colmap_formats(
            str(images_dir),
            extrinsic,
            intrinsic,
            image_paths,
            sampled_points,
            sampled_colors,
            width=args.resolution,
            height=args.resolution,
            output_path=str(colmap_dir),
        )

    metrics = {
        "method": "VGGT",
        "Scene": str(args.data_path),
        "Preprocessed Images": len(image_paths),
        "inference time": render_time,
        **runtime,
        "Peak GPU Memory": peak_memory,
        "checkpoint": args.ckpt_path,
        **spatial_policy_metrics(args.spatial_policy, is_shared=args.spatial_policy == "shared"),
    }
    (args.output_path / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    run()
