#!/usr/bin/env python
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Run Co-Me through its Hydra inference entry point and export artifacts."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from adapter_utils import (
    canonical_image_root,
    batched_image_paths,
    c2w_to_w2c,
    cleanup_intermediate_paths,
    export_camera_pose_priors,
    export_depth_predictions,
    export_spatial_depth_predictions,
    export_standard_artifacts,
    find_image_paths,
    materialize_resized_images,
    normalize_depths,
    patch_grid_compatible_resolution,
    plan_spatial_transform,
    prepare_model_input,
    print_dry_run,
    register_preprocessor,
    restore_camera_export,
    se3_xyzw_to_c2w,
    select_interval,
    should_keep_intermediate,
    spatial_policy_metrics,
    validate_shared_spatial_policy_depth_only,
    write_adapter_metrics,
)
from PIL import Image, ImageOps


METHOD = "COME"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Co-Me benchmark adapter.")
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=os.environ.get("COME_CKPT_PATH", ""),
        help="Reserved for local Co-Me/backbone checkpoints; most Co-Me configs carry their own ckpt paths.",
    )
    parser.add_argument(
        "--come_code_path",
        type=Path,
        default=Path(os.environ["COME_CODE_DIR"]) if "COME_CODE_DIR" in os.environ else None,
    )
    parser.add_argument(
        "--come_model",
        type=str,
        default=os.environ.get("COME_MODEL", "co_me_fused_vggt_3x3"),
        help="Hydra model config name, e.g. co_me_fused_vggt_3x3 or co_me_fused_pi3x_3x3.",
    )
    parser.add_argument(
        "--come_extra_override",
        action="append",
        default=[],
        help="Additional Hydra override passed through to Co-Me. Can be repeated.",
    )
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=504)
    parser.add_argument("--image_load_resolution", type=int, default=504)
    parser.add_argument(
        "--internal_resolution",
        type=int,
        default=0,
        help=(
            "Co-Me model resolution. Defaults to the nearest larger size whose "
            "VGGT patch grid is compatible with the configured 3x3 merger."
        ),
    )
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--token_merge_group", type=int, default=3)
    parser.add_argument("--max_images_per_run", type=int, default=0)
    parser.add_argument("--max_points", type=int, default=1_000_000)
    parser.add_argument("--conf_thresh", type=float, default=0.0)
    parser.add_argument("--depth_conf_thresh", type=float, default=None)
    parser.add_argument("--export_depth", action="store_true")
    parser.add_argument("--export_colmap", action="store_true")
    parser.add_argument("--depth_only", action="store_true")
    parser.add_argument("--colmap_model_subdir", type=str, default="0")
    parser.add_argument("--keep_intermediate", action="store_true")
    parser.add_argument(
        "--spatial_policy",
        choices=("legacy", "shared"),
        default="legacy",
        help="'legacy' (default): depth stays on the padded square canvas Co-Me actually sees. "
        "'shared': restore depth/confidence/intrinsics to each image's true source resolution "
        "via restore_prediction/restore_camera_export (depth_only export only).",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load_come_archive(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Co-Me raw output not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def run_come(args: argparse.Namespace, image_paths: list[str], run_name: str = "") -> Path:
    if args.come_code_path is None:
        raise ValueError("--come_code_path or COME_CODE_DIR is required for real Co-Me runs")
    code_path = args.come_code_path.expanduser().resolve()
    if not code_path.is_dir():
        raise FileNotFoundError(f"Co-Me code path not found: {code_path}")

    # Co-Me's own src.infer hardcodes sample.multiview_input.resize(504, 504) and never actually
    # reads the +infer_size override passed below (that Hydra key is dead as far as that entry
    # point goes) -- MultiViewInput.resize's own docstring is "uniform scale to cover target, then
    # center crop", a real crop that discards image content whenever its input isn't already
    # exactly 504x504. materialize_resized_images below pads to a square canvas sized
    # --image_load_resolution, so this is only a no-op (nothing cropped) because that currently
    # defaults to the same 504. Guard against silently cropping if that ever changes.
    if args.image_load_resolution != 504:
        raise ValueError(
            "Co-Me's own src.infer hardcodes .resize(504, 504) regardless of --image_load_resolution; "
            "changing --image_load_resolution away from 504 would make that resize crop real image "
            "content instead of a no-op. Leave it at the default (504) until that upstream hardcoding "
            "is fixed."
        )
    args.output_path.mkdir(parents=True, exist_ok=True)
    suffix = f"_{run_name}" if run_name else ""
    temp_images = args.output_path / f"_come_input_images{suffix}"
    raw_dir = args.output_path / f"_come_raw{suffix}"
    materialize_resized_images(image_paths, temp_images, args.image_load_resolution)
    internal_resolution = (
        args.internal_resolution
        if args.internal_resolution > 0
        else patch_grid_compatible_resolution(
            args.resolution,
            patch_size=args.patch_size,
            grid_multiple=args.token_merge_group,
        )
    )

    command = [
        sys.executable,
        "-m",
        "src.infer",
        "dataset=custom_image",
        f"model={args.come_model}",
        "render=save_raw",
        f"dataset.data_root={temp_images}",
        f"dataset.max_length={len(image_paths)}",
        "dataset.step_size=1",
        "dataset.start=0",
        "first_n=1",
        f"+infer_size=[{internal_resolution},{internal_resolution}]",
        f"output_directory={raw_dir}",
        f"hydra.run.dir={raw_dir / 'hydra'}",
    ]
    command.extend(args.come_extra_override)
    env = os.environ.copy()
    env.setdefault("HYDRA_FULL_ERROR", "1")
    subprocess.run(command, cwd=code_path, env=env, check=True)
    return raw_dir / "sample_0_00.npz"


def preprocess_come(
    image_paths: list[str],
    *,
    args: argparse.Namespace,
    device=None,
    internal_resolution: "int | None" = None,
    **_unused,
):
    """Co-Me's preprocess_<method>(), registered below for prepare_model_input("COME", ...).

    Co-Me's own run_come() (called separately, at the caller) already pads every image onto a
    square canvas via materialize_resized_images before its Hydra entry point resizes that square
    canvas to internal_resolution -- a uniform scale of an already-square image preserves the
    padding ratio, so this only computes the matching SpatialTransform (one "vggt"-style
    pad-to-square transform straight from each image's true source size to internal_resolution,
    matching the composed effect of both resizes exactly, same reasoning as vggt.py's two-stage
    resize) -- trusting run_come's own padding rather than reimplementing it here.
    Returns (image_paths, transforms) -- image_paths unchanged.
    """
    transforms = []
    for image_path in image_paths:
        with Image.open(image_path) as probe:
            source_wh = ImageOps.exif_transpose(probe).size
        transforms.append(plan_spatial_transform(source_wh, method="vggt", size=internal_resolution, mode="pad"))
    return image_paths, transforms


register_preprocessor("COME", preprocess_come)


def come_intermediate_paths(args: argparse.Namespace, raw_path: Path) -> list[Path]:
    raw_dir = raw_path.parent
    input_name = raw_dir.name.replace("_come_raw", "_come_input_images", 1)
    return [args.output_path / input_name, raw_dir]


def parse_scene_arrays(raw: dict[str, np.ndarray]) -> tuple:
    depths = None
    if "scene_depths" in raw:
        depths = raw["scene_depths"]
        if depths.ndim == 4 and depths.shape[1] == 1:
            depths = depths[:, 0]

    points = None
    if "scene_points" in raw:
        points = raw["scene_points"]
        if points.ndim == 4 and points.shape[1] == 3:
            points = np.transpose(points, (0, 2, 3, 1))

    confidence = None
    for key in ("scene_points_conf", "scene_depths_conf"):
        if key in raw:
            confidence = raw[key]
            if confidence.ndim == 4 and confidence.shape[1] == 1:
                confidence = confidence[:, 0]
            break

    colors = None
    if "input_images" in raw:
        colors = raw["input_images"]
        if colors.ndim == 4 and colors.shape[1] == 3:
            colors = np.transpose(colors, (0, 2, 3, 1))
        colors = (colors.clip(0.0, 1.0) * 255.0).astype(np.uint8)

    extrinsics_w2c = None
    if "scene_poses" in raw:
        extrinsics_w2c = c2w_to_w2c(se3_xyzw_to_c2w(raw["scene_poses"]))

    intrinsics = raw.get("scene_intrinsics")
    return depths, points, colors, confidence, extrinsics_w2c, intrinsics


def run() -> None:
    args = parse_args()
    images_dir = canonical_image_root(args.data_path)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images directory not found: {images_dir}")
    image_paths = select_interval(find_image_paths(images_dir), args.interval, args.max_images_per_run)
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")

    if args.dry_run:
        print_dry_run(args, METHOD, len(image_paths))
        return

    started_at = time.time()
    keep_intermediate = should_keep_intermediate(args)
    is_shared_spatial_policy = args.spatial_policy == "shared"
    validate_shared_spatial_policy_depth_only(is_shared=is_shared_spatial_policy, depth_only=args.depth_only)
    if args.depth_only:
        if not args.export_depth:
            raise ValueError("--depth_only requires --export_depth")
        images_per_run = args.max_images_per_run if args.max_images_per_run > 0 else len(image_paths)
        internal_resolution = (
            args.internal_resolution
            if args.internal_resolution > 0
            else patch_grid_compatible_resolution(
                args.resolution, patch_size=args.patch_size, grid_multiple=args.token_merge_group
            )
        )
        pose_extrinsics: list[np.ndarray] = []
        pose_intrinsics: list[np.ndarray] = []
        pose_output_size: tuple[int, int] | None = None
        for batch_index, batch_image_paths in enumerate(batched_image_paths(image_paths, images_per_run)):
            raw_path = run_come(args, batch_image_paths, run_name=f"batch_{batch_index:04d}")
            raw = load_come_archive(raw_path)
            depths, _points, _colors, confidence, extrinsics_w2c, intrinsics = parse_scene_arrays(raw)
            if depths is None:
                raise ValueError("Co-Me raw output did not include depth predictions")
            if is_shared_spatial_policy:
                _paths, transforms = prepare_model_input(
                    "COME", batch_image_paths, args=args, device=None, internal_resolution=internal_resolution
                )
                export_spatial_depth_predictions(
                    args.output_path, batch_image_paths, normalize_depths(depths), confidence, transforms
                )
                if args.export_colmap and extrinsics_w2c is not None and intrinsics is not None:
                    restored_intrinsics, width, height = restore_camera_export(np.asarray(intrinsics), transforms)
                    pose_extrinsics.append(np.asarray(extrinsics_w2c))
                    pose_intrinsics.append(restored_intrinsics)
                    if pose_output_size is None:
                        pose_output_size = (width, height)
                cleanup_intermediate_paths(
                    come_intermediate_paths(args, raw_path),
                    keep=keep_intermediate,
                )
                continue
            export_depth_predictions(args.output_path, batch_image_paths, depths, confidence)
            if args.export_colmap and extrinsics_w2c is not None and intrinsics is not None:
                pose_extrinsics.append(np.asarray(extrinsics_w2c))
                pose_intrinsics.append(np.asarray(intrinsics))
                if pose_output_size is None:
                    depth_array = np.asarray(depths)
                    pose_output_size = (depth_array.shape[2], depth_array.shape[1])
            cleanup_intermediate_paths(
                come_intermediate_paths(args, raw_path),
                keep=keep_intermediate,
            )
        if pose_extrinsics:
            width, height = pose_output_size
            export_camera_pose_priors(
                args.output_path,
                images_dir,
                image_paths,
                np.concatenate(pose_extrinsics, axis=0),
                np.concatenate(pose_intrinsics, axis=0),
                width=width,
                height=height,
                colmap_model_subdir=args.colmap_model_subdir,
            )
        write_adapter_metrics(
            output_path=args.output_path,
            method=METHOD,
            image_paths=image_paths,
            started_at=started_at,
            checkpoint=args.ckpt_path or None,
            depth_only=True,
            metrics_extra={
                "come_model": args.come_model,
                "max_images_per_run": images_per_run,
                "resolution": args.resolution,
                "image_load_resolution": args.image_load_resolution,
                "internal_resolution": (
                    args.internal_resolution
                    if args.internal_resolution > 0
                    else patch_grid_compatible_resolution(
                        args.resolution,
                        patch_size=args.patch_size,
                        grid_multiple=args.token_merge_group,
                    )
                ),
                "patch_size": args.patch_size,
                "token_merge_group": args.token_merge_group,
                "resized_input": True,
                **spatial_policy_metrics(args.spatial_policy, is_shared=is_shared_spatial_policy),
            },
        )
        return

    raw_path = run_come(args, image_paths)
    raw = load_come_archive(raw_path)
    depths, points, colors, confidence, extrinsics_w2c, intrinsics = parse_scene_arrays(raw)
    conf_thresh = args.depth_conf_thresh if args.depth_conf_thresh is not None else args.conf_thresh

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
        intrinsics=intrinsics,
        export_depth=args.export_depth,
        export_colmap=args.export_colmap,
        depth_only=args.depth_only,
        colmap_model_subdir=args.colmap_model_subdir,
        max_points=args.max_points,
        conf_thresh=conf_thresh,
        started_at=started_at,
        checkpoint=args.ckpt_path or None,
        metrics_extra={
            "come_model": args.come_model,
            "raw_output": str(raw_path),
            "max_images_per_run": args.max_images_per_run,
            "resolution": args.resolution,
            "image_load_resolution": args.image_load_resolution,
            "internal_resolution": (
                args.internal_resolution
                if args.internal_resolution > 0
                else patch_grid_compatible_resolution(
                    args.resolution,
                    patch_size=args.patch_size,
                    grid_multiple=args.token_merge_group,
                )
            ),
            "patch_size": args.patch_size,
            "token_merge_group": args.token_merge_group,
            "resized_input": True,
        },
    )
    cleanup_intermediate_paths(
        come_intermediate_paths(args, raw_path),
        keep=keep_intermediate,
    )


if __name__ == "__main__":
    run()
