#!/usr/bin/env python
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Run OmniVGGT and export benchmark prediction artifacts."""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from adapter_utils import (
    SpatialTransform,
    mark_model_loaded,
    record_model_input,
    add_code_path,
    as_numpy,
    canonical_image_root,
    cleanup_intermediate_paths,
    colors_from_torch_images,
    export_camera_pose_priors,
    export_spatial_depth_predictions,
    export_standard_artifacts,
    find_image_paths,
    load_colmap_camera_priors,
    load_masked_depth_priors,
    load_optional_depth_priors,
    load_sparse_depth_priors,
    prepare_model_input,
    print_dry_run,
    register_preprocessor,
    resolve_camera_conditioned_image_paths,
    restore_camera_export,
    select_interval,
    should_keep_intermediate,
    single_view_mode,
    write_adapter_metrics,
)


METHOD = "OMNIVGGT"

TRACKS = (
    "rgb", "rgb_depth", "rgb_cameras", "rgb_masked_depth", "rgb_masked_depth_cameras", "rgb_sparse_depth",
    "rgb_sparse_depth_cameras",
)
CAMERA_TRACKS = ("rgb_cameras", "rgb_masked_depth_cameras", "rgb_sparse_depth_cameras")
DEPTH_TRACKS = (
    "rgb_depth", "rgb_masked_depth", "rgb_masked_depth_cameras", "rgb_sparse_depth", "rgb_sparse_depth_cameras",
)
MASKED_DEPTH_TRACKS = ("rgb_masked_depth", "rgb_masked_depth_cameras")
SPARSE_DEPTH_TRACKS = ("rgb_sparse_depth", "rgb_sparse_depth_cameras")
# The input's long side the sparse points are drawn for (adapter_utils.load_sparse_depth_priors).
SPARSE_MODEL_LONG_SIDE = 518


def _load_track_depth_priors(data_path, image_paths, track):
    """Depth priors for a depth track: masked GT on the masked-depth tracks (spatialarena/depth_prior_masking.py),
    the real sparse depth (LiDAR / COLMAP keypoints) on the sparse-depth tracks (spatialarena/sparse_depth.py)."""
    if track in SPARSE_DEPTH_TRACKS:
        return load_sparse_depth_priors(data_path, image_paths, model_long_side=SPARSE_MODEL_LONG_SIDE)
    loader = load_masked_depth_priors if track in MASKED_DEPTH_TRACKS else load_optional_depth_priors
    return loader(data_path, image_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--ckpt_path", default=os.environ.get("OMNIVGGT_CKPT_PATH", "Livioni/OmniVGGT"))
    parser.add_argument(
        "--omnivggt_code_path",
        type=Path,
        default=Path(os.environ["OMNIVGGT_CODE_DIR"]) if "OMNIVGGT_CODE_DIR" in os.environ else None,
    )
    parser.add_argument("--track", choices=TRACKS, default="rgb")
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=518)
    parser.add_argument("--image_load_resolution", type=int, default=518)
    parser.add_argument("--max_images_per_run", type=int, default=0)
    parser.add_argument("--max_points", type=int, default=1_000_000)
    parser.add_argument("--conf_thresh", type=float, default=0.0)
    parser.add_argument("--depth_conf_thresh", type=float, default=None)
    parser.add_argument("--export_depth", action="store_true")
    parser.add_argument("--export_colmap", action="store_true")
    parser.add_argument("--depth_only", action="store_true")
    parser.add_argument("--colmap_path", type=Path, default=None)
    parser.add_argument("--colmap_model_subdir", default="0")
    parser.add_argument("--keep_intermediate", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def resolve_checkpoint(ckpt_path: str) -> str:
    path = Path(ckpt_path).expanduser()
    if path.is_file():
        return str(path)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=ckpt_path, filename="OmniVGGT.safetensors")


def preprocess_omnivggt(image_paths: list[str], *, args: argparse.Namespace, device=None, **_unused):
    """OmniVGGT's preprocess_<method>(), registered below for prepare_model_input("OMNIVGGT", ...).
    Unconditional (no --spatial_policy toggle). Writes padded images/depths/cameras directories
    (bypassing OmniVGGT's own loader) and the matching per-image SpatialTransform list.
    Returns ({"image_dir": ..., "depth_dir": ... or None, "camera_dir": ... or None}, transforms).
    """
    image_dir, depth_dir, camera_dir, transforms = _materialize_omnivggt_inputs(args, image_paths)
    return {"image_dir": image_dir, "depth_dir": depth_dir, "camera_dir": camera_dir}, transforms


register_preprocessor("OMNIVGGT", preprocess_omnivggt)


def _materialize_omnivggt_inputs(
    args: argparse.Namespace,
    image_paths: list[str],
) -> tuple[Path, Path | None, Path | None, list[SpatialTransform]]:
    input_root = args.output_path / "_omnivggt_inputs"
    image_dir = input_root / "images"
    depth_dir = input_root / "depths"
    camera_dir = input_root / "cameras"
    image_dir.mkdir(parents=True, exist_ok=True)
    for existing in image_dir.iterdir():
        if existing.is_file() or existing.is_symlink():
            existing.unlink()

    target_size = int(args.image_load_resolution)
    stems: list[str] = []
    image_transforms: list[tuple[float, int, int]] = []
    spatial_transforms: list[SpatialTransform] = []
    for index, image_path in enumerate(image_paths):
        src = Path(image_path)
        stem = f"{index:06d}_{src.stem}"
        stems.append(stem)
        dst = image_dir / f"{stem}.png"
        with Image.open(src) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            width, height = image.size
            scale = min(target_size / float(width), target_size / float(height))
            resized_width = max(1, int(round(width * scale)))
            resized_height = max(1, int(round(height * scale)))
            offset_x = (target_size - resized_width) // 2
            offset_y = (target_size - resized_height) // 2
            resized = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
            canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
            canvas.paste(resized, (offset_x, offset_y))
            canvas.save(dst)
            image_transforms.append((scale, offset_x, offset_y))
            # Wraps this same pad computation (not a separate/different transform) into a
            # SpatialTransform, so restore_prediction/restore_camera_export can map the model's
            # predictions on this padded canvas back to this image's true (width, height) later.
            spatial_transforms.append(
                SpatialTransform(
                    source_wh=(width, height),
                    resized_wh=(resized_width, resized_height),
                    padding=(
                        offset_x,
                        offset_y,
                        target_size - resized_width - offset_x,
                        target_size - resized_height - offset_y,
                    ),
                    policy="omnivggt_resize_pad_square",
                )
            )

    active_depth_dir = None
    if args.track in DEPTH_TRACKS:
        depth_dir.mkdir(parents=True, exist_ok=True)
        for existing in depth_dir.iterdir():
            if existing.is_file() or existing.is_symlink():
                existing.unlink()
        depths, has_depth, _missing = _load_track_depth_priors(args.data_path, image_paths, args.track)
        if not has_depth.any():
            raise FileNotFoundError(f"No matched depth priors found for {args.data_path}")
        for stem, depth in zip(stems, depths, strict=True):
            if depth is None:
                continue
            np.save(
                depth_dir / f"{stem}.npy",
                resize_depth_prior_for_omnivggt(depth, target_size).astype(
                    np.float32,
                    copy=False,
                ),
            )
        active_depth_dir = depth_dir

    active_camera_dir = None
    if args.track in CAMERA_TRACKS:
        camera_dir.mkdir(parents=True, exist_ok=True)
        for existing in camera_dir.iterdir():
            if existing.is_file() or existing.is_symlink():
                existing.unlink()
        c2w, intrinsics = load_colmap_camera_priors(
            args.data_path,
            image_paths,
            colmap_path=args.colmap_path,
            colmap_model_subdir=args.colmap_model_subdir,
        )
        for stem, pose, intrinsic, transform in zip(stems, c2w, intrinsics, image_transforms):
            adjusted_intrinsic = adjust_intrinsics_for_square_materialization(
                intrinsic,
                transform,
            )
            with (camera_dir / f"{stem}.txt").open("w", encoding="utf-8") as handle:
                for row in pose[:3, :4]:
                    handle.write(" ".join(f"{value:.9g}" for value in row) + "\n")
                for row in adjusted_intrinsic:
                    handle.write(" ".join(f"{value:.9g}" for value in row) + "\n")
        active_camera_dir = camera_dir

    return image_dir, active_depth_dir, active_camera_dir, spatial_transforms


def resize_depth_prior_for_omnivggt(depth: np.ndarray, target_size: int) -> np.ndarray:
    """Resize/pad a depth prior to the square materialized OmniVGGT input."""

    depth_array = np.asarray(depth, dtype=np.float32)
    if depth_array.ndim == 3:
        depth_array = depth_array[..., 0]
    height, width = depth_array.shape
    scale = min(target_size / float(width), target_size / float(height))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    try:
        import cv2

        resized = cv2.resize(
            depth_array,
            (resized_width, resized_height),
            interpolation=cv2.INTER_NEAREST,
        )
    except ImportError:
        resized = np.asarray(
            Image.fromarray(depth_array).resize(
                (resized_width, resized_height),
                Image.Resampling.NEAREST,
            ),
            dtype=np.float32,
        )
    canvas = np.zeros((target_size, target_size), dtype=np.float32)
    offset_x = (target_size - resized_width) // 2
    offset_y = (target_size - resized_height) // 2
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
    return canvas


def adjust_intrinsics_for_square_materialization(
    intrinsic: np.ndarray,
    transform: tuple[float, int, int],
) -> np.ndarray:
    """Apply resize/pad transform to a pinhole intrinsic matrix."""

    scale, offset_x, offset_y = transform
    adjusted = np.asarray(intrinsic, dtype=np.float32).copy()
    adjusted[0, 0] *= scale
    adjusted[1, 1] *= scale
    adjusted[0, 2] = adjusted[0, 2] * scale + float(offset_x)
    adjusted[1, 2] = adjusted[1, 2] * scale + float(offset_y)
    return adjusted


def run() -> None:
    args = parse_args()
    images_dir = canonical_image_root(args.data_path)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images directory not found: {images_dir}")
    image_paths = find_image_paths(images_dir)
    if args.track in CAMERA_TRACKS:
        image_paths = resolve_camera_conditioned_image_paths(
            args.data_path,
            image_paths,
            colmap_path=args.colmap_path,
        )
    # select_interval also applies args.max_images_per_run as a total-frame
    # cap: the chunking loop below (chunk_size == max_images_per_run) runs
    # unconditionally in both --depth_only and full mode, but each chunk is
    # an independent forward pass with its own arbitrary coordinate frame
    # and results are simply concatenated -- no cross-chunk registration.
    # Capping total frames first means the loop always produces exactly one
    # chunk; doesn't change per-chunk memory footprint since chunk_size
    # (already <= max_images_per_run) is unchanged.
    image_paths = select_interval(image_paths, args.interval, args.max_images_per_run)
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")
    if args.dry_run:
        print_dry_run(args, METHOD, len(image_paths))
        return
    if args.depth_only and not args.export_depth:
        raise ValueError("--depth_only requires --export_depth")

    add_code_path(args.omnivggt_code_path, ("omnivggt",))

    import torch
    from omnivggt.models.omnivggt import OmniVGGT  # pylint: disable=no-name-in-module
    from omnivggt.utils.pose_enc import pose_encoding_to_extri_intri  # pylint: disable=no-name-in-module
    from safetensors.torch import load_file
    from visual_util import load_images_and_cameras

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started_at = time.time()

    model = OmniVGGT()
    model.load_state_dict(load_file(resolve_checkpoint(args.ckpt_path)), strict=True)
    model.to(device).eval()
    mark_model_loaded()

    chunk_size = args.max_images_per_run if args.max_images_per_run > 0 else len(image_paths)
    if single_view_mode():
        chunk_size = 1  # MonoDepth: one image per forward pass, never a multi-view chunk
    depth_chunks = []
    confidence_chunks = []
    extrinsic_chunks = []
    intrinsic_chunks = []
    color_chunks = []
    all_transforms: list[SpatialTransform] = []
    total_depth_prior_frames = 0
    total_camera_prior_frames = 0
    peak_memory = 0.0
    for start in range(0, len(image_paths), chunk_size):
        chunk_paths = image_paths[start : start + chunk_size]
        model_input, chunk_transforms = prepare_model_input("OMNIVGGT", chunk_paths, args=args, device=device)
        materialized_image_dir = model_input["image_dir"]
        materialized_depth_dir = model_input["depth_dir"]
        materialized_camera_dir = model_input["camera_dir"]
        all_transforms.extend(chunk_transforms)
        images, extrinsics, intrinsics, depthmaps, masks, depth_indices, camera_indices = load_images_and_cameras(
            str(materialized_image_dir),
            str(materialized_camera_dir) if materialized_camera_dir is not None else None,
            str(materialized_depth_dir) if materialized_depth_dir is not None else None,
            args.image_load_resolution,
        )
        record_model_input(images.shape[-1], images.shape[-2])
        inputs = {
            "images": images.to(device),
            "extrinsics": extrinsics.to(device),
            "intrinsics": intrinsics.to(device),
            "depth": depthmaps.to(device),
            "mask": masks.to(device),
            "depth_gt_index": depth_indices,
            "camera_gt_index": camera_indices,
        }
        with torch.no_grad():
            predictions = model.inference(**inputs)
            pred_extrinsics, pred_intrinsics = pose_encoding_to_extri_intri(
                predictions["pose_enc"],
                images.shape[-2:],
            )
        depths = as_numpy(predictions["depth"]).astype(np.float32)
        if depths.ndim == 5:
            depths = depths[0, ..., 0]
        depth_chunks.append(depths)
        confidence = predictions.get("depth_conf", None)
        if confidence is not None:
            confidence = as_numpy(confidence)
            if confidence.ndim == 4:
                confidence = confidence[0]
            confidence_chunks.append(confidence)
        if args.export_colmap:
            extrinsic_chunks.append(as_numpy(pred_extrinsics)[0].astype(np.float32))
            intrinsic_chunks.append(as_numpy(pred_intrinsics)[0].astype(np.float32))
        if not args.depth_only:
            color_chunks.append(colors_from_torch_images(images))
        total_depth_prior_frames += len(depth_indices)
        total_camera_prior_frames += len(camera_indices)
        if torch.cuda.is_available():
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated() / (1024.0**2))
        cleanup_intermediate_paths([args.output_path / "_omnivggt_inputs"], keep=should_keep_intermediate(args))

    depths = np.concatenate(depth_chunks, axis=0)
    confidence = np.concatenate(confidence_chunks, axis=0) if confidence_chunks else None
    extrinsics_w2c = np.concatenate(extrinsic_chunks, axis=0) if extrinsic_chunks else None
    intrinsics_np = np.concatenate(intrinsic_chunks, axis=0) if intrinsic_chunks else None
    colors = np.concatenate(color_chunks, axis=0) if color_chunks else None
    points = None

    metrics_extra = {
        "Peak GPU Memory": peak_memory,
        "track": args.track,
        "image_load_resolution": args.image_load_resolution,
        "depth_prior_frames": total_depth_prior_frames,
        "camera_prior_frames": total_camera_prior_frames,
        "max_images_per_run": chunk_size,
    }
    if args.depth_only:
        # Unlike export_standard_artifacts's plain export_depth_predictions (which would write
        # depths at the padded materialize_inputs() canvas resolution, like every other method
        # before this fix), restore each image's own true source resolution here -- the same
        # restore_prediction/restore_camera_export machinery already proven on
        # vggt.py/fastvggt.py/da3.py/vggt_omega.py/cut3r.py.
        if args.export_depth:
            export_spatial_depth_predictions(args.output_path, image_paths, depths, confidence, all_transforms)
        if args.export_colmap and extrinsics_w2c is not None and intrinsics_np is not None:
            restored_intrinsics, width, height = restore_camera_export(intrinsics_np, all_transforms)
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
            checkpoint=args.ckpt_path,
            depth_only=True,
            metrics_extra=metrics_extra,
        )
    else:
        # Dense/points3D export derives pixel-indexed geometry from the model's own padded grid --
        # not restored to source resolution, same restriction every other patched adapter's
        # "shared" policy has for non-depth-only export.
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
            conf_thresh=args.depth_conf_thresh if args.depth_conf_thresh is not None else args.conf_thresh,
            started_at=started_at,
            checkpoint=args.ckpt_path,
            metrics_extra=metrics_extra,
        )
    cleanup_intermediate_paths([args.output_path / "_omnivggt_inputs"], keep=should_keep_intermediate(args))
    if not should_keep_intermediate(args):
        shutil.rmtree(args.output_path / "_omnivggt_inputs", ignore_errors=True)


if __name__ == "__main__":
    run()
