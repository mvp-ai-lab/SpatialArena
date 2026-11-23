#!/usr/bin/env python
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Run MapAnything and export benchmark prediction artifacts."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from adapter_utils import (
    SpatialTransform,
    mark_model_loaded,
    add_code_path,
    as_numpy,
    batched_image_paths,
    build_default_camera_priors,
    canonical_image_root,
    c2w_to_w2c,
    colors_from_torch_images,
    export_camera_pose_priors,
    export_depth_predictions,
    export_spatial_depth_predictions,
    export_standard_artifacts,
    find_image_paths,
    load_colmap_camera_priors,
    load_masked_depth_priors,
    load_optional_depth_priors,
    load_sparse_depth_priors,
    normalize_confidence,
    normalize_depths,
    normalize_points,
    prepare_model_input,
    preprocess_images,
    print_dry_run,
    register_preprocessor,
    resolve_camera_conditioned_image_paths,
    resize_depth_to_image_size,
    restore_camera_export,
    select_interval,
    write_adapter_metrics,
)

# MapAnything's own patch size is 14 (unlike the DUSt3R-family wrapper's 16); mean/std for its
# default norm_type="dinov2" (see uniception.models.encoders.image_normalizations
# .IMAGE_NORMALIZATION_DICT["dinov2"] -- the ImageNet stats every DINOv2-backed encoder uses).
_MAPANYTHING_PATCH_SIZE = 14
_MAPANYTHING_NORM_TYPE = "dinov2"
_MAPANYTHING_NORM_MEAN = (0.485, 0.456, 0.406)
_MAPANYTHING_NORM_STD = (0.229, 0.224, 0.225)

METHOD = "MAPANYTHING"

TRACKS = (
    "rgb",
    "rgb_depth",
    "rgb_cameras",
    "rgb_masked_depth",
    "rgb_masked_depth_cameras",
    "rgb_sparse_depth",
    "rgb_sparse_depth_cameras",
)
CAMERA_TRACKS = ("rgb_cameras", "rgb_masked_depth_cameras",
                 "rgb_sparse_depth_cameras")
DEPTH_TRACKS = (
    "rgb_depth",
    "rgb_masked_depth",
    "rgb_masked_depth_cameras",
    "rgb_sparse_depth",
    "rgb_sparse_depth_cameras",
)
MASKED_DEPTH_TRACKS = ("rgb_masked_depth", "rgb_masked_depth_cameras")
SPARSE_DEPTH_TRACKS = ("rgb_sparse_depth", "rgb_sparse_depth_cameras")
# The input's long side the sparse points are drawn for (adapter_utils.load_sparse_depth_priors).
SPARSE_MODEL_LONG_SIDE = 518


def _load_track_depth_priors(data_path, image_paths, track):
    """Depth priors for a depth track: masked GT on the masked-depth tracks (spatialarena/depth_prior_masking.py),
    the real sparse depth (LiDAR / COLMAP keypoints) on the sparse-depth tracks (spatialarena/sparse_depth.py)."""
    if track in SPARSE_DEPTH_TRACKS:
        return load_sparse_depth_priors(data_path,
                                        image_paths,
                                        model_long_side=SPARSE_MODEL_LONG_SIDE)
    loader = load_masked_depth_priors if track in MASKED_DEPTH_TRACKS else load_optional_depth_priors
    return loader(data_path, image_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--ckpt_path",
                        default=os.environ.get("MAPANYTHING_CKPT_PATH",
                                               "facebook/map-anything"))
    parser.add_argument(
        "--mapanything_code_path",
        type=Path,
        default=Path(os.environ["MAPANYTHING_CODE_DIR"])
        if "MAPANYTHING_CODE_DIR" in os.environ else None,
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
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def build_rgb_views(
        image_paths: list[str],
        resolution_set: int) -> tuple[list[dict], list[SpatialTransform]]:
    """Build MapAnything view dicts (img/data_norm_type/true_shape/idx/instance) for the plain
    "rgb" track directly from a padded tensor, bypassing mapanything.utils.image.load_images --
    like the shared mapanything_dense_wrapper.py's build_views, that loader computes one target
    size from the whole batch's *average* aspect ratio (fixed_mapping mode) and crops every image
    to it, discarding real content; this track never conditions on depth/camera priors, so (unlike
    rgb_depth/rgb_cameras below, not yet converted) the simple per-image pad case applies directly.
    """
    import torch

    size = resolution_set if resolution_set % _MAPANYTHING_PATCH_SIZE == 0 else (
        ((resolution_set + _MAPANYTHING_PATCH_SIZE - 1) //
         _MAPANYTHING_PATCH_SIZE) * _MAPANYTHING_PATCH_SIZE)
    # exif_transpose=True to match mapanything.utils.image.load_images's own convention (the
    # loader this bypasses still applies it).
    images, transforms = preprocess_images(image_paths,
                                           method="vggt",
                                           size=size,
                                           mode="pad",
                                           exif_transpose=True)
    mean = torch.tensor(_MAPANYTHING_NORM_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(_MAPANYTHING_NORM_STD).view(1, 3, 1, 1)
    images = (images - mean) / std
    views = []
    for index, image_path in enumerate(image_paths):
        frame = images[index:index + 1]
        height, width = frame.shape[-2:]
        views.append({
            "img": frame,
            "data_norm_type": [_MAPANYTHING_NORM_TYPE],
            "true_shape": np.int32([[height, width]]),
            "idx": index,
            "instance": [Path(image_path).name],
        })
    return views, transforms


def preprocess_mapanything(image_paths: list[str],
                           *,
                           args: argparse.Namespace,
                           device=None,
                           **_unused):
    """MapAnything's preprocess_<method>(), registered below for prepare_model_input("MAPANYTHING",
    ...). "rgb" track: delegates to build_rgb_views (the pad-based fix). "rgb_depth"/"rgb_cameras":
    still routes through the vendored mapanything.utils.image.preprocess_inputs (real, documented
    content loss -- see the TODO below; not fixed, since padding the image without also padding
    the real depth/camera prior it's conditioned on in lockstep would desync the two).
    Returns (views, transforms or None)."""
    import torch
    from PIL import Image
    from mapanything.utils.image import preprocess_inputs  # pylint: disable=no-name-in-module

    resolution_set = args.image_load_resolution
    if resolution_set not in {504, 512, 518}:
        resolution_set = 518

    if args.track == "rgb":
        return build_rgb_views(image_paths, resolution_set)

    raw_views = []
    depths = None
    camera_poses = None
    intrinsics = None
    if args.track in DEPTH_TRACKS:
        depths, _has_depth, _missing = _load_track_depth_priors(
            args.data_path, image_paths, args.track)
    if args.track in CAMERA_TRACKS:
        camera_poses, intrinsics = load_colmap_camera_priors(
            args.data_path,
            image_paths,
            colmap_path=args.colmap_path,
            colmap_model_subdir=args.colmap_model_subdir,
        )

    for index, image_path in enumerate(image_paths):
        view = {"img": Image.open(image_path).convert("RGB")}
        metric_scale = torch.ones(1, dtype=torch.bool)
        if depths is not None and depths[index] is not None:
            if intrinsics is None:
                try:
                    _camera_poses, intrinsics = load_colmap_camera_priors(
                        args.data_path,
                        image_paths,
                        colmap_path=args.colmap_path,
                        colmap_model_subdir=args.colmap_model_subdir,
                    )
                except FileNotFoundError:
                    _camera_poses, intrinsics = build_default_camera_priors(
                        image_paths)
            view["depth_z"] = resize_depth_to_image_size(
                depths[index], view["img"].size)
            view["intrinsics"] = intrinsics[index]
            view["is_metric_scale"] = metric_scale
        if camera_poses is not None:
            view["camera_poses"] = camera_poses[index]
            view["intrinsics"] = intrinsics[index]
            view["is_metric_scale"] = metric_scale
        raw_views.append(view)
    # TODO(rgb_depth/rgb_cameras content loss): preprocess_inputs's crop_resize_if_necessary crops
    # depth/intrinsics/image together to one batch-average-aspect-ratio target, same real content
    # loss as the "rgb" track had -- not yet converted to the pad-based fix above, since these
    # tracks also need the depth prior and the input camera intrinsics padded/rescaled correctly
    # in lockstep with the image, not just the image alone. Flagged, not silently left broken.
    return preprocess_inputs(
        raw_views,
        resize_mode="fixed_mapping",
        resolution_set=resolution_set,
    ), None


register_preprocessor("MAPANYTHING", preprocess_mapanything)


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
    image_paths = select_interval(image_paths, args.interval,
                                  args.max_images_per_run)
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")
    if args.dry_run:
        print_dry_run(args, METHOD, len(image_paths))
        return
    if args.depth_only and not args.export_depth:
        raise ValueError("--depth_only requires --export_depth")

    add_code_path(args.mapanything_code_path, ("mapanything",))

    import torch
    from mapanything.models import MapAnything  # pylint: disable=no-name-in-module

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started_at = time.time()

    model = MapAnything.from_pretrained(args.ckpt_path).to(device)
    model.eval()
    mark_model_loaded()

    if args.depth_only:
        batch_size = args.max_images_per_run if args.max_images_per_run > 0 else len(
            image_paths)
        peak_memory = 0.0
        pose_image_paths: list[str] = []
        pose_extrinsics: list[np.ndarray] = []
        pose_intrinsics: list[np.ndarray] = []
        pose_output_size: tuple[int, int] | None = None
        for batch_paths in batched_image_paths(image_paths, batch_size):
            views, transforms = prepare_model_input("MAPANYTHING",
                                                    batch_paths,
                                                    args=args,
                                                    device=device)
            with torch.no_grad():
                # Dense depth (MonoDepth / VideoDepth / MVRecon) is unmasked: MapAnything's non-ambiguous and edge
                # masks write depth 0 on the pixels they reject (up to ~14% of KITTI's GT pixels), a method-specific
                # coverage change; MVRecon fusion filters with the engine's thresholds as for every other method.
                outputs = model.infer(
                    views,
                    memory_efficient_inference=True,
                    minibatch_size=1,
                    use_amp=True,
                    amp_dtype="bf16",
                    apply_mask=False,
                    mask_edges=False,
                )
            depths = normalize_depths(
                [as_numpy(pred["depth_z"])[0, ..., 0] for pred in outputs])
            confidence = normalize_confidence(
                [as_numpy(pred["conf"])[0] for pred in outputs], depths.shape)
            if transforms is not None:
                export_spatial_depth_predictions(args.output_path, batch_paths,
                                                 depths, confidence, transforms)
            else:
                export_depth_predictions(args.output_path, batch_paths, depths,
                                         confidence)
            if args.export_colmap:
                pose_image_paths.extend(batch_paths)
                pose_extrinsics.append(
                    c2w_to_w2c([
                        as_numpy(pred["camera_poses"])[0] for pred in outputs
                    ]))
                intrinsics = as_numpy([
                    as_numpy(pred["intrinsics"])[0] for pred in outputs
                ]).astype("float32")
                if transforms is not None:
                    restored_intrinsics, width, height = restore_camera_export(
                        intrinsics, transforms)
                    pose_intrinsics.append(restored_intrinsics)
                    pose_output_size = (width, height)
                else:
                    pose_intrinsics.append(intrinsics)
                    if pose_output_size is None:
                        pose_output_size = (depths.shape[2], depths.shape[1])
            if torch.cuda.is_available():
                peak_memory = max(
                    peak_memory,
                    torch.cuda.max_memory_allocated() / (1024.0**2))
                torch.cuda.empty_cache()
        if pose_extrinsics:
            width, height = pose_output_size
            export_camera_pose_priors(
                args.output_path,
                images_dir,
                pose_image_paths,
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
            checkpoint=args.ckpt_path,
            depth_only=True,
            metrics_extra={
                "Peak GPU Memory": peak_memory,
                "track": args.track,
                "image_load_resolution": args.image_load_resolution,
                "max_images_per_run": batch_size,
            },
        )
        return

    views, _transforms = prepare_model_input("MAPANYTHING",
                                             image_paths,
                                             args=args,
                                             device=device)
    with torch.no_grad():
        outputs = model.infer(
            views,
            memory_efficient_inference=True,
            minibatch_size=1,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
        )

    depths = normalize_depths(
        [as_numpy(pred["depth_z"])[0, ..., 0] for pred in outputs])
    points = normalize_points([as_numpy(pred["pts3d"])[0] for pred in outputs])
    colors = colors_from_torch_images(
        [as_numpy(pred["img_no_norm"])[0] for pred in outputs])
    confidence = normalize_confidence(
        [as_numpy(pred["conf"])[0] for pred in outputs], depths.shape)
    extrinsics_w2c = c2w_to_w2c(
        [as_numpy(pred["camera_poses"])[0] for pred in outputs])
    intrinsics = as_numpy([as_numpy(pred["intrinsics"])[0] for pred in outputs
                          ]).astype("float32")

    peak_memory = (torch.cuda.max_memory_allocated() /
                   (1024.0**2) if torch.cuda.is_available() else 0.0)
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
        conf_thresh=args.depth_conf_thresh
        if args.depth_conf_thresh is not None else args.conf_thresh,
        started_at=started_at,
        checkpoint=args.ckpt_path,
        metrics_extra={
            "Peak GPU Memory": peak_memory,
            "track": args.track,
            "image_load_resolution": args.image_load_resolution,
        },
    )


if __name__ == "__main__":
    run()
