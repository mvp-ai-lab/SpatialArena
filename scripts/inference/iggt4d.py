#!/usr/bin/env python
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Run IGGT4D (HorizonRobotics/IGGT4D) and export benchmark prediction artifacts.

IGGT4D's public API (``iggt4d.api.IGGT4D.inference``) follows Depth Anything 3's: images in, a ``Prediction`` with
per-frame ``depth`` and ``conf`` at the processing resolution (``process_res``, longest side, default 504),
world-to-camera ``extrinsics`` and ``intrinsics`` out. The instance head is not needed by the benchmark and is
skipped (``run_instance=False``). With camera priors (``rgb_cameras``) the input poses condition the model and the
prediction is aligned to them, as in the official ``--camera`` mode.

The checkpoint is a single ``model.safetensors`` (Hugging Face ``HorizonRobotics/IGGT4D``); upstream loads it from
the relative path ``checkpoints/model.safetensors``, so the adapter points that path at ``--ckpt_path``.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from adapter_utils import (
    SpatialTransform,
    add_code_path,
    batched_image_paths,
    canonical_image_root,
    cleanup_intermediate_paths,
    export_camera_pose_priors,
    export_depth_predictions,
    export_spatial_depth_predictions,
    export_standard_artifacts,
    find_image_paths,
    load_colmap_camera_priors,
    mark_model_loaded,
    materialize_resized_images,
    prepare_model_input,
    print_dry_run,
    register_preprocessor,
    record_model_input,
    resolve_camera_conditioned_image_paths,
    restore_camera_export,
    select_interval,
    should_keep_intermediate,
    unproject_depths_to_world,
    write_adapter_metrics,
)

METHOD = "IGGT4D"


def parse_args() -> argparse.Namespace:
    """Benchmark adapter arguments (the shared run_method_case_list.sh command line)."""
    parser = argparse.ArgumentParser(
        description="Run the IGGT4D benchmark adapter.")
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--ckpt_path",
                        type=str,
                        default=os.environ.get("IGGT4D_CKPT_PATH"),
                        help="IGGT4D model.safetensors.")
    parser.add_argument("--iggt4d_code_path",
                        type=Path,
                        default=Path(os.environ["IGGT4D_CODE_PATH"])
                        if "IGGT4D_CODE_PATH" in os.environ else None)
    parser.add_argument("--track",
                        choices=("rgb", "rgb_cameras"),
                        default="rgb")
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--resolution",
                        type=int,
                        default=504,
                        help="process_res: longest processed side.")
    parser.add_argument("--image_load_resolution", type=int, default=504)
    parser.add_argument("--max_images_per_run", type=int, default=0)
    parser.add_argument("--max_points", type=int, default=1_000_000)
    parser.add_argument("--conf_thresh", type=float, default=0.0)
    parser.add_argument("--depth_conf_thresh", type=float, default=None)
    parser.add_argument("--export_depth", action="store_true")
    parser.add_argument("--export_colmap", action="store_true")
    parser.add_argument("--depth_only", action="store_true")
    parser.add_argument("--colmap_path", type=Path, default=None)
    parser.add_argument("--colmap_model_subdir", type=str, default="0")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load_model(code_path: Path | None, ckpt_path: str, device: str):
    """IGGT4D with its weights from ``ckpt_path`` (upstream reads them from ``checkpoints/model.safetensors``)."""
    add_code_path(code_path, ("iggt4d",))
    from iggt4d.utils import model_loading  # pylint: disable=import-outside-toplevel

    checkpoint = Path(ckpt_path).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"IGGT4D checkpoint not found: {checkpoint} (HorizonRobotics/IGGT4D model.safetensors)"
        )
    model_loading.DEFAULT_MODEL_CHECKPOINT_PATH = checkpoint
    from iggt4d.api import IGGT4D  # pylint: disable=import-outside-toplevel

    model = IGGT4D()
    model.device = device
    return model.to(device).eval()


def preprocess_iggt4d(image_paths: list[str],
                      *,
                      args: argparse.Namespace,
                      device=None,
                      camera_priors=None,
                      **_unused):
    """IGGT4D's preprocess_<method>(), registered below for prepare_model_input("IGGT4D", ...).

    Without camera priors (the "rgb" track), this bypasses IGGT4D's own cross-image crop
    (InputProcessor._unify_batch_shapes: a real, batch-dependent center-crop needed only because
    its per-image upper_bound_resize can round different images to slightly different shapes) by
    pre-padding every image to one shared canvas itself (materialize_resized_images, the same
    lossless pad used elsewhere this session) and passing a fixed (width, height) process_res --
    IGGT4D's own _resize_fixed is then a no-op for every image (all already exactly that size), so
    _unify_batch_shapes never needs to crop anything. Depth/intrinsics are then restored to each
    image's true source resolution afterward.

    With camera priors (the "rgb_cameras" track), left as before (not yet converted): pre-padding
    the image here would desync it from the camera prior's own intrinsics, which IGGT4D's
    _resize_ixt expects to rescale together with the image's *true* source resolution -- feeding
    it an already-padded file would make it read the wrong "orig_w, orig_h" and silently miscompute
    that rescale. Flagged, not silently left broken.

    Returns ({"inference_paths": ..., "process_res": ..., "resized_dir": Path or None}, transforms).
    """
    from PIL import Image, ImageOps

    if camera_priors is not None:
        return {
            "inference_paths": image_paths,
            "process_res": args.resolution,
            "resized_dir": None
        }, None

    target = ((args.resolution + 13) // 14) * 14
    source_whs = []
    for image_path in image_paths:
        # Match materialize_resized_images's own ImageOps.exif_transpose exactly -- an
        # un-transposed probe would report width/height swapped (relative to the padded
        # canvas it actually writes) for any image whose EXIF orientation tag requires a
        # 90/270 degree rotation.
        with Image.open(image_path) as probe:
            source_whs.append(ImageOps.exif_transpose(probe).size)
    resized_dir = args.output_path / "_iggt4d_resized_input_images"
    inference_paths = materialize_resized_images(image_paths, resized_dir,
                                                 target)
    transforms = []
    for source_w, source_h in source_whs:
        scale = min(target / source_w, target / source_h)
        resized_w = max(1, round(source_w * scale))
        resized_h = max(1, round(source_h * scale))
        pad_l, pad_t = (target - resized_w) // 2, (target - resized_h) // 2
        transforms.append(
            SpatialTransform(
                source_wh=(source_w, source_h),
                resized_wh=(resized_w, resized_h),
                padding=(pad_l, pad_t, target - resized_w - pad_l,
                         target - resized_h - pad_t),
            ))
    return {
        "inference_paths": inference_paths,
        "process_res": (target, target),
        "resized_dir": resized_dir
    }, transforms


register_preprocessor("IGGT4D", preprocess_iggt4d)


def predict(model, image_paths: list[str], args: argparse.Namespace,
            camera_priors, *, keep_intermediate: bool):
    """One forward pass over ``image_paths``: depth, confidence, w2c extrinsics, intrinsics, colors
    (uint8), and per-image SpatialTransforms (None when camera_priors is given)."""
    extrinsics = intrinsics = None
    if camera_priors is not None:
        poses_c2w, prior_intrinsics = camera_priors
        extrinsics = np.linalg.inv(poses_c2w).astype(np.float32)
        intrinsics = prior_intrinsics.astype(np.float32)

    model_input, transforms = prepare_model_input("IGGT4D",
                                                  image_paths,
                                                  args=args,
                                                  device=None,
                                                  camera_priors=camera_priors)
    resized_dir = model_input["resized_dir"]

    prediction = model.inference(
        model_input["inference_paths"],
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        process_res=model_input["process_res"],
        run_instance=False,
        cluster_instances=False,
    )
    if resized_dir is not None:
        cleanup_intermediate_paths([resized_dir], keep=keep_intermediate)
    depth = np.asarray(prediction.depth, dtype=np.float32)
    record_model_input(depth.shape[-1], depth.shape[-2])
    extrinsics_w2c = np.asarray(prediction.extrinsics, dtype=np.float32)
    if extrinsics_w2c.shape[-2:] == (3, 4):
        bottom = np.broadcast_to(np.array([0, 0, 0, 1], dtype=np.float32),
                                 (len(extrinsics_w2c), 1, 4))
        extrinsics_w2c = np.concatenate([extrinsics_w2c, bottom], axis=1)
    colors = prediction.processed_images
    confidence = None if prediction.conf is None else np.asarray(
        prediction.conf, dtype=np.float32)
    return (
        depth,
        confidence,
        extrinsics_w2c,
        np.asarray(prediction.intrinsics, dtype=np.float32),
        colors,
        transforms,
    )


def run() -> None:
    """Select the frames, run IGGT4D and export depth / confidence / COLMAP / points."""
    args = parse_args()
    images_dir = canonical_image_root(args.data_path)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images directory not found: {images_dir}")
    image_paths = find_image_paths(images_dir)
    if args.track == "rgb_cameras":
        image_paths = resolve_camera_conditioned_image_paths(
            args.data_path, image_paths, colmap_path=args.colmap_path)
    image_paths = select_interval(image_paths, args.interval,
                                  args.max_images_per_run)
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")
    if args.dry_run:
        print_dry_run(args, METHOD, len(image_paths))
        return
    if not args.ckpt_path:
        raise ValueError("IGGT4D requires --ckpt_path or IGGT4D_CKPT_PATH")

    import torch  # pylint: disable=import-outside-toplevel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    started_at = time.time()
    model = load_model(args.iggt4d_code_path, args.ckpt_path, device)
    mark_model_loaded()

    def priors_for(paths: list[str]):
        if args.track != "rgb_cameras":
            return None
        return load_colmap_camera_priors(
            args.data_path,
            paths,
            colmap_path=args.colmap_path,
            colmap_model_subdir=args.colmap_model_subdir)

    metrics_extra = {
        "track": args.track,
        "process_res": args.resolution,
        "max_images_per_run": args.max_images_per_run
    }
    if args.depth_only:
        if not args.export_depth:
            raise ValueError("--depth_only requires --export_depth")
        images_per_run = args.max_images_per_run if args.max_images_per_run > 0 else len(
            image_paths)
        keep_intermediate = should_keep_intermediate(args)
        pose_paths, pose_extrinsics, pose_intrinsics, pose_size = [], [], [], None
        for batch_paths in batched_image_paths(image_paths, images_per_run):
            priors = priors_for(batch_paths)
            depth, confidence, extrinsics_w2c, intrinsics, _, transforms = predict(
                model,
                batch_paths,
                args,
                priors,
                keep_intermediate=keep_intermediate)
            if transforms is not None:
                export_spatial_depth_predictions(args.output_path, batch_paths,
                                                 depth, confidence, transforms)
            else:
                export_depth_predictions(args.output_path, batch_paths, depth,
                                         confidence)
            if args.export_colmap:
                pose_paths.extend(batch_paths)
                pose_extrinsics.append(extrinsics_w2c)
                if transforms is not None:
                    restored_intrinsics, width, height = restore_camera_export(
                        intrinsics, transforms)
                    pose_intrinsics.append(restored_intrinsics)
                    pose_size = (width, height)
                else:
                    pose_intrinsics.append(intrinsics)
                    pose_size = pose_size or (depth.shape[2], depth.shape[1])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if pose_extrinsics:
            export_camera_pose_priors(
                args.output_path,
                images_dir,
                pose_paths,
                np.concatenate(pose_extrinsics),
                np.concatenate(pose_intrinsics),
                width=pose_size[0],
                height=pose_size[1],
                colmap_model_subdir=args.colmap_model_subdir)
        write_adapter_metrics(output_path=args.output_path,
                              method=METHOD,
                              image_paths=image_paths,
                              started_at=started_at,
                              checkpoint=args.ckpt_path,
                              depth_only=True,
                              metrics_extra=dict(
                                  metrics_extra,
                                  max_images_per_run=images_per_run))
        return

    depth, confidence, extrinsics_w2c, intrinsics, colors, _transforms = predict(
        model,
        image_paths,
        args,
        priors_for(image_paths),
        keep_intermediate=should_keep_intermediate(args))
    points = unproject_depths_to_world(depth, extrinsics_w2c, intrinsics)
    export_standard_artifacts(
        method=METHOD,
        output_path=args.output_path,
        images_dir=images_dir,
        image_paths=image_paths,
        depths=depth,
        points=points,
        colors=colors,
        confidence=confidence,
        extrinsics_w2c=extrinsics_w2c,
        intrinsics=intrinsics,
        export_depth=args.export_depth,
        export_colmap=args.export_colmap,
        depth_only=False,
        colmap_model_subdir=args.colmap_model_subdir,
        max_points=args.max_points,
        conf_thresh=args.depth_conf_thresh
        if args.depth_conf_thresh is not None else args.conf_thresh,
        started_at=started_at,
        checkpoint=args.ckpt_path,
        metrics_extra=metrics_extra,
    )


if __name__ == "__main__":
    run()
