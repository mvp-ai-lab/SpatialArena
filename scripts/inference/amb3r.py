#!/usr/bin/env python
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Run AMB3R and export benchmark prediction artifacts.

AMB3R's front-end reuses VGGT's own camera/depth/point heads (see
amb3r/frontend.py's ``decode_heads``, which imports
``vggt.utils.pose_enc.pose_encoding_to_extri_intri`` from AMB3R's vendored
``thirdparty/vggt`` copy), so its prediction dict has the same
pose/intrinsic/depth/world_points shape convention VGGT's own adapter
(scripts/inference/vggt.py) exports -- this adapter mirrors that.

Preprocessing goes through AMB3R's own ``amb3r.datasets.Demo`` dataset
(pointed at a scratch directory containing only this run's selected/interval-
sampled images, symlinked in) rather than a hand-rolled resize, since it's
the same loader AMB3R's own demo.py/benchmark scripts use and reproducing
its dust3r-family crop-resize math independently would risk a subtle mismatch.

Demo's underlying _crop_resize_if_necessary (thirdparty/dust3r's
base_stereo_view_dataset.py) supports disable_crop=True: a pure anisotropic
resize straight to the target W x H with no crop and no pad, keeping every
source pixel (at the cost of a slight aspect-ratio distortion where the
source ratio isn't already 518:392) -- used here instead of its default
crop behavior so no image content is ever discarded, matching every other
adapter's "shared" spatial policy. Depth/camera predictions are restored
back to each image's true source resolution afterward via
adapter_utils.restore_prediction/export_spatial_depth_predictions/
restore_camera_export, since this is a pure stretch transform (no
padding/crop margins) their formulas already handle as a degenerate case.
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.inference.adapter_utils import (  # noqa: E402
    SpatialTransform,
    canonical_image_root,
    mark_model_loaded,
    add_code_path,
    batched_image_paths,
    cleanup_intermediate_paths,
    export_camera_pose_priors,
    export_spatial_depth_predictions,
    export_standard_artifacts,
    find_image_paths,
    fixed_input_size,
    prepare_model_input,
    print_dry_run,
    register_preprocessor,
    record_model_input,
    restore_camera_export,
    select_interval,
    should_keep_intermediate,
    write_adapter_metrics,
)


METHOD = "AMB3R"

# AMB3R's own benchmark harness (benchmark/eval_mvrecon.py etc.) hardcodes
# this exact non-square resolution for the base model; it is not a tunable
# knob in their own scripts either.
AMB3R_WIDTH = 518
AMB3R_HEIGHT = 392


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AMB3R benchmark adapter.")
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=os.environ.get("AMB3R_CKPT_PATH", ""),
    )
    parser.add_argument(
        "--amb3r_code_path",
        type=Path,
        default=Path(os.environ["AMB3R_CODE_DIR"]) if "AMB3R_CODE_DIR" in os.environ else None,
    )
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--max_images_per_run", type=int, default=0)
    parser.add_argument("--max_points", type=int, default=1_000_000)
    parser.add_argument("--conf_thresh", type=float, default=5.0)
    parser.add_argument("--depth_conf_thresh", type=float, default=None)
    parser.add_argument("--export_depth", action="store_true")
    parser.add_argument("--export_colmap", action="store_true")
    parser.add_argument("--depth_only", action="store_true")
    parser.add_argument("--colmap_model_subdir", type=str, default="0")
    parser.add_argument("--keep_intermediate", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def materialize_symlinked_images(image_paths: list[str], dest_dir: Path) -> None:
    """AMB3R's Demo dataset lists ROOT via os.listdir (no explicit file-list
    argument), so feed it only this run's selected/interval-sampled images by
    symlinking them into a scratch directory with zero-padded names -- that
    keeps Demo's natural_sort_key ordering identical to our own image_paths
    order without copying image bytes."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for index, image_path in enumerate(image_paths):
        source = Path(image_path).expanduser().resolve()
        target = dest_dir / f"{index:06d}{source.suffix.lower()}"
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source)


def load_model(ckpt_path: str, device: str):
    from amb3r.model_zoo import load_model as amb3r_load_model

    model = amb3r_load_model("amb3r", ckpt_path=ckpt_path)
    model.eval()
    return model.to(device)


def preprocess_amb3r(image_paths: list[str], *, args: argparse.Namespace, device=None, run_name: str = "", **_unused):
    """AMB3R's preprocess_<method>(), registered below for prepare_model_input("AMB3R", ...).
    Unconditional (no --spatial_policy toggle). Symlinks the real images into a temp directory
    AMB3R's own Demo dataset reads (disable_crop=True, so its loader never crops), builds the
    matching per-image SpatialTransform list, and returns the views AMB3R's model forward pass
    needs. Returns ({"views": ..., "views_all": ..., "temp_images_dir": Path}, transforms).
    """
    from amb3r.datasets import Demo
    from PIL import Image, ImageOps
    from torch.utils.data import DataLoader

    # Fixed-resolution inputs (full-slam) keep their own size (source == target, an exact no-op
    # regardless of disable_crop); otherwise 518x392.
    target_w, target_h = fixed_input_size(image_paths) or (AMB3R_WIDTH, AMB3R_HEIGHT)
    transforms: list[SpatialTransform] = []
    for image_path in image_paths:
        # Match AMB3R's own Demo dataset (thirdparty/dust3r/utils/image.py's load loop, which
        # calls exif_transpose) exactly -- an un-transposed probe would report width/height
        # swapped for any image whose EXIF orientation tag requires a 90/270 degree rotation.
        with Image.open(image_path) as image:
            source_w, source_h = ImageOps.exif_transpose(image).size
        transforms.append(SpatialTransform(source_wh=(source_w, source_h), resized_wh=(target_w, target_h)))

    suffix = f"_{run_name}" if run_name else ""
    temp_images_dir = args.output_path / f"_amb3r_input_images{suffix}"
    materialize_symlinked_images(image_paths, temp_images_dir)
    data = Demo(
        ROOT=str(temp_images_dir),
        resolution=(target_w, target_h),
        num_seq=1,
        full_video=True,
        kf_every=1,
        disable_crop=True,
        max_images=len(image_paths),
    )
    dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)
    views, views_all = next(iter(dataloader))
    for key in list(views_all.keys()):
        views_all[key] = views_all[key].to(device)
    return {"views": views, "views_all": views_all, "temp_images_dir": temp_images_dir}, transforms


register_preprocessor("AMB3R", preprocess_amb3r)


def run_amb3r(
    model,
    image_paths: list[str],
    args: argparse.Namespace,
    device: str,
    run_name: str = "",
    keep_intermediate: bool = False,
):
    import torch

    model_input, transforms = prepare_model_input("AMB3R", image_paths, args=args, device=device, run_name=run_name)
    temp_images_dir = model_input["temp_images_dir"]
    try:
        views, views_all = model_input["views"], model_input["views_all"]
        record_model_input(views[0]["img"].shape[-1], views[0]["img"].shape[-2])

        if device == "cuda":
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        elif device == "cpu":
            # The checkpoint's weights are bf16; without autocast the input stays fp32 and
            # conv2d raises "Input type (float) and bias type (c10::BFloat16) should be the same".
            autocast_ctx = torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        else:
            autocast_ctx = nullcontext()
        with torch.no_grad(), autocast_ctx:
            res_all = model(views_all)
        res = res_all[-1]

        pose = res["pose"][0].float().detach().cpu().numpy()
        extrinsic = res["extrinsic"][0].float().detach().cpu().numpy()
        intrinsic = res["intrinsic"][0].float().detach().cpu().numpy()
        depth = res["depth"][0].float().detach().cpu().numpy()
        world_points = res["world_points"][0].float().detach().cpu().numpy()
        world_points_conf = res["world_points_conf"][0].float().detach().cpu().numpy()
        # decode_heads' own docstring: images are [0, 1] at this point.
        images_out = res["images"][0].float().detach().cpu().numpy()
    finally:
        cleanup_intermediate_paths([temp_images_dir], keep=keep_intermediate)

    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if world_points_conf.ndim == 4 and world_points_conf.shape[-1] == 1:
        world_points_conf = world_points_conf[..., 0]

    colors = (images_out.transpose(0, 2, 3, 1).clip(0.0, 1.0) * 255.0).astype(np.uint8)
    return extrinsic, intrinsic, pose, depth, world_points, world_points_conf, colors, transforms


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

    add_code_path(args.amb3r_code_path, ("amb3r",))

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    args.output_path.mkdir(parents=True, exist_ok=True)
    model = load_model(args.ckpt_path, device)
    started_at = mark_model_loaded()
    keep_intermediate = should_keep_intermediate(args)

    if args.depth_only:
        if not args.export_depth:
            raise ValueError("--depth_only requires --export_depth")
        images_per_run = args.max_images_per_run if args.max_images_per_run > 0 else len(image_paths)
        pose_extrinsics: list[np.ndarray] = []
        pose_intrinsics: list[np.ndarray] = []
        pose_output_size: tuple[int, int] | None = None
        for batch_index, batch_image_paths in enumerate(batched_image_paths(image_paths, images_per_run)):
            extrinsic, intrinsic, _pose, depth, _points, conf, _colors, transforms = run_amb3r(
                model,
                batch_image_paths,
                args,
                device,
                run_name=f"batch_{batch_index:04d}",
                keep_intermediate=keep_intermediate,
            )
            export_spatial_depth_predictions(args.output_path, batch_image_paths, depth, conf, transforms)
            if args.export_colmap:
                pose_extrinsics.append(extrinsic)
                restored_intrinsic, width, height = restore_camera_export(intrinsic, transforms)
                pose_intrinsics.append(restored_intrinsic)
                pose_output_size = (width, height)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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
        peak_memory = (
            torch.cuda.max_memory_allocated() / (1024.0**2) if torch.cuda.is_available() else 0.0
        )
        write_adapter_metrics(
            output_path=args.output_path,
            method=METHOD,
            image_paths=image_paths,
            started_at=started_at,
            checkpoint=args.ckpt_path or None,
            depth_only=True,
            metrics_extra={
                "Peak GPU Memory": peak_memory,
                "max_images_per_run": images_per_run,
                "resolution": f"{AMB3R_WIDTH}x{AMB3R_HEIGHT}",
            },
        )
        return

    extrinsic, intrinsic, _pose, depth, points, confidence, colors, _transforms = run_amb3r(
        model, image_paths, args, device, keep_intermediate=keep_intermediate
    )
    conf_thresh = args.depth_conf_thresh if args.depth_conf_thresh is not None else args.conf_thresh
    peak_memory = (
        torch.cuda.max_memory_allocated() / (1024.0**2) if torch.cuda.is_available() else 0.0
    )

    export_standard_artifacts(
        method=METHOD,
        output_path=args.output_path,
        images_dir=images_dir,
        image_paths=image_paths,
        depths=depth,
        points=points,
        colors=colors,
        confidence=confidence,
        extrinsics_w2c=extrinsic,
        intrinsics=intrinsic,
        export_depth=args.export_depth,
        export_colmap=args.export_colmap,
        depth_only=False,
        colmap_model_subdir=args.colmap_model_subdir,
        max_points=args.max_points,
        conf_thresh=conf_thresh,
        started_at=started_at,
        checkpoint=args.ckpt_path or None,
        metrics_extra={
            "Peak GPU Memory": peak_memory,
            "resolution": f"{AMB3R_WIDTH}x{AMB3R_HEIGHT}",
        },
    )


if __name__ == "__main__":
    run()
