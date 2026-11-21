# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Run Déjà View / DVLT and export benchmark prediction artifacts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from adapter_utils import (
    SpatialTransform,
    canonical_image_root,
    mark_model_loaded,
    add_code_path,
    as_numpy,
    c2w_to_w2c,
    colors_from_torch_images,
    export_camera_pose_priors,
    export_spatial_depth_predictions,
    export_standard_artifacts,
    find_image_paths,
    prepare_model_input,
    print_dry_run,
    register_preprocessor,
    restore_camera_export,
    select_interval,
    single_view_mode,
    write_adapter_metrics,
)


METHOD = "DEJAVIEW"


# Shared method checkouts (the engine's METHOD_CODE_ROOT); the engine also passes the *_CODE_PATH below.
# see spatialarena/paths.py
SPATIALARENA_ROOT = os.path.expanduser(os.environ.get("SPATIALARENA_ROOT", "~/spatialarena"))
METHOD_CODE_ROOT = Path(
    os.environ.get("METHOD_CODE_ROOT", f"{SPATIALARENA_ROOT}/benchmarks/_model_codes")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--ckpt_path", default=os.environ.get("DEJAVIEW_CKPT_PATH", "nvidia/dvlt"))
    parser.add_argument(
        "--dejaview_code_path",
        type=Path,
        default=Path(os.environ["DEJAVIEW_CODE_PATH"])
        if "DEJAVIEW_CODE_PATH" in os.environ
        else METHOD_CODE_ROOT / "dvlt",
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


def to_4x4_camera_to_world(camera_to_worlds):
    import torch
    from dvlt.common.pose import to4x4

    c2w = camera_to_worlds
    if c2w.shape[-2:] == (3, 4):
        c2w = to4x4(c2w)
    if c2w.shape[-2:] != (4, 4):
        raise ValueError(f"Expected DVLT camera_to_worlds with shape Sx3x4 or Sx4x4, got {tuple(c2w.shape)}")
    if isinstance(c2w, torch.Tensor):
        return c2w.detach().float().cpu().numpy()
    return as_numpy(c2w).astype("float32", copy=False)


def preprocess_dejaview(image_paths: list[str], *, args, device=None, patch_size: int = 14, **_unused):
    """DejaView's preprocess_<method>(), registered below for prepare_model_input("DEJAVIEW", ...).

    Builds DVLT's own batch dict (DataField.IMAGES/"gradio_valid_pixels", matching
    dvlt.util.preprocess.preprocess_images's output exactly) without its lossy per-image step.

    preprocess_images resizes each image's longest side to img_size (preserving aspect), then
    *center-crops* both dimensions down to the nearest lower patch_size multiple before padding
    the whole batch to a shared canvas -- real (if bounded, under one patch) content loss. This
    builds the same shared canvas directly: resize each image (unchanged), then pad (not crop)
    straight to the batch's shared, already patch_size-divisible canvas (the max of each image's
    own *ceiling*-rounded size) -- the batch-level pad step preprocess_images already does is
    reused as-is, since it was already lossless.
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageOps
    from dvlt.common.constants import DataField

    resized_images = []
    source_and_resized = []
    ceil_dims = []
    for path in image_paths:
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        source_w, source_h = img.size
        scale = args.image_load_resolution / max(source_h, source_w)
        new_h, new_w = max(1, round(source_h * scale)), max(1, round(source_w * scale))
        resized_images.append(img.resize((new_w, new_h), Image.Resampling.LANCZOS))
        source_and_resized.append((source_w, source_h, new_w, new_h))
        ceil_dims.append(
            (((new_w + patch_size - 1) // patch_size) * patch_size,
             ((new_h + patch_size - 1) // patch_size) * patch_size)
        )

    max_w = max(ceil_w for ceil_w, _ in ceil_dims)
    max_h = max(ceil_h for _, ceil_h in ceil_dims)

    aligned: list[torch.Tensor] = []
    valid_masks: list[torch.Tensor] = []
    transforms: list[SpatialTransform] = []
    for resized, (source_w, source_h, new_w, new_h) in zip(resized_images, source_and_resized):
        tensor = torch.from_numpy(np.array(resized)).permute(2, 0, 1).float() / 255.0
        pad_w, pad_h = max_w - new_w, max_h - new_h
        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
        aligned.append(F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), value=0.0))
        valid = F.pad(torch.ones(1, new_h, new_w, dtype=torch.float32), (pad_left, pad_right, pad_top, pad_bottom),
                      value=0.0)
        valid_masks.append(valid.squeeze(0).bool())
        transforms.append(
            SpatialTransform(
                source_wh=(source_w, source_h),
                resized_wh=(new_w, new_h),
                padding=(pad_left, pad_top, pad_right, pad_bottom),
            )
        )

    images = torch.stack(aligned, dim=0).unsqueeze(0).to(device)
    gradio_valid_pixels = torch.stack(valid_masks, dim=0).unsqueeze(0).to(device)
    return {DataField.IMAGES: images, "gradio_valid_pixels": gradio_valid_pixels}, transforms


register_preprocessor("DEJAVIEW", preprocess_dejaview)


def run() -> None:
    args = parse_args()
    images_dir = canonical_image_root(args.data_path)
    # select_interval also applies args.max_images_per_run as a total-frame
    # cap: the chunking loop below (chunk_size == max_images_per_run) runs
    # unconditionally in both --depth_only and full mode, but each chunk is
    # an independent forward pass with its own arbitrary coordinate frame
    # and results are simply concatenated -- no cross-chunk registration.
    # Capping total frames first means the loop always produces exactly one
    # chunk; doesn't change per-chunk memory footprint since chunk_size
    # (already <= max_images_per_run) is unchanged.
    image_paths = select_interval(find_image_paths(images_dir), args.interval, args.max_images_per_run)
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")
    if args.dry_run:
        print_dry_run(args, METHOD, len(image_paths))
        return

    add_code_path(args.dejaview_code_path, ("dvlt",))
    source_dir = args.dejaview_code_path / "src"
    if source_dir.is_dir():
        add_code_path(source_dir, ("dvlt",))

    import torch
    if os.environ.get("DEJAVIEW_DISABLE_TORCH_COMPILE", "1") == "1":
        torch.compile = lambda fn, *args, **kwargs: fn
    from accelerate import Accelerator
    from dvlt.model.dvlt.model import DVLT

    accelerator = Accelerator(mixed_precision="bf16" if torch.cuda.is_available() else "no")
    model = DVLT(img_size=args.image_load_resolution)
    model.load_pretrained(args.ckpt_path, strict=True)
    model.setup_test(accelerator)

    started_at = mark_model_loaded()
    chunk_size = args.max_images_per_run if args.max_images_per_run > 0 else len(image_paths)
    if single_view_mode():
        chunk_size = 1  # MonoDepth: one image per forward pass, never a multi-view chunk
    depth_chunks = []
    point_chunks = []
    color_chunks = []
    confidence_chunks = []
    extrinsic_chunks = []
    intrinsic_chunks = []
    all_transforms: list[SpatialTransform] = []
    for start in range(0, len(image_paths), chunk_size):
        chunk_paths = image_paths[start : start + chunk_size]
        batch, chunk_transforms = prepare_model_input(
            "DEJAVIEW", chunk_paths, args=args, device=accelerator.device
        )
        all_transforms.extend(chunk_transforms)
        with torch.no_grad(), accelerator.autocast():
            predictions = model.predict(batch, accelerator)

        depths = as_numpy(predictions["depths"][0]).astype("float32", copy=False)
        depth_chunks.append(depths)
        confidence = predictions.get("depth_conf", None)
        if confidence is None:
            confidence = predictions.get("depths_conf", None)
        if isinstance(confidence, (list, tuple)):
            confidence = confidence[0]
        if confidence is not None:
            confidence = as_numpy(confidence).squeeze()
            # A one-image chunk squeezes to (H, W); keep the frame axis so chunks concatenate per frame.
            confidence_chunks.append(confidence[None] if confidence.ndim == 2 else confidence)

        if args.export_colmap:
            cameras = predictions["cameras"][0]
            c2w = to_4x4_camera_to_world(cameras.camera_to_worlds)
            extrinsic_chunks.append(c2w_to_w2c(c2w))
            intrinsic_chunks.append(
                as_numpy(cameras.get_intrinsics_matrices()).astype("float32", copy=False)
            )
        if not args.depth_only:
            point_chunks.append(
                as_numpy(predictions["world_points"][0]).astype("float32", copy=False)
            )
            color_chunks.append(colors_from_torch_images(batch["images"]))

    confidence = np.concatenate(confidence_chunks, axis=0) if confidence_chunks else None
    extrinsics_w2c = np.concatenate(extrinsic_chunks, axis=0) if extrinsic_chunks else None
    intrinsics = np.concatenate(intrinsic_chunks, axis=0) if intrinsic_chunks else None
    metrics_extra = {"track": args.track, "adapter": "dvlt.model.dvlt.model.DVLT"}

    if args.depth_only:
        # Restore depth/camera predictions to each image's true source resolution -- the same
        # restore_prediction/restore_camera_export machinery already proven on
        # vggt.py/fastvggt.py/da3.py/vggt_omega.py/cut3r.py/etc.
        if args.export_depth:
            depths = np.concatenate(depth_chunks, axis=0)
            export_spatial_depth_predictions(args.output_path, image_paths, depths, confidence, all_transforms)
        if args.export_colmap and extrinsics_w2c is not None and intrinsics is not None:
            restored_intrinsics, width, height = restore_camera_export(intrinsics, all_transforms)
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
        return

    # Dense/points3D export derives pixel-indexed geometry from the model's own padded grid -- not
    # restored to source resolution, same restriction every other patched adapter's "shared"
    # policy has for non-depth-only export.
    depths = np.concatenate(depth_chunks, axis=0)
    points = np.concatenate(point_chunks, axis=0) if point_chunks else None
    colors = np.concatenate(color_chunks, axis=0) if color_chunks else None
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
        conf_thresh=args.depth_conf_thresh if args.depth_conf_thresh is not None else args.conf_thresh,
        started_at=started_at,
        checkpoint=args.ckpt_path,
        metrics_extra=metrics_extra,
    )


if __name__ == "__main__":
    run()
