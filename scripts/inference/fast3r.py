#!/usr/bin/env python
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Run Fast3R and export benchmark prediction artifacts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from adapter_utils import (
    canonical_image_root,
    mark_model_loaded,
    record_model_input,
    add_code_path,
    as_numpy,
    c2w_to_w2c,
    export_camera_pose_priors,
    export_spatial_depth_predictions,
    export_standard_artifacts,
    find_image_paths,
    intrinsics_from_focals,
    prepare_model_input,
    preprocess_images,
    print_dry_run,
    register_preprocessor,
    restore_camera_export,
    select_interval,
    single_view_mode,
    write_adapter_metrics,
)

METHOD = "FAST3R"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Fast3R benchmark adapter.")
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=os.environ.get("FAST3R_CKPT_PATH",
                               "jedyang97/Fast3R_ViT_Large_512"),
    )
    parser.add_argument(
        "--fast3r_code_path",
        type=Path,
        default=Path(os.environ["FAST3R_CODE_DIR"])
        if "FAST3R_CODE_DIR" in os.environ else None,
    )
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--image_load_resolution", type=int, default=512)
    parser.add_argument("--max_images_per_run", type=int, default=0)
    parser.add_argument("--max_points", type=int, default=1_000_000)
    parser.add_argument("--conf_thresh", type=float, default=0.0)
    parser.add_argument("--depth_conf_thresh", type=float, default=None)
    parser.add_argument("--niter_pnp", type=int, default=100)
    parser.add_argument("--is_lightning_checkpoint", action="store_true")
    parser.add_argument("--export_depth", action="store_true")
    parser.add_argument("--export_colmap", action="store_true")
    parser.add_argument("--depth_only", action="store_true")
    parser.add_argument("--colmap_model_subdir", type=str, default="0")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def stack_prediction_key(preds: list[dict], key: str) -> np.ndarray:
    arrays = []
    for pred in preds:
        value = pred[key]
        array = as_numpy(value)
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        arrays.append(array)
    return stack_spatial_arrays(arrays)


def stack_spatial_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        raise ValueError("No arrays to stack")
    if len({array.shape for array in arrays}) == 1:
        return np.stack(arrays, axis=0)
    if not all(array.ndim >= 2 for array in arrays):
        return np.stack(arrays, axis=0)
    target_h = min(array.shape[0] for array in arrays)
    target_w = min(array.shape[1] for array in arrays)
    cropped = []
    for array in arrays:
        height, width = array.shape[:2]
        top = max(0, (height - target_h) // 2)
        left = max(0, (width - target_w) // 2)
        cropped.append(array[top:top + target_h, left:left + target_w, ...])
    return np.stack(cropped, axis=0)


def colors_from_views(views: list[dict]) -> np.ndarray:
    colors = []
    for view in views:
        image = as_numpy(view["img"])
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        colors.append(((image + 1.0) * 127.5).clip(0, 255).astype(np.uint8))
    return stack_spatial_arrays(colors)


def preprocess_fast3r(image_paths: list[str], *, args, device, **_unused):
    """Fast3R's preprocess_<method>(), registered below for prepare_model_input("FAST3R", ...).
    Unconditional (no --spatial_policy toggle): builds Fast3R's own view-dict format
    (img/true_shape/idx/instance, matching its vendored fast3r.dust3r.utils.image.load_images's
    output exactly -- that fork has the identical resize-then-center-crop-to-16px-grid as CUT3R's
    own load_images, same source, same lossy square-input special case, confirmed by reading it
    directly) from a padded tensor instead, bypassing that loader entirely so no source pixel is
    ever discarded. Returns (views, transforms)."""
    import torch

    # exif_transpose=True to match fast3r.dust3r.utils.image.load_images's own convention (the
    # loader this bypasses still applies it).
    images, transforms = preprocess_images(image_paths,
                                           method="vggt_omega",
                                           size=args.resolution,
                                           mode="pad",
                                           exif_transpose=True)
    # fast3r's ImgNorm is ToTensor (-> [0,1]) + Normalize(0.5, 0.5) (-> [-1,1]); preprocess_images
    # already returns [0,1], so only the normalization step remains.
    images = (images * 2.0 - 1.0).to(device)
    views = []
    for index in range(images.shape[0]):
        frame = images[index:index + 1]
        height, width = frame.shape[-2:]
        views.append({
            "img": frame,
            "true_shape": np.int32([[height, width]]),
            "idx": index,
            "instance": str(index),
        })
    return views, transforms


register_preprocessor("FAST3R", preprocess_fast3r)


def extend_view_position_embeddings(model, num_views: int) -> dict:
    """Extend Fast3R view-position tables for scenes with more than 1000 images."""
    import torch
    from fast3r.croco.models.pos_embed import get_1d_sincos_pos_embed_from_grid

    decoder = getattr(model, "decoder", None)
    if decoder is None:
        return {}

    metadata = {"num_views": int(num_views)}
    if hasattr(decoder,
               "image_idx_emb") and decoder.image_idx_emb.shape[0] < num_views:
        current = decoder.image_idx_emb
        extended = torch.from_numpy(
            get_1d_sincos_pos_embed_from_grid(current.shape[1],
                                              np.arange(num_views)))
        decoder.image_idx_emb = extended.to(device=current.device,
                                            dtype=current.dtype)
        metadata["image_idx_emb_extended_from"] = int(current.shape[0])
        metadata["image_idx_emb_extended_to"] = int(num_views)

    if (hasattr(decoder, "precomputed_freqs_cis") and
            decoder.precomputed_freqs_cis.shape[0] < num_views and
            hasattr(decoder, "_precompute_freqs_cis")):
        current = decoder.precomputed_freqs_cis
        decoder.precomputed_freqs_cis = decoder._precompute_freqs_cis(
            num_views).to(device=current.device)
        metadata["precomputed_freqs_cis_extended_from"] = int(current.shape[0])
        metadata["precomputed_freqs_cis_extended_to"] = int(num_views)

    return metadata


def run() -> None:
    args = parse_args()
    images_dir = canonical_image_root(args.data_path)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images directory not found: {images_dir}")
    image_paths = select_interval(find_image_paths(images_dir), args.interval,
                                  args.max_images_per_run)
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")
    if args.dry_run:
        print_dry_run(args, METHOD, len(image_paths))
        return

    add_code_path(args.fast3r_code_path, ("fast3r",))

    import torch
    from fast3r.dust3r.inference_multiview import inference
    from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
    from fast3r.utils.checkpoint_utils import load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model, _lit_module = load_model(
        args.ckpt_path,
        device=device,
        is_lightning_checkpoint=args.is_lightning_checkpoint,
    )
    single_view = single_view_mode()
    position_metadata = extend_view_position_embeddings(
        model, 1 if single_view else len(image_paths))
    started_at = mark_model_loaded()

    def predict(view_paths: list[str]):
        """Fast3R on ``view_paths`` as one multi-view input."""
        images, transforms = prepare_model_input("FAST3R",
                                                 view_paths,
                                                 args=args,
                                                 device=device)
        record_model_input(images[0]["img"].shape[-1],
                           images[0]["img"].shape[-2])
        for index, (view, image_path) in enumerate(
                zip(images, view_paths, strict=True)):
            view.setdefault("dataset", args.data_path.parent.name)
            view.setdefault("label", args.data_path.name)
            view.setdefault("instance", Path(image_path).name)
            view.setdefault("idx", index)
        output_dict, _profiling_info = inference(
            images,
            model,
            device,
            dtype=torch.float32,
            verbose=True,
            profiling=True,
        )
        preds = output_dict["preds"]
        poses_c2w_batch, estimated_focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
            preds,
            niter_PnP=args.niter_pnp,
            focal_length_estimation_method="first_view_from_global_head",
        )
        c2w = as_numpy(poses_c2w_batch[0]).astype(np.float32, copy=False)
        view_extrinsics_w2c = c2w_to_w2c(c2w)
        view_points = stack_prediction_key(preds, "pts3d_in_other_view")
        if "pts3d_local" in preds[0]:
            view_depths = stack_prediction_key(preds, "pts3d_local")[..., 2]
            view_confidence = stack_prediction_key(
                preds, "conf_local") if "conf_local" in preds[0] else None
        else:
            camera_points = np.einsum(
                "nij,nhwj->nhwi",
                view_extrinsics_w2c[:, :3, :3],
                view_points,
            ) + view_extrinsics_w2c[:, None, None, :3, 3]
            view_depths = camera_points[..., 2]
            view_confidence = stack_prediction_key(
                preds, "conf") if "conf" in preds[0] else None
        view_height, view_width = view_depths.shape[1:]
        view_intrinsics = intrinsics_from_focals(estimated_focals[0],
                                                 height=view_height,
                                                 width=view_width)
        return (view_depths, view_points,
                colors_from_views(output_dict["views"]), view_confidence,
                view_extrinsics_w2c, view_intrinsics, transforms)

    if single_view:
        # MonoDepth: every image is its own one-view input; the per-image results are stacked.
        per_image = [predict([path]) for path in image_paths]
        depths, points, colors, confidence, extrinsics_w2c, intrinsics = (
            None if any(result[index] is None
                        for result in per_image) else np.concatenate(
                            [result[index]
                             for result in per_image], axis=0)
            for index in range(6))
        transforms = [
            transform for result in per_image for transform in result[6]
        ]
    else:
        depths, points, colors, confidence, extrinsics_w2c, intrinsics, transforms = predict(
            image_paths)
    conf_thresh = args.depth_conf_thresh if args.depth_conf_thresh is not None else args.conf_thresh
    peak_memory = (torch.cuda.max_memory_allocated() /
                   (1024.0**2) if torch.cuda.is_available() else 0.0)

    if args.depth_only:
        # Unlike export_standard_artifacts's plain export_depth_predictions (which would write
        # depths at the padded preprocess_fast3r() canvas resolution), restore each image's own
        # true source resolution here -- the same restore_prediction/restore_camera_export
        # machinery already proven on vggt.py/fastvggt.py/da3.py/vggt_omega.py/cut3r.py/etc.
        if args.export_depth:
            export_spatial_depth_predictions(args.output_path, image_paths,
                                             depths, confidence, transforms)
        if args.export_colmap:
            restored_intrinsics, width, height = restore_camera_export(
                intrinsics, transforms)
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
            metrics_extra={
                "Peak GPU Memory": peak_memory,
                "resolution": args.resolution,
                "niter_pnp": args.niter_pnp,
                "is_lightning_checkpoint": args.is_lightning_checkpoint,
                "max_images_per_run": args.max_images_per_run,
                "single_view": single_view,
                **position_metadata,
            },
        )
        return

    # Dense/points3D export derives pixel-indexed geometry from the model's own padded grid -- not
    # restored to source resolution, same restriction every other patched adapter's "shared"
    # policy has for non-depth-only export.
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
        checkpoint=args.ckpt_path,
        metrics_extra={
            "Peak GPU Memory": peak_memory,
            "resolution": args.resolution,
            "niter_pnp": args.niter_pnp,
            "is_lightning_checkpoint": args.is_lightning_checkpoint,
            "max_images_per_run": args.max_images_per_run,
            "single_view": single_view,
            **position_metadata,
        },
    )


if __name__ == "__main__":
    run()
