#!/usr/bin/env python
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Run FastVGGT and export benchmark prediction artifacts."""

from __future__ import annotations

import argparse
import inspect
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from adapter_utils import (
    canonical_image_root,
    mark_model_loaded,
    record_model_input,
    add_code_path,
    batched_image_paths,
    cleanup_intermediate_paths,
    colors_from_torch_images,
    export_camera_pose_priors,
    export_depth_predictions,
    export_depths_for_policy,
    export_standard_artifacts,
    find_image_paths,
    materialize_resized_images,
    prepare_model_input,
    preprocess_images,
    print_dry_run,
    register_preprocessor,
    restore_camera_export,
    select_interval,
    should_keep_intermediate,
    spatial_policy_metrics,
    unproject_depths_to_world,
    validate_shared_spatial_policy_depth_only,
    write_adapter_metrics,
)

METHOD = "FASTVGGT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FastVGGT benchmark adapter.")
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=os.environ.get("FASTVGGT_CKPT_PATH",
                               "./ckpt/model_tracker_fixed_e20.pt"),
    )
    parser.add_argument(
        "--fastvggt_code_path",
        type=Path,
        default=Path(os.environ["FASTVGGT_CODE_DIR"])
        if "FASTVGGT_CODE_DIR" in os.environ else None,
    )
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=518)
    parser.add_argument("--image_load_resolution", type=int, default=518)
    parser.add_argument("--max_images_per_run", type=int, default=0)
    parser.add_argument("--max_points", type=int, default=1_000_000)
    parser.add_argument("--conf_thresh", type=float, default=0.0)
    parser.add_argument("--depth_conf_thresh", type=float, default=None)
    parser.add_argument("--merging", type=int, default=0)
    parser.add_argument("--merge_ratio", type=float, default=0.9)
    parser.add_argument("--export_depth", action="store_true")
    parser.add_argument("--export_colmap", action="store_true")
    parser.add_argument("--depth_only", action="store_true")
    parser.add_argument("--colmap_model_subdir", type=str, default="0")
    parser.add_argument("--keep_intermediate", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--spatial_policy",
        choices=("legacy", "shared"),
        default="legacy",
        help=
        ("'legacy' (default): unchanged two-stage resize (materialize_resized_images pad-to-"
         "square at --image_load_resolution, written to disk, then reopened and plain-resized "
         "to --resolution), depth exported at that resolution, not restored to source. "
         "'shared': a single direct pad-to-square resize straight to --resolution via "
         "adapter_utils.plan_spatial_transform(method='fastvggt'), entirely in-memory, with "
         "depth exported at the true original per-image resolution."),
    )
    return parser.parse_args()


def preprocess_fastvggt(image_paths: list[str],
                        *,
                        args: argparse.Namespace,
                        device: str,
                        batch_suffix: str = ""):
    """FastVGGT's preprocess_<method>(), registered below for prepare_model_input("FASTVGGT", ...).

    --spatial_policy legacy (default): materialize_resized_images pads each image onto a square
    canvas on disk (patch 14); run_fastvggt loads and further resizes those files itself.
    --spatial_policy shared: preprocess_images builds the padded tensor directly in memory, one
    resize straight to --resolution, restorable.

    `batch_suffix`: the depth_only loop's temp directory for each batch must not collide with the
    others', so it passes e.g. "_batch_0003"; the single-batch (non depth_only) call site leaves
    it at the default, giving the historical "_resized_input_images" name.

    Returns ({"paths": ..., "images": tensor or None, "resized_dir": Path or None}, transforms):
    pass model_input["paths"]/["images"] straight through to run_fastvggt(..., images=...); clean
    up model_input["resized_dir"] afterward the same way the legacy path's temp files always were.
    """
    if args.spatial_policy == "shared":
        images, transforms = preprocess_images(image_paths,
                                               method="fastvggt",
                                               size=args.resolution,
                                               mode="pad")
        return {
            "paths": image_paths,
            "images": images.to(device),
            "resized_dir": None
        }, transforms
    resized_paths = materialize_resized_images(
        image_paths,
        args.output_path / f"_resized_input_images{batch_suffix}",
        args.image_load_resolution,
    )
    resized_dir = Path(resized_paths[0]).parent
    return {
        "paths": resized_paths,
        "images": None,
        "resized_dir": resized_dir
    }, None


register_preprocessor("FASTVGGT", preprocess_fastvggt)


def run_fastvggt(
    model,
    image_paths: list[str],
    args: argparse.Namespace,
    device: str,
    images: "torch.Tensor | None" = None,
):
    import torch
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    if images is None:
        from PIL import Image
        from torchvision.transforms import functional as TF

        tensors = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                image = image.convert("RGB").resize(
                    (args.resolution, args.resolution),
                    Image.Resampling.BICUBIC,
                )
                tensors.append(TF.to_tensor(image))
        if len(tensors) < 1:
            raise ValueError("FastVGGT did not load any images")
        vgg_input = torch.stack(tensors, dim=0)
    else:
        vgg_input = images
    record_model_input(vgg_input.shape[-1], vgg_input.shape[-2])
    if hasattr(model, "update_patch_dimensions"):
        patch_width = vgg_input.shape[-1] // 14
        patch_height = vgg_input.shape[-2] // 14
        model.update_patch_dimensions(patch_width, patch_height)
    vgg_input = vgg_input.to(device)

    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability(
    )[0] >= 8 else torch.float16
    autocast_context = torch.amp.autocast(
        "cuda", dtype=dtype) if device == "cuda" else nullcontext()
    with torch.no_grad():
        with autocast_context:
            forward_signature = inspect.signature(model.forward)
            forward_kwargs = {}
            if "image_paths" in forward_signature.parameters:
                forward_kwargs["image_paths"] = [
                    Path(p).name for p in image_paths
                ]
            predictions = model(vgg_input, **forward_kwargs)
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"],
        (vgg_input.shape[-2], vgg_input.shape[-1]),
    )
    extrinsic_np = extrinsic[0].detach().float().cpu().numpy()
    intrinsic_np = intrinsic[0].detach().float().cpu().numpy()
    depths = predictions["depth"][0].detach().float().cpu().numpy()
    depth_conf = predictions["depth_conf"][0].detach().float().cpu().numpy()
    points = unproject_depths_to_world(depths, extrinsic_np, intrinsic_np)
    colors = colors_from_torch_images(vgg_input)
    return extrinsic_np, intrinsic_np, depths, depth_conf, points, colors


def run() -> None:
    args = parse_args()
    validate_shared_spatial_policy_depth_only(
        is_shared=args.spatial_policy == "shared", depth_only=args.depth_only)
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

    add_code_path(args.fastvggt_code_path, ("vggt",))

    import torch
    from vggt.models.vggt import VGGT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    init_signature = inspect.signature(VGGT)
    init_kwargs = {}
    if "merging" in init_signature.parameters:
        init_kwargs["merging"] = args.merging
    if "merge_ratio" in init_signature.parameters:
        init_kwargs["merge_ratio"] = args.merge_ratio
    if "vis_attn_map" in init_signature.parameters:
        init_kwargs["vis_attn_map"] = False
    model = VGGT(**init_kwargs)
    ckpt_path = Path(args.ckpt_path).expanduser()
    if ckpt_path.is_file():
        state = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(state, dict):
            state = state.get("model", state.get("state_dict", state))
        model.load_state_dict(state, strict=False)
    else:
        model = VGGT.from_pretrained(args.ckpt_path)
    model = model.to(device).eval()

    started_at = mark_model_loaded()
    conf_thresh = args.depth_conf_thresh if args.depth_conf_thresh is not None else args.conf_thresh
    keep_intermediate = should_keep_intermediate(args)
    if args.depth_only:
        if not args.export_depth:
            raise ValueError("--depth_only requires --export_depth")
        images_per_run = args.max_images_per_run if args.max_images_per_run > 0 else len(
            image_paths)
        peak_memory = 0.0
        pose_extrinsics: list[np.ndarray] = []
        pose_intrinsics: list[np.ndarray] = []
        pose_output_size: tuple[int, int] | None = None
        for batch_index, batch_original_paths in enumerate(
                batched_image_paths(image_paths, images_per_run)):
            model_input, transforms = prepare_model_input(
                "FASTVGGT",
                batch_original_paths,
                args=args,
                device=device,
                batch_suffix=f"_batch_{batch_index:04d}")
            resized_dir = model_input["resized_dir"]
            extrinsic, intrinsic, depths, depth_conf, _points, _colors = run_fastvggt(
                model,
                model_input["paths"],
                args,
                device,
                images=model_input["images"])
            export_depths_for_policy(
                args.output_path,
                batch_original_paths,
                depths,
                depth_conf,
                transforms,
                legacy_export=lambda: export_depth_predictions(
                    args.output_path, batch_original_paths, depths, depth_conf),
            )
            if args.export_colmap and extrinsic is not None and intrinsic is not None:
                pose_extrinsics.append(np.asarray(extrinsic))
                if transforms is not None:
                    restored_intrinsic, width, height = restore_camera_export(
                        intrinsic, transforms)
                    pose_intrinsics.append(restored_intrinsic)
                    pose_output_size = (width, height)
                else:
                    pose_intrinsics.append(np.asarray(intrinsic))
                    if pose_output_size is None:
                        depth_array = np.asarray(depths)
                        pose_output_size = (depth_array.shape[2],
                                            depth_array.shape[1])
            cleanup_intermediate_paths([resized_dir], keep=keep_intermediate)
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
            checkpoint=args.ckpt_path,
            depth_only=True,
            metrics_extra={
                "Peak GPU Memory":
                    peak_memory,
                "merging":
                    args.merging,
                "merge_ratio":
                    args.merge_ratio,
                "max_images_per_run":
                    images_per_run,
                "image_load_resolution":
                    args.image_load_resolution,
                "resized_input":
                    True,
                **spatial_policy_metrics(args.spatial_policy,
                                         is_shared=args.spatial_policy == "shared"),
            },
        )
        return

    model_input, _transforms = prepare_model_input("FASTVGGT",
                                                   image_paths,
                                                   args=args,
                                                   device=device)
    resized_dir = model_input["resized_dir"]
    extrinsic_np, intrinsic_np, depths, depth_conf, points, colors = run_fastvggt(
        model, model_input["paths"], args, device, images=model_input["images"])
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
        confidence=depth_conf,
        extrinsics_w2c=extrinsic_np,
        intrinsics=intrinsic_np,
        export_depth=args.export_depth,
        export_colmap=args.export_colmap,
        depth_only=args.depth_only,
        colmap_model_subdir=args.colmap_model_subdir,
        max_points=args.max_points,
        conf_thresh=conf_thresh,
        started_at=started_at,
        checkpoint=args.ckpt_path,
        metrics_extra={
            "Peak GPU Memory":
                peak_memory,
            "merging":
                args.merging,
            "merge_ratio":
                args.merge_ratio,
            "max_images_per_run":
                args.max_images_per_run,
            "image_load_resolution":
                args.image_load_resolution,
            "resized_input":
                True,
            **spatial_policy_metrics(args.spatial_policy,
                                     is_shared=args.spatial_policy == "shared"),
        },
    )
    cleanup_intermediate_paths([resized_dir], keep=keep_intermediate)


if __name__ == "__main__":
    run()
