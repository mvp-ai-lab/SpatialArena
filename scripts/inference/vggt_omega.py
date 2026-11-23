#!/usr/bin/env python
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Run VGGT-Omega and export benchmark prediction artifacts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from adapter_utils import (
    canonical_image_root,
    mark_model_loaded,
    add_code_path,
    batched_image_paths,
    cleanup_intermediate_paths,
    colors_from_torch_images,
    export_camera_pose_priors,
    export_depth_predictions,
    export_depths_for_policy,
    export_standard_artifacts,
    find_image_paths,
    fixed_input_size,
    load_patch_grid_tensor,
    materialize_resized_images,
    patch_grid_compatible_resolution,
    prepare_model_input,
    preprocess_images,
    print_dry_run,
    register_preprocessor,
    restore_camera_export,
    select_interval,
    record_model_input,
    should_keep_intermediate,
    spatial_policy_metrics,
    unproject_depths_to_world,
    validate_shared_spatial_policy_depth_only,
    write_adapter_metrics,
)

METHOD = "VGGT_OMEGA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VGGT-Omega benchmark adapter.")
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=os.environ.get(
            "VGGT_OMEGA_CKPT_PATH",
            os.path.expanduser(
                "~/Projects/FoundationModels/pretrained/vggt_omega_1b_512.pt"),
        ),
    )
    parser.add_argument(
        "--text_ckpt_path",
        type=str,
        default=os.environ.get(
            "VGGT_OMEGA_TEXT_CKPT_PATH",
            os.path.expanduser(
                "~/Projects/FoundationModels/pretrained/vggt_omega_1b_256_text.pt"
            ),
        ),
    )
    parser.add_argument(
        "--vggt_omega_code_path",
        type=Path,
        default=Path(os.environ["VGGT_OMEGA_CODE_DIR"])
        if "VGGT_OMEGA_CODE_DIR" in os.environ else None,
    )
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--image_load_resolution", type=int, default=512)
    parser.add_argument(
        "--internal_resolution",
        type=int,
        default=0,
        help=
        "VGGT-Omega model resolution. Defaults to the nearest larger patch-compatible size.",
    )
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--preprocess_mode",
                        choices=("balanced", "max_size"),
                        default="balanced")
    parser.add_argument("--max_images_per_run", type=int, default=0)
    parser.add_argument("--max_points", type=int, default=1_000_000)
    parser.add_argument("--conf_thresh", type=float, default=0.0)
    parser.add_argument("--depth_conf_thresh", type=float, default=None)
    parser.add_argument("--enable_alignment", action="store_true")
    parser.add_argument("--export_depth", action="store_true")
    parser.add_argument("--export_colmap", action="store_true")
    parser.add_argument("--depth_only", action="store_true")
    parser.add_argument("--colmap_model_subdir", type=str, default="0")
    parser.add_argument("--keep_intermediate", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--spatial_policy",
        choices=("legacy", "legacy-pad", "shared-pad", "shared-rect-pad",
                 "shared-resize"),
        default="legacy",
        help=
        ("Temporary port of PR #21 (eval_refactor_hsy): 'shared-*' routes preprocessing through "
         "adapter_utils.plan_spatial_transform's vggt_omega policy (pad to a square canvas, "
         "rect-pad to the next patch boundary, or plain patch-16 resize) and restores predictions "
         "to source-image coordinates; 'legacy-pad' keeps the original preprocessing path but is "
         "reserved for matched-resolution camera/geometry export."),
    )
    return parser.parse_args()


def resolve_internal_resolution(args: argparse.Namespace) -> int:
    if args.internal_resolution > 0:
        return args.internal_resolution
    return patch_grid_compatible_resolution(args.resolution,
                                            patch_size=args.patch_size)


def preprocess_vggt_omega(
    image_paths: list[str],
    *,
    args: argparse.Namespace,
    device: str,
    load_and_preprocess_images=None,
    internal_resolution: "int | None" = None,
    batch_suffix: str = "",
):
    """VGGT-Omega's preprocess_<method>(), registered below for prepare_model_input("VGGT_OMEGA", ...).

    --spatial_policy shared-pad/shared-rect-pad/shared-resize: preprocess_images builds the
    padded/resized tensor directly from the true source images, restorable.
    --spatial_policy legacy (default) on a SLAM-protocol fixed-resolution input: only
    center-cropped to the 16px patch grid via load_patch_grid_tensor, never rescaled.
    --spatial_policy legacy otherwise: materialize_resized_images pre-pads to disk first
    (`batch_suffix` keeps each depth_only batch's temp dir distinct), then VGGT-Omega's own
    vendored load_and_preprocess_images further resizes those files.
    --spatial_policy legacy-pad (dense/camera export only, never depth_only -- enforced in
    run()): always pre-pads to disk too, then the same vendored loader resizes those files; only
    valid when --image_load_resolution == internal_resolution, so that second resize is a no-op.

    Returns ({"images": tensor, "resized_dir": Path or None}, transforms).
    """
    if args.spatial_policy not in ("legacy", "legacy-pad"):
        images, transforms = preprocess_images(
            image_paths,
            method="vggt_omega",
            size=args.image_load_resolution,
            mode=args.spatial_policy.removeprefix("shared-"),
        )
        return {"images": images.to(device), "resized_dir": None}, transforms

    fixed = args.spatial_policy == "legacy" and fixed_input_size(image_paths)
    if fixed:
        model_input_paths = image_paths
        resized_dir = None
    else:
        resized_paths = materialize_resized_images(
            image_paths,
            args.output_path / f"_resized_input_images{batch_suffix}",
            args.image_load_resolution,
        )
        model_input_paths = resized_paths
        resized_dir = Path(resized_paths[0]).parent

    if fixed:
        # Fixed-resolution inputs (full-slam): only center-cropped to the 16 px patch grid, never rescaled.
        images = load_patch_grid_tensor(model_input_paths,
                                        args.patch_size).to(device)
    else:
        images = load_and_preprocess_images(
            model_input_paths,
            mode=args.preprocess_mode,
            image_resolution=internal_resolution,
            patch_size=args.patch_size,
        ).to(device)
    return {"images": images, "resized_dir": resized_dir}, None


register_preprocessor("VGGT_OMEGA", preprocess_vggt_omega)


def run_vggt_omega_model(*, model, images, encoding_to_camera) -> tuple:
    import torch

    record_model_input(images.shape[-1], images.shape[-2])
    with torch.inference_mode():
        predictions = model(images)
    extrinsic, intrinsic = encoding_to_camera(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
    )
    extrinsic_np = extrinsic[0].detach().float().cpu().numpy()
    intrinsic_np = intrinsic[0].detach().float().cpu().numpy()
    depths = predictions["depth"][0].detach().float().cpu().numpy()
    depth_conf = predictions["depth_conf"][0].detach().float().cpu().numpy()
    points = unproject_depths_to_world(depths, extrinsic_np, intrinsic_np)
    colors = colors_from_torch_images(predictions["images"])
    return extrinsic_np, intrinsic_np, depths, depth_conf, points, colors


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

    add_code_path(args.vggt_omega_code_path, ("vggt_omega",))

    import torch
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError(
            "VGGT-Omega requires CUDA in the upstream implementation.")
    torch.cuda.reset_peak_memory_stats()

    checkpoint = args.text_ckpt_path if args.enable_alignment else args.ckpt_path
    ckpt_path = Path(checkpoint).expanduser()
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"VGGT-Omega checkpoint not found: {ckpt_path}. "
            "Set VGGT_OMEGA_CKPT_PATH/VGGT_OMEGA_TEXT_CKPT_PATH or run the download script."
        )
    model = VGGTOmega(enable_alignment=args.enable_alignment).to(device).eval()
    state = torch.load(str(ckpt_path), map_location="cpu")
    load_result = model.load_state_dict(state, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        print("VGGT-Omega checkpoint loaded with "
              f"{len(load_result.missing_keys)} missing keys and "
              f"{len(load_result.unexpected_keys)} unexpected keys.")

    started_at = mark_model_loaded()
    conf_thresh = args.depth_conf_thresh if args.depth_conf_thresh is not None else args.conf_thresh
    internal_resolution = resolve_internal_resolution(args)
    keep_intermediate = should_keep_intermediate(args)

    # Temporary port of PR #21 (eval_refactor_hsy, commit d88025f4).
    is_shared_spatial_policy = args.spatial_policy not in ("legacy",
                                                           "legacy-pad")
    validate_shared_spatial_policy_depth_only(
        is_shared=is_shared_spatial_policy, depth_only=args.depth_only)
    if args.spatial_policy == "legacy-pad" and args.depth_only:
        raise ValueError(
            "legacy-pad is for camera geometry export; use shared-pad for depth evaluation"
        )
    if args.spatial_policy == "legacy-pad" and args.image_load_resolution != internal_resolution:
        raise ValueError(
            "legacy-pad requires matching image_load_resolution and internal resolution"
        )

    if args.depth_only:
        if not args.export_depth:
            raise ValueError("--depth_only requires --export_depth")
        images_per_run = args.max_images_per_run if args.max_images_per_run > 0 else len(
            image_paths)
        images_per_run = int(
            os.environ.get("DEPTH_INFERENCE_BATCH_SIZE", images_per_run))
        if images_per_run < 1:
            raise ValueError(
                "DEPTH_INFERENCE_BATCH_SIZE/--max_images_per_run must be >= 1")
        peak_memory = 0.0
        pose_extrinsics: list[np.ndarray] = []
        pose_intrinsics: list[np.ndarray] = []
        pose_output_size: tuple[int, int] | None = None
        for batch_index, batch_original_paths in enumerate(
                batched_image_paths(image_paths, images_per_run)):
            model_input, transforms = prepare_model_input(
                "VGGT_OMEGA",
                batch_original_paths,
                args=args,
                device=device,
                load_and_preprocess_images=load_and_preprocess_images,
                internal_resolution=internal_resolution,
                batch_suffix=f"_batch_{batch_index:04d}",
            )
            resized_dir = model_input["resized_dir"]
            extrinsic, intrinsic, depths, depth_conf, _points, _colors = run_vggt_omega_model(
                model=model,
                images=model_input["images"],
                encoding_to_camera=encoding_to_camera)
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
            cleanup_intermediate_paths(
                [resized_dir], keep=keep_intermediate)  # None for fixed inputs
            peak_memory = max(peak_memory,
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
            checkpoint=str(ckpt_path),
            depth_only=True,
            metrics_extra={
                "Peak GPU Memory":
                    peak_memory,
                "enable_alignment":
                    args.enable_alignment,
                "preprocess_mode":
                    args.preprocess_mode,
                "image_load_resolution":
                    args.image_load_resolution,
                "internal_resolution":
                    internal_resolution,
                "patch_size":
                    args.patch_size,
                "max_images_per_run":
                    images_per_run,
                "resized_input":
                    True,
                **spatial_policy_metrics(args.spatial_policy,
                                         is_shared=is_shared_spatial_policy),
            },
        )
        return

    model_input, _transforms = prepare_model_input(
        "VGGT_OMEGA",
        image_paths,
        args=args,
        device=device,
        load_and_preprocess_images=load_and_preprocess_images,
        internal_resolution=internal_resolution,
    )
    resized_dir = model_input["resized_dir"]
    extrinsic_np, intrinsic_np, depths, depth_conf, points, colors = run_vggt_omega_model(
        model=model,
        images=model_input["images"],
        encoding_to_camera=encoding_to_camera)
    peak_memory = torch.cuda.max_memory_allocated() / (1024.0**2)

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
        checkpoint=str(ckpt_path),
        metrics_extra={
            "Peak GPU Memory":
                peak_memory,
            "enable_alignment":
                args.enable_alignment,
            "preprocess_mode":
                args.preprocess_mode,
            "image_load_resolution":
                args.image_load_resolution,
            "internal_resolution":
                internal_resolution,
            "patch_size":
                args.patch_size,
            "max_images_per_run":
                args.max_images_per_run,
            "resized_input":
                True,
            **spatial_policy_metrics(args.spatial_policy,
                                     is_shared=is_shared_spatial_policy),
        },
    )
    cleanup_intermediate_paths([resized_dir], keep=keep_intermediate)


if __name__ == "__main__":
    run()
