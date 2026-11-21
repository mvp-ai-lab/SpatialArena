#!/usr/bin/env python
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Run Pi3/Pi3X and export benchmark prediction artifacts."""

from __future__ import annotations

import argparse
import os
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from adapter_utils import (
    SpatialTransform,
    mark_model_loaded,
    add_code_path,
    batched_image_paths,
    canonical_image_root,
    c2w_to_w2c,
    cleanup_intermediate_paths,
    colors_from_torch_images,
    export_camera_pose_priors,
    export_spatial_depth_predictions,
    export_standard_artifacts,
    find_image_paths,
    fixed_input_size,
    fold_single_view_outputs,
    load_colmap_camera_priors,
    load_masked_depth_priors,
    load_optional_depth_priors,
    load_sparse_depth_priors,
    resolve_camera_conditioned_image_paths,
    normalize_depths,
    normalize_points,
    prepare_model_input,
    print_dry_run,
    register_preprocessor,
    record_model_input,
    restore_camera_export,
    select_interval,
    should_keep_intermediate,
    single_view_mode,
    write_adapter_metrics,
)


METHOD = "PI3"

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
    parser = argparse.ArgumentParser(description="Run Pi3/Pi3X benchmark adapter.")
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=os.environ.get("PI3_CKPT_PATH", "yyfz233/Pi3X"),
        help="Hugging Face model id or local Pi3/Pi3X checkpoint.",
    )
    parser.add_argument(
        "--pi3_code_path",
        type=Path,
        default=Path(os.environ["PI3_CODE_DIR"]) if "PI3_CODE_DIR" in os.environ else None,
    )
    parser.add_argument(
        "--model_variant",
        choices=("pi3", "pi3x"),
        default=os.environ.get("PI3_MODEL_VARIANT", "pi3x"),
    )
    parser.add_argument("--track", choices=TRACKS, default="rgb")
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--image_load_resolution", type=int, default=512)
    parser.add_argument("--pixel_limit", type=int, default=262144)
    parser.add_argument("--max_images_per_run", type=int, default=0)
    parser.add_argument("--max_points", type=int, default=1_000_000)
    parser.add_argument("--conf_thresh", type=float, default=0.0)
    parser.add_argument("--depth_conf_thresh", type=float, default=None)
    parser.add_argument("--export_depth", action="store_true")
    parser.add_argument("--export_colmap", action="store_true")
    parser.add_argument("--depth_only", action="store_true")
    parser.add_argument("--colmap_path", type=Path, default=None)
    parser.add_argument("--colmap_model_subdir", type=str, default="0")
    parser.add_argument("--keep_intermediate", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load_model(args: argparse.Namespace, device: str, use_multimodal: bool = False):
    import torch

    if args.model_variant == "pi3x":
        from pi3.models.pi3x import Pi3X  # pylint: disable=no-name-in-module

        ckpt = Path(args.ckpt_path).expanduser()
        if ckpt.is_file():
            model = Pi3X(use_multimodal=use_multimodal).eval()
            if ckpt.suffix == ".safetensors":
                from safetensors.torch import load_file

                state = load_file(str(ckpt))
            else:
                state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
            model.load_state_dict(state, strict=False)
        else:
            model = Pi3X.from_pretrained(args.ckpt_path).eval()
            if not use_multimodal and hasattr(model, "disable_multimodal"):
                model.disable_multimodal()
    else:
        from pi3.models.pi3 import Pi3  # pylint: disable=no-name-in-module

        ckpt = Path(args.ckpt_path).expanduser()
        if ckpt.is_file():
            model = Pi3().eval()
            if ckpt.suffix == ".safetensors":
                from safetensors.torch import load_file

                state = load_file(str(ckpt))
            else:
                state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
            model.load_state_dict(state, strict=False)
        else:
            model = Pi3.from_pretrained(args.ckpt_path).eval()

    return model.to(device)


def _zero_depth_like_image(image_path: str) -> np.ndarray:
    from PIL import Image

    with Image.open(image_path) as image:
        width, height = image.size
    return np.zeros((height, width), dtype=np.float32)


def load_multimodal_conditions(args: argparse.Namespace, image_paths: list[str]) -> dict[str, object] | None:
    if args.track == "rgb":
        return None
    if args.model_variant != "pi3x":
        raise ValueError("Pi3 multimodal tracks require --model_variant pi3x")

    conditions: dict[str, object] = {
        "poses": None,
        "depths": None,
        "intrinsics": None,
        "mask_add_depth": None,
    }
    if args.track not in TRACKS[1:]:
        raise ValueError(f"Unsupported benchmark track: {args.track}")
    if args.track in DEPTH_TRACKS:
        # Pi3's multimodal loader resizes every prior depth map to the
        # run-specific uniform image size. Keep this as a list because several
        # full-pose datasets (for example IMC) contain mixed-resolution images
        # and depth priors that cannot be stacked before resizing.
        # (nearest-neighbour: pi3.utils.basic.load_multimodal_data), so masked cells stay empty.
        optional_depths, has_depth, _missing = _load_track_depth_priors(args.data_path, image_paths, args.track)
        if not has_depth.any():
            raise FileNotFoundError(f"No matched depth priors found for {args.data_path}")
        conditions["depths"] = [
            depth if depth is not None else _zero_depth_like_image(image_path)
            for image_path, depth in zip(image_paths, optional_depths, strict=True)
        ]
        conditions["mask_add_depth"] = has_depth
    if args.track in CAMERA_TRACKS:
        poses, intrinsics = load_colmap_camera_priors(
            args.data_path,
            image_paths,
            colmap_path=args.colmap_path,
            colmap_model_subdir=args.colmap_model_subdir,
        )
        conditions["poses"] = poses
        conditions["intrinsics"] = intrinsics
    return conditions


def materialize_selected_original_images(image_paths: list[str], output_dir: Path) -> list[str]:
    """Expose selected original images in a flat directory for Pi3's multimodal loader."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.iterdir():
        if existing.is_file() or existing.is_symlink():
            existing.unlink()
    linked_paths: list[str] = []
    for index, image_path in enumerate(image_paths):
        source = Path(image_path).resolve()
        suffix = source.suffix if source.suffix else ".png"
        destination = output_dir / f"{index:06d}{suffix.lower()}"
        try:
            destination.symlink_to(source)
        except OSError:
            shutil.copy2(source, destination)
        linked_paths.append(str(destination))
    return linked_paths


def load_pi3_official_eval_rgb_tensor(
    image_paths: list[str],
    *,
    new_width: int,
    device: str,
    verbose: bool,
):
    """Load Pi3 RGB inputs with the official evaluation resize convention.

    The Pi3 evaluation branch uses ``load_img_size=512`` and the helper
    ``load_and_resize14``: resize the first image's width to ``new_width``,
    preserve aspect ratio for the shared batch size, then force both axes onto
    the model patch grid. This differs from Pi3's public example loader, which
    chooses dimensions from a pixel budget and can change the effective
    evaluation resolution.
    """

    import torch
    import torch.nn.functional as F
    import torchvision.transforms as tvf
    from PIL import Image

    sources = []
    for image_path in image_paths:
        try:
            sources.append(Image.open(image_path).convert("RGB"))
        except Exception as exc:
            print(f"Could not load image {image_path}: {exc}")
    if not sources:
        return torch.empty(0, device=device)

    first_image = sources[0]
    original_width, original_height = first_image.size
    target_width = int(new_width)
    target_height = max(14, round(original_height * (target_width / original_width) / 14) * 14)
    if verbose:
        print(f"Pi3 official-eval RGB resize: ({target_width}, {target_height}) before patch-grid interpolation")

    to_tensor = tvf.ToTensor()
    tensor_list = [
        to_tensor(image.resize((target_width, target_height), Image.Resampling.LANCZOS))
        for image in sources
    ]
    images = torch.stack(tensor_list, dim=0).to(device)
    patch_height, patch_width = images.shape[-2] // 14, images.shape[-1] // 14
    images = F.interpolate(
        images,
        (patch_height * 14, patch_width * 14),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return images


def preprocess_pi3(image_paths: list[str], *, args, device=None, target_wh: "tuple[int, int] | None" = None, **_unused):
    """Pi3's preprocess_<method>(), registered below for prepare_model_input("PI3_X", ...).

    Pi3's own official loaders (load_pi3_official_eval_rgb_tensor for the plain "rgb" track;
    the vendored multimodal pi3.utils.basic.load_multimodal_data for rgb_depth/rgb_cameras) are
    trusted directly (the same "trust the vendored resize" pattern as da3.py/iggt.py) -- both are
    pure stretches to one size shared by the whole batch, no crop, no pad margins -- so this only
    computes the matching SpatialTransform, not the tensor itself: each image's restoration
    transform needs only its own true (width, height) (read here, not reimplementing either
    resize's internal formula) and that one shared target size the caller already resolved from
    the resulting tensor's shape (`target_wh`). Returns (image_paths, transforms) -- image_paths
    unchanged, since the actual loader call happens at the caller, after this.
    """
    from PIL import Image

    transforms = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            source_wh = image.size
        transforms.append(SpatialTransform(source_wh=source_wh, resized_wh=target_wh))
    return image_paths, transforms


register_preprocessor("PI3_X", preprocess_pi3)


def run() -> None:
    args = parse_args()
    images_dir = canonical_image_root(args.data_path)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images directory not found: {images_dir}")
    image_paths = find_image_paths(images_dir)
    if args.track in CAMERA_TRACKS:
        if args.colmap_path is None:
            raise FileNotFoundError(f"{args.track} requires --colmap_path")
        image_paths = resolve_camera_conditioned_image_paths(
            args.data_path,
            image_paths,
            colmap_path=args.colmap_path,
        )
    image_paths = select_interval(image_paths, args.interval, args.max_images_per_run)
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")
    if args.dry_run:
        print_dry_run(args, METHOD, len(image_paths))
        return

    add_code_path(args.pi3_code_path, ("pi3",))

    import torch
    import torch.nn.functional as F
    from pi3.utils.basic import load_multimodal_data  # pylint: disable=no-name-in-module
    from pi3.utils.geometry import recover_intrinsic_from_rays_d  # pylint: disable=no-name-in-module

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    use_multimodal = args.track != "rgb"
    fixed = fixed_input_size(image_paths)
    if fixed:
        # Fixed-resolution inputs (full-slam): a pixel budget of exactly width x height makes the multimodal loader's
        # 14 px rounding keep the frames as they are (the RGB loader keeps them at new_width = their width).
        args.pixel_limit = fixed[0] * fixed[1]
    model = load_model(args, device, use_multimodal=use_multimodal)
    mark_model_loaded()

    single_view = single_view_mode()

    def infer_from_dir(input_dir: Path, condition_image_paths: list[str] | None = None):
        conditions = None
        if not use_multimodal:
            images = load_pi3_official_eval_rgb_tensor(
                condition_image_paths or image_paths,
                new_width=args.image_load_resolution,
                device=device,
                verbose=True,
            )
            model_kwargs = {}
        else:
            multimodal_conditions = load_multimodal_conditions(args, condition_image_paths or image_paths)
            images, conditions = load_multimodal_data(
                str(input_dir),
                multimodal_conditions,
                interval=1,
                PIXEL_LIMIT=args.pixel_limit,
                verbose=True,
                device=device,
            )
            model_kwargs = conditions
            if (
                args.track in DEPTH_TRACKS
                and multimodal_conditions is not None
                and multimodal_conditions.get("mask_add_depth") is not None
            ):
                depth_mask = torch.as_tensor(
                    multimodal_conditions["mask_add_depth"],
                    dtype=torch.bool,
                    device=device,
                )
                model_kwargs["mask_add_depth"] = depth_mask[None, : images.shape[1]]
        if images.numel() == 0:
            raise ValueError(f"Pi3 did not load any images from {input_dir}")
        record_model_input(images.shape[-1], images.shape[-2])
        _paths, transforms = prepare_model_input(
            "PI3_X",
            condition_image_paths or image_paths,
            args=args,
            device=device,
            target_wh=(images.shape[-1], images.shape[-2]),
        )

        dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        autocast_context = torch.amp.autocast("cuda", dtype=dtype) if device == "cuda" else nullcontext()
        with torch.no_grad():
            with autocast_context:
                if not use_multimodal and single_view:
                    # Independent single-view samples ([B, 1, 3, H, W]), folded back to the sequence layout.
                    result = fold_single_view_outputs(model(images[:, None]), images.shape[0])
                elif not use_multimodal:
                    result = model(images[None])
                else:
                    result = model(imgs=images, **model_kwargs)

        points = normalize_points(result["points"])
        local_points = result.get("local_points")
        if local_points is None:
            raise KeyError("Pi3 output missing local_points; cannot export depth")
        local_points_np = normalize_points(local_points)
        depths = normalize_depths(local_points_np[..., 2])
        # result["conf"] keeps the leading batch dim (unlike points/local_points, routed through
        # normalize_points above): normalize it the same way depths/points are, or the depth_only
        # export path's own normalize_depths(confidence) rejects the un-squeezed NxHxWx1 shape.
        confidence = normalize_depths(torch.sigmoid(result["conf"][..., 0])) if "conf" in result else None

        rays_d = F.normalize(result["local_points"], dim=-1)
        intrinsics = recover_intrinsic_from_rays_d(
            rays_d,
            force_center_principal_point=True,
        )[0].detach().float().cpu().numpy()
        extrinsics_w2c = c2w_to_w2c(result["camera_poses"][0])
        colors = colors_from_torch_images(images[0] if images.ndim == 5 else images)
        return depths, points, colors, confidence, extrinsics_w2c, intrinsics, transforms

    started_at = time.time()
    conf_thresh = args.depth_conf_thresh if args.depth_conf_thresh is not None else args.conf_thresh
    keep_intermediate = should_keep_intermediate(args)
    if args.depth_only:
        if not args.export_depth:
            raise ValueError("--depth_only requires --export_depth")
        images_per_run = args.max_images_per_run if args.max_images_per_run > 0 else len(image_paths)
        peak_memory = 0.0
        pose_image_paths: list[str] = []
        pose_extrinsics: list[np.ndarray] = []
        pose_intrinsics: list[np.ndarray] = []
        pose_output_size: tuple[int, int] | None = None
        batches = batched_image_paths(image_paths, images_per_run, independent_batch=not use_multimodal)
        for batch_index, batch_original_paths in enumerate(batches):
            if use_multimodal:
                input_paths = materialize_selected_original_images(
                    batch_original_paths,
                    args.output_path / f"_selected_input_images_batch_{batch_index:04d}",
                )
                batch_input_dir = Path(input_paths[0]).parent
            else:
                batch_input_dir = args.data_path / "images"
            depths, _points, _colors, confidence, extrinsics_w2c, intrinsics, transforms = infer_from_dir(
                batch_input_dir,
                batch_original_paths,
            )
            export_spatial_depth_predictions(args.output_path, batch_original_paths, depths, confidence, transforms)
            if args.export_colmap and extrinsics_w2c is not None and intrinsics is not None:
                pose_image_paths.extend(batch_original_paths)
                pose_extrinsics.append(np.asarray(extrinsics_w2c))
                restored_intrinsics, width, height = restore_camera_export(intrinsics, transforms)
                pose_intrinsics.append(restored_intrinsics)
                pose_output_size = (width, height)
            if use_multimodal:
                cleanup_intermediate_paths([batch_input_dir], keep=keep_intermediate)
            if torch.cuda.is_available():
                peak_memory = max(peak_memory, torch.cuda.max_memory_allocated() / (1024.0**2))
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
                "model_variant": args.model_variant,
                "track": args.track,
                "pixel_limit": args.pixel_limit,
                "max_images_per_run": images_per_run,
                "single_view": single_view,
                "image_load_resolution": args.image_load_resolution,
                "pi3_rgb_preprocess": "official_width_resize14",
                "resized_input": not use_multimodal,
            },
        )
        return

    if use_multimodal:
        input_paths = materialize_selected_original_images(
            image_paths,
            args.output_path / "_selected_input_images",
        )
        resized_dir = Path(input_paths[0]).parent
    else:
        resized_dir = args.data_path / "images"
    depths, points, colors, confidence, extrinsics_w2c, intrinsics, _transforms = infer_from_dir(
        resized_dir,
        image_paths,
    )

    peak_memory = (
        torch.cuda.max_memory_allocated() / (1024.0**2)
        if torch.cuda.is_available()
        else 0.0
    )
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
            "model_variant": args.model_variant,
            "track": args.track,
            "pixel_limit": args.pixel_limit,
            "max_images_per_run": args.max_images_per_run,
            "image_load_resolution": args.image_load_resolution,
            "pi3_rgb_preprocess": "official_width_resize14",
            "resized_input": not use_multimodal,
        },
    )
    if use_multimodal:
        cleanup_intermediate_paths([resized_dir], keep=keep_intermediate)


if __name__ == "__main__":
    run()
