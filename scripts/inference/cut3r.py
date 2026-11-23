#!/usr/bin/env python
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Run CUT3R and export benchmark prediction artifacts."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from adapter_utils import (
    canonical_image_root,
    mark_model_loaded,
    record_model_input,
    add_code_path,
    fixed_input_size,
    as_numpy,
    batched_image_paths,
    c2w_to_w2c,
    cleanup_intermediate_paths,
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
    validate_shared_spatial_policy_depth_only,
    write_adapter_metrics,
)

METHOD = "CUT3R"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CUT3R benchmark adapter.")
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=os.environ.get("CUT3R_CKPT_PATH", "src/cut3r_512_dpt_4_64.pth"),
    )
    parser.add_argument(
        "--cut3r_code_path",
        type=Path,
        default=Path(os.environ["CUT3R_CODE_DIR"])
        if "CUT3R_CODE_DIR" in os.environ else None,
    )
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--image_load_resolution", type=int, default=512)
    parser.add_argument("--max_images_per_run", type=int, default=0)
    parser.add_argument("--max_points", type=int, default=1_000_000)
    parser.add_argument("--conf_thresh", type=float, default=0.0)
    parser.add_argument("--depth_conf_thresh", type=float, default=None)
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
        ("'legacy' (default): unchanged -- CUT3R's own demo.prepare_input/dust3r.utils.image."
         "load_images resize-then-center-crop-to-16px-grid, which discards real border pixels "
         "(up to 15px per side) and additionally forces a lossy 4:3 crop whenever the resized "
         "image happens to be square. 'shared': bypasses that loader entirely -- a single direct "
         "pad-to-square resize straight to --resolution via "
         "adapter_utils.plan_spatial_transform(method='vggt_omega') (CUT3R's own patch size is "
         "also 16, matching VGGT-Omega's), with CUT3R's view dicts (img/ray_map/true_shape/idx/"
         "instance) built directly from that padded tensor instead of re-loading from disk, so no "
         "image content is ever discarded. Depth exported at the true original per-image "
         "resolution via restore_prediction/export_spatial_depth_predictions."),
    )
    return parser.parse_args()


def stack_list(values: list) -> np.ndarray:
    arrays = [as_numpy(value) for value in values]
    return np.concatenate(arrays, axis=0)


def intrinsics_from_cam_dict(cam_dict: dict) -> np.ndarray:
    focal = np.asarray(cam_dict["focal"], dtype=np.float32).reshape(-1)
    pp = np.asarray(cam_dict["pp"], dtype=np.float32).reshape(-1, 2)
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None],
                           focal.shape[0],
                           axis=0)
    intrinsics[:, 0, 0] = focal
    intrinsics[:, 1, 1] = focal
    intrinsics[:, 0, 2] = pp[:, 0]
    intrinsics[:, 1, 2] = pp[:, 1]
    return intrinsics


def patch_rope_negative_positions() -> None:
    """Handle CUT3R pose tokens when the CUDA RoPE extension is unavailable."""
    import torch
    import torch.nn.functional as F
    from models import pos_embed

    rope_cls = getattr(pos_embed, "RoPE2D", None)
    if rope_cls is None or getattr(rope_cls, "_ffb_negative_position_patch",
                                   False):
        return
    if not hasattr(rope_cls, "get_cos_sin") or not hasattr(
            rope_cls, "rotate_half"):
        return

    def apply_rope1d(self, tokens, pos1d, cos, sin):
        assert pos1d.ndim == 2
        valid = pos1d >= 0
        safe_pos = pos1d.clamp_min(0)
        cos_pos = F.embedding(safe_pos, cos)[:, None, :, :]
        sin_pos = F.embedding(safe_pos, sin)[:, None, :, :]
        rotated = (tokens * cos_pos) + (self.rotate_half(tokens) * sin_pos)
        if bool(valid.all()):
            return rotated
        return torch.where(valid[:, None, :, None], rotated, tokens)

    def forward(self, tokens, positions):
        assert tokens.size(
            3) % 2 == 0, "number of dimensions should be a multiple of two"
        assert positions.ndim == 3 and positions.shape[-1] == 2
        dim = tokens.size(3) // 2
        max_position = int(positions.clamp_min(0).max()) + 1
        cos, sin = self.get_cos_sin(dim, max_position, tokens.device,
                                    tokens.dtype)
        y, x = tokens.chunk(2, dim=-1)
        y = self.apply_rope1d(y, positions[:, :, 0], cos, sin)
        x = self.apply_rope1d(x, positions[:, :, 1], cos, sin)
        return torch.cat((y, x), dim=-1)

    rope_cls.apply_rope1d = apply_rope1d
    rope_cls.forward = forward
    rope_cls._ffb_negative_position_patch = True


def build_shared_views(image_paths: list[str], args: argparse.Namespace,
                       device: str):
    """Build CUT3R's own view-dict format (img/ray_map/true_shape/idx/instance, matching
    demo.prepare_input's output exactly) directly from a padded tensor, bypassing
    demo.prepare_input/dust3r.utils.image.load_images entirely -- that loader's own
    resize-then-center-crop-to-16px-grid (and its lossy square-image special case) is never
    invoked, so no source pixel is ever discarded."""
    import torch

    # exif_transpose=True to match dust3r.utils.image.load_images's own convention (the loader
    # this bypasses still applies it).
    images, transforms = preprocess_images(image_paths,
                                           method="vggt_omega",
                                           size=args.resolution,
                                           mode="pad",
                                           exif_transpose=True)
    # dust3r's ImgNorm is ToTensor (-> [0,1]) + Normalize(0.5, 0.5) (-> [-1,1]); preprocess_images
    # already returns [0,1], so only the normalization step remains.
    images = (images * 2.0 - 1.0).to(device)
    views = []
    for index in range(images.shape[0]):
        frame = images[index:index + 1]
        height, width = frame.shape[-2:]
        views.append({
            "img":
                frame,
            "ray_map":
                torch.full((1, 6, height, width), float("nan"), device=device),
            "true_shape":
                torch.from_numpy(np.int32([[height, width]])),
            "idx":
                index,
            "instance":
                str(index),
            # Matching demo.py's prepare_input image-only branch exactly: a fixed identity
            "camera_pose":
                torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0),
            "img_mask":
                torch.tensor(True, device=device).unsqueeze(0),
            "ray_mask":
                torch.tensor(False, device=device).unsqueeze(0),
            "update":
                torch.tensor(True, device=device).unsqueeze(0),
            "reset":
                torch.tensor(False, device=device).unsqueeze(0),
        })
    return views, transforms


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

    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    add_code_path(args.cut3r_code_path, ("src", "dust3r"))
    if args.cut3r_code_path is not None:
        src_path = args.cut3r_code_path.expanduser().resolve() / "src"
        if src_path.is_dir() and str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

    import torch
    from add_ckpt_path import add_path_to_dust3r
    from demo import prepare_input, prepare_output
    from src.dust3r.inference import inference
    from src.dust3r.model import ARCroco3DStereo

    patch_rope_negative_positions()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    checkpoint = args.ckpt_path
    if args.cut3r_code_path is not None:
        candidate = args.cut3r_code_path / args.ckpt_path
        if not Path(
                args.ckpt_path).expanduser().is_file() and candidate.is_file():
            checkpoint = str(candidate)
    add_path_to_dust3r(checkpoint)

    original_torch_load = torch.load

    def trusted_torch_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = trusted_torch_load
    try:
        model = ARCroco3DStereo.from_pretrained(checkpoint).to(device).eval()
    finally:
        torch.load = original_torch_load
    started_at = mark_model_loaded()

    def infer_views(views, raw_dir: Path, num_frames: int):
        record_model_input(views[0]["img"].shape[-1], views[0]["img"].shape[-2])
        outputs, _state_args = inference(views, model, device)
        points_list, colors_list, conf_list, cam_dict = prepare_output(
            outputs,
            str(raw_dir),
            revisit=1,
            use_pose=True,
        )
        points = stack_list(points_list)
        colors = stack_list(colors_list)
        confidence = stack_list(conf_list)
        depths = []
        for frame_id in range(num_frames):
            depths.append(np.load(raw_dir / "depth" / f"{frame_id:06d}.npy"))
        depths_np = np.stack(depths, axis=0)
        return points, colors, confidence, depths_np, cam_dict

    def preprocess_cut3r(image_paths: list[str],
                         *,
                         args,
                         device,
                         batch_suffix: str = ""):
        """CUT3R's preprocess_<method>(), registered just below for prepare_model_input("CUT3R",
        ...). A closure (not module-level) because CUT3R's own demo.prepare_input is only
        importable after add_ckpt_path/sys.path setup above, inside run().

        Fixed-resolution frames (full-slam): go to CUT3R's own loader as they are -- it
        center-crops them to its 16px grid (504x378 -> 496x368); padding them onto a square
        first would make its loader crop a 4:3 window of 496x372, off the grid.
        --spatial_policy shared (non-fixed input): build_shared_views bypasses CUT3R's own
        loader entirely, resizing+padding straight from these original paths.
        --spatial_policy legacy otherwise: materialize_resized_images pre-pads to disk first
        (`batch_suffix` keeps each depth_only batch's temp dir distinct), then CUT3R's own
        demo.prepare_input/dust3r.utils.image.load_images (lossy center-crop) reads those files.

        Returns ({"views": ..., "resized_dir": Path or None}, transforms).
        """
        fixed = fixed_input_size(image_paths)
        if fixed or args.spatial_policy == "shared":
            model_input_paths = image_paths
            resized_dir = None
        else:
            resized = materialize_resized_images(
                image_paths,
                args.output_path / f"_resized_input_images{batch_suffix}",
                args.image_load_resolution,
            )
            model_input_paths = resized
            resized_dir = Path(resized[0]).parent

        if args.spatial_policy == "shared" and not fixed:
            views, transforms = build_shared_views(model_input_paths, args,
                                                   device)
        else:
            views = prepare_input(
                img_paths=model_input_paths,
                img_mask=[True] * len(model_input_paths),
                size=args.resolution,
                revisit=1,
                update=True,
            )
            transforms = None
        return {"views": views, "resized_dir": resized_dir}, transforms

    register_preprocessor("CUT3R", preprocess_cut3r)

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
                "CUT3R",
                batch_original_paths,
                args=args,
                device=device,
                batch_suffix=f"_batch_{batch_index:04d}")
            resized_dir = model_input["resized_dir"]
            raw_dir = args.output_path / f"_cut3r_raw_batch_{batch_index:04d}"
            _points, _colors, confidence, depths_np, cam_dict = infer_views(
                model_input["views"], raw_dir, len(batch_original_paths))
            export_depths_for_policy(
                args.output_path,
                batch_original_paths,
                depths_np,
                confidence,
                transforms,
                legacy_export=lambda: export_depth_predictions(
                    args.output_path, batch_original_paths, depths_np,
                    confidence),
            )
            if args.export_colmap:
                batch_c2w = np.repeat(np.eye(4, dtype=np.float32)[None],
                                      len(batch_original_paths),
                                      axis=0)
                batch_c2w[:, :3, :3] = np.asarray(cam_dict["R"],
                                                  dtype=np.float32)
                batch_c2w[:, :3, 3] = np.asarray(cam_dict["t"],
                                                 dtype=np.float32)
                pose_extrinsics.append(c2w_to_w2c(batch_c2w))
                batch_intrinsics = intrinsics_from_cam_dict(cam_dict)
                if transforms is not None:
                    restored_intrinsic, width, height = restore_camera_export(
                        batch_intrinsics, transforms)
                    pose_intrinsics.append(restored_intrinsic)
                    pose_output_size = (width, height)
                else:
                    pose_intrinsics.append(batch_intrinsics)
                    if pose_output_size is None:
                        depth_array = np.asarray(depths_np)
                        pose_output_size = (depth_array.shape[2],
                                            depth_array.shape[1])
            cleanup_intermediate_paths([resized_dir, raw_dir],
                                       keep=keep_intermediate)
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
            checkpoint=checkpoint,
            depth_only=True,
            metrics_extra={
                "Peak GPU Memory":
                    peak_memory,
                "resolution":
                    args.resolution,
                "image_load_resolution":
                    args.image_load_resolution,
                "max_images_per_run":
                    images_per_run,
                "resized_input":
                    True,
                **spatial_policy_metrics(args.spatial_policy,
                                         is_shared=args.spatial_policy == "shared"),
            },
        )
        return

    model_input, _transforms = prepare_model_input("CUT3R",
                                                   image_paths,
                                                   args=args,
                                                   device=device)
    resized_dir = model_input["resized_dir"]
    raw_dir = args.output_path / "_cut3r_raw"
    points, colors, confidence, depths_np, cam_dict = infer_views(
        model_input["views"], raw_dir, len(image_paths))

    c2w = np.repeat(np.eye(4, dtype=np.float32)[None], len(image_paths), axis=0)
    c2w[:, :3, :3] = np.asarray(cam_dict["R"], dtype=np.float32)
    c2w[:, :3, 3] = np.asarray(cam_dict["t"], dtype=np.float32)
    extrinsics_w2c = c2w_to_w2c(c2w)
    intrinsics = intrinsics_from_cam_dict(cam_dict)
    peak_memory = (torch.cuda.max_memory_allocated() /
                   (1024.0**2) if torch.cuda.is_available() else 0.0)

    export_standard_artifacts(
        method=METHOD,
        output_path=args.output_path,
        images_dir=images_dir,
        image_paths=image_paths,
        depths=depths_np,
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
        checkpoint=checkpoint,
        metrics_extra={
            "Peak GPU Memory":
                peak_memory,
            "resolution":
                args.resolution,
            "image_load_resolution":
                args.image_load_resolution,
            "max_images_per_run":
                args.max_images_per_run,
            "resized_input":
                True,
            **spatial_policy_metrics(args.spatial_policy,
                                     is_shared=args.spatial_policy == "shared"),
        },
    )
    cleanup_intermediate_paths([resized_dir, raw_dir], keep=keep_intermediate)


if __name__ == "__main__":
    run()
