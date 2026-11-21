# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Shared adapter for MapAnything-style external dense reconstruction wrappers."""

from __future__ import annotations

import argparse
import errno
import importlib
import json
import getpass
import os
import shutil
import sys
import tempfile
import time
from urllib.parse import urlparse
from pathlib import Path
from typing import Any

import numpy as np

from adapter_utils import (
    canonical_image_root,
    mark_model_loaded,
    add_code_path,
    as_numpy,
    c2w_to_w2c,
    export_camera_pose_priors,
    export_spatial_depth_predictions,
    export_standard_artifacts,
    find_image_paths,
    batched_image_paths,
    normalize_confidence,
    normalize_depths,
    normalize_points,
    prepare_model_input,
    preprocess_images,
    print_dry_run,
    register_preprocessor,
    quaternion_xyzw_to_matrix,
    restore_camera_export,
    select_interval,
)

# Wrappers that predict from image pairs; a single image (single-view mode, MonoDepth) is run as the pair
# (image, image) -- DUSt3R's own monocular mode -- and only the first view's prediction is kept.
PAIRWISE_WRAPPERS = ("DUSt3RBAWrapper", "MASt3RSGAWrapper", "Pow3RBAWrapper")


# Shared method checkouts (the engine's METHOD_CODE_ROOT); the engine also passes the *_CODE_PATH below.
# see spatialarena/paths.py
SPATIALARENA_ROOT = os.path.expanduser(os.environ.get("SPATIALARENA_ROOT", "~/spatialarena"))
METHOD_CODE_ROOT = Path(
    os.environ.get("METHOD_CODE_ROOT", f"{SPATIALARENA_ROOT}/benchmarks/_model_codes")
)


def add_optional_path(path: Path | None, package_names: tuple[str, ...]) -> None:
    if path is not None:
        add_code_path(path, package_names)


def add_nested_method_paths(path: Path | None) -> None:
    """Expose common nested external-method submodules to Python imports."""

    if path is None:
        return
    root = Path(path).expanduser().resolve()
    for nested in (
        root / "dust3r",
        root / "dust3r" / "croco",
        root / "croco",
    ):
        if nested.is_dir():
            add_code_path(nested, ())


def import_symbol(module_name: str, symbol_name: str) -> Any:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - preserve useful setup context
        raise RuntimeError(
            f"Could not import {module_name!r}. Install/clone the method dependencies "
            f"and pass the appropriate *_CODE_PATH to the benchmark runner."
        ) from exc
    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise RuntimeError(f"{module_name!r} does not expose {symbol_name!r}") from exc


def describe_structure(value: Any, *, depth: int = 0) -> str:
    """Return a compact nested type summary for adapter diagnostics."""

    if depth >= 3:
        return type(value).__name__
    if isinstance(value, dict):
        keys = list(value.keys())
        preview = ", ".join(str(key) for key in keys[:8])
        if len(keys) > 8:
            preview += ", ..."
        return f"dict(keys=[{preview}])"
    if isinstance(value, (list, tuple)):
        inner = describe_structure(value[0], depth=depth + 1) if value else "empty"
        return f"{type(value).__name__}(len={len(value)}, first={inner})"
    if hasattr(value, "shape"):
        return f"{type(value).__name__}(shape={getattr(value, 'shape', None)})"
    return type(value).__name__


# The scene of the last DUSt3R-family global alignment (patch_dust3r_global_aligner): its per-view confidence,
# which MapAnything's BA wrappers drop from their outputs, is attached back by attach_aligned_confidence().
_ALIGNED_SCENE: dict[str, Any] = {}


def capture_aligned_scene(wrapper_module: str) -> None:
    """Keep the scene returned by a DUSt3R-family wrapper's alignment step (MASt3R: ``sparse_global_alignment``;
    DUSt3R / Pow3R: ``global_aligner``, patched by patch_dust3r_global_aligner)."""
    try:
        module = importlib.import_module(wrapper_module)
    except Exception:  # pylint: disable=broad-except
        return
    original = getattr(module, "sparse_global_alignment", None)
    if original is None or getattr(original, "_feedforward_capture", False):
        return

    def capturing_alignment(*args: Any, **kwargs: Any) -> Any:
        scene = original(*args, **kwargs)
        _ALIGNED_SCENE["scene"] = scene
        return scene

    capturing_alignment._feedforward_capture = True  # type: ignore[attr-defined]
    module.sparse_global_alignment = capturing_alignment


def attach_aligned_confidence(outputs: list[dict[str, Any]]) -> None:
    """Give each view's output the aligned scene's confidence as ``conf`` (the model's own, ``exp`` scale).

    DUSt3R / Pow3R (PointCloudOptimizer) hold it in ``scene.im_conf``; MASt3R's sparse scene returns it from
    ``get_dense_pts3d()``. Outputs that already have ``conf``, or a scene without confidence, are left alone.
    """
    scene = _ALIGNED_SCENE.pop("scene", None)
    if scene is None or not outputs or all("conf" in output for output in outputs):
        return
    if hasattr(scene, "im_conf"):
        confs = list(scene.im_conf)
    elif hasattr(scene, "get_dense_pts3d"):
        confs = list(scene.get_dense_pts3d()[2])
    else:
        return
    if len(confs) != len(outputs):
        return
    for output, conf in zip(outputs, confs):
        depth = output.get("depth_along_ray")
        if depth is None:
            continue
        height, width = as_numpy(depth).squeeze().shape[:2]
        conf = as_numpy(conf).astype(np.float32).reshape(-1)
        if conf.size == height * width:
            output["conf"] = conf.reshape(height, width)


def patch_dust3r_global_aligner(wrapper_module: str) -> None:
    """Make MapAnything DUSt3R-family wrappers accept list-form inference output.

    Some checked-out DUSt3R/Pow3R revisions return a list of per-batch dictionaries
    when input pairs contain mixed image shapes. MapAnything's wrappers pass that
    value directly to the module-level ``global_aligner`` imported from
    ``dust3r.cloud_opt``, while the aligner expects a dictionary with
    ``view1/view2/pred1/pred2`` keys. Normalizing at this adapter boundary keeps the
    benchmark self-contained and avoids patching external repositories.
    """

    if not (
        wrapper_module.endswith(".dust3r")
        or wrapper_module.endswith(".pow3r")
        or wrapper_module.endswith(".mast3r")
    ):
        return
    try:
        module = importlib.import_module(wrapper_module)
        device_utils = importlib.import_module("dust3r.utils.device")
    except Exception:
        return
    original_global_aligner = getattr(module, "global_aligner", None)
    collate_with_cat = getattr(device_utils, "collate_with_cat", None)
    if original_global_aligner is None or collate_with_cat is None:
        return
    if getattr(original_global_aligner, "_feedforward_list_patch", False):
        return

    def has_dust3r_keys(value: Any) -> bool:
        return isinstance(value, dict) and {"view1", "view2", "pred1", "pred2"}.issubset(value)

    def tuple_to_dust3r_dict(value: Any) -> dict[str, Any] | None:
        if isinstance(value, (list, tuple)) and len(value) >= 4 and all(isinstance(item, dict) for item in value[:4]):
            return {
                "view1": value[0],
                "view2": value[1],
                "pred1": value[2],
                "pred2": value[3],
            }
        return None

    def normalize_dust3r_output(dust3r_output: Any) -> Any:
        if has_dust3r_keys(dust3r_output):
            return dust3r_output
        if isinstance(dust3r_output, (list, tuple)) and dust3r_output:
            if len(dust3r_output) == 1:
                return normalize_dust3r_output(dust3r_output[0])
            first = dust3r_output[0]
            if all(has_dust3r_keys(item) for item in dust3r_output):
                dust3r_output = collate_with_cat(list(dust3r_output), lists=True)
            else:
                tuple_dict = tuple_to_dust3r_dict(dust3r_output)
                if tuple_dict is not None:
                    return tuple_dict
                if all(tuple_to_dust3r_dict(item) is not None for item in dust3r_output):
                    collated_tuple = collate_with_cat(list(dust3r_output), lists=True)
                    tuple_dict = tuple_to_dust3r_dict(collated_tuple)
                    if tuple_dict is not None:
                        return tuple_dict
            print(
                "Unrecognized DUSt3R-family output structure: "
                f"{describe_structure(dust3r_output)}",
                file=sys.stderr,
                flush=True,
            )
        return dust3r_output

    def patched_global_aligner(dust3r_output: Any, *args: Any, **kwargs: Any) -> Any:
        normalized_output = normalize_dust3r_output(dust3r_output)
        try:
            scene = original_global_aligner(normalized_output, *args, **kwargs)
            _ALIGNED_SCENE["scene"] = scene
            return scene
        except TypeError:
            print(
                "DUSt3R-family global_aligner failed with structures: "
                f"original={describe_structure(dust3r_output)}, "
                f"normalized={describe_structure(normalized_output)}",
                file=sys.stderr,
                flush=True,
            )
            raise

    patched_global_aligner._feedforward_list_patch = True  # type: ignore[attr-defined]
    module.global_aligner = patched_global_aligner


def candidate_checkpoint_files(root: Path) -> list[Path]:
    suffixes = (".pth", ".pt", ".bin", ".safetensors")
    return [
        path
        for suffix in suffixes
        for path in sorted(root.rglob(f"*{suffix}"))
        if "retrieval" not in path.name.lower() and "codebook" not in path.name.lower()
    ]


def resolve_checkpoint_arg(checkpoint: str) -> str:
    """Resolve local files, local dirs, HF repo ids, and direct checkpoint URLs."""

    if not checkpoint:
        return checkpoint
    parsed = urlparse(checkpoint)
    if parsed.scheme in {"http", "https"}:
        import torch

        cache_root = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
        target_dir = cache_root / "hub" / "checkpoints"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / Path(parsed.path).name
        if not target_path.is_file():
            torch.hub.download_url_to_file(checkpoint, str(target_path), progress=True)
        return str(target_path)

    path = Path(checkpoint).expanduser()
    if path.is_file():
        return str(path)
    if path.is_dir():
        candidates = candidate_checkpoint_files(path)
        if candidates:
            return str(candidates[0])
        raise FileNotFoundError(f"No checkpoint file found under {path}")

    if "/" in checkpoint:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                f"Checkpoint {checkpoint!r} looks like an HF repo id, but huggingface_hub is unavailable."
            ) from exc
        snapshot_dir = Path(snapshot_download(repo_id=checkpoint, repo_type="model"))
        candidates = candidate_checkpoint_files(snapshot_dir)
        if candidates:
            return str(candidates[0])
        raise FileNotFoundError(f"No checkpoint file found in HF snapshot for {checkpoint}")

    return checkpoint


def parse_common_args(method_name: str, ckpt_env: str, code_env: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Run {method_name} and export benchmark artifacts.")
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--ckpt_path", default=os.environ.get(ckpt_env, ""))
    parser.add_argument("--method_code_path", type=Path,
                        default=Path(os.environ[code_env]) if code_env in os.environ else None)
    parser.add_argument(
        "--mapanything_code_path",
        type=Path,
        default=Path(os.environ["MAPANYTHING_CODE_PATH"])
        if "MAPANYTHING_CODE_PATH" in os.environ
        else METHOD_CODE_ROOT / "map-anything",
    )
    parser.add_argument("--track", choices=("rgb", "rgb_depth", "rgb_cameras"), default="rgb")
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--image_load_resolution", type=int, default=512)
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


def preprocess_dust3r_family(
    image_paths: list[str],
    *,
    args: argparse.Namespace,
    device=None,
    norm_type: str = "dust3r",
    **_unused,
) -> tuple[list[dict[str, Any]], list]:
    """DUST3R/MAST3R/POW3R's shared preprocess_<method>() (one implementation, registered below
    for all three method names -- their wrappers never differ in how they read an image, only in
    the global-alignment math applied afterward). Builds MapAnything-family view dicts
    (img/data_norm_type/true_shape/label/instance/idx/camera_intrinsics/camera_pose/depthmap,
    matching mapanything.utils.image.preprocess_inputs's own output exactly -- confirmed by
    reading it directly) from a padded tensor instead, bypassing that function's
    crop_resize_if_necessary entirely.

    preprocess_inputs computes ONE target size from the whole batch's *average* aspect ratio
    (fixed_mapping: the closest of a few preset buckets, used for DUSt3R/MASt3R; square: a fixed
    patch-aligned square, used for Pow3R), then crops every image to it -- real, batch-dependent
    content loss (the same DUST3R-family crop CUT3R's own load_images had, just computed over a
    batch average instead of per image). Every DUSt3R/MASt3R/Pow3R-family wrapper here only ever
    gets a plain {"img": ...} view (no depth/camera conditioning is ever threaded through this
    shared wrapper), so all three degrade to the same clean per-image pad case this session's
    other adapters already use -- the original resize_mode distinction (which of those two target
    sizes preprocess_inputs would have picked) no longer applies, since this bypasses that choice
    entirely in favor of one pad policy for all three.
    """
    import torch

    size = args.image_load_resolution if args.image_load_resolution % 16 == 0 else 512
    if size not in {512}:
        size = 512
    # exif_transpose=True to match mapanything.utils.image.preprocess_inputs's own convention
    # (the function this bypasses still applies it).
    images, transforms = preprocess_images(
        image_paths, method="vggt_omega", size=size, mode="pad", exif_transpose=True
    )
    # The "dust3r" UniCeption normalization (used by every wrapper this shared code supports) is
    # ToTensor (-> [0,1]) + Normalize(0.5, 0.5); preprocess_images already returns [0,1], so only
    # the normalization step remains.
    images = images * 2.0 - 1.0
    views: list[dict[str, Any]] = []
    for index, image_path in enumerate(image_paths):
        frame = images[index : index + 1]
        height, width = frame.shape[-2:]
        views.append(
            {
                "img": frame,
                "data_norm_type": [norm_type],
                "true_shape": np.int32([[height, width]]),
                "label": [Path(args.data_path).name],
                "instance": [Path(image_path).name],
                "idx": index,
                "camera_intrinsics": default_intrinsics(width, height)[None],
                "camera_pose": torch.eye(4, dtype=torch.float32)[None],
                "depthmap": torch.zeros((1, height, width, 1), dtype=torch.float32),
            }
        )
    return views, transforms


register_preprocessor("DUST3R", preprocess_dust3r_family)
register_preprocessor("MAST3R", preprocess_dust3r_family)
register_preprocessor("POW3R", preprocess_dust3r_family)


def default_intrinsics(width: int, height: int):
    import torch

    focal = float(max(width, height))
    intrinsics = torch.eye(3, dtype=torch.float32)
    intrinsics[0, 0] = focal
    intrinsics[1, 1] = focal
    intrinsics[0, 2] = (width - 1) / 2.0
    intrinsics[1, 2] = (height - 1) / 2.0
    return intrinsics


def move_views_to_device(views: list[dict[str, Any]], device: str) -> list[dict[str, Any]]:
    ignore = {"label", "instance", "data_norm_type", "idx", "true_shape"}
    for view in views:
        for key, value in list(view.items()):
            if key in ignore:
                continue
            if hasattr(value, "to"):
                view[key] = value.to(device)
    return views


def outputs_to_standard(
    outputs: list[dict[str, Any]], views: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from mapanything.utils.image import rgb

    points = normalize_points([as_numpy(output["pts3d"])[0] for output in outputs])
    if "pts3d_cam" in outputs[0]:
        depths = normalize_depths([as_numpy(output["pts3d_cam"])[0, ..., 2] for output in outputs])
    else:
        depths = normalize_depths([as_numpy(output["depth_along_ray"])[0, ..., 0] for output in outputs])
    confidence_maps = []
    for index, output in enumerate(outputs):
        conf = as_numpy(output.get("conf", np.ones_like(depths[index]))).squeeze()
        if conf.shape == depths[index].shape:
            confidence_maps.append(conf)
        elif conf.ndim == 1 and conf.shape[0] == depths.shape[1]:
            confidence_maps.append(np.broadcast_to(conf[:, None], depths[index].shape))
        elif conf.ndim == 1 and conf.shape[0] == depths.shape[2]:
            confidence_maps.append(np.broadcast_to(conf[None, :], depths[index].shape))
        else:
            confidence_maps.append(np.ones_like(depths[index], dtype=np.float32))
    # No confidence head at all: None (point sampling treats it as all-ones, as before), so no all-ones map is
    # exported as if the model had predicted it.
    has_confidence = any("conf" in output for output in outputs)
    confidence = normalize_confidence(confidence_maps, depths.shape) if has_confidence else None
    colors = np.stack(
        [
            (rgb(view["img"][0].detach().cpu(), view["data_norm_type"][0]).clip(0.0, 1.0) * 255).astype(np.uint8)
            for view in views
        ],
        axis=0,
    )
    c2w = quats_trans_to_c2w(
        [as_numpy(output["cam_quats"])[0] for output in outputs],
        [as_numpy(output["cam_trans"])[0] for output in outputs],
    )
    extrinsics_w2c = c2w_to_w2c(c2w)
    intrinsics = infer_intrinsics(outputs, depths.shape[1], depths.shape[2])
    return depths, points, colors, confidence, extrinsics_w2c, intrinsics


def quats_trans_to_c2w(quats: list[np.ndarray], trans: list[np.ndarray]) -> np.ndarray:
    quat_array = np.asarray(quats, dtype=np.float32)
    trans_array = np.asarray(trans, dtype=np.float32)
    c2w = np.repeat(np.eye(4, dtype=np.float32)[None], quat_array.shape[0], axis=0)
    c2w[:, :3, :3] = quaternion_xyzw_to_matrix(quat_array)
    c2w[:, :3, 3] = trans_array
    return c2w


def infer_intrinsics(outputs: list[dict[str, Any]], height: int, width: int) -> np.ndarray:
    intrinsics: list[np.ndarray] = []
    for output in outputs:
        ray_dirs = output.get("ray_directions")
        if ray_dirs is None:
            intrinsics.append(as_numpy(default_intrinsics(width, height)))
            continue
        rays = as_numpy(ray_dirs)[0]
        center = rays[height // 2, width // 2]
        fx = abs(center[2]) / max(abs(rays[height // 2, min(width - 1, width // 2 + 1), 0] - center[0]), 1e-6)
        fy = abs(center[2]) / max(abs(rays[min(height - 1, height // 2 + 1), width // 2, 1] - center[1]), 1e-6)
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = (width - 1) / 2.0
        K[1, 2] = (height - 1) / 2.0
        intrinsics.append(K)
    return np.stack(intrinsics, axis=0)


def run_wrapper_method(
    *,
    method: str,
    wrapper_module: str,
    wrapper_class: str,
    ckpt_env: str,
    code_env: str,
    code_package_names: tuple[str, ...],
    wrapper_kwargs: dict[str, Any] | None = None,
) -> None:
    args = parse_common_args(method, ckpt_env, code_env)
    images_dir = canonical_image_root(args.data_path)
    image_paths = select_interval(find_image_paths(images_dir), args.interval, args.max_images_per_run)
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")
    if args.dry_run:
        print_dry_run(args, method, len(image_paths))
        return
    if not args.ckpt_path:
        raise ValueError(f"{method} requires --ckpt_path or {ckpt_env}.")
    checkpoint_path = resolve_checkpoint_arg(args.ckpt_path)

    add_optional_path(args.mapanything_code_path, ("mapanything",))
    add_optional_path(args.method_code_path, code_package_names)
    add_nested_method_paths(args.method_code_path)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    started_at = time.time()
    Wrapper = import_symbol(wrapper_module, wrapper_class)
    patch_dust3r_global_aligner(wrapper_module)
    capture_aligned_scene(wrapper_module)
    kwargs = dict(wrapper_kwargs or {})
    cleanup_cache_dir: Path | None = None
    if wrapper_class == "MASt3RSGAWrapper":
        default_tmp_root = (
            Path("/tmp")
            / (os.environ.get("USER") or getpass.getuser())
            / "feedforward_benchmark"
            / os.environ.get("SLURM_JOB_ID", os.environ.get("RUN_ID", "manual"))
        )
        cache_root = Path(os.environ.get("BENCHMARK_TMP_ROOT", str(default_tmp_root))).expanduser()
        try:
            cache_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            if exc.errno != errno.ENOSPC:
                raise
            fallback_root = (  # shared benchmark scratch, never the home folder
                Path(os.environ.get("BENCHMARK_AUX_ROOT",
                                    f"{SPATIALARENA_ROOT}/benchmarks"))
                / "_scratch"
                / (os.environ.get("USER") or getpass.getuser())
                / os.environ.get("SLURM_JOB_ID", os.environ.get("RUN_ID", "manual"))
            )
            fallback_root.mkdir(parents=True, exist_ok=True)
            cache_root = fallback_root
        cleanup_cache_dir = Path(tempfile.mkdtemp(prefix="mast3r_sga_", dir=cache_root))
        kwargs.setdefault("cache_dir", str(cleanup_cache_dir))
    try:
        model = Wrapper(name=method, ckpt_path=checkpoint_path, **kwargs).to(device).eval()
        mark_model_loaded()
        if args.depth_only:
            if not args.export_depth:
                raise ValueError("--depth_only requires --export_depth")
            batch_size = args.max_images_per_run if args.max_images_per_run > 0 else len(image_paths)
            if wrapper_class == "MASt3RSGAWrapper":
                mast3r_depth_batch_size = int(os.environ.get("MAST3R_DEPTH_BATCH_SIZE", "16"))
                if mast3r_depth_batch_size > 0:
                    batch_size = min(batch_size, mast3r_depth_batch_size)
            peak_memory = 0.0
            pose_extrinsics: list[np.ndarray] = []
            pose_intrinsics: list[np.ndarray] = []
            pose_output_size: tuple[int, int] | None = None
            for batch_index, batch_paths in enumerate(batched_image_paths(image_paths, batch_size), start=1):
                print(
                    f"[{method}] depth batch {batch_index}: {len(batch_paths)} images "
                    f"(batch_size={batch_size})",
                    flush=True,
                )
                duplicate_pair = len(batch_paths) == 1 and wrapper_class in PAIRWISE_WRAPPERS
                view_paths = batch_paths * 2 if duplicate_pair else batch_paths
                views, transforms = prepare_model_input(method, view_paths, args=args, device=device)
                views = move_views_to_device(views, device)
                with torch.no_grad():
                    outputs = model(views)
                if duplicate_pair:
                    outputs, views, transforms = outputs[:1], views[:1], transforms[:1]
                attach_aligned_confidence(outputs)
                depths, _points, _colors, confidence, extrinsics_w2c, intrinsics = outputs_to_standard(outputs, views)
                export_spatial_depth_predictions(args.output_path, batch_paths, depths, confidence, transforms)
                if args.export_colmap and extrinsics_w2c is not None and intrinsics is not None:
                    pose_extrinsics.append(np.asarray(extrinsics_w2c))
                    restored_intrinsics, width, height = restore_camera_export(intrinsics, transforms)
                    pose_intrinsics.append(restored_intrinsics)
                    pose_output_size = (width, height)
                if torch.cuda.is_available():
                    peak_memory = max(peak_memory, torch.cuda.max_memory_allocated() / (1024.0**2))
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
            from adapter_utils import write_adapter_metrics

            write_adapter_metrics(
                output_path=args.output_path,
                method=method,
                image_paths=image_paths,
                started_at=started_at,
                checkpoint=checkpoint_path,
                depth_only=True,
                metrics_extra={
                    "track": args.track,
                    "adapter": wrapper_module,
                    "max_images_per_run": batch_size,
                    "Peak GPU Memory": peak_memory,
                },
            )
            return

        views, _transforms = prepare_model_input(method, image_paths, args=args, device=device)
        views = move_views_to_device(views, device)
        with torch.no_grad():
            outputs = model(views)
        attach_aligned_confidence(outputs)
        depths, points, colors, confidence, extrinsics_w2c, intrinsics = outputs_to_standard(outputs, views)
        export_standard_artifacts(
            method=method,
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
            checkpoint=checkpoint_path,
            metrics_extra={"track": args.track, "adapter": wrapper_module, "selected_images": len(image_paths)},
        )
    finally:
        if cleanup_cache_dir is not None:
            shutil.rmtree(cleanup_cache_dir, ignore_errors=True)
