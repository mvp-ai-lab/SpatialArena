# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Shared helpers for external-method benchmark adapters.

Adapters in this directory are allowed to import and run external model code.
The benchmark evaluators remain artifact-only and consume only the files written
by these adapters.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import time
import struct
import sys
import random
from dataclasses import dataclass
from bisect import bisect_left
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

import numpy as np
from PIL import Image, ImageOps

if TYPE_CHECKING:
    import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from spatialarena.eval.colmap_view_resolver import (  # noqa: E402
    IMAGE_PATTERNS, canonical_image_root as shared_canonical_image_root,
    resolve_colmap_image_mappings,
)

ETH3D_RAW_DEPTH_SHAPE = (4032, 6048)
ETH3D_RAW_DEPTH_BYTES = ETH3D_RAW_DEPTH_SHAPE[0] * ETH3D_RAW_DEPTH_SHAPE[1] * 4
SINTEL_DPT_TAG_FLOAT = 202021.25


def has_colmap_model_files(path: Path) -> bool:
    """Return True if a directory contains minimal COLMAP camera/image/point files."""

    path = Path(path)
    if not path.is_dir():
        return False
    return all(
        any((path / f"{stem}{extension}").is_file()
            for extension in (".bin", ".txt"))
        for stem in ("cameras", "images", "points3D"))


def colmap_prior_candidates(data_path: Path,
                            colmap_model_subdir: str = "0") -> list[Path]:
    """Return benchmark camera-prior candidate directories in preference order."""

    data_path = Path(data_path)
    return [
        data_path / "sparse" / colmap_model_subdir,
        data_path / "sparse",
        data_path / "dslr_calibration_undistorted",
        data_path / "dslr_calibration_jpg",
    ]


def resolve_colmap_prior_path(
    data_path: Path,
    *,
    colmap_model_subdir: str = "0",
) -> Path | None:
    """Resolve the first available COLMAP camera-prior directory for a scene."""

    for candidate in colmap_prior_candidates(data_path, colmap_model_subdir):
        if has_colmap_model_files(candidate):
            return candidate
    return None


def find_image_paths(images_dir: Path) -> list[str]:
    image_paths: list[str] = []
    for pattern in IMAGE_PATTERNS:
        image_paths.extend(
            glob.glob(str(images_dir / "**" / pattern), recursive=True))
    return sorted(image_paths)


def canonical_image_root(data_path: Path) -> Path:
    """Return the image directory used by the benchmark for a scene."""

    return shared_canonical_image_root(data_path)


ORDER_NAMES = ("perm0", "perm1", "perm2", "perm3", "perm4", "sequential",
               "sorted")
ORDER_SEEDS = {
    "perm0": 7,
    "perm1": 13,
    "perm2": 23,
    "perm3": 37,
    "perm4": 53,
}


def apply_image_order(image_paths: list[str],
                      order_name: str | None) -> list[str]:
    if not order_name or order_name in {"sorted", "sequential"}:
        return image_paths
    if order_name not in ORDER_SEEDS:
        raise ValueError(f"Unsupported benchmark image order {order_name!r}; "
                         f"expected one of {', '.join(ORDER_NAMES)}")
    indices = list(range(len(image_paths)))
    rng = random.Random(ORDER_SEEDS[order_name])
    rng.shuffle(indices)
    return [image_paths[index] for index in indices]


def resolve_capped_image_indices(num_images: int,
                                 interval: int,
                                 max_images_per_run: int = 0) -> list[int]:
    """Index-only version of the same three-step selection `select_interval`
    applies, for callers that carry other per-image arrays (e.g. da3.py's
    COLMAP camera-pose/intrinsics priors) needing the identical subsample
    applied in lockstep -- `select_interval` alone can't do that since it
    only ever sees a flat path list. See `select_interval`'s docstring for
    what each step does and why (DA3-BENCH's max_frames cap, applied before
    any --depth_only-specific batching).
    """
    indices = list(range(num_images))
    if interval > 1:
        indices = indices[::interval]
    if single_view_mode():
        # MonoDepth scores every frame: no total cap. run_method_case_list.sh splits the scene into adapter calls of
        # at most max_images_per_run frames (scripts/inference/single_view_groups.py) instead.
        return indices
    max_input_frames = int(
        os.environ.get("BENCHMARK_MAX_INPUT_FRAMES", "0") or 0)
    if max_input_frames > 0 and len(indices) > max_input_frames:
        rng = random.Random(
            int(os.environ.get("BENCHMARK_MAX_INPUT_FRAMES_SEED", "42") or 42))
        shuffled = list(indices)
        rng.shuffle(shuffled)
        indices = sorted(shuffled[:max_input_frames])
    if max_images_per_run > 0 and len(indices) > max_images_per_run:
        positions = np.linspace(0,
                                len(indices) - 1,
                                max_images_per_run).round().astype(int)
        indices = [
            indices[position] for position in sorted(set(positions.tolist()))
        ]
    return indices


def select_interval(image_paths: list[str],
                    interval: int,
                    max_images_per_run: int = 0) -> list[str]:
    """The single shared entry point every adapter uses to go from "every
    image found on disk" to "the exact frames this run will process."

    Three independent steps, always applied in this order, regardless of
    --depth_only (a scene needing more than one depth_only batch is an
    independent model forward pass per batch, each with its own arbitrary
    coordinate frame; capping total frames here -- before any adapter-specific
    branching -- is what keeps a scene to a single, globally consistent batch):

    1. `interval`: keep every Nth frame (adapter CLI arg).
    2. `BENCHMARK_MAX_INPUT_FRAMES` (env var, set by run_method_case_list.sh
       per suite): the DA3-BENCH-matching total-frame cap for the whole
       scene (eval_bench.yaml's `max_frames: 100`, applied to pose and
       reconstruction modes alike) -- fixed-seed random subset, matching
       DA3's own `evaluator.py::_sample_frames`.
    3. `max_images_per_run` (adapter CLI arg, e.g. depth_only batch size):
       an optional further, method-specific cap (e.g. IGGT's 12, driven by
       its lack of any batching support) -- evenly-spaced subset of what's
       left after step 2.

    In single-view mode (MonoDepth) steps 2 and 3 are skipped: every frame of the scene is predicted, and the engine
    bounds each adapter call to ``max_images_per_run`` frames through ``BENCHMARK_IMAGE_SUBSET`` lists.
    """
    indices = resolve_capped_image_indices(len(image_paths), interval,
                                           max_images_per_run)
    image_paths = [image_paths[index] for index in indices]
    order_name = os.environ.get("BENCHMARK_IMAGE_ORDER", "").strip() or None
    return restrict_to_image_subset(apply_image_order(image_paths, order_name))


# Set by run_method_case_list.sh to one adapter call's image list (scripts/inference/single_view_groups.py) when a
# single-view scene mixes image sizes or has more than max_images_per_run frames; the lists together are the scene.
IMAGE_SUBSET_ENV = "BENCHMARK_IMAGE_SUBSET"


def restrict_to_image_subset(image_paths: list[str]) -> list[str]:
    """Keep only the images listed in the ``BENCHMARK_IMAGE_SUBSET`` file (all of them when it is unset)."""
    subset_file = os.environ.get(IMAGE_SUBSET_ENV, "").strip()
    if not subset_file:
        return image_paths
    subset = set(
        Path(subset_file).read_text(encoding="utf-8").split("\n")) - {""}
    kept = [path for path in image_paths if path in subset]
    if len(kept) != len(subset):
        raise ValueError(
            f"{len(subset) - len(kept)} image(s) of {subset_file} are not among the selected frames"
        )
    return kept


def _image_lookup_keys(image_path: str | Path,
                       images_dir: Path | None = None) -> tuple[str, ...]:
    path = Path(image_path)
    normalized = str(image_path).replace("\\", "/")
    keys = [normalized, path.name, path.stem]
    if images_dir is not None:
        try:
            relative = path.relative_to(images_dir)
            keys.extend([relative.as_posix(), str(relative), relative.stem])
        except ValueError:
            pass
    return tuple(dict.fromkeys(keys))


def _normalized_relative_image_name(value: str | Path) -> str:
    text = str(value).replace("\\", "/").lstrip("./")
    parts = text.split("/")
    if len(parts) > 1:
        directory = parts[-2].lower()
        directory = directory.removesuffix("_undistorted").removesuffix("_jpg")
        parts[-2] = directory
    return "/".join(parts)


def _colmap_name_keys(image_name: str) -> tuple[str, ...]:
    """Build normalized keys for a COLMAP image name.

    COLMAP names may contain dataset-specific directory prefixes while the
    canonical benchmark image path may not. Keep exact relative names first,
    then use basename and stem only when they identify one record.
    """

    path = Path(str(image_name).replace("\\", "/"))
    return tuple(
        dict.fromkeys((
            path.as_posix(),
            _normalized_relative_image_name(path),
            path.name,
            path.stem,
        )))


def match_colmap_images_to_real_paths(
    data_path: Path,
    image_paths: list[str],
    *,
    colmap_path: Path,
) -> list[str]:
    """Return real RGB paths represented by a COLMAP model.

    COLMAP is authoritative for camera-conditioned inference. This mirrors
    DepthAnything3's ColmapManager behavior: partial models, such as
    ScanNet++, intentionally evaluate only reconstructed views.
    """

    mappings, _diagnostics = resolve_colmap_image_mappings(
        data_path,
        image_paths,
        colmap_path=colmap_path,
        strict=True,
    )
    return [str(mapping.image_path) for mapping in mappings]


@dataclass(frozen=True)
class CameraConditionedInputs:
    image_paths: list[str]
    poses: np.ndarray
    intrinsics: np.ndarray


def resolve_camera_conditioned_inputs(
    data_path: Path,
    *,
    colmap_path: Path | None,
    colmap_model_subdir: str = "0",
) -> CameraConditionedInputs:
    """Resolve images and camera priors in one authoritative ordering."""

    images_dir = canonical_image_root(data_path)
    image_paths = find_image_paths(images_dir)
    image_paths = resolve_camera_conditioned_image_paths(
        data_path,
        image_paths,
        colmap_path=colmap_path,
    )
    poses, intrinsics = load_colmap_camera_priors(
        data_path,
        image_paths,
        colmap_path=colmap_path,
        colmap_model_subdir=colmap_model_subdir,
    )
    return CameraConditionedInputs(image_paths, poses, intrinsics)


def resolve_camera_conditioned_image_paths(
    data_path: Path,
    image_paths: list[str],
    *,
    colmap_path: Path | None,
) -> list[str]:
    """Select real images represented by the camera-prior COLMAP model."""

    if colmap_path is None:
        raise FileNotFoundError("rgb_cameras requires --colmap_path")
    return match_colmap_images_to_real_paths(
        data_path,
        image_paths,
        colmap_path=colmap_path,
    )


def _colmap_camera_to_intrinsics(camera: Any) -> np.ndarray:
    params = np.asarray(camera.params, dtype=np.float32).reshape(-1)
    intrinsics = np.eye(3, dtype=np.float32)
    if camera.model == "PINHOLE":
        intrinsics[0,
                   0], intrinsics[1,
                                  1], intrinsics[0,
                                                 2], intrinsics[1,
                                                                2] = params[:4]
    elif camera.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        intrinsics[0, 0] = params[0]
        intrinsics[1, 1] = params[0]
        intrinsics[0, 2] = params[1]
        intrinsics[1, 2] = params[2]
    elif camera.model in {"OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
        intrinsics[0,
                   0], intrinsics[1,
                                  1], intrinsics[0,
                                                 2], intrinsics[1,
                                                                2] = params[:4]
    else:
        raise ValueError(
            f"Unsupported COLMAP camera model for priors: {camera.model}")
    return intrinsics


def load_colmap_camera_priors(
    data_path: Path,
    image_paths: list[str],
    *,
    colmap_path: Path | None = None,
    colmap_model_subdir: str = "0",
) -> tuple[np.ndarray, np.ndarray]:
    """Load per-image camera-to-world poses and intrinsics from canonical COLMAP priors."""

    sparse_dir = None
    if colmap_path is not None:
        candidate = Path(colmap_path)
        if has_colmap_model_files(candidate):
            sparse_dir = candidate
        else:
            raise FileNotFoundError(
                f"Explicit COLMAP prior directory is invalid: {candidate}")
    else:
        sparse_dir = resolve_colmap_prior_path(
            data_path,
            colmap_model_subdir=colmap_model_subdir,
        )
    if sparse_dir is None:
        candidates = ", ".join(
            str(path)
            for path in colmap_prior_candidates(data_path, colmap_model_subdir))
        raise FileNotFoundError(
            f"COLMAP prior directory not found. Tried: {candidates}")
    # Non-strict: callers may request a batch subset of a larger COLMAP model
    # (for example one inference chunk of a multi-thousand-frame scene), and
    # COLMAP records outside that subset are not a mapping failure. Every
    # requested path is still required to resolve, checked below.
    mappings, _diagnostics = resolve_colmap_image_mappings(
        data_path,
        image_paths,
        colmap_path=sparse_dir,
        strict=False,
    )
    by_path = {str(mapping.image_path): mapping.record for mapping in mappings}
    missing = [str(path) for path in image_paths if str(path) not in by_path]
    if missing:
        preview = ", ".join(Path(path).name for path in missing[:5])
        raise FileNotFoundError(
            f"Missing COLMAP camera priors for {len(missing)} selected images: {preview}"
        )

    c2w_list: list[np.ndarray] = []
    intrinsics_list: list[np.ndarray] = []
    for image_path in image_paths:
        record = by_path[str(image_path)]
        c2w_list.append(record.c2w)
        intrinsics_list.append(record.intrinsic)
    return np.stack(c2w_list, axis=0), np.stack(intrinsics_list, axis=0)


def build_default_camera_priors(
        image_paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Build identity poses and centered pinhole intrinsics for image-only scenes.

    Some modality-conditioned methods require intrinsics even when the benchmark
    scene provides RGB+depth but no calibrated camera model. These priors are a
    deterministic fallback for depth-conditioned inference only; they should not
    be used to evaluate camera-conditioned tracks.
    """

    c2w_list: list[np.ndarray] = []
    intrinsics_list: list[np.ndarray] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            width, height = image.size
        focal = float(max(width, height))
        intrinsics = np.array(
            [
                [focal, 0.0, (float(width) - 1.0) * 0.5],
                [0.0, focal, (float(height) - 1.0) * 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        c2w_list.append(np.eye(4, dtype=np.float32))
        intrinsics_list.append(intrinsics)
    return np.stack(c2w_list, axis=0), np.stack(intrinsics_list, axis=0)


def depth_prior_candidates(data_path: Path) -> list[Path]:
    """Return candidate directories containing per-image depth priors.

    Most benchmark cases materialize RGB+depth priors under ``depths/`` next to
    the images. Sintel keeps RGB frames under ``training/final/<scene>`` or
    ``training/clean/<scene>`` and metric depth maps under the sibling
    ``training/depth/<scene>`` directory, so adapters must resolve that layout
    before deciding priors are unavailable.
    """

    data_path = Path(data_path)
    candidates = [
        data_path / "depths",
        data_path / "depth",
    ]
    if data_path.name in {"images", "rgb", "color"}:
        scene_root = data_path.parent
        candidates.extend([scene_root / "depths", scene_root / "depth"])
    if data_path.parent.name in {"final", "clean"}:
        candidates.append(data_path.parent.parent / "depth" / data_path.name)

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve() if candidate.exists() else candidate
        if resolved not in seen:
            unique.append(candidate)
            seen.add(resolved)
    return unique


def read_sintel_dpt(path: Path) -> np.ndarray:
    """Read a Sintel ``.dpt`` depth map as float32 meters."""

    with path.open("rb") as handle:
        tag = struct.unpack("f", handle.read(4))[0]
        if abs(tag - SINTEL_DPT_TAG_FLOAT) > 1e-4:
            raise ValueError(f"Invalid Sintel .dpt tag in {path}: {tag}")
        width = struct.unpack("i", handle.read(4))[0]
        height = struct.unpack("i", handle.read(4))[0]
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid Sintel .dpt shape in {path}: {width}x{height}")
        depth = np.fromfile(handle, dtype=np.float32, count=width * height)
    if depth.size != width * height:
        raise ValueError(
            f"Truncated Sintel .dpt file {path}: expected {width * height} values"
        )
    return depth.reshape(height, width)


def read_pfm_depth(path: Path) -> np.ndarray:
    """Read a PFM depth map as float32.

    RobustMVD-style benchmark samples commonly keep key-view metric depths as
    ``.pfm`` files. Adapter-side multimodal depth loading must support the same
    format as the evaluator so RGB+depth tracks can consume these priors.
    """

    with path.open("rb") as handle:
        header = handle.readline().decode("ascii").rstrip()
        if header not in {"PF", "Pf"}:
            raise ValueError(f"Invalid PFM header in {path}")
        is_color = header == "PF"
        dims_line = handle.readline().decode("ascii").strip()
        while dims_line.startswith("#"):
            dims_line = handle.readline().decode("ascii").strip()
        width, height = (int(value) for value in dims_line.split())
        scale = float(handle.readline().decode("ascii").strip())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(handle, endian + "f")
    shape = (height, width, 3) if is_color else (height, width)
    return np.flipud(data.reshape(shape)).astype(np.float32, copy=False)


def parse_numeric_stem(path: Path) -> float | None:
    """Parse numeric timestamp-like file stems used by RGB-D video datasets."""

    try:
        return float(path.stem)
    except ValueError:
        return None


def nearest_numeric_depth_path(
    image_path: Path,
    numeric_depth_files: list[tuple[float, Path]],
    *,
    max_delta_seconds: float = 0.12,
) -> Path | None:
    """Return the nearest timestamp-matched depth path for an RGB frame."""

    timestamp = parse_numeric_stem(image_path)
    if timestamp is None or not numeric_depth_files:
        return None
    timestamps = [item[0] for item in numeric_depth_files]
    insert_at = bisect_left(timestamps, timestamp)
    candidates: list[tuple[float, Path]] = []
    if insert_at < len(numeric_depth_files):
        candidates.append(numeric_depth_files[insert_at])
    if insert_at > 0:
        candidates.append(numeric_depth_files[insert_at - 1])
    if not candidates:
        return None
    nearest_timestamp, nearest_path = min(
        candidates,
        key=lambda item: abs(item[0] - timestamp),
    )
    if abs(nearest_timestamp - timestamp) <= max_delta_seconds:
        return nearest_path
    return None


DEPTH_PRIOR_PATTERNS = (
    "*.npy",
    "*.npz",
    "*.dpt",
    "*.DPT",
    "*.pfm",
    "*.PFM",
    "*.png",
    "*.PNG",
    "*.tif",
    "*.TIF",
    "*.tiff",
    "*.TIFF",
    "*.exr",
    "*.jpg",
    "*.JPG",
    "*.jpeg",
    "*.JPEG",
    "*.pt",
    "*.pth",
)


def _build_depth_prior_index(
        data_path: Path) -> tuple[dict[str, Path], list[tuple[float, Path]]]:
    depth_dirs = [
        candidate for candidate in depth_prior_candidates(data_path)
        if candidate.is_dir()
    ]
    if not depth_dirs:
        return {}, []
    depth_files: dict[str, Path] = {}
    numeric_depth_files: list[tuple[float, Path]] = []
    for suffix in DEPTH_PRIOR_PATTERNS:
        for depth_dir in depth_dirs:
            for path in depth_dir.glob(f"**/{suffix}"):
                for key in (path.name, path.stem):
                    depth_files.setdefault(key, path)
                numeric_stem = parse_numeric_stem(path)
                if numeric_stem is not None:
                    numeric_depth_files.append((numeric_stem, path))
    numeric_depth_files.sort(key=lambda item: item[0])
    return depth_files, numeric_depth_files


def _match_depth_prior_path(
    image_path: str | Path,
    depth_files: dict[str, Path],
    numeric_depth_files: list[tuple[float, Path]],
) -> Path | None:
    path = Path(image_path)
    stem_alternatives = [
        path.stem,
        path.name,
        path.stem.replace(".color", ".depth"),
        path.stem.replace(".color", ".depth.proj"),
        path.name.replace(".color.", ".depth."),
        path.name.replace(".color.", ".depth.proj."),
    ]
    depth_path = None
    for key in stem_alternatives:
        depth_path = depth_files.get(key)
        if depth_path is not None:
            break
    if depth_path is None:
        depth_path = nearest_numeric_depth_path(path, numeric_depth_files)
    return depth_path


def resolve_optional_depth_prior_paths(
    data_path: Path,
    image_paths: list[str],
) -> tuple[list[Path | None], list[str]]:
    """Resolve per-image depth priors while preserving missing-frame positions."""

    depth_files, numeric_depth_files = _build_depth_prior_index(data_path)
    if not depth_files and not numeric_depth_files:
        return [None for _path in image_paths
               ], [str(path) for path in image_paths]

    depth_paths: list[Path | None] = []
    missing: list[str] = []
    for image_path in image_paths:
        depth_path = _match_depth_prior_path(image_path, depth_files,
                                             numeric_depth_files)
        if depth_path is None:
            missing.append(str(image_path))
        depth_paths.append(depth_path)
    return depth_paths, missing


def resolve_depth_prior_paths(
        data_path: Path,
        image_paths: list[str]) -> tuple[list[Path], list[str]]:
    """Resolve canonical per-image depth prior paths matching selected images."""

    optional_paths, missing = resolve_optional_depth_prior_paths(
        data_path, image_paths)
    depth_paths: list[Path] = []
    for depth_path in optional_paths:
        if depth_path is None:
            continue
        depth_paths.append(depth_path)
    return depth_paths, missing


def dataset_depth_png_scale(data_path: Path) -> float | None:
    """Integer depth-PNG units per metre declared by the scene's dataset (``"depth_png_scale"`` in the dataset's
    ``metadata.json``, e.g. 5000 for TUM RGB-D, 1000 for millimetre PNGs), or None if the dataset declares none."""
    for metadata_path in (Path(data_path) / "metadata.json",
                          Path(data_path).parent / "metadata.json"):
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and payload.get("depth_png_scale"):
            return float(payload["depth_png_scale"])
    return None


def load_depth_prior_file(depth_path: Path,
                          image_path: str | Path,
                          png_scale: float | None = None) -> np.ndarray:
    """Load one depth prior file as a finite float32 ``H x W`` array in metres.

    ``png_scale`` (from :func:`dataset_depth_png_scale`) converts integer image formats (PNG/TIFF) explicitly. Without
    it, image depths whose maximum exceeds 100 are assumed to be millimetres -- right for 7-Scenes/DTU-style PNGs,
    wrong for e.g. TUM's 5000-per-metre PNGs, which is why datasets declare their scale.
    """

    path = Path(image_path)
    if depth_path.suffix.lower() == ".npy":
        depth = np.load(depth_path).astype(np.float32)
    elif depth_path.suffix.lower() == ".npz":
        with np.load(depth_path) as data:
            if "depth" in data:
                depth = data["depth"].astype(np.float32)
            else:
                depth = data[data.files[0]].astype(np.float32)
    elif depth_path.suffix.lower() == ".dpt":
        depth = read_sintel_dpt(depth_path)
    elif depth_path.suffix.lower() == ".pfm":
        depth = read_pfm_depth(depth_path)
    elif (depth_path.suffix.lower() in {".jpg", ".jpeg"} and
          depth_path.stat().st_size == ETH3D_RAW_DEPTH_BYTES):
        depth = np.fromfile(depth_path,
                            dtype=np.float32).reshape(ETH3D_RAW_DEPTH_SHAPE)
    elif depth_path.suffix.lower() in {".pt", ".pth"}:
        import torch

        payload = torch.load(depth_path, map_location="cpu", weights_only=False)
        if isinstance(payload,
                      dict) and "depth" in payload and "coord" in payload:
            with Image.open(path) as image:
                width, height = image.size
            sparse_depth = np.asarray(payload["depth"],
                                      dtype=np.float32).reshape(-1)
            sparse_coord = np.asarray(payload["coord"],
                                      dtype=np.float32).reshape(-1, 2)
            x = np.rint(sparse_coord[:, 0]).astype(np.int64)
            y = np.rint(sparse_coord[:, 1]).astype(np.int64)
            valid = (np.isfinite(sparse_depth) &
                     np.isfinite(sparse_coord).all(axis=1) & (x >= 0) &
                     (x < width) & (y >= 0) & (y < height))
            depth_sum = np.zeros((height, width), dtype=np.float32)
            depth_count = np.zeros((height, width), dtype=np.float32)
            np.add.at(depth_sum, (y[valid], x[valid]), sparse_depth[valid])
            np.add.at(depth_count, (y[valid], x[valid]), 1.0)
            depth = np.zeros((height, width), dtype=np.float32)
            filled = depth_count > 0
            depth[filled] = depth_sum[filled] / depth_count[filled]
        elif isinstance(payload, dict) and "depth" in payload:
            depth = np.asarray(payload["depth"], dtype=np.float32)
        else:
            depth = np.asarray(payload, dtype=np.float32)
    else:
        depth = np.asarray(Image.open(depth_path), dtype=np.float32)
        # 16-bit depth images mark missing depth near the type's maximum: 65535 in raw 7-Scenes frames, and
        # 65000-65534 in its colour-registered *.depth.proj.png (resampled around that value), with nothing valid
        # between 5 m and 65 m. Read as values they are a far-away wall of points. 65000 is beyond any valid depth at
        # the 16-bit scales used here (65 m in mm, 13 m at TUM's 5000/m, 254 m at KITTI's 256/m).
        depth[depth >= 65000.0] = 0.0
        if png_scale is not None:
            if depth.ndim == 3:
                depth = depth[..., 0]
            depth[~np.isfinite(depth)] = 0.0
            return (depth / png_scale).astype(np.float32, copy=False)
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth[~np.isfinite(depth)] = 0.0
    if depth_path.suffix.lower() not in {".npy", ".npz"
                                        } and depth.max(initial=0.0) > 100.0:
        depth = depth / 1000.0
    return depth.astype(np.float32, copy=False)


def load_depth_priors(data_path: Path,
                      image_paths: list[str]) -> list[np.ndarray]:
    """Load canonical per-image depth priors matching selected image stems."""

    depth_paths, missing = resolve_depth_prior_paths(data_path, image_paths)
    if not depth_paths and missing:
        candidates = ", ".join(
            str(candidate) for candidate in depth_prior_candidates(data_path))
        raise FileNotFoundError(
            f"Depth prior directory not found. Tried: {candidates}")
    if missing:
        preview = ", ".join(Path(path).name for path in missing[:5])
        raise FileNotFoundError(
            f"Missing depth priors for {len(missing)} selected images: {preview}"
        )

    png_scale = dataset_depth_png_scale(data_path)
    depths: list[np.ndarray] = []
    for image_path, depth_path in zip(image_paths, depth_paths, strict=True):
        depths.append(load_depth_prior_file(depth_path, image_path, png_scale))
    return depths


def load_optional_depth_priors(
    data_path: Path,
    image_paths: list[str],
) -> tuple[list[np.ndarray | None], np.ndarray, list[str]]:
    """Load depth priors as an aligned list and an explicit per-view availability mask."""

    depth_paths, missing = resolve_optional_depth_prior_paths(
        data_path, image_paths)
    png_scale = dataset_depth_png_scale(data_path)
    depths: list[np.ndarray | None] = []
    has_depth = np.zeros((len(image_paths),), dtype=bool)
    for index, (image_path, depth_path) in enumerate(
            zip(image_paths, depth_paths, strict=True)):
        if depth_path is None:
            depths.append(None)
            continue
        depths.append(load_depth_prior_file(depth_path, image_path, png_scale))
        has_depth[index] = True
    return depths, has_depth, missing


def load_masked_depth_priors(
    data_path: Path,
    image_paths: list[str],
) -> tuple[list[np.ndarray | None], np.ndarray, list[str]]:
    """``load_optional_depth_priors`` with the masked-depth tracks' cells dropped from every map.

    The masking (``spatialarena/depth_prior_masking.py``) is applied at each prior's own resolution, before an
    adapter resizes it, so every method of ``rgb_masked_depth``/``rgb_masked_depth_cameras`` is given the same input.
    """
    from spatialarena.depth_prior_masking import mask_depth_prior

    depths, has_depth, missing = load_optional_depth_priors(
        data_path, image_paths)
    masked = [
        None
        if depth is None else mask_depth_prior(depth, data_path, image_path)
        for depth, image_path in zip(depths, image_paths, strict=True)
    ]
    return masked, has_depth, missing


def load_sparse_depth_priors(
    data_path: Path,
    image_paths: list[str],
    *,
    model_long_side: int | None = None,
) -> tuple[list[np.ndarray | None], np.ndarray, list[str]]:
    """The sparse-depth tracks' priors (``spatialarena/sparse_depth.py``) in ``load_optional_depth_priors``' form.

    Each image's sparse points (KITTI LiDAR or COLMAP keypoints) are drawn as a dense map at the image's own size
    (0 = no depth), each point as a small square sized so it survives the adapter's nearest-neighbour resize to a
    ``model_long_side`` input as about one pixel; an image without a prior is ``None`` and listed as missing.
    """
    from spatialarena.sparse_depth import load_sparse_depth  # pylint: disable=import-outside-toplevel

    depths: list[np.ndarray | None] = []
    has_depth = np.zeros((len(image_paths),), dtype=bool)
    missing: list[str] = []
    for index, image_path in enumerate(image_paths):
        with Image.open(image_path) as image:
            size = image.size
        depth = load_sparse_depth(data_path,
                                  image_path,
                                  image_size=size,
                                  model_long_side=model_long_side)
        if depth is None or not np.any(depth > 0):
            depths.append(None)
            missing.append(str(image_path))
            continue
        depths.append(depth)
        has_depth[index] = True
    return depths, has_depth, missing


def resize_depth_to_image_size(
    depth: np.ndarray,
    image_size: tuple[int, int],
    *,
    resample: int = Image.Resampling.NEAREST,
) -> np.ndarray:
    """Resize a depth prior to match a PIL image ``(width, height)`` size.

    Some benchmark layouts, notably ETH3D RGB+depth priors, pair raw or
    undistorted depth maps with RGB images whose pixel dimensions differ after
    upstream preprocessing. External methods such as MapAnything assert exact
    image/depth shape equality before applying their own crop/resize path, so
    adapters must normalize the prior shape first.
    """

    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    target_width, target_height = image_size
    if depth.shape[:2] == (target_height, target_width):
        return depth.astype(np.float32, copy=False)
    depth_image = Image.fromarray(depth.astype(np.float32, copy=False),
                                  mode="F")
    resized = depth_image.resize((target_width, target_height),
                                 resample=resample)
    out = np.asarray(resized, dtype=np.float32).copy()
    out[~np.isfinite(out)] = 0.0
    return out


# Set to "1" by run_method_case_list.sh for MonoDepth cases: every image is predicted on its own, never as one view
# of a multi-view sequence. Adapters either run independent single-view samples as a [B, 1, 3, H, W] batch
# (fold_single_view_outputs) or call the model once per image (batched_image_paths returns one-image chunks).
SINGLE_VIEW_ENV = "BENCHMARK_SINGLE_VIEW"


def single_view_mode() -> bool:
    """Whether each image must be predicted independently (MonoDepth)."""
    return os.environ.get(SINGLE_VIEW_ENV, "0") == "1"


def batched_image_paths(
    image_paths: list[str],
    max_images_per_run: int,
    *,
    independent_batch: bool = False,
) -> list[list[str]]:
    """Chunks of image paths, one model call each. In single-view mode a chunk is one image, unless the caller runs
    the chunk as independent single-view samples (``independent_batch``, the [B, 1, 3, H, W] path)."""
    if single_view_mode() and not independent_batch:
        return [[path] for path in image_paths]
    if max_images_per_run <= 0 or max_images_per_run >= len(image_paths):
        return [image_paths]
    return [
        image_paths[start:start + max_images_per_run]
        for start in range(0, len(image_paths), max_images_per_run)
    ]


def fold_single_view_outputs(outputs: Any, num_samples: int) -> Any:
    """Map the outputs of a [B, 1, ...] forward over ``num_samples`` independent single-view samples back to the
    [1, B, ...] layout of one B-view sequence, so an adapter's multi-view post-processing applies unchanged.

    Every tensor or array whose leading dims are (num_samples, 1) gets them swapped; dicts, attribute objects,
    lists and tuples are folded recursively; anything else (scalars, metadata) is returned as is.
    """
    shape = getattr(outputs, "shape", None)
    if shape is not None and len(shape) >= 2 and tuple(
            shape[:2]) == (num_samples, 1):
        if isinstance(outputs, np.ndarray):
            return np.swapaxes(outputs, 0, 1)
        return outputs.transpose(0, 1)  # torch.Tensor
    if isinstance(outputs, dict):
        folded = type(
            outputs)()  # keeps dict subclasses such as addict.Dict (DA3)
        for key, value in outputs.items():
            folded[key] = fold_single_view_outputs(value, num_samples)
        return folded
    if isinstance(outputs, list):
        return [
            fold_single_view_outputs(value, num_samples) for value in outputs
        ]
    if isinstance(outputs, tuple):
        return tuple(
            fold_single_view_outputs(value, num_samples) for value in outputs)
    return outputs


def round_up_to_multiple(value: int, multiple: int) -> int:
    if value <= 0:
        raise ValueError(f"Expected a positive value, got {value}")
    if multiple <= 0:
        raise ValueError(f"Expected a positive multiple, got {multiple}")
    return ((value + multiple - 1) // multiple) * multiple


def patch_grid_compatible_resolution(
    requested_resolution: int,
    *,
    patch_size: int,
    grid_multiple: int = 1,
) -> int:
    """Return the nearest larger square size that satisfies patch-grid constraints."""
    if patch_size <= 0:
        raise ValueError(f"Expected positive patch_size, got {patch_size}")
    if grid_multiple <= 0:
        raise ValueError(
            f"Expected positive grid_multiple, got {grid_multiple}")
    return round_up_to_multiple(requested_resolution,
                                patch_size * grid_multiple)


FIXED_INPUT_ENV = "FIXED_INPUT_RESOLUTION"


def fixed_input_size(image_paths: list[str]) -> tuple[int, int] | None:
    """(width, height) of the inputs when the engine feeds pre-sized frames every method must use as they are
    (``FIXED_INPUT_RESOLUTION=1``, the full-slam suite: slam_protocol.DATASET_INPUT_SIZE), else None.

    Adapters then skip their own rescale (a pad to the model's required shape, or a center crop to a 16 px patch
    grid, is all they may still do) and size every model-specific setting from this.
    """
    if os.environ.get(FIXED_INPUT_ENV, "0") != "1" or not image_paths:
        return None
    sizes = set()
    for image_path in image_paths:
        with Image.open(image_path) as image:
            sizes.add(image.size)
    if len(sizes) != 1:
        raise ValueError(
            f"Fixed-resolution inputs must share one size, found {sorted(sizes)}"
        )
    return sizes.pop()


# Optional root for the long-sequence SLAM adapters' scratch (their methods keep per-chunk dense results on disk while
# running: hundreds of GB on an 18,846-frame VBR sequence). Unset: the scratch lives under the case's output path.
SLAM_SCRATCH_ENV = "SLAM_SCRATCH_ROOT"


def slam_work_dir(method: str, output_path: Path) -> Path:
    """A fresh scratch directory for one SLAM case: under ``$SLAM_SCRATCH_ROOT`` when set, else ``output_path``."""
    import tempfile

    root = Path(os.environ[SLAM_SCRATCH_ENV]) if os.environ.get(
        SLAM_SCRATCH_ENV) else Path(output_path)
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{method.lower()}_", dir=root))


_MODEL_INPUT_SIZE: list[int] | None = None


def record_model_input(width: int, height: int) -> None:
    """Record the (width, height) of the image tensor the model actually received; reported in ``metrics.json``
    as ``model_input_size`` (``runtime_metrics``) so the fixed-resolution protocol can be checked per case."""
    global _MODEL_INPUT_SIZE  # pylint: disable=global-statement  # per-process record read by runtime_metrics
    _MODEL_INPUT_SIZE = [int(width), int(height)]


def load_patch_grid_tensor(image_paths: list[str], patch_size: int):
    """[N, 3, H, W] float tensor in [0, 1] of fixed-resolution frames, each center-cropped to a multiple of
    ``patch_size`` (e.g. 504x378 -> 496x368 for 16 px patches) and never rescaled."""
    import torch
    from torchvision.transforms import functional as TF

    tensors = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            crop_w, crop_h = width // patch_size * patch_size, height // patch_size * patch_size
            left, top = (width - crop_w) // 2, (height - crop_h) // 2
            tensors.append(
                TF.to_tensor(
                    image.crop((left, top, left + crop_w, top + crop_h))))
    return torch.stack(tensors)


def materialize_resized_images(
    image_paths: list[str],
    output_dir: Path,
    size: int | tuple[int, int],
) -> list[str]:
    """Write fixed-size padded RGB copies for adapters that stack inputs."""
    if isinstance(size, int):
        target_width = target_height = size
    else:
        target_width, target_height = size
    if target_width <= 0 or target_height <= 0:
        raise ValueError(f"Invalid target image size: {size}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.iterdir():
        if existing.is_file() or existing.is_symlink():
            existing.unlink()

    resized_paths: list[str] = []
    manifest: list[dict[str, Any]] = []
    for index, image_path in enumerate(image_paths):
        source_path = Path(image_path)
        with Image.open(source_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            original_width, original_height = image.size
            scale = min(
                target_width / max(original_width, 1),
                target_height / max(original_height, 1),
            )
            resized_width = max(1, int(round(original_width * scale)))
            resized_height = max(1, int(round(original_height * scale)))
            resample = Image.Resampling.LANCZOS if scale < 1.0 else Image.Resampling.BICUBIC
            resized = image.resize((resized_width, resized_height), resample)
            canvas = Image.new("RGB", (target_width, target_height), (0, 0, 0))
            offset_x = (target_width - resized_width) // 2
            offset_y = (target_height - resized_height) // 2
            canvas.paste(resized, (offset_x, offset_y))

        destination = output_dir / f"{index:06d}.png"
        canvas.save(destination)
        resized_paths.append(str(destination))
        manifest.append({
            "index": index,
            "source": str(source_path),
            "resized": str(destination),
            "source_size": [original_width, original_height],
            "resized_size": [resized_width, resized_height],
            "canvas_size": [target_width, target_height],
            "offset": [offset_x, offset_y],
        })

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return resized_paths


def should_keep_intermediate(args: Any) -> bool:
    """Return True when adapter temporary files should be retained for debugging."""
    if bool(getattr(args, "keep_intermediate", False)):
        return True
    return os.environ.get("KEEP_INTERMEDIATE",
                          "0").lower() in {"1", "true", "yes", "on"}


def cleanup_intermediate_paths(paths: list[Path | None], *, keep: bool) -> None:
    """Remove adapter scratch directories after successful artifact export (None entries: nothing was staged)."""
    if keep:
        return
    for path in paths:
        if path is not None and path.is_dir():
            shutil.rmtree(path)


def planned_layout(args: Any, method: str, num_images: int) -> dict[str, Any]:
    return {
        "method":
            method,
        "data_path":
            str(args.data_path),
        "output_path":
            str(args.output_path),
        "num_images":
            num_images,
        "checkpoint":
            getattr(args, "ckpt_path", None),
        "metrics":
            str(args.output_path / "metrics.json"),
        "dense_points":
            None if args.depth_only else str(args.output_path /
                                             "points3D_dense.ply"),
        "depths":
            str(args.output_path / "depths") if args.export_depth else None,
        "sparse": (str(args.output_path / "sparse" / args.colmap_model_subdir)
                   if args.export_colmap and not args.depth_only else None),
    }


def print_dry_run(args: Any, method: str, num_images: int) -> None:
    print(json.dumps(planned_layout(args, method, num_images), indent=2))


def add_code_path(
    path: Path | None, package_names: tuple[str, ...] = ()) -> None:
    script_dir = str(Path(__file__).resolve().parent)
    sys.path[:] = [
        entry for entry in sys.path
        if str(Path(entry or ".").resolve()) != script_dir
    ]
    for package_name in package_names:
        shadowed = sys.modules.get(package_name)
        if shadowed is not None and not hasattr(shadowed, "__path__"):
            del sys.modules[package_name]
    if path is None:
        return
    code_path = Path(path).expanduser().resolve()
    if str(code_path) not in sys.path:
        sys.path.insert(0, str(code_path))


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().float().numpy()
    return np.asarray(value)


def normalize_depths(depths: Any) -> np.ndarray:
    depth_array = as_numpy(depths).astype(np.float32, copy=False)
    if depth_array.ndim == 5 and depth_array.shape[0] == 1:
        depth_array = depth_array[0]
    # A trailing singleton channel ("N x H x W x 1", e.g. VGGT's depth head) goes first: with a
    # single frame (N == 1, every MonoDepth run) the leading-batch rule below would otherwise
    # mistake N for a batch dim and leave H x W x 1, i.e. H "frames".
    if depth_array.ndim == 4 and depth_array.shape[-1] == 1:
        depth_array = depth_array[..., 0]
    # A leading singleton batch dim ("1 x N x H x W", e.g. Pi3's raw confidence output) is
    # unambiguous whenever shape[1] != 1 too, so it is safe to drop before the N-x-1-x-H-x-W
    # check below (which instead keeps shape[0] as N and drops a spurious shape[1]).
    if depth_array.ndim == 4 and depth_array.shape[
            0] == 1 and depth_array.shape[1] != 1:
        depth_array = depth_array[0]
    if depth_array.ndim == 4 and depth_array.shape[1] == 1:
        depth_array = depth_array[:, 0]
    if depth_array.ndim != 3:
        raise ValueError(
            f"Expected depth array with shape NxHxW, got {depth_array.shape}")
    return depth_array


def normalize_points(points: Any) -> np.ndarray:
    points_array = as_numpy(points).astype(np.float32, copy=False)
    if points_array.ndim == 5 and points_array.shape[0] == 1:
        points_array = points_array[0]
    if points_array.ndim == 4 and points_array.shape[1] == 3:
        points_array = np.transpose(points_array, (0, 2, 3, 1))
    if points_array.ndim != 4 or points_array.shape[-1] != 3:
        raise ValueError(
            f"Expected point map with shape NxHxWx3, got {points_array.shape}")
    return points_array


def normalize_confidence(confidence: Any | None,
                         shape: tuple[int, int, int]) -> np.ndarray:
    if confidence is None:
        return np.ones(shape, dtype=np.float32)
    conf_array = as_numpy(confidence).astype(np.float32, copy=False)
    if conf_array.size == int(np.prod(shape)):
        # Exactly one value per depth pixel: the layout is unambiguous, including N == 1 (e.g. 1xHxWx1, where the
        # leading-1 rule below would drop the frame axis instead of the channel axis).
        return conf_array.reshape(shape)
    if conf_array.ndim == 5 and conf_array.shape[0] == 1:
        conf_array = conf_array[0]
    if conf_array.ndim == 4 and conf_array.shape[0] == 1:
        conf_array = conf_array[0]
    if conf_array.ndim == 4 and conf_array.shape[1] == 1:
        conf_array = conf_array[:, 0]
    if conf_array.ndim == 4 and conf_array.shape[-1] == 1:
        conf_array = conf_array[..., 0]
    if conf_array.ndim != 3:
        raise ValueError(
            f"Expected confidence map with shape NxHxW, got {conf_array.shape}")
    return conf_array


def colors_from_torch_images(images: Any) -> np.ndarray:
    image_array = as_numpy(images)
    if image_array.ndim == 5 and image_array.shape[0] == 1:
        image_array = image_array[0]
    if image_array.ndim == 4 and image_array.shape[1] == 3:
        image_array = np.transpose(image_array, (0, 2, 3, 1))
    if image_array.ndim != 4 or image_array.shape[-1] != 3:
        raise ValueError(
            f"Expected images with shape NxHxWx3, got {image_array.shape}")
    if image_array.dtype != np.uint8:
        image_array = (image_array.clip(0.0, 1.0) * 255.0).astype(np.uint8)
    return image_array


def ensure_uint8_colors(colors: Any) -> np.ndarray:
    color_array = as_numpy(colors)
    if color_array.ndim == 5 and color_array.shape[0] == 1:
        color_array = color_array[0]
    if color_array.ndim == 4 and color_array.shape[1] == 3:
        color_array = np.transpose(color_array, (0, 2, 3, 1))
    if color_array.ndim != 4 or color_array.shape[-1] != 3:
        raise ValueError(
            f"Expected colors with shape NxHxWx3, got {color_array.shape}")
    if color_array.dtype != np.uint8:
        if color_array.max(initial=0.0) <= 1.0:
            color_array = color_array * 255.0
        color_array = color_array.clip(0, 255).astype(np.uint8)
    return color_array


def c2w_to_w2c(camera_to_world: Any) -> np.ndarray:
    c2w = as_numpy(camera_to_world).astype(np.float32, copy=False)
    if c2w.ndim == 4 and c2w.shape[0] == 1:
        c2w = c2w[0]
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(
            f"Expected camera-to-world matrices with shape Nx4x4, got {c2w.shape}"
        )
    w2c = np.repeat(np.eye(4, dtype=np.float32)[None], c2w.shape[0], axis=0)
    rotation = c2w[:, :3, :3]
    translation = c2w[:, :3, 3]
    rotation_inv = np.transpose(rotation, (0, 2, 1))
    w2c[:, :3, :3] = rotation_inv
    w2c[:, :3,
        3] = np.negative(np.einsum("nij,nj->ni", rotation_inv, translation))
    return w2c


def quaternion_xyzw_to_matrix(quaternions: Any) -> np.ndarray:
    q = as_numpy(quaternions).astype(np.float64, copy=False)
    if q.shape[-1] != 4:
        raise ValueError(f"Expected xyzw quaternions, got {q.shape}")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.clip(norm, 1e-12, None)
    x, y, z, w = np.moveaxis(q, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    matrix = np.empty(q.shape[:-1] + (3, 3), dtype=np.float32)
    matrix[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
    matrix[..., 0, 1] = 2.0 * (xy - wz)
    matrix[..., 0, 2] = 2.0 * (xz + wy)
    matrix[..., 1, 0] = 2.0 * (xy + wz)
    matrix[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
    matrix[..., 1, 2] = 2.0 * (yz - wx)
    matrix[..., 2, 0] = 2.0 * (xz - wy)
    matrix[..., 2, 1] = 2.0 * (yz + wx)
    matrix[..., 2, 2] = 1.0 - 2.0 * (xx + yy)
    return matrix


def se3_xyzw_to_c2w(poses: Any) -> np.ndarray:
    pose_array = as_numpy(poses).astype(np.float32, copy=False)
    if pose_array.ndim == 3 and pose_array.shape[0] == 1:
        pose_array = pose_array[0]
    if pose_array.ndim != 2 or pose_array.shape[1] != 7:
        raise ValueError(
            f"Expected SE3 pose vectors with shape Nx7, got {pose_array.shape}")
    c2w = np.repeat(np.eye(4, dtype=np.float32)[None],
                    pose_array.shape[0],
                    axis=0)
    c2w[:, :3, :3] = quaternion_xyzw_to_matrix(pose_array[:, 3:])
    c2w[:, :3, 3] = pose_array[:, :3]
    return c2w


def intrinsics_from_focals(focals: Any, height: int, width: int) -> np.ndarray:
    focal_array = as_numpy(focals).astype(np.float32, copy=False).reshape(-1)
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None],
                           focal_array.shape[0],
                           axis=0)
    intrinsics[:, 0, 0] = focal_array
    intrinsics[:, 1, 1] = focal_array
    intrinsics[:, 0, 2] = (width - 1) / 2.0
    intrinsics[:, 1, 2] = (height - 1) / 2.0
    return intrinsics


def unproject_depths_to_world(
    depths: Any,
    extrinsics_w2c: Any,
    intrinsics: Any,
) -> np.ndarray:
    depth_array = normalize_depths(depths)
    extrinsics = as_numpy(extrinsics_w2c).astype(np.float32, copy=False)
    intrinsics_array = as_numpy(intrinsics).astype(np.float32, copy=False)
    num_frames, height, width = depth_array.shape
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width)).astype(np.float32)
    y = np.broadcast_to(y[None], (num_frames, height, width)).astype(np.float32)

    fx = intrinsics_array[:, 0, 0][:, None, None]
    fy = intrinsics_array[:, 1, 1][:, None, None]
    cx = intrinsics_array[:, 0, 2][:, None, None]
    cy = intrinsics_array[:, 1, 2][:, None, None]
    camera_points = np.stack(
        [
            (x - cx) / np.clip(fx, 1e-12, None) * depth_array,
            (y - cy) / np.clip(fy, 1e-12, None) * depth_array,
            depth_array,
        ],
        axis=-1,
    )

    rotation = extrinsics[:, :3, :3]
    translation = extrinsics[:, :3, 3]
    return np.einsum(
        "nij,nhwj->nhwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )


def export_depth_predictions(
    output_path: Path,
    image_paths: list[str],
    depths: Any,
    confidence: Any | None = None,
) -> None:
    """Export one numeric depth prediction per frame for benchmark evaluation, and its confidence when given.

    The evaluators can read several depth formats, but writing multiple copies of
    the same dense map multiplies storage and can create duplicate frame keys.
    Every map is one float32 compressed ``.npz`` (``save_prediction_map``); generate
    visualization/compatibility files in a separate post-processing step when needed.

    ``confidence`` (the method's own per-pixel confidence, on its own scale) goes to
    ``confidences/<stem>.npz`` as float16, pixel-aligned with the depth of the same name.
    """

    depth_array = normalize_depths(depths)
    depth_dir = output_path / "depths"
    for image_path, depth in zip(image_paths, depth_array):
        save_prediction_map(depth_dir, Path(image_path).stem, depth)
    if confidence is not None:
        export_confidence_predictions(output_path, image_paths, confidence,
                                      depth_array.shape)


# Arrays of the per-frame prediction maps (<dir>/<stem>.npz): depths/, raw_disparities/, raw_log_depths/ hold "depth"
# (the evaluator's loader also takes the first array), confidences/ holds "confidence".
PREDICTION_MAP_KEYS = {"confidences": "confidence"}
FLOAT16_MAX = float(np.finfo(np.float16).max)


def save_prediction_map(directory: Path,
                        stem: str,
                        array: Any,
                        key: str | None = None,
                        dtype: Any = np.float32) -> Path:
    """Write one frame's prediction map as ``<directory>/<stem>.npz``, zip-compressed (``np.savez_compressed``), the
    array as ``dtype`` (float32: depths; float16: confidences, clipped to its range) under ``key`` (default by
    folder: "confidence" in confidences/, else "depth"). An older ``<stem>.npy`` of the same frame is removed, so a
    frame never has two files. Returns the path written."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    key = key or PREDICTION_MAP_KEYS.get(directory.name, "depth")
    values = np.asarray(array, dtype=np.float32)
    if np.dtype(dtype) == np.float16:
        values = np.clip(values, -FLOAT16_MAX, FLOAT16_MAX)  # NaN stays NaN
    path = directory / f"{stem}.npz"
    np.savez_compressed(path, **{key: values.astype(dtype)})
    (directory / f"{stem}.npy").unlink(missing_ok=True)
    return path


def export_confidence_predictions(
    output_path: Path,
    image_paths: list[str],
    confidence: Any,
    shape: tuple[int, int, int],
) -> None:
    """Write ``confidences/<stem>.npz`` (float16, one H x W map per frame, the same pixels as ``depths/``).

    Values are the method's raw confidence (VGGT-style ``1 + exp(x)``, Pi3's sigmoid, ...), clipped to the float16
    range; NaN stays NaN. A map whose shape differs from the depth's is an adapter bug (a crop or resize applied to
    one and not the other), so it raises instead of writing misaligned maps.
    """
    conf_array = normalize_confidence(confidence, shape)
    if conf_array.shape != tuple(shape):
        raise ValueError(
            f"confidence {conf_array.shape} does not match depth {tuple(shape)}"
        )
    for image_path, conf in zip(image_paths, conf_array):
        save_confidence_map(output_path, image_path, conf)


def save_confidence_map(output_path: Path, image_path: str,
                        confidence: np.ndarray) -> None:
    """One frame's ``confidences/<stem>.npz``: float16, clipped to its range (for adapters that crop per frame)."""
    save_prediction_map(Path(output_path) / "confidences",
                        Path(image_path).stem,
                        confidence,
                        dtype=np.float16)


def export_camera_pose_priors(
    output_path: Path,
    images_dir: Path,
    image_paths: list[str],
    extrinsics_w2c: Any,
    intrinsics: Any,
    *,
    width: "int | list[int]",
    height: "int | list[int]",
    colmap_model_subdir: str = "0",
) -> None:
    """Export camera poses/intrinsics as a COLMAP model, without dense points. ``width``/``height`` are one size
    for every image or one per image (see restore_camera_export).

    MVRecon evaluation fuses per-frame depth with these poses itself (see
    ``spatialarena/eval/mvrecon_evaluator.py::prepare_fusion_inputs``), so
    depth-only inference batches do not need to hold a full dense point cloud
    in memory just to satisfy the MVRecon output contract. Poses are tiny
    (one 4x4 + one 3x3 per image) and accumulate cheaply across batches even
    for scenes with thousands of frames.
    """

    from spatialarena.geometry.utils import write_colmap_formats

    extrinsics_array = as_numpy(extrinsics_w2c).astype(np.float32, copy=False)
    intrinsics_array = as_numpy(intrinsics).astype(np.float32, copy=False)
    colmap_dir = output_path / "sparse" / colmap_model_subdir
    colmap_dir.mkdir(parents=True, exist_ok=True)
    write_colmap_formats(
        str(images_dir),
        extrinsics_array,
        intrinsics_array,
        image_paths,
        np.empty((0, 3), dtype=np.float32),
        np.empty((0, 3), dtype=np.uint8),
        width=width,
        height=height,
        output_path=str(colmap_dir),
    )


# Standard runtime metrics, comparable across adapters (see mark_model_loaded / runtime_metrics). The legacy
# "inference time" / "Peak GPU Memory" keys stay as each adapter defined them (their scopes differ per adapter).
_PROCESS_STARTED_AT = time.time()
_MODEL_LOADED_AT: float | None = None


def mark_model_loaded() -> float:
    """Call once the model is on its device and before the first input is processed.

    Starts the standard inference timer and resets the CUDA peak-memory counter, so ``inference_seconds`` and
    ``peak_gpu_memory_mib`` cover inference only (the weights stay allocated, so the peak still includes them) and
    ``model_load_seconds`` covers everything from adapter start to here. Returns the timestamp.
    """
    global _MODEL_LOADED_AT  # pylint: disable=global-statement  # per-process timer read by runtime_metrics
    _MODEL_LOADED_AT = time.time()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)
    return _MODEL_LOADED_AT


def _peak_gpu_memory_mib() -> float | None:
    """Peak device memory allocated by this process's framework (summed over devices), or None if unknown."""
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available(
    ) and torch.cuda.is_initialized():
        return sum(
            torch.cuda.max_memory_allocated(i)
            for i in range(torch.cuda.device_count())) / 1024.0**2
    jax = sys.modules.get("jax")
    if jax is not None:
        try:
            stats = [
                device.memory_stats() or {}
                for device in jax.local_devices()
                if device.platform == "gpu"
            ]
            peaks = [
                s["peak_bytes_in_use"]
                for s in stats
                if "peak_bytes_in_use" in s
            ]
            return sum(peaks) / 1024.0**2 if peaks else None
        except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught  # memory stats are best effort
            return None
    return None


def runtime_metrics(started_at: float) -> dict[str, Any]:
    """Standard runtime metrics at the moment inference finished (call before exporting artifacts).

    ``runtime_scope`` is ``after_model_load`` when the adapter called :func:`mark_model_loaded`; otherwise the timer
    falls back to the adapter's own ``started_at`` and the scope is ``adapter_started_at``.
    """
    now = time.time()
    loaded = _MODEL_LOADED_AT
    return {
        "inference_seconds":
            now - (loaded if loaded is not None else started_at),
        "model_load_seconds":
            loaded - _PROCESS_STARTED_AT if loaded is not None else None,
        "peak_gpu_memory_mib":
            _peak_gpu_memory_mib(),
        "runtime_scope":
            "after_model_load" if loaded is not None else "adapter_started_at",
        "model_input_size":
            _MODEL_INPUT_SIZE,
    }


def write_adapter_metrics(
    *,
    output_path: Path,
    method: str,
    image_paths: list[str],
    started_at: float,
    checkpoint: str | None,
    depth_only: bool,
    metrics_extra: dict[str, Any] | None = None,
    runtime: dict[str, Any] | None = None,
) -> None:
    metrics: dict[str, Any] = {
        "method": method,
        "Preprocessed Images": len(image_paths),
        "inference time": time.time() - started_at,
        "checkpoint": checkpoint,
        "depth_only": depth_only,
    }
    if metrics_extra:
        metrics.update(metrics_extra)
    metrics.update(
        runtime if runtime is not None else runtime_metrics(started_at))
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )


def sample_points(
    points: Any,
    colors: Any,
    confidence: Any | None,
    conf_thresh: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    point_array = normalize_points(points)
    color_array = ensure_uint8_colors(colors)
    conf_array = normalize_confidence(confidence, point_array.shape[:3])
    if color_array.shape[:3] != point_array.shape[:3]:
        raise ValueError(
            f"Point/color spatial shapes do not match: {point_array.shape} vs {color_array.shape}"
        )
    finite_mask = np.isfinite(point_array).all(axis=-1)
    conf_mask = finite_mask & (conf_array >= conf_thresh)
    if max_points > 0:
        true_indices = np.flatnonzero(conf_mask)
        if true_indices.size > max_points:
            sampled_indices = np.random.choice(true_indices,
                                               size=max_points,
                                               replace=False)
            limited_mask = np.zeros(conf_mask.size, dtype=bool)
            limited_mask[sampled_indices] = True
            conf_mask = limited_mask.reshape(conf_mask.shape)
    return point_array[conf_mask], color_array[conf_mask]


def export_standard_artifacts(
    *,
    method: str,
    output_path: Path,
    images_dir: Path,
    image_paths: list[str],
    depths: Any | None,
    points: Any | None,
    colors: Any | None,
    confidence: Any | None,
    extrinsics_w2c: Any | None,
    intrinsics: Any | None,
    export_depth: bool,
    export_colmap: bool,
    depth_only: bool,
    colmap_model_subdir: str,
    max_points: int,
    conf_thresh: float,
    started_at: float,
    checkpoint: str | None = None,
    metrics_extra: dict[str, Any] | None = None,
) -> None:
    runtime = runtime_metrics(
        started_at)  # before exporting, so artifact writing is not timed
    output_path.mkdir(parents=True, exist_ok=True)
    if export_depth:
        if depths is None:
            raise ValueError(
                "--export_depth requested but the adapter did not produce depths"
            )
        export_depth_predictions(output_path, image_paths, depths, confidence)

    sampled_points = None
    sampled_colors = None
    if not depth_only:
        if points is None:
            if depths is None or extrinsics_w2c is None or intrinsics is None:
                raise ValueError(
                    "Dense point export requires point maps or depth + camera predictions"
                )
            points = unproject_depths_to_world(depths, extrinsics_w2c,
                                               intrinsics)
        if colors is None:
            raise ValueError("Dense point export requires colors")
        sampled_points, sampled_colors = sample_points(
            points,
            colors,
            confidence,
            conf_thresh=conf_thresh,
            max_points=max_points,
        )
        from spatialarena.utils.point_utils import export_ply

        export_ply(str(output_path / "points3D_dense.ply"), sampled_points,
                   sampled_colors)

    if export_colmap and not depth_only:
        if extrinsics_w2c is None or intrinsics is None:
            raise ValueError(
                "--export_colmap requested but the adapter did not produce cameras"
            )
        if sampled_points is None or sampled_colors is None:
            raise ValueError(
                "--export_colmap requested but no point cloud was sampled")
        intrinsics_array = as_numpy(intrinsics).astype(np.float32, copy=False)
        if depths is not None:
            depth_array = normalize_depths(depths)
            height, width = depth_array.shape[1:]
        elif points is not None:
            point_array = normalize_points(points)
            height, width = point_array.shape[1:3]
        else:
            raise ValueError("Cannot determine COLMAP image size")
        from spatialarena.geometry.utils import write_colmap_formats

        colmap_dir = output_path / "sparse" / colmap_model_subdir
        colmap_dir.mkdir(parents=True, exist_ok=True)
        write_colmap_formats(
            str(images_dir),
            as_numpy(extrinsics_w2c).astype(np.float32, copy=False),
            intrinsics_array,
            image_paths,
            sampled_points,
            sampled_colors,
            width=width,
            height=height,
            output_path=str(colmap_dir),
        )

    write_adapter_metrics(
        output_path=output_path,
        method=method,
        image_paths=image_paths,
        started_at=started_at,
        checkpoint=checkpoint,
        depth_only=depth_only,
        metrics_extra=metrics_extra,
        runtime=runtime,
    )


# --- Temporary port of FeedForward-Benchmark PR #21 (eval_refactor_hsy,
# commits 21f0a728/d88025f4): shared image spatial-policy transform plus
# depth-space/inverse-depth GT-only alignment protocols. Ported directly
# rather than merging the full PR since it still has other things to
# correct; remove this block once the PR itself lands. ---


@dataclass(frozen=True)
class SpatialTransform:
    """Image-coordinate transform. Sizes are (width, height), margins are L,T,R,B."""

    source_wh: tuple[int, int]
    resized_wh: tuple[int, int]
    padding: tuple[int, int, int, int] = (0, 0, 0, 0)
    crop: tuple[int, int, int, int] = (0, 0, 0, 0)
    policy: str = "resize"

    def __post_init__(self) -> None:
        if min(*self.source_wh, *self.resized_wh) <= 0 or min(
                *self.padding, *self.crop) < 0:
            raise ValueError("Invalid spatial dimensions or margins")
        if self.crop[0] + self.crop[2] >= self.resized_wh[
                0] or self.crop[1] + self.crop[3] >= self.resized_wh[1]:
            raise ValueError("Crop removes entire image")

    @property
    def model_wh(self) -> tuple[int, int]:
        w, h = self.resized_wh
        l, t, r, b = self.padding
        cl, ct, cr, cb = self.crop
        return w - cl - cr + l + r, h - ct - cb + t + b

    def metadata(self) -> dict:
        from dataclasses import asdict

        return dict(asdict(self),
                    model_wh=self.model_wh,
                    output_frame="source_rgb")


def plan_spatial_transform(source_wh: tuple[int, int],
                           *,
                           method: str,
                           size: int = 512,
                           mode: str = "pad") -> SpatialTransform:
    """Explicit policies for Omega, DA3 upper_bound_resize, DVD and the local camera adapter."""
    import math

    w, h = source_wh
    if min(w, h, size) <= 0:
        raise ValueError("Sizes must be positive")
    method = method.lower()
    if method == "dvd":
        scale = max(480 / h, 640 / w)
        resized = tuple((math.ceil(x * scale) + 15) // 16 * 16 for x in (w, h))
        return SpatialTransform(source_wh, resized, policy="dvd_training_scale")
    if method == "dvd_compact":
        if source_wh != (1024, 436):
            raise ValueError("DVD compact policy requires Sintel 1024x436")
        return SpatialTransform(source_wh, (1008, 426), (0, 3, 0, 3),
                                policy="dvd_compact")
    scale = size / max(w, h)
    rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
    if method == "da3":
        # Official upper_bound_resize: nearest 14 multiple, ties round upwards.
        rounded = tuple(max(1, ((x + 7) // 14) * 14) for x in (rw, rh))
        return SpatialTransform(source_wh,
                                rounded,
                                policy="da3_upper_bound_resize")
    if method in ("vggt", "fastvggt"):
        # VGGT/FastVGGT's own pad-to-square (vggt.utils.load_fn.load_and_preprocess_images_square,
        # scripts/inference/fastvggt.py's materialize_resized_images) use patch 14, not 16 -- only
        # "pad" mode is supported here since "resize"/"rect-pad" below are hardcoded to round to 16.
        if mode != "pad":
            raise ValueError(
                f"Unsupported spatial policy: {method}/{mode} (only 'pad' is supported)"
            )
        if size % 14:
            raise ValueError(
                f"{method} square canvas size must be a multiple of 14")
        dw, dh = size - rw, size - rh
        return SpatialTransform(source_wh, (rw, rh),
                                (dw // 2, dh // 2, dw - dw // 2, dh - dh // 2),
                                policy=f"{method}_resize_pad_square")
    if method != "vggt_omega" or mode not in ("pad", "rect-pad", "resize"):
        raise ValueError(f"Unsupported spatial policy: {method}/{mode}")
    if mode == "resize":
        rounded = tuple(max(16, round(x / 16) * 16) for x in (rw, rh))
        return SpatialTransform(source_wh,
                                rounded,
                                policy="omega_resize_patch16")
    if mode == "pad":
        if size % 16:
            raise ValueError(
                "Omega square canvas size must be a multiple of 16")
        dw, dh = size - rw, size - rh
        return SpatialTransform(source_wh, (rw, rh),
                                (dw // 2, dh // 2, dw - dw // 2, dh - dh // 2),
                                policy="omega_resize_pad_square")
    # Explicit rectangular experiment: pad only to the next patch boundary.
    dw, dh = (-rw) % 16, (-rh) % 16
    return SpatialTransform(source_wh, (rw, rh),
                            (dw // 2, dh // 2, dw - dw // 2, dh - dh // 2),
                            policy="omega_resize_pad_patch16")


def apply_spatial_transform_tensor(
        images: "torch.Tensor",
        transform: SpatialTransform,
        *,
        interpolation: str = "bilinear") -> "torch.Tensor":
    """NCHW float RGB -> model grid. No normalization or temporal processing.

    ``interpolation="nearest"`` resizes a map whose zeros mark missing values (a masked or sparse depth prior)
    without blending them into their valid neighbours.
    """
    import torch.nn.functional as functional

    if tuple(images.shape[-2:]) != transform.source_wh[::-1]:
        raise ValueError("RGB does not match recorded source dimensions")
    if transform.resized_wh != transform.source_wh:
        if interpolation == "nearest":
            images = functional.interpolate(images,
                                            size=transform.resized_wh[::-1],
                                            mode="nearest-exact")
        else:
            images = functional.interpolate(images,
                                            size=transform.resized_wh[::-1],
                                            mode="bilinear",
                                            align_corners=False)
    l, t, r, b = transform.crop
    images = images[..., t:images.shape[-2] - b, l:images.shape[-1] - r]
    l, t, r, b = transform.padding
    if any(transform.padding):
        if transform.policy in ("omega_resize_pad_square",
                                "vggt_resize_pad_square",
                                "fastvggt_resize_pad_square"):
            images = functional.pad(images, (l, r, t, b),
                                    mode="constant",
                                    value=0)
        else:
            images = functional.pad(images, (l, r, t, b), mode="replicate")
    return images


def preprocess_images(
    image_paths: list[str],
    *,
    method: str,
    size: int = 512,
    mode: str = "pad",
    exif_transpose: bool = False
) -> tuple["torch.Tensor", list[SpatialTransform]]:
    """Return NCHW RGB [0,1] and per-frame transforms. Reject incompatible batch shapes.

    ``exif_transpose``: match whichever convention the method's own *legacy* loader uses, since
    the resulting SpatialTransform's source_wh must describe the same pixel grid "shared" mode's
    model input and the legacy loader's actual output agree on. VGGT-family loaders
    (load_and_preprocess_images_square, vggt_omega/omnivggt's load_fn.py) never exif_transpose;
    DUST3R-family loaders (CUT3R/FAST3R/AMB3R's load_images, MapAnything's load_images/
    preprocess_inputs) always do -- pass True for those, leave the default for the former.
    """
    import torch

    images, transforms = [], []
    for path in image_paths:
        with Image.open(path) as image:
            if exif_transpose:
                image = ImageOps.exif_transpose(image)
            rgb = np.asarray(image.convert("RGB")).copy()
        transform = plan_spatial_transform((rgb.shape[1], rgb.shape[0]),
                                           method=method,
                                           size=size,
                                           mode=mode)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()[None] / 255
        images.append(apply_spatial_transform_tensor(tensor, transform)[0])
        transforms.append(transform)
    if not images or len({tuple(image.shape) for image in images}) != 1:
        raise ValueError(
            "A batch must have a common model grid; split heterogeneous image shapes"
        )
    return torch.stack(images), transforms


def restore_prediction(prediction: np.ndarray,
                       transform: SpatialTransform) -> np.ndarray:
    """Restore one scalar map to source RGB coordinates; never invent cropped-away borders."""
    import cv2

    array = np.asarray(prediction, dtype=np.float32)
    if array.shape != transform.model_wh[::-1]:
        raise ValueError(
            f"Prediction grid {array.shape} != recorded model grid {transform.model_wh[::-1]}"
        )
    l, t, r, b = transform.padding
    array = array[t:array.shape[0] - b, l:array.shape[1] - r]
    cl, ct, cr, cb = transform.crop
    canvas = np.full(transform.resized_wh[::-1], np.nan, dtype=np.float32)
    canvas[ct:canvas.shape[0] - cb, cl:canvas.shape[1] - cr] = array
    # Conservative validity: any missing interpolation contributor invalidates the output.
    valid = np.isfinite(canvas).astype(np.float32)
    values = cv2.resize(np.nan_to_num(canvas),
                        transform.source_wh,
                        interpolation=cv2.INTER_LINEAR)
    coverage = cv2.resize(valid,
                          transform.source_wh,
                          interpolation=cv2.INTER_LINEAR)
    values[coverage < 1 - 1e-6] = np.nan
    return values


def export_spatial_depth_predictions(
    output_path: Path,
    image_paths: list[str],
    depths: np.ndarray,
    confidence: "np.ndarray | None",
    transforms: list[SpatialTransform],
) -> None:
    """Export original-grid depth/confidence plus per-frame spatial metadata."""
    arrays = normalize_depths(depths)
    if not (len(image_paths) == len(arrays) == len(transforms)):
        raise ValueError("Frame/transform count mismatch")
    conf = normalize_depths(confidence) if confidence is not None else None
    if conf is not None and len(conf) != len(arrays):
        raise ValueError("Confidence frame count mismatch")
    output_path = Path(output_path)
    records_path = output_path / "spatial_transforms.json"
    records = json.loads(
        records_path.read_text()) if records_path.exists() else {}
    for i, (path, depth,
            transform) in enumerate(zip(image_paths, arrays, transforms)):
        restored = restore_prediction(depth, transform)
        restored_conf = restore_prediction(
            conf[i], transform) if conf is not None else None
        export_depth_predictions(
            output_path, [path], restored[None],
            None if restored_conf is None else restored_conf[None])
        records[Path(path).name] = transform.metadata()
    records_path.write_text(json.dumps(records, indent=2))


SHARED_SPATIAL_POLICY_DEPTH_ONLY_MESSAGE = "--spatial_policy shared currently supports --depth_only export only"


def validate_shared_spatial_policy_depth_only(*, is_shared: bool,
                                              depth_only: bool) -> None:
    """Raise if a "shared" spatial_policy value is combined with non-depth-only export.

    Every adapter's "shared" policy restores depth (and, via restore_camera_export, camera intrinsics)
    to each image's true source resolution, but points3D/dense-point export derives pixel-indexed
    geometry from the model's own padded-square grid -- the two would disagree, so dense/points3D
    export isn't supported under "shared" (camera export alongside depth is fine; see
    restore_camera_export). `is_shared` lets callers with more than two --spatial_policy choices (e.g.
    vggt_omega.py's "shared-pad"/"legacy-pad") pass their own "is this a shared-family value" test.
    """
    if is_shared and not depth_only:
        raise ValueError(SHARED_SPATIAL_POLICY_DEPTH_ONLY_MESSAGE)


def spatial_policy_metrics(spatial_policy: str, *,
                           is_shared: bool) -> dict[str, str]:
    """The `metrics.json` fields recording which --spatial_policy an adapter run used."""
    return {
        "depth_export":
            "restored_to_source_resolution"
            if is_shared else "cropped_to_original_image_region",
        "spatial_policy":
            spatial_policy,
    }


def restore_intrinsics(intrinsics: Any,
                       transform: "SpatialTransform") -> np.ndarray:
    """Map a (3,3) or (N,3,3) pinhole intrinsics matrix from the model's grid (after
    apply_spatial_transform_tensor's resize, crop, then pad) back to pixel coordinates of
    transform.source_wh -- the geometric inverse of that same image transform.

    fx/fy scale by the resize factor; cx/cy additionally shift by padding (removed) and crop
    (added back) before that same scale is undone, in the same coordinate frame restore_prediction
    maps scalar maps (depth/confidence) back to.
    """
    k = np.array(intrinsics, dtype=np.float64, copy=True)
    single = k.ndim == 2
    if single:
        k = k[None]
    source_w, source_h = transform.source_wh
    resized_w, resized_h = transform.resized_wh
    scale_x, scale_y = resized_w / source_w, resized_h / source_h
    pad_l, pad_t, _pad_r, _pad_b = transform.padding
    crop_l, crop_t, _crop_r, _crop_b = transform.crop
    k[:, 0, 0] /= scale_x
    k[:, 1, 1] /= scale_y
    k[:, 0, 2] = (k[:, 0, 2] - pad_l + crop_l) / scale_x
    k[:, 1, 2] = (k[:, 1, 2] - pad_t + crop_t) / scale_y
    return k[0] if single else k


def transform_intrinsics(intrinsics: Any,
                         transform: "SpatialTransform") -> np.ndarray:
    """Map a (3,3) or (N,3,3) pinhole intrinsics matrix from pixel coordinates of transform.source_wh onto the
    model's grid (apply_spatial_transform_tensor's resize, crop, then pad): the inverse of restore_intrinsics,
    for camera priors given to a model as input."""
    k = np.array(intrinsics, dtype=np.float64, copy=True)
    single = k.ndim == 2
    if single:
        k = k[None]
    source_w, source_h = transform.source_wh
    resized_w, resized_h = transform.resized_wh
    scale_x, scale_y = resized_w / source_w, resized_h / source_h
    pad_l, pad_t, _pad_r, _pad_b = transform.padding
    crop_l, crop_t, _crop_r, _crop_b = transform.crop
    k[:, 0, 0] *= scale_x
    k[:, 1, 1] *= scale_y
    k[:, 0, 2] = k[:, 0, 2] * scale_x - crop_l + pad_l
    k[:, 1, 2] = k[:, 1, 2] * scale_y - crop_t + pad_t
    return k[0] if single else k


def restore_camera_export(
    intrinsics: Any, transforms: list["SpatialTransform"]
) -> "tuple[np.ndarray, int | list[int], int | list[int]]":
    """Adapter export boundary for camera/COLMAP output alongside shared-policy depth.

    Restores each image's intrinsics (restore_intrinsics) to its own transform.source_wh, and
    returns the (width, height) to pass to write_colmap_formats/export_camera_pose_priors: one
    common size when every transform agrees on source_wh, else one per image (e.g. ETH3D, whose
    undistorted frames differ by a few pixels), which those writers record per COLMAP camera.
    """
    source_whs = [t.source_wh for t in transforms]
    if len(set(source_whs)) == 1:
        width, height = source_whs[0]
    else:
        width, height = [wh[0] for wh in source_whs
                        ], [wh[1] for wh in source_whs]
    restored = np.stack([
        restore_intrinsics(k, t)
        for k, t in zip(np.asarray(intrinsics), transforms)
    ]).astype(np.float32)
    return restored, width, height


def legacy_or_shared_tensor(
    image_paths: list[str],
    *,
    method: str,
    args: Any,
    device: str,
    legacy_loader: "Callable[[list[str]], tuple[torch.Tensor, Any]]",
) -> tuple["torch.Tensor", Any, "list[SpatialTransform] | None"]:
    """Shared building block for a preprocess_<method>() whose method exposes --spatial_policy
    (default "legacy", unchanged behavior; "shared" opts into restoration): resolve it once here
    so the low-level run_<method>() model-call function takes a plain images tensor and stays
    unaware of spatial_policy entirely. Not called directly by adapter run()/infer() code -- see
    prepare_model_input(), the one call every adapter actually uses.

    "shared": a single pad-to-square resize straight to args.resolution
    (plan_spatial_transform(method=...) via preprocess_images); returns (images, None, transforms).
    Anything else ("legacy" default, or an adapter-specific value like vggt_omega.py's
    "legacy-pad"): defers to the method's own historical resize, `legacy_loader`, called with
    `image_paths`; returns (images, legacy_loader's second return value, None) -- e.g. VGGT's
    `original_coords`, for its own crop_to_original_image_region-based legacy export.
    """
    if args.spatial_policy == "shared":
        images, transforms = preprocess_images(image_paths,
                                               method=method,
                                               size=args.resolution,
                                               mode="pad")
        return images.to(device), None, transforms
    images, legacy_extra = legacy_loader(image_paths)
    return images.to(device), legacy_extra, None


# --- prepare_model_input: the one preprocessing call every adapter's run()/infer() uses ---
#
# Historically each adapter called its own differently-named helper directly -- build_shared_views
# (cut3r.py, fast3r.py), build_pure_resize_transforms (pi3.py), build_dejaview_batch (dejaview.py),
# materialize_resized_images (iggt4d.py, come.py), or a bare preprocess_images() call duplicated
# inline (fastvggt.py, da3.py, vggt_omega.py) -- so "how does this method preprocess its images"
# meant finding and reading a different function in every file. prepare_model_input() is the one
# name every adapter calls instead; it dispatches by METHOD (a spatialarena.methods.MethodSpec
# name, e.g. "CUT3R") to that method's own preprocess_<method_lower>() function, registered below
# each one's definition via register_preprocessor(). The implementation still varies -- a tensor,
# a view-dict list, a batch dict, or a list of file paths the model reads from disk, depending on
# what that method's own vendored code needs -- but the call site and the registered function's
# name are both uniform, so the per-method detail is one `grep "^def preprocess_"` away instead of
# a different name to look up per file.
_PREPROCESSOR_REGISTRY: "dict[str, Callable[..., tuple[Any, list[SpatialTransform] | None]]]" = {}


def register_preprocessor(
        method: str,
        fn: "Callable[..., tuple[Any, list[SpatialTransform] | None]]") -> None:
    """Register `fn` (that method's own preprocess_<method_lower>()) as the implementation
    prepare_model_input(method, ...) dispatches to. Call once, right after defining it, in that
    method's own adapter module -- e.g. cut3r.py: `register_preprocessor("CUT3R", preprocess_cut3r)`.
    """
    _PREPROCESSOR_REGISTRY[method.upper()] = fn


def prepare_model_input(
        method: str,
        image_paths: list[str],
        *,
        args: Any,
        device: "str | None" = None,
        **extra: Any) -> tuple[Any, "list[SpatialTransform] | None"]:
    """The one preprocessing entry point every adapter's run()/infer() calls. Looks up `method`'s
    own preprocess_<method_lower>() (registered via register_preprocessor, normally at import
    time in that method's adapter module) and delegates to it with (image_paths, args, device).

    Returns (model_input, transforms): model_input is whatever that model's own code needs (a
    tensor, a view-dict list, a batch dict, a list of file paths written to disk for a subprocess
    to read) -- never normalized across methods, since the real vendored model call sites differ.
    transforms is the per-image SpatialTransform list for restore_prediction/restore_camera_export,
    or None when the method has no restoration path (no --spatial_policy support at all, or
    "legacy" was selected for one that does).

    `**extra`: forwarded verbatim to that method's preprocess_<method_lower>() as keyword
    arguments, for the rare per-call extra a specific method's own implementation needs (e.g. a
    batch index to keep each depth_only batch's temp directory name distinct) -- not part of the
    common (image_paths, args, device) contract every method shares.
    """
    try:
        preprocess = _PREPROCESSOR_REGISTRY[method.upper()]
    except KeyError as exc:
        raise ValueError(
            f"No preprocessor registered for method {method!r}. That adapter module must call "
            "register_preprocessor(method, preprocess_<method_lower>) before prepare_model_input "
            "can be used for it.") from exc
    return preprocess(image_paths, args=args, device=device, **extra)


def export_depths_for_policy(
    output_path: Path,
    image_paths: list[str],
    depths: Any,
    confidence: "Any | None",
    transforms: "list[SpatialTransform] | None",
    *,
    legacy_export: "Callable[[], None]",
) -> None:
    """Adapter export boundary mirroring prepare_model_input/legacy_or_shared_tensor: restore depth/confidence to
    source resolution when transforms is not None (shared policy); otherwise call `legacy_export`
    (the method's own historical export, e.g. VGGT's crop_to_original_image_region-based export).
    """
    if transforms is not None:
        export_spatial_depth_predictions(output_path, image_paths, depths,
                                         confidence, transforms)
    else:
        legacy_export()
