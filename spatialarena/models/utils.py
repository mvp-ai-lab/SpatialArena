# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Shared pre- and post-processing helpers of the inference models.

The helpers cover:

    frames           image discovery, interval sampling, frame caps, the
                     engine's image subsets and single-view batching;
    priors           COLMAP camera priors and depth priors of a scene;
    spatial policy   the resize, crop and pad from a source image to the
                     model grid (`SpatialTransform`), and restoring depth,
                     confidence and intrinsics back to the source image;
    export           prediction maps (.npz), COLMAP models, dense points;
    runtime          the standard timers and metrics.json.

Models (`spatialarena.models.base.BaseModel` subclasses and the scripts in
scripts/inference) may run external method code; the benchmark evaluators
only read the files written here. scripts/inference/adapter_utils.py is an
alias of this module for scripts that import it by that name.
"""

from __future__ import annotations

import bisect
import dataclasses
import glob
import json
import os
import pathlib
import random
import shutil
import struct
import sys
import time
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from PIL import Image
from PIL import ImageOps

if TYPE_CHECKING:
    import torch

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# pylint: disable-next=wrong-import-position
from spatialarena.eval import colmap_view_resolver  # noqa: E402

ETH3D_RAW_DEPTH_SHAPE = (4032, 6048)
ETH3D_RAW_DEPTH_BYTES = ETH3D_RAW_DEPTH_SHAPE[0] * ETH3D_RAW_DEPTH_SHAPE[1] * 4
SINTEL_DPT_TAG_FLOAT = 202021.25

# ----------------------------------------------------------------------------
# Frames.
# ----------------------------------------------------------------------------


def has_colmap_model_files(path: pathlib.Path) -> bool:
    """Returns whether a directory holds COLMAP cameras, images and points."""
    path = pathlib.Path(path)
    if not path.is_dir():
        return False
    for stem in ("cameras", "images", "points3D"):
        if not any((path / f"{stem}{extension}").is_file()
                   for extension in (".bin", ".txt")):
            return False
    return True


def colmap_prior_candidates(
        data_path: pathlib.Path,
        colmap_model_subdir: str = "0") -> list[pathlib.Path]:
    """Returns the camera-prior directories of a scene, preferred first."""
    data_path = pathlib.Path(data_path)
    return [
        data_path / "sparse" / colmap_model_subdir,
        data_path / "sparse",
        data_path / "dslr_calibration_undistorted",
        data_path / "dslr_calibration_jpg",
    ]


def resolve_colmap_prior_path(
        data_path: pathlib.Path,
        *,
        colmap_model_subdir: str = "0") -> pathlib.Path | None:
    """Returns the first camera-prior directory with a COLMAP model, or None."""
    for candidate in colmap_prior_candidates(data_path, colmap_model_subdir):
        if has_colmap_model_files(candidate):
            return candidate
    return None


def find_image_paths(images_dir: pathlib.Path) -> list[str]:
    """Returns every image under `images_dir`, recursively, sorted."""
    image_paths = []
    for pattern in colmap_view_resolver.IMAGE_PATTERNS:
        image_paths.extend(
            glob.glob(str(images_dir / "**" / pattern), recursive=True))
    return sorted(image_paths)


def canonical_image_root(data_path: pathlib.Path) -> pathlib.Path:
    """Returns the image directory the benchmark uses for a scene."""
    return colmap_view_resolver.canonical_image_root(data_path)


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
    """Returns the images in a named order (BENCHMARK_IMAGE_ORDER).

    Args:
        image_paths: Images in sorted order.
        order_name: One of ORDER_NAMES, or None for sorted order.

    Returns:
        The images, shuffled with the order's fixed seed for "perm*".

    Raises:
        ValueError: The order is unknown.
    """
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
    """Returns the indices `select_interval` keeps, for per-image arrays.

    Callers with other per-image arrays (e.g. DA3's camera priors) need the
    same subsample applied in lockstep; see `select_interval` for the steps.

    Args:
        num_images: Number of images found.
        interval: Keep every interval-th image.
        max_images_per_run: Optional further cap (0: none).

    Returns:
        The kept indices, in order.
    """
    indices = list(range(num_images))
    if interval > 1:
        indices = indices[::interval]
    if single_view_mode():
        # MonoDepth scores every frame: no total cap. The engine splits the
        # scene into calls of at most max_images_per_run frames instead
        # (scripts/inference/single_view_groups.py).
        return indices

    max_input_frames = int(
        os.environ.get("BENCHMARK_MAX_INPUT_FRAMES", "0") or 0)
    if max_input_frames > 0 and len(indices) > max_input_frames:
        seed = int(
            os.environ.get("BENCHMARK_MAX_INPUT_FRAMES_SEED", "42") or 42)
        shuffled = list(indices)
        random.Random(seed).shuffle(shuffled)
        indices = sorted(shuffled[:max_input_frames])

    if max_images_per_run > 0 and len(indices) > max_images_per_run:
        positions = np.linspace(0, len(indices) - 1, max_images_per_run)
        positions = positions.round().astype(int)
        indices = [
            indices[position] for position in sorted(set(positions.tolist()))
        ]
    return indices


def select_interval(image_paths: list[str],
                    interval: int,
                    max_images_per_run: int = 0) -> list[str]:
    """Returns the frames a run processes, out of every image found.

    Three steps, always in this order and regardless of --depth_only (each
    depth-only batch is a separate forward pass with its own coordinate
    frame, so the total cap comes first to keep a scene in one batch):

    1. `interval`: keep every interval-th frame.
    2. BENCHMARK_MAX_INPUT_FRAMES (set by the engine per suite): the
       DA3-BENCH total-frame cap (eval_bench.yaml's max_frames: 100), a
       fixed-seed random subset as in DA3's evaluator.py::_sample_frames.
    3. `max_images_per_run`: an optional method-specific cap (e.g. IGGT's
       12), an evenly spaced subset of what is left.

    In single-view mode (MonoDepth) steps 2 and 3 are skipped: every frame
    is predicted, and the engine bounds each call through
    BENCHMARK_IMAGE_SUBSET lists. BENCHMARK_IMAGE_ORDER and
    BENCHMARK_IMAGE_SUBSET are applied last.

    Args:
        image_paths: Every image of the scene, sorted.
        interval: Keep every interval-th image.
        max_images_per_run: Optional further cap (0: none).

    Returns:
        The selected image paths.
    """
    indices = resolve_capped_image_indices(len(image_paths), interval,
                                           max_images_per_run)
    image_paths = [image_paths[index] for index in indices]
    order_name = os.environ.get("BENCHMARK_IMAGE_ORDER", "").strip() or None
    return restrict_to_image_subset(apply_image_order(image_paths, order_name))


# Set by the engine to one call's image list (single_view_groups.py) when a
# single-view scene mixes image sizes or has more than max_images_per_run
# frames; the lists together make up the scene.
IMAGE_SUBSET_ENV = "BENCHMARK_IMAGE_SUBSET"


def restrict_to_image_subset(image_paths: list[str]) -> list[str]:
    """Keeps the images listed in the BENCHMARK_IMAGE_SUBSET file.

    Args:
        image_paths: Candidate images.

    Returns:
        The listed images (all of them when the variable is unset).

    Raises:
        ValueError: A listed image is not among `image_paths`.
    """
    subset_file = os.environ.get(IMAGE_SUBSET_ENV, "").strip()
    if not subset_file:
        return image_paths
    lines = pathlib.Path(subset_file).read_text(encoding="utf-8").split("\n")
    subset = set(lines) - {""}
    kept = [path for path in image_paths if path in subset]
    if len(kept) != len(subset):
        raise ValueError(f"{len(subset) - len(kept)} image(s) of {subset_file} "
                         "are not among the selected frames")
    return kept


def _image_lookup_keys(
        image_path: str | pathlib.Path,
        images_dir: pathlib.Path | None = None) -> tuple[str, ...]:
    """Returns the names an image can be looked up by, most specific first."""
    path = pathlib.Path(image_path)
    keys = [str(image_path).replace("\\", "/"), path.name, path.stem]
    if images_dir is not None:
        try:
            relative = path.relative_to(images_dir)
        except ValueError:
            pass
        else:
            keys.extend([relative.as_posix(), str(relative), relative.stem])
    return tuple(dict.fromkeys(keys))


def _normalized_relative_image_name(value: str | pathlib.Path) -> str:
    """Returns a relative image name with a normalized parent directory."""
    parts = str(value).replace("\\", "/").lstrip("./").split("/")
    if len(parts) > 1:
        directory = parts[-2].lower()
        parts[-2] = directory.removesuffix("_undistorted").removesuffix("_jpg")
    return "/".join(parts)


def _colmap_name_keys(image_name: str) -> tuple[str, ...]:
    """Returns the normalized lookup keys of a COLMAP image name.

    COLMAP names may carry dataset-specific directory prefixes the benchmark
    image path lacks. Exact relative names come first, then the basename and
    stem, used only when they identify one record.
    """
    path = pathlib.Path(str(image_name).replace("\\", "/"))
    keys = (
        path.as_posix(),
        _normalized_relative_image_name(path),
        path.name,
        path.stem,
    )
    return tuple(dict.fromkeys(keys))


# ----------------------------------------------------------------------------
# Camera priors.
# ----------------------------------------------------------------------------


def match_colmap_images_to_real_paths(data_path: pathlib.Path,
                                      image_paths: list[str], *,
                                      colmap_path: pathlib.Path) -> list[str]:
    """Returns the real image paths a COLMAP model has records for.

    COLMAP is authoritative for camera-conditioned inference. As in
    DepthAnything3's ColmapManager, a partial model (e.g. ScanNet++) means
    only its reconstructed views are evaluated.
    """
    mappings, _ = colmap_view_resolver.resolve_colmap_image_mappings(
        data_path, image_paths, colmap_path=colmap_path, strict=True)
    return [str(mapping.image_path) for mapping in mappings]


@dataclasses.dataclass(frozen=True)
class CameraConditionedInputs:
    """The images of a camera-conditioned run and their priors.

    Attributes:
        image_paths: The images, in prior order.
        poses: Camera-to-world matrices, (N, 4, 4).
        intrinsics: Pinhole intrinsics, (N, 3, 3).
    """

    image_paths: list[str]
    poses: np.ndarray
    intrinsics: np.ndarray


def resolve_camera_conditioned_inputs(
        data_path: pathlib.Path,
        *,
        colmap_path: pathlib.Path | None,
        colmap_model_subdir: str = "0") -> CameraConditionedInputs:
    """Returns the images and camera priors of a scene in one ordering."""
    image_paths = find_image_paths(canonical_image_root(data_path))
    image_paths = resolve_camera_conditioned_image_paths(
        data_path, image_paths, colmap_path=colmap_path)
    poses, intrinsics = load_colmap_camera_priors(
        data_path,
        image_paths,
        colmap_path=colmap_path,
        colmap_model_subdir=colmap_model_subdir,
    )
    return CameraConditionedInputs(image_paths, poses, intrinsics)


def resolve_camera_conditioned_image_paths(
        data_path: pathlib.Path, image_paths: list[str], *,
        colmap_path: pathlib.Path | None) -> list[str]:
    """Returns the images the camera-prior COLMAP model has records for.

    Raises:
        FileNotFoundError: No COLMAP model was given.
    """
    if colmap_path is None:
        raise FileNotFoundError("rgb_cameras requires --colmap_path")
    return match_colmap_images_to_real_paths(data_path,
                                             image_paths,
                                             colmap_path=colmap_path)


def _colmap_camera_to_intrinsics(camera: Any) -> np.ndarray:
    """Returns the 3x3 pinhole matrix of a COLMAP camera."""
    params = np.asarray(camera.params, dtype=np.float32).reshape(-1)
    intrinsics = np.eye(3, dtype=np.float32)
    if camera.model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
        fx, fy, cx, cy = params[:4]
    elif camera.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        fx, fy, cx, cy = params[0], params[0], params[1], params[2]
    else:
        raise ValueError(
            f"Unsupported COLMAP camera model for priors: {camera.model}")
    intrinsics[0, 0], intrinsics[1, 1] = fx, fy
    intrinsics[0, 2], intrinsics[1, 2] = cx, cy
    return intrinsics


def _resolve_prior_dir(data_path, colmap_path, colmap_model_subdir):
    """Returns the COLMAP prior directory to read.

    Raises:
        FileNotFoundError: The given or found directory has no model.
    """
    if colmap_path is not None:
        candidate = pathlib.Path(colmap_path)
        if not has_colmap_model_files(candidate):
            raise FileNotFoundError(
                f"Explicit COLMAP prior directory is invalid: {candidate}")
        return candidate
    sparse_dir = resolve_colmap_prior_path(
        data_path, colmap_model_subdir=colmap_model_subdir)
    if sparse_dir is None:
        tried = colmap_prior_candidates(data_path, colmap_model_subdir)
        raise FileNotFoundError("COLMAP prior directory not found. Tried: " +
                                ", ".join(str(path) for path in tried))
    return sparse_dir


def load_colmap_camera_priors(
        data_path: pathlib.Path,
        image_paths: list[str],
        *,
        colmap_path: pathlib.Path | None = None,
        colmap_model_subdir: str = "0") -> tuple[np.ndarray, np.ndarray]:
    """Loads the camera priors of the images from the scene's COLMAP model.

    Args:
        data_path: Scene root.
        image_paths: Images to load priors for.
        colmap_path: Explicit COLMAP model directory, or None to search the
            scene's usual places.
        colmap_model_subdir: Model subdirectory under sparse/.

    Returns:
        A (camera-to-world (N, 4, 4), intrinsics (N, 3, 3)) tuple.

    Raises:
        FileNotFoundError: No model was found, or an image has no record.
    """
    sparse_dir = _resolve_prior_dir(data_path, colmap_path, colmap_model_subdir)
    # Not strict: a caller may ask for one batch of a larger model, so model
    # records outside the request are fine; every requested image must
    # still resolve, checked below.
    mappings, _ = colmap_view_resolver.resolve_colmap_image_mappings(
        data_path, image_paths, colmap_path=sparse_dir, strict=False)
    by_path = {str(mapping.image_path): mapping.record for mapping in mappings}
    missing = [str(path) for path in image_paths if str(path) not in by_path]
    if missing:
        preview = ", ".join(pathlib.Path(path).name for path in missing[:5])
        raise FileNotFoundError(f"Missing COLMAP camera priors for "
                                f"{len(missing)} selected images: {preview}")

    records = [by_path[str(image_path)] for image_path in image_paths]
    c2w = np.stack([record.c2w for record in records], axis=0)
    intrinsics = np.stack([record.intrinsic for record in records], axis=0)
    return c2w, intrinsics


def build_default_camera_priors(
        image_paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Returns identity poses and centered pinhole intrinsics per image.

    Some depth-conditioned methods need intrinsics even when a scene has no
    calibrated cameras. These are a deterministic fallback for those; they
    must not be used for camera-conditioned tracks.

    Args:
        image_paths: The images.

    Returns:
        A (camera-to-world (N, 4, 4), intrinsics (N, 3, 3)) tuple.
    """
    c2w_list = []
    intrinsics_list = []
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


# ----------------------------------------------------------------------------
# Depth priors.
# ----------------------------------------------------------------------------


def depth_prior_candidates(data_path: pathlib.Path) -> list[pathlib.Path]:
    """Returns the directories that may hold a scene's depth priors.

    Most scenes keep them in depths/ next to images/. Sintel keeps RGB in
    training/final/<scene> (or clean) and depth in the sibling
    training/depth/<scene>, which must be found before deciding a scene has
    no priors.
    """
    data_path = pathlib.Path(data_path)
    candidates = [data_path / "depths", data_path / "depth"]
    if data_path.name in {"images", "rgb", "color"}:
        scene_root = data_path.parent
        candidates.extend([scene_root / "depths", scene_root / "depth"])
    if data_path.parent.name in {"final", "clean"}:
        candidates.append(data_path.parent.parent / "depth" / data_path.name)

    unique = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve() if candidate.exists() else candidate
        if resolved not in seen:
            unique.append(candidate)
            seen.add(resolved)
    return unique


def read_sintel_dpt(path: pathlib.Path) -> np.ndarray:
    """Reads a Sintel .dpt depth map as float32 meters.

    Raises:
        ValueError: The file is not a valid .dpt file.
    """
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
        raise ValueError(f"Truncated Sintel .dpt file {path}: expected "
                         f"{width * height} values")
    return depth.reshape(height, width)


def read_pfm_depth(path: pathlib.Path) -> np.ndarray:
    """Reads a PFM depth map as float32.

    RobustMVD-style samples keep key-view metric depth as .pfm, so the
    depth priors support the same format as the evaluator.

    Raises:
        ValueError: The file has no PFM header.
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


# (timestamp, path) pairs of depth files whose stems are timestamps.
TimedPaths = list[tuple[float, pathlib.Path]]


def parse_numeric_stem(path: pathlib.Path) -> float | None:
    """Returns a timestamp-like file stem as a number, or None."""
    try:
        return float(path.stem)
    except ValueError:
        return None


def nearest_numeric_depth_path(
        image_path: pathlib.Path,
        numeric_depth_files: TimedPaths,
        *,
        max_delta_seconds: float = 0.12) -> pathlib.Path | None:
    """Returns the depth file nearest in time to an RGB frame.

    Args:
        image_path: Frame whose stem is a timestamp.
        numeric_depth_files: (timestamp, path) pairs sorted by timestamp.
        max_delta_seconds: Largest accepted time difference.

    Returns:
        The nearest depth path, or None when none is close enough.
    """
    timestamp = parse_numeric_stem(image_path)
    if timestamp is None or not numeric_depth_files:
        return None
    timestamps = [item[0] for item in numeric_depth_files]
    insert_at = bisect.bisect_left(timestamps, timestamp)
    candidates = []
    if insert_at < len(numeric_depth_files):
        candidates.append(numeric_depth_files[insert_at])
    if insert_at > 0:
        candidates.append(numeric_depth_files[insert_at - 1])
    if not candidates:
        return None
    nearest_timestamp, nearest_path = min(
        candidates, key=lambda item: abs(item[0] - timestamp))
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


def _build_depth_prior_index(data_path):
    """Indexes a scene's depth files by name and stem, and by timestamp.

    Returns:
        A ({name or stem: path}, [(timestamp, path)] sorted) tuple.
    """
    depth_dirs = [
        candidate for candidate in depth_prior_candidates(data_path)
        if candidate.is_dir()
    ]
    if not depth_dirs:
        return {}, []
    depth_files = {}
    numeric_depth_files = []
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


def _match_depth_prior_path(image_path, depth_files, numeric_depth_files):
    """Returns the depth file of an image, by name, then by timestamp."""
    path = pathlib.Path(image_path)
    name_alternatives = [
        path.stem,
        path.name,
        path.stem.replace(".color", ".depth"),
        path.stem.replace(".color", ".depth.proj"),
        path.name.replace(".color.", ".depth."),
        path.name.replace(".color.", ".depth.proj."),
    ]
    for key in name_alternatives:
        depth_path = depth_files.get(key)
        if depth_path is not None:
            return depth_path
    return nearest_numeric_depth_path(path, numeric_depth_files)


def resolve_optional_depth_prior_paths(
        data_path: pathlib.Path,
        image_paths: list[str]) -> tuple[list[pathlib.Path | None], list[str]]:
    """Returns each image's depth file, keeping the missing ones' places.

    Returns:
        A ([path or None per image], [images without a prior]) tuple.
    """
    depth_files, numeric_depth_files = _build_depth_prior_index(data_path)
    if not depth_files and not numeric_depth_files:
        return [None for _ in image_paths], [str(p) for p in image_paths]

    depth_paths = []
    missing = []
    for image_path in image_paths:
        depth_path = _match_depth_prior_path(image_path, depth_files,
                                             numeric_depth_files)
        if depth_path is None:
            missing.append(str(image_path))
        depth_paths.append(depth_path)
    return depth_paths, missing


def resolve_depth_prior_paths(
        data_path: pathlib.Path,
        image_paths: list[str]) -> tuple[list[pathlib.Path], list[str]]:
    """Returns the depth files found, and the images without one."""
    optional_paths, missing = resolve_optional_depth_prior_paths(
        data_path, image_paths)
    depth_paths = [path for path in optional_paths if path is not None]
    return depth_paths, missing


def dataset_depth_png_scale(data_path: pathlib.Path) -> float | None:
    """Returns the integer depth-PNG units per meter of a scene's dataset.

    The dataset declares it as "depth_png_scale" in metadata.json (e.g. 5000
    for TUM RGB-D, 1000 for millimeter PNGs); None when it declares none.
    """
    data_path = pathlib.Path(data_path)
    for metadata_path in (data_path / "metadata.json",
                          data_path.parent / "metadata.json"):
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and payload.get("depth_png_scale"):
            return float(payload["depth_png_scale"])
    return None


def _splat_sparse_depth(payload, image_path):
    """Draws a {"depth", "coord"} point payload as a map of the image size.

    Pixels hit by several points take their mean; pixels without a point
    are 0.
    """
    with Image.open(image_path) as image:
        width, height = image.size
    sparse_depth = np.asarray(payload["depth"], dtype=np.float32).reshape(-1)
    sparse_coord = np.asarray(payload["coord"], dtype=np.float32)
    sparse_coord = sparse_coord.reshape(-1, 2)
    x = np.rint(sparse_coord[:, 0]).astype(np.int64)
    y = np.rint(sparse_coord[:, 1]).astype(np.int64)
    valid = (np.isfinite(sparse_depth) & np.isfinite(sparse_coord).all(axis=1) &
             (x >= 0) & (x < width) & (y >= 0) & (y < height))
    depth_sum = np.zeros((height, width), dtype=np.float32)
    depth_count = np.zeros((height, width), dtype=np.float32)
    np.add.at(depth_sum, (y[valid], x[valid]), sparse_depth[valid])
    np.add.at(depth_count, (y[valid], x[valid]), 1.0)
    depth = np.zeros((height, width), dtype=np.float32)
    filled = depth_count > 0
    depth[filled] = depth_sum[filled] / depth_count[filled]
    return depth


def _load_torch_depth(depth_path, image_path):
    """Loads a .pt/.pth depth prior: a map, or sparse points to splat."""
    import torch  # pylint: disable=import-outside-toplevel

    payload = torch.load(depth_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "depth" in payload and "coord" in payload:
        return _splat_sparse_depth(payload, image_path)
    if isinstance(payload, dict) and "depth" in payload:
        return np.asarray(payload["depth"], dtype=np.float32)
    return np.asarray(payload, dtype=np.float32)


def load_depth_prior_file(depth_path: pathlib.Path,
                          image_path: str | pathlib.Path,
                          png_scale: float | None = None) -> np.ndarray:
    """Loads one depth prior as a finite float32 (H, W) map in meters.

    Args:
        depth_path: The depth file.
        image_path: Its image (for the size of sparse .pt priors).
        png_scale: Units per meter of integer image formats (PNG/TIFF), from
            `dataset_depth_png_scale`. Without it, image depths whose maximum
            exceeds 100 are taken as millimeters: right for 7-Scenes/DTU
            PNGs, wrong for TUM's 5000-per-meter PNGs, which is why datasets
            declare their scale.

    Returns:
        The depth map; missing depth is 0.
    """
    suffix = depth_path.suffix.lower()
    if suffix == ".npy":
        depth = np.load(depth_path).astype(np.float32)
    elif suffix == ".npz":
        with np.load(depth_path) as data:
            key = "depth" if "depth" in data else data.files[0]
            depth = data[key].astype(np.float32)
    elif suffix == ".dpt":
        depth = read_sintel_dpt(depth_path)
    elif suffix == ".pfm":
        depth = read_pfm_depth(depth_path)
    elif (suffix in {".jpg", ".jpeg"} and
          depth_path.stat().st_size == ETH3D_RAW_DEPTH_BYTES):
        depth = np.fromfile(depth_path, dtype=np.float32)
        depth = depth.reshape(ETH3D_RAW_DEPTH_SHAPE)
    elif suffix in {".pt", ".pth"}:
        depth = _load_torch_depth(depth_path, pathlib.Path(image_path))
    else:
        depth = np.asarray(Image.open(depth_path), dtype=np.float32)
        # 16-bit depth images mark missing depth near the type's maximum:
        # 65535 in raw 7-Scenes frames and 65000-65534 in its registered
        # *.depth.proj.png, with nothing valid between 5 m and 65 m. 65000 is
        # beyond any valid depth at the 16-bit scales used here (65 m in mm,
        # 13 m at TUM's 5000/m, 254 m at KITTI's 256/m).
        depth[depth >= 65000.0] = 0.0
        if png_scale is not None:
            if depth.ndim == 3:
                depth = depth[..., 0]
            depth[~np.isfinite(depth)] = 0.0
            return (depth / png_scale).astype(np.float32, copy=False)

    if depth.ndim == 3:
        depth = depth[..., 0]
    depth[~np.isfinite(depth)] = 0.0
    if suffix not in {".npy", ".npz"} and depth.max(initial=0.0) > 100.0:
        depth = depth / 1000.0
    return depth.astype(np.float32, copy=False)


def load_depth_priors(data_path: pathlib.Path,
                      image_paths: list[str]) -> list[np.ndarray]:
    """Loads the depth prior of every image.

    Raises:
        FileNotFoundError: The scene has no priors, or an image has none.
    """
    depth_paths, missing = resolve_depth_prior_paths(data_path, image_paths)
    if not depth_paths and missing:
        candidates = ", ".join(
            str(candidate) for candidate in depth_prior_candidates(data_path))
        raise FileNotFoundError(
            f"Depth prior directory not found. Tried: {candidates}")
    if missing:
        preview = ", ".join(pathlib.Path(path).name for path in missing[:5])
        raise FileNotFoundError(f"Missing depth priors for {len(missing)} "
                                f"selected images: {preview}")

    png_scale = dataset_depth_png_scale(data_path)
    return [
        load_depth_prior_file(depth_path, image_path, png_scale)
        for image_path, depth_path in zip(image_paths, depth_paths, strict=True)
    ]


def load_optional_depth_priors(
    data_path: pathlib.Path, image_paths: list[str]
) -> tuple[list[np.ndarray | None], np.ndarray, list[str]]:
    """Loads the depth priors that exist, with a per-image availability mask.

    Returns:
        A ([map or None per image], has_depth (N,) bool,
        [images without a prior]) tuple.
    """
    depth_paths, missing = resolve_optional_depth_prior_paths(
        data_path, image_paths)
    png_scale = dataset_depth_png_scale(data_path)
    depths = []
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
    data_path: pathlib.Path, image_paths: list[str]
) -> tuple[list[np.ndarray | None], np.ndarray, list[str]]:
    """Loads the depth priors of the masked-depth tracks.

    The masking (spatialarena/depth_prior_masking.py) is applied at each
    prior's own resolution, before any model resizes it, so every method of
    rgb_masked_depth(_cameras) gets the same input.

    Returns:
        The same tuple as `load_optional_depth_priors`, with masked maps.
    """
    # pylint: disable-next=import-outside-toplevel
    from spatialarena import depth_prior_masking

    depths, has_depth, missing = load_optional_depth_priors(
        data_path, image_paths)
    masked = []
    for depth, image_path in zip(depths, image_paths, strict=True):
        if depth is not None:
            depth = depth_prior_masking.mask_depth_prior(
                depth, data_path, image_path)
        masked.append(depth)
    return masked, has_depth, missing


def load_sparse_depth_priors(
    data_path: pathlib.Path,
    image_paths: list[str],
    *,
    model_long_side: int | None = None
) -> tuple[list[np.ndarray | None], np.ndarray, list[str]]:
    """Loads the depth priors of the sparse-depth tracks.

    Each image's sparse points (KITTI LiDAR or COLMAP keypoints, see
    spatialarena/sparse_depth.py) are drawn as a map of the image's size
    (0 = no depth), each point a small square sized to survive a
    nearest-neighbour resize to a `model_long_side` input as about one pixel.

    Returns:
        The same tuple as `load_optional_depth_priors`; an image without
        points is None and listed as missing.
    """
    # pylint: disable-next=import-outside-toplevel
    from spatialarena import sparse_depth

    depths = []
    has_depth = np.zeros((len(image_paths),), dtype=bool)
    missing = []
    for index, image_path in enumerate(image_paths):
        with Image.open(image_path) as image:
            size = image.size
        depth = sparse_depth.load_sparse_depth(data_path,
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
        resample: int = Image.Resampling.NEAREST) -> np.ndarray:
    """Resizes a depth prior to an image's (width, height).

    Some layouts (e.g. ETH3D) pair depth maps with images whose size differs
    after upstream preprocessing, and methods such as MapAnything require
    equal image and depth shapes before their own resize.

    Returns:
        The resized float32 map; non-finite values become 0.
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


# ----------------------------------------------------------------------------
# Batching and model inputs.
# ----------------------------------------------------------------------------

# Set to "1" by the engine for MonoDepth: every image is predicted on its
# own, never as a view of a sequence. Models either run independent views as
# one [B, 1, 3, H, W] batch (fold_single_view_outputs) or call the model once
# per image (batched_image_paths returns one-image chunks).
SINGLE_VIEW_ENV = "BENCHMARK_SINGLE_VIEW"


def single_view_mode() -> bool:
    """Returns whether each image is predicted independently (MonoDepth)."""
    return os.environ.get(SINGLE_VIEW_ENV, "0") == "1"


def batched_image_paths(image_paths: list[str],
                        max_images_per_run: int,
                        *,
                        independent_batch: bool = False) -> list[list[str]]:
    """Returns the image paths of each model call.

    Args:
        image_paths: The images.
        max_images_per_run: Images per call (0: all in one call).
        independent_batch: The caller runs a chunk as independent views
            ([B, 1, 3, H, W]); otherwise single-view mode gives one image
            per chunk.

    Returns:
        The chunks.
    """
    if single_view_mode() and not independent_batch:
        return [[path] for path in image_paths]
    if max_images_per_run <= 0 or max_images_per_run >= len(image_paths):
        return [image_paths]
    return [
        image_paths[start:start + max_images_per_run]
        for start in range(0, len(image_paths), max_images_per_run)
    ]


def fold_single_view_outputs(outputs: Any, num_samples: int) -> Any:
    """Maps [B, 1, ...] outputs of independent views to the [1, B, ...] layout.

    Then a model's sequence post-processing applies unchanged. Every array
    or tensor whose leading dims are (num_samples, 1) has them swapped;
    dicts, lists and tuples are folded recursively; anything else (scalars,
    metadata) is returned as is.

    Args:
        outputs: The model outputs.
        num_samples: Number of independent views B.

    Returns:
        The outputs in sequence layout.
    """
    shape = getattr(outputs, "shape", None)
    if (shape is not None and len(shape) >= 2 and
            tuple(shape[:2]) == (num_samples, 1)):
        if isinstance(outputs, np.ndarray):
            return np.swapaxes(outputs, 0, 1)
        return outputs.transpose(0, 1)  # A torch.Tensor.
    if isinstance(outputs, dict):
        folded = type(outputs)()  # Keeps dict subclasses (DA3's addict.Dict).
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
    """Returns the smallest multiple of `multiple` at or above `value`.

    Raises:
        ValueError: An argument is not positive.
    """
    if value <= 0:
        raise ValueError(f"Expected a positive value, got {value}")
    if multiple <= 0:
        raise ValueError(f"Expected a positive multiple, got {multiple}")
    return ((value + multiple - 1) // multiple) * multiple


def patch_grid_compatible_resolution(requested_resolution: int,
                                     *,
                                     patch_size: int,
                                     grid_multiple: int = 1) -> int:
    """Returns the nearest larger square size on the patch grid.

    Raises:
        ValueError: `patch_size` or `grid_multiple` is not positive.
    """
    if patch_size <= 0:
        raise ValueError(f"Expected positive patch_size, got {patch_size}")
    if grid_multiple <= 0:
        raise ValueError(
            f"Expected positive grid_multiple, got {grid_multiple}")
    return round_up_to_multiple(requested_resolution,
                                patch_size * grid_multiple)


FIXED_INPUT_ENV = "FIXED_INPUT_RESOLUTION"


def fixed_input_size(image_paths: list[str]) -> tuple[int, int] | None:
    """Returns the (width, height) of pre-sized inputs, or None.

    With FIXED_INPUT_RESOLUTION=1 (the full-slam suite, see
    slam_protocol.DATASET_INPUT_SIZE) the engine feeds frames every method
    must use as they are: a model then skips its own rescale (a pad to its
    required shape, or a center crop to a 16-pixel patch grid, is all it may
    still do) and sizes its settings from this.

    Raises:
        ValueError: The inputs do not share one size.
    """
    if os.environ.get(FIXED_INPUT_ENV, "0") != "1" or not image_paths:
        return None
    sizes = set()
    for image_path in image_paths:
        with Image.open(image_path) as image:
            sizes.add(image.size)
    if len(sizes) != 1:
        raise ValueError(f"Fixed-resolution inputs must share one size, "
                         f"found {sorted(sizes)}")
    return sizes.pop()


# Optional root of the long-sequence SLAM models' scratch: they keep
# per-chunk dense results on disk while running (hundreds of GB on an
# 18,846-frame VBR sequence). Unset: under the case's output path.
SLAM_SCRATCH_ENV = "SLAM_SCRATCH_ROOT"


def slam_work_dir(method: str, output_path: pathlib.Path) -> pathlib.Path:
    """Returns a fresh scratch directory for one SLAM case."""
    import tempfile  # pylint: disable=import-outside-toplevel

    if os.environ.get(SLAM_SCRATCH_ENV):
        root = pathlib.Path(os.environ[SLAM_SCRATCH_ENV])
    else:
        root = pathlib.Path(output_path)
    root.mkdir(parents=True, exist_ok=True)
    return pathlib.Path(tempfile.mkdtemp(prefix=f"{method.lower()}_", dir=root))


_MODEL_INPUT_SIZE: list[int] | None = None


def record_model_input(width: int, height: int) -> None:
    """Records the size of the image tensor the model received.

    metrics.json reports it as model_input_size (`runtime_metrics`), so the
    fixed-resolution protocol can be checked per case.
    """
    # pylint: disable-next=global-statement  # Read by runtime_metrics.
    global _MODEL_INPUT_SIZE
    _MODEL_INPUT_SIZE = [int(width), int(height)]


def load_patch_grid_tensor(image_paths: list[str], patch_size: int):
    """Loads fixed-resolution frames center-cropped to the patch grid.

    The frames are never rescaled (e.g. 504x378 -> 496x368 for 16-pixel
    patches).

    Returns:
        A [N, 3, H, W] float tensor in [0, 1].
    """
    # pylint: disable=import-outside-toplevel
    import torch
    from torchvision.transforms import functional as TF

    tensors = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            crop_w = width // patch_size * patch_size
            crop_h = height // patch_size * patch_size
            left = (width - crop_w) // 2
            top = (height - crop_h) // 2
            box = (left, top, left + crop_w, top + crop_h)
            tensors.append(TF.to_tensor(image.crop(box)))
    return torch.stack(tensors)


def _pad_to_canvas(image, target_width, target_height):
    """Resizes an image to fit a canvas, keeping its aspect, and centers it.

    Returns:
        A (canvas, resized (width, height), offset (x, y)) tuple.
    """
    original_width, original_height = image.size
    scale = min(target_width / max(original_width, 1),
                target_height / max(original_height, 1))
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))
    if scale < 1.0:
        resample = Image.Resampling.LANCZOS
    else:
        resample = Image.Resampling.BICUBIC
    resized = image.resize((resized_width, resized_height), resample)
    canvas = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas, (resized_width, resized_height), (offset_x, offset_y)


def materialize_resized_images(image_paths: list[str], output_dir: pathlib.Path,
                               size: int | tuple[int, int]) -> list[str]:
    """Writes copies of the images padded to one size, for stacked inputs.

    Each image is resized to fit the canvas, keeping its aspect ratio, and
    centered on black. A manifest.json records every copy's geometry.

    Args:
        image_paths: Source images.
        output_dir: Directory to write; its old files are removed.
        size: Canvas side, or (width, height).

    Returns:
        The paths of the copies, named by index.

    Raises:
        ValueError: The size is not positive.
    """
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

    resized_paths = []
    manifest = []
    for index, image_path in enumerate(image_paths):
        source_path = pathlib.Path(image_path)
        with Image.open(source_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            original_size = list(image.size)
            canvas, resized_size, offset = _pad_to_canvas(
                image, target_width, target_height)

        destination = output_dir / f"{index:06d}.png"
        canvas.save(destination)
        resized_paths.append(str(destination))
        manifest.append({
            "index": index,
            "source": str(source_path),
            "resized": str(destination),
            "source_size": original_size,
            "resized_size": list(resized_size),
            "canvas_size": [target_width, target_height],
            "offset": list(offset),
        })

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) +
                                              "\n",
                                              encoding="utf-8")
    return resized_paths


def should_keep_intermediate(args: Any) -> bool:
    """Returns whether scratch files are kept (--keep_intermediate or env)."""
    if bool(getattr(args, "keep_intermediate", False)):
        return True
    keep = os.environ.get("KEEP_INTERMEDIATE", "0").lower()
    return keep in {"1", "true", "yes", "on"}


def cleanup_intermediate_paths(paths: list[pathlib.Path | None], *,
                               keep: bool) -> None:
    """Removes scratch directories after a successful export.

    Args:
        paths: The directories; None entries had nothing staged.
        keep: Keep them instead (debugging).
    """
    if keep:
        return
    for path in paths:
        if path is not None and path.is_dir():
            shutil.rmtree(path)


def planned_layout(args: Any, method: str, num_images: int) -> dict[str, Any]:
    """Returns the outputs a run would write (the dry run)."""
    output_path = args.output_path
    dense_points = None
    if not args.depth_only:
        dense_points = str(output_path / "points3D_dense.ply")
    depths = str(output_path / "depths") if args.export_depth else None
    sparse = None
    if args.export_colmap and not args.depth_only:
        sparse = str(output_path / "sparse" / args.colmap_model_subdir)
    return {
        "method": method,
        "data_path": str(args.data_path),
        "output_path": str(output_path),
        "num_images": num_images,
        "checkpoint": getattr(args, "ckpt_path", None),
        "metrics": str(output_path / "metrics.json"),
        "dense_points": dense_points,
        "depths": depths,
        "sparse": sparse,
    }


def print_dry_run(args: Any, method: str, num_images: int) -> None:
    """Prints `planned_layout` as JSON."""
    print(json.dumps(planned_layout(args, method, num_images), indent=2))


# The adapter scripts live here, and several share a name with the package
# they wrap (scripts/inference/vggt.py and `vggt`), so add_code_path drops
# this directory from sys.path before the method's code is imported.
INFERENCE_SCRIPTS_DIR = ROOT_DIR / "scripts" / "inference"


def add_code_path(
    path: pathlib.Path | None, package_names: tuple[str, ...] = ()) -> None:
    """Puts a method's code on sys.path, ahead of anything shadowing it.

    Args:
        path: Checkout of the method's code, or None to only clean up.
        package_names: The method's top-level packages; a same-named
            non-package module already imported (an adapter script) is
            removed from sys.modules.
    """
    script_dir = str(INFERENCE_SCRIPTS_DIR.resolve())
    sys.path[:] = [
        entry for entry in sys.path
        if str(pathlib.Path(entry or ".").resolve()) != script_dir
    ]
    for package_name in package_names:
        shadowed = sys.modules.get(package_name)
        if shadowed is not None and not hasattr(shadowed, "__path__"):
            del sys.modules[package_name]
    if path is None:
        return
    code_path = pathlib.Path(path).expanduser().resolve()
    if str(code_path) not in sys.path:
        sys.path.insert(0, str(code_path))


# ----------------------------------------------------------------------------
# Array conversions.
# ----------------------------------------------------------------------------


def as_numpy(value: Any) -> np.ndarray:
    """Returns a tensor (as float) or array-like as a numpy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().float().numpy()
    return np.asarray(value)


def normalize_depths(depths: Any) -> np.ndarray:
    """Returns depth maps as a float32 (N, H, W) array.

    Accepts the layouts models return: a leading batch axis of 1, a
    trailing or second channel axis of 1.

    Raises:
        ValueError: The array cannot be read as (N, H, W).
    """
    depth_array = as_numpy(depths).astype(np.float32, copy=False)
    if depth_array.ndim == 5 and depth_array.shape[0] == 1:
        depth_array = depth_array[0]
    # The trailing channel ("N x H x W x 1", e.g. VGGT's depth head) goes
    # first: with one frame (N == 1, every MonoDepth run) the batch rule
    # below would take N for a batch axis and leave H "frames".
    if depth_array.ndim == 4 and depth_array.shape[-1] == 1:
        depth_array = depth_array[..., 0]
    # A leading batch axis of 1 ("1 x N x H x W", e.g. Pi3's confidence) is
    # unambiguous when shape[1] != 1, so it goes before the "N x 1 x H x W"
    # rule below.
    if (depth_array.ndim == 4 and depth_array.shape[0] == 1 and
            depth_array.shape[1] != 1):
        depth_array = depth_array[0]
    if depth_array.ndim == 4 and depth_array.shape[1] == 1:
        depth_array = depth_array[:, 0]
    if depth_array.ndim != 3:
        raise ValueError(
            f"Expected depth array with shape NxHxW, got {depth_array.shape}")
    return depth_array


def normalize_points(points: Any) -> np.ndarray:
    """Returns point maps as a float32 (N, H, W, 3) array.

    Raises:
        ValueError: The array cannot be read as (N, H, W, 3).
    """
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
    """Returns confidence maps as a float32 array of the depth `shape`.

    Args:
        confidence: The model's confidence, or None for all ones.
        shape: The (N, H, W) of the depth maps.

    Raises:
        ValueError: The array cannot be read as (N, H, W).
    """
    if confidence is None:
        return np.ones(shape, dtype=np.float32)
    conf_array = as_numpy(confidence).astype(np.float32, copy=False)
    if conf_array.size == int(np.prod(shape)):
        # One value per depth pixel: the layout is unambiguous, also for
        # N == 1 (e.g. 1xHxWx1, where the batch rule below would drop the
        # frame axis instead of the channel axis).
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
        raise ValueError(f"Expected confidence map with shape NxHxW, got "
                         f"{conf_array.shape}")
    return conf_array


def _channels_last_images(images: Any, label: str) -> np.ndarray:
    """Returns images as an (N, H, W, 3) array, from NCHW or NHWC."""
    array = as_numpy(images)
    if array.ndim == 5 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 4 and array.shape[1] == 3:
        array = np.transpose(array, (0, 2, 3, 1))
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(
            f"Expected {label} with shape NxHxWx3, got {array.shape}")
    return array


def colors_from_torch_images(images: Any) -> np.ndarray:
    """Returns [0, 1] model images as uint8 (N, H, W, 3) colors."""
    image_array = _channels_last_images(images, "images")
    if image_array.dtype != np.uint8:
        image_array = (image_array.clip(0.0, 1.0) * 255.0).astype(np.uint8)
    return image_array


def ensure_uint8_colors(colors: Any) -> np.ndarray:
    """Returns colors in [0, 1] or [0, 255] as uint8 (N, H, W, 3)."""
    color_array = _channels_last_images(colors, "colors")
    if color_array.dtype != np.uint8:
        if color_array.max(initial=0.0) <= 1.0:
            color_array = color_array * 255.0
        color_array = color_array.clip(0, 255).astype(np.uint8)
    return color_array


def c2w_to_w2c(camera_to_world: Any) -> np.ndarray:
    """Inverts (N, 4, 4) camera-to-world matrices to world-to-camera.

    Raises:
        ValueError: The matrices are not (N, 4, 4).
    """
    c2w = as_numpy(camera_to_world).astype(np.float32, copy=False)
    if c2w.ndim == 4 and c2w.shape[0] == 1:
        c2w = c2w[0]
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"Expected camera-to-world matrices with shape "
                         f"Nx4x4, got {c2w.shape}")
    w2c = np.repeat(np.eye(4, dtype=np.float32)[None], c2w.shape[0], axis=0)
    rotation_inv = np.transpose(c2w[:, :3, :3], (0, 2, 1))
    translation = c2w[:, :3, 3]
    w2c[:, :3, :3] = rotation_inv
    w2c[:, :3,
        3] = np.negative(np.einsum("nij,nj->ni", rotation_inv, translation))
    return w2c


def quaternion_xyzw_to_matrix(quaternions: Any) -> np.ndarray:
    """Returns the rotation matrices of (..., 4) xyzw quaternions.

    Raises:
        ValueError: The last axis is not 4.
    """
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
    """Returns (N, 4, 4) matrices of (N, 7) [tx, ty, tz, qx, qy, qz, qw] poses.

    Raises:
        ValueError: The poses are not (N, 7).
    """
    pose_array = as_numpy(poses).astype(np.float32, copy=False)
    if pose_array.ndim == 3 and pose_array.shape[0] == 1:
        pose_array = pose_array[0]
    if pose_array.ndim != 2 or pose_array.shape[1] != 7:
        raise ValueError(f"Expected SE3 pose vectors with shape Nx7, got "
                         f"{pose_array.shape}")
    c2w = np.repeat(np.eye(4, dtype=np.float32)[None],
                    pose_array.shape[0],
                    axis=0)
    c2w[:, :3, :3] = quaternion_xyzw_to_matrix(pose_array[:, 3:])
    c2w[:, :3, 3] = pose_array[:, :3]
    return c2w


def intrinsics_from_focals(focals: Any, height: int, width: int) -> np.ndarray:
    """Returns (N, 3, 3) intrinsics of per-image focals, centered."""
    focal_array = as_numpy(focals).astype(np.float32, copy=False).reshape(-1)
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None],
                           focal_array.shape[0],
                           axis=0)
    intrinsics[:, 0, 0] = focal_array
    intrinsics[:, 1, 1] = focal_array
    intrinsics[:, 0, 2] = (width - 1) / 2.0
    intrinsics[:, 1, 2] = (height - 1) / 2.0
    return intrinsics


def unproject_depths_to_world(depths: Any, extrinsics_w2c: Any,
                              intrinsics: Any) -> np.ndarray:
    """Returns the (N, H, W, 3) world points of depth maps and cameras."""
    depth_array = normalize_depths(depths)
    extrinsics = as_numpy(extrinsics_w2c).astype(np.float32, copy=False)
    intrinsics_array = as_numpy(intrinsics).astype(np.float32, copy=False)
    num_frames, height, width = depth_array.shape
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    grid_shape = (num_frames, height, width)
    x = np.broadcast_to(x[None], grid_shape).astype(np.float32)
    y = np.broadcast_to(y[None], grid_shape).astype(np.float32)

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


# ----------------------------------------------------------------------------
# Export.
# ----------------------------------------------------------------------------

# Array names of the per-frame maps (<dir>/<stem>.npz): depths/,
# raw_disparities/ and raw_log_depths/ hold "depth" (the evaluator's loader
# also takes the first array), confidences/ holds "confidence".
PREDICTION_MAP_KEYS = {"confidences": "confidence"}
FLOAT16_MAX = float(np.finfo(np.float16).max)


def export_depth_predictions(output_path: pathlib.Path,
                             image_paths: list[str],
                             depths: Any,
                             confidence: Any | None = None) -> None:
    """Writes one depth map per frame, and its confidence when given.

    Every map is one compressed .npz (`save_prediction_map`); writing several
    formats of the same map would multiply storage and the frame keys.

    Args:
        output_path: Method scene output directory.
        image_paths: Source images, one per map; maps are named by stem.
        depths: Depth maps, (N, H, W) in any `normalize_depths` layout.
        confidence: The method's own confidence on its own scale, written to
            confidences/<stem>.npz as float16, or None.
    """
    depth_array = normalize_depths(depths)
    depth_dir = output_path / "depths"
    for image_path, depth in zip(image_paths, depth_array):
        save_prediction_map(depth_dir, pathlib.Path(image_path).stem, depth)
    if confidence is not None:
        export_confidence_predictions(output_path, image_paths, confidence,
                                      depth_array.shape)


def save_prediction_map(directory: pathlib.Path,
                        stem: str,
                        array: Any,
                        key: str | None = None,
                        dtype: Any = np.float32) -> pathlib.Path:
    """Writes one frame's map as <directory>/<stem>.npz, compressed.

    An older <stem>.npy of the frame is removed, so a frame never has two
    files.

    Args:
        directory: Map directory (depths/, confidences/, ...).
        stem: The frame's file stem.
        array: The map.
        key: Array name; defaults to "confidence" in confidences/, else
            "depth".
        dtype: float32 for depth; float16 for confidence (clipped to its
            range, NaN kept).

    Returns:
        The path written.
    """
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    key = key or PREDICTION_MAP_KEYS.get(directory.name, "depth")
    values = np.asarray(array, dtype=np.float32)
    if np.dtype(dtype) == np.float16:
        values = np.clip(values, -FLOAT16_MAX, FLOAT16_MAX)  # NaN stays NaN.
    path = directory / f"{stem}.npz"
    np.savez_compressed(path, **{key: values.astype(dtype)})
    (directory / f"{stem}.npy").unlink(missing_ok=True)
    return path


def export_confidence_predictions(output_path: pathlib.Path,
                                  image_paths: list[str], confidence: Any,
                                  shape: tuple[int, int, int]) -> None:
    """Writes confidences/<stem>.npz: float16, the pixels of depths/.

    Values are the method's raw confidence (VGGT's 1 + exp(x), Pi3's
    sigmoid, ...), clipped to the float16 range.

    Args:
        output_path: Method scene output directory.
        image_paths: Source images, one per map.
        confidence: The confidence maps.
        shape: The (N, H, W) of the depth maps.

    Raises:
        ValueError: The maps do not match the depth (a crop or resize
            applied to one and not the other).
    """
    conf_array = normalize_confidence(confidence, shape)
    if conf_array.shape != tuple(shape):
        raise ValueError(f"confidence {conf_array.shape} does not match depth "
                         f"{tuple(shape)}")
    for image_path, conf in zip(image_paths, conf_array):
        save_confidence_map(output_path, image_path, conf)


def save_confidence_map(output_path: pathlib.Path, image_path: str,
                        confidence: np.ndarray) -> None:
    """Writes one frame's confidences/<stem>.npz as float16."""
    save_prediction_map(pathlib.Path(output_path) / "confidences",
                        pathlib.Path(image_path).stem,
                        confidence,
                        dtype=np.float16)


def export_camera_pose_priors(output_path: pathlib.Path,
                              images_dir: pathlib.Path,
                              image_paths: list[str],
                              extrinsics_w2c: Any,
                              intrinsics: Any,
                              *,
                              width: int | list[int],
                              height: int | list[int],
                              colmap_model_subdir: str = "0") -> None:
    """Writes cameras as a COLMAP model without points.

    MVRecon evaluation fuses depth with these poses itself
    (mvrecon_evaluator.py::prepare_fusion_inputs), so depth-only batches
    need no dense point cloud; poses are tiny and accumulate cheaply even
    over thousands of frames.

    Args:
        output_path: Method scene output directory.
        images_dir: The scene's image directory (for relative names).
        image_paths: Images, one per camera.
        extrinsics_w2c: World-to-camera matrices.
        intrinsics: Pinhole intrinsics.
        width: Image width, one for all or one per image.
        height: Image height, one for all or one per image.
        colmap_model_subdir: Model subdirectory under sparse/.
    """
    # pylint: disable-next=import-outside-toplevel
    from spatialarena.geometry import utils as geometry_utils

    colmap_dir = output_path / "sparse" / colmap_model_subdir
    colmap_dir.mkdir(parents=True, exist_ok=True)
    geometry_utils.write_colmap_formats(
        str(images_dir),
        as_numpy(extrinsics_w2c).astype(np.float32, copy=False),
        as_numpy(intrinsics).astype(np.float32, copy=False),
        image_paths,
        np.empty((0, 3), dtype=np.float32),
        np.empty((0, 3), dtype=np.uint8),
        width=width,
        height=height,
        output_path=str(colmap_dir),
    )


# ----------------------------------------------------------------------------
# Runtime metrics.
# ----------------------------------------------------------------------------

# Standard runtime metrics, comparable across models (mark_model_loaded,
# runtime_metrics). The older "inference time" / "Peak GPU Memory" keys keep
# each adapter's own scope.
_PROCESS_STARTED_AT = time.time()
_MODEL_LOADED_AT: float | None = None


def mark_model_loaded() -> float:
    """Starts the inference timer; call once the model is on its device.

    Also resets the CUDA peak-memory counter, so inference_seconds and
    peak_gpu_memory_mib cover inference only (the resident weights still
    count in the peak) and model_load_seconds covers everything before.

    Returns:
        The timestamp.
    """
    # pylint: disable-next=global-statement  # Read by runtime_metrics.
    global _MODEL_LOADED_AT
    _MODEL_LOADED_AT = time.time()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)
    return _MODEL_LOADED_AT


def _peak_gpu_memory_mib() -> float | None:
    """Returns the peak device memory of this process in MiB, or None.

    Summed over devices, from torch or jax, whichever the method uses.
    """
    torch = sys.modules.get("torch")
    if (torch is not None and torch.cuda.is_available() and
            torch.cuda.is_initialized()):
        total = sum(
            torch.cuda.max_memory_allocated(i)
            for i in range(torch.cuda.device_count()))
        return total / 1024.0**2
    jax = sys.modules.get("jax")
    if jax is None:
        return None
    try:
        stats = [
            device.memory_stats() or {}
            for device in jax.local_devices()
            if device.platform == "gpu"
        ]
    except Exception:  # pylint: disable=broad-exception-caught  # Best effort.
        return None
    peaks = [s["peak_bytes_in_use"] for s in stats if "peak_bytes_in_use" in s]
    if not peaks:
        return None
    return sum(peaks) / 1024.0**2


def runtime_metrics(started_at: float) -> dict[str, Any]:
    """Returns the standard runtime metrics; call when inference finishes.

    Args:
        started_at: The model's own start time, used when
            `mark_model_loaded` was never called (runtime_scope
            "adapter_started_at" instead of "after_model_load").

    Returns:
        The metrics.json runtime fields.
    """
    now = time.time()
    loaded = _MODEL_LOADED_AT
    if loaded is None:
        return {
            "inference_seconds": now - started_at,
            "model_load_seconds": None,
            "peak_gpu_memory_mib": _peak_gpu_memory_mib(),
            "runtime_scope": "adapter_started_at",
            "model_input_size": _MODEL_INPUT_SIZE,
        }
    return {
        "inference_seconds": now - loaded,
        "model_load_seconds": loaded - _PROCESS_STARTED_AT,
        "peak_gpu_memory_mib": _peak_gpu_memory_mib(),
        "runtime_scope": "after_model_load",
        "model_input_size": _MODEL_INPUT_SIZE,
    }


def write_adapter_metrics(*,
                          output_path: pathlib.Path,
                          method: str,
                          image_paths: list[str],
                          started_at: float,
                          checkpoint: str | None,
                          depth_only: bool,
                          metrics_extra: dict[str, Any] | None = None,
                          runtime: dict[str, Any] | None = None) -> None:
    """Writes metrics.json.

    Args:
        output_path: Method scene output directory.
        method: The "method" field.
        image_paths: The processed images.
        started_at: Start time of the legacy "inference time" field.
        checkpoint: The checkpoint used.
        depth_only: Whether this was a depth-only run.
        metrics_extra: The model's own fields.
        runtime: `runtime_metrics` taken before the export, or None to take
            them now.
    """
    metrics = {
        "method": method,
        "Preprocessed Images": len(image_paths),
        "inference time": time.time() - started_at,
        "checkpoint": checkpoint,
        "depth_only": depth_only,
    }
    if metrics_extra:
        metrics.update(metrics_extra)
    if runtime is None:
        runtime = runtime_metrics(started_at)
    metrics.update(runtime)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "metrics.json").write_text(json.dumps(metrics, indent=2) +
                                              "\n",
                                              encoding="utf-8")


def sample_points(points: Any, colors: Any, confidence: Any | None,
                  conf_thresh: float,
                  max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns the confident finite points, at most `max_points` of them.

    Args:
        points: Point maps, (N, H, W, 3).
        colors: Their colors.
        confidence: Per-pixel confidence, or None.
        conf_thresh: Lowest kept confidence.
        max_points: Random subset size (0: keep all).

    Returns:
        A (points (M, 3), colors (M, 3) uint8) tuple.

    Raises:
        ValueError: Points and colors differ in shape.
    """
    point_array = normalize_points(points)
    color_array = ensure_uint8_colors(colors)
    conf_array = normalize_confidence(confidence, point_array.shape[:3])
    if color_array.shape[:3] != point_array.shape[:3]:
        raise ValueError(f"Point/color spatial shapes do not match: "
                         f"{point_array.shape} vs {color_array.shape}")
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


def _colmap_image_size(depths, points):
    """Returns the (height, width) of the COLMAP cameras of a prediction."""
    if depths is not None:
        return normalize_depths(depths).shape[1:]
    if points is not None:
        return normalize_points(points).shape[1:3]
    raise ValueError("Cannot determine COLMAP image size")


def _export_dense_points(output_path, depths, points, colors, confidence,
                         extrinsics_w2c, intrinsics, conf_thresh, max_points):
    """Samples and writes points3D_dense.ply; returns the sampled points.

    Without point maps, the points are unprojected from depth and cameras.
    """
    # pylint: disable-next=import-outside-toplevel
    from spatialarena.utils import point_utils

    if points is None:
        if depths is None or extrinsics_w2c is None or intrinsics is None:
            raise ValueError("Dense point export requires point maps or "
                             "depth + camera predictions")
        points = unproject_depths_to_world(depths, extrinsics_w2c, intrinsics)
    if colors is None:
        raise ValueError("Dense point export requires colors")
    sampled_points, sampled_colors = sample_points(points,
                                                   colors,
                                                   confidence,
                                                   conf_thresh=conf_thresh,
                                                   max_points=max_points)
    point_utils.export_ply(str(output_path / "points3D_dense.ply"),
                           sampled_points, sampled_colors)
    return sampled_points, sampled_colors


def export_standard_artifacts(
        *,
        method: str,
        output_path: pathlib.Path,
        images_dir: pathlib.Path,
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
        metrics_extra: dict[str, Any] | None = None) -> None:
    """Writes the standard artifacts of a whole-scene prediction.

    That is depth (with --export_depth), the sampled dense points (unless
    depth-only), a COLMAP model of the cameras and those points (with
    --export_colmap), and metrics.json. Runtime is measured before the
    export, so writing files is not timed.

    Raises:
        ValueError: A requested artifact needs an output the model did not
            produce.
    """
    runtime = runtime_metrics(started_at)
    output_path.mkdir(parents=True, exist_ok=True)
    if export_depth:
        if depths is None:
            raise ValueError("--export_depth requested but the adapter did "
                             "not produce depths")
        export_depth_predictions(output_path, image_paths, depths, confidence)

    sampled_points = None
    sampled_colors = None
    if not depth_only:
        sampled_points, sampled_colors = _export_dense_points(
            output_path, depths, points, colors, confidence, extrinsics_w2c,
            intrinsics, conf_thresh, max_points)

    if export_colmap and not depth_only:
        # pylint: disable-next=import-outside-toplevel
        from spatialarena.geometry import utils as geometry_utils

        if extrinsics_w2c is None or intrinsics is None:
            raise ValueError("--export_colmap requested but the adapter did "
                             "not produce cameras")
        if sampled_points is None or sampled_colors is None:
            raise ValueError("--export_colmap requested but no point cloud "
                             "was sampled")
        intrinsics_array = as_numpy(intrinsics).astype(np.float32, copy=False)
        height, width = _colmap_image_size(depths, points)
        colmap_dir = output_path / "sparse" / colmap_model_subdir
        colmap_dir.mkdir(parents=True, exist_ok=True)
        geometry_utils.write_colmap_formats(
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


# ----------------------------------------------------------------------------
# Spatial policy: source image <-> model grid.
#
# Ported from FeedForward-Benchmark PR #21 (eval_refactor_hsy, commits
# 21f0a728/d88025f4) rather than merging the whole PR, which still has
# other things to correct; remove this note once the PR lands.
# ----------------------------------------------------------------------------

# Policies that pad with zeros (a square canvas); the others replicate edges.
_ZERO_PAD_POLICIES = (
    "omega_resize_pad_square",
    "vggt_resize_pad_square",
    "fastvggt_resize_pad_square",
)


@dataclasses.dataclass(frozen=True)
class SpatialTransform:
    """The resize, crop and pad from a source image to the model grid.

    Applied in that order. Sizes are (width, height); margins are (left,
    top, right, bottom).

    Attributes:
        source_wh: Size of the source image.
        resized_wh: Size after the resize.
        padding: Margins added after the crop.
        crop: Margins removed after the resize.
        policy: Name of the policy that planned it.
    """

    source_wh: tuple[int, int]
    resized_wh: tuple[int, int]
    padding: tuple[int, int, int, int] = (0, 0, 0, 0)
    crop: tuple[int, int, int, int] = (0, 0, 0, 0)
    policy: str = "resize"

    def __post_init__(self):
        if (min(*self.source_wh, *self.resized_wh) <= 0 or
                min(*self.padding, *self.crop) < 0):
            raise ValueError("Invalid spatial dimensions or margins")
        crop_left, crop_top, crop_right, crop_bottom = self.crop
        if (crop_left + crop_right >= self.resized_wh[0] or
                crop_top + crop_bottom >= self.resized_wh[1]):
            raise ValueError("Crop removes entire image")

    @property
    def model_wh(self) -> tuple[int, int]:
        """The (width, height) of the model grid."""
        width, height = self.resized_wh
        pad_left, pad_top, pad_right, pad_bottom = self.padding
        crop_left, crop_top, crop_right, crop_bottom = self.crop
        return (width - crop_left - crop_right + pad_left + pad_right,
                height - crop_top - crop_bottom + pad_top + pad_bottom)

    def metadata(self) -> dict:
        """Returns the transform as a JSON-ready dict."""
        return dict(dataclasses.asdict(self),
                    model_wh=self.model_wh,
                    output_frame="source_rgb")


def _centered_padding(pad_width, pad_height):
    """Splits a total padding into (left, top, right, bottom) margins."""
    return (pad_width // 2, pad_height // 2, pad_width - pad_width // 2,
            pad_height - pad_height // 2)


def plan_spatial_transform(source_wh: tuple[int, int],
                           *,
                           method: str,
                           size: int = 512,
                           mode: str = "pad") -> SpatialTransform:
    """Plans a method's restorable resize of one source image.

    Policies by `method`:
        dvd: DVD's training scale, rounded up to 16-pixel multiples.
        dvd_compact: DVD's Sintel 1024x436 -> 1008x426 with 3-pixel padding.
        da3: DA3's upper_bound_resize (nearest 14 multiple, ties up).
        vggt, fastvggt: pad to a `size` square (patch 14; "pad" only).
        vggt_omega: "pad" to a `size` square, "rect-pad" to the next 16-pixel
            boundary, or "resize" to 16-pixel multiples.

    Args:
        source_wh: Source image (width, height).
        method: The policy, see above.
        size: Long side (and square side) of the model input.
        mode: The vggt_omega mode, see above.

    Returns:
        The transform.

    Raises:
        ValueError: The method, mode or sizes are not supported.
    """
    import math  # pylint: disable=import-outside-toplevel

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
        # Official upper_bound_resize: nearest 14 multiple, ties round up.
        rounded = tuple(max(1, ((x + 7) // 14) * 14) for x in (rw, rh))
        return SpatialTransform(source_wh,
                                rounded,
                                policy="da3_upper_bound_resize")
    if method in ("vggt", "fastvggt"):
        # Their own pad-to-square loaders (VGGT's
        # load_and_preprocess_images_square, FastVGGT's
        # materialize_resized_images) use patch 14, not 16, so only "pad" is
        # supported: "resize" and "rect-pad" below round to 16.
        if mode != "pad":
            raise ValueError(f"Unsupported spatial policy: {method}/{mode} "
                             "(only 'pad' is supported)")
        if size % 14:
            raise ValueError(
                f"{method} square canvas size must be a multiple of 14")
        return SpatialTransform(source_wh, (rw, rh),
                                _centered_padding(size - rw, size - rh),
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
        return SpatialTransform(source_wh, (rw, rh),
                                _centered_padding(size - rw, size - rh),
                                policy="omega_resize_pad_square")
    # Rectangular experiment: pad only to the next patch boundary.
    return SpatialTransform(source_wh, (rw, rh),
                            _centered_padding((-rw) % 16, (-rh) % 16),
                            policy="omega_resize_pad_patch16")


def apply_spatial_transform_tensor(
        images: torch.Tensor,
        transform: SpatialTransform,
        *,
        interpolation: str = "bilinear") -> torch.Tensor:
    """Maps NCHW float images to the model grid (no normalization).

    Args:
        images: Images of the transform's source size.
        transform: The transform.
        interpolation: "bilinear", or "nearest" for a map whose zeros mark
            missing values (a masked or sparse depth prior), so they are not
            blended into their valid neighbours.

    Returns:
        The images on the model grid.

    Raises:
        ValueError: The images do not have the source size.
    """
    # pylint: disable-next=import-outside-toplevel
    import torch.nn.functional as F

    if tuple(images.shape[-2:]) != transform.source_wh[::-1]:
        raise ValueError("RGB does not match recorded source dimensions")
    if transform.resized_wh != transform.source_wh:
        target_hw = transform.resized_wh[::-1]
        if interpolation == "nearest":
            images = F.interpolate(images, size=target_hw, mode="nearest-exact")
        else:
            images = F.interpolate(images,
                                   size=target_hw,
                                   mode="bilinear",
                                   align_corners=False)

    crop_left, crop_top, crop_right, crop_bottom = transform.crop
    images = images[..., crop_top:images.shape[-2] - crop_bottom,
                    crop_left:images.shape[-1] - crop_right]
    if any(transform.padding):
        pad_left, pad_top, pad_right, pad_bottom = transform.padding
        margins = (pad_left, pad_right, pad_top, pad_bottom)
        if transform.policy in _ZERO_PAD_POLICIES:
            images = F.pad(images, margins, mode="constant", value=0)
        else:
            images = F.pad(images, margins, mode="replicate")
    return images


def preprocess_images(
    image_paths: list[str],
    *,
    method: str,
    size: int = 512,
    mode: str = "pad",
    exif_transpose: bool = False
) -> tuple[torch.Tensor, list[SpatialTransform]]:
    """Loads images onto the model grid of a shared spatial policy.

    Args:
        image_paths: Source images.
        method: The `plan_spatial_transform` policy.
        size: Long side of the model input.
        mode: The `plan_spatial_transform` mode.
        exif_transpose: Apply EXIF orientation first. Match the method's own
            legacy loader, so `source_wh` describes the same pixel grid:
            VGGT-family loaders never transpose; DUSt3R-family loaders
            (CUT3R, Fast3R, AMB3R, MapAnything) always do.

    Returns:
        An (images [N, 3, H, W] in [0, 1], transforms) tuple.

    Raises:
        ValueError: The images do not share one model grid; split them.
    """
    import torch  # pylint: disable=import-outside-toplevel

    images = []
    transforms = []
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
        raise ValueError("A batch must have a common model grid; split "
                         "heterogeneous image shapes")
    return torch.stack(images), transforms


def restore_prediction(prediction: np.ndarray,
                       transform: SpatialTransform) -> np.ndarray:
    """Restores one scalar map from the model grid to the source image.

    Cropped-away borders are never invented: any pixel whose interpolation
    touches a missing value is NaN.

    Raises:
        ValueError: The map is not on the transform's model grid.
    """
    import cv2  # pylint: disable=import-outside-toplevel

    array = np.asarray(prediction, dtype=np.float32)
    if array.shape != transform.model_wh[::-1]:
        raise ValueError(f"Prediction grid {array.shape} != recorded model "
                         f"grid {transform.model_wh[::-1]}")
    pad_left, pad_top, pad_right, pad_bottom = transform.padding
    array = array[pad_top:array.shape[0] - pad_bottom,
                  pad_left:array.shape[1] - pad_right]
    crop_left, crop_top, crop_right, crop_bottom = transform.crop
    canvas = np.full(transform.resized_wh[::-1], np.nan, dtype=np.float32)
    canvas[crop_top:canvas.shape[0] - crop_bottom,
           crop_left:canvas.shape[1] - crop_right] = array

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
        output_path: pathlib.Path, image_paths: list[str], depths: np.ndarray,
        confidence: np.ndarray | None,
        transforms: list[SpatialTransform]) -> None:
    """Writes depth and confidence restored to the source images.

    Also records each image's transform in spatial_transforms.json.

    Raises:
        ValueError: Images, maps and transforms differ in number.
    """
    arrays = normalize_depths(depths)
    if not len(image_paths) == len(arrays) == len(transforms):
        raise ValueError("Frame/transform count mismatch")
    conf = None
    if confidence is not None:
        conf = normalize_depths(confidence)
        if len(conf) != len(arrays):
            raise ValueError("Confidence frame count mismatch")

    output_path = pathlib.Path(output_path)
    records_path = output_path / "spatial_transforms.json"
    records = {}
    if records_path.exists():
        records = json.loads(records_path.read_text())
    for index, (path, depth,
                transform) in enumerate(zip(image_paths, arrays, transforms)):
        restored = restore_prediction(depth, transform)
        restored_conf = None
        if conf is not None:
            restored_conf = restore_prediction(conf[index], transform)[None]
        export_depth_predictions(output_path, [path], restored[None],
                                 restored_conf)
        records[pathlib.Path(path).name] = transform.metadata()
    records_path.write_text(json.dumps(records, indent=2))


SHARED_SPATIAL_POLICY_DEPTH_ONLY_MESSAGE = (
    "--spatial_policy shared currently supports --depth_only export only")


def validate_shared_spatial_policy_depth_only(*, is_shared: bool,
                                              depth_only: bool) -> None:
    """Rejects a shared spatial policy outside depth-only export.

    The shared policy restores depth and intrinsics to each source image,
    but dense points are indexed by the model's padded grid; the two would
    disagree. Camera export alongside depth is fine.

    Args:
        is_shared: Whether the run's policy is a shared one.
        depth_only: Whether the run is depth-only.

    Raises:
        ValueError: The policy is shared and the run is not depth-only.
    """
    if is_shared and not depth_only:
        raise ValueError(SHARED_SPATIAL_POLICY_DEPTH_ONLY_MESSAGE)


def spatial_policy_metrics(spatial_policy: str, *,
                           is_shared: bool) -> dict[str, str]:
    """Returns the metrics.json fields recording the run's spatial policy."""
    if is_shared:
        depth_export = "restored_to_source_resolution"
    else:
        depth_export = "cropped_to_original_image_region"
    return {"depth_export": depth_export, "spatial_policy": spatial_policy}


def _intrinsics_batch(intrinsics):
    """Returns (N, 3, 3) float64 copies and whether the input was one 3x3."""
    k = np.array(intrinsics, dtype=np.float64, copy=True)
    single = k.ndim == 2
    if single:
        k = k[None]
    return k, single


def _transform_scales(transform):
    """Returns the (x, y) resize factors of a transform."""
    source_w, source_h = transform.source_wh
    resized_w, resized_h = transform.resized_wh
    return resized_w / source_w, resized_h / source_h


def restore_intrinsics(intrinsics: Any,
                       transform: SpatialTransform) -> np.ndarray:
    """Maps intrinsics from the model grid back to the source image.

    The inverse of the image transform (resize, crop, then pad): the focal
    lengths undo the resize; the principal point also drops the padding and
    adds the crop back, as `restore_prediction` does for maps.

    Args:
        intrinsics: A (3, 3) or (N, 3, 3) pinhole matrix on the model grid.
        transform: The image's transform.

    Returns:
        The intrinsics in source pixels, in the input's shape.
    """
    k, single = _intrinsics_batch(intrinsics)
    scale_x, scale_y = _transform_scales(transform)
    pad_left, pad_top, _, _ = transform.padding
    crop_left, crop_top, _, _ = transform.crop
    k[:, 0, 0] /= scale_x
    k[:, 1, 1] /= scale_y
    k[:, 0, 2] = (k[:, 0, 2] - pad_left + crop_left) / scale_x
    k[:, 1, 2] = (k[:, 1, 2] - pad_top + crop_top) / scale_y
    return k[0] if single else k


def transform_intrinsics(intrinsics: Any,
                         transform: SpatialTransform) -> np.ndarray:
    """Maps intrinsics from the source image onto the model grid.

    The inverse of `restore_intrinsics`, for camera priors given to a model.

    Args:
        intrinsics: A (3, 3) or (N, 3, 3) pinhole matrix in source pixels.
        transform: The image's transform.

    Returns:
        The intrinsics on the model grid, in the input's shape.
    """
    k, single = _intrinsics_batch(intrinsics)
    scale_x, scale_y = _transform_scales(transform)
    pad_left, pad_top, _, _ = transform.padding
    crop_left, crop_top, _, _ = transform.crop
    k[:, 0, 0] *= scale_x
    k[:, 1, 1] *= scale_y
    k[:, 0, 2] = k[:, 0, 2] * scale_x - crop_left + pad_left
    k[:, 1, 2] = k[:, 1, 2] * scale_y - crop_top + pad_top
    return k[0] if single else k


def restore_camera_export(
    intrinsics: Any, transforms: list[SpatialTransform]
) -> tuple[np.ndarray, int | list[int], int | list[int]]:
    """Restores a batch's intrinsics for the COLMAP export.

    Args:
        intrinsics: (N, 3, 3) intrinsics on the model grid.
        transforms: One transform per image.

    Returns:
        A (restored intrinsics float32, width, height) tuple. Width and
        height are one size when every image has the same source size, else
        one per image (e.g. ETH3D, whose undistorted frames differ by a few
        pixels), which the COLMAP writers record per camera.
    """
    source_whs = [transform.source_wh for transform in transforms]
    if len(set(source_whs)) == 1:
        width, height = source_whs[0]
    else:
        width = [wh[0] for wh in source_whs]
        height = [wh[1] for wh in source_whs]
    restored = [
        restore_intrinsics(k, transform)
        for k, transform in zip(np.asarray(intrinsics), transforms)
    ]
    return np.stack(restored).astype(np.float32), width, height


def legacy_or_shared_tensor(
    image_paths: list[str], *, method: str, args: Any, device: str,
    legacy_loader: Callable[[list[str]], tuple[torch.Tensor, Any]]
) -> tuple[torch.Tensor, Any, list[SpatialTransform] | None]:
    """Loads a method's input under its --spatial_policy.

    "shared": one pad-to-square resize to args.resolution
    (`preprocess_images`). Anything else ("legacy", or a method's own value
    such as VGGT-Omega's "legacy-pad"): the method's own `legacy_loader`.

    Args:
        image_paths: Source images.
        method: The `plan_spatial_transform` policy of "shared".
        args: The parsed options (spatial_policy, resolution).
        device: Device of the returned tensor.
        legacy_loader: The method's loader: paths -> (images, extra).

    Returns:
        An (images, extra, transforms) tuple: extra is the legacy loader's
        second value (e.g. VGGT's crop coordinates) or None; transforms is
        None for the legacy loader.
    """
    if args.spatial_policy == "shared":
        images, transforms = preprocess_images(image_paths,
                                               method=method,
                                               size=args.resolution,
                                               mode="pad")
        return images.to(device), None, transforms
    images, legacy_extra = legacy_loader(image_paths)
    return images.to(device), legacy_extra, None


# ----------------------------------------------------------------------------
# Preprocessor registry of the adapters not yet on BaseModel.
#
# Every adapter calls prepare_model_input(METHOD, ...), which dispatches to
# that method's own preprocess_<method>() registered with
# register_preprocessor(), so `grep "^def preprocess_"` lists them all. What
# a preprocessor returns still varies (a tensor, view dicts, a batch dict,
# file paths) with what the method's own code needs. BaseModel subclasses
# implement `preprocess` instead.
# ----------------------------------------------------------------------------

_PREPROCESSOR_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_preprocessor(method: str, fn: Callable[..., Any]) -> None:
    """Registers a method's preprocess_<method>() for `prepare_model_input`.

    Call it once, right after defining the function, in the method's own
    module, e.g. `register_preprocessor("CUT3R", preprocess_cut3r)`.
    """
    _PREPROCESSOR_REGISTRY[method.upper()] = fn


def prepare_model_input(
        method: str,
        image_paths: list[str],
        *,
        args: Any,
        device: str | None = None,
        **extra: Any) -> tuple[Any, list[SpatialTransform] | None]:
    """Runs a method's registered preprocessor.

    Args:
        method: The method's registry name, e.g. "CUT3R".
        image_paths: Source images.
        args: The parsed options.
        device: Device of the input.
        **extra: Method-specific keyword arguments of its preprocessor (e.g.
            a batch index naming its scratch directory).

    Returns:
        A (model_input, transforms) tuple: model_input is whatever the
        method's own code takes; transforms are the per-image transforms,
        or None when the input cannot be restored (no or legacy policy).

    Raises:
        ValueError: The method registered no preprocessor.
    """
    try:
        preprocess = _PREPROCESSOR_REGISTRY[method.upper()]
    except KeyError as error:
        raise ValueError(
            f"No preprocessor registered for method {method!r}. That adapter "
            "module must call register_preprocessor(method, "
            "preprocess_<method_lower>) before prepare_model_input can be "
            "used for it.") from error
    return preprocess(image_paths, args=args, device=device, **extra)


def export_depths_for_policy(output_path: pathlib.Path, image_paths: list[str],
                             depths: Any, confidence: Any | None,
                             transforms: list[SpatialTransform] | None, *,
                             legacy_export: Callable[[], None]) -> None:
    """Writes depth and confidence under the run's spatial policy.

    Args:
        output_path: Method scene output directory.
        image_paths: Source images, one per map.
        depths: Depth maps on the model grid.
        confidence: Confidence maps, or None.
        transforms: Per-image transforms: the maps are restored to the
            source images. None: `legacy_export` writes them instead.
        legacy_export: The method's own export (e.g. VGGT's padding crop).
    """
    if transforms is not None:
        export_spatial_depth_predictions(output_path, image_paths, depths,
                                         confidence, transforms)
    else:
        legacy_export()
