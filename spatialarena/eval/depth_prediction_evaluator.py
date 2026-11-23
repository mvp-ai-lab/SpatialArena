# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Evaluates standardized depth prediction artifacts against canonical GT.

The evaluator scores the depth maps a method wrote for MonoDepth, VideoDepth
and MVRecon scenes. For each scene it pairs the GT depth files with the
predicted ones, builds the valid-pixel mask, fits the protocol's alignment
(per frame for MonoDepth, one fit per sequence for VideoDepth), and reports
the depth metrics and, when intrinsics are known, normal and point-cloud
metrics.

Command line (eval_mono_depth.py and eval_video_depth.py wrap `main` with a
default task):

    python -m spatialarena.eval.depth_prediction_evaluator \
        --experiment_root <root> --methods <method> --tasks MonoDepth

Library use: `resolve_depth_config` turns a named protocol into concrete
options, and `evaluate_depth_scene` scores one scene.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import dataclasses
import json
import math
import os
import pathlib
import sys
from typing import Any, Iterable

import numpy as np
from scipy import spatial

from spatialarena import spatialarena
from spatialarena.eval import canonical_dataset
from spatialarena.eval import dataset_registry
from spatialarena.eval import depth_tools
from spatialarena.eval import prediction_artifacts
# Kept as a name of this module: other modules import `find_depth_files` from
# here, and tests patch this module's binding.
from spatialarena.eval.prediction_artifacts import find_depth_files
from spatialarena.eval import result_layout


@dataclasses.dataclass(frozen=True, kw_only=True)
class DepthEvaluationConfig:
    """Options of a depth evaluation.

    Attributes:
        protocol: Named protocol ("standard", "pi3_mono", "custom", a GT-only
            protocol, ...); `resolve_depth_config` expands it.
        alignment: Alignment mode, or "standard" to take the protocol's.
        resize_interpolation: How predictions are resized to GT resolution.
        min_depth: Lower GT depth bound and the metrics' depth floor.
        max_depth: Upper GT depth bound, or None for none.
        gt_depth_scale: Factor from GT file values to metres.
        pred_depth_scale: Factor applied to the predicted values.
        post_clip_min: Lower clip of the aligned prediction, or None.
        post_clip_max: Upper clip of the aligned prediction, or None.
        post_clip_policy: "fixed", or "valid_gt_max" to clip each frame at
            its valid GT maximum.
        diode_mask_policy: "original", or "marigold_v2" for the official V2
            DIODE mask.
        require_positive_prediction: Also mask pixels whose prediction is not
            above `min_depth`.
        alignment_max_resolution: Fit the alignment at most at this
            resolution, or None for full resolution.
        filter_gt_depth_range: Use `min_depth` and `max_depth` in the GT
            validity mask.
        compute_geometry_metrics: Compute normal and point-cloud metrics.
        geometry_max_points: Points sampled per frame for the KD-tree
            metrics.
        geometry_f1_threshold: Distance threshold of point_f1, in metres.
        inverse_floor_policy: How inverse_l1_gt floors the aligned inverse
            depth.
        gt_source: Which GT the scene is scored on, "dense" or "lidar".
    """
    protocol: str = "standard"
    alignment: str = "standard"
    resize_interpolation: str = "linear"
    min_depth: float = 1e-6
    max_depth: float | None = None
    gt_depth_scale: float = 1.0
    pred_depth_scale: float = 1.0
    post_clip_min: float | None = None
    post_clip_max: float | None = None
    post_clip_policy: str = "fixed"
    diode_mask_policy: str = "original"
    require_positive_prediction: bool = False
    alignment_max_resolution: int | None = None
    filter_gt_depth_range: bool = True
    compute_geometry_metrics: bool = True
    geometry_max_points: int = 10000
    geometry_f1_threshold: float = 0.05
    # Temporary port of PR #21 (eval_refactor_hsy, commit d88025f4): selects
    # how inverse_l1_gt.score_sequence floors the aligned inverse depth
    # before reciprocating back to depth ("fixed" 1e-3, or "gt_max" ==
    # 1/max(GT) for per-image MonoDepth protocols).
    inverse_floor_policy: str = "fixed"
    # Which GT the scene is scored on: "dense" (depths/, or the MonoDepth
    # records' `depth`) or "lidar" (SpatialArena's sparse LiDAR depth,
    # lidar_depths/ or the records' `lidar_depth`, uint16 centimetres).
    gt_source: str = "dense"


# SpatialArena scores MonoDepth and VideoDepth on both its GGGS dense depth
# (`metrics`) and its sparse LiDAR depth (the same metrics prefixed `lidar_`,
# each from its own alignment fit); geometry metrics need dense GT, so the
# LiDAR pass reports only these.
LIDAR_GT_TASKS = frozenset({"MonoDepth", "VideoDepth"})
LIDAR_DEPTH_SCALE = 0.01  # The lidar_depth PNGs are in centimetres.
LIDAR_METRIC_EXCLUDE_PREFIXES = ("point_", "mae_normal", "mean_normal",
                                 "median_normal", "normal_")


@dataclasses.dataclass(frozen=True, kw_only=True)
class DepthPairData:
    """One GT frame and its prediction, loaded and masked for scoring.

    Attributes:
        gt_path: GT depth file.
        pred_path: Prediction file that was read (depth, raw disparity or
            raw log depth, depending on the protocol).
        ground_truth: GT depth in metres.
        prediction: Prediction resized to the GT resolution.
        valid_mask: Pixels that are scored.
        intrinsics: 3x3 intrinsics at GT resolution, or None if unknown.
    """
    gt_path: pathlib.Path
    pred_path: pathlib.Path
    ground_truth: np.ndarray
    prediction: np.ndarray
    valid_mask: np.ndarray
    intrinsics: np.ndarray | None = None

    @property
    def num_valid_pixels(self) -> int:
        """The number of scored pixels."""
        return int(self.valid_mask.sum())


# Per-frame artifacts of the aligned evaluation (--error_artifact_dir), one
# file per GT frame, named by its stem:
#
#   error_maps/<stem>.npy            float16 absolute relative error
#                                    |aligned - gt| / gt on evaluated pixels,
#                                    else NaN;
#   error_visualizations/<stem>.png  that error on one fixed scale for every
#                                    method and frame (ERROR_VIS_*);
#   aligned_depths/<stem>.npy        float16 aligned (and clipped)
#                                    prediction, only with
#                                    --save_aligned_depth: it is
#                                    prediction * scale + shift, and
#                                    frame_results records scale and shift.
#
# Maps are at GT resolution, downsampled (nearest) so the longer side is at
# most max_side (ETH3D GT is 6048x4032).
ERROR_MAP_DIR = "error_maps"
ERROR_VIS_DIR = "error_visualizations"
ALIGNED_DEPTH_DIR = "aligned_depths"
ERROR_VIS_CMAP = "turbo"
# The delta < 1.25 boundary; larger errors saturate.
ERROR_VIS_MAX_ABSREL = 0.25
ERROR_VIS_INVALID_RGB = (128, 128, 128)  # Pixels without valid GT.
FLOAT16_MAX = float(np.finfo(np.float16).max)


@dataclasses.dataclass(kw_only=True)
class DepthErrorArtifactWriter:
    """Writes each evaluated frame's error maps and collects frame records.

    Attributes:
        root: Output directory of the maps.
        max_side: Longer side of the saved maps; 0 keeps the GT size.
        save_aligned_depth: Also save each frame's aligned prediction.
        records: One record per written frame: files, alignment, metrics.
    """

    root: pathlib.Path
    max_side: int = 1024
    save_aligned_depth: bool = False

    def __post_init__(self) -> None:
        self.records: list[dict] = []

    def describe(self) -> dict:
        """Returns the layout and visualization scale, for the scene result."""
        return {
            "root":
                str(self.root),
            "error_maps":
                ERROR_MAP_DIR,
            "error_visualizations":
                ERROR_VIS_DIR,
            "aligned_depths":
                ALIGNED_DEPTH_DIR if self.save_aligned_depth else None,
            "error":
                "absolute relative error |aligned - gt| / gt",
            "visualization": {
                "cmap": ERROR_VIS_CMAP,
                "vmin": 0.0,
                "vmax": ERROR_VIS_MAX_ABSREL,
                "invalid_rgb": list(ERROR_VIS_INVALID_RGB)
            },
            "max_side":
                self.max_side,
        }

    def write(self, pair: DepthPairData, aligned: np.ndarray, scale: float,
              shift: float, metrics: Mapping[str, float]) -> None:
        """Saves one frame's maps and records its alignment and metrics.

        Args:
            pair: The frame.
            aligned: Aligned (and clipped) prediction at GT resolution.
            scale: Alignment scale.
            shift: Alignment shift.
            metrics: The frame's metrics.
        """
        import matplotlib  # pylint: disable=import-outside-toplevel
        from PIL import Image  # pylint: disable=import-outside-toplevel

        stem = pair.gt_path.stem
        valid = pair.valid_mask
        error = np.full(pair.ground_truth.shape, np.nan, dtype=np.float32)
        error[valid] = np.abs(aligned[valid] - pair.ground_truth[valid]
                             ) / pair.ground_truth[valid]
        error = self._downsample(error)
        self._save_float16(ERROR_MAP_DIR, stem, error)
        normalized = np.clip(
            np.nan_to_num(error, nan=0.0) / ERROR_VIS_MAX_ABSREL, 0.0, 1.0)
        rgb = (matplotlib.colormaps[ERROR_VIS_CMAP](normalized)[..., :3] *
               255.0).astype(np.uint8)
        rgb[~np.isfinite(error)] = ERROR_VIS_INVALID_RGB
        (self.root / ERROR_VIS_DIR).mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(self.root / ERROR_VIS_DIR / f"{stem}.png")
        if self.save_aligned_depth:
            self._save_float16(ALIGNED_DEPTH_DIR, stem,
                               self._downsample(aligned.astype(np.float32)))
        self.records.append({
            "gt": pair.gt_path.name,
            "prediction": pair.pred_path.name,
            "scale": float(scale),
            "shift": float(shift),
            "num_valid_pixels": pair.num_valid_pixels,
            "metrics": dict(metrics),
        })

    def _downsample(self, array: np.ndarray) -> np.ndarray:
        height, width = array.shape
        if self.max_side > 0:
            factor = self.max_side / float(max(height, width))
        else:
            factor = 1.0
        if factor >= 1.0:
            return array
        return _resize_to_match(
            array,
            (max(1, round(height * factor)), max(1, round(width * factor))),
            "nearest")

    def _save_float16(self, dirname: str, stem: str, array: np.ndarray) -> None:
        (self.root / dirname).mkdir(parents=True, exist_ok=True)
        np.save(self.root / dirname / f"{stem}.npy",
                np.clip(array, -FLOAT16_MAX, FLOAT16_MAX).astype(np.float16))


# Bonn RGB-D Dynamic depth PNGs: uint16, metres x 5000 (the TUM RGB-D
# convention).
BONN_DEPTH_SCALE = 1.0 / 5000.0
ETH3D_RAW_DEPTH_SHAPE = (4032, 6048)
ETH3D_RAW_DEPTH_BYTES = ETH3D_RAW_DEPTH_SHAPE[0] * ETH3D_RAW_DEPTH_SHAPE[1] * 4
SINTEL_INVALID_DEPTH_THRESHOLD = 1e9
# When a frame has two files, the first wins (.npz: the current export
# format).
DEPTH_EXTENSION_PRIORITY = {
    ".npz": 0,
    ".npy": 1,
    ".pfm": 2,
    ".dpt": 3,
    ".exr": 4,
    ".tif": 5,
    ".tiff": 5,
    ".png": 6,
    ".jpg": 7,
    ".jpeg": 7,
}


def _read_pfm(path: pathlib.Path) -> np.ndarray:
    with pathlib.Path(path).open("rb") as file:
        header = file.readline().decode("ascii").rstrip()
        if header not in {"PF", "Pf"}:
            raise ValueError(f"Invalid PFM header in {path}")
        color = header == "PF"
        dims_line = file.readline().decode("ascii").strip()
        while dims_line.startswith("#"):
            dims_line = file.readline().decode("ascii").strip()
        width, height = (int(value) for value in dims_line.split())
        scale = float(file.readline().decode("ascii").strip())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width)
    return np.flipud(data.reshape(shape))


def _read_sintel_dpt(path: pathlib.Path) -> np.ndarray:
    with pathlib.Path(path).open("rb") as file:
        tag = np.fromfile(file, np.float32, count=1)
        if tag.size != 1 or not np.isclose(float(tag[0]), 202021.25):
            raise ValueError(f"Invalid Sintel DPT tag in {path}")
        width = np.fromfile(file, np.int32, count=1)
        height = np.fromfile(file, np.int32, count=1)
        if width.size != 1 or height.size != 1:
            raise ValueError(f"Invalid Sintel DPT dimensions in {path}")
        width_int = int(width[0])
        height_int = int(height[0])
        data = np.fromfile(file, np.float32, count=width_int * height_int)
    if data.size != width_int * height_int:
        raise ValueError(f"Unexpected Sintel DPT payload size in {path}")
    depth = data.reshape((height_int, width_int))
    depth[depth >= SINTEL_INVALID_DEPTH_THRESHOLD] = np.nan
    return depth


def _read_image_depth(path: pathlib.Path) -> np.ndarray:
    try:
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not read depth image {path}")
        return image
    except ImportError:
        from PIL import Image

        return np.asarray(Image.open(path))


def _read_eth3d_raw_depth(path: pathlib.Path) -> np.ndarray:
    if path.stat().st_size != ETH3D_RAW_DEPTH_BYTES:
        raise ValueError(f"Unsupported JPEG depth payload in {path}")
    return np.fromfile(path, dtype=np.float32).reshape(ETH3D_RAW_DEPTH_SHAPE)


def load_depth_map(path: pathlib.Path, scale: float = 1.0) -> np.ndarray:
    """Reads a depth map file as float32, times `scale`.

    Reads .npy, .npz (the "depth", "pred", "prediction" or "arr_0" array,
    else the first), .pfm, Sintel .dpt and image files; a .jpg/.jpeg that is
    not an image is read as an ETH3D raw float depth map. Multi-channel maps
    keep their first channel.

    Args:
        path: The depth file.
        scale: Factor from file values to depth.

    Returns:
        The (H, W) depth map.

    Raises:
        ValueError: The extension or the file content is not supported.
    """
    path = pathlib.Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        depth = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as data:
            for key in ("depth", "pred", "prediction", "arr_0"):
                if key in data:
                    depth = data[key]
                    break
            else:
                depth = data[data.files[0]]
    elif suffix == ".pfm":
        depth = _read_pfm(path)
    elif suffix == ".dpt":
        depth = _read_sintel_dpt(path)
    elif suffix in {".png", ".tif", ".tiff", ".exr"}:
        depth = _read_image_depth(path)
    elif suffix in {".jpg", ".jpeg"}:
        try:
            depth = _read_image_depth(path)
        except Exception as image_exc:  # pylint: disable=broad-exception-caught
            try:
                depth = _read_eth3d_raw_depth(path)
            except ValueError as raw_exc:
                raise image_exc from raw_exc
    else:
        raise ValueError(f"Unsupported depth file extension: {path}")

    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth * float(scale)


def list_depth_files_in_directory(
        depth_dir: pathlib.Path) -> list[pathlib.Path]:
    """Returns every depth file under `depth_dir`, recursively, sorted."""
    depth_dir = pathlib.Path(depth_dir)
    if not depth_dir.is_dir():
        return []
    return sorted(path for path in depth_dir.rglob("*") if path.is_file() and
                  path.suffix.lower() in prediction_artifacts.DEPTH_EXTENSIONS)


def _materialized_manifest_path(
        scene_info: canonical_dataset.BenchmarkSceneInfo) -> pathlib.Path:
    return scene_info.root_path / "materialized_full.json"


def _load_materialized_manifest(
        scene_info: canonical_dataset.BenchmarkSceneInfo) -> dict | None:
    manifest_path = _materialized_manifest_path(scene_info)
    if scene_info.task != "MonoDepth" or not manifest_path.is_file():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _resolve_manifest_path(manifest_path: pathlib.Path,
                           path_value: str | None) -> pathlib.Path | None:
    if not path_value:
        return None
    path = pathlib.Path(path_value)
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def _canonical_existing_path(path: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(path).expanduser().resolve(strict=False)


def list_scene_gt_depth_files(scene_info: canonical_dataset.BenchmarkSceneInfo,
                              gt_source: str = "dense") -> list[pathlib.Path]:
    """Returns a scene's canonical GT depth files.

    The MonoDepth materialization manifest's records win when present.

    Args:
        scene_info: The scene.
        gt_source: "dense" for the dense GT depth; "lidar" for the sparse
            LiDAR depth (the records' `lidar_depth`, else lidar_depths/).

    Returns:
        The GT depth files.
    """
    manifest_path = _materialized_manifest_path(scene_info)
    manifest = _load_materialized_manifest(scene_info)
    if gt_source == "lidar":
        if manifest is not None:
            paths = [
                _resolve_manifest_path(manifest_path, record.get("lidar_depth"))
                for record in manifest.get("records", [])
                if record.get("lidar_depth")
            ]
            return [
                path for path in paths if path is not None and path.is_file()
            ]
        return list_depth_files_in_directory(
            pathlib.Path(scene_info.root_path) / "lidar_depths")
    if manifest is not None:
        paths = [
            _resolve_manifest_path(manifest_path, record.get("depth"))
            for record in manifest.get("records", [])
            if record.get("depth")
        ]
        existing = [
            path for path in paths if path.is_file() and
            path.suffix.lower() in prediction_artifacts.DEPTH_EXTENSIONS
        ]
        if existing:
            return existing
        if scene_info.depth_gt_path is not None:
            return list_depth_files_in_directory(scene_info.depth_gt_path)
        return []

    if scene_info.depth_gt_path is None:
        return []
    return list_depth_files_in_directory(scene_info.depth_gt_path)


def materialized_mask_paths_by_depth(
    scene_info: canonical_dataset.BenchmarkSceneInfo
) -> dict[pathlib.Path, pathlib.Path]:
    """Returns the per-frame masks of the MonoDepth materialization manifest.

    Args:
        scene_info: The scene.

    Returns:
        Mask file by resolved GT depth path; empty without a manifest.
    """
    manifest_path = _materialized_manifest_path(scene_info)
    manifest = _load_materialized_manifest(scene_info)
    if manifest is None:
        return {}

    mask_by_depth: dict[pathlib.Path, pathlib.Path] = {}
    for record in manifest.get("records", []):
        depth_path = _resolve_manifest_path(manifest_path, record.get("depth"))
        mask_path = _resolve_manifest_path(manifest_path, record.get("mask"))
        if (depth_path is not None and not depth_path.is_file() and
                scene_info.depth_gt_path is not None):
            depth_path = scene_info.depth_gt_path / depth_path.name
        if (mask_path is not None and not mask_path.is_file() and
                scene_info.mask_path is not None):
            mask_path = scene_info.mask_path / mask_path.name
        if depth_path is None or mask_path is None:
            continue
        if not depth_path.is_file() or not mask_path.is_file():
            continue
        mask_by_depth[_canonical_existing_path(depth_path)] = mask_path
    return mask_by_depth


def _as_intrinsics_matrix(value: object) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        for key in ("intrinsics", "K", "matrix", "camera_matrix",
                    "depth_intrinsics"):
            intrinsic = _as_intrinsics_matrix(value.get(key))
            if intrinsic is not None:
                return intrinsic
        if {"fx", "fy", "cx", "cy"} <= set(value):
            return np.array(
                [
                    [float(value["fx"]), 0.0,
                     float(value["cx"])],
                    [0.0, float(value["fy"]),
                     float(value["cy"])],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
        return None

    array = np.asarray(value, dtype=np.float64)
    if array.shape == (3, 3):
        return array
    if array.shape == (4, 4):
        return array[:3, :3]
    if array.shape == (4,):
        fx, fy, cx, cy = array
        return np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    return None


def _read_image_size(path: pathlib.Path) -> tuple[int, int] | None:
    if not path.is_file():
        return None
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def _scale_intrinsics_to_shape(
    intrinsics: np.ndarray,
    source_size: tuple[int, int] | None,
    target_shape: tuple[int, int],
) -> np.ndarray:
    scaled = np.asarray(intrinsics, dtype=np.float64).copy()
    if source_size is None:
        return scaled
    source_width, source_height = source_size
    target_height, target_width = target_shape
    if source_width <= 0 or source_height <= 0:
        return scaled
    scaled[0, :] *= float(target_width) / float(source_width)
    scaled[1, :] *= float(target_height) / float(source_height)
    return scaled


def materialized_intrinsics_by_depth(
    scene_info: canonical_dataset.BenchmarkSceneInfo
) -> dict[pathlib.Path, np.ndarray]:
    """Returns the per-frame intrinsics of a MonoDepth materialization manifest.

    Args:
        scene_info: The scene.

    Returns:
        Intrinsics scaled to the GT depth resolution, by resolved GT depth
        path; empty without a manifest.
    """
    manifest_path = _materialized_manifest_path(scene_info)
    manifest = _load_materialized_manifest(scene_info)
    if manifest is None:
        return {}

    intrinsics_by_depth: dict[pathlib.Path, np.ndarray] = {}
    for record in manifest.get("records", []):
        intrinsics = _as_intrinsics_matrix(record.get("intrinsics"))
        if intrinsics is None:
            continue
        depth_path = _resolve_manifest_path(manifest_path, record.get("depth"))
        if (depth_path is not None and not depth_path.is_file() and
                scene_info.depth_gt_path is not None):
            depth_path = scene_info.depth_gt_path / depth_path.name
        if depth_path is None or not depth_path.is_file():
            continue
        image_path = _resolve_manifest_path(manifest_path, record.get("image"))
        if image_path is not None:
            source_size = _read_image_size(image_path)
        else:
            source_size = None
        target_shape = load_depth_map(depth_path).shape
        scaled = _scale_intrinsics_to_shape(
            intrinsics,
            source_size,
            target_shape,
        )
        intrinsics_by_depth[_canonical_existing_path(depth_path)] = scaled
    return intrinsics_by_depth


def _sample_json_depth_matches(scene_root: pathlib.Path, payload: Mapping,
                               gt_path: pathlib.Path) -> bool:
    depth_record = payload.get("depth")
    if not isinstance(depth_record, Mapping):
        return True
    canonical_path = depth_record.get("canonical_path")
    if not canonical_path:
        return True
    return _canonical_existing_path(
        scene_root / str(canonical_path)) == _canonical_existing_path(gt_path)


def _sample_json_image_size(scene_root: pathlib.Path, payload: Mapping,
                            view_index: int) -> tuple[int, int] | None:
    images = payload.get("images")
    if not isinstance(images, list) or view_index >= len(images):
        return None
    record = images[view_index]
    if not isinstance(record, Mapping):
        return None
    canonical_path = record.get("canonical_path")
    if not canonical_path:
        return None
    return _read_image_size(scene_root / str(canonical_path))


def _sample_json_intrinsics(
    scene_info: canonical_dataset.BenchmarkSceneInfo,
    gt_path: pathlib.Path,
    target_shape: tuple[int, int],
) -> np.ndarray | None:
    """Returns the key view's intrinsics from the scene's sample.json."""
    sample_path = scene_info.root_path / "sample.json"
    if sample_path.is_file():
        payload = json.loads(sample_path.read_text(encoding="utf-8"))
        intrinsics_list = payload.get("intrinsics")
        if isinstance(intrinsics_list,
                      list) and intrinsics_list and _sample_json_depth_matches(
                          scene_info.root_path,
                          payload,
                          gt_path,
                      ):
            view_index = int(payload.get("keyview_idx", 0))
            if view_index < len(intrinsics_list):
                intrinsics = _as_intrinsics_matrix(intrinsics_list[view_index])
                if intrinsics is not None:
                    source_size = _sample_json_image_size(
                        scene_info.root_path, payload, view_index)
                    return _scale_intrinsics_to_shape(intrinsics, source_size,
                                                      target_shape)
    return None


def resolve_depth_intrinsics(
    scene_info: canonical_dataset.BenchmarkSceneInfo,
    gt_path: pathlib.Path,
    target_shape: tuple[int, int],
    manifest_intrinsics: Mapping[pathlib.Path, np.ndarray] | None = None,
) -> np.ndarray | None:
    """Returns a GT frame's intrinsics at the GT depth resolution.

    Sources, first match wins: the materialization manifest, the scene's
    sample.json, the scene metadata, and the NYUv2 intrinsics for Marigold's
    NYUv2 subset.

    Args:
        scene_info: The scene.
        gt_path: The GT depth file.
        target_shape: (height, width) of the GT depth map.
        manifest_intrinsics: `materialized_intrinsics_by_depth` of the scene.

    Returns:
        The 3x3 intrinsics, or None if no source has them.
    """
    if manifest_intrinsics:
        intrinsics = manifest_intrinsics.get(_canonical_existing_path(gt_path))
        if intrinsics is not None:
            return intrinsics

    intrinsics = _sample_json_intrinsics(scene_info, gt_path, target_shape)
    if intrinsics is not None:
        return intrinsics

    if scene_info.metadata:
        for key in ("depth_intrinsics", "intrinsics", "K", "camera_matrix"):
            intrinsics = _as_intrinsics_matrix(scene_info.metadata.get(key))
            if intrinsics is not None:
                return _scale_intrinsics_to_shape(intrinsics, None,
                                                  target_shape)

    if (scene_info.task == "MonoDepth" and
            scene_info.dataset.lower() == "marigold" and
            scene_info.scene.lower() in {"nyuv2", "nyu-v2"}):
        fallback = np.array(
            [
                [518.8579011745018, 0.0, 325.58244941119034],
                [0.0, 519.4696111212748, 253.73616633400462],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return _scale_intrinsics_to_shape(fallback, (640, 480), target_shape)
    return None


def _depth_key(path: pathlib.Path) -> str:
    stem = path.stem.lower()
    for suffix in (".depth.proj", ".depth", ".color"):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
    for prefix in ("depth_map_", "depth_", "disp_", "frame_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    if stem.endswith("_depth"):
        stem = stem[:-len("_depth")]
    return stem


def _depth_sort_key(path: pathlib.Path) -> tuple[int, str]:
    return (DEPTH_EXTENSION_PRIORITY.get(path.suffix.lower(), 99), str(path))


def _depth_files_by_key(
        files: Iterable[pathlib.Path]) -> dict[str, pathlib.Path]:
    selected: dict[str, pathlib.Path] = {}
    for path in files:
        key = _depth_key(path)
        if key not in selected or _depth_sort_key(path) < _depth_sort_key(
                selected[key]):
            selected[key] = path
    return selected


def gt_excluded_frame_keys(
        scene_info: canonical_dataset.BenchmarkSceneInfo) -> set[str]:
    """Returns the depth keys of the frames whose GT a manual review left out.

    These are the `gt_excluded` entries of the scene's frame_manifest.json,
    as SpatialArena's VideoDepth clips carry them: their images stay model
    input, but they are not scored, nor reported as predictions without GT.

    Args:
        scene_info: The scene.

    Returns:
        The excluded frames' depth keys.
    """
    manifest_path = pathlib.Path(scene_info.root_path) / "frame_manifest.json"
    if not manifest_path.is_file():
        return set()
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if isinstance(manifest, list):
        entries = manifest
    else:
        entries = manifest.get("frames", [])
    return {
        _depth_key(pathlib.Path(entry["rgb"]))
        for entry in entries
        if entry.get("gt_excluded") and entry.get("rgb")
    }


def pair_depth_files(
    gt_files: Iterable[pathlib.Path],
    pred_files: Iterable[pathlib.Path],
) -> tuple[list[tuple[pathlib.Path, pathlib.Path]], list[str], list[str], bool]:
    """Pairs GT and predicted depth files by their depth key.

    Without a common key, equally many files are paired in sorted order.

    Args:
        gt_files: GT depth files.
        pred_files: Predicted depth files.

    Returns:
        (pairs, keys of predictions without GT, keys of GT frames without a
        prediction, whether the sorted-order fallback was used).
    """
    gt_by_key = _depth_files_by_key(gt_files)
    pred_by_key = _depth_files_by_key(pred_files)
    common_keys = sorted(set(gt_by_key) & set(pred_by_key))
    if common_keys:
        pairs = [(gt_by_key[key], pred_by_key[key]) for key in common_keys]
        missing_gt = sorted(set(pred_by_key) - set(gt_by_key))
        missing_pred = sorted(set(gt_by_key) - set(pred_by_key))
        return pairs, missing_gt, missing_pred, False

    gt_sorted = sorted(gt_by_key.values())
    pred_sorted = sorted(pred_by_key.values())
    if gt_sorted and len(gt_sorted) == len(pred_sorted):
        return list(zip(gt_sorted, pred_sorted)), [], [], True
    return [], sorted(pred_by_key), sorted(gt_by_key), False


def _timestamp_depth_files(
        files: Iterable[pathlib.Path]) -> list[tuple[float, pathlib.Path]]:
    timestamped = []
    for path in files:
        try:
            timestamped.append((float(_depth_key(path)), path))
        except ValueError:
            continue
    return sorted(timestamped)


def pair_depth_files_by_nearest_timestamp(
    gt_files: Iterable[pathlib.Path],
    pred_files: Iterable[pathlib.Path],
    max_delta_seconds: float = 0.25,
) -> tuple[list[tuple[pathlib.Path, pathlib.Path]], list[str], list[str]]:
    """Pairs each prediction with the nearest unused GT frame by timestamp.

    The depth keys are timestamps in seconds (Bonn RGB-D Dynamic).

    Args:
        gt_files: GT depth files.
        pred_files: Predicted depth files.
        max_delta_seconds: Largest accepted timestamp difference.

    Returns:
        (pairs, keys of unmatched predictions, keys of unmatched GT frames).
    """
    gt_timestamped = _timestamp_depth_files(gt_files)
    pred_timestamped = _timestamp_depth_files(pred_files)
    if not gt_timestamped or not pred_timestamped:
        unmatched_keys = [_depth_key(path) for path in pred_files]
        missing_keys = [_depth_key(path) for path in gt_files]
        return [], unmatched_keys, missing_keys

    used_gt_indices: set[int] = set()
    pairs: list[tuple[pathlib.Path, pathlib.Path]] = []
    unmatched_pred: list[str] = []

    for pred_time, pred_path in pred_timestamped:
        candidates = [(abs(gt_time - pred_time), index, gt_path)
                      for index, (gt_time, gt_path) in enumerate(gt_timestamped)
                      if index not in used_gt_indices]
        if not candidates:
            unmatched_pred.append(_depth_key(pred_path))
            continue
        delta, index, gt_path = min(candidates, key=lambda item: item[0])
        if delta > max_delta_seconds:
            unmatched_pred.append(_depth_key(pred_path))
            continue
        used_gt_indices.add(index)
        pairs.append((gt_path, pred_path))

    missing_pred = [
        _depth_key(gt_path)
        for index, (_, gt_path) in enumerate(gt_timestamped)
        if index not in used_gt_indices
    ]
    return pairs, unmatched_pred, missing_pred


def _integer_depth_files(
        files: Iterable[pathlib.Path]) -> list[tuple[int, pathlib.Path]]:
    indexed = []
    for path in files:
        try:
            indexed.append((int(_depth_key(path)), path))
        except ValueError:
            continue
    return sorted(indexed)


def pair_depth_files_by_integer_offset(
    gt_files: Iterable[pathlib.Path],
    pred_files: Iterable[pathlib.Path],
    gt_index_offset: int,
) -> tuple[list[tuple[pathlib.Path, pathlib.Path]], list[str], list[str]]:
    """Pairs prediction index i with GT index i + `gt_index_offset`.

    The depth keys are frame indices (Tanks and Temples).

    Args:
        gt_files: GT depth files.
        pred_files: Predicted depth files.
        gt_index_offset: GT index minus prediction index.

    Returns:
        (pairs, keys of unmatched predictions, keys of unmatched GT frames).
    """
    gt_indexed = _integer_depth_files(gt_files)
    pred_indexed = _integer_depth_files(pred_files)
    if not gt_indexed or not pred_indexed:
        unmatched_keys = [_depth_key(path) for path in pred_files]
        missing_keys = [_depth_key(path) for path in gt_files]
        return [], unmatched_keys, missing_keys

    gt_by_index = {index: path for index, path in gt_indexed}
    pairs: list[tuple[pathlib.Path, pathlib.Path]] = []
    unmatched_pred: list[str] = []
    used_gt_indices: set[int] = set()
    for pred_index, pred_path in pred_indexed:
        gt_index = pred_index + gt_index_offset
        gt_path = gt_by_index.get(gt_index)
        if gt_path is None:
            unmatched_pred.append(_depth_key(pred_path))
            continue
        used_gt_indices.add(gt_index)
        pairs.append((gt_path, pred_path))

    missing_pred = [
        _depth_key(gt_path)
        for gt_index, gt_path in gt_indexed
        if gt_index not in used_gt_indices
    ]
    return pairs, unmatched_pred, missing_pred


def _resize_to_match(
    prediction: np.ndarray,
    target_shape: tuple[int, int],
    interpolation: str = "linear",
) -> np.ndarray:
    if prediction.shape == target_shape:
        return prediction
    interpolation = interpolation.lower()
    try:
        import cv2
    except ImportError:
        from PIL import Image

        resampling = {
            "nearest": Image.Resampling.NEAREST,
            "linear": Image.Resampling.BILINEAR,
            "bilinear": Image.Resampling.BILINEAR,
            "cubic": Image.Resampling.BICUBIC,
            "bicubic": Image.Resampling.BICUBIC,
        }.get(interpolation)
        if resampling is None:
            raise ValueError(
                f"Unknown depth resize interpolation: {interpolation}"
            ) from None
        height, width = target_shape
        image = Image.fromarray(prediction.astype(np.float32, copy=False),
                                mode="F")
        resized = image.resize(
            (width, height),
            resampling,
        )
        return np.asarray(resized, dtype=np.float32)
    height, width = target_shape
    interpolation_flag = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "bilinear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "bicubic": cv2.INTER_CUBIC,
    }.get(interpolation)
    if interpolation_flag is None:
        raise ValueError(f"Unknown depth resize interpolation: {interpolation}")
    return cv2.resize(prediction, (width, height),
                      interpolation=interpolation_flag)


def align_prediction_depth(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    valid_mask: np.ndarray,
    alignment: str,
    max_resolution: int | None = None,
) -> np.ndarray:
    """Returns a frame's prediction aligned to its GT (see fit_frame_alignment).
    """
    scale, shift = fit_frame_alignment(prediction, ground_truth, valid_mask,
                                       alignment, max_resolution)
    return apply_depth_alignment(prediction, scale, shift)


def fit_frame_alignment(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    valid_mask: np.ndarray,
    alignment: str,
    max_resolution: int | None = None,
) -> tuple[float, float]:
    """Fits the (scale, shift) that aligns one frame's prediction to its GT.

    Args:
        prediction: Predicted depth at GT resolution.
        ground_truth: GT depth.
        valid_mask: Pixels the fit uses.
        alignment: Alignment mode (see `fit_depth_alignment`).
        max_resolution: Fit at most at this resolution (nearest
            downsampling), or None for full resolution.

    Returns:
        (scale, shift).
    """
    fit_prediction = prediction
    fit_ground_truth = ground_truth
    fit_mask = valid_mask
    if max_resolution is not None:
        fit_prediction, fit_ground_truth, fit_mask = (
            _downsample_alignment_inputs(
                prediction,
                ground_truth,
                valid_mask,
                max_resolution=max_resolution,
            ))
    return fit_depth_alignment(
        fit_prediction[fit_mask],
        fit_ground_truth[fit_mask],
        alignment,
    )


def _downsample_alignment_inputs(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    valid_mask: np.ndarray,
    max_resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = prediction.shape[-2:]
    scale_factor = min(
        float(max_resolution) / float(height),
        float(max_resolution) / float(width))
    if scale_factor >= 1.0:
        return prediction, ground_truth, valid_mask
    new_height = max(1, int(height * scale_factor))
    new_width = max(1, int(width * scale_factor))
    try:
        import cv2
    except ImportError:
        from PIL import Image

        target_size = (new_width, new_height)
        prediction_image = Image.fromarray(prediction.astype(np.float32,
                                                             copy=False),
                                           mode="F")
        resized_prediction = np.asarray(
            prediction_image.resize(
                target_size,
                Image.Resampling.NEAREST,
            ),
            dtype=np.float32,
        )
        ground_truth_image = Image.fromarray(ground_truth.astype(np.float32,
                                                                 copy=False),
                                             mode="F")
        resized_ground_truth = np.asarray(
            ground_truth_image.resize(
                target_size,
                Image.Resampling.NEAREST,
            ),
            dtype=np.float32,
        )
        mask_image = Image.fromarray(valid_mask.astype(np.uint8))
        resized_mask = np.asarray(
            mask_image.resize(
                target_size,
                Image.Resampling.NEAREST,
            ),
            dtype=np.uint8,
        ).astype(bool)
        return (resized_prediction, resized_ground_truth, resized_mask)
    target_size = (new_width, new_height)
    return (
        cv2.resize(prediction, target_size, interpolation=cv2.INTER_NEAREST),
        cv2.resize(ground_truth, target_size, interpolation=cv2.INTER_NEAREST),
        cv2.resize(valid_mask.astype(np.uint8),
                   target_size,
                   interpolation=cv2.INTER_NEAREST).astype(bool),
    )


def fit_depth_alignment(
    pred_values: np.ndarray,
    gt_values: np.ndarray,
    alignment: str,
) -> tuple[float, float]:
    """Fits the (scale, shift) that maps predicted values onto GT values.

    Args:
        pred_values: Predicted depths of the valid pixels.
        gt_values: GT depths of the same pixels.
        alignment: "none", "median" (median ratio), "pi3_median_scale" /
            "scale" (robust scale only), "least_squares" (scale and shift),
            or "pi3_scale_shift_lad" / "scale_shift_lad" / "scale&shift"
            (absolute-error scale and shift).

    Returns:
        (scale, shift); (1.0, 0.0) without values.

    Raises:
        ValueError: The alignment mode is unknown.
    """
    if alignment == "none":
        return 1.0, 0.0

    if pred_values.size == 0:
        return 1.0, 0.0

    if alignment == "median":
        pred_median = np.median(pred_values)
        if abs(pred_median) < 1e-12:
            return 1.0, 0.0
        return float(np.median(gt_values) / pred_median), 0.0

    if alignment in {"pi3_median_scale", "scale"}:
        # Matches the scale-only robust fitting used by Pi3/CUT3R evaluation
        # (github.com/yyfz/Pi3, evaluation branch, utils/depth.py).
        scale = float(np.nanmean(gt_values) / np.nanmean(pred_values))
        if not np.isfinite(scale):
            return 1.0, 0.0
        for _ in range(10):
            residuals = scale * pred_values - gt_values
            weights = 1.0 / (np.abs(residuals) + 1e-8)
            denominator = np.sum(weights * pred_values**2)
            if denominator <= 1e-12:
                break
            scale = float(
                np.sum(weights * pred_values * gt_values) / denominator)
        return max(scale, 1e-3), 0.0

    if alignment == "least_squares":
        matrix = np.stack([pred_values, np.ones_like(pred_values)], axis=1)
        scale, shift = np.linalg.lstsq(matrix, gt_values, rcond=None)[0]
        return float(scale), float(shift)

    if alignment in {"pi3_scale_shift_lad", "scale_shift_lad", "scale&shift"}:
        # LAD2 follows the Adam-based absolute-error fit in Pi3/CUT3R
        # video-depth evaluation. The caller decides whether values are one
        # frame or a whole sequence; VideoDepth passes the full sequence.
        import torch

        pred_tensor = torch.from_numpy(np.asarray(pred_values,
                                                  dtype=np.float32))
        gt_tensor = torch.from_numpy(np.asarray(gt_values, dtype=np.float32))
        pred_median = torch.median(pred_tensor)
        if torch.abs(pred_median) <= 1e-12:
            return 1.0, 0.0
        scale_init = (torch.median(gt_tensor) / pred_median).item()
        scale, shift = depth_tools.absolute_value_scaling2(
            pred_tensor,
            gt_tensor,
            s_init=scale_init,
        )
        return float(scale), float(shift)

    raise ValueError(f"Unknown depth alignment mode: {alignment}")


def apply_depth_alignment(
    prediction: np.ndarray,
    scale: float,
    shift: float,
) -> np.ndarray:
    """Returns prediction * scale + shift."""
    return prediction * float(scale) + float(shift)


# Standard-protocol MonoDepth limits of the Marigold subsets, used when a
# scene's metadata.json entry does not declare them: (min_depth, max_depth,
# depth_scale, alignment_max_resolution, filter_gt_depth_range).
MARIGOLD_MONODEPTH_LIMITS = {
    "nyuv2": (1e-3, 10.0, 1e-3, None, True),
    "kitti": (1e-5, 80.0, 1.0 / 256.0, None, True),
    "eth3d": (1e-5, None, 1.0, 1024, True),
    "diode": (1e-5, 80.0, 1.0, None, False),
    "scannet": (1e-3, 10.0, 1e-3, None, True),
}
MONODEPTH_LIMIT_KEYS = ("min_depth", "max_depth", "depth_scale",
                        "alignment_max_resolution", "filter_gt_depth_range")


def monodepth_scene_limits(
        scene_info: canonical_dataset.BenchmarkSceneInfo) -> dict | None:
    """Returns a MonoDepth scene's depth limits for the standard protocol.

    A key the scene's metadata.json entry declares wins: `min_depth`,
    `max_depth` (null: no upper bound), `depth_scale` (GT file value to
    metres), `alignment_max_resolution`, `filter_gt_depth_range`. A missing
    key falls back to the Marigold table for Marigold subsets, else to the
    evaluator default.

    Args:
        scene_info: The scene.

    Returns:
        The limits by MONODEPTH_LIMIT_KEYS, or None to keep the config
        defaults.
    """
    defaults = dict(
        zip(MONODEPTH_LIMIT_KEYS,
            (DepthEvaluationConfig.min_depth, None, 1.0, None, True)))
    table = None
    if scene_info.dataset.lower() == "marigold":
        table = MARIGOLD_MONODEPTH_LIMITS.get(scene_info.scene.lower())
        if table is not None:
            defaults = dict(zip(MONODEPTH_LIMIT_KEYS, table))
    metadata = scene_info.metadata or {}
    declared = {
        key: metadata[key] for key in MONODEPTH_LIMIT_KEYS if key in metadata
    }
    if table is None and not declared:
        return None
    limits = dict(defaults)
    limits.update(declared)
    limits["min_depth"] = float(limits["min_depth"])
    if limits["max_depth"] is not None:
        limits["max_depth"] = float(limits["max_depth"])
    limits["depth_scale"] = float(limits["depth_scale"])
    resolution = limits["alignment_max_resolution"]
    if resolution is not None:
        limits["alignment_max_resolution"] = int(resolution)
    limits["filter_gt_depth_range"] = bool(limits["filter_gt_depth_range"])
    return limits


# Temporary port of PR #21 (eval_refactor_hsy, commit d88025f4): GT-only
# depth-alignment protocols (no prediction-dependent pixel exclusion).
GT_ONLY_PROTOCOLS = frozenset({
    "inverse_l1_gt",
    "inverse_huber_irls_gt",
    "inverse_l1_l2init_gt",
    "depth_huber_irls_gt",
    "gt_bounds_direct",
    "gt_bounds_depth_huber_irls",
    # Normalized log depth (raw_log_depths/), see
    # spatialarena/eval/log_depth_gt.py.
    "log_huber_irls_gt",
})


def resolve_depth_config(
    scene_info: canonical_dataset.BenchmarkSceneInfo,
    config: DepthEvaluationConfig,
) -> DepthEvaluationConfig:
    """Resolves a named depth protocol into concrete alignment and ranges.

    Args:
        scene_info: The scene; its task and dataset select the options.
        config: The requested config.

    Returns:
        The resolved config; `config` itself for a non-"standard" alignment.

    Raises:
        ValueError: The protocol is unknown.
    """
    protocol = config.protocol.lower()
    if protocol in GT_ONLY_PROTOCOLS:
        base = resolve_depth_config(
            scene_info, dataclasses.replace(config, protocol="standard"))
        # GT-only scoring reports mae_normal_deg (mean_normal_error) but no
        # point-cloud metrics.
        return dataclasses.replace(base,
                                   protocol=protocol,
                                   resize_interpolation="linear")
    if protocol in {"pi3_mono", "pi3_monodepth"}:
        return resolve_pi3_depth_config(
            scene_info,
            dataclasses.replace(
                config,
                alignment="pi3_median_scale",
                resize_interpolation="cubic",
            ))
    if protocol in {"pi3_video", "pi3_videodepth"}:
        return resolve_pi3_depth_config(
            scene_info,
            dataclasses.replace(
                config,
                alignment="pi3_scale_shift_lad",
                resize_interpolation="cubic",
            ))
    if protocol not in {"standard", "custom"}:
        raise ValueError(
            f"Unknown depth evaluation protocol: {config.protocol}")
    if config.alignment != "standard":
        return config

    task = scene_info.task
    dataset = scene_info.dataset.lower()
    resolved = dataclasses.replace(config, alignment="median")

    if task == "MonoDepth":
        resolved = _standard_monodepth_config(scene_info, resolved)
    elif task == "VideoDepth":
        resolved = _standard_videodepth_config(dataset, resolved)
    elif task == "MVRecon":
        resolved = _standard_mvrecon_config(dataset, resolved)

    return resolved


def _standard_monodepth_config(
    scene_info: canonical_dataset.BenchmarkSceneInfo,
    resolved: DepthEvaluationConfig,
) -> DepthEvaluationConfig:
    """Returns the standard-protocol options of a MonoDepth scene."""
    dataset = scene_info.dataset.lower()
    resolved = dataclasses.replace(resolved, alignment="least_squares")
    limits = monodepth_scene_limits(scene_info)
    if limits is not None:
        resolved = dataclasses.replace(
            resolved,
            min_depth=limits["min_depth"],
            max_depth=limits["max_depth"],
            gt_depth_scale=limits["depth_scale"],
            post_clip_min=limits["min_depth"],
            post_clip_max=limits["max_depth"],
            alignment_max_resolution=limits["alignment_max_resolution"],
            filter_gt_depth_range=limits["filter_gt_depth_range"],
        )
    if dataset == "marigold" and scene_info.scene.lower() == "diode":
        resolved = dataclasses.replace(resolved,
                                       post_clip_min=1e-3,
                                       post_clip_max=None,
                                       post_clip_policy="valid_gt_max")
    return resolved


def _standard_videodepth_config(
    dataset: str,
    resolved: DepthEvaluationConfig,
) -> DepthEvaluationConfig:
    """Returns the standard-protocol options of a VideoDepth dataset."""
    resolved = dataclasses.replace(resolved,
                                   alignment="pi3_scale_shift_lad",
                                   resize_interpolation="cubic")
    if dataset == "bonn":
        # Bonn RGB-D Dynamic stores depth PNGs in the TUM format, metres x
        # 5000 (as MonST3R / CUT3R read them).
        resolved = dataclasses.replace(resolved,
                                       max_depth=10.0,
                                       gt_depth_scale=BONN_DEPTH_SCALE)
    elif dataset == "kitti":
        # KITTI projected depth PNGs are encoded with the standard 256x
        # depth factor.
        resolved = dataclasses.replace(resolved,
                                       max_depth=300.0,
                                       gt_depth_scale=1.0 / 256.0)
    if dataset == "sintel":
        resolved = dataclasses.replace(resolved, max_depth=70.0)
    elif dataset == "tum_dynamics":
        resolved = dataclasses.replace(resolved,
                                       max_depth=10.0,
                                       gt_depth_scale=1.0 / 5000.0)
    elif dataset == "dycheck":
        resolved = dataclasses.replace(resolved,
                                       max_depth=10.0,
                                       gt_depth_scale=1.0)
    elif dataset == spatialarena.SPATIALARENA:
        # GGGS dense depth in metres (.npy); the same 0.1-100 m range as
        # SpatialArena's MonoDepth track.
        resolved = dataclasses.replace(resolved,
                                       min_depth=0.1,
                                       max_depth=100.0,
                                       gt_depth_scale=1.0,
                                       post_clip_min=0.1)
    if resolved.max_depth is not None:
        resolved = dataclasses.replace(resolved,
                                       post_clip_max=resolved.max_depth)
    return resolved


def _standard_mvrecon_config(
    dataset: str,
    resolved: DepthEvaluationConfig,
) -> DepthEvaluationConfig:
    """Returns the standard-protocol options of an MVRecon dataset."""
    # MVRecon depth artifacts are clipped after their dataset protocol.
    resolved = dataclasses.replace(resolved,
                                   post_clip_min=0.1,
                                   post_clip_max=100.0)
    if dataset == "dtu":
        resolved = dataclasses.replace(resolved, gt_depth_scale=1e-3)
    if dataset == "7scenes":
        resolved = dataclasses.replace(resolved, gt_depth_scale=1e-3)
    if dataset == "nrgbd":
        # The depth_filtered/ PNGs use the official codebase's own
        # depth_shift=1000.0 convention (mm -> m), same scale as dtu/7scenes.
        resolved = dataclasses.replace(resolved, gt_depth_scale=1e-3)
    return resolved


def resolve_pi3_depth_config(
    scene_info: canonical_dataset.BenchmarkSceneInfo,
    config: DepthEvaluationConfig,
) -> DepthEvaluationConfig:
    """Resolves the Pi3-style MonoDepth/VideoDepth protocol settings.

    Args:
        scene_info: The scene.
        config: The config with the Pi3 alignment already set.

    Returns:
        The config with the dataset's depth range and scale.
    """
    task = scene_info.task
    dataset = scene_info.dataset.lower()
    scene = scene_info.scene.lower()
    resolved = config

    if task == "MonoDepth":
        if dataset == "marigold":
            if scene in {"nyuv2", "nyu-v2"}:
                resolved = dataclasses.replace(
                    resolved,
                    min_depth=1e-3,
                    max_depth=10.0,
                    gt_depth_scale=1e-3,
                )
            elif scene == "kitti":
                resolved = dataclasses.replace(
                    resolved,
                    min_depth=1e-5,
                    max_depth=80.0,
                    gt_depth_scale=1.0 / 256.0,
                )
            elif scene == "diode":
                resolved = dataclasses.replace(
                    resolved,
                    min_depth=1e-5,
                    max_depth=300.0,
                    filter_gt_depth_range=False,
                )
        if resolved.max_depth is not None:
            resolved = dataclasses.replace(resolved,
                                           post_clip_max=resolved.max_depth)
    elif task == "VideoDepth":
        if dataset == "bonn":
            resolved = dataclasses.replace(resolved,
                                           max_depth=10.0,
                                           gt_depth_scale=BONN_DEPTH_SCALE)
        elif dataset == "kitti":
            resolved = dataclasses.replace(resolved,
                                           max_depth=300.0,
                                           gt_depth_scale=1.0 / 256.0)
        elif dataset == "sintel":
            resolved = dataclasses.replace(resolved, max_depth=70.0)
        if resolved.max_depth is not None:
            resolved = dataclasses.replace(resolved,
                                           post_clip_max=resolved.max_depth)

    return resolved


def clip_prediction_depth(
    prediction: np.ndarray,
    min_depth: float | None,
    max_depth: float | None,
) -> np.ndarray:
    """Clips a depth map to [min_depth, max_depth]; None leaves a side open."""
    if min_depth is None and max_depth is None:
        return prediction
    lower = -np.inf if min_depth is None else float(min_depth)
    upper = np.inf if max_depth is None else float(max_depth)
    return np.clip(prediction, a_min=lower, a_max=upper)


def compute_depth_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    valid_mask: np.ndarray,
    min_depth: float = 1e-6,
) -> dict[str, float]:
    """Returns the depth metrics of a map on its valid pixels.

    See `compute_depth_metrics_from_values`.
    """
    return compute_depth_metrics_from_values(
        prediction[valid_mask],
        ground_truth[valid_mask],
        min_depth=min_depth,
    )


def compute_depth_metrics_from_values(
    prediction_values: np.ndarray,
    ground_truth_values: np.ndarray,
    min_depth: float = 1e-6,
) -> dict[str, float]:
    """Returns the depth metrics of aligned predicted values against GT.

    Args:
        prediction_values: Aligned predicted depths.
        ground_truth_values: GT depths of the same pixels.
        min_depth: Both are floored at this depth.

    Returns:
        AbsRel, SqRel, RMSE, log RMSE, log10, the delta accuracies, inverse
        RMSE and SILog RMSE, and the percent forms of AbsRel and the delta
        accuracies.
    """
    pred = np.maximum(np.asarray(prediction_values, dtype=np.float64),
                      min_depth)
    gt = np.maximum(np.asarray(ground_truth_values, dtype=np.float64),
                    min_depth)
    diff = pred - gt
    log_diff = np.log(pred) - np.log(gt)
    thresh = np.maximum(gt / pred, pred / gt)
    silog_var = np.mean(log_diff**2) - np.mean(log_diff)**2
    absrel = float(np.mean(np.abs(diff) / gt))
    delta103 = float(np.mean(thresh < 1.03))
    delta125 = float(np.mean(thresh < 1.25))

    return {
        "abs_relative_difference": absrel,
        "squared_relative_difference": float(np.mean((diff**2) / gt)),
        "rmse_linear": float(np.sqrt(np.mean(diff**2))),
        "rmse_log": float(np.sqrt(np.mean(log_diff**2))),
        "log10": float(np.mean(np.abs(np.log10(pred) - np.log10(gt)))),
        "delta103_acc": delta103,
        "delta125_acc": delta125,
        "delta125_2_acc": float(np.mean(thresh < 1.25**2)),
        "delta125_3_acc": float(np.mean(thresh < 1.25**3)),
        "i_rmse": float(np.sqrt(np.mean((1.0 / pred - 1.0 / gt)**2))),
        "silog_rmse": float(math.sqrt(max(silog_var, 0.0)) * 100.0),
        "absrel_percent": absrel * 100.0,
        "delta103_percent": delta103 * 100.0,
        "delta125_percent": delta125 * 100.0,
    }


def depth_to_camera_points(
    depth: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Unprojects a depth map to an (H, W, 3) OpenCV camera point map.

    Raises:
        ValueError: A focal length is zero.
    """
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    if abs(fx) < 1e-12 or abs(fy) < 1e-12:
        raise ValueError("Depth intrinsics must have non-zero focal lengths")

    height, width = depth.shape
    yy, xx = np.indices((height, width), dtype=np.float64)
    z = np.asarray(depth, dtype=np.float64)
    x = (xx - cx) * z / fx
    y = (yy - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def point_map_to_normals(
    points: np.ndarray,
    valid_mask: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimates SurGe-style point-map normals from local neighborhoods.

    Reference: SurGe point-map normal MAE forms four local normals around each
    pixel from adjacent finite differences and averages the valid normals.

    Args:
        points: (H, W, 3) point map.
        valid_mask: Pixels with a valid point.
        eps: Smallest norm of a usable normal.

    Returns:
        (unit normals, mask of the pixels that have one).
    """
    normals = np.zeros_like(points, dtype=np.float64)
    normal_mask = np.zeros(points.shape[:2], dtype=bool)
    if points.shape[0] < 2 or points.shape[1] < 2:
        return normals, normal_mask

    points = np.asarray(points, dtype=np.float64)
    valid_mask = np.asarray(valid_mask,
                            dtype=bool) & np.isfinite(points).all(axis=-1)
    accumulated = np.zeros_like(points, dtype=np.float64)
    counts = np.zeros(points.shape[:2], dtype=np.int32)

    def add_local_normal(
        center_slice: tuple[slice, slice],
        first_slice: tuple[slice, slice],
        second_slice: tuple[slice, slice],
    ) -> None:
        center = points[center_slice]
        first = points[first_slice] - center
        second = points[second_slice] - center
        local_valid = (valid_mask[center_slice] & valid_mask[first_slice] &
                       valid_mask[second_slice] &
                       np.isfinite(first).all(axis=-1) &
                       np.isfinite(second).all(axis=-1))
        local_normals = np.cross(first, second)
        norms = np.linalg.norm(local_normals, axis=-1)
        local_valid &= norms > eps
        if not np.any(local_valid):
            return
        local_normals[local_valid] /= norms[local_valid, None]
        accumulated_view = accumulated[center_slice]
        counts_view = counts[center_slice]
        accumulated_view[local_valid] += local_normals[local_valid]
        counts_view[local_valid] += 1

    # Neighbor pairs are ordered around the center so a fronto-parallel OpenCV
    # point map gives +Z normals consistently.
    add_local_normal(
        (slice(1, None), slice(1, None)),
        (slice(1, None), slice(0, -1)),
        (slice(0, -1), slice(1, None)),
    )
    add_local_normal(
        (slice(1, None), slice(0, -1)),
        (slice(0, -1), slice(0, -1)),
        (slice(1, None), slice(1, None)),
    )
    add_local_normal(
        (slice(0, -1), slice(0, -1)),
        (slice(0, -1), slice(1, None)),
        (slice(1, None), slice(0, -1)),
    )
    add_local_normal(
        (slice(0, -1), slice(1, None)),
        (slice(1, None), slice(1, None)),
        (slice(0, -1), slice(0, -1)),
    )

    normal_mask = counts > 0
    averaged = accumulated[normal_mask]
    norms = np.linalg.norm(averaged, axis=-1)
    finite = np.isfinite(averaged).all(axis=-1) & (norms > eps)
    valid_indices = np.argwhere(normal_mask)
    if np.any(finite):
        selected_indices = valid_indices[finite]
        normals[selected_indices[:, 0],
                selected_indices[:,
                                 1]] = (averaged[finite] / norms[finite, None])
    final_mask = np.zeros_like(normal_mask)
    if np.any(finite):
        final_mask[selected_indices[:, 0], selected_indices[:, 1]] = True
    return normals, final_mask


def compute_mae_normal_deg(
    prediction_points: np.ndarray,
    ground_truth_points: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, float]:
    """Returns the mean normal angle error in degrees, or {} without normals.
    """
    pred_normals, pred_normal_mask = point_map_to_normals(
        prediction_points, valid_mask)
    gt_normals, gt_normal_mask = point_map_to_normals(ground_truth_points,
                                                      valid_mask)
    normal_mask = pred_normal_mask & gt_normal_mask
    if not np.any(normal_mask):
        return {}

    dots = np.sum(pred_normals[normal_mask] * gt_normals[normal_mask], axis=-1)
    angles = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
    return {"mae_normal_deg": float(np.mean(angles))}


def _limit_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    indices = np.linspace(0,
                          points.shape[0] - 1,
                          num=max_points,
                          dtype=np.int64)
    return points[indices]


def compute_point_reconstruction_metrics(
    prediction_points: np.ndarray,
    ground_truth_points: np.ndarray,
    valid_mask: np.ndarray,
    *,
    max_points: int,
    f1_threshold: float,
) -> dict[str, float]:
    """Computes depth-unprojected point metrics as the FDMS recon evaluation.

    Args:
        prediction_points: (H, W, 3) predicted point map.
        ground_truth_points: (H, W, 3) GT point map.
        valid_mask: Pixels that are scored.
        max_points: Evenly spaced points kept per cloud (0: all).
        f1_threshold: Distance threshold of point_f1; 0 skips it.

    Returns:
        Accuracy, completion and Chamfer distance (mean squared nearest
        neighbor distances) and point_f1, or {} without points.
    """
    point_mask = (valid_mask & np.isfinite(prediction_points).all(axis=-1) &
                  np.isfinite(ground_truth_points).all(axis=-1))
    if not np.any(point_mask):
        return {}

    pred_points = _limit_points(prediction_points[point_mask].reshape(-1, 3),
                                max_points)
    gt_points = _limit_points(ground_truth_points[point_mask].reshape(-1, 3),
                              max_points)
    if pred_points.size == 0 or gt_points.size == 0:
        return {}

    pred_to_gt_dist, _ = spatial.cKDTree(gt_points).query(pred_points, k=1)
    gt_to_pred_dist, _ = spatial.cKDTree(pred_points).query(gt_points, k=1)
    accuracy = float(np.mean(pred_to_gt_dist**2))
    completion = float(np.mean(gt_to_pred_dist**2))
    metrics = {
        "point_accuracy": accuracy,
        "point_completion": completion,
        "point_chamfer": (accuracy + completion) / 2.0,
    }
    if f1_threshold > 0:
        precision = float(np.mean(pred_to_gt_dist <= f1_threshold))
        recall = float(np.mean(gt_to_pred_dist <= f1_threshold))
        metrics["point_f1"] = (0.0 if precision + recall <= 0 else 2.0 *
                               precision * recall / (precision + recall))
    return metrics


def mean_normal_error(
    pairs: Iterable[DepthPairData],
    depth_maps: Iterable[np.ndarray],
    config: DepthEvaluationConfig,
) -> dict[str, float]:
    """Returns the frames' mean `mae_normal_deg` of their aligned depth maps.

    The GT-only protocols fit on flattened GT-valid pixel values; they pass
    each frame's full aligned map (same fit, floor and clipping) here so
    normals see the image neighbourhoods. Frames without intrinsics are
    skipped.

    Args:
        pairs: The frames.
        depth_maps: Each frame's aligned depth map; may be a generator.
        config: The resolved config.

    Returns:
        {"mae_normal_deg": mean}, or {} without any frame's normals.
    """
    if not config.compute_geometry_metrics:
        return {}
    values = []
    for pair, depth in zip(pairs, depth_maps):
        if pair.intrinsics is None:
            continue
        metric = compute_mae_normal_deg(
            depth_to_camera_points(depth, pair.intrinsics),
            depth_to_camera_points(pair.ground_truth, pair.intrinsics),
            pair.valid_mask,
        )
        if metric:
            values.append(metric["mae_normal_deg"])
    if not values:
        return {}
    return {"mae_normal_deg": float(np.mean(values))}


def compute_depth_geometry_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    valid_mask: np.ndarray,
    intrinsics: np.ndarray | None,
    config: DepthEvaluationConfig,
) -> dict[str, float]:
    """Returns a frame's normal and point-cloud metrics.

    Returns {} when geometry metrics are off or the intrinsics are unknown.
    """
    if not config.compute_geometry_metrics or intrinsics is None:
        return {}
    pred_points = depth_to_camera_points(prediction, intrinsics)
    gt_points = depth_to_camera_points(ground_truth, intrinsics)
    metrics = compute_mae_normal_deg(pred_points, gt_points, valid_mask)
    metrics.update(
        compute_point_reconstruction_metrics(
            pred_points,
            gt_points,
            valid_mask,
            max_points=config.geometry_max_points,
            f1_threshold=config.geometry_f1_threshold,
        ))
    return metrics


def _protocol_prediction_path(pred_path: pathlib.Path,
                              config: DepthEvaluationConfig) -> pathlib.Path:
    """Returns the prediction file a protocol scores for a depths/ file.

    Raises:
        ValueError: The protocol's raw prediction is missing.
    """
    # Temporary port of PR #21 (eval_refactor_hsy, commit d88025f4): GT-only
    # alignment protocols read raw disparity (not depth-space) predictions,
    # except depth_huber_irls_gt, which stays in depth space.
    gt_only = config.protocol in GT_ONLY_PROTOCOLS
    if config.protocol == "log_huber_irls_gt":
        # Scored only from the normalized log depth; the decoded depths/ next
        # to it are not a fallback.
        raw_log_path = prediction_artifacts.frame_file(
            pred_path.parent.parent / "raw_log_depths", pred_path.stem)
        if raw_log_path is None:
            raise ValueError(f"log_huber_irls_gt needs raw_log_depths/"
                             f"{pred_path.stem}.npz ({pred_path.parent})")
        pred_path = raw_log_path
    elif gt_only and config.protocol != "depth_huber_irls_gt":
        raw_pred_path = prediction_artifacts.frame_file(
            pred_path.parent.parent / "raw_disparities", pred_path.stem)
        if raw_pred_path is not None:
            pred_path = raw_pred_path
        elif (pred_path.parent.parent / "dvd_protocol.json").exists():
            raise ValueError(f"{config.protocol} needs raw_disparities; "
                             f"rerun DVD export ({pred_path})")
    return pred_path


def _depth_range_mask(gt_depth: np.ndarray, pred_depth: np.ndarray,
                      config: DepthEvaluationConfig,
                      gt_only: bool) -> np.ndarray:
    """Returns the pixels with finite values inside the configured GT range.

    The GT-only protocols do not exclude pixels by the prediction.
    """
    if gt_only:
        valid_mask = np.isfinite(gt_depth)
        if config.filter_gt_depth_range:
            valid_mask &= gt_depth > config.min_depth
        if config.filter_gt_depth_range and config.max_depth is not None:
            valid_mask &= gt_depth < config.max_depth
    else:
        valid_mask = np.isfinite(gt_depth) & np.isfinite(pred_depth)
        if config.filter_gt_depth_range:
            valid_mask &= gt_depth > config.min_depth
        if config.require_positive_prediction:
            valid_mask &= pred_depth > config.min_depth
        if config.filter_gt_depth_range and config.max_depth is not None:
            valid_mask &= gt_depth < config.max_depth
    return valid_mask


def _marigold_v2_diode_mask(gt_depth: np.ndarray) -> np.ndarray:
    """Returns the official Marigold V2 DIODE mask of a GT depth map."""
    from scipy import ndimage

    # Match the official V2 PPD mask: Sobel on the full native GT map,
    # not on a masked/fill-in map. The same mask is used for fit and score.
    gt_depth_float = gt_depth.astype(np.float32, copy=False)
    gradient = np.abs(ndimage.sobel(gt_depth_float, 0)) + np.abs(
        ndimage.sobel(gt_depth_float, 1))
    return (gt_depth > 0.6) & (gt_depth < 350.0) & (gradient <= 0.3)


def prepare_depth_pair(
    *,
    scene_info: canonical_dataset.BenchmarkSceneInfo,
    gt_path: pathlib.Path,
    pred_path: pathlib.Path,
    config: DepthEvaluationConfig,
    manifest_mask_paths: Mapping[pathlib.Path, pathlib.Path] | None = None,
    manifest_intrinsics: Mapping[pathlib.Path, np.ndarray] | None = None,
) -> DepthPairData:
    """Loads a GT frame and its prediction and builds the scored-pixel mask.

    Args:
        scene_info: The scene.
        gt_path: The GT depth file.
        pred_path: The predicted depth file (depths/); GT-only protocols
            read the raw disparity or raw log depth next to it.
        config: The resolved config.
        manifest_mask_paths: `materialized_mask_paths_by_depth` of the
            scene.
        manifest_intrinsics: `materialized_intrinsics_by_depth` of the
            scene.

    Returns:
        The loaded pair.

    Raises:
        ValueError: A required raw prediction is missing, the DIODE mask
            policy is unknown, or a GT-only protocol's prediction is not
            finite on GT-valid pixels.
    """
    gt_only = config.protocol in GT_ONLY_PROTOCOLS
    pred_path = _protocol_prediction_path(pred_path, config)

    gt_depth = load_depth_map(gt_path, scale=config.gt_depth_scale)
    pred_depth = load_depth_map(pred_path, scale=config.pred_depth_scale)
    pred_depth = _resize_to_match(pred_depth, gt_depth.shape,
                                  config.resize_interpolation)

    valid_mask = _depth_range_mask(gt_depth, pred_depth, config, gt_only)

    mask = None
    if manifest_mask_paths:
        mask_path = manifest_mask_paths.get(_canonical_existing_path(gt_path))
        if mask_path is not None:
            mask = load_depth_map(mask_path) > 0
    if mask is None:
        mask = _load_optional_mask(scene_info.mask_path, _depth_key(gt_path))
    if mask is not None:
        valid_mask &= _resize_to_match(mask.astype(np.float32), gt_depth.shape,
                                       "nearest") > 0
    if config.diode_mask_policy not in {"original", "marigold_v2"}:
        raise ValueError(
            f"Unknown DIODE mask policy: {config.diode_mask_policy}")
    if (config.diode_mask_policy == "marigold_v2" and
            scene_info.task == "MonoDepth" and
            scene_info.dataset.lower() == "marigold" and
            scene_info.scene.lower() == "diode"):
        valid_mask &= _marigold_v2_diode_mask(gt_depth)
    if gt_only and valid_mask.any() and not np.isfinite(
            pred_depth[valid_mask]).all():
        raise ValueError(
            f"Nonfinite prediction on GT-valid pixels ({pred_path})")
    intrinsics = resolve_depth_intrinsics(
        scene_info,
        gt_path,
        gt_depth.shape,
        manifest_intrinsics=manifest_intrinsics,
    )

    return DepthPairData(
        gt_path=gt_path,
        pred_path=pred_path,
        ground_truth=gt_depth,
        prediction=pred_depth,
        valid_mask=valid_mask,
        intrinsics=intrinsics,
    )


def _apply_post_clip_policy(
        pair_data: DepthPairData,
        config: DepthEvaluationConfig) -> DepthEvaluationConfig:
    """Returns the config with a frame's post-clip policy applied.

    Raises:
        ValueError: The policy is unknown, or "valid_gt_max" finds no finite
            positive valid GT.
    """
    if config.post_clip_policy == "valid_gt_max":
        valid_gt = pair_data.ground_truth[pair_data.valid_mask]
        if valid_gt.size == 0 or not np.isfinite(valid_gt).all() or (
                valid_gt <= 0).any():
            raise ValueError(
                "Per-image GT clipping requires finite positive valid GT")
        upper = float(valid_gt.max())
        # The mask has already been prepared. These bounds only affect scoring,
        # including GT-only scorers which use min_depth/max_depth for clipping.
        return dataclasses.replace(config,
                                   min_depth=1e-3,
                                   max_depth=upper,
                                   post_clip_min=1e-3,
                                   post_clip_max=upper)
    if config.post_clip_policy != "fixed":
        raise ValueError(f"Unknown post-clip policy: {config.post_clip_policy}")
    return config


def evaluate_single_depth_pair(
    pair_data: DepthPairData,
    config: DepthEvaluationConfig,
    artifacts: DepthErrorArtifactWriter | None = None,
) -> dict[str, float]:
    """Aligns and scores one frame on its own.

    Args:
        pair_data: The frame.
        config: The resolved config.
        artifacts: Optional writer of the frame's error maps.

    Returns:
        The frame's depth and geometry metrics.
    """
    config = _apply_post_clip_policy(pair_data, config)
    if config.protocol in GT_ONLY_PROTOCOLS:
        # Temporary port of PR #21 (eval_refactor_hsy, commit d88025f4).
        from spatialarena.eval import inverse_l1_gt

        return inverse_l1_gt.score_sequence(
            [pair_data],
            config,
            diagnostic_suffix=pair_data.pred_path.stem,
            artifacts=artifacts)
    scale, shift = fit_frame_alignment(
        pair_data.prediction,
        pair_data.ground_truth,
        pair_data.valid_mask,
        config.alignment,
        max_resolution=config.alignment_max_resolution,
    )
    aligned_pred = clip_prediction_depth(
        apply_depth_alignment(pair_data.prediction, scale, shift),
        min_depth=config.post_clip_min,
        max_depth=config.post_clip_max,
    )
    depth_metrics = compute_depth_metrics(
        aligned_pred,
        pair_data.ground_truth,
        pair_data.valid_mask,
        min_depth=config.min_depth,
    )
    geometry_metrics = compute_depth_geometry_metrics(
        aligned_pred,
        pair_data.ground_truth,
        pair_data.valid_mask,
        pair_data.intrinsics,
        config,
    )
    metrics = depth_metrics | geometry_metrics
    if artifacts is not None:
        artifacts.write(pair_data, aligned_pred, scale, shift, metrics)
    return metrics


def evaluate_sequence_depth_pairs(
    pair_data: list[DepthPairData],
    config: DepthEvaluationConfig,
    artifacts: DepthErrorArtifactWriter | None = None,
) -> dict[str, float]:
    """Aligns a whole sequence with one fit and scores it.

    Args:
        pair_data: The sequence's frames.
        config: The resolved config.
        artifacts: Optional writer of the frames' error maps.

    Returns:
        The depth metrics over all frames' pixels, and the frame mean of
        each geometry metric.
    """
    if config.protocol in GT_ONLY_PROTOCOLS:
        # Temporary port of PR #21 (eval_refactor_hsy, commit d88025f4).
        from spatialarena.eval import inverse_l1_gt

        return inverse_l1_gt.score_sequence(pair_data,
                                            config,
                                            artifacts=artifacts)
    pred_values = np.concatenate(
        [pair.prediction[pair.valid_mask] for pair in pair_data],)
    gt_values = np.concatenate(
        [pair.ground_truth[pair.valid_mask] for pair in pair_data],)
    scale, shift = fit_depth_alignment(pred_values, gt_values, config.alignment)

    aligned_pred_values = []
    aligned_gt_values = []
    geometry_sums: dict[str, float] = {}
    geometry_counts: dict[str, int] = {}
    for pair in pair_data:
        aligned_pred = apply_depth_alignment(pair.prediction, scale, shift)
        aligned_pred = clip_prediction_depth(
            aligned_pred,
            min_depth=config.post_clip_min,
            max_depth=config.post_clip_max,
        )
        aligned_pred_values.append(aligned_pred[pair.valid_mask])
        aligned_gt_values.append(pair.ground_truth[pair.valid_mask])
        if artifacts is not None:
            # One sequence-wide scale/shift; per-frame metrics are for
            # inspection only.
            artifacts.write(
                pair, aligned_pred, scale, shift,
                compute_depth_metrics(aligned_pred,
                                      pair.ground_truth,
                                      pair.valid_mask,
                                      min_depth=config.min_depth))
        for name, value in compute_depth_geometry_metrics(
                aligned_pred,
                pair.ground_truth,
                pair.valid_mask,
                pair.intrinsics,
                config,
        ).items():
            geometry_sums[name] = geometry_sums.get(name, 0.0) + value
            geometry_counts[name] = geometry_counts.get(name, 0) + 1

    metrics = compute_depth_metrics_from_values(
        np.concatenate(aligned_pred_values),
        np.concatenate(aligned_gt_values),
        min_depth=config.min_depth,
    )
    metrics.update({
        name: value / geometry_counts[name]
        for name, value in geometry_sums.items()
        if geometry_counts.get(name, 0) > 0
    })
    return metrics


def _load_optional_mask(mask_path: pathlib.Path | None,
                        key: str) -> np.ndarray | None:
    if mask_path is None:
        return None
    mask_files = {
        _depth_key(path): path
        for path in list_depth_files_in_directory(mask_path)
    }
    if key not in mask_files:
        return None
    return load_depth_map(mask_files[key]) > 0


@dataclasses.dataclass
class SceneDepthPairs:
    """A scene's GT and predicted depth files and how they pair up.

    Built by `scene_depth_pairs`.

    Attributes:
        gt_files: The scene's GT depth files.
        pred_files: The prediction's depth files.
        pairs: (GT file, prediction file) pairs, without GT-excluded frames.
        missing_gt: Keys of predictions without GT.
        missing_pred: Keys of GT frames without a prediction.
        used_index_fallback: The files were paired in sorted order.
        gt_excluded: Keys of the frames whose GT is excluded.
    """

    gt_files: list[pathlib.Path]
    pred_files: list[pathlib.Path]
    pairs: list[tuple[pathlib.Path, pathlib.Path]]
    missing_gt: list[str]
    missing_pred: list[str]
    used_index_fallback: bool
    gt_excluded: set[str]


def scene_depth_pairs(scene_info: canonical_dataset.BenchmarkSceneInfo,
                      prediction_scene_path: pathlib.Path,
                      config: DepthEvaluationConfig) -> SceneDepthPairs:
    """Pairs a scene's GT depth files with a prediction's.

    Files pair by key, then by the Bonn (timestamp) or Tanks and Temples
    (index offset) fallbacks; the scene's GT-excluded frames are left out.

    Args:
        scene_info: The scene.
        prediction_scene_path: The prediction's scene directory.
        config: A resolved config (`resolve_depth_config`).

    Returns:
        The files and their pairing.

    Raises:
        ValueError: The scene has no GT depth directory, or nothing pairs.
    """
    if scene_info.depth_gt_path is None:
        raise ValueError(
            f"Scene has no canonical GT depth directory: {scene_info.scene}")

    gt_files = list_scene_gt_depth_files(scene_info, config.gt_source)
    pred_files = find_depth_files(prediction_scene_path)
    pairs, missing_gt, missing_pred, used_index_fallback = pair_depth_files(
        gt_files,
        pred_files,
    )
    if not pairs and scene_info.dataset.lower() == "bonn":
        pairs, missing_gt, missing_pred = pair_depth_files_by_nearest_timestamp(
            gt_files,
            pred_files,
        )
    if not pairs and scene_info.dataset.lower() == "tanks_and_temples":
        pairs, missing_gt, missing_pred = pair_depth_files_by_integer_offset(
            gt_files,
            pred_files,
            gt_index_offset=1,
        )
    if not pairs:
        raise ValueError(
            f"No matching depth predictions for scene {scene_info.scene} in "
            f"{prediction_scene_path}")
    gt_excluded = gt_excluded_frame_keys(scene_info)
    if gt_excluded:
        pairs = [
            pair for pair in pairs if _depth_key(pair[0]) not in gt_excluded
        ]
        missing_gt = [key for key in missing_gt if key not in gt_excluded]
    return SceneDepthPairs(gt_files, pred_files, pairs, missing_gt,
                           missing_pred, used_index_fallback, gt_excluded)


# GT-only sequence protocols whose per-frame alignment can be rebuilt from the
# scorer's saved fit (`<prediction>/<protocol>_fit.json`): its scale/shift,
# and for the inverse protocol its floor and representation.
STORED_FIT_PROTOCOLS = ("depth_huber_irls_gt", "inverse_huber_irls_gt")


def write_sequence_viewer_from_fit(
    scene_info: canonical_dataset.BenchmarkSceneInfo,
    prediction_scene_path: pathlib.Path,
    viewer,
    metrics: Mapping[str, float],
    config: DepthEvaluationConfig | None = None,
    fit: Mapping[str, Any] | None = None,
) -> int:
    """Writes a VideoDepth case's per-frame viewer images from its saved fit.

    `evaluate_depth_scene` holds every frame's GT, prediction and mask to fit
    the sequence (SpatialArena's longest sequences need over 90 GB); the fit
    only gives one scale/shift, which the evaluation saves in
    `<prediction>/<protocol>_fit.json`. Applying it frame by frame gives the
    same aligned maps with one frame in memory. Frames are paired and skipped
    as in `evaluate_depth_scene`.

    Args:
        scene_info: The scene.
        prediction_scene_path: The prediction's scene directory.
        viewer: The viewer writer (`viewer_assets.DepthViewerWriter`).
        metrics: The case's stored sequence metrics, written to the viewer's
            frames.json.
        config: The requested config; resolved here.
        fit: The saved fit; read from the prediction directory when None.

    Returns:
        The number of frames written.

    Raises:
        ValueError: The case has no stored-fit path, the fit is missing or
            does not match, or no frame has a valid pixel.
    """
    # pylint: disable=import-outside-toplevel
    from spatialarena.eval import depth_space_gt
    from spatialarena.eval import inverse_l1_gt

    config = resolve_depth_config(scene_info, config or DepthEvaluationConfig())
    if (scene_info.task != "VideoDepth" or
            config.protocol not in STORED_FIT_PROTOCOLS):
        raise ValueError(f"No stored-fit viewer path for {scene_info.task} / "
                         f"{config.protocol}")
    if fit is None:
        fit_path = (pathlib.Path(prediction_scene_path) /
                    f"{config.protocol}_fit.json")
        if not fit_path.is_file():
            raise ValueError(f"No saved sequence fit {fit_path}")
        fit = json.loads(fit_path.read_text(encoding="utf-8"))
    if fit.get("protocol") != config.protocol:
        raise ValueError(f"Saved fit is for {fit.get('protocol')}, the case "
                         f"was scored with {config.protocol}")
    scale = float(fit["scale"])
    shift = float(fit["shift"])
    if config.protocol == "inverse_huber_irls_gt":
        raw = fit["representation"] == "raw_disparity"
        inverse_floor = float(fit["inverse_floor"])

        def align(prediction: np.ndarray) -> np.ndarray:
            return inverse_l1_gt.aligned_inverse_depth_map(
                prediction,
                scale,
                shift,
                raw=raw,
                inverse_floor=inverse_floor,
                config=config)
    else:
        if fit.get("oracle_gt_bounds"):
            raise ValueError("GT-bounds fits are not rebuilt from a saved fit")

        def align(prediction: np.ndarray) -> np.ndarray:
            return depth_space_gt.aligned_depth_space_map(
                prediction, scale, shift, config)

    manifest_mask_paths = materialized_mask_paths_by_depth(scene_info)
    manifest_intrinsics = materialized_intrinsics_by_depth(scene_info)
    written = 0
    for gt_path, pred_path in scene_depth_pairs(scene_info,
                                                prediction_scene_path,
                                                config).pairs:
        pair = prepare_depth_pair(scene_info=scene_info,
                                  gt_path=gt_path,
                                  pred_path=pred_path,
                                  config=config,
                                  manifest_mask_paths=manifest_mask_paths,
                                  manifest_intrinsics=manifest_intrinsics)
        # Skipped by the evaluation too (VideoDepth needs one valid pixel).
        if pair.num_valid_pixels < 1:
            continue
        aligned = align(pair.prediction)
        viewer.write(
            pair, aligned, scale, shift,
            compute_depth_metrics(aligned,
                                  pair.ground_truth,
                                  pair.valid_mask,
                                  min_depth=config.min_depth))
        written += 1
    if not written:
        raise ValueError(f"No valid depth pixels for scene {scene_info.scene}")
    viewer.finish(scene_info, metrics)
    return written


def _min_valid_pixels(scene_info: canonical_dataset.BenchmarkSceneInfo,
                      config: DepthEvaluationConfig) -> int:
    """Returns the fewest valid pixels a frame needs to be scored."""
    # Temporary port of PR #21 (eval_refactor_hsy, commit d88025f4): the
    # depth-space/inverse-depth Huber-IRLS protocols need at least 2 valid
    # pixels per frame to fit an affine scale/shift; everything else only
    # needs 1.
    if scene_info.task == "MonoDepth" and config.protocol in {
            "inverse_huber_irls_gt", "depth_huber_irls_gt", "log_huber_irls_gt"
    }:
        return 2
    return 1


def _add_lidar_metrics(
    scene_info: canonical_dataset.BenchmarkSceneInfo,
    prediction_scene_path: pathlib.Path,
    requested: DepthEvaluationConfig,
    metrics: dict[str, float],
) -> tuple[dict[str, float], dict]:
    """Adds the `lidar_` metrics of a SpatialArena dense-GT pass.

    Args:
        scene_info: The scene.
        prediction_scene_path: The prediction's scene directory.
        requested: The requested (unresolved) config.
        metrics: The dense metrics; not modified.

    Returns:
        (metrics with the `lidar_` metrics, the LiDAR pass summary without
        its metrics, or {} when the scene has no LiDAR pass).
    """
    lidar = {}
    if (requested.gt_source == "dense" and
            scene_info.dataset.lower() == spatialarena.SPATIALARENA and
            scene_info.task in LIDAR_GT_TASKS):
        lidar = evaluate_lidar_gt(scene_info, prediction_scene_path, requested)
        # A new dict: the viewer keeps the dense metrics it was given.
        metrics = metrics | {
            f"lidar_{name}": value
            for name, value in lidar.pop("metrics", {}).items()
        }
    return metrics, lidar


def evaluate_depth_scene(
    scene_info: canonical_dataset.BenchmarkSceneInfo,
    prediction_scene_path: pathlib.Path,
    config: DepthEvaluationConfig | None = None,
    artifacts: DepthErrorArtifactWriter | None = None,
    viewer=None,
) -> dict:
    """Evaluates one scene.

    `artifacts` and `viewer` both receive every frame's aligned prediction.
    SpatialArena's MonoDepth and VideoDepth scenes are also scored on their
    LiDAR GT (`lidar_` metrics).

    Args:
        scene_info: The scene.
        prediction_scene_path: The prediction's scene directory.
        config: The requested config; resolved here.
        artifacts: Optional writer of error maps.
        viewer: Optional writer of the dashboard's per-frame images
            (`viewer_assets.DepthViewerWriter`).

    Returns:
        The scene result: counts, the resolved options, the metrics and the
        unpaired frames.

    Raises:
        ValueError: The files do not pair or no frame has a valid pixel.
    """
    # pylint: disable-next=import-outside-toplevel
    from spatialarena.eval import viewer_assets

    requested = config or DepthEvaluationConfig()
    config = resolve_depth_config(scene_info, requested)
    if config.gt_source == "lidar":
        config = dataclasses.replace(config,
                                     gt_depth_scale=LIDAR_DEPTH_SCALE,
                                     compute_geometry_metrics=False)
    if viewer is not None:
        frame_writer = viewer_assets.TeeDepthWriter(artifacts, viewer)
    else:
        frame_writer = artifacts
    scene_pairs = scene_depth_pairs(scene_info, prediction_scene_path, config)

    metric_sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    total_valid = 0
    evaluated_frames = 0
    skipped_frames = 0
    skipped_frame_details: list[dict] = []
    min_valid = _min_valid_pixels(scene_info, config)
    prepared_pairs: list[DepthPairData] = []
    manifest_mask_paths = materialized_mask_paths_by_depth(scene_info)
    manifest_intrinsics = materialized_intrinsics_by_depth(scene_info)

    for gt_path, pred_path in scene_pairs.pairs:
        pair_data = prepare_depth_pair(
            scene_info=scene_info,
            gt_path=gt_path,
            pred_path=pred_path,
            config=config,
            manifest_mask_paths=manifest_mask_paths,
            manifest_intrinsics=manifest_intrinsics,
        )
        num_valid = pair_data.num_valid_pixels
        if num_valid < min_valid:
            skipped_frames += 1
            skipped_frame_details.append({
                "gt_path": str(gt_path),
                "pred_path": str(pred_path),
                "num_valid_pixels": num_valid
            })
            continue
        total_valid += num_valid
        evaluated_frames += 1
        if scene_info.task == "VideoDepth":
            prepared_pairs.append(pair_data)
        else:
            frame_metrics = evaluate_single_depth_pair(pair_data, config,
                                                       frame_writer)
            for name, value in frame_metrics.items():
                metric_sums[name] = metric_sums.get(name, 0.0) + value
                metric_counts[name] = metric_counts.get(name, 0) + 1

    if total_valid == 0:
        raise ValueError(f"No valid depth pixels for scene {scene_info.scene}")

    if scene_info.task == "VideoDepth":
        metrics = evaluate_sequence_depth_pairs(prepared_pairs, config,
                                                frame_writer)
        metric_units = 1
        metric_aggregation = "sequence"
    else:
        metrics = {
            name: value / metric_counts[name]
            for name, value in metric_sums.items()
            if metric_counts.get(name, 0) > 0
        }
        if scene_info.task == "MonoDepth":
            metric_units = evaluated_frames
            metric_aggregation = "frame_mean"
        else:
            metric_units = 1
            metric_aggregation = "sample"
    if viewer is not None:
        viewer.finish(scene_info, metrics)
    metrics, lidar = _add_lidar_metrics(scene_info, prediction_scene_path,
                                        requested, metrics)

    runtime = prediction_artifacts.read_adapter_runtime(prediction_scene_path)
    result = {
        "task": scene_info.task,
        "dataset": scene_info.dataset,
        "scene": scene_info.scene,
        "prediction_scene_path": str(prediction_scene_path),
        "num_gt_depth_files": len(scene_pairs.gt_files),
        "num_pred_depth_files": len(scene_pairs.pred_files),
        "num_evaluated_frames": evaluated_frames,
        "num_skipped_frames": skipped_frames,
        "skipped_frame_details": skipped_frame_details,
        "num_metric_units": metric_units,
        "num_valid_pixels": total_valid,
        "runtime": runtime,
        "alignment": config.alignment,
        "protocol": config.protocol,
        "resize_interpolation": config.resize_interpolation,
        "metric_aggregation": metric_aggregation,
        "min_depth": config.min_depth,
        "max_depth": config.max_depth,
        "gt_depth_scale": config.gt_depth_scale,
        "pred_depth_scale": config.pred_depth_scale,
        "post_clip_min": config.post_clip_min,
        "post_clip_max": config.post_clip_max,
        "post_clip_policy": config.post_clip_policy,
        "diode_mask_policy": config.diode_mask_policy,
        "alignment_max_resolution": config.alignment_max_resolution,
        "filter_gt_depth_range": config.filter_gt_depth_range,
        "compute_geometry_metrics": config.compute_geometry_metrics,
        "geometry_max_points": config.geometry_max_points,
        "geometry_f1_threshold": config.geometry_f1_threshold,
        "inverse_floor_policy": config.inverse_floor_policy,
        "missing_gt_predictions": scene_pairs.missing_gt,
        "gt_excluded_frames": sorted(scene_pairs.gt_excluded),
        "missing_predicted_depths": scene_pairs.missing_pred,
        "used_index_fallback": scene_pairs.used_index_fallback,
        "metrics": metrics,
    }
    if artifacts:
        result["error_artifacts"] = artifacts.describe()
        result["frame_results"] = artifacts.records
    if viewer is not None:
        result["viewer_dir"] = str(viewer.case_dir)
    if lidar:
        result["lidar_gt"] = lidar
    return result


def evaluate_lidar_gt(scene_info: canonical_dataset.BenchmarkSceneInfo,
                      prediction_scene_path: pathlib.Path,
                      config: DepthEvaluationConfig) -> dict:
    """Scores the same predictions on the scene's sparse LiDAR depth.

    The LiDAR pass has its own alignment fit and the dense pass's range and
    protocol.

    Args:
        scene_info: The scene.
        prediction_scene_path: The prediction's scene directory.
        config: The requested (unresolved) config of the dense pass.

    Returns:
        The depth metrics and the pass's counts, or {"error": reason} if it
        could not run.
    """
    try:
        result = evaluate_depth_scene(scene_info,
                                      prediction_scene_path,
                                      config=dataclasses.replace(
                                          config, gt_source="lidar"))
    except ValueError as exc:
        return {"error": str(exc)}
    metrics = {
        name: value
        for name, value in result["metrics"].items()
        if not name.startswith(LIDAR_METRIC_EXCLUDE_PREFIXES)
    }
    summary = {"metrics": metrics, "gt_depth_scale": result["gt_depth_scale"]}
    for key in ("num_gt_depth_files", "num_evaluated_frames",
                "num_skipped_frames", "num_valid_pixels",
                "missing_gt_predictions", "gt_excluded_frames"):
        summary[key] = result[key]
    return summary


def aggregate_depth_results(scene_results: list[dict]) -> dict[str, float]:
    """Returns each metric's unit-weighted mean over the scenes reporting it.

    Not every scene reports every metric: SpatialArena's `lidar_` metrics
    exist only for its scenes.

    Args:
        scene_results: `evaluate_depth_scene` results.

    Returns:
        The mean of each metric.
    """
    metric_sums: dict[str, float] = {}
    metric_units: dict[str, int] = {}
    for result in scene_results:
        weight = int(result.get("num_metric_units", 0))
        if weight <= 0:
            weight = int(result.get("num_evaluated_frames", 0)) or 1
        for name, value in result["metrics"].items():
            metric_sums[name] = metric_sums.get(name, 0.0) + value * weight
            metric_units[name] = metric_units.get(name, 0) + weight
    return {
        name: value / metric_units[name] for name, value in metric_sums.items()
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses the command line."""
    parser = argparse.ArgumentParser(
        description="Evaluate standardized depth prediction artifacts.")
    parser.add_argument("--benchmark_root",
                        type=pathlib.Path,
                        default=dataset_registry.DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--experiment_root", type=pathlib.Path, required=True)
    parser.add_argument("--experiment_name", type=str, default="exp")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--scenes", nargs="*", default=[])
    parser.add_argument("--include_scannet", action="store_true")
    parser.add_argument(
        "--protocol",
        choices=[
            "standard",
            "pi3_mono",
            "pi3_monodepth",
            "pi3_video",
            "pi3_videodepth",
            "custom",
            "inverse_l1_gt",
            "inverse_huber_irls_gt",
            "inverse_l1_l2init_gt",
            "depth_huber_irls_gt",
            "gt_bounds_direct",
            "gt_bounds_depth_huber_irls",
            "log_huber_irls_gt",
        ],
        default="standard",
        help=("Named depth evaluation protocol. Use custom with explicit "
              "alignment/range flags."),
    )
    parser.add_argument(
        "--inverse_floor_policy",
        choices=["fixed", "gt_max"],
        default="fixed",
        help=("Temporary port of PR #21: how inverse_l1_gt floors the aligned "
              "inverse depth."),
    )
    parser.add_argument(
        "--alignment",
        choices=[
            "standard",
            "none",
            "median",
            "least_squares",
            "pi3_median_scale",
            "scale",
            "pi3_scale_shift_lad",
            "scale_shift_lad",
            "scale&shift",
        ],
        default="standard",
    )
    parser.add_argument(
        "--resize_interpolation",
        choices=["nearest", "linear", "bilinear", "cubic", "bicubic"],
        default="linear",
        help=("Interpolation used when resizing prediction depths to GT "
              "resolution."),
    )
    parser.add_argument("--min_depth", type=float, default=1e-6)
    parser.add_argument("--max_depth", type=float, default=None)
    parser.add_argument("--gt_depth_scale", type=float, default=1.0)
    parser.add_argument("--pred_depth_scale", type=float, default=1.0)
    parser.add_argument("--post_clip_min", type=float, default=None)
    parser.add_argument("--post_clip_max", type=float, default=None)
    parser.add_argument("--require_positive_prediction", action="store_true")
    parser.add_argument("--alignment_max_resolution", type=int, default=None)
    parser.add_argument(
        "--no_filter_gt_depth_range",
        action="store_true",
        help="Do not use min/max depth to form the GT validity mask.",
    )
    parser.add_argument(
        "--no_geometry_metrics",
        action="store_true",
        help="Disable normal, Chamfer, and F1 geometry metrics.",
    )
    parser.add_argument(
        "--geometry_max_points",
        type=int,
        default=10000,
        help=("Maximum valid points sampled per frame for KD-tree geometry "
              "metrics."),
    )
    parser.add_argument(
        "--geometry_f1_threshold",
        type=float,
        default=0.05,
        help="Euclidean distance threshold in meters for point_f1.",
    )
    parser.add_argument("--output_path", type=pathlib.Path, default=None)
    parser.add_argument(
        "--error_artifact_dir",
        type=pathlib.Path,
        default=None,
        help=("Write per-frame error maps (error_maps/, "
              "error_visualizations/) and frame_results there."),
    )
    parser.add_argument(
        "--error_artifact_max_side",
        type=int,
        default=1024,
        help=("Longer side of the saved maps (nearest downsampling of the "
              "GT-size map); 0 = GT size."))
    parser.add_argument(
        "--save_aligned_depth",
        action="store_true",
        help=("Also save each frame's aligned prediction (aligned_depths/, "
              "float16)."))
    parser.add_argument(
        "--viewer_dir",
        type=pathlib.Path,
        default=None,
        help=("Write the dashboard viewer's per-frame depth and error images "
              "there (viewer_assets)."))
    parser.add_argument(
        "--viewer_scene_root",
        type=pathlib.Path,
        default=None,
        help=("Root of the viewer assets every method of a scene shares (GT "
              "and RGB frames)."))
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--diode_mask_policy",
        choices=["original", "marigold_v2"],
        default=os.environ.get("FFB_DIODE_MASK_POLICY", "original"),
    )
    return parser.parse_args(argv)


def run_depth_evaluation(
    *,
    benchmark_root: pathlib.Path,
    experiment_root: pathlib.Path,
    experiment_name: str,
    methods: Iterable[str],
    tasks: Iterable[str],
    datasets: Iterable[str],
    scenes: Iterable[str],
    include_scannet: bool,
    config: DepthEvaluationConfig,
    error_artifact_dir: pathlib.Path | None = None,
    error_artifact_max_side: int = 1024,
    save_aligned_depth: bool = False,
    viewer_dir: pathlib.Path | None = None,
    viewer_scene_root: pathlib.Path | None = None,
) -> dict:
    """Evaluates every selected scene of every method.

    A scene that fails is recorded in "errors" and does not stop the run.

    Args:
        benchmark_root: Root of the benchmark datasets.
        experiment_root: Root of the method outputs.
        experiment_name: Experiment folder under `experiment_root`.
        methods: Methods to evaluate.
        tasks: Tasks to evaluate.
        datasets: Datasets to evaluate (empty: all).
        scenes: Scenes to evaluate (empty: all).
        include_scannet: Also select the ScanNet datasets.
        config: The requested config; resolved per scene.
        error_artifact_dir: Optional root of the per-frame error maps.
        error_artifact_max_side: Longer side of the saved maps.
        save_aligned_depth: Also save the aligned predictions.
        viewer_dir: Optional root of the dashboard viewer images.
        viewer_scene_root: Root of the viewer assets a scene's methods share.

    Returns:
        The scene results, their aggregated metrics and the errors.
    """
    # pylint: disable-next=import-outside-toplevel
    from spatialarena.eval import viewer_assets

    scene_filter = list(scenes)
    methods = list(methods)
    datasets = list(datasets)
    # One method on one named scene (the engine's per-case call) writes
    # straight into error_artifact_dir.
    flat_artifacts = len(methods) == 1 and len(scene_filter) == 1 and len(
        datasets) == 1
    scene_results: list[dict] = []
    errors: list[dict] = []

    for spec in dataset_registry.select_dataset_specs(
            tasks=tasks,
            datasets=datasets,
            include_scannet=include_scannet,
    ):
        if not (spec.data_path(benchmark_root) / "metadata.json").is_file():
            errors.append({
                "task": spec.task,
                "dataset": spec.dataset,
                "error": "metadata.json is missing",
            })
            continue

        dataset = spec.create_dataset(benchmark_root, scenes=scene_filter)
        for scene_info in dataset.list_scenes():
            for method in methods:
                prediction_scene_path = (
                    result_layout.resolve_method_scene_output_path(
                        experiment_root=experiment_root,
                        experiment_name=experiment_name,
                        dataset_name=scene_info.dataset,
                        method_name=method,
                        scene_name=scene_info.scene,
                        task_name=scene_info.task,
                    ))
                artifacts = None
                if error_artifact_dir is not None:
                    if flat_artifacts:
                        artifact_root = error_artifact_dir
                    else:
                        artifact_root = (error_artifact_dir / method /
                                         scene_info.task / scene_info.dataset /
                                         scene_info.scene)
                    artifacts = DepthErrorArtifactWriter(
                        root=artifact_root,
                        max_side=error_artifact_max_side,
                        save_aligned_depth=save_aligned_depth,
                    )
                try:
                    viewer = None
                    if viewer_dir is not None and scene_info.task in {
                            "MonoDepth", "VideoDepth"
                    }:
                        viewer = viewer_assets.make_depth_viewer_writer(
                            scene_info,
                            config,
                            viewer_assets.case_viewer_dir(
                                viewer_dir, flat_artifacts, method,
                                scene_info.task, scene_info.dataset,
                                scene_info.scene),
                            viewer_scene_root,
                        )
                    scene_results.append(
                        evaluate_depth_scene(
                            scene_info,
                            prediction_scene_path,
                            config=config,
                            artifacts=artifacts,
                            viewer=viewer,
                        ) | {"method": method})
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    errors.append({
                        "task": scene_info.task,
                        "dataset": scene_info.dataset,
                        "scene": scene_info.scene,
                        "method": method,
                        "prediction_scene_path": str(prediction_scene_path),
                        "error": str(exc),
                    })

    return {
        "num_scene_results": len(scene_results),
        "num_errors": len(errors),
        "metrics": aggregate_depth_results(scene_results),
        "scene_results": scene_results,
        "errors": errors,
    }


def main(
    argv: list[str] | None = None,
    default_tasks: list[str] | None = None,
) -> None:
    """Runs the evaluation and prints or writes the result JSON.

    Args:
        argv: Command-line arguments; None reads sys.argv.
        default_tasks: Tasks to evaluate when --tasks is empty; None
            evaluates MonoDepth, VideoDepth and MVRecon.
    """
    args = parse_args(argv)
    result_layout.validate_output_path_outside_benchmark_root(
        args.output_path,
        args.benchmark_root,
        label="--output_path",
    )
    result_layout.validate_output_path_outside_benchmark_root(
        args.error_artifact_dir,
        args.benchmark_root,
        label="--error_artifact_dir",
    )
    for label in ("viewer_dir", "viewer_scene_root"):
        result_layout.validate_output_path_outside_benchmark_root(
            getattr(args, label), args.benchmark_root, label=f"--{label}")
    tasks = args.tasks or default_tasks or [
        "MonoDepth", "VideoDepth", "MVRecon"
    ]
    config = DepthEvaluationConfig(
        protocol=args.protocol,
        alignment=args.alignment,
        resize_interpolation=args.resize_interpolation,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        gt_depth_scale=args.gt_depth_scale,
        pred_depth_scale=args.pred_depth_scale,
        post_clip_min=args.post_clip_min,
        post_clip_max=args.post_clip_max,
        require_positive_prediction=args.require_positive_prediction,
        alignment_max_resolution=args.alignment_max_resolution,
        filter_gt_depth_range=not args.no_filter_gt_depth_range,
        compute_geometry_metrics=not args.no_geometry_metrics,
        geometry_max_points=args.geometry_max_points,
        geometry_f1_threshold=args.geometry_f1_threshold,
        inverse_floor_policy=args.inverse_floor_policy,
        diode_mask_policy=args.diode_mask_policy,
    )
    result = run_depth_evaluation(
        benchmark_root=args.benchmark_root,
        experiment_root=args.experiment_root,
        experiment_name=args.experiment_name,
        methods=args.methods,
        tasks=tasks,
        datasets=args.datasets,
        scenes=args.scenes,
        include_scannet=args.include_scannet,
        config=config,
        error_artifact_dir=args.error_artifact_dir,
        error_artifact_max_side=args.error_artifact_max_side,
        save_aligned_depth=args.save_aligned_depth,
        viewer_dir=args.viewer_dir,
        viewer_scene_root=args.viewer_scene_root,
    )

    text = json.dumps(result, indent=2)
    if args.output_path:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)

    if args.strict and result["num_errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
