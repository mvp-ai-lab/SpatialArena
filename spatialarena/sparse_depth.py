# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Sparse depth priors for the ``rgb_sparse_depth`` / ``rgb_sparse_depth_cameras`` tracks.

``rgb_depth`` and ``rgb_masked_depth`` hand a method (part of) the dataset's dense GT depth. The sparse-depth tracks
hand it the kind of sparse depth a real system has:

- **LiDAR** (KITTI, VideoDepth): each frame's raw Velodyne scan, projected into the left colour camera (``image_02``)
  with KITTI's calibration (``P_rect_02 @ R_rect_00 @ Tr_velo_to_cam``): about 4-5% of the pixels, against the
  accumulated GT's ~20%.
- **SfM points** (DTU, ETH3D, Tanks and Temples on MVRecon; those plus IMC and TexturePoorSfM on SfM):
  the keypoints each image observes in the scene's COLMAP model (``points2D`` with a 3D point), at the depth of that
  3D point in the image's camera, typically 0.1-1% of the pixels. These points come from the same reconstruction the
  SfM poses are scored against, so SfM results on these tracks are SfM-assisted, not pose estimation.

The priors are materialized once (``scripts/preprocess/materialize_sparse_depth_priors.py``) as point lists, one
``.npz`` per image under ``<benchmark_root>/_sparse_depth/<dataset>/<scene>/<image stem>.npz`` (``xy``: [N, 2] pixel
coordinates in the image as stored, ``z``: [N] depth in the dataset's GT units, ``size``: (width, height),
``source``: ``"lidar"`` or ``"sfm"``). Points rather than images, so that no point is lost before a method resizes
its input: ``splat_sparse_depth`` draws each point as a small square block (one value, never interpolated) sized so
that it survives a nearest-neighbour resize to the method's input resolution as about one pixel.

Scoring is unchanged (every GT-valid pixel, after each task's usual alignment).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

TRACK_RGB_SPARSE_DEPTH = "rgb_sparse_depth"
TRACK_RGB_SPARSE_DEPTH_CAMERAS = "rgb_sparse_depth_cameras"
SPARSE_DEPTH_TRACKS = frozenset({TRACK_RGB_SPARSE_DEPTH, TRACK_RGB_SPARSE_DEPTH_CAMERAS})
SPARSE_DEPTH_DIRNAME = "_sparse_depth"
PROTOCOL = "sparse-v1"


def sparse_depth_root(benchmark_root: Path) -> Path:
    return Path(benchmark_root) / SPARSE_DEPTH_DIRNAME


def scene_dir(data_path: Path, benchmark_root: Path | None = None) -> Path:
    """``<benchmark_root>/_sparse_depth/<dataset>/<scene>`` for a scene folder ``<root>/<Task>/<dataset>/<scene>``.

    Keyed by dataset and scene only, so a scene shared by two tasks (SfM/dtu and MVRecon/dtu) has one set of priors.
    """
    data_path = Path(data_path)
    root = Path(benchmark_root) if benchmark_root is not None else data_path.parents[2]
    return sparse_depth_root(root) / data_path.parent.name / data_path.name


def prior_path(data_path: Path, image_path: str | Path, benchmark_root: Path | None = None) -> Path:
    return scene_dir(data_path, benchmark_root) / f"{Path(image_path).stem}.npz"


def write_points(path: Path, xy: np.ndarray, z: np.ndarray, size: tuple[int, int], source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, xy=np.asarray(xy, np.float32).reshape(-1, 2), z=np.asarray(z, np.float32).reshape(-1),
                        size=np.asarray(size, np.int32), source=np.asarray(source))


def read_points(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int], str]:
    with np.load(path) as data:
        size = np.asarray(data["size"]).reshape(-1)
        return data["xy"], data["z"], (int(size[0]), int(size[1])), str(data["source"])


def block_size(image_size: tuple[int, int], model_long_side: int | None) -> int:
    """Side of the square each point is drawn as, so it survives a nearest-neighbour resize to ``model_long_side``."""
    if not model_long_side:
        return 1
    return max(1, math.ceil(max(image_size) / float(model_long_side)))


def splat_sparse_depth(
    xy: np.ndarray, z: np.ndarray, size: tuple[int, int], *, block: int = 1, target_size: tuple[int, int] | None = None
) -> np.ndarray:
    """A dense [H, W] float32 map (0 = no depth) with each point drawn as a ``block`` x ``block`` square of its depth.

    ``size`` is the (width, height) the points' coordinates refer to; ``target_size`` (default: ``size``) the map's
    size, the coordinates being scaled to it. Where squares overlap, the nearest point wins.
    """
    width, height = size
    out_w, out_h = target_size or size
    xy = np.asarray(xy, np.float64).reshape(-1, 2) * [out_w / float(width), out_h / float(height)]
    z = np.asarray(z, np.float32).reshape(-1)
    keep = np.isfinite(z) & (z > 0) & np.all(np.isfinite(xy), axis=1)
    xy, z = xy[keep], z[keep]
    depth = np.full((out_h, out_w), np.inf, np.float32)
    if len(z):
        cols = np.floor(xy[:, 0]).astype(np.int64)
        rows = np.floor(xy[:, 1]).astype(np.int64)
        half = (block - 1) // 2
        for dy in range(-half, block - half):
            for dx in range(-half, block - half):
                r, c = rows + dy, cols + dx
                inside = (r >= 0) & (r < out_h) & (c >= 0) & (c < out_w)
                np.minimum.at(depth, (r[inside], c[inside]), z[inside])
    depth[~np.isfinite(depth)] = 0.0
    return depth


def load_sparse_depth(
    data_path: Path,
    image_path: str | Path,
    *,
    image_size: tuple[int, int] | None = None,
    model_long_side: int | None = None,
) -> np.ndarray | None:
    """The sparse prior of one image as a dense map at ``image_size`` (default: the size the points refer to), points
    drawn as blocks for a model whose input's long side is ``model_long_side``; None if the image has no prior."""
    path = prior_path(data_path, image_path)
    if not path.is_file():
        return None
    xy, z, size, _source = read_points(path)
    target = tuple(image_size) if image_size else size
    return splat_sparse_depth(xy, z, size, block=block_size(target, model_long_side), target_size=target)
