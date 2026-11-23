# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Sparse depth priors for the sparse-depth tracks.

The tracks are `rgb_sparse_depth` and `rgb_sparse_depth_cameras`. `rgb_depth`
and `rgb_masked_depth` hand a method (part of) the dataset's dense GT depth.
The sparse-depth tracks hand it the kind of sparse depth a real system has:

- **LiDAR** (KITTI, VideoDepth): each frame's raw Velodyne scan, projected
  into the left colour camera (`image_02`) with KITTI's calibration
  (`P_rect_02 @ R_rect_00 @ Tr_velo_to_cam`): about 4-5% of the pixels,
  against the accumulated GT's ~20%.
- **SfM points** (DTU, ETH3D, Tanks and Temples on MVRecon; those plus IMC and
  TexturePoorSfM on SfM): the keypoints each image observes in the scene's
  COLMAP model (`points2D` with a 3D point), at the depth of that 3D point in
  the image's camera, typically 0.1-1% of the pixels. These points come from
  the same reconstruction the SfM poses are scored against, so SfM results on
  these tracks are SfM-assisted, not pose estimation.

The priors are materialized once
(`scripts/preprocess/materialize_sparse_depth_priors.py`) as point lists, one
`.npz` per image under
`<benchmark_root>/_sparse_depth/<dataset>/<scene>/<image stem>.npz` (`xy`:
[N, 2] pixel coordinates in the image as stored, `z`: [N] depth in the
dataset's GT units, `size`: (width, height), `source`: `"lidar"` or `"sfm"`).
Points rather than images, so that no point is lost before a method resizes
its input: `splat_sparse_depth` draws each point as a small square block (one
value, never interpolated) sized so that it survives a nearest-neighbour
resize to the method's input resolution as about one pixel.

Scoring is unchanged (every GT-valid pixel, after each task's usual
alignment).
"""

from __future__ import annotations

import math
import pathlib

import numpy as np

TRACK_RGB_SPARSE_DEPTH = "rgb_sparse_depth"
TRACK_RGB_SPARSE_DEPTH_CAMERAS = "rgb_sparse_depth_cameras"
SPARSE_DEPTH_TRACKS = frozenset(
    {TRACK_RGB_SPARSE_DEPTH, TRACK_RGB_SPARSE_DEPTH_CAMERAS})
SPARSE_DEPTH_DIRNAME = "_sparse_depth"
PROTOCOL = "sparse-v1"


def sparse_depth_root(benchmark_root: pathlib.Path) -> pathlib.Path:
    """Returns the folder of all sparse priors of a benchmark root."""
    return pathlib.Path(benchmark_root) / SPARSE_DEPTH_DIRNAME


def scene_dir(data_path: pathlib.Path,
              benchmark_root: pathlib.Path | None = None) -> pathlib.Path:
    """Returns the prior folder of a scene.

    The folder is `<benchmark_root>/_sparse_depth/<dataset>/<scene>` for a
    scene folder `<root>/<Task>/<dataset>/<scene>`. It is keyed by dataset and
    scene only, so a scene shared by two tasks (SfM/dtu and MVRecon/dtu) has
    one set of priors.

    Args:
        data_path: The scene folder.
        benchmark_root: The benchmark root; by default the scene folder's
            `<root>`.

    Returns:
        The prior folder.
    """
    data_path = pathlib.Path(data_path)
    if benchmark_root is not None:
        root = pathlib.Path(benchmark_root)
    else:
        root = data_path.parents[2]
    return sparse_depth_root(root) / data_path.parent.name / data_path.name


def prior_path(data_path: pathlib.Path,
               image_path: str | pathlib.Path,
               benchmark_root: pathlib.Path | None = None) -> pathlib.Path:
    """Returns the prior file of one image of a scene."""
    stem = pathlib.Path(image_path).stem
    return scene_dir(data_path, benchmark_root) / f"{stem}.npz"


def write_points(path: pathlib.Path, xy: np.ndarray, z: np.ndarray,
                 size: tuple[int, int], source: str) -> None:
    """Writes the prior points of one image.

    Args:
        path: The prior file.
        xy: [N, 2] pixel coordinates.
        z: [N] depths.
        size: (width, height) the coordinates refer to.
        source: "lidar" or "sfm".
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path,
                        xy=np.asarray(xy, np.float32).reshape(-1, 2),
                        z=np.asarray(z, np.float32).reshape(-1),
                        size=np.asarray(size, np.int32),
                        source=np.asarray(source))


def read_points(
        path: pathlib.Path
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], str]:
    """Reads the prior points of one image.

    Args:
        path: The prior file.

    Returns:
        The pixel coordinates, the depths, the (width, height) they refer to
        and the source.
    """
    with np.load(path) as data:
        size = np.asarray(data["size"]).reshape(-1)
        return (data["xy"], data["z"], (int(size[0]), int(size[1])),
                str(data["source"]))


def block_size(image_size: tuple[int, int], model_long_side: int | None) -> int:
    """Returns the side of the square each point is drawn as.

    The square survives a nearest-neighbour resize to `model_long_side` as
    about one pixel.

    Args:
        image_size: (width, height) of the map.
        model_long_side: Long side of the model input; 1 if not given.

    Returns:
        The side in pixels.
    """
    if not model_long_side:
        return 1
    return max(1, math.ceil(max(image_size) / float(model_long_side)))


def splat_sparse_depth(
        xy: np.ndarray,
        z: np.ndarray,
        size: tuple[int, int],
        *,
        block: int = 1,
        target_size: tuple[int, int] | None = None) -> np.ndarray:
    """Draws sparse points as a dense depth map.

    Each point is drawn as a `block` x `block` square of its depth. Where
    squares overlap, the nearest point wins.

    Args:
        xy: [N, 2] pixel coordinates.
        z: [N] depths; non-finite and non-positive ones are dropped.
        size: (width, height) the coordinates refer to.
        block: Side of the square of each point.
        target_size: (width, height) of the map, the coordinates being scaled
            to it; `size` by default.

    Returns:
        A dense [H, W] float32 map, 0 where there is no depth.
    """
    width, height = size
    out_w, out_h = target_size or size
    scale = [out_w / float(width), out_h / float(height)]
    xy = np.asarray(xy, np.float64).reshape(-1, 2) * scale
    z = np.asarray(z, np.float32).reshape(-1)
    keep = np.isfinite(z) & (z > 0) & np.all(np.isfinite(xy), axis=1)
    xy = xy[keep]
    z = z[keep]
    depth = np.full((out_h, out_w), np.inf, np.float32)
    if len(z):
        cols = np.floor(xy[:, 0]).astype(np.int64)
        rows = np.floor(xy[:, 1]).astype(np.int64)
        half = (block - 1) // 2
        for dy in range(-half, block - half):
            for dx in range(-half, block - half):
                block_rows = rows + dy
                block_cols = cols + dx
                inside = ((block_rows >= 0) & (block_rows < out_h) &
                          (block_cols >= 0) & (block_cols < out_w))
                np.minimum.at(depth, (block_rows[inside], block_cols[inside]),
                              z[inside])
    depth[~np.isfinite(depth)] = 0.0
    return depth


def load_sparse_depth(
    data_path: pathlib.Path,
    image_path: str | pathlib.Path,
    *,
    image_size: tuple[int, int] | None = None,
    model_long_side: int | None = None,
) -> np.ndarray | None:
    """Loads the sparse prior of one image as a dense map.

    Args:
        data_path: The scene folder.
        image_path: The image.
        image_size: (width, height) of the map; by default the size the points
            refer to.
        model_long_side: Long side of the model input; the points are drawn
            as blocks that survive the resize to it.

    Returns:
        The map, or None if the image has no prior.
    """
    path = prior_path(data_path, image_path)
    if not path.is_file():
        return None
    xy, z, size, _ = read_points(path)
    target = tuple(image_size) if image_size else size
    return splat_sparse_depth(xy,
                              z,
                              size,
                              block=block_size(target, model_long_side),
                              target_size=target)
