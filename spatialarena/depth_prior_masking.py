# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The masked-depth protocol of the masked-depth tracks.

The tracks are `rgb_masked_depth` and `rgb_masked_depth_cameras`. `rgb_depth`
hands a method the dataset's GT depth as it is. The masked-depth tracks hand
it only part of that depth, which is what a depth-completion model meets in
practice (a phone LiDAR, a stereo matcher's holes, SfM points): the method has
to fill in the rest. Every method of these tracks gets the same input, so the
masking is defined here, not in an adapter.

Per depth map (at its source resolution, before any method resizes it):

1. The map is divided into a grid of square cells, `grid` cells along its
   longer side (default 32: about one 16 px ViT patch per cell at a 512 px
   model input).
2. A fraction `ratio` of the cells is dropped (default 0.5): their pixels
   become invalid (0). The cells are put in a fixed random order by a
   generator seeded from the grid, the scene name and the image's path inside
   the scene, and the first `round(ratio * cells)` of that order are dropped.
   The mask is therefore the same on every run, for every method, at every
   frame interval, and for a scene shared by two tasks (SfM/dtu and
   MVRecon/dtu are one directory); and the masks of one image are nested
   across ratios (the cells dropped at 0.5 are also dropped at 0.7), so a
   ratio sweep only ever removes more of the same input.
3. Pixels that were already invalid stay invalid. Datasets whose GT depth is
   itself sparse (DTU's SfM points, ETH3D's laser scan, KITTI's LiDAR)
   therefore get a sparser input still, and dense datasets get a masked one.

The default ratio is 0.5: half of each depth prior's patches are hidden.

Both settings can be given from outside through the environment
(`BENCHMARK_DEPTH_MASK_RATIO`, in [0, 1); `BENCHMARK_DEPTH_MASK_GRID`, a
positive integer), e.g. per model entry of a run config:

    models:
      - {name: PI3_X, track: rgb_masked_depth,
         env: {BENCHMARK_DEPTH_MASK_RATIO: "0.7"}}

A setting other than the default is part of the method ID (`method_variant`:
`PI3_X-mask0.7_rgb_masked_depth`), so each setting has its own report folder,
and of the prediction cache key (`protocol`), so no setting reuses another's
predictions. `protocol` is also written into each case's adapter metrics.

Scoring is unchanged from the other tracks (all GT-valid pixels, after each
task's usual alignment), so a method that copies its input still has to
predict the dropped cells of every map.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import sys
from typing import Mapping

import numpy as np

TRACK_RGB_MASKED_DEPTH = "rgb_masked_depth"
TRACK_RGB_MASKED_DEPTH_CAMERAS = "rgb_masked_depth_cameras"
MASKED_DEPTH_TRACKS = frozenset(
    {TRACK_RGB_MASKED_DEPTH, TRACK_RGB_MASKED_DEPTH_CAMERAS})

RATIO_ENV = "BENCHMARK_DEPTH_MASK_RATIO"
GRID_ENV = "BENCHMARK_DEPTH_MASK_GRID"
DEFAULT_RATIO = 0.5
DEFAULT_GRID = 32
_VERSION = "v1"


def mask_settings(env: Mapping[str, str] | None = None) -> tuple[float, int]:
    """Returns the mask `(ratio, grid)` set in an environment.

    Args:
        env: The environment; the process environment by default. Unset or
            empty values take the defaults.

    Returns:
        The drop ratio and the grid size.

    Raises:
        ValueError: If a value does not parse or is out of range.
    """
    if env is None:
        env = os.environ
    ratio_text = str(env.get(RATIO_ENV, "") or "").strip()
    grid_text = str(env.get(GRID_ENV, "") or "").strip()
    try:
        ratio = float(ratio_text) if ratio_text else DEFAULT_RATIO
        grid = int(grid_text) if grid_text else DEFAULT_GRID
    except ValueError as exc:
        raise ValueError(
            f"{RATIO_ENV}={ratio_text!r} / {GRID_ENV}={grid_text!r}: {exc}"
        ) from exc
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"{RATIO_ENV} must be in [0, 1), got {ratio}")
    if grid < 1:
        raise ValueError(f"{GRID_ENV} must be a positive integer, got {grid}")
    return ratio, grid


def protocol(env: Mapping[str, str] | None = None) -> str:
    """Returns the protocol name, e.g. `cells32-drop0.5-v1`.

    It is recorded with each case and part of its cache key.
    """
    ratio, grid = mask_settings(env)
    return f"cells{grid}-drop{ratio:g}-{_VERSION}"


def method_variant(track: str, env: Mapping[str, str] | None = None) -> str:
    """Returns the method-name suffix of a masked-depth run's setting.

    Args:
        track: The input track.
        env: The environment with the mask settings.

    Returns:
        `-mask0.7` or `-mask0.7-grid16` for a non-default setting; empty for
        the default setting and for every other track.
    """
    if track not in MASKED_DEPTH_TRACKS:
        return ""
    ratio, grid = mask_settings(env)
    variant = f"-mask{ratio:g}" if ratio != DEFAULT_RATIO else ""
    if grid != DEFAULT_GRID:
        variant = f"-mask{ratio:g}-grid{grid}"
    return variant


# pylint: disable-next=redefined-outer-name
def mask_seed(scene_key: str, image_key: str, grid: int = DEFAULT_GRID) -> int:
    """Returns the stable 64-bit seed of one image's cell order.

    The ratio is not part of it, so the masks of one image are nested across
    ratios.

    Args:
        scene_key: The scene directory name.
        image_key: The image's path in the scene (`image_key()`).
        grid: Cells along the longer side.

    Returns:
        The seed.
    """
    seed_text = f"cells{grid}-{_VERSION}:{scene_key}:{image_key}"
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def cell_keep_mask(height: int,
                   width: int,
                   seed: int,
                   ratio: float = DEFAULT_RATIO,
                   grid: int = DEFAULT_GRID) -> np.ndarray:
    """Returns where the protocol keeps the depth: the cells not dropped.

    Args:
        height: Map height.
        width: Map width.
        seed: Seed of the cell order (`mask_seed`).
        ratio: Fraction of cells dropped.
        grid: Cells along the longer side.

    Returns:
        A boolean [H, W] mask.
    """
    cell = max(1, int(np.ceil(max(height, width) / grid)))
    rows = int(np.ceil(height / cell))
    cols = int(np.ceil(width / cell))
    num_cells = rows * cols
    order = np.random.default_rng(seed).permutation(num_cells)
    keep_cells = np.ones(num_cells, dtype=bool)
    keep_cells[order[:int(round(ratio * num_cells))]] = False
    keep_rows = np.repeat(keep_cells.reshape(rows, cols), cell, axis=0)
    keep = np.repeat(keep_rows, cell, axis=1)
    return keep[:height, :width]


def image_key(data_path: pathlib.Path, image_path: str | pathlib.Path) -> str:
    """Returns the image's path relative to the scene directory.

    An image outside the scene directory is keyed by its file name.
    """
    path = pathlib.Path(image_path)
    scene_path = pathlib.Path(data_path).resolve()
    try:
        return path.resolve().relative_to(scene_path).as_posix()
    except ValueError:
        return path.name


def mask_depth_prior(
    depth: np.ndarray,
    data_path: pathlib.Path,
    image_path: str | pathlib.Path,
    *,
    env: Mapping[str, str] | None = None,
) -> np.ndarray:
    """Drops the protocol's cells of a depth prior.

    Args:
        depth: The depth map, [H, W] or [H, W, C] (the first channel is used).
        data_path: The scene folder.
        image_path: The image the map belongs to.
        env: The environment with the mask settings.

    Returns:
        The float32 map with the dropped cells set to 0; invalid pixels (<= 0
        or non-finite) also become 0.
    """
    ratio, grid = mask_settings(env)
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    seed = mask_seed(
        pathlib.Path(data_path).name, image_key(data_path, image_path), grid)
    keep = cell_keep_mask(*depth.shape, seed, ratio=ratio, grid=grid)
    valid = np.isfinite(depth) & (depth > 0) & keep
    return np.where(valid, depth, 0.0).astype(np.float32)


def masked_valid_fraction(depth: np.ndarray) -> float:
    """Returns the share of pixels still valid in a (masked) depth map."""
    depth = np.asarray(depth)
    if not depth.size:
        return 0.0
    return float((np.isfinite(depth) & (depth > 0)).mean())


def main() -> None:
    """Prints the method variant or the protocol for run_method_case_list.sh.

    Usage: `python -m spatialarena.depth_prior_masking {variant,protocol}
    TRACK`.
    """
    if len(sys.argv) != 3 or sys.argv[1] not in ("variant", "protocol"):
        raise SystemExit("usage: python -m spatialarena.depth_prior_masking "
                         "{variant,protocol} TRACK")
    if sys.argv[1] == "variant":
        print(method_variant(sys.argv[2]))
    else:
        print(protocol())


if __name__ == "__main__":
    main()
