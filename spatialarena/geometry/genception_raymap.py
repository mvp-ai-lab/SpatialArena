# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Decodes camera poses and intrinsics from a GenCeption "raymap" prediction.

The decode follows the official evaluation: it is ported from
representations4d/genception/evals/eval_camera_pose.py, released by the
GenCeption authors with their answer to
https://github.com/google-deepmind/representations4d/issues/10 (closed
2026-11-10). The prompt is `"Input: RGB. Output: Raymap."` (official
embedding: `assets/genception_prompt_embeddings/raymap.npy`).

Layout of one 480x832 raymap frame (values in [0, 1]; `2 * v - 1` gives the
encoded vector), in 16-px patches:

- Origin block: patch rows 10-20, columns 18-34 (pixel rows 160-320, columns
  288-544). It holds the camera centre, log-scaled:
  `2 * v - 1 = sign(o) * log(1 + |o|)`, so decoding applies
  `sign(x) * (exp(|x|) - 1)`.
- Everything else holds world-space ray directions (`2 * v - 1`,
  normalized).
- Decoding leaves a 2-patch margin on both sides of the block edge: the
  camera centre is the median over the block shrunk by 2 patches, and the
  directions come from outside the block grown by 2 patches.

Per frame, the centre is that median and the rotation comes from the
directions by one of two solvers: `robust2` (forward axis = median ray, x
axis from the column-weighted rays by Gram-Schmidt) or `svd` (Kabsch against
pinhole rays with focal `max(H, W)`). One pinhole `K` is then fitted for the
whole clip, by a two-step MAD-robust line fit of pixel coordinates to the
camera-frame ray slopes. The official evaluation reports both the raw
per-frame poses and poses smoothed over time (window 5: median then mean
filter on the centres, mean then SO(3) projection on the rotations); both are
returned here.
"""

from __future__ import annotations

import dataclasses

import numpy as np

RAYMAP_HEIGHT = 480
RAYMAP_WIDTH = 832
PATCH_SIZE = 16
# Origin block in patches: (row start, row end, column start, column end).
ORIGIN_BLOCK_PATCHES = (10, 20, 18, 34)
MARGIN_PATCHES = 2
SMOOTHING_WINDOW = 5
SOLVERS = ("robust2", "svd")
DEFAULT_SOLVER = "robust2"


@dataclasses.dataclass
class RaymapPoseDecode:
    """The decode of one clip.

    Attributes:
        intrinsics: The (3, 3) pinhole K shared by all frames.
        camera_to_world: (T, 4, 4) poses smoothed over time.
        camera_to_world_raw: (T, 4, 4) per-frame poses.
        solver: The rotation solver used, one of `SOLVERS`.
    """

    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    camera_to_world_raw: np.ndarray
    solver: str

    def poses(self, smoothed: bool = True) -> np.ndarray:
        """Returns the poses.

        Args:
            smoothed: Whether to return the smoothed poses (the official
                headline numbers) instead of the raw per-frame ones.
        """
        if smoothed:
            return self.camera_to_world
        return self.camera_to_world_raw


def _block_pixels(top: int, bottom: int, left: int,
                  right: int) -> tuple[slice, slice]:
    """Returns the pixel slices of a block given in patches."""
    rows = slice(top * PATCH_SIZE, bottom * PATCH_SIZE)
    columns = slice(left * PATCH_SIZE, right * PATCH_SIZE)
    return rows, columns


def raymap_masks(height: int = RAYMAP_HEIGHT,
                 width: int = RAYMAP_WIDTH) -> tuple[np.ndarray, np.ndarray]:
    """Returns the pixels the camera centre and the rotation are read from.

    Args:
        height: The raymap frame height.
        width: The raymap frame width.

    Returns:
        (origin_mask, direction_mask), each an (H, W) bool array: the origin
        block shrunk by the 2-patch margin, and the complement of the block
        grown by it.

    Raises:
        ValueError: If the frame is smaller than the raymap layout.
    """
    top, bottom, left, right = ORIGIN_BLOCK_PATCHES
    if bottom * PATCH_SIZE > height or right * PATCH_SIZE > width:
        raise ValueError(f"raymap frame {height}x{width} is smaller than the "
                         f"{RAYMAP_HEIGHT}x{RAYMAP_WIDTH} layout")
    origin_mask = np.zeros((height, width), dtype=bool)
    origin_mask[_block_pixels(top + MARGIN_PATCHES, bottom - MARGIN_PATCHES,
                              left + MARGIN_PATCHES,
                              right - MARGIN_PATCHES)] = True
    direction_mask = np.ones((height, width), dtype=bool)
    direction_mask[_block_pixels(top - MARGIN_PATCHES, bottom + MARGIN_PATCHES,
                                 left - MARGIN_PATCHES,
                                 right + MARGIN_PATCHES)] = False
    return origin_mask, direction_mask


def unscale_origin(encoded: np.ndarray) -> np.ndarray:
    """Inverts the log scaling of the origin block.

    Args:
        encoded: The encoded values `x = 2 * v - 1`.

    Returns:
        `sign(x) * (exp(|x|) - 1)`.
    """
    return np.sign(encoded) * np.expm1(np.abs(encoded))


def _robust_line_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fits `y ~ m * x + c` and returns (m, c).

    A least-squares fit, then a refit on the points within
    max(5 * MAD, 1e-3) of it (MAD over every 10th residual, as upstream).
    """

    def fit(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
        n = len(xs)
        sum_x, sum_y = np.sum(xs), np.sum(ys)
        slope = (n * np.dot(xs, ys) - sum_x * sum_y) / (n * np.dot(xs, xs) -
                                                        sum_x * sum_x + 1e-12)
        return float(slope), float((sum_y - slope * sum_x) / n)

    slope, intercept = fit(x, y)
    residuals = np.abs(slope * x + intercept - y)
    inliers = residuals < max(5.0 * np.median(residuals[::10]), 1e-3)
    return fit(x[inliers], y[inliers])


def _window(length: int, window: int) -> int:
    """Returns the window clipped to an odd size of at most `length`."""
    return min(window, length if length % 2 else length - 1)


def _smooth_centres(centres: np.ndarray,
                    window: int = SMOOTHING_WINDOW) -> np.ndarray:
    """Smooths a (T, 3) trajectory by a sliding median then a sliding mean.

    Both filters pad the trajectory at its ends by edge values.
    """
    length = len(centres)
    if length < 3:
        return centres
    w = _window(length, window)
    pad = w // 2
    padded = np.pad(centres, ((pad, pad), (0, 0)), mode="edge")
    median = np.stack(
        [np.median(padded[i:i + w], axis=0) for i in range(length)])
    padded = np.pad(median, ((pad, pad), (0, 0)), mode="edge")
    return np.stack([np.mean(padded[i:i + w], axis=0) for i in range(length)])


def _project_to_so3(matrix: np.ndarray) -> np.ndarray:
    """Returns the rotation closest to a (3, 3) matrix."""
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u = u.copy()
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def _smooth_rotations(rotations: np.ndarray,
                      window: int = SMOOTHING_WINDOW) -> np.ndarray:
    """Smooths (T, 3, 3) rotations by a sliding mean projected onto SO(3).

    The mean pads the sequence at its ends by edge values.
    """
    length = len(rotations)
    if length < 3:
        return rotations
    w = _window(length, window)
    pad = w // 2
    padded = np.pad(rotations, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    return np.stack([
        _project_to_so3(np.mean(padded[i:i + w], axis=0)) for i in range(length)
    ])


def _rotation_robust2(rays: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Returns the camera-to-world rotation from world rays.

    z is the median ray; x is the column-centred ray sum, orthogonalized
    against z; y = z x x.
    """
    z_axis = np.median(rays, axis=0)
    z_axis /= np.linalg.norm(z_axis) + 1e-6
    x_raw = (cols - np.mean(cols)) @ rays
    x_axis = x_raw - np.dot(x_raw, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis) + 1e-6
    return np.stack([x_axis, np.cross(z_axis, x_axis), z_axis], axis=1)


def _rotation_svd(rays: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                  height: int, width: int) -> np.ndarray:
    """Returns the camera-to-world rotation by Kabsch.

    The rotation R minimizes ||proxy - rays @ R|| between pinhole rays
    (focal max(H, W), centre (W/2, H/2)) and the world rays.
    """
    proxy = np.stack([
        cols - width / 2.0,
        rows - height / 2.0,
        np.full(cols.shape, float(max(height, width))),
    ],
                     axis=1)
    proxy /= np.linalg.norm(proxy, axis=1, keepdims=True) + 1e-6
    u, _, vt = np.linalg.svd(rays.T @ proxy)
    return u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt


def decode_poses_from_raymap(raymap_video: np.ndarray,
                             *,
                             solver: str = DEFAULT_SOLVER) -> RaymapPoseDecode:
    """Decodes the poses and the shared intrinsics of a raymap video.

    Args:
        raymap_video: (T, 480, 832, 3) raymap video with values in [0, 1]
            (GenCeption's float output, not quantized).
        solver: The rotation solver, one of `SOLVERS`.

    Returns:
        The smoothed and raw poses and the fitted intrinsics.

    Raises:
        ValueError: If the video shape or the solver is invalid.
    """
    if raymap_video.ndim != 4 or raymap_video.shape[-1] != 3:
        raise ValueError("expected a [T, H, W, 3] raymap video, got shape "
                         f"{raymap_video.shape}")
    if solver not in SOLVERS:
        raise ValueError(
            f"unknown solver {solver!r}; expected one of {SOLVERS}")
    _, height, width, _ = raymap_video.shape
    origin_mask, direction_mask = raymap_masks(height, width)
    rows, cols = (grid[direction_mask].astype(np.float64)
                  for grid in np.mgrid[:height, :width])

    centres, rotations, slopes_x, slopes_y = [], [], [], []
    for frame in raymap_video:
        encoded = 2.0 * frame.astype(np.float64) - 1.0
        centres.append(np.median(unscale_origin(encoded[origin_mask]), axis=0))
        rays = encoded[direction_mask]
        rays = rays / (np.linalg.norm(rays, axis=-1, keepdims=True) + 1e-6)
        if solver == "robust2":
            rotation = _rotation_robust2(rays, cols)
        else:
            rotation = _rotation_svd(rays, rows, cols, height, width)
        rotations.append(rotation)
        rays_camera = rays @ rotation
        slopes_x.append(rays_camera[:, 0] / (rays_camera[:, 2] + 1e-8))
        slopes_y.append(rays_camera[:, 1] / (rays_camera[:, 2] + 1e-8))

    num_frames = len(raymap_video)
    fx, cx = _robust_line_fit(np.concatenate(slopes_x),
                              np.tile(cols, num_frames))
    fy, cy = _robust_line_fit(np.concatenate(slopes_y),
                              np.tile(rows, num_frames))
    centres_raw, rotations_raw = np.stack(centres), np.stack(rotations)

    def compose(rotation_seq: np.ndarray, centre_seq: np.ndarray) -> np.ndarray:
        poses = np.repeat(np.eye(4)[None], num_frames, axis=0)
        poses[:, :3, :3], poses[:, :3, 3] = rotation_seq, centre_seq
        return poses

    return RaymapPoseDecode(
        intrinsics=np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]),
        camera_to_world=compose(_smooth_rotations(rotations_raw),
                                _smooth_centres(centres_raw)),
        camera_to_world_raw=compose(rotations_raw, centres_raw),
        solver=solver,
    )
