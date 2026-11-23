# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Replay assets of the demo mode of FeedForward-Benchmark's dashboard.

The demo mode belongs to FeedForward-Benchmark's internal dashboard (this
repository's dashboard has no demo mode): it replays one method's prediction
on one scene, frame by frame.

A demo replays a prediction as the method saw its input: each frame's camera
(frustum), its RGB and predicted depth, and the frame's points unprojected
from that depth, added to the cloud in frame order. Everything is written once
per case into
<eval root>/viewer_assets/demos/<suite>/<method>/<task>/<dataset>/<scene>/:

    demo.json       frames (camera-to-world, intrinsics, the frame's slice of
                    the point buffer), bounds, metadata
    points.u16      every frame's points, frame after frame, quantized to the
                    bounds (3 x uint16 per point)
    colors.u8       their RGB (3 x uint8 per point)
    confidence.f32  their predicted confidence (float32 per point, NaN where a
                    frame has none), when the prediction saved confidence maps
                    (confidences/); demo.json then has "confidence": true
    rgb/NNNN.webp   the frames' input images (small: the frustum texture and
                    the inset)
    depth/NNNN.webp the frames' predicted depth, on one color scale for the
                    whole demo

Sources (that dashboard's build_demo_assets.py reads them from the prediction
artifacts, no evaluation needed):

- SfM / SLAM / MVRecon: the predicted depth and the predicted cameras (the
  COLMAP model every adapter exports), in the method's own frame and scale.
- VideoDepth: predictions carry no cameras, so the frames are replayed in the
  camera frame (a fixed camera, each frame replacing the last) with the depth
  aligned to the GT exactly as scored (scale/shift per sequence), and the
  dataset's nominal intrinsics (`VIDEO_DEPTH_INTRINSICS`).

Sizes are bounded (`MAX_FRAMES` frames, `MAX_POINTS` points;
`UNPOSED_MAX_POINTS` for camera-frame replays), so long SLAM sequences replay
every k-th frame.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Callable, Iterable, Sequence

import numpy as np

from spatialarena.eval import viewer_assets

DEMO_ROOT = pathlib.Path("viewer_assets") / "demos"
DEMO_FILE = "demo.json"
POINTS_FILE = "points.u16"
COLORS_FILE = "colors.u8"
CONFIDENCE_FILE = "confidence.f32"
MAX_FRAMES = 240
MAX_POINTS = 1_000_000
# A camera-frame replay shows one frame at a time, so each frame gets more
# points.
UNPOSED_MAX_POINTS = 2_500_000
FRAME_POINTS = 30_000  # Per frame, before the MAX_POINTS budget.
# Depth is unprojected at this resolution (points per frame are subsampled
# anyway).
UNPROJECT_MAX_SIDE = 320
IMAGE_MAX_SIDE = 320
IMAGE_QUALITY = 78
# A pixel whose 3x3 neighbourhood spans more than this depth ratio is a depth
# edge: dropped.
EDGE_RATIO = 1.06
BOUNDS_PERCENTILES = (0.5, 99.5)
# Of the percentile box, on every side; points outside are dropped.
BOUNDS_MARGIN = 0.1
SAMPLE_SEED = 0

# Nominal pinhole intrinsics of the VideoDepth datasets, at the resolution
# given (fx, fy, cx, cy, width, height): Bonn's calibration, KITTI's 2011_09_26
# left color camera moved into the 1216 x 352 depth-selection crop, Sintel's
# usual focal length. Other datasets get a 60-degree horizontal field of view.
VIDEO_DEPTH_INTRINSICS = {
    "bonn": (542.822841, 542.576870, 315.593520, 237.756098, 640, 480),
    "kitti": (721.5377, 721.5377, 596.5593, 149.854, 1216, 352),
    "sintel": (688.0, 688.0, 511.5, 217.5, 1024, 436),
}
DEFAULT_HFOV_DEGREES = 60.0


@dataclasses.dataclass(frozen=True)
class DemoFrame:
    """One input frame: its name, image, depth loader and camera.

    Attributes:
        name: Frame name.
        image_path: The input image, if any.
        load_depth: Loads the predicted depth.
        intrinsic: 3 x 3 intrinsics, for an image of `size`.
        size: (height, width) the intrinsics refer to.
        c2w: Camera-to-world matrix; None for a camera-frame replay.
        load_confidence: Loads the predicted confidence, pixel-aligned with
            the depth, in the method's own scale; None without one.
    """

    name: str
    image_path: pathlib.Path | None
    load_depth: Callable[[], np.ndarray]
    intrinsic: np.ndarray
    size: tuple[int, int]
    c2w: np.ndarray | None = None
    load_confidence: Callable[[], np.ndarray] | None = None


def select_frames(count: int, limit: int = MAX_FRAMES) -> np.ndarray:
    """Returns evenly spaced frame indices, at most `limit` of them.

    Returns all frames when there are fewer.
    """
    if count <= limit:
        return np.arange(count)
    return np.unique(np.linspace(0, count - 1, limit).round().astype(int))


def nominal_intrinsics(dataset: str, shape: tuple[int, int]) -> np.ndarray:
    """Returns a VideoDepth dataset's nominal intrinsics at (height, width)."""
    height, width = shape
    nominal = VIDEO_DEPTH_INTRINSICS.get(dataset.lower())
    if nominal is None:
        focal = 0.5 * width / np.tan(np.radians(DEFAULT_HFOV_DEGREES) / 2.0)
        return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0],
                         [0.0, 0.0, 1.0]])
    fx, fy, cx, cy, nominal_width, nominal_height = nominal
    sx, sy = width / nominal_width, height / nominal_height
    return np.array([[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy],
                     [0.0, 0.0, 1.0]])


def scale_intrinsic(intrinsic: np.ndarray, source: tuple[int, int],
                    target: tuple[int, int]) -> np.ndarray:
    """Returns intrinsics of a (height, width) image resized to target."""
    scaled = np.asarray(intrinsic, dtype=np.float64).copy()
    scaled[0, :] *= target[1] / float(source[1])
    scaled[1, :] *= target[0] / float(source[0])
    return scaled


def resize_array(array: np.ndarray, shape: tuple[int, int],
                 nearest: bool) -> np.ndarray:
    """Returns array at shape (height, width).

    Uses nearest-neighbour interpolation for depth and area averaging for
    images.
    """
    import cv2  # pylint: disable=import-outside-toplevel

    if array.shape[:2] == tuple(shape):
        return array
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return cv2.resize(array, (shape[1], shape[0]), interpolation=interpolation)


def edge_free_mask(depth: np.ndarray, ratio: float = EDGE_RATIO) -> np.ndarray:
    """Returns the valid depth (finite, > 0) away from depth edges.

    At depth edges unprojection leaves "flying" points between surfaces.
    """
    valid = np.isfinite(depth) & (depth > 0)
    high = np.where(valid, depth, 0.0)
    low = np.where(valid, depth, np.inf)
    padded_high = np.pad(high, 1, mode="edge")
    padded_low = np.pad(low, 1, mode="edge")
    height, width = depth.shape
    local_max = np.zeros_like(high)
    local_min = np.full_like(low, np.inf)
    for dy in range(3):
        for dx in range(3):
            local_max = np.maximum(local_max, padded_high[dy:dy + height,
                                                          dx:dx + width])
            local_min = np.minimum(local_min, padded_low[dy:dy + height,
                                                         dx:dx + width])
    return valid & (local_max <= ratio * local_min)


def unproject(depth: np.ndarray, intrinsic: np.ndarray,
              mask: np.ndarray) -> np.ndarray:
    """Returns the camera-frame points (OpenCV axes) of the masked pixels."""
    rows, cols = np.nonzero(mask)
    z = depth[rows, cols].astype(np.float64)
    x = (cols + 0.5 - intrinsic[0, 2]) / intrinsic[0, 0] * z
    y = (rows + 0.5 - intrinsic[1, 2]) / intrinsic[1, 1] * z
    return np.stack([x, y, z], axis=1)


def load_rgb(path: pathlib.Path | None, shape: tuple[int,
                                                     int]) -> np.ndarray | None:
    """Returns the frame's image at shape, or None when it cannot be read."""
    if path is None or not pathlib.Path(path).is_file():
        return None
    from PIL import Image  # pylint: disable=import-outside-toplevel

    with Image.open(path) as image:
        return resize_array(np.asarray(image.convert("RGB")),
                            shape,
                            nearest=False)


def demo_depth_range(depths: Iterable[np.ndarray]) -> tuple[float, float]:
    """Returns one color scale for every frame.

    The scale is the 2nd-98th percentile of the frames' valid depth.
    """
    values = [depth[np.isfinite(depth) & (depth > 0)] for depth in depths]
    samples = [
        value[::max(1, value.size // 20000)] for value in values if value.size
    ]
    values = np.concatenate(samples or [[1.0]])
    low, high = np.percentile(values, [2.0, 98.0])
    return float(low), float(max(high, low * 1.01 + 1e-6))


@dataclasses.dataclass
class _FramePoints:
    """The points of one frame, in world (or camera) coordinates.

    Attributes:
        points: (N, 3) float32 points.
        colors: (N, 3) uint8 colors.
        confidence: (N,) float32 confidence; NaN where the frame has no
            confidence map.
    """

    points: np.ndarray
    colors: np.ndarray
    confidence: np.ndarray


def _frame_points(frame: DemoFrame, depth: np.ndarray, rgb: np.ndarray | None,
                  budget: int, rng: np.random.Generator) -> _FramePoints:
    """Unprojects a frame's edge-free depth, subsampled to budget points."""
    intrinsic = scale_intrinsic(frame.intrinsic, frame.size, depth.shape)
    mask = edge_free_mask(depth)
    points = unproject(depth, intrinsic, mask)
    if rgb is not None:
        colors = rgb[mask]
    else:
        colors = np.full((points.shape[0], 3), 160, dtype=np.uint8)
    confidence = np.full(points.shape[0], np.nan, dtype=np.float32)
    if frame.load_confidence is not None:
        confidence_map = np.asarray(frame.load_confidence(), dtype=np.float32)
        if confidence_map.ndim == 3:
            confidence_map = confidence_map[..., 0]
        confidence = resize_array(confidence_map, depth.shape,
                                  nearest=True)[mask]
    if points.shape[0] > budget:
        keep = np.sort(rng.choice(points.shape[0], budget, replace=False))
        points = points[keep]
        colors = colors[keep]
        confidence = confidence[keep]
    if frame.c2w is not None:
        points = points @ frame.c2w[:3, :3].T + frame.c2w[:3, 3]
    return _FramePoints(points.astype(np.float32), colors.astype(np.uint8),
                        confidence.astype(np.float32))


def robust_bounds(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns the box holding the bulk of the points.

    The box is the percentile box plus a margin, so outliers do not waste
    quantization precision.
    """
    sample = points[::max(1, points.shape[0] // 200_000)]
    low, high = np.percentile(sample, BOUNDS_PERCENTILES, axis=0)
    margin = (high - low) * BOUNDS_MARGIN + 1e-6
    return low - margin, high + margin


def write_demo(out_dir: pathlib.Path, frames: Sequence[DemoFrame], *,
               metadata: dict, posed: bool, intrinsics_source: str) -> dict:
    """Writes one demo from its frames, already in replay order.

    Args:
        out_dir: The demo's directory.
        frames: The frames, in replay order.
        metadata: Entries copied to the front of demo.json.
        posed: Whether the frames have cameras (False: camera-frame replay).
        intrinsics_source: Where the intrinsics come from, for demo.json.

    Returns:
        The demo.json payload.

    Raises:
        ValueError: There are no frames.
    """
    if not frames:
        raise ValueError("demo has no frames")
    picked = [frames[index] for index in select_frames(len(frames))]
    max_points = MAX_POINTS if posed else UNPOSED_MAX_POINTS
    budget = max(1000, min(FRAME_POINTS, max_points // len(picked)))
    rng = np.random.default_rng(SAMPLE_SEED)
    depths = []
    for frame in picked:
        depth = np.asarray(frame.load_depth(), dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        depths.append(
            resize_array(depth,
                         viewer_assets.downsample_shape(depth.shape,
                                                        UNPROJECT_MAX_SIDE),
                         nearest=True))
    depth_range = demo_depth_range(depths)

    clouds, entries = [], []
    for index, (frame, depth) in enumerate(zip(picked, depths)):
        rgb = load_rgb(frame.image_path, depth.shape)
        clouds.append(_frame_points(frame, depth, rgb, budget, rng))
        image_shape = viewer_assets.downsample_shape(depth.shape,
                                                     IMAGE_MAX_SIDE)
        if rgb is not None:
            viewer_assets.save_webp(
                resize_array(rgb, image_shape, nearest=False),
                out_dir / "rgb" / f"{index:04d}.webp", IMAGE_QUALITY)
        viewer_assets.save_webp(
            viewer_assets.colorize_depth(
                resize_array(depth, image_shape, nearest=True), depth_range),
            out_dir / "depth" / f"{index:04d}.webp", IMAGE_QUALITY)
        intrinsic = scale_intrinsic(frame.intrinsic, frame.size, depth.shape)
        valid = depth[np.isfinite(depth) & (depth > 0)]
        c2w = frame.c2w if frame.c2w is not None else np.eye(4)
        median_depth = float(np.median(valid)) if valid.size else None
        entries.append({
            "name": frame.name,
            "c2w": [
                round(float(value), 6)
                for value in np.asarray(c2w, dtype=np.float64).reshape(-1)
            ],
            "intrinsics": [
                round(float(intrinsic[0, 0]), 3),
                round(float(intrinsic[1, 1]), 3),
                round(float(intrinsic[0, 2]), 3),
                round(float(intrinsic[1, 2]), 3)
            ],
            "size": [int(depth.shape[1]),
                     int(depth.shape[0])],
            "median_depth": median_depth,
            "rgb": rgb is not None,
        })

    all_points = np.concatenate([cloud.points for cloud in clouds])
    low, high = robust_bounds(all_points)
    span = np.maximum(high - low, 1e-9)
    start = 0
    quantized, colors, confidences = [], [], []
    for entry, cloud in zip(entries, clouds):
        inside = np.all((cloud.points >= low) & (cloud.points <= high), axis=1)
        values = np.round(
            (cloud.points[inside] - low) / span * 65535.0).astype("<u2")
        quantized.append(values)
        colors.append(cloud.colors[inside])
        confidences.append(cloud.confidence[inside])
        entry["start"] = start
        entry["count"] = int(values.shape[0])
        start += int(values.shape[0])
    viewer_assets.write_array(out_dir / POINTS_FILE, np.concatenate(quantized))
    viewer_assets.write_array(out_dir / COLORS_FILE, np.concatenate(colors))
    confidence = np.concatenate(confidences)
    has_confidence = bool(np.isfinite(confidence).any())
    if has_confidence:
        viewer_assets.write_array(out_dir / CONFIDENCE_FILE,
                                  confidence.astype("<f4"))
    else:
        (out_dir / CONFIDENCE_FILE).unlink(missing_ok=True)

    # Keys already in metadata keep their position, as in a dict merge.
    payload = dict(metadata)
    payload["posed"] = posed
    payload["intrinsics_source"] = intrinsics_source
    payload["frames_total"] = len(frames)
    payload["points"] = start
    payload["confidence"] = has_confidence
    payload["bounds"] = [[float(value) for value in low],
                         [float(value) for value in high]]
    payload["depth_range"] = list(depth_range)
    if posed:
        payload["up"] = _up_vector(entries)
    else:
        payload["up"] = [0.0, -1.0, 0.0]
    payload["frames"] = entries
    viewer_assets.write_json(out_dir / DEMO_FILE, payload)
    return payload


def _up_vector(entries: list[dict]) -> list[float]:
    """Returns the world's up: the cameras' mean "-y" (OpenCV image up).

    This holds for the usual upright capture.
    """
    ups = np.array([[-entry["c2w"][1], -entry["c2w"][5], -entry["c2w"][9]]
                    for entry in entries])
    mean = ups.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm > 1e-6:
        up = mean / norm
    else:
        up = np.array([0.0, -1.0, 0.0])
    return [float(value) for value in up]
