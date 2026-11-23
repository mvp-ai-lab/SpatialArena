# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Viewer assets written by the evaluators for the dashboard's comparison modes.

Each evaluator writes the assets of a case while it holds the aligned data, so
nothing is recomputed afterwards:

- `slam_trajectory_evaluator` / `sparse_reconstruction_evaluator`
  (`--viewer_dir`): `trajectory.json`, the GT and the prediction aligned
  exactly as the ATE (`aligned_trajectory`).
- `depth_prediction_evaluator` (`--viewer_dir`, `--viewer_scene_root`): per
  evaluated frame, the aligned prediction and its absolute relative error as
  WebP, plus `frames.json`. All methods of a scene are drawn on one depth
  color scale fixed by the scene's GT (2nd-98th percentile), so they compare
  directly and video playback shows temporal flicker.
- `mvrecon_evaluator` (`--viewer_dir`, `--viewer_scene_root`): the fused cloud
  in the GT frame (subsampled), its fused colors and per-point error (distance
  to the nearest GT point), plus `cloud.json`, and the fused views' cameras
  (`cameras.json`: predicted and GT, in the cloud's frame) and, where the
  prediction saved confidence maps, each point's predicted confidence
  (`confidence.f32`, `point_confidence`); for the scene, the GT points and,
  where the GT scan is colored, their colors (`gt_colors.u8`).

The per-case assets go into the case's report folder (`<case>/viewer/`), so
reruns overwrite them in place. Assets shared by every method of a scene (GT
and RGB frames, GT points) go to `<scene root>/<task>/<dataset>/<scene>/` and
are written by whichever evaluation reaches the scene first (atomically; every
method computes the same files). The dashboard collector indexes both
(`app/webpage/python/viewer_index.py`).
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
from typing import Mapping

import numpy as np

VIEWER_DIR_NAME = "viewer"
TRAJECTORY_FILE = "trajectory.json"
DEPTH_FRAMES_FILE = "frames.json"
CLOUD_FILE = "cloud.json"
SCENE_FILE = "scene.json"
CAMERAS_FILE = "cameras.json"
CONFIDENCE_FILE = "confidence.f32"

DEPTH_MAX_SIDE = 640
WEBP_QUALITY = 88
# Near red, far blue, as materialize_depth_visualizations.py.
DEPTH_COLORMAP = "Spectral"
DEPTH_RANGE_PERCENTILES = (2.0, 98.0)
DEPTH_RANGE_SAMPLE_FRAMES = 40
DEPTH_INVALID_RGB = (0, 0, 0)
ERROR_COLORMAP = "turbo"
# depth_prediction_evaluator.ERROR_VIS_MAX_ABSREL: the delta < 1.25 boundary.
ERROR_MAX_ABSREL = 0.25
ERROR_INVALID_RGB = (128, 128, 128)

THUMBNAIL_DIR = "thumbs"  # In the scene's shared folder: <image name>.webp.
THUMBNAIL_WIDTH = 320
THUMBNAIL_QUALITY = 80
THUMBNAIL_WORKERS = 8  # Threads; PIL releases the GIL while decoding.

CLOUD_MAX_POINTS = 500_000
GT_CLOUD_MAX_POINTS = 300_000
# point_confidence: the depth buffer that decides which views see a point
# (cells on the longer image side), and how far behind a cell's nearest point a
# point may lie and still count as seen (relative depth).
VISIBILITY_GRID = 160
VISIBILITY_TOLERANCE = 0.03
GT_POINTS_FILE = "gt_points.f32"
GT_COLORS_FILE = "gt_colors.u8"
CLOUD_SAMPLE_SEED = 0


def case_viewer_dir(root: pathlib.Path, flat: bool, method: str, task: str,
                    dataset: str, scene: str) -> pathlib.Path:
    """Returns a case's viewer folder.

    That is `root` itself for the engine's one-case calls (`flat`), else a
    folder nested by case.
    """
    if flat:
        return root
    return root / method / task / dataset / scene


def scene_asset_dir(scene_root: pathlib.Path, task: str, dataset: str,
                    scene: str) -> pathlib.Path:
    """Folder of the assets every method of a scene shares."""
    return scene_root / task / dataset / scene


def write_json(path: pathlib.Path,
               payload: Mapping,
               *,
               indent: int | None = None) -> None:
    """Writes JSON atomically (readers never see a partial file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if indent:
        separators = None
    else:
        separators = (",", ":")
    temporary.write_text(json.dumps(payload,
                                    indent=indent,
                                    separators=separators),
                         encoding="utf-8")
    temporary.replace(path)


def write_array(path: pathlib.Path, array: np.ndarray) -> None:
    """Writes a raw little-endian array atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    np.ascontiguousarray(array).tofile(temporary)
    temporary.replace(path)


def save_webp(rgb: np.ndarray,
              path: pathlib.Path,
              quality: int = WEBP_QUALITY) -> None:
    """Writes an RGB uint8 array as WebP atomically."""
    from PIL import Image  # pylint: disable=import-outside-toplevel

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{id(rgb)}.tmp")
    Image.fromarray(rgb).save(temporary,
                              format="WEBP",
                              quality=quality,
                              method=4)
    temporary.replace(path)


# ---- trajectories ------------------------------------------------------------


def write_trajectory(viewer_dir: pathlib.Path,
                     payload: Mapping) -> pathlib.Path:
    """Writes a case's aligned trajectory (`aligned_trajectory` payload)."""
    path = viewer_dir / TRAJECTORY_FILE
    write_json(path, payload)
    return path


def resolve_image(image_root: pathlib.Path, name: str) -> pathlib.Path | None:
    """Returns the input image of a COLMAP image name.

    It is looked up under the scene's image folder, else under its parent.
    """
    for candidate in (image_root / name, image_root.parent / name):
        if candidate.is_file():
            return candidate
    return None


def write_frame_thumbnails(scene_dir: pathlib.Path,
                           image_root: pathlib.Path,
                           names: list[str],
                           workers: int = THUMBNAIL_WORKERS) -> int:
    """Writes small WebP thumbnails of the frames that have none yet.

    The thumbnails (`thumbs/<name>.webp`) are shared by every method of the
    scene, so a later evaluation only checks that they exist.

    Returns:
        How many thumbnails were written.
    """
    from concurrent import futures  # pylint: disable=import-outside-toplevel

    from PIL import Image  # pylint: disable=import-outside-toplevel

    folder = scene_dir / THUMBNAIL_DIR
    todo = [name for name in names if not (folder / f"{name}.webp").is_file()]

    def write(name: str) -> bool:
        source = resolve_image(image_root, name)
        if source is None:
            return False
        with Image.open(source) as image:
            # Fast JPEG downscale while decoding.
            image.draft("RGB", (THUMBNAIL_WIDTH * 2, THUMBNAIL_WIDTH * 2))
            image = image.convert("RGB")
            height = max(1, round(image.height * THUMBNAIL_WIDTH / image.width))
            thumbnail = image.resize((THUMBNAIL_WIDTH, height), Image.BILINEAR)
            save_webp(np.asarray(thumbnail),
                      folder / f"{name}.webp",
                      quality=THUMBNAIL_QUALITY)
        return True

    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return sum(pool.map(write, todo))


def add_case_trajectory(viewer_root: pathlib.Path | None,
                        flat: bool,
                        method: str,
                        scene_info,
                        result: dict,
                        scene_root: pathlib.Path | None = None) -> dict:
    """Writes the aligned trajectory of one evaluated pose case.

    SLAM results (`slam_range` / `stride` set) use the SLAM alignment over the
    strided inputs, the others the SfM one. The evaluator's own fit
    (`alignment_transform`) is reused; a result without it (older
    evaluations) is aligned here. With `scene_root`, the frames' thumbnails go
    to the scene's shared folder (the viewer shows the image of the selected
    camera). A viewer failure is recorded (`viewer_error`) and never fails the
    evaluation.

    Args:
        viewer_root: Viewer root folder; None writes nothing.
        flat: Write straight into `viewer_root` (one-case calls).
        method: Method name.
        scene_info: The benchmark scene.
        result: The evaluator's scene result.
        scene_root: Root of the assets every method of a scene shares.

    Returns:
        The result with the viewer file (or the viewer error) noted.
    """
    if viewer_root is None:
        return result
    # pylint: disable=import-outside-toplevel
    from spatialarena.eval import aligned_trajectory

    try:
        if "stride" in result:
            payload = aligned_trajectory.slam_trajectory(
                pathlib.Path(result["sparse_gt_path"]),
                pathlib.Path(result["prediction_sparse_path"]),
                stride=int(result["stride"]),
                alignment=result.get("alignment", "sim3"),
                alignment_estimator=result.get("alignment_estimator",
                                               "ransac_gtsam_irls"),
                transform=result.get("alignment_transform"),
            )
        else:
            payload = aligned_trajectory.sfm_trajectory(
                pathlib.Path(result["sparse_gt_path"]),
                pathlib.Path(result["prediction_sparse_path"]),
                transform=result.get("alignment_transform"))
        payload.update({
            "task": scene_info.task,
            "dataset": scene_info.dataset,
            "scene": scene_info.scene,
            "slam_range": result.get("slam_range", ""),
            "method": method,
            "metric_ate_rmse": (result.get("metrics") or {}).get("ate_rmse")
        })
        directory = case_viewer_dir(viewer_root, flat, method, scene_info.task,
                                    scene_info.dataset, scene_info.scene)
        trajectory_path = write_trajectory(directory, payload)
        result = result | {"viewer_trajectory": str(trajectory_path)}
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return result | {"viewer_error": f"{type(exc).__name__}: {exc}"}
    image_root = getattr(scene_info, "image_path", None)
    if scene_root is not None and image_root:
        try:
            write_frame_thumbnails(
                scene_asset_dir(scene_root, scene_info.task,
                                scene_info.dataset, scene_info.scene),
                pathlib.Path(image_root), payload["names"])
        except Exception as exc:  # pylint: disable=broad-exception-caught
            result = result | {
                "viewer_error": f"thumbnails: {type(exc).__name__}: {exc}"
            }
    return result


# ---- depth -------------------------------------------------------------------


def downsample_shape(shape: tuple[int, int],
                     max_side: int = DEPTH_MAX_SIDE) -> tuple[int, int]:
    """Returns `shape` scaled down so its longer side is at most `max_side`."""
    height, width = shape[:2]
    factor = max_side / float(max(height, width))
    if factor >= 1.0:
        return height, width
    return max(1, round(height * factor)), max(1, round(width * factor))


def _resized(array: np.ndarray, shape: tuple[int, int],
             interpolation: str) -> np.ndarray:
    """Returns `array` resized to `shape` (itself if it already has it)."""
    # pylint: disable=import-outside-toplevel
    from spatialarena.eval import depth_prediction_evaluator

    if array.shape[:2] == tuple(shape):
        return array
    return depth_prediction_evaluator._resize_to_match(array, shape,
                                                       interpolation)


def colorize_depth(depth: np.ndarray, depth_range: tuple[float,
                                                         float]) -> np.ndarray:
    """Colors `depth` on the fixed `depth_range` as uint8 RGB.

    Invalid (non-finite or <= 0) pixels are black.
    """
    import matplotlib  # pylint: disable=import-outside-toplevel

    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    low, high = depth_range
    normalized = np.clip(
        (np.nan_to_num(depth, nan=low) - low) / max(high - low, 1e-6), 0.0, 1.0)
    rgb = (matplotlib.colormaps[DEPTH_COLORMAP](normalized)[..., :3] *
           255.0).astype(np.uint8)
    rgb[~valid] = DEPTH_INVALID_RGB
    return rgb


def colorize_error(error: np.ndarray) -> np.ndarray:
    """Colors an absolute-relative-error map as uint8 RGB; NaN (no GT) grey."""
    import matplotlib  # pylint: disable=import-outside-toplevel

    normalized = np.clip(
        np.nan_to_num(error, nan=0.0) / ERROR_MAX_ABSREL, 0.0, 1.0)
    rgb = (matplotlib.colormaps[ERROR_COLORMAP](normalized)[..., :3] *
           255.0).astype(np.uint8)
    rgb[~np.isfinite(error)] = ERROR_INVALID_RGB
    return rgb


def scene_depth_range(scene_info, config) -> tuple[float, float]:
    """Returns the scene's depth color scale.

    That is the 2nd-98th percentile of valid GT depth over up to 40 evenly
    spaced frames. It depends only on the GT and the dataset's resolved
    configuration, so every method's evaluation gets the same scale.

    Raises:
        FileNotFoundError: If the scene has no GT depth.
        ValueError: If the sampled GT depth has no valid value.
    """
    # pylint: disable=import-outside-toplevel
    from spatialarena.eval import depth_prediction_evaluator

    config = depth_prediction_evaluator.resolve_depth_config(scene_info, config)
    files = depth_prediction_evaluator.list_scene_gt_depth_files(scene_info)
    if not files:
        raise FileNotFoundError(
            f"No GT depth for {scene_info.dataset}/{scene_info.scene}")
    num_picks = min(DEPTH_RANGE_SAMPLE_FRAMES, len(files))
    picks = np.unique(
        np.linspace(0,
                    len(files) - 1, num_picks).round().astype(int))
    samples = []
    for index in picks:
        depth = depth_prediction_evaluator.load_depth_map(
            files[index], scale=config.gt_depth_scale)
        valid = np.isfinite(depth) & (depth > 0)
        if config.max_depth:
            valid &= depth <= config.max_depth
        values = depth[valid]
        if values.size:
            samples.append(values[::max(1, values.size // 20000)])
    if not samples:
        raise ValueError(
            f"No valid GT depth for {scene_info.dataset}/{scene_info.scene}")
    low, high = np.percentile(np.concatenate(samples), DEPTH_RANGE_PERCENTILES)
    return float(low), float(max(high, low + 1e-3))


def match_rgb_files(scene_info) -> dict[str, pathlib.Path]:
    """Maps each GT depth stem to its input image.

    Frames are matched by the evaluator's frame key, else by position when the
    counts agree.
    """
    # pylint: disable=import-outside-toplevel
    from spatialarena.eval import depth_prediction_evaluator

    if scene_info.image_path is None or not pathlib.Path(
            scene_info.image_path).is_dir():
        return {}
    gt_files = depth_prediction_evaluator.list_scene_gt_depth_files(scene_info)
    images = sorted(
        (path for path in pathlib.Path(scene_info.image_path).iterdir()
         if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}),
        key=depth_prediction_evaluator._depth_sort_key,
    )
    by_key = {
        depth_prediction_evaluator._depth_key(path): path for path in images
    }
    matched = {
        gt.stem: by_key[depth_prediction_evaluator._depth_key(gt)]
        for gt in gt_files
        if depth_prediction_evaluator._depth_key(gt) in by_key
    }
    if len(matched) < len(gt_files) and len(images) == len(gt_files):
        matched = {
            gt.stem: image for gt, image in zip(
                sorted(gt_files,
                       key=depth_prediction_evaluator._depth_sort_key), images)
        }
    return matched


@dataclasses.dataclass(kw_only=True)
class DepthViewerWriter:
    """Per-frame viewer images of one depth case.

    The writer is fed by the evaluator's `write(pair, aligned, ...)` hook.

    Attributes:
        case_dir: The case's viewer folder.
        scene_dir: The scene's shared asset folder, or None.
        depth_range: The scene's depth color scale.
        rgb_files: Input image of each GT depth stem.
        max_side: Longer image side of the written images.
        frames: Per-frame records for frames.json.
        shape: Size of the written images, once a frame was written.
    """

    case_dir: pathlib.Path
    scene_dir: pathlib.Path | None
    depth_range: tuple[float, float]
    rgb_files: Mapping[str,
                       pathlib.Path] = dataclasses.field(default_factory=dict)
    max_side: int = DEPTH_MAX_SIDE

    def __post_init__(self) -> None:
        self.frames: list[dict] = []
        self.shape: tuple[int, int] | None = None

    def write(
            self,
            pair,
            aligned: np.ndarray,
            scale: float,
            shift: float,  # pylint: disable=unused-argument
            metrics: Mapping[str, float]) -> None:
        """Saves one frame's aligned depth and error images.

        The scene's GT / RGB frame is written too if it is not there yet.
        """
        stem = pair.gt_path.stem
        shape = downsample_shape(pair.ground_truth.shape, self.max_side)
        self.shape = shape
        save_webp(
            colorize_depth(
                _resized(aligned.astype(np.float32), shape, "nearest"),
                self.depth_range), self.case_dir / "depth" / f"{stem}.webp")
        error = np.full(pair.ground_truth.shape, np.nan, dtype=np.float32)
        valid = pair.valid_mask
        error[valid] = (np.abs(aligned[valid] - pair.ground_truth[valid]) /
                        pair.ground_truth[valid])
        save_webp(colorize_error(_resized(error, shape, "nearest")),
                  self.case_dir / "error" / f"{stem}.webp")
        if self.scene_dir is not None:
            self._write_scene_frame(pair, stem, shape)
        self.frames.append({
            "frame": stem,
            "gt": pair.gt_path.name,
            "absrel": _finite_or_none(metrics.get("abs_relative_difference")),
            "delta125": _finite_or_none(metrics.get("delta125_acc")),
        })

    def _write_scene_frame(self, pair, stem: str, shape: tuple[int,
                                                               int]) -> None:
        from PIL import Image  # pylint: disable=import-outside-toplevel

        gt_target = self.scene_dir / "gt" / f"{stem}.webp"
        if not gt_target.is_file():
            gt = np.where(pair.valid_mask, pair.ground_truth,
                          np.nan).astype(np.float32)
            save_webp(
                colorize_depth(_resized(gt, shape, "nearest"),
                               self.depth_range), gt_target)
        rgb_path = self.rgb_files.get(pair.gt_path.stem)
        rgb_target = self.scene_dir / "rgb" / f"{stem}.webp"
        if rgb_path is not None and not rgb_target.is_file():
            image = Image.open(rgb_path).convert("RGB").resize(
                (shape[1], shape[0]), Image.BILINEAR)
            save_webp(np.asarray(image), rgb_target)

    def finish(self, scene_info, metrics: Mapping[str, float]) -> None:
        """Writes the case's `frames.json` and, if missing, `scene.json`."""
        if self.shape:
            size = {"height": self.shape[0], "width": self.shape[1]}
        else:
            size = None
        write_json(self.case_dir / DEPTH_FRAMES_FILE, {
            "task": scene_info.task,
            "dataset": scene_info.dataset,
            "scene": scene_info.scene,
            "frames": self.frames,
            "size": size,
            "depth_range": list(self.depth_range),
            "metrics": {
                key: _finite_or_none(value) for key, value in metrics.items()
            },
        },
                   indent=1)
        if self.scene_dir is not None and not (self.scene_dir /
                                               SCENE_FILE).is_file():
            write_json(self.scene_dir / SCENE_FILE, {
                "task": scene_info.task,
                "dataset": scene_info.dataset,
                "scene": scene_info.scene,
                "size": size,
                "depth_range": list(self.depth_range),
                "depth_colormap": DEPTH_COLORMAP,
                "error": {
                    "colormap": ERROR_COLORMAP,
                    "vmin": 0.0,
                    "vmax": ERROR_MAX_ABSREL,
                    "label": "|aligned - gt| / gt"
                },
            },
                       indent=1)


def make_depth_viewer_writer(
        scene_info, config, case_dir: pathlib.Path,
        scene_root: pathlib.Path | None) -> DepthViewerWriter:
    """Returns the viewer writer of one depth case.

    This computes the scene's GT color scale and RGB matching.
    """
    if scene_root:
        scene_dir = scene_asset_dir(scene_root, scene_info.task,
                                    scene_info.dataset, scene_info.scene)
    else:
        scene_dir = None
    depth_range = scene_depth_range(scene_info, config)
    if scene_dir is not None:
        rgb_files = match_rgb_files(scene_info)
    else:
        rgb_files = {}
    return DepthViewerWriter(
        case_dir=case_dir,
        scene_dir=scene_dir,
        depth_range=depth_range,
        rgb_files=rgb_files,
    )


class TeeDepthWriter:
    """Forwards the evaluator's per-frame `write` to several writers.

    The writers are e.g. the error artifacts and the viewer images.

    Attributes:
        writers: The writers that are not None.
    """

    def __init__(self, *writers) -> None:
        self.writers = [writer for writer in writers if writer is not None]

    def write(self, *args, **kwargs) -> None:
        """Forwards one frame to every writer."""
        for writer in self.writers:
            writer.write(*args, **kwargs)


def _finite_or_none(value) -> float | None:
    """Returns `value` as a float if it is finite, else None."""
    if value is not None and np.isfinite(value):
        return float(value)
    return None


# ---- reconstruction ----------------------------------------------------------


def _subsample(count: int, limit: int) -> np.ndarray:
    """Returns sorted indices of at most `limit` of `count` items (seeded)."""
    if count <= limit:
        return np.arange(count)
    return np.sort(
        np.random.default_rng(CLOUD_SAMPLE_SEED).choice(count,
                                                        size=limit,
                                                        replace=False))


def gt_point_colors(geometry_path: pathlib.Path | None,
                    points: np.ndarray) -> np.ndarray | None:
    """Returns the RGB colors (uint8, N x 3) of `points` from the GT geometry.

    Each point takes the color of the nearest vertex of the colored GT mesh or
    point cloud at `geometry_path` (the evaluator's GT points are samples of
    it, in its frame).

    Returns:
        The colors, or None if the geometry has no colors.
    """
    if geometry_path is None or not len(points):
        return None
    # pylint: disable=import-outside-toplevel
    import open3d as o3d
    from scipy import spatial

    mesh = o3d.io.read_triangle_mesh(str(geometry_path))
    if len(mesh.vertices) and mesh.has_vertex_colors():
        vertices = np.asarray(mesh.vertices)
        colors = np.asarray(mesh.vertex_colors)
    else:
        cloud = o3d.io.read_point_cloud(str(geometry_path))
        if not len(cloud.points) or not cloud.has_colors():
            return None
        vertices, colors = np.asarray(cloud.points), np.asarray(cloud.colors)
    _, nearest = spatial.cKDTree(vertices).query(points, k=1)
    return np.clip(colors[nearest] * 255.0 + 0.5, 0, 255).astype(np.uint8)


def write_gt_colors(scene_dir: pathlib.Path, scene_info) -> bool:
    """Writes `gt_colors.u8` for a scene's `gt_points.f32` if the GT is colored.

    Records `gt_colors` in scene.json.

    Returns:
        Whether the colors were written.
    """
    # pylint: disable-next=import-outside-toplevel
    from spatialarena.eval import mvrecon_evaluator

    points = np.fromfile(scene_dir / GT_POINTS_FILE, dtype="<f4").reshape(-1, 3)
    geometry_path = mvrecon_evaluator.resolve_gt_geometry_path(scene_info)
    colors = gt_point_colors(geometry_path, points)
    if colors is not None:
        write_array(scene_dir / GT_COLORS_FILE, colors)
    scene = json.loads((scene_dir / SCENE_FILE).read_text(encoding="utf-8"))
    scene["gt_colors"] = colors is not None
    write_json(scene_dir / SCENE_FILE, scene, indent=1)
    return colors is not None


def cameras_payload(cameras: Mapping) -> dict:
    """Returns the fused-view cameras as a JSON-ready dict.

    Per view: the image size, pinhole intrinsics and the predicted and GT
    camera-to-world poses as [x, y, z, qw, qx, qy, qz] (OpenCV axes: x right,
    y down, z forward).
    """
    # pylint: disable-next=import-outside-toplevel
    from spatialarena.eval import aligned_trajectory

    def poses(c2w):
        if c2w is None:
            return None
        c2w = np.asarray(c2w, dtype=np.float64)
        quaternions = aligned_trajectory.rotation_to_quaternion_wxyz(
            c2w[:, :3, :3])
        values = np.concatenate([c2w[:, :3, 3], quaternions], axis=1)
        return np.round(values, 5).tolist()

    intrinsics = np.asarray(cameras["intrinsics"], dtype=np.float64)
    # fx, fy, cx, cy.
    pinhole = np.round(intrinsics[:, [0, 1, 0, 1], [0, 1, 2, 2]], 3)
    return {
        "sizes": np.asarray(cameras["sizes"]).astype(int).tolist(),
        "intrinsics": pinhole.tolist(),
        "pred": poses(cameras["pred_c2w"]),
        "gt": poses(cameras.get("gt_c2w")),
    }


def point_confidence(points: np.ndarray, cameras: Mapping) -> np.ndarray | None:
    """Returns each fused point's predicted confidence.

    That is the mean, over the fused views that see the point, of the view's
    confidence map at the pixel it projects to (NaN where no view with a
    confidence map sees it).

    TSDF fusion keeps no per-point confidence, so it is read back from the
    views: `cameras` is `FusedPrediction.cameras` (or its `cameras.json` poses
    with `depth_paths`), in the points' frame. A view sees a point when it
    projects inside the image and is within `VISIBILITY_TOLERANCE` of the
    nearest point in its cell of a coarse depth buffer (the cloud's own
    surface occludes it otherwise).

    Returns:
        The per-point confidence, or None when no view has a map.
    """
    # pylint: disable-next=import-outside-toplevel
    from spatialarena.eval import prediction_artifacts

    depth_paths = cameras.get("depth_paths") or []
    files = []
    for path in depth_paths:
        if path is not None:
            files.append(prediction_artifacts.confidence_file_for(path))
        else:
            files.append(None)
    if not any(files):
        return None
    points = np.asarray(points, dtype=np.float64)
    total = np.zeros(len(points), dtype=np.float64)
    count = np.zeros(len(points), dtype=np.int32)
    c2ws = np.asarray(cameras["pred_c2w"], dtype=np.float64)
    views = zip(c2ws, np.asarray(cameras["intrinsics"]), cameras["sizes"],
                files)
    for c2w, intrinsic, (width, height), path in views:
        if path is None:
            continue
        confidence = prediction_artifacts.load_confidence_map(path)
        w2c = np.linalg.inv(c2w)
        local = points @ w2c[:3, :3].T + w2c[:3, 3]
        depth = local[:, 2]
        index = np.flatnonzero(depth > 1e-9)
        u = intrinsic[0, 0] * local[index, 0] / depth[index] + intrinsic[0, 2]
        v = intrinsic[1, 1] * local[index, 1] / depth[index] + intrinsic[1, 2]
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        index, u, v = index[inside], u[inside], v[inside]
        if not len(index):
            continue
        cell = VISIBILITY_GRID / max(width, height)
        grid_width = max(1, int(np.ceil(width * cell)))
        cells = (v * cell).astype(np.int64) * grid_width + (u * cell).astype(
            np.int64)
        nearest = np.full(cells.max() + 1, np.inf)
        np.minimum.at(nearest, cells, depth[index])
        seen = depth[index] <= nearest[cells] * (1.0 + VISIBILITY_TOLERANCE)
        index, u, v = index[seen], u[seen], v[seen]
        rows = np.clip((v * confidence.shape[0] / height).astype(np.int64), 0,
                       confidence.shape[0] - 1)
        columns = np.clip((u * confidence.shape[1] / width).astype(np.int64), 0,
                          confidence.shape[1] - 1)
        values = confidence[rows, columns].astype(np.float64)
        finite = np.isfinite(values)
        np.add.at(total, index[finite], values[finite])
        np.add.at(count, index[finite], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1),
                        np.nan).astype(np.float32)


def cameras_from_payload(payload: Mapping, depth_paths: list) -> dict:
    """Returns `point_confidence`'s cameras from a saved `cameras.json`.

    Uses its predicted poses and the views' depth files.
    """
    from scipy.spatial import transform  # pylint: disable=import-outside-toplevel

    poses = np.asarray(payload["pred"], dtype=np.float64)
    c2w = np.tile(np.eye(4), (len(poses), 1, 1))
    # Stored as w, x, y, z.
    c2w[:, :3, :3] = transform.Rotation.from_quat(
        poses[:, [4, 5, 6, 3]]).as_matrix()
    c2w[:, :3, 3] = poses[:, :3]
    intrinsics = np.tile(np.eye(3), (len(poses), 1, 1))
    values = np.asarray(payload["intrinsics"], dtype=np.float64)
    fx, fy, cx, cy = values.T
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy
    return {
        "pred_c2w": c2w,
        "intrinsics": intrinsics,
        "sizes": payload["sizes"],
        "depth_paths": list(depth_paths)
    }


def write_point_confidence(case_dir: pathlib.Path, points: np.ndarray,
                           cameras: Mapping | None) -> bool:
    """Writes a case's `confidence.f32` (or removes a stale one).

    Returns:
        Whether the file was written.
    """
    if cameras:
        confidence = point_confidence(points, cameras)
    else:
        confidence = None
    if confidence is None:
        (case_dir / CONFIDENCE_FILE).unlink(missing_ok=True)
        return False
    write_array(case_dir / CONFIDENCE_FILE, confidence.astype("<f4"))
    return True


def write_recon_viewer(case_dir: pathlib.Path, scene_dir: pathlib.Path | None,
                       scene_info, fused, *, metrics: Mapping,
                       eval_threshold: float) -> None:
    """Writes a case's fused cloud and, if missing, the scene's GT points.

    The fused cloud is in the GT frame (subsampled), with colors and
    nearest-GT error.

    Args:
        case_dir: The case's viewer folder.
        scene_dir: The scene's shared asset folder, or None.
        scene_info: The benchmark scene.
        fused: `mvrecon_evaluator.FusedPrediction` (before DTU's
            calibration-frame transform).
        metrics: The case's metrics.
        eval_threshold: The evaluation distance threshold.
    """
    from scipy import spatial  # pylint: disable=import-outside-toplevel

    keep = _subsample(len(fused.points), CLOUD_MAX_POINTS)
    points = np.asarray(fused.points)[keep]
    if len(points):
        error, _ = spatial.cKDTree(fused.gt_points).query(points, k=1)
    else:
        error, _ = (np.zeros(0), None)
    write_array(case_dir / "points.f32", points.astype("<f4"))
    colors = np.clip(np.asarray(fused.colors)[keep] * 255.0, 0, 255)
    write_array(case_dir / "colors.u8", colors.astype(np.uint8))
    write_array(case_dir / "error.f32", np.asarray(error).astype("<f4"))
    if getattr(fused, "cameras", None):
        write_json(case_dir / CAMERAS_FILE, cameras_payload(fused.cameras))
    has_confidence = write_point_confidence(case_dir, points,
                                            getattr(fused, "cameras", None))
    if len(points):
        bounds = [points.min(axis=0).tolist(), points.max(axis=0).tolist()]
        error_percentiles = {
            str(q): float(np.percentile(error, q)) for q in (50, 90)
        }
    else:
        bounds = None
        error_percentiles = {}
    write_json(case_dir / CLOUD_FILE, {
        "task": scene_info.task,
        "dataset": scene_info.dataset,
        "scene": scene_info.scene,
        "points": int(len(points)),
        "fused_points": int(len(fused.points)),
        "confidence": has_confidence,
        "bounds": bounds,
        "error_percentiles": error_percentiles,
        "eval_threshold": eval_threshold,
        "metrics": {
            key: _finite_or_none(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        },
    },
               indent=1)
    if scene_dir is not None and not (scene_dir / SCENE_FILE).is_file():
        gt_keep = _subsample(len(fused.gt_points), GT_CLOUD_MAX_POINTS)
        gt_points = np.asarray(fused.gt_points)[gt_keep]
        write_array(scene_dir / GT_POINTS_FILE, gt_points.astype("<f4"))
        write_json(scene_dir / SCENE_FILE, {
            "task": scene_info.task,
            "dataset": scene_info.dataset,
            "scene": scene_info.scene,
            "gt_points": int(len(gt_points)),
            "bounds": [
                gt_points.min(axis=0).tolist(),
                gt_points.max(axis=0).tolist()
            ],
            "error": {
                "colormap": ERROR_COLORMAP,
                "vmin": 0.0,
                "vmax": 2.0 * eval_threshold,
                "eval_threshold": eval_threshold,
                "label": "distance to the nearest GT point"
            },
        },
                   indent=1)
        write_gt_colors(scene_dir, scene_info)
