#!/usr/bin/env python
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Materialize benchmark camera priors as a minimal COLMAP model.

The benchmark adapters consume camera priors through COLMAP's cameras/images/
points3D triplet. Several depth benchmark layouts keep calibrated poses in their
native metadata instead of an already-materialized COLMAP folder. This script
exports those native pose/intrinsic records into a minimal text COLMAP model.

Supported native layouts:
- RobustMVD sampled RMVD cases with ``sample.json`` containing ``poses`` and
  ``intrinsics``.
- Bonn RGB-D scenes with TUM-style ``rgb.txt`` and ``groundtruth.txt``.
- Sintel scenes with ``training/camdata_left/<scene>/*.cam``.
- DyCheck scenes (``VideoDepth/dycheck``) whose images link to the MegaSaM archive's camera-0 frames, with
  Nerfies-style ``dense/dycheck/camera/<frame>.json``.
- KITTI video-depth gathered scenes with OXTS poses and per-frame intrinsics
  from the depth-selection layout. KITTI calibration-to-camera extrinsics are
  not present in the current benchmark copy, so the OXTS vehicle trajectory is
  used as a metric pose prior with image-space intrinsics.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from bisect import bisect_left
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from spatialarena.geometry.read_write_model import (  # noqa: E402
    Camera,
    Image as ColmapImage,
    rotmat2qvec,
    write_model,
)
from scripts.inference.adapter_utils import find_image_paths  # noqa: E402


TUM_BONN_INTRINSICS = np.array(
    [542.822841, 0.0, 315.593520, 0.0, 542.576870, 237.756098, 0.0, 0.0, 1.0],
    dtype=np.float64,
).reshape(3, 3)

SINTEL_CAM_TAG_FLOAT = 202021.25
EARTH_RADIUS_METERS = 6378137.0


def _read_tum_trajectory(path: Path) -> dict[float, np.ndarray]:
    poses: dict[float, np.ndarray] = {}
    if not path.is_file():
        return poses
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values = [float(part) for part in line.split()]
        if len(values) < 8:
            continue
        timestamp, tx, ty, tz, qx, qy, qz, qw = values[:8]
        poses[timestamp] = _quat_xyzw_to_c2w(qx, qy, qz, qw, tx, ty, tz)
    return poses


def _read_tum_rgb_index(path: Path) -> dict[str, float]:
    mapping: dict[str, float] = {}
    if not path.is_file():
        return mapping
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        mapping[Path(parts[1]).name] = float(parts[0])
        mapping[Path(parts[1]).stem] = float(parts[0])
    return mapping


def _nearest_pose(
    timestamp: float, poses: dict[float, np.ndarray], max_delta_seconds: float = 0.12
) -> np.ndarray | None:
    if not poses:
        return None
    keys = sorted(poses)
    index = bisect_left(keys, timestamp)
    candidates = []
    if index < len(keys):
        candidates.append(keys[index])
    if index > 0:
        candidates.append(keys[index - 1])
    if not candidates:
        return None
    nearest = min(candidates, key=lambda value: abs(value - timestamp))
    if abs(nearest - timestamp) > max_delta_seconds:
        return None
    return poses[nearest]


def _quat_xyzw_to_c2w(qx: float, qy: float, qz: float, qw: float, tx: float, ty: float, tz: float) -> np.ndarray:
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    qw, qx, qy, qz = q.tolist()
    rotation = np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qx * qz + 2 * qw * qy],
            [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx],
            [2 * qx * qz - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = [tx, ty, tz]
    return pose


def _read_sintel_cam(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        tag = np.fromfile(handle, dtype=np.float32, count=1)[0]
        if abs(float(tag) - SINTEL_CAM_TAG_FLOAT) > 1e-4:
            raise ValueError(f"Invalid Sintel camera tag in {path}: {tag}")
        intrinsic = np.fromfile(handle, dtype=np.float64, count=9).reshape(3, 3)
        extrinsic = np.fromfile(handle, dtype=np.float64, count=12).reshape(3, 4)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :4] = extrinsic
    return np.linalg.inv(w2c), intrinsic


def _latlon_to_mercator(lat: float, lon: float, scale: float) -> tuple[float, float]:
    mx = scale * lon * math.pi * EARTH_RADIUS_METERS / 180.0
    my = scale * EARTH_RADIUS_METERS * math.log(math.tan((90.0 + lat) * math.pi / 360.0))
    return mx, my


def _rotz(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _roty(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rotx(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _read_kitti_oxts_scene(
    scene_root: Path, image_paths: list[Path]
) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
    dataset_root = scene_root.parent
    parts = scene_root.name.split("_")
    if len(parts) < 7 or parts[-1] not in {"02", "03"}:
        return None
    date = "_".join(parts[:3])
    drive = "_".join(parts[:6])
    cam = parts[-1]
    oxts_dir = dataset_root / date / drive / "oxts" / "data"
    intrinsics_dir = dataset_root / "depth_selection" / "val_selection_cropped" / "intrinsics"
    if not oxts_dir.is_dir() or not intrinsics_dir.is_dir():
        return None

    oxts_by_index: dict[str, np.ndarray] = {}
    origin = None
    scale = None
    for oxts_path in sorted(oxts_dir.glob("*.txt")):
        values = [float(part) for part in oxts_path.read_text(encoding="utf-8").split()]
        if len(values) < 6:
            continue
        lat, lon, alt, roll, pitch, yaw = values[:6]
        if origin is None:
            scale = math.cos(lat * math.pi / 180.0)
            mx, my = _latlon_to_mercator(lat, lon, scale)
            origin = np.array([mx, my, alt], dtype=np.float64)
        assert scale is not None and origin is not None
        mx, my = _latlon_to_mercator(lat, lon, scale)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = _rotz(yaw) @ _roty(pitch) @ _rotx(roll)
        pose[:3, 3] = np.array([mx, my, alt], dtype=np.float64) - origin
        oxts_by_index[oxts_path.stem] = pose

    intrinsic_fallback = None
    fallback_candidates = sorted(intrinsics_dir.glob(f"{drive}_image_*_image_{cam}.txt"))
    if fallback_candidates:
        intrinsic_values = [
            float(part)
            for part in fallback_candidates[0].read_text(encoding="utf-8").split()
        ]
        intrinsic_fallback = np.asarray(intrinsic_values, dtype=np.float64).reshape(3, 3)

    poses: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    for image_path in image_paths:
        frame = image_path.stem
        pose = oxts_by_index.get(frame)
        intrinsic_path = intrinsics_dir / f"{drive}_image_{frame}_image_{cam}.txt"
        if pose is None:
            return None
        if intrinsic_path.is_file():
            intrinsic_values = [
                float(part)
                for part in intrinsic_path.read_text(encoding="utf-8").split()
            ]
            intrinsic = np.asarray(intrinsic_values, dtype=np.float64).reshape(3, 3)
        elif intrinsic_fallback is not None:
            intrinsic = intrinsic_fallback
        else:
            return None
        poses.append(pose)
        intrinsics.append(intrinsic)
    return poses, intrinsics


def _image_name_for_colmap(image_path: Path, images_dir: Path) -> str:
    try:
        return image_path.relative_to(images_dir).as_posix()
    except ValueError:
        return image_path.name


def _intrinsics_to_params(intrinsic: np.ndarray, image_path: Path) -> tuple[int, int, np.ndarray]:
    with Image.open(image_path) as image:
        width, height = image.size
    intrinsic = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    params = np.array([intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]], dtype=np.float64)
    return width, height, params


def _write_colmap_model(
    *,
    output_dir: Path,
    images_dir: Path,
    image_paths: list[Path],
    c2w_poses: Iterable[np.ndarray],
    intrinsics: Iterable[np.ndarray],
) -> int:
    cameras = {}
    images = {}
    for index, (image_path, c2w, intrinsic) in enumerate(zip(image_paths, c2w_poses, intrinsics, strict=True), start=1):
        width, height, params = _intrinsics_to_params(intrinsic, image_path)
        cameras[index] = Camera(id=index, model="PINHOLE", width=width, height=height, params=params)
        c2w = np.asarray(c2w, dtype=np.float64).reshape(4, 4)
        w2c = np.linalg.inv(c2w)
        images[index] = ColmapImage(
            id=index,
            qvec=rotmat2qvec(w2c[:3, :3]),
            tvec=w2c[:3, 3],
            camera_id=index,
            name=_image_name_for_colmap(image_path, images_dir),
            xys=np.empty((0, 2), dtype=np.float64),
            point3D_ids=np.empty((0,), dtype=np.int64),
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_model(cameras, images, {}, str(output_dir), ext=".txt")
    return len(images)


def materialize_sample_json(scene_root: Path, output_dir: Path) -> int | None:
    sample_path = scene_root / "sample.json"
    images_dir = scene_root / "images"
    if not sample_path.is_file() or not images_dir.is_dir():
        return None
    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    if "poses" not in payload or "intrinsics" not in payload or "images" not in payload:
        return None
    image_paths = [scene_root / record["canonical_path"] for record in payload["images"]]
    if any(not path.is_file() for path in image_paths):
        return None
    return _write_colmap_model(
        output_dir=output_dir,
        images_dir=images_dir,
        image_paths=image_paths,
        c2w_poses=[np.asarray(pose, dtype=np.float64) for pose in payload["poses"]],
        intrinsics=[np.asarray(intrinsic, dtype=np.float64) for intrinsic in payload["intrinsics"]],
    )


def materialize_bonn(scene_root: Path, output_dir: Path) -> int | None:
    images_dir = scene_root / "images"
    if not images_dir.is_dir():
        return None
    raw_scene = (images_dir.resolve()).parent
    rgb_index = _read_tum_rgb_index(raw_scene / "rgb.txt")
    poses_by_time = _read_tum_trajectory(raw_scene / "groundtruth.txt")
    if not rgb_index or not poses_by_time:
        return None
    image_paths = [Path(path) for path in find_image_paths(images_dir)]
    poses: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    for image_path in image_paths:
        timestamp = rgb_index.get(image_path.name) or rgb_index.get(image_path.stem)
        if timestamp is None:
            try:
                timestamp = float(image_path.stem)
            except ValueError:
                return None
        pose = _nearest_pose(timestamp, poses_by_time)
        if pose is None:
            return None
        poses.append(pose)
        intrinsics.append(TUM_BONN_INTRINSICS)
    return _write_colmap_model(
        output_dir=output_dir,
        images_dir=images_dir,
        image_paths=image_paths,
        c2w_poses=poses,
        intrinsics=intrinsics,
    )


def materialize_sintel(scene_root: Path, output_dir: Path) -> int | None:
    images_dir = scene_root / "images"
    if not images_dir.is_dir():
        return None
    dataset_root = scene_root.parent
    cam_dir = dataset_root / "training" / "camdata_left" / scene_root.name
    if not cam_dir.is_dir():
        return None
    image_paths = [Path(path) for path in find_image_paths(images_dir)]
    poses: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    for image_path in image_paths:
        cam_path = cam_dir / f"{image_path.stem}.cam"
        if not cam_path.is_file():
            return None
        pose, intrinsic = _read_sintel_cam(cam_path)
        poses.append(pose)
        intrinsics.append(intrinsic)
    return _write_colmap_model(
        output_dir=output_dir,
        images_dir=images_dir,
        image_paths=image_paths,
        c2w_poses=poses,
        intrinsics=intrinsics,
    )


def materialize_kitti(scene_root: Path, output_dir: Path) -> int | None:
    images_dir = scene_root / "images"
    if not images_dir.is_dir():
        return None
    image_paths = [Path(path) for path in find_image_paths(images_dir)]
    loaded = _read_kitti_oxts_scene(scene_root, image_paths)
    if loaded is None:
        return None
    poses, intrinsics = loaded
    return _write_colmap_model(
        output_dir=output_dir,
        images_dir=images_dir,
        image_paths=image_paths,
        c2w_poses=poses,
        intrinsics=intrinsics,
    )


def _read_dycheck_camera(path: Path, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """DyCheck (Nerfies) camera JSON -> (c2w, K) at ``image_size``.

    ``orientation`` is the world-to-camera rotation and ``position`` the camera centre, in OpenCV axes; the
    intrinsics are given at the JSON's ``image_size`` and rescaled to the image actually linked.
    """
    camera = json.loads(path.read_text(encoding="utf-8"))
    distortion = [*camera.get("radial_distortion", []), *camera.get("tangential_distortion", [])]
    if camera.get("skew", 0) or any(distortion):
        raise ValueError(f"DyCheck camera has skew or distortion a PINHOLE prior cannot express: {path}")
    c2w = np.eye(4)
    c2w[:3, :3] = np.asarray(camera["orientation"], dtype=np.float64).T
    c2w[:3, 3] = np.asarray(camera["position"], dtype=np.float64)
    sx, sy = image_size[0] / camera["image_size"][0], image_size[1] / camera["image_size"][1]
    focal = float(camera["focal_length"])
    intrinsic = np.array([
        [focal * sx, 0.0, camera["principal_point"][0] * sx],
        [0.0, focal * camera.get("pixel_aspect_ratio", 1.0) * sy, camera["principal_point"][1] * sy],
        [0.0, 0.0, 1.0],
    ])
    return c2w, intrinsic


def materialize_dycheck(scene_root: Path, output_dir: Path) -> int | None:
    """VideoDepth/dycheck: images link to <scene>/dense/rgb/1x/<frame>.png; cameras are dense/dycheck/camera."""
    images_dir = scene_root / "images"
    if not images_dir.is_dir() or not (scene_root / "frame_manifest.json").is_file():
        return None
    image_paths = [Path(path) for path in find_image_paths(images_dir)]
    poses: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    for image_path in image_paths:
        source = image_path.resolve()
        cam_path = source.parents[2] / "dycheck" / "camera" / f"{source.stem}.json"
        if not cam_path.is_file():
            return None
        with Image.open(image_path) as image:
            pose, intrinsic = _read_dycheck_camera(cam_path, image.size)
        poses.append(pose)
        intrinsics.append(intrinsic)
    return _write_colmap_model(
        output_dir=output_dir,
        images_dir=images_dir,
        image_paths=image_paths,
        c2w_poses=poses,
        intrinsics=intrinsics,
    )


def materialize(scene_root: Path, output_dir: Path, overwrite: bool = False) -> int:
    existing = [output_dir / name for name in ("cameras.txt", "images.txt", "points3D.txt")]
    if all(path.is_file() for path in existing) and not overwrite:
        return -1
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    for converter in (materialize_sample_json, materialize_bonn, materialize_sintel, materialize_kitti,
                      materialize_dycheck):
        result = converter(scene_root, output_dir)
        if result is not None:
            return result
    raise ValueError(f"No supported camera-prior source found for {scene_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", type=Path, required=True, help="Benchmark scene root containing images/.")
    parser.add_argument("--output_path", type=Path, required=True, help="Output COLMAP model directory.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = materialize(args.data_path, args.output_path, overwrite=args.overwrite)
    if count == -1:
        print(f"Camera priors already exist: {args.output_path}")
    else:
        print(f"Wrote {count} camera priors to {args.output_path}")


if __name__ == "__main__":
    main()
