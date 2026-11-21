# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Scene-folder helpers of ``eval_scannetpp_depth``: list scenes, read their COLMAP cameras, find per-image sidecar
files (depths, masks) and load depth maps.

Moved from the DL3DV preprocessing scripts (``scripts/preprocess/dl3dv/common.py`` and
``generate_depth_masks.load_depth_map``) on 2026-11-21: the evaluator was their only user here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from spatialarena.geometry.read_write_model import qvec2rotmat, read_model

DEPTH_EXTENSIONS = (".npy", ".npz", ".png", ".tiff", ".tif", ".exr")
MASK_EXTENSIONS = (".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class SceneCamera:
    """Camera metadata in COLMAP world-to-camera convention."""

    index: int
    image_id: int
    image_name: str
    camera_id: int
    image_path: Path
    width: int
    height: int
    w2c: np.ndarray
    intrinsics: np.ndarray
    center: np.ndarray
    center_ray: np.ndarray


def read_scene_ids(path: Optional[str]) -> Optional[List[str]]:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def parse_scene_ids(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    return [scene_id.strip() for scene_id in value.split(",") if scene_id.strip()]


def list_scene_dirs(
    dataset_root: Path,
    scene_ids: Optional[Sequence[str]] = None,
    scene_ids_file: Optional[str] = None,
) -> List[Path]:
    file_scene_ids = read_scene_ids(scene_ids_file)
    merged_scene_ids: Optional[List[str]] = None
    if scene_ids is not None:
        merged_scene_ids = list(scene_ids)
    if file_scene_ids is not None:
        merged_scene_ids = file_scene_ids if merged_scene_ids is None else (
            merged_scene_ids + file_scene_ids
        )
    if merged_scene_ids is not None:
        return [dataset_root / scene_id for scene_id in sorted(set(merged_scene_ids))]

    scenes = []
    for path in sorted(dataset_root.iterdir()):
        if path.name.startswith("."):
            continue
        try:
            if path.is_dir():
                scenes.append(path)
        except PermissionError:
            print(f"Skipping {path}: permission denied")
    return scenes


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def find_existing_colmap_dir(scene_dir: Path, model_folder: str) -> Path:
    candidates = [
        scene_dir / "sparse" / model_folder,
        scene_dir / "colmap" / "sparse" / model_folder,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find COLMAP model folder '{model_folder}' under {scene_dir}"
    )


def resolve_image_path(images_dir: Path, image_name: str) -> Path:
    direct = images_dir / image_name
    if direct.exists():
        return direct
    matches = list(images_dir.rglob(Path(image_name).name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Missing image for COLMAP entry: {direct}")


def camera_params_to_intrinsics(camera) -> np.ndarray:
    camera_params = camera.params
    simple_focal_models = {
        "SIMPLE_PINHOLE",
        "SIMPLE_RADIAL",
        "RADIAL",
        "SIMPLE_RADIAL_FISHEYE",
        "RADIAL_FISHEYE",
    }
    two_focal_models = {
        "PINHOLE",
        "OPENCV",
        "OPENCV_FISHEYE",
        "FULL_OPENCV",
        "FOV",
        "THIN_PRISM_FISHEYE",
    }
    if camera.model in simple_focal_models:
        fx = fy = camera_params[0]
        cx, cy = camera_params[1:3]
    elif camera.model in two_focal_models:
        fx, fy, cx, cy = camera_params[:4]
    else:
        raise ValueError(f"Unsupported COLMAP camera model: {camera.model}")
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def scale_intrinsics_to_size(camera, width: int, height: int) -> np.ndarray:
    intrinsics = camera_params_to_intrinsics(camera)
    scale_x = width / camera.width
    scale_y = height / camera.height
    intrinsics[0, :] *= scale_x
    intrinsics[1, :] *= scale_y
    return intrinsics


def scale_intrinsics_between_sizes(
    intrinsics: np.ndarray,
    source_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> np.ndarray:
    source_width, source_height = source_size
    target_width, target_height = target_size
    scaled = np.asarray(intrinsics, dtype=np.float32).copy()
    scaled[0, :] *= target_width / source_width
    scaled[1, :] *= target_height / source_height
    return scaled


def load_scene_cameras(
    scene_dir: Path,
    image_folder: str,
    model_folder: str,
) -> List[SceneCamera]:
    model_dir = find_existing_colmap_dir(scene_dir, model_folder)
    cameras, images, _ = read_model(str(model_dir))
    images_dir = scene_dir / image_folder
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing image folder: {images_dir}")

    scene_cameras: List[SceneCamera] = []
    bottom = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32).reshape(1, 4)
    for index, (image_id, image) in enumerate(
        sorted(images.items(), key=lambda item: item[1].name)
    ):
        image_path = resolve_image_path(images_dir, image.name)
        with Image.open(image_path) as pil_image:
            width, height = pil_image.size
        camera = cameras[image.camera_id]
        intrinsics = scale_intrinsics_to_size(camera, width, height)
        rotation = qvec2rotmat(image.qvec).astype(np.float32)
        translation = image.tvec.astype(np.float32)
        w2c = np.concatenate(
            [np.concatenate([rotation, translation.reshape(3, 1)], axis=1), bottom],
            axis=0,
        ).astype(np.float32)
        center = (-rotation.T @ translation).astype(np.float32)
        center_ray = (rotation.T @ np.array([0.0, 0.0, 1.0], dtype=np.float32))
        ray_norm = np.linalg.norm(center_ray)
        if ray_norm > 0:
            center_ray = center_ray / ray_norm
        scene_cameras.append(
            SceneCamera(
                index=index,
                image_id=image_id,
                image_name=image.name,
                camera_id=image.camera_id,
                image_path=image_path,
                width=width,
                height=height,
                w2c=w2c,
                intrinsics=intrinsics,
                center=center,
                center_ray=center_ray.astype(np.float32),
            )
        )
    return scene_cameras


def find_sidecar_file(
    source_dir: Path,
    image_name: str,
    extensions: Iterable[str],
) -> Optional[Path]:
    relative_stem = Path(image_name).with_suffix("")
    flat_stem = relative_stem.name
    for ext in extensions:
        candidates = [
            source_dir / f"{relative_stem}{ext}",
            source_dir / f"{flat_stem}{ext}",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return None


def load_depth_map(
    path: Path,
    npz_key: Optional[str],
    png_depth_scale: float,
    allow_rgb_depth_png: bool,
) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        depth = np.load(path)
    elif suffix == ".npz":
        payload = np.load(path)
        key = npz_key or ("depth" if "depth" in payload else payload.files[0])
        depth = payload[key]
    else:
        depth = np.asarray(Image.open(path))
        if depth.ndim == 3:
            if not allow_rgb_depth_png:
                raise ValueError(
                    f"{path} is an RGB/RGBA image. Depth masks require raw scalar "
                    "depths; pass --allow_rgb_depth_png only for debugging."
                )
            depth = depth[..., 0]
        depth = depth.astype(np.float32) * png_depth_scale

    depth = np.asarray(depth)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth map at {path}, got shape {depth.shape}")
    return depth.astype(np.float32, copy=False)
