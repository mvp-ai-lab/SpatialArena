# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Materialize the Neural RGB-D Surface Reconstruction dataset
(Azinovic et al., CVPR 2022 -- https://github.com/dazinovic/
neural-rgbd-surface-reconstruction) into the benchmark_root/MVRecon/nrgbd
canonical layout.

Source layout (one `neural_rgbd_data.zip` + one `meshes.zip`, see
scripts/download/nrgbd.sh):

    <source_root>/<scene>/
        images/img{N}.png
        depth_filtered/depth{N}.png     # network-quality depth (used here,
                                         # matches the official codebase's
                                         # own get_intrinsics()/training use)
        focal.txt                       # single float, fx == fy
        poses.txt                       # N blocks of 4 lines: 4x4
                                         # camera-to-world matrix in OpenGL
                                         # convention (Y-up, looks down -Z),
                                         # "nan" rows mark invalid frames
    <source_root>/meshes/<scene>/gt_mesh.ply

Poses are converted from OpenGL camera-to-world to COLMAP/OpenCV
world-to-camera (see `opengl_c2w_to_colmap_w2c()`) before being written out
as a `sparse/0` COLMAP text model via the shared
`spatialarena.geometry.read_write_model` writer -- same approach as
`materialize_texturepoorsfm.py::export_sparse_model()`.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatialarena.geometry.read_write_model import BaseImage, Camera, rotmat2qvec, write_model  # noqa: E402

SCENES = (
    "breakfast_room",
    "complete_kitchen",
    "green_room",
    "grey_white_room",
    "kitchen",
    "morning_apartment",
    "staircase",
    "thin_geometry",
    "whiteroom",
)

# OpenGL camera basis (right, up, -forward) -> OpenCV/COLMAP camera basis
# (right, down, forward): flip the Y and Z columns, keep the same origin.
_GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def _alphanum_key(name: str) -> list:
    return [int(chunk) if chunk.isdigit() else chunk for chunk in re.split(r"(\d+)", name)]


def load_focal(path: Path) -> float:
    return float(path.read_text().strip())


def load_poses(path: Path) -> list[tuple[np.ndarray | None, bool]]:
    lines = path.read_text().splitlines()
    poses = []
    for start in range(0, len(lines), 4):
        block = lines[start : start + 4]
        if len(block) < 4:
            break
        if any("nan" in line.lower() for line in block):
            poses.append((None, False))
            continue
        matrix = np.array([[float(x) for x in line.split()] for line in block], dtype=np.float64)
        poses.append((matrix, True))
    return poses


def opengl_c2w_to_colmap_w2c(c2w_gl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert one OpenGL-convention camera-to-world matrix into a COLMAP
    world-to-camera (R, t) pair."""
    c2w_cv = c2w_gl @ _GL_TO_CV
    w2c_cv = np.linalg.inv(c2w_cv)
    return w2c_cv[:3, :3], w2c_cv[:3, 3]


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def export_sparse_model(
    scene_root: Path,
    output_sparse: Path,
    focal: float,
    poses: list[tuple[np.ndarray | None, bool]],
    image_names: list[str],
    overwrite: bool,
) -> int:
    if output_sparse.exists() and not overwrite:
        return sum(1 for _, valid in poses if valid)

    cameras: dict[int, Camera] = {}
    images: dict[int, BaseImage] = {}
    output_sparse.mkdir(parents=True, exist_ok=True)

    for frame_index, image_name in enumerate(image_names):
        if frame_index >= len(poses):
            break
        c2w_gl, valid = poses[frame_index]
        if not valid or c2w_gl is None:
            continue
        image_path = scene_root / "images" / image_name
        width, height = image_size(image_path)
        R, t = opengl_c2w_to_colmap_w2c(c2w_gl)
        image_id = frame_index + 1
        cameras[image_id] = Camera(
            id=image_id,
            model="PINHOLE",
            width=width,
            height=height,
            params=np.array([focal, focal, width / 2.0, height / 2.0], dtype=np.float64),
        )
        images[image_id] = BaseImage(
            id=image_id,
            qvec=rotmat2qvec(R).astype(np.float64),
            tvec=t.astype(np.float64),
            camera_id=image_id,
            name=image_name,
            xys=np.empty((0, 2), dtype=np.float64),
            point3D_ids=np.empty((0,), dtype=np.int64),
        )

    write_model(cameras, images, {}, str(output_sparse), ext=".txt")
    return len(images)


def link_path(src: Path, dst: Path, overwrite: bool, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())


def materialize_scene(
    source_root: Path,
    meshes_root: Path,
    scene: str,
    output_root: Path,
    overwrite: bool,
    copy: bool,
) -> dict | None:
    scene_root = source_root / scene
    if not scene_root.is_dir():
        return None

    image_names = sorted(
        (p.name for p in (scene_root / "images").glob("*.png")),
        key=_alphanum_key,
    )
    if not image_names:
        return None

    out_scene = output_root / scene
    link_path(scene_root / "images", out_scene / "images", overwrite=overwrite, copy=copy)
    link_path(scene_root / "depth_filtered", out_scene / "depths", overwrite=overwrite, copy=copy)

    gt_mesh = meshes_root / scene / "gt_mesh.ply"
    if gt_mesh.is_file():
        link_path(gt_mesh, out_scene / "gt_mesh.ply", overwrite=overwrite, copy=copy)

    focal = load_focal(scene_root / "focal.txt")
    poses = load_poses(scene_root / "poses.txt")
    num_sparse = export_sparse_model(
        scene_root, out_scene / "sparse" / "0", focal, poses, image_names, overwrite=overwrite
    )

    return {
        "dataset_name": "nrgbd",
        "scene_name": scene,
        "root_directory": str(out_scene),
        "total_images": len(image_names),
        "num_sparse_images": num_sparse,
        "gt_mesh_path": "gt_mesh.ply" if gt_mesh.is_file() else None,
    }


def write_metadata(output_root: Path, scenes: list[dict]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    total_images = sum(int(scene.get("total_images", 0)) for scene in scenes)
    payload = {
        "dataset_name": "nrgbd",
        "root_directory": str(output_root),
        "total_images": total_images,
        "average_images_per_scene": total_images / len(scenes) if scenes else 0.0,
        "scenes": scenes,
    }
    (output_root / "metadata.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def materialize_dataset(source_root: Path, benchmark_root: Path, overwrite: bool, copy: bool = True) -> dict:
    meshes_root = source_root / "meshes"
    output_root = benchmark_root / "MVRecon" / "nrgbd"
    records = []
    for scene in SCENES:
        record = materialize_scene(source_root, meshes_root, scene, output_root, overwrite, copy)
        if record is not None:
            records.append(record)
    write_metadata(output_root, records)
    return {"dataset": "nrgbd", "num_scenes": len(records), "output_root": str(output_root)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source_root", type=Path, required=True, help="Extracted neural_rgbd_data.zip root.")
    parser.add_argument("--benchmark_root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--symlink", action="store_true", help="Symlink instead of copying (not self-contained).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = materialize_dataset(
        args.source_root, args.benchmark_root, overwrite=args.overwrite, copy=not args.symlink
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
