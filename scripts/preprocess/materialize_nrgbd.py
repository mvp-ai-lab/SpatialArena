# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Materialize Neural RGB-D into benchmark_root/MVRecon/nrgbd.

The Neural RGB-D Surface Reconstruction dataset (Azinovic et al., CVPR 2022 --
https://github.com/dazinovic/neural-rgbd-surface-reconstruction) is written in
the canonical layout.

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
import pathlib
import shutil
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from spatialarena.geometry import read_write_model  # noqa: E402
# pylint: enable=wrong-import-position

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
    """Sort key that orders embedded numbers numerically (img2 before img10)."""
    return [
        int(chunk) if chunk.isdigit() else chunk
        for chunk in re.split(r"(\d+)", name)
    ]


def load_focal(path: pathlib.Path) -> float:
    """Reads the single focal length (fx == fy) in `focal.txt`."""
    return float(path.read_text().strip())


def load_poses(path: pathlib.Path) -> list[tuple[np.ndarray | None, bool]]:
    """Reads `poses.txt` as 4-line blocks of 4x4 camera-to-world matrices.

    Args:
        path: Pose file.

    Returns:
        One `(matrix, True)` per block, or `(None, False)` for a block that
        contains "nan" (an invalid frame). A trailing partial block is ignored.
    """
    lines = path.read_text().splitlines()
    poses = []
    for start in range(0, len(lines), 4):
        block = lines[start:start + 4]
        if len(block) < 4:
            break
        if any("nan" in line.lower() for line in block):
            poses.append((None, False))
            continue
        matrix = np.array(
            [[float(value) for value in line.split()] for line in block],
            dtype=np.float64,
        )
        poses.append((matrix, True))
    return poses


def opengl_c2w_to_colmap_w2c(
        c2w_gl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Converts an OpenGL camera-to-world matrix to COLMAP world-to-camera.

    Args:
        c2w_gl: 4x4 camera-to-world matrix in the OpenGL convention.

    Returns:
        The world-to-camera rotation (3x3) and translation (3,) in the
        OpenCV/COLMAP convention.
    """
    c2w_cv = c2w_gl @ _GL_TO_CV
    w2c_cv = np.linalg.inv(c2w_cv)
    return w2c_cv[:3, :3], w2c_cv[:3, 3]


def image_size(path: pathlib.Path) -> tuple[int, int]:
    """Returns the (width, height) of an image file."""
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def export_sparse_model(
    scene_root: pathlib.Path,
    output_sparse: pathlib.Path,
    focal: float,
    poses: list[tuple[np.ndarray | None, bool]],
    image_names: list[str],
    overwrite: bool,
) -> int:
    """Writes the valid poses as a COLMAP text model, one camera per image.

    Args:
        scene_root: Source scene folder (for the image sizes).
        output_sparse: Output model folder.
        focal: Focal length; the principal point is the image center.
        poses: Output of `load_poses`.
        image_names: Image file names in frame order.
        overwrite: Rewrite an existing model; otherwise it is kept.

    Returns:
        The number of posed images (counted from `poses` when the existing
        model is kept).
    """
    if output_sparse.exists() and not overwrite:
        return sum(1 for _, valid in poses if valid)

    cameras: dict[int, read_write_model.Camera] = {}
    images: dict[int, read_write_model.BaseImage] = {}
    output_sparse.mkdir(parents=True, exist_ok=True)

    for frame_index, image_name in enumerate(image_names):
        if frame_index >= len(poses):
            break
        c2w_gl, valid = poses[frame_index]
        if not valid or c2w_gl is None:
            continue
        image_path = scene_root / "images" / image_name
        width, height = image_size(image_path)
        rotation, translation = opengl_c2w_to_colmap_w2c(c2w_gl)
        image_id = frame_index + 1
        cameras[image_id] = read_write_model.Camera(
            id=image_id,
            model="PINHOLE",
            width=width,
            height=height,
            params=np.array([focal, focal, width / 2.0, height / 2.0],
                            dtype=np.float64),
        )
        images[image_id] = read_write_model.BaseImage(
            id=image_id,
            qvec=read_write_model.rotmat2qvec(rotation).astype(np.float64),
            tvec=translation.astype(np.float64),
            camera_id=image_id,
            name=image_name,
            xys=np.empty((0, 2), dtype=np.float64),
            point3D_ids=np.empty((0,), dtype=np.int64),
        )

    points3d = {}
    read_write_model.write_model(
        cameras,
        images,
        points3d,
        str(output_sparse),
        ext=".txt",
    )
    return len(images)


def link_path(src: pathlib.Path, dst: pathlib.Path, overwrite: bool,
              copy: bool) -> None:
    """Copies or symlinks `src` to `dst`.

    Args:
        src: Source file or folder.
        dst: Destination path.
        overwrite: Replace an existing `dst`; otherwise it is left alone.
        copy: Copy; otherwise create an absolute symlink.
    """
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
    source_root: pathlib.Path,
    meshes_root: pathlib.Path,
    scene: str,
    output_root: pathlib.Path,
    overwrite: bool,
    copy: bool,
) -> dict | None:
    """Materializes one scene: images, filtered depths, GT mesh and poses.

    Args:
        source_root: Extracted `neural_rgbd_data.zip` root.
        meshes_root: Extracted `meshes.zip` root.
        scene: Scene name.
        output_root: Dataset output folder.
        overwrite: Replace existing outputs.
        copy: Copy files; otherwise symlink them.

    Returns:
        The scene's metadata record, or None when the scene or its images are
        missing.
    """
    scene_root = source_root / scene
    if not scene_root.is_dir():
        return None

    image_names = sorted(
        (path.name for path in (scene_root / "images").glob("*.png")),
        key=_alphanum_key,
    )
    if not image_names:
        return None

    out_scene = output_root / scene
    link_path(scene_root / "images",
              out_scene / "images",
              overwrite=overwrite,
              copy=copy)
    link_path(scene_root / "depth_filtered",
              out_scene / "depths",
              overwrite=overwrite,
              copy=copy)

    gt_mesh = meshes_root / scene / "gt_mesh.ply"
    if gt_mesh.is_file():
        link_path(gt_mesh,
                  out_scene / "gt_mesh.ply",
                  overwrite=overwrite,
                  copy=copy)

    focal = load_focal(scene_root / "focal.txt")
    poses = load_poses(scene_root / "poses.txt")
    num_sparse = export_sparse_model(scene_root,
                                     out_scene / "sparse" / "0",
                                     focal,
                                     poses,
                                     image_names,
                                     overwrite=overwrite)

    return {
        "dataset_name": "nrgbd",
        "scene_name": scene,
        "root_directory": str(out_scene),
        "total_images": len(image_names),
        "num_sparse_images": num_sparse,
        "gt_mesh_path": "gt_mesh.ply" if gt_mesh.is_file() else None,
    }


def write_metadata(output_root: pathlib.Path, scenes: list[dict]) -> None:
    """Writes the dataset's `metadata.json` from the scene records."""
    output_root.mkdir(parents=True, exist_ok=True)
    total_images = sum(int(scene.get("total_images", 0)) for scene in scenes)
    average_images = 0.0
    if scenes:
        average_images = total_images / len(scenes)
    payload = {
        "dataset_name": "nrgbd",
        "root_directory": str(output_root),
        "total_images": total_images,
        "average_images_per_scene": average_images,
        "scenes": scenes,
    }
    text = json.dumps(payload, indent=2) + "\n"
    (output_root / "metadata.json").write_text(text, encoding="utf-8")


def materialize_dataset(source_root: pathlib.Path,
                        benchmark_root: pathlib.Path,
                        overwrite: bool,
                        copy: bool = True) -> dict:
    """Materializes every scene in `SCENES` and writes the metadata.

    Args:
        source_root: Extracted `neural_rgbd_data.zip` root, with `meshes/`.
        benchmark_root: Benchmark root.
        overwrite: Replace existing outputs.
        copy: Copy files; otherwise symlink them.

    Returns:
        A summary with the dataset name, scene count and output folder.
    """
    meshes_root = source_root / "meshes"
    output_root = benchmark_root / "MVRecon" / "nrgbd"
    records = []
    for scene in SCENES:
        record = materialize_scene(
            source_root,
            meshes_root,
            scene,
            output_root,
            overwrite,
            copy,
        )
        if record is not None:
            records.append(record)
    write_metadata(output_root, records)
    return {
        "dataset": "nrgbd",
        "num_scenes": len(records),
        "output_root": str(output_root)
    }


def parse_args() -> argparse.Namespace:
    """Parses the command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source_root",
                        type=pathlib.Path,
                        required=True,
                        help="Extracted neural_rgbd_data.zip root.")
    parser.add_argument("--benchmark_root", type=pathlib.Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--symlink",
                        action="store_true",
                        help="Symlink instead of copying (not self-contained).")
    return parser.parse_args()


def main() -> None:
    """Runs the script with the command-line arguments."""
    args = parse_args()
    result = materialize_dataset(args.source_root,
                                 args.benchmark_root,
                                 overwrite=args.overwrite,
                                 copy=not args.symlink)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
