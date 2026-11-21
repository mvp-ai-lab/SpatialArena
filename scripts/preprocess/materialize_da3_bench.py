# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Materialize DA3-BENCH subsets into the canonical benchmark layout.

This script does not add a DA3-specific dataloader. It copies data into the
existing scene contract:

  MVRecon/<dataset>/<scene>/{images,depths,masks,sparse/0}

Every file is a real copy, never a symlink -- the materialized layout is
meant to be a self-contained, redistributable dataset (e.g. for a Hugging
Face release), so it must not depend on the DA3-BENCH source directory (or
any other machine-local path) staying in place.

The conversion logic follows the public Depth-Anything-3 DA3-BENCH dataset
wrappers for ScanNet++ and HiRoom. Source references are kept in comments where
the path conventions differ from the benchmark canonical layout.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatialarena.geometry.read_write_model import (  # noqa: E402
    BaseImage,
    Camera,
    rotmat2qvec,
    write_model,
)


SCANNETPP_SCENES = (
    "09c1414f1b",
    "1ada7a0617",
    "40aec5fffa",
    "3e8bba0176",
    "acd95847c5",
    "578511c8a9",
    "5f99900f09",
    "c4c04e6d6c",
    "f3d64c30f8",
    "7bc286c1b6",
    "c5439f4607",
    "286b55a2bf",
    "fb5a96b1a2",
    "7831862f02",
    "38d58a7a31",
    "bde1e479ad",
    "9071e139d9",
    "21d970d8de",
    "bcd2436daf",
    "cc5237fd77",
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def copy_path(src: Path, dst: Path, overwrite: bool = False) -> str:
    """Copy a file or directory tree, resolving through any source symlinks.

    Always produces a real file/directory at ``dst`` -- never a symlink --
    so the materialized dataset stays self-contained and redistributable.
    """
    src = src.resolve(strict=False)
    if not src.exists():
        return "missing_source"
    if dst.exists() or dst.is_symlink():
        if overwrite:
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        else:
            return "exists"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, symlinks=False)
    else:
        shutil.copy2(src, dst, follow_symlinks=True)
    return "copied"


def read_image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def colmap_image_from_w2c(image_id: int, image_name: str, w2c: np.ndarray) -> BaseImage:
    return BaseImage(
        id=image_id,
        qvec=rotmat2qvec(w2c[:3, :3]).astype(np.float64),
        tvec=w2c[:3, 3].astype(np.float64),
        camera_id=1,
        name=image_name,
        xys=np.empty((0, 2), dtype=np.float64),
        point3D_ids=np.empty((0,), dtype=np.int64),
    )


def export_hiroom_sparse(scene_root: Path, output_sparse: Path, image_names: list[str], overwrite: bool) -> int:
    if output_sparse.exists() and not overwrite:
        return 0
    intrinsics = np.load(scene_root / "cam_K.npy").astype(np.float64)
    first_image = scene_root / "image" / image_names[0]
    width, height = read_image_size(first_image)
    cameras = {
        1: Camera(
            id=1,
            model="PINHOLE",
            width=width,
            height=height,
            params=np.array([intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]]),
        )
    }
    images = {}
    for image_name in image_names:
        pose_path = scene_root / "pose" / f"{Path(image_name).stem}.npy"
        if not pose_path.is_file():
            continue
        w2c = np.load(pose_path).astype(np.float64)
        images[len(images) + 1] = colmap_image_from_w2c(len(images) + 1, image_name, w2c)
    output_sparse.mkdir(parents=True, exist_ok=True)
    write_model(cameras, images, {}, str(output_sparse), ext=".txt")
    return len(images)


def materialize_hiroom(source_root: Path, benchmark_root: Path, overwrite: bool) -> dict:
    # Matches Depth-Anything-3 bench/datasets/hiroom.py:
    #   data/<scene>/image, depth, pose, cam_K.npy, aliasing_mask
    #   fused_pcd/<last-three-path-components>.ply
    data_root = source_root / "hiroom" / "data"
    if not data_root.is_dir():
        data_root = source_root / "data"
    gt_root = source_root / "hiroom" / "fused_pcd"
    if not gt_root.is_dir():
        gt_root = source_root / "fused_pcd"
    scene_list = source_root / "hiroom" / "selected_scene_list_val.txt"
    if not scene_list.is_file():
        scene_list = source_root / "selected_scene_list_val.txt"

    scenes = (
        [line.strip() for line in scene_list.read_text(encoding="utf-8").splitlines()]
        if scene_list.is_file()
        else []
    )
    if not scenes:
        scenes = [
            path.relative_to(data_root).as_posix()
            for path in data_root.rglob("*")
            if path.is_dir() and (path / "image").is_dir() and (path / "cam_K.npy").is_file()
        ]

    output_root = benchmark_root / "MVRecon" / "hiroom"
    records = []
    for scene in sorted(scenes):
        src_scene = data_root / scene
        if not (src_scene / "image").is_dir():
            continue
        scene_name = scene.replace("/", "__")
        out_scene = output_root / scene_name
        image_names = sorted(
            path.name
            for path in (src_scene / "image").iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_names:
            continue
        copy_path(src_scene / "image", out_scene / "images", overwrite=overwrite)
        copy_path(src_scene / "depth", out_scene / "depths", overwrite=overwrite)
        copy_path(src_scene / "aliasing_mask", out_scene / "masks", overwrite=overwrite)
        gt_name = "-".join(scene.split("/")[-3:]) + ".ply"
        gt_path = gt_root / gt_name
        if gt_path.is_file():
            copy_path(gt_path, out_scene / "gt_pointcloud.ply", overwrite=overwrite)
        num_sparse = export_hiroom_sparse(src_scene, out_scene / "sparse" / "0", image_names, overwrite=overwrite)
        records.append(
            {
                "dataset_name": "hiroom",
                "scene_name": scene_name,
                "source_scene": scene,
                "root_directory": str(out_scene),
                "total_images": len(image_names),
                "gt_point_path": "gt_pointcloud.ply",
                "num_sparse_images": num_sparse,
            }
        )
    write_metadata(output_root, "hiroom", records)
    return {"dataset": "hiroom", "num_scenes": len(records), "output_root": str(output_root)}


def materialize_scannetpp(source_root: Path, benchmark_root: Path, overwrite: bool) -> dict:
    # Matches Depth-Anything-3 bench/datasets/scannetpp.py:
    #   <scene>/iphone/{rgb,depth,colmap}
    #   <scene>/scans/mesh_aligned_0.05.ply
    data_root = source_root / "scannetpp"
    if not data_root.is_dir():
        data_root = source_root
    output_root = benchmark_root / "MVRecon" / "scannetpp"
    records = []
    for scene in SCANNETPP_SCENES:
        scene_root = data_root / scene
        raw_iphone_root = scene_root / "iphone"
        legacy_root = scene_root / "merge_dslr_iphone"
        if raw_iphone_root.is_dir():
            image_root = raw_iphone_root / "rgb"
            depth_root = raw_iphone_root / "depth"
            sparse_root = raw_iphone_root / "colmap"
        else:
            image_root = legacy_root / "images"
            depth_root = legacy_root / "render_depth"
            sparse_root = legacy_root / "colmap" / "sparse_render_rgb"
        gt_mesh = scene_root / "scans" / "mesh_aligned_0.05.ply"
        if not image_root.is_dir():
            continue
        out_scene = output_root / scene
        copy_path(image_root, out_scene / "images", overwrite=overwrite)
        copy_path(depth_root, out_scene / "depths", overwrite=overwrite)
        if sparse_root.is_dir():
            copy_path(sparse_root, out_scene / "sparse" / "0", overwrite=overwrite)
        if gt_mesh.is_file():
            copy_path(gt_mesh, out_scene / "gt_mesh.ply", overwrite=overwrite)
        total_images = sum(
            1
            for path in image_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        records.append(
            {
                "dataset_name": "scannetpp",
                "scene_name": scene,
                "root_directory": str(out_scene),
                "total_images": total_images,
                "gt_mesh_path": "gt_mesh.ply",
            }
        )
    write_metadata(output_root, "scannetpp", records)
    return {"dataset": "scannetpp", "num_scenes": len(records), "output_root": str(output_root)}


def materialize_7scenes_gt_mesh(source_root: Path, benchmark_root: Path, overwrite: bool) -> dict:
    """Attach DA3-BENCH's fused 7Scenes ground-truth meshes to the existing
    canonical MVRecon/7scenes/<scene> layout.

    Unlike materialize_hiroom/materialize_scannetpp, this does not rebuild a
    scene's images/depths/sparse -- those are already materialized (by
    scripts/download/7scenes.sh, from the original Microsoft RGB-D release)
    and validated end to end. DA3-BENCH's 7scenes.zip is only the source for
    the TSDF-fused ground-truth mesh that
    depth_anything_3/bench/datasets/sevenscenes.py expects at
    <data_root>/7Scenes/meshes/<scene>.ply (see the DA3 benchmark docs:
    "fuse RGBD images with TSDF fusion and prepare ground-truth 3D shapes").
    """
    mesh_root = source_root / "7Scenes" / "meshes"
    output_root = benchmark_root / "MVRecon" / "7scenes"
    metadata_path = output_root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"7scenes metadata not found; expected it already materialized: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    records = []
    for scene_meta in metadata.get("scenes", []):
        scene = scene_meta["scene_name"]
        mesh_path = mesh_root / f"{scene}.ply"
        status = copy_path(mesh_path, output_root / scene / "gt_mesh.ply", overwrite=overwrite)
        if status in ("copied", "exists"):
            scene_meta["gt_mesh_path"] = "gt_mesh.ply"
        records.append({"scene_name": scene, "status": status})
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {"dataset": "7scenes", "num_scenes": len(records), "output_root": str(output_root), "records": records}


def materialize_eth3d_gt_mesh(source_root: Path, benchmark_root: Path, overwrite: bool) -> dict:
    """Attach DA3-BENCH's ETH3D ground-truth meshes to the existing
    canonical MVRecon/eth3d/<scene> layout.

    Like materialize_7scenes_gt_mesh, this does not rebuild a scene's
    images/depths/calibration -- those are already materialized (by
    scripts/download/eth3d.sh and scripts/preprocess/finalize_mvrecon_layout.sh,
    from the original ETH3D release). DA3-BENCH's eth3d.zip is only the
    source for the ground-truth mesh that depth_anything_3/bench/datasets/eth3d.py
    expects at
    <data_root>/eth3d/<scene>/combined_mesh.ply. DA3-BENCH ships 11 of
    ETH3D's scenes; any scene it does not cover (e.g. "meadow", "terrace")
    is reported as missing_source and left without a gt_mesh_path.
    """
    data_root = source_root / "eth3d"
    if not data_root.is_dir():
        data_root = source_root
    output_root = benchmark_root / "MVRecon" / "eth3d"
    metadata_path = output_root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"eth3d metadata not found; expected it already materialized: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    records = []
    for scene_meta in metadata.get("scenes", []):
        scene = scene_meta["scene_name"]
        mesh_path = data_root / scene / "combined_mesh.ply"
        status = copy_path(mesh_path, output_root / scene / "gt_mesh.ply", overwrite=overwrite)
        if status in ("copied", "exists"):
            scene_meta["gt_mesh_path"] = "gt_mesh.ply"
        records.append({"scene_name": scene, "status": status})
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {"dataset": "eth3d", "num_scenes": len(records), "output_root": str(output_root), "records": records}


def write_metadata(output_root: Path, dataset_name: str, scenes: list[dict]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    total_images = sum(int(scene.get("total_images", 0)) for scene in scenes)
    payload = {
        "dataset_name": dataset_name,
        "root_directory": str(output_root),
        "total_images": total_images,
        "average_images_per_scene": total_images / len(scenes) if scenes else 0.0,
        "scenes": scenes,
    }
    (output_root / "metadata.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--benchmark_root", type=Path, default=Path(os.path.expanduser("~/datasets/benchmarks")))
    parser.add_argument("--datasets", nargs="+", default=["scannetpp", "hiroom"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    materializers = {
        "hiroom": materialize_hiroom,
        "scannetpp": materialize_scannetpp,
        "scannet++": materialize_scannetpp,
        "7scenes": materialize_7scenes_gt_mesh,
        "eth3d": materialize_eth3d_gt_mesh,
    }
    summaries = []
    for dataset in args.datasets:
        key = dataset.lower()
        if key not in materializers:
            raise ValueError(f"Unsupported DA3-BENCH dataset materializer: {dataset}")
        summaries.append(materializers[key](args.source_root, args.benchmark_root, args.overwrite))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
