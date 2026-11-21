# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Materialize the DetectorFreeSfM weak-texture dataset into the benchmark layout."""

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

from spatialarena.geometry.read_write_model import BaseImage, Camera, rotmat2qvec, write_model  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def link_path(src: Path, dst: Path, overwrite: bool = False, copy: bool = True) -> str:
    """Populate `dst` from `src`. Copies by default so the materialized
    layout is self-contained (matches normalize_benchmark_layout.py's "every
    entry is a real copy, never a symlink" convention) and doesn't depend on
    the raw source_root staying on disk; pass copy=False for the old
    space-saving symlink behavior."""
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
    if copy:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        return "copied"
    dst.symlink_to(src, target_is_directory=src.is_dir())
    return "linked"


def discover_dataset_root(source_root: Path) -> Path:
    candidates = [
        source_root / "TexturePoorSfM_dataset",
        source_root / "SfM_dataset" / "TexturePoorSfM_dataset",
        source_root / "TexturePoorSfM",
        source_root,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return source_root


def image_dir(scene_root: Path) -> Path | None:
    for candidate in (scene_root / "images", scene_root / "image", scene_root / "rgb"):
        if candidate.is_dir():
            return candidate
    return None


def pose_dir(scene_root: Path) -> Path | None:
    candidate = scene_root / "poses"
    return candidate if candidate.is_dir() else None


def intrinsic_dir(scene_root: Path) -> Path | None:
    candidate = scene_root / "intrins"
    return candidate if candidate.is_dir() else None


def read_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float64)


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def export_sparse_model(scene_root: Path, output_sparse: Path, image_names: list[str], overwrite: bool) -> int:
    if output_sparse.exists() and not overwrite:
        return 0

    cameras: dict[int, Camera] = {}
    images: dict[int, BaseImage] = {}
    output_sparse.mkdir(parents=True, exist_ok=True)

    for image_id, image_name in enumerate(image_names, start=1):
        image_path = scene_root / "images" / image_name
        pose_path = scene_root / "poses" / f"{Path(image_name).stem}.txt"
        intrinsic_path = scene_root / "intrins" / f"{Path(image_name).stem}.txt"
        if not pose_path.is_file() or not intrinsic_path.is_file():
            continue

        width, height = image_size(image_path)
        k_matrix = read_matrix(intrinsic_path)
        pose_matrix = read_matrix(pose_path)
        if pose_matrix.shape != (4, 4) or k_matrix.shape != (3, 3):
            continue

        # TexturePoorSfM pose files are already world-to-camera (DetectorFreeSfM's evaluator loads them as
        # "w2c pose"). An earlier version inverted them here, which broke epipolar consistency with the images.
        world_to_camera = pose_matrix
        cameras[image_id] = Camera(
            id=image_id,
            model="PINHOLE",
            width=width,
            height=height,
            params=np.array(
                [
                    float(k_matrix[0, 0]),
                    float(k_matrix[1, 1]),
                    float(k_matrix[0, 2]),
                    float(k_matrix[1, 2]),
                ],
                dtype=np.float64,
            ),
        )
        images[image_id] = BaseImage(
            id=image_id,
            qvec=rotmat2qvec(world_to_camera[:3, :3]).astype(np.float64),
            tvec=world_to_camera[:3, 3].astype(np.float64),
            camera_id=image_id,
            name=image_name,
            xys=np.empty((0, 2), dtype=np.float64),
            point3D_ids=np.empty((0,), dtype=np.int64),
        )

    write_model(cameras, images, {}, str(output_sparse), ext=".txt")
    return len(images)


def count_images(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)


def is_scene_root(path: Path) -> bool:
    return image_dir(path) is not None and pose_dir(path) is not None and intrinsic_dir(path) is not None


def discover_scene_roots(dataset_root: Path) -> list[Path]:
    candidates = [path for path in dataset_root.rglob("*") if path.is_dir() and is_scene_root(path)]
    selected: list[Path] = []
    for candidate in sorted(candidates, key=lambda path: (len(path.parts), str(path))):
        if any(candidate == existing or candidate.is_relative_to(existing) for existing in selected):
            continue
        selected.append(candidate)
    return selected


def scene_name_from_root(dataset_root: Path, scene_root: Path) -> str:
    relative = scene_root.relative_to(dataset_root).as_posix()
    return relative.replace("/", "__") or scene_root.name


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


def materialize_dataset(source_root: Path, benchmark_root: Path, overwrite: bool, copy: bool = True) -> dict:
    dataset_root = discover_dataset_root(source_root)
    output_root = benchmark_root / "SfM" / "texturepoorsfm"
    records = []
    for scene_root in discover_scene_roots(dataset_root):
        scene_name = scene_name_from_root(dataset_root, scene_root)
        out_scene = output_root / scene_name
        img_dir = image_dir(scene_root)
        pose_root = pose_dir(scene_root)
        intrins_root = intrinsic_dir(scene_root)
        if img_dir is None or pose_root is None or intrins_root is None:
            continue
        link_path(img_dir, out_scene / "images", overwrite=overwrite, copy=copy)
        depth_dir = scene_root / "depths"
        if depth_dir.is_dir():
            link_path(depth_dir, out_scene / "depths", overwrite=overwrite, copy=copy)
        image_names = sorted(
            path.name
            for path in img_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        num_sparse = export_sparse_model(scene_root, out_scene / "sparse" / "0", image_names, overwrite=overwrite)
        records.append(
            {
                "dataset_name": "texturepoorsfm",
                "scene_name": scene_name,
                "source_scene": str(scene_root.relative_to(dataset_root)),
                "root_directory": str(out_scene),
                "total_images": count_images(img_dir),
                "num_sparse_images": num_sparse,
            }
        )
    write_metadata(output_root, "texturepoorsfm", records)
    return {"dataset": "texturepoorsfm", "num_scenes": len(records), "output_root": str(output_root)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--benchmark_root", type=Path, default=Path(os.path.expanduser("~/datasets/benchmarks")))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Symlink images/depths back to source_root instead of copying them (old, space-saving "
        "but not self-contained behavior; source_root must stay on disk).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = materialize_dataset(
        args.source_root.expanduser(),
        args.benchmark_root.expanduser(),
        args.overwrite,
        copy=not args.symlink,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
