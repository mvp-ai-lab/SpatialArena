# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Materialize the DetectorFreeSfM weak-texture dataset into the benchmark.

Writes benchmark_root/SfM/texturepoorsfm: one scene per DetectorFreeSfM scene
with `images/`, `depths/` (when present) and a text COLMAP model in `sparse/0`
built from the per-image `poses/` and `intrins/` files.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatialarena.geometry import read_write_model  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def link_path(src: Path,
              dst: Path,
              overwrite: bool = False,
              copy: bool = True) -> str:
    """Populates `dst` from `src`.

    Copies by default so the materialized layout is self-contained (matches
    normalize_benchmark_layout.py's "every entry is a real copy, never a
    symlink" convention) and doesn't depend on the raw source_root staying on
    disk; pass copy=False for the old space-saving symlink behavior.

    Args:
        src: The source file or directory.
        dst: The destination path.
        overwrite: Replace an existing `dst`.
        copy: Copy `src`; when False, symlink to it.

    Returns:
        "missing_source", "exists", "copied" or "linked".
    """
    src = src.resolve(strict=False)
    if not src.exists():
        return "missing_source"
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return "exists"
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
        return "copied"
    dst.symlink_to(src, target_is_directory=src.is_dir())
    return "linked"


def discover_dataset_root(source_root: Path) -> Path:
    """Finds the TexturePoorSfM dataset directory under `source_root`."""
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
    """Returns the scene's image directory, or None if it has none."""
    for candidate in (scene_root / "images", scene_root / "image",
                      scene_root / "rgb"):
        if candidate.is_dir():
            return candidate
    return None


def pose_dir(scene_root: Path) -> Path | None:
    """Returns the scene's `poses/` directory, or None if it has none."""
    candidate = scene_root / "poses"
    if candidate.is_dir():
        return candidate
    return None


def intrinsic_dir(scene_root: Path) -> Path | None:
    """Returns the scene's `intrins/` directory, or None if it has none."""
    candidate = scene_root / "intrins"
    if candidate.is_dir():
        return candidate
    return None


def read_matrix(path: Path) -> np.ndarray:
    """Reads a whitespace-separated matrix as float64."""
    return np.loadtxt(path, dtype=np.float64)


def image_size(path: Path) -> tuple[int, int]:
    """Returns an image's (width, height)."""
    from PIL import Image  # pylint: disable=import-outside-toplevel

    with Image.open(path) as image:
        return image.size


def export_sparse_model(scene_root: Path, output_sparse: Path,
                        image_names: list[str], overwrite: bool) -> int:
    """Writes a text COLMAP model from the scene's poses and intrinsics.

    Images without a pose or intrinsics file, or with malformed matrices, are
    left out of the model.

    Args:
        scene_root: The source scene with images/, poses/ and intrins/.
        output_sparse: The output model directory (sparse/0).
        image_names: The image file names to export, in id order.
        overwrite: Rewrite an existing model.

    Returns:
        The number of images written; 0 when the model exists and is kept.
    """
    if output_sparse.exists() and not overwrite:
        return 0

    cameras: dict[int, read_write_model.Camera] = {}
    images: dict[int, read_write_model.BaseImage] = {}
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

        # TexturePoorSfM pose files are already world-to-camera
        # (DetectorFreeSfM's evaluator loads them as "w2c pose"). An earlier
        # version inverted them here, which broke epipolar consistency with
        # the images.
        world_to_camera = pose_matrix
        cameras[image_id] = read_write_model.Camera(
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
        images[image_id] = read_write_model.BaseImage(
            id=image_id,
            qvec=read_write_model.rotmat2qvec(world_to_camera[:3, :3]).astype(
                np.float64),
            tvec=world_to_camera[:3, 3].astype(np.float64),
            camera_id=image_id,
            name=image_name,
            xys=np.empty((0, 2), dtype=np.float64),
            point3D_ids=np.empty((0,), dtype=np.int64),
        )

    read_write_model.write_model(
        cameras,
        images,
        {},
        str(output_sparse),
        ext=".txt",
    )
    return len(images)


def count_images(path: Path) -> int:
    """Counts the image files anywhere under `path`."""
    return sum(1 for item in path.rglob("*")
               if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)


def is_scene_root(path: Path) -> bool:
    """Whether `path` has image, pose and intrinsics directories."""
    return (image_dir(path) is not None and pose_dir(path) is not None and
            intrinsic_dir(path) is not None)


def discover_scene_roots(dataset_root: Path) -> list[Path]:
    """Finds the outermost scene directories under `dataset_root`."""
    candidates = [
        path for path in dataset_root.rglob("*")
        if path.is_dir() and is_scene_root(path)
    ]
    selected: list[Path] = []
    for candidate in sorted(candidates,
                            key=lambda path: (len(path.parts), str(path))):
        if any(candidate == existing or candidate.is_relative_to(existing)
               for existing in selected):
            continue
        selected.append(candidate)
    return selected


def scene_name_from_root(dataset_root: Path, scene_root: Path) -> str:
    """Names a scene by its relative path, with "/" replaced by "__"."""
    relative = scene_root.relative_to(dataset_root).as_posix()
    return relative.replace("/", "__") or scene_root.name


def write_metadata(output_root: Path, dataset_name: str,
                   scenes: list[dict]) -> None:
    """Writes the dataset-level metadata.json over the scene records."""
    output_root.mkdir(parents=True, exist_ok=True)
    total_images = sum(int(scene.get("total_images", 0)) for scene in scenes)
    payload = {
        "dataset_name":
            dataset_name,
        "root_directory":
            str(output_root),
        "total_images":
            total_images,
        "average_images_per_scene":
            total_images / len(scenes) if scenes else 0.0,
        "scenes":
            scenes,
    }
    (output_root / "metadata.json").write_text(json.dumps(payload, indent=2) +
                                               "\n",
                                               encoding="utf-8")


def materialize_dataset(source_root: Path,
                        benchmark_root: Path,
                        overwrite: bool,
                        copy: bool = True) -> dict:
    """Materializes every scene into benchmark_root/SfM/texturepoorsfm.

    Args:
        source_root: The downloaded DetectorFreeSfM dataset.
        benchmark_root: The benchmark root to write into.
        overwrite: Replace existing outputs.
        copy: Copy images and depths; when False, symlink to them.

    Returns:
        A summary with the dataset name, scene count and output root.
    """
    dataset_root = discover_dataset_root(source_root)
    output_root = benchmark_root / "SfM" / "texturepoorsfm"
    records = []
    for scene_root in discover_scene_roots(dataset_root):
        scene_name = scene_name_from_root(dataset_root, scene_root)
        out_scene = output_root / scene_name
        images_root = image_dir(scene_root)
        pose_root = pose_dir(scene_root)
        intrins_root = intrinsic_dir(scene_root)
        if images_root is None or pose_root is None or intrins_root is None:
            continue
        link_path(images_root,
                  out_scene / "images",
                  overwrite=overwrite,
                  copy=copy)
        depth_dir = scene_root / "depths"
        if depth_dir.is_dir():
            link_path(depth_dir,
                      out_scene / "depths",
                      overwrite=overwrite,
                      copy=copy)
        image_names = sorted(
            path.name
            for path in images_root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        num_sparse = export_sparse_model(scene_root,
                                         out_scene / "sparse" / "0",
                                         image_names,
                                         overwrite=overwrite)
        records.append({
            "dataset_name": "texturepoorsfm",
            "scene_name": scene_name,
            "source_scene": str(scene_root.relative_to(dataset_root)),
            "root_directory": str(out_scene),
            "total_images": count_images(images_root),
            "num_sparse_images": num_sparse,
        })
    write_metadata(output_root, "texturepoorsfm", records)
    return {
        "dataset": "texturepoorsfm",
        "num_scenes": len(records),
        "output_root": str(output_root)
    }


def parse_args() -> argparse.Namespace:
    """Parses the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--benchmark_root",
                        type=Path,
                        default=Path(
                            os.path.expanduser("~/datasets/benchmarks")))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--symlink",
        action="store_true",
        help=("Symlink images/depths back to source_root instead of copying "
              "them (old, space-saving but not self-contained behavior; "
              "source_root must stay on disk)."),
    )
    return parser.parse_args()


def main() -> None:
    """Materializes the dataset and prints the summary."""
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
