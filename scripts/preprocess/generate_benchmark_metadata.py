# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Generate metadata.json files for benchmark datasets.

The metadata schema is intentionally small and matches the benchmark task:

{
  "dataset_name": "...",
  "root_directory": "...",
  "total_images": 123,
  "average_images_per_scene": 12.3,
  "scenes": [
    {
      "dataset_name": "...",
      "scene_name": "...",
      "root_directory": "...",
      "total_images": 10
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

DEFAULT_TASK_DIRS = (
    "MonoDepth",
    "VideoDepth",
    "SfM",
    "WeakTextureSfM",
    "MVRecon",
)

EXCLUDED_IMAGE_PARTS = {
    "depth",
    "depths",
    "depth_110",
    "depthmap",
    "depthmaps",
    "disp",
    "disparity",
    "flow",
    "flows",
    "mask",
    "masks",
    "normal",
    "normals",
    "pose",
    "poses",
    "intrinsic",
    "intrinsics",
    "sparse",
    "segmentation",
}


PREFERRED_IMAGE_DIRS = (
    ("rgb_110",),
    ("images",),
    ("image",),
    ("dense", "images"),
    ("final",),
    ("color",),
    ("rgb",),
)


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _looks_like_depth_or_mask(path: Path) -> bool:
    lowered_parts = {part.lower() for part in path.parts}
    if lowered_parts & EXCLUDED_IMAGE_PARTS:
        return True

    stem = path.stem.lower()
    return any(
        token in stem
        for token in (
            "depth",
            "mask",
            "normal",
            "seg",
            "flow",
            "disp",
        )
    )


def count_images(scene_root: Path) -> int:
    """Count image files below a scene while skipping depth/mask-like files."""
    tar_files = sorted(scene_root.glob("*.tar"))
    if tar_files:
        return sum(count_images_in_tar(tar_path) for tar_path in tar_files)

    for relative_parts in PREFERRED_IMAGE_DIRS:
        candidate = scene_root.joinpath(*relative_parts)
        if candidate.is_dir():
            return sum(1 for path in candidate.rglob("*") if path.is_file() and _is_image_file(path))

    count = 0
    for path in scene_root.rglob("*"):
        if path.is_file() and _is_image_file(path) and not _looks_like_depth_or_mask(path):
            count += 1
    return count


def count_images_in_tar(tar_path: Path) -> int:
    count = 0
    with tarfile.open(tar_path) as tar:
        for member in tar:
            if not member.isfile():
                continue
            member_path = Path(member.name)
            if _is_image_file(member_path) and not _looks_like_depth_or_mask(member_path):
                count += 1
    return count


def _child_dirs(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(child for child in path.iterdir() if child.is_dir())


def _named_children(path: Path, names: Iterable[str]) -> list[Path]:
    return [path / name for name in names if (path / name).is_dir()]


def find_scene_roots(dataset_root: Path) -> list[Path]:
    """Return scene roots for known raw/preprocessed benchmark layouts."""
    name = dataset_root.name.lower()

    if name in {"sintel"} and (dataset_root / "training" / "final").is_dir():
        return _child_dirs(dataset_root / "training" / "final")

    if name in {"bonn"} and (dataset_root / "rgbd_bonn_dataset").is_dir():
        return _child_dirs(dataset_root / "rgbd_bonn_dataset")

    kitti_images = dataset_root / "depth_selection" / "val_selection_cropped" / "image_gathered"
    if name in {"kitti"} and kitti_images.is_dir():
        return _child_dirs(kitti_images)

    if name in {"7scenes", "seven_scenes"}:
        return _named_children(
            dataset_root,
            ("chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"),
        )

    if name == "dtu":
        return [
            child
            for child in sorted(dataset_root.iterdir())
            if child.is_dir()
            and child.name.startswith("scan")
            and (child / "images").is_dir()
        ]

    children = [
        child
        for child in _child_dirs(dataset_root)
        if child.name != "__MACOSX" and not child.name.startswith(".")
    ]
    if children:
        return children
    return [dataset_root] if dataset_root.is_dir() else []


def build_metadata(dataset_root: Path, dataset_name: str | None = None) -> dict:
    dataset_root = dataset_root.expanduser().absolute()
    dataset_name = dataset_name or dataset_root.name
    scene_roots = find_scene_roots(dataset_root)

    scenes = []
    total_images = 0
    for scene_root in scene_roots:
        num_images = count_images(scene_root)
        total_images += num_images
        scenes.append(
            {
                "dataset_name": dataset_name,
                "scene_name": scene_root.name,
                "root_directory": str(scene_root),
                "total_images": num_images,
            }
        )

    avg_images = total_images / len(scenes) if scenes else 0.0
    return {
        "dataset_name": dataset_name,
        "root_directory": str(dataset_root),
        "total_images": total_images,
        "average_images_per_scene": avg_images,
        "scenes": scenes,
    }


def discover_dataset_roots(benchmark_root: Path) -> list[Path]:
    roots: list[Path] = []
    for task_name in DEFAULT_TASK_DIRS:
        task_root = benchmark_root / task_name
        if not task_root.is_dir():
            continue
        roots.extend(
            child
            for child in sorted(task_root.iterdir())
            if child.is_dir() and child.name != "__MACOSX" and not child.name.startswith(".")
        )
    return roots


def write_metadata(dataset_root: Path, metadata: dict, filename: str = "metadata.json") -> Path:
    output_path = dataset_root / filename
    output_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark_root",
        type=Path,
        default=Path(os.path.expanduser("~/datasets/benchmarks")),
        help="Root containing task folders such as MonoDepth, VideoDepth, SfM, and MVRecon.",
    )
    parser.add_argument(
        "--dataset_roots",
        type=Path,
        nargs="*",
        default=None,
        help="Specific dataset roots to process. Defaults to all datasets discovered under benchmark_root.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print metadata summaries without writing metadata.json files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_roots = args.dataset_roots or discover_dataset_roots(args.benchmark_root)

    summaries = []
    for dataset_root in dataset_roots:
        if not dataset_root.is_dir():
            print(f"Skipping missing dataset root: {dataset_root}")
            continue
        metadata = build_metadata(dataset_root)
        summaries.append(
            {
                "dataset": metadata["dataset_name"],
                "root": metadata["root_directory"],
                "scenes": len(metadata["scenes"]),
                "total_images": metadata["total_images"],
            }
        )
        if not args.dry_run:
            output_path = write_metadata(dataset_root, metadata)
            print(f"Wrote {output_path}")

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
