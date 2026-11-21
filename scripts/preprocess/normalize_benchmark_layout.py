# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Create canonical benchmark dataset layouts by copying files.

This script preserves downloaded raw data and adds a task-friendly layout:

DATASET_ROOT/
  metadata.json
  scene/
    images/
    depths/        optional
    masks/         optional
    sparse/0/      optional COLMAP model copy

Every entry is a real copy, never a symlink -- the normalized layout is
meant to be a self-contained, redistributable dataset (e.g. for a Hugging
Face release). Re-running this script also converts any symlink left by an
older version of it into a real copy. Only sources that already exist are
copied. Missing optional folders are skipped.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SEVEN_SCENES = ("chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs")


def copy_entry(src: Path, dst: Path, dry_run: bool = False) -> str:
    src = src.absolute()
    if not src.exists():
        return "missing_source"
    if dst.is_symlink():
        # Convert a pre-existing symlink (from an older run of this script)
        # into a real copy.
        if dry_run:
            return "would_copy"
        dst.unlink()
    elif dst.exists():
        return "exists"
    if dry_run:
        return "would_copy"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, symlinks=False)
    else:
        shutil.copy2(src, dst, follow_symlinks=True)
    return "copied"


def link_dir(src: Path, dst: Path, stats: dict[str, int], dry_run: bool = False) -> None:
    status = copy_entry(src, dst, dry_run=dry_run)
    stats[status] = stats.get(status, 0) + 1


def link_matching_files(
    src_dir: Path,
    dst_dir: Path,
    pattern: str,
    stats: dict[str, int],
    dry_run: bool = False,
    prefix: str | None = None,
) -> None:
    if not src_dir.is_dir():
        stats["missing_source"] = stats.get("missing_source", 0) + 1
        return

    for src in sorted(src_dir.glob(pattern)):
        name = f"{prefix}_{src.name}" if prefix else src.name
        status = copy_entry(src, dst_dir / name, dry_run=dry_run)
        stats[status] = stats.get(status, 0) + 1


def normalize_bonn(root: Path, dry_run: bool = False) -> dict[str, int]:
    stats: dict[str, int] = {}
    raw_root = root / "rgbd_bonn_dataset"
    for scene_root in sorted(raw_root.iterdir()) if raw_root.is_dir() else []:
        if not scene_root.is_dir():
            continue
        out_scene = root / scene_root.name
        link_dir(scene_root / "rgb_110", out_scene / "images", stats, dry_run=dry_run)
        link_dir(scene_root / "depth_110", out_scene / "depths", stats, dry_run=dry_run)
    return stats


def normalize_sintel(root: Path, dry_run: bool = False) -> dict[str, int]:
    stats: dict[str, int] = {}
    image_root = root / "training" / "final"
    depth_root = root / "training" / "depth"
    for scene_images in sorted(image_root.iterdir()) if image_root.is_dir() else []:
        if not scene_images.is_dir():
            continue
        out_scene = root / scene_images.name
        link_dir(scene_images, out_scene / "images", stats, dry_run=dry_run)
        link_dir(depth_root / scene_images.name, out_scene / "depths", stats, dry_run=dry_run)
    return stats


def normalize_kitti(root: Path, dry_run: bool = False) -> dict[str, int]:
    stats: dict[str, int] = {}
    base = root / "depth_selection" / "val_selection_cropped"
    image_root = base / "image_gathered"
    depth_root = base / "groundtruth_depth_gathered"
    for scene_images in sorted(image_root.iterdir()) if image_root.is_dir() else []:
        if not scene_images.is_dir():
            continue
        out_scene = root / scene_images.name
        link_dir(scene_images, out_scene / "images", stats, dry_run=dry_run)
        link_dir(depth_root / scene_images.name, out_scene / "depths", stats, dry_run=dry_run)
    return stats


def normalize_imc(root: Path, dry_run: bool = False) -> dict[str, int]:
    stats: dict[str, int] = {}
    for scene_root in sorted(root.iterdir()) if root.is_dir() else []:
        if not scene_root.is_dir() or scene_root.name.startswith("."):
            continue
        dense_root = scene_root / "dense"
        link_dir(dense_root / "images", scene_root / "images", stats, dry_run=dry_run)
        link_dir(dense_root / "sparse", scene_root / "sparse" / "0", stats, dry_run=dry_run)
        clean_depths = dense_root / "stereo" / "depth_maps_clean_300_th_0.10"
        if clean_depths.is_dir():
            link_dir(clean_depths, scene_root / "depths", stats, dry_run=dry_run)
    return stats


def normalize_7scenes(root: Path, dry_run: bool = False) -> dict[str, int]:
    stats: dict[str, int] = {}
    for scene_name in SEVEN_SCENES:
        scene_root = root / scene_name
        if not scene_root.is_dir():
            stats["missing_source"] = stats.get("missing_source", 0) + 1
            continue
        for seq_root in sorted(scene_root.glob("seq-*")):
            if not seq_root.is_dir():
                continue
            prefix = seq_root.name
            link_matching_files(
                seq_root,
                scene_root / "images",
                "frame-*.color.png",
                stats,
                dry_run=dry_run,
                prefix=prefix,
            )
            projected_depths = list(seq_root.glob("frame-*.depth.proj.png"))
            depth_pattern = "frame-*.depth.proj.png" if projected_depths else "frame-*.depth.png"
            link_matching_files(
                seq_root,
                scene_root / "depths",
                depth_pattern,
                stats,
                dry_run=dry_run,
                prefix=prefix,
            )
    return stats


NORMALIZERS = {
    "7scenes": normalize_7scenes,
    "seven_scenes": normalize_7scenes,
    "bonn": normalize_bonn,
    "imc": normalize_imc,
    "kitti": normalize_kitti,
    "sintel": normalize_sintel,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_roots", type=Path, nargs="+")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for root in args.dataset_roots:
        dataset_name = root.name.lower()
        normalizer = NORMALIZERS.get(dataset_name)
        if normalizer is None:
            print(f"{root}: no normalizer registered")
            continue
        stats = normalizer(root, dry_run=args.dry_run)
        print(f"{root}: {stats}")


if __name__ == "__main__":
    main()
