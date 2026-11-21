#!/usr/bin/env python
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Materialize canonical Marigold MonoDepth subsets from tar-backed archives."""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatialarena.eval.dataset_registry import DEFAULT_BENCHMARK_ROOT  # noqa: E402


DEFAULT_MARIGOLD_ROOT = DEFAULT_BENCHMARK_ROOT / "MonoDepth" / "marigold"
SPLIT_ROOT = REPO_ROOT / "spatialarena" / "eval" / "monodepth_splits"
NYUV2_TAR = "nyu_labeled_extracted.tar"
RGB_PATTERN = re.compile(r"^(?P<prefix>.+/)(?P<kind>rgb|depth|filled)_(?P<index>\d+)\.png$")
NYUV2_DEFAULT_INTRINSICS = [
    [518.8579011745018, 0.0, 325.58244941119034],
    [0.0, 519.4696111212748, 253.73616633400462],
    [0.0, 0.0, 1.0],
]

SCENE_CONFIGS = {
    "nyuv2": {
        "archive": Path("nyuv2") / NYUV2_TAR,
        "split": SPLIT_ROOT / "nyu" / "labeled" / "filename_list_test.txt",
        "has_filled": True,
        "mask": "nyu_eigen",
        "depth_scale": 1e-3,
        "min_depth": 1e-3,
        "max_depth": 10.0,
    },
    "kitti": {
        "archive": Path("kitti") / "kitti_eigen_split_test.tar",
        "split": SPLIT_ROOT / "kitti" / "eigen_test_files_with_gt.txt",
        "mask": "kitti_eigen",
        "crop": "kitti_benchmark",
        "depth_scale": 1.0 / 256.0,
        "min_depth": 1e-5,
        "max_depth": 80.0,
    },
    "eth3d": {
        "archive": Path("eth3d") / "eth3d.tar",
        "split": SPLIT_ROOT / "eth3d" / "eth3d_filename_list.txt",
        "depth_scale": 1.0,
        "min_depth": 1e-5,
        "max_depth": None,
    },
    "diode": {
        "archive": Path("diode") / "diode_val.tar",
        "split": SPLIT_ROOT / "diode" / "diode_val_all_filename_list.txt",
        "mask": "third_column",
        "depth_scale": 1.0,
        "min_depth": 1e-5,
        "max_depth": 80.0,
    },
    "scannet": {
        "archive": Path("scannet") / "scannet_val_sampled_800_1.tar",
        "split": SPLIT_ROOT / "scannet" / "scannet_val_sampled_list_800_1.txt",
        "depth_scale": 1e-3,
        "min_depth": 1e-3,
        "max_depth": 10.0,
    },
}


@dataclass(frozen=True, kw_only=True)
class Nyuv2Pair:
    key: str
    rgb_member: str
    depth_member: str
    filled_member: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marigold_root", type=Path, default=DEFAULT_MARIGOLD_ROOT)
    parser.add_argument("--scene", choices=sorted(SCENE_CONFIGS), default="nyuv2")
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=sorted(SCENE_CONFIGS),
        default=None,
        help="Materialize multiple Marigold subsets. Overrides --scene.",
    )
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def relative_to(path: Path, base: Path) -> str:
    """``path`` relative to ``base`` (manifests store no absolute, machine-specific paths)."""
    return Path(path).resolve().relative_to(Path(base).resolve()).as_posix()


def normalized_member_name(name: str) -> str:
    return name[2:] if name.startswith("./") else name


def nyuv2_tar_path(marigold_root: Path) -> Path:
    return Path(marigold_root) / "nyuv2" / NYUV2_TAR


def collect_nyuv2_pairs(tar_path: Path) -> list[Nyuv2Pair]:
    grouped: dict[str, dict[str, str]] = {}
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = normalized_member_name(member.name)
            match = RGB_PATTERN.match(name)
            if match is None:
                continue
            prefix = match.group("prefix")
            index = match.group("index")
            key = f"{prefix.rstrip('/').replace('/', '__')}_{index}"
            grouped.setdefault(key, {})[match.group("kind")] = member.name

    pairs = []
    for key in sorted(grouped):
        files = grouped[key]
        if "rgb" not in files or "depth" not in files:
            continue
        pairs.append(
            Nyuv2Pair(
                key=key,
                rgb_member=files["rgb"],
                depth_member=files["depth"],
                filled_member=files.get("filled"),
            )
        )
    return pairs


def write_member(tar: tarfile.TarFile, member_name: str, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        return
    extracted = tar.extractfile(member_name)
    if extracted is None:
        raise ValueError(f"Could not extract {member_name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(extracted.read())


def read_split_lines(split_path: Path) -> list[list[str]]:
    with Path(split_path).open("r", encoding="utf-8") as file:
        return [line.split() for line in file.read().splitlines() if line.strip()]


def indexed_members(tar: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    return {
        normalized_member_name(member.name): member
        for member in tar.getmembers()
        if member.isfile()
    }


def extract_member_bytes(
    tar: tarfile.TarFile,
    members: dict[str, tarfile.TarInfo],
    relative_path: str,
) -> bytes:
    normalized = normalized_member_name(relative_path)
    member = members.get(normalized)
    if member is None:
        member = members.get(f"./{normalized}")
    if member is None:
        raise FileNotFoundError(f"Archive member not found: {relative_path}")
    extracted = tar.extractfile(member)
    if extracted is None:
        raise ValueError(f"Could not extract {relative_path}")
    return extracted.read()


def canonical_key(relative_path: str) -> str:
    path = Path(relative_path)
    return path.with_suffix("").as_posix().replace("/", "__")


def write_bytes(path: Path, content: bytes, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def read_image_bytes(content: bytes) -> Image.Image:
    return Image.open(io.BytesIO(content))


def kitti_benchmark_crop_array(array: np.ndarray) -> np.ndarray:
    crop_height = 352
    crop_width = 1216
    height, width = array.shape[:2]
    top = int(height - crop_height)
    left = int((width - crop_width) / 2)
    return array[top:top + crop_height, left:left + crop_width, ...]


def save_png_array(path: Path, array: np.ndarray, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def save_mask(path: Path, mask: np.ndarray, overwrite: bool) -> None:
    save_png_array(path, (mask.astype(np.uint8) * 255), overwrite=overwrite)


def nyu_eigen_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[45:min(471, height), 41:min(601, width)] = True
    return mask


def nyuv2_intrinsics() -> list[list[float]]:
    return [row[:] for row in NYUV2_DEFAULT_INTRINSICS]


def kitti_eigen_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[
        int(0.3324324 * height): int(0.91351351 * height),
        int(0.0359477 * width): int(0.96405229 * width),
    ] = True
    return mask


def materialize_nyuv2(
    *,
    marigold_root: Path,
    max_samples: int,
    overwrite: bool,
) -> dict:
    tar_path = nyuv2_tar_path(marigold_root)
    if not tar_path.is_file():
        raise FileNotFoundError(f"Marigold NYUv2 archive not found: {tar_path}")

    pairs = collect_nyuv2_pairs(tar_path)
    if max_samples > 0:
        pairs = pairs[:max_samples]
    if not pairs:
        raise ValueError(f"No RGB/depth pairs found in {tar_path}")

    scene_root = marigold_root / "nyuv2"
    images_dir = scene_root / "images"
    depths_dir = scene_root / "depths"
    filled_dir = scene_root / "filled_depths"
    records = []

    with tarfile.open(tar_path, "r") as tar:
        for pair in pairs:
            image_path = images_dir / f"{pair.key}.png"
            depth_path = depths_dir / f"{pair.key}.png"
            write_member(tar, pair.rgb_member, image_path, overwrite=overwrite)
            write_member(tar, pair.depth_member, depth_path, overwrite=overwrite)
            record = {
                "key": pair.key,
                "image": relative_to(image_path, scene_root),
                "depth": relative_to(depth_path, scene_root),
                "rgb_member": pair.rgb_member,
                "depth_member": pair.depth_member,
                "intrinsics": nyuv2_intrinsics(),
            }
            if pair.filled_member is not None:
                filled_path = filled_dir / f"{pair.key}.png"
                write_member(tar, pair.filled_member, filled_path, overwrite=overwrite)
                record["filled_depth"] = relative_to(filled_path, scene_root)
                record["filled_member"] = pair.filled_member
            records.append(record)

    manifest = {
        "dataset": "marigold",
        "scene": "nyuv2",
        "source_archive": relative_to(tar_path, marigold_root),
        "num_samples": len(records),
        "images_dir": relative_to(images_dir, scene_root),
        "depths_dir": relative_to(depths_dir, scene_root),
        "records": records,
    }
    manifest_path = scene_root / "materialized_sample.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def materialize_split_scene(
    *,
    marigold_root: Path,
    scene: str,
    max_samples: int,
    overwrite: bool,
) -> dict:
    config = SCENE_CONFIGS[scene]
    archive_path = marigold_root / config["archive"]
    split_path = Path(config["split"])
    if not archive_path.is_file():
        raise FileNotFoundError(f"Marigold archive not found: {archive_path}")
    if not split_path.is_file():
        raise FileNotFoundError(f"Marigold split file not found: {split_path}")

    split_rows = read_split_lines(split_path)
    if max_samples > 0:
        split_rows = split_rows[:max_samples]
    if not split_rows:
        raise ValueError(f"No samples selected for {scene}")

    scene_root = marigold_root / scene
    images_dir = scene_root / "images"
    depths_dir = scene_root / "depths"
    masks_dir = scene_root / "masks"
    filled_dir = scene_root / "filled_depths"
    records = []

    with tarfile.open(archive_path, "r") as tar:
        members = indexed_members(tar)
        for row_index, row in enumerate(split_rows):
            if len(row) < 2 or row[1] == "None":
                continue

            rgb_rel = row[0]
            depth_rel = row[1]
            filled_rel = row[2] if config.get("has_filled") and len(row) > 2 else None
            key = canonical_key(rgb_rel)
            image_suffix = Path(rgb_rel).suffix or ".png"
            depth_suffix = Path(depth_rel).suffix or ".dat"
            image_path = images_dir / f"{key}{image_suffix}"
            depth_path = depths_dir / f"{key}{depth_suffix}"

            if config.get("crop") == "kitti_benchmark":
                rgb_image = read_image_bytes(extract_member_bytes(tar, members, rgb_rel)).convert("RGB")
                depth_image = read_image_bytes(extract_member_bytes(tar, members, depth_rel))
                rgb_array = kitti_benchmark_crop_array(np.asarray(rgb_image))
                depth_array = kitti_benchmark_crop_array(np.asarray(depth_image))
                save_png_array(image_path, rgb_array, overwrite=overwrite)
                save_png_array(depth_path, depth_array, overwrite=overwrite)
                mask = kitti_eigen_mask(depth_array.shape[0], depth_array.shape[1])
                save_mask(masks_dir / f"{key}.png", mask, overwrite=overwrite)
            else:
                write_bytes(
                    image_path,
                    extract_member_bytes(tar, members, rgb_rel),
                    overwrite=overwrite,
                )
                write_bytes(
                    depth_path,
                    extract_member_bytes(tar, members, depth_rel),
                    overwrite=overwrite,
                )

                if config.get("mask") == "nyu_eigen":
                    depth_image = read_image_bytes(extract_member_bytes(tar, members, depth_rel))
                    mask = nyu_eigen_mask(depth_image.height, depth_image.width)
                    save_mask(masks_dir / f"{key}.png", mask, overwrite=overwrite)
                elif config.get("mask") == "third_column" and len(row) > 2:
                    mask_rel = row[2]
                    mask_path = masks_dir / f"{key}{Path(mask_rel).suffix or '.npy'}"
                    write_bytes(
                        mask_path,
                        extract_member_bytes(tar, members, mask_rel),
                        overwrite=overwrite,
                    )

            record = {
                "key": key,
                "image": relative_to(image_path, scene_root),
                "depth": relative_to(depth_path, scene_root),
                "rgb_member": rgb_rel,
                "depth_member": depth_rel,
                "split_row_index": row_index,
            }
            if filled_rel is not None:
                filled_path = filled_dir / f"{key}{Path(filled_rel).suffix or '.png'}"
                write_bytes(
                    filled_path,
                    extract_member_bytes(tar, members, filled_rel),
                    overwrite=overwrite,
                )
                record["filled_depth"] = relative_to(filled_path, scene_root)
                record["filled_member"] = filled_rel
            if (masks_dir / f"{key}.png").exists():
                record["mask"] = relative_to(masks_dir / f"{key}.png", scene_root)
            elif config.get("mask") == "third_column" and len(row) > 2:
                mask_path = masks_dir / f"{key}{Path(row[2]).suffix or '.npy'}"
                if mask_path.exists():
                    record["mask"] = relative_to(mask_path, scene_root)
                    record["mask_member"] = row[2]
            records.append(record)

    manifest = {
        "dataset": "marigold",
        "scene": scene,
        "source_archive": relative_to(archive_path, marigold_root),
        "split_file": relative_to(split_path, REPO_ROOT),
        "num_samples": len(records),
        "images_dir": relative_to(images_dir, scene_root),
        "depths_dir": relative_to(depths_dir, scene_root),
        "masks_dir": relative_to(masks_dir, scene_root) if masks_dir.exists() else None,
        "depth_scale": config["depth_scale"],
        "min_depth": config["min_depth"],
        "max_depth": config["max_depth"],
        "records": records,
    }
    scene_root.mkdir(parents=True, exist_ok=True)
    (scene_root / "materialized_full.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def write_marigold_metadata(marigold_root: Path, manifests: list[dict]) -> dict:
    """Write ``metadata.json`` for the materialized scenes, keeping the entries of other scenes already in it (so
    materializing one subset, e.g. ScanNet after a Hugging Face download, adds it instead of replacing the rest)."""
    metadata_path = marigold_root / "metadata.json"
    entries = {}
    if metadata_path.is_file():
        entries = {scene["scene_name"]: scene for scene in json.loads(metadata_path.read_text()).get("scenes", [])}
    for manifest in manifests:
        scene = manifest["scene"]
        total = int(manifest["num_samples"])
        entries[scene] = {
            "dataset_name": "marigold",
            "scene_name": scene,
            "root_directory": str(marigold_root / scene),
            "total_images": total,
            "total_depth_maps": total,
            "layout": "marigold_materialized",
            "depth_intrinsics": nyuv2_intrinsics() if scene == "nyuv2" else None,
            "source_archive": manifest["source_archive"],
            "split_file": manifest.get("split_file"),
            "depth_scale": manifest.get("depth_scale"),
            "min_depth": manifest.get("min_depth"),
            "max_depth": manifest.get("max_depth"),
        }
    order = list(SCENE_CONFIGS)
    scenes = sorted(entries.values(), key=lambda s: order.index(s["scene_name"]) if s["scene_name"] in order else 99)
    total_images = sum(int(scene["total_images"]) for scene in scenes)

    metadata = {
        "dataset_name": "marigold",
        "task": "MonoDepth",
        "root_directory": str(marigold_root),
        "total_images": total_images,
        "average_images_per_scene": total_images / len(scenes) if scenes else 0.0,
        "scenes": scenes,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def materialize_marigold_scenes(
    *,
    marigold_root: Path,
    scenes: list[str],
    max_samples: int,
    overwrite: bool,
) -> dict:
    manifests = [
        materialize_split_scene(
            marigold_root=marigold_root,
            scene=scene,
            max_samples=max_samples,
            overwrite=overwrite,
        )
        for scene in scenes
    ]
    metadata = write_marigold_metadata(marigold_root, manifests)
    return {
        "manifests": manifests,
        "metadata": metadata,
    }


def main() -> None:
    args = parse_args()
    if args.max_samples == 0:
        raise SystemExit("--max_samples must be positive or negative for all samples")
    selected_scenes = args.scenes or [args.scene]
    result = materialize_marigold_scenes(
        marigold_root=args.marigold_root.expanduser(),
        scenes=selected_scenes,
        max_samples=args.max_samples,
        overwrite=args.overwrite,
    )
    print(json.dumps(result["metadata"], indent=2))


if __name__ == "__main__":
    main()
