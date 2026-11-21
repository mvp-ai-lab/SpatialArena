# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Link extracted MegaSaM-distributed DyCheck camera-0 RGB/depth into VideoDepth.

Source: <scene>/dense/rgb/1x/0_*.png and <scene>/depth/2x/0_*.npy.
Native arrays and units are preserved. No resizing, prediction or GT filtering
is performed. Optional validity masks must be supplied explicitly.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

SCENES = (
    "apple", "backpack", "block", "creeper", "handwavy", "haru-sit",
    "mochi-high-five", "paper-windmill", "pillow", "spin", "sriracha-tree", "teddy",
)


def _files(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    result = {}
    for path in directory.glob("0_*"):
        if path.suffix.lower() not in suffixes or not path.is_file():
            continue
        if path.stem in result:
            raise ValueError(f"Duplicate camera-0 frame: {path.stem}")
        int(path.stem.removeprefix("0_"))
        result[path.stem] = path
    return result


def _scene_records(source: Path, scene: str, mask_subdir: str | None) -> list[dict]:
    rgb = _files(source / scene / "dense/rgb/1x", {".png", ".jpg", ".jpeg"})
    depth = _files(source / scene / "depth/2x", {".npy"})
    if not rgb or set(rgb) != set(depth):
        raise ValueError(f"Unmatched camera-0 RGB/depth in {scene}: RGB-only={sorted(rgb.keys() - depth.keys())}, "
                         f"depth-only={sorted(depth.keys() - rgb.keys())}")
    masks = _files(source / scene / mask_subdir, {".npy", ".png"}) if mask_subdir else {}
    if mask_subdir and set(masks) != set(rgb):
        raise ValueError(f"Unmatched validity masks in {scene}")
    records = []
    for index, stem in enumerate(sorted(rgb, key=lambda value: int(value.removeprefix("0_")))):
        with Image.open(rgb[stem]) as image:
            width, height = image.size
        array = np.load(depth[stem], mmap_mode="r", allow_pickle=False).squeeze()
        if array.ndim != 2 or array.size == 0 or not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"Invalid depth array: {depth[stem]}")
        if height * array.shape[1] != width * array.shape[0]:
            raise ValueError(f"RGB/depth aspect ratios differ: {scene}/{stem}")
        record = {"frame": f"{index:06d}", "source_frame": stem,
                  "rgb": rgb[stem], "depth": depth[stem],
                  "rgb_wh": [width, height], "depth_wh": list(array.shape[::-1])}
        if masks:
            path = masks[stem]
            if path.suffix == ".npy":
                mask = np.load(path, allow_pickle=False).squeeze()
            else:
                with Image.open(path) as image:
                    mask = np.asarray(image).squeeze()
            if mask.shape != array.shape:
                raise ValueError(f"Mask/depth shapes differ: {path}")
            record["mask"] = path
        records.append(record)
    return records


def _write_json(path: Path, data: object) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def materialize_dataset(
    source_root: Path, benchmark_root: Path, *, scenes: list[str] | None = None,
    mask_subdir: str | None = None,
) -> Path:
    """Validate all pairs before linking; reruns cannot overwrite different data."""
    source_root = Path(source_root).resolve()
    root = Path(benchmark_root).resolve() / "VideoDepth/dycheck"
    selected = list(SCENES) if scenes is None else scenes
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Choose a nonempty, unique scene list")
    if any(Path(scene).name != scene or scene in {".", ".."} for scene in selected):
        raise ValueError("Scene names must be single directory names")
    if mask_subdir and (Path(mask_subdir).is_absolute() or ".." in Path(mask_subdir).parts):
        raise ValueError("mask_subdir must be relative to each source scene")
    records = {scene: _scene_records(source_root, scene, mask_subdir) for scene in selected}
    metadata_path = root / "metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if {s["scene_name"] for s in previous["scenes"]} != set(selected):
            raise FileExistsError("Existing dataset has a different scene selection; use a separate output root")
    links = []
    manifests = {}
    for scene, frames in records.items():
        manifest = []
        for record in frames:
            entry = {key: record[key] for key in ("frame", "source_frame", "rgb_wh", "depth_wh")}
            for key, folder in (("rgb", "images"), ("depth", "depths"), ("mask", "masks")):
                if key not in record:
                    continue
                source = record[key]
                target = root / scene / folder / (record["frame"] + source.suffix)
                if target.exists() or target.is_symlink():
                    if not target.is_symlink() or target.resolve() != source.resolve():
                        raise FileExistsError(f"Refusing to replace existing data: {target}")
                links.append((source, target))
                entry[key] = target.relative_to(root / scene).as_posix()
                entry[f"source_{key}"] = source.relative_to(source_root).as_posix()
            manifest.append(entry)
        manifest_path = root / scene / "frame_manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise FileExistsError(f"Existing frame manifest differs: {manifest_path}")
        manifests[scene] = manifest
    for source, target in links:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_symlink():
            target.symlink_to(os.path.relpath(source, target.parent))
    summaries = []
    for scene, frames in records.items():
        _write_json(root / scene / "frame_manifest.json", manifests[scene])
        summaries.append({"scene_name": scene, "root_directory": scene, "total_images": len(frames),
                          "gt_depth_scale": 1.0, "rgb_wh": frames[0]["rgb_wh"],
                          "depth_wh": frames[0]["depth_wh"], "camera": 0,
                          "pairing": "matching source stems, numeric frame order", "has_masks": bool(mask_subdir)})
    _write_json(metadata_path, {"dataset_name": "dycheck", "root_directory": ".",
                               "total_images": sum(len(frames) for frames in records.values()),
                               "scenes": summaries})
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--benchmark_root", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", choices=SCENES)
    parser.add_argument("--mask_subdir", help="Optional camera-0 masks, relative to each source scene")
    args = parser.parse_args()
    print(materialize_dataset(args.source_root, args.benchmark_root, scenes=args.scenes,
                              mask_subdir=args.mask_subdir))


if __name__ == "__main__":
    main()
