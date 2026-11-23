# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Link extracted MegaSaM DyCheck camera-0 RGB/depth into VideoDepth.

Source: <scene>/dense/rgb/1x/0_*.png and <scene>/depth/2x/0_*.npy.
Native arrays and units are preserved. No resizing, prediction or GT filtering
is performed. Optional validity masks must be supplied explicitly.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib

import numpy as np
from PIL import Image

SCENES = (
    "apple",
    "backpack",
    "block",
    "creeper",
    "handwavy",
    "haru-sit",
    "mochi-high-five",
    "paper-windmill",
    "pillow",
    "spin",
    "sriracha-tree",
    "teddy",
)


def _files(directory: pathlib.Path,
           suffixes: set[str]) -> dict[str, pathlib.Path]:
    """Maps each camera-0 file stem (`0_<n>`) in `directory` to its path.

    Raises:
        ValueError: Two files share a stem, or a stem is not `0_<integer>`.
    """
    result = {}
    for path in directory.glob("0_*"):
        if path.suffix.lower() not in suffixes or not path.is_file():
            continue
        if path.stem in result:
            raise ValueError(f"Duplicate camera-0 frame: {path.stem}")
        int(path.stem.removeprefix("0_"))
        result[path.stem] = path
    return result


def _load_mask(path: pathlib.Path) -> np.ndarray:
    """Loads a validity mask from `.npy` or an image file."""
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False).squeeze()
    with Image.open(path) as image:
        return np.asarray(image).squeeze()


def _scene_records(source: pathlib.Path, scene: str,
                   mask_subdir: str | None) -> list[dict]:
    """Pairs and validates one scene's camera-0 RGB, depth and mask files.

    Args:
        source: Source root.
        scene: Scene name.
        mask_subdir: Optional mask folder, relative to the scene.

    Returns:
        One record per frame in numeric frame order, with the output frame name,
        source stem, source paths and RGB/depth sizes.

    Raises:
        ValueError: The files do not pair up, a depth array is invalid, the
            RGB/depth aspect ratios differ or a mask's shape differs from the
            depth's.
    """
    rgb = _files(source / scene / "dense/rgb/1x", {".png", ".jpg", ".jpeg"})
    depth = _files(source / scene / "depth/2x", {".npy"})
    if not rgb or set(rgb) != set(depth):
        raise ValueError(f"Unmatched camera-0 RGB/depth in {scene}: "
                         f"RGB-only={sorted(rgb.keys() - depth.keys())}, "
                         f"depth-only={sorted(depth.keys() - rgb.keys())}")
    masks = {}
    if mask_subdir:
        masks = _files(source / scene / mask_subdir, {".npy", ".png"})
    if mask_subdir and set(masks) != set(rgb):
        raise ValueError(f"Unmatched validity masks in {scene}")
    records = []
    stems = sorted(rgb, key=lambda value: int(value.removeprefix("0_")))
    for index, stem in enumerate(stems):
        with Image.open(rgb[stem]) as image:
            width, height = image.size
        array = np.load(depth[stem], mmap_mode="r",
                        allow_pickle=False).squeeze()
        if array.ndim != 2 or array.size == 0 or not np.issubdtype(
                array.dtype, np.number):
            raise ValueError(f"Invalid depth array: {depth[stem]}")
        if height * array.shape[1] != width * array.shape[0]:
            raise ValueError(f"RGB/depth aspect ratios differ: {scene}/{stem}")
        record = {
            "frame": f"{index:06d}",
            "source_frame": stem,
            "rgb": rgb[stem],
            "depth": depth[stem],
            "rgb_wh": [width, height],
            "depth_wh": list(array.shape[::-1])
        }
        if masks:
            path = masks[stem]
            mask = _load_mask(path)
            if mask.shape != array.shape:
                raise ValueError(f"Mask/depth shapes differ: {path}")
            record["mask"] = path
        records.append(record)
    return records


def _write_json(path: pathlib.Path, data: object) -> None:
    """Writes JSON through a temporary file so readers never see a partial."""
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _check_selection(selected: list[str], mask_subdir: str | None) -> None:
    """Rejects empty, duplicate or path-like scene names and unsafe masks."""
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Choose a nonempty, unique scene list")
    if any(
            pathlib.Path(scene).name != scene or scene in {".", ".."}
            for scene in selected):
        raise ValueError("Scene names must be single directory names")
    if mask_subdir and (pathlib.Path(mask_subdir).is_absolute() or
                        ".." in pathlib.Path(mask_subdir).parts):
        raise ValueError("mask_subdir must be relative to each source scene")


def _plan_links(
    records: dict[str, list[dict]],
    root: pathlib.Path,
    source_root: pathlib.Path,
) -> tuple[list[tuple[pathlib.Path, pathlib.Path]], dict[str, list[dict]]]:
    """Plans the symlinks and frame manifests without writing anything.

    Args:
        records: Scene name to its frame records.
        root: Output dataset folder.
        source_root: Source root, for the manifests' source paths.

    Returns:
        The (source, target) links to create and each scene's manifest.

    Raises:
        FileExistsError: A target exists and is not a link to the same source,
            or an existing frame manifest differs.
    """
    links = []
    manifests = {}
    for scene, frames in records.items():
        manifest = []
        for record in frames:
            entry = {
                key: record[key]
                for key in ("frame", "source_frame", "rgb_wh", "depth_wh")
            }
            for key, folder in (("rgb", "images"), ("depth", "depths"),
                                ("mask", "masks")):
                if key not in record:
                    continue
                source = record[key]
                target = root / scene / folder / (record["frame"] +
                                                  source.suffix)
                if target.exists() or target.is_symlink():
                    if (not target.is_symlink() or
                            target.resolve() != source.resolve()):
                        raise FileExistsError(
                            f"Refusing to replace existing data: {target}")
                links.append((source, target))
                entry[key] = target.relative_to(root / scene).as_posix()
                entry[f"source_{key}"] = source.relative_to(
                    source_root).as_posix()
            manifest.append(entry)
        manifest_path = root / scene / "frame_manifest.json"
        if manifest_path.exists() and json.loads(
                manifest_path.read_text()) != manifest:
            raise FileExistsError(
                f"Existing frame manifest differs: {manifest_path}")
        manifests[scene] = manifest
    return links, manifests


def materialize_dataset(
    source_root: pathlib.Path,
    benchmark_root: pathlib.Path,
    *,
    scenes: list[str] | None = None,
    mask_subdir: str | None = None,
) -> pathlib.Path:
    """Links DyCheck scenes into `<benchmark_root>/VideoDepth/dycheck`.

    All pairs are validated before anything is linked, and reruns cannot
    overwrite different data.

    Args:
        source_root: Extracted MegaSaM DyCheck release.
        benchmark_root: Benchmark root.
        scenes: Scenes to link; defaults to `SCENES`.
        mask_subdir: Optional camera-0 mask folder, relative to each scene.

    Returns:
        The dataset folder.

    Raises:
        ValueError: The scene selection or the source files are invalid.
        FileExistsError: Existing output differs from what would be written.
    """
    source_root = pathlib.Path(source_root).resolve()
    root = pathlib.Path(benchmark_root).resolve() / "VideoDepth/dycheck"
    selected = list(SCENES) if scenes is None else scenes
    _check_selection(selected, mask_subdir)
    records = {
        scene: _scene_records(source_root, scene, mask_subdir)
        for scene in selected
    }
    metadata_path = root / "metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        previous_scenes = {
            summary["scene_name"] for summary in previous["scenes"]
        }
        if previous_scenes != set(selected):
            raise FileExistsError("Existing dataset has a different scene "
                                  "selection; use a separate output root")
    links, manifests = _plan_links(records, root, source_root)
    for source, target in links:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_symlink():
            target.symlink_to(os.path.relpath(source, target.parent))
    summaries = []
    for scene, frames in records.items():
        _write_json(root / scene / "frame_manifest.json", manifests[scene])
        summaries.append({
            "scene_name": scene,
            "root_directory": scene,
            "total_images": len(frames),
            "gt_depth_scale": 1.0,
            "rgb_wh": frames[0]["rgb_wh"],
            "depth_wh": frames[0]["depth_wh"],
            "camera": 0,
            "pairing": "matching source stems, numeric frame order",
            "has_masks": bool(mask_subdir)
        })
    _write_json(
        metadata_path, {
            "dataset_name": "dycheck",
            "root_directory": ".",
            "total_images": sum(len(frames) for frames in records.values()),
            "scenes": summaries
        })
    return root


def main() -> None:
    """Runs the script with the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark_root", type=pathlib.Path, required=True)
    parser.add_argument("--scenes", nargs="+", choices=SCENES)
    parser.add_argument(
        "--mask_subdir",
        help="Optional camera-0 masks, relative to each source scene")
    args = parser.parse_args()
    print(
        materialize_dataset(args.source_root,
                            args.benchmark_root,
                            scenes=args.scenes,
                            mask_subdir=args.mask_subdir))


if __name__ == "__main__":
    main()
