# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Shared writer for the canonical MonoDepth layout used by the non-Marigold MonoDepth datasets.

Layout of one dataset root (``<benchmark root>/MonoDepth/<dataset>``)::

    metadata.json                 dataset entry + one entry per scene, carrying the scene's depth limits
                                  (min_depth, max_depth, depth_scale, filter_gt_depth_range, alignment_max_resolution)
                                  that depth_prediction_evaluator.monodepth_scene_limits reads
    <scene>/images/<key>.<ext>    RGB as shipped by the source (jpg/png)
    <scene>/depths/<key>.npy      float32 depth in metres (or the source's unit for non-metric sets); 0 = invalid
    <scene>/masks/<key>.png       optional extra evaluation mask (255 = evaluate), only when the source has one
    <scene>/materialized_full.json  per-frame records (paths relative to the scene folder, pixel intrinsics, source)

Frame keys are the source sample path with "/" replaced by "__"; the committed split file
``spatialarena/eval/monodepth_splits/<dataset>/<scene>.txt`` lists them in canonical (sorted) order.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_ROOT = REPO_ROOT / "spatialarena" / "eval" / "monodepth_splits"
MANIFEST_NAME = "materialized_full.json"


@dataclass(kw_only=True)
class FrameRecord:
    """One materialized frame; paths are relative to the scene folder."""

    key: str
    image: str
    depth: str
    source: str
    intrinsics: list[list[float]] | None = None
    mask: str | None = None
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """The manifest record (optional fields only when set)."""
        record = {"key": self.key, "image": self.image, "depth": self.depth, "source": self.source}
        if self.intrinsics is not None:
            record["intrinsics"] = self.intrinsics
        if self.mask is not None:
            record["mask"] = self.mask
        record.update(self.extra)
        return record


@dataclass(kw_only=True)
class SceneLimits:
    """Standard-protocol depth limits of one scene (see monodepth_scene_limits)."""

    min_depth: float
    max_depth: float | None
    depth_scale: float = 1.0
    filter_gt_depth_range: bool = True
    alignment_max_resolution: int | None = None

    def as_dict(self) -> dict:
        """The limit keys as written to metadata.json and the manifest."""
        return {
            "min_depth": self.min_depth,
            "max_depth": self.max_depth,
            "depth_scale": self.depth_scale,
            "filter_gt_depth_range": self.filter_gt_depth_range,
            "alignment_max_resolution": self.alignment_max_resolution,
        }


def frame_key(sample_path: str) -> str:
    """Canonical frame key of a source sample path (``a/b/c.jpg`` -> ``a__b__c``)."""
    path = Path(sample_path)
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".npy", ".exr"}:
        path = path.with_suffix("")
    return path.as_posix().strip("/").replace("/", "__")


def decode_moge_depth(content: bytes) -> np.ndarray:
    """Decode MoGe's 16-bit log-encoded depth PNG (moge/utils/io.py read_depth): 0 = unknown (NaN), 65535 = inf."""
    with Image.open(io.BytesIO(content)) as image:
        near = float(image.info["near"])
        far = float(image.info["far"])
        unit = float(image.info["unit"]) if "unit" in image.info else 1.0
        encoded = np.array(image)
    depth = (encoded.astype(np.float32) - 1.0) / 65533.0
    depth = near ** (1.0 - depth) * far ** depth * unit
    depth[encoded == 0] = np.nan
    depth[encoded == 65535] = np.inf
    return depth.astype(np.float32)


def drop_far_depth(depth: np.ndarray, factor: float = 1000.0) -> np.ndarray:
    """Valid-depth mask with MoGe's far-depth rule: finite, > 0 and <= factor x the 1% depth quantile."""
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return valid
    limit = float(np.quantile(depth[valid], 0.01)) * factor
    return valid & (depth <= limit)


def normalized_to_pixel_intrinsics(intrinsics: np.ndarray, width: int, height: int) -> list[list[float]]:
    """Scale UV-normalized intrinsics (fx / W, cx / W, ...) to pixels for an image of ``width`` x ``height``."""
    pixel = np.asarray(intrinsics, dtype=np.float64).copy()
    pixel[0, :] *= float(width)
    pixel[1, :] *= float(height)
    pixel[2, :] = (0.0, 0.0, 1.0)
    return pixel.tolist()


def save_depth(path: Path, depth: np.ndarray, valid: np.ndarray | None = None) -> None:
    """Write float32 depth to .npy with invalid (non-finite, <= 0 or outside ``valid``) pixels set to 0."""
    depth = np.asarray(depth, dtype=np.float32)
    keep = np.isfinite(depth) & (depth > 0)
    if valid is not None:
        keep &= valid
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.where(keep, depth, 0.0).astype(np.float32))


def save_mask(path: Path, mask: np.ndarray) -> None:
    """Write a boolean evaluation mask as an 8-bit PNG (255 = evaluate)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=bool).astype(np.uint8) * 255).save(path)


def write_bytes(path: Path, content: bytes) -> None:
    """Write ``content`` to ``path``, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def write_scene(
    *,
    dataset_root: Path,
    dataset: str,
    scene: str,
    records: list[FrameRecord],
    limits: SceneLimits,
    provenance: dict,
) -> dict:
    """Write ``<scene>/materialized_full.json`` and return the scene's metadata.json entry."""
    if not records:
        raise ValueError(f"No frames materialized for {dataset}/{scene}")
    scene_root = dataset_root / scene
    records = sorted(records, key=lambda record: record.key)
    manifest = {
        "dataset": dataset,
        "scene": scene,
        "num_samples": len(records),
        "images_dir": "images",
        "depths_dir": "depths",
        "masks_dir": "masks" if (scene_root / "masks").is_dir() else None,
        **limits.as_dict(),
        "provenance": provenance,
        "records": [record.as_dict() for record in records],
    }
    scene_root.mkdir(parents=True, exist_ok=True)
    (scene_root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "dataset_name": dataset,
        "scene_name": scene,
        "root_directory": str(scene_root),
        "total_images": len(records),
        "total_depth_maps": len(records),
        "layout": "monodepth_materialized",
        **limits.as_dict(),
        "provenance": provenance,
    }


def write_dataset_metadata(
    dataset_root: Path,
    dataset: str,
    scene_entries: list[dict],
    keep_scenes: tuple[str, ...] | None = None,
) -> dict:
    """Write ``metadata.json``, keeping entries of scenes that were not re-materialized in this run (only those in
    ``keep_scenes`` when it is given)."""
    metadata_path = dataset_root / "metadata.json"
    entries = {}
    if metadata_path.is_file():
        for entry in json.loads(metadata_path.read_text(encoding="utf-8")).get("scenes", []):
            if keep_scenes is None or entry["scene_name"] in keep_scenes:
                entries[entry["scene_name"]] = entry
    for entry in scene_entries:
        entries[entry["scene_name"]] = entry
    scenes = [entries[name] for name in sorted(entries)]
    total = sum(int(entry["total_images"]) for entry in scenes)
    metadata = {
        "dataset_name": dataset,
        "task": "MonoDepth",
        "root_directory": str(dataset_root),
        "total_images": total,
        "average_images_per_scene": total / len(scenes) if scenes else 0.0,
        "scenes": scenes,
    }
    dataset_root.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def split_path(dataset: str, scene: str) -> Path:
    """Committed frame-list file of a scene."""
    return SPLIT_ROOT / dataset / f"{scene}.txt"


def read_split(dataset: str, scene: str) -> list[str] | None:
    """The committed frame list of a scene, or None when there is none yet."""
    path = split_path(dataset, scene)
    if not path.is_file():
        return None
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_split(dataset: str, scene: str, sources: list[str]) -> Path:
    """Commit the source sample paths of a scene (one per line, sorted) as its frame list."""
    path = split_path(dataset, scene)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{source}\n" for source in sorted(sources)), encoding="utf-8")
    return path


def fraction_count(total: int, fraction: float) -> int:
    """How many of ``total`` items a ``fraction`` in (0, 1] keeps: rounded, at least one of a non-empty list."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"frame fraction must be in (0, 1], got {fraction}")
    return min(total, max(1, round(total * fraction))) if total else 0


def uniform_subsample(items: list[str], count: int) -> list[str]:
    """``count`` items evenly spaced over the sorted list (all of them when there are fewer)."""
    items = sorted(items)
    if count <= 0 or len(items) <= count:
        return items
    positions = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[position] for position in sorted(set(positions.tolist()))]
