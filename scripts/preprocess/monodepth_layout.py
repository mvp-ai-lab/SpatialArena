# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Shared writer for the canonical MonoDepth layout.

The non-Marigold MonoDepth datasets are all materialized into this layout.
Layout of one dataset root (`<benchmark root>/MonoDepth/<dataset>`)::

    metadata.json                 dataset entry + one entry per scene,
                                  carrying the scene's depth limits
                                  (min_depth, max_depth, depth_scale,
                                  filter_gt_depth_range,
                                  alignment_max_resolution) that
                                  depth_prediction_evaluator.
                                  monodepth_scene_limits reads
    <scene>/images/<key>.<ext>    RGB as shipped by the source (jpg/png)
    <scene>/depths/<key>.npy      float32 depth in metres (or the source's
                                  unit for non-metric sets); 0 = invalid
    <scene>/masks/<key>.png       optional extra evaluation mask
                                  (255 = evaluate), only when the source
                                  has one
    <scene>/materialized_full.json  per-frame records (paths relative to the
                                  scene folder, pixel intrinsics, source)

Frame keys are the source sample path with "/" replaced by "__"; the committed
split file `spatialarena/eval/monodepth_splits/<dataset>/<scene>.txt` lists
them in canonical (sorted) order.
"""

from __future__ import annotations

import dataclasses
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_ROOT = REPO_ROOT / "spatialarena" / "eval" / "monodepth_splits"
MANIFEST_NAME = "materialized_full.json"


@dataclasses.dataclass(kw_only=True)
class FrameRecord:
    """One materialized frame.

    Attributes:
        key: Canonical frame key (see `frame_key`).
        image: Image path, relative to the scene folder.
        depth: Depth `.npy` path, relative to the scene folder.
        source: Source sample path the frame was read from.
        intrinsics: Optional 3x3 pixel intrinsics.
        mask: Optional evaluation mask path, relative to the scene folder.
        extra: Additional keys copied into the manifest record.
    """

    key: str
    image: str
    depth: str
    source: str
    intrinsics: list[list[float]] | None = None
    mask: str | None = None
    extra: dict = dataclasses.field(default_factory=dict)

    def as_dict(self) -> dict:
        """Returns the manifest record (optional fields only when set)."""
        record = {
            "key": self.key,
            "image": self.image,
            "depth": self.depth,
            "source": self.source
        }
        if self.intrinsics is not None:
            record["intrinsics"] = self.intrinsics
        if self.mask is not None:
            record["mask"] = self.mask
        record.update(self.extra)
        return record


@dataclasses.dataclass(kw_only=True)
class SceneLimits:
    """Standard-protocol depth limits of one scene.

    See `monodepth_scene_limits` in the depth prediction evaluator.

    Attributes:
        min_depth: Smallest evaluated GT depth.
        max_depth: Largest evaluated GT depth, or None for no limit.
        depth_scale: Factor from stored depth to metres.
        filter_gt_depth_range: Whether GT outside the range is dropped.
        alignment_max_resolution: Optional resolution cap for alignment.
    """

    min_depth: float
    max_depth: float | None
    depth_scale: float = 1.0
    filter_gt_depth_range: bool = True
    alignment_max_resolution: int | None = None

    def as_dict(self) -> dict:
        """Returns the limit keys as written to metadata.json and manifest."""
        return {
            "min_depth": self.min_depth,
            "max_depth": self.max_depth,
            "depth_scale": self.depth_scale,
            "filter_gt_depth_range": self.filter_gt_depth_range,
            "alignment_max_resolution": self.alignment_max_resolution,
        }


def frame_key(sample_path: str) -> str:
    """Returns the canonical frame key of a source sample path.

    For example `a/b/c.jpg` becomes `a__b__c`.
    """
    path = Path(sample_path)
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".npy", ".exr"}:
        path = path.with_suffix("")
    return path.as_posix().strip("/").replace("/", "__")


def decode_moge_depth(content: bytes) -> np.ndarray:
    """Decodes MoGe's 16-bit log-encoded depth PNG.

    Follows `read_depth` in moge/utils/io.py: 0 is unknown (NaN) and 65535 is
    infinity.
    """
    with Image.open(io.BytesIO(content)) as image:
        near = float(image.info["near"])
        far = float(image.info["far"])
        unit = float(image.info["unit"]) if "unit" in image.info else 1.0
        encoded = np.array(image)
    depth = (encoded.astype(np.float32) - 1.0) / 65533.0
    depth = near**(1.0 - depth) * far**depth * unit
    depth[encoded == 0] = np.nan
    depth[encoded == 65535] = np.inf
    return depth.astype(np.float32)


def drop_far_depth(depth: np.ndarray, factor: float = 1000.0) -> np.ndarray:
    """Returns the valid-depth mask under MoGe's far-depth rule.

    A pixel is valid when its depth is finite, > 0 and at most `factor` times
    the 1% depth quantile.
    """
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return valid
    limit = float(np.quantile(depth[valid], 0.01)) * factor
    return valid & (depth <= limit)


def normalized_to_pixel_intrinsics(intrinsics: np.ndarray, width: int,
                                   height: int) -> list[list[float]]:
    """Scales UV-normalized intrinsics (fx / W, cx / W, ...) to pixels.

    Args:
        intrinsics: 3x3 normalized intrinsics.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        The 3x3 pixel intrinsics as nested lists.
    """
    pixel = np.asarray(intrinsics, dtype=np.float64).copy()
    pixel[0, :] *= float(width)
    pixel[1, :] *= float(height)
    pixel[2, :] = (0.0, 0.0, 1.0)
    return pixel.tolist()


def save_depth(path: Path,
               depth: np.ndarray,
               valid: np.ndarray | None = None) -> None:
    """Writes float32 depth to .npy with invalid pixels set to 0.

    Invalid pixels are non-finite, <= 0, or outside `valid` when it is given.
    """
    depth = np.asarray(depth, dtype=np.float32)
    keep = np.isfinite(depth) & (depth > 0)
    if valid is not None:
        keep &= valid
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.where(keep, depth, 0.0).astype(np.float32))


def save_mask(path: Path, mask: np.ndarray) -> None:
    """Writes a boolean evaluation mask as an 8-bit PNG (255 = evaluate)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.asarray(mask, dtype=bool).astype(np.uint8) * 255
    Image.fromarray(pixels).save(path)


def write_bytes(path: Path, content: bytes) -> None:
    """Writes `content` to `path`, creating parent folders."""
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
    """Writes `<scene>/materialized_full.json`.

    Args:
        dataset_root: The dataset folder holding the scene folders.
        dataset: Dataset name.
        scene: Scene name.
        records: The scene's frames, in any order.
        limits: The scene's depth limits.
        provenance: Source description recorded in both outputs.

    Returns:
        The scene's metadata.json entry.

    Raises:
        ValueError: If `records` is empty.
    """
    if not records:
        raise ValueError(f"No frames materialized for {dataset}/{scene}")
    scene_root = dataset_root / scene
    records = sorted(records, key=lambda record: record.key)
    masks_dir = None
    if (scene_root / "masks").is_dir():
        masks_dir = "masks"
    manifest = {
        "dataset": dataset,
        "scene": scene,
        "num_samples": len(records),
        "images_dir": "images",
        "depths_dir": "depths",
        "masks_dir": masks_dir,
    }
    manifest.update(limits.as_dict())
    manifest["provenance"] = provenance
    manifest["records"] = [record.as_dict() for record in records]
    scene_root.mkdir(parents=True, exist_ok=True)
    (scene_root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) +
                                            "\n",
                                            encoding="utf-8")
    entry = {
        "dataset_name": dataset,
        "scene_name": scene,
        "root_directory": str(scene_root),
        "total_images": len(records),
        "total_depth_maps": len(records),
        "layout": "monodepth_materialized",
    }
    entry.update(limits.as_dict())
    entry["provenance"] = provenance
    return entry


def write_dataset_metadata(
    dataset_root: Path,
    dataset: str,
    scene_entries: list[dict],
    keep_scenes: tuple[str, ...] | None = None,
) -> dict:
    """Writes the dataset's `metadata.json`.

    Entries of scenes that were not re-materialized in this run are kept from
    the existing file (only those in `keep_scenes` when it is given).

    Args:
        dataset_root: The dataset folder.
        dataset: Dataset name.
        scene_entries: Entries of the scenes materialized in this run.
        keep_scenes: Scenes whose existing entries may be kept, or None for
            all of them.

    Returns:
        The written metadata.
    """
    metadata_path = dataset_root / "metadata.json"
    entries = {}
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        for entry in existing.get("scenes", []):
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
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n",
                             encoding="utf-8")
    return metadata


def split_path(dataset: str, scene: str) -> Path:
    """Returns the committed frame-list file of a scene."""
    return SPLIT_ROOT / dataset / f"{scene}.txt"


def read_split(dataset: str, scene: str) -> list[str] | None:
    """Returns the committed frame list of a scene, or None if there is none."""
    path = split_path(dataset, scene)
    if not path.is_file():
        return None
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_split(dataset: str, scene: str, sources: list[str]) -> Path:
    """Commits the source sample paths of a scene as its frame list.

    The file holds one path per line, sorted.
    """
    path = split_path(dataset, scene)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{source}\n" for source in sorted(sources)),
                    encoding="utf-8")
    return path


def fraction_count(total: int, fraction: float) -> int:
    """Returns how many of `total` items a `fraction` in (0, 1] keeps.

    The count is rounded and at least one for a non-empty list.

    Raises:
        ValueError: If `fraction` is outside (0, 1].
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"frame fraction must be in (0, 1], got {fraction}")
    if not total:
        return 0
    return min(total, max(1, round(total * fraction)))


def uniform_subsample(items: list[str], count: int) -> list[str]:
    """Returns `count` items evenly spaced over the sorted list.

    All items are returned when there are at most `count` of them.
    """
    items = sorted(items)
    if count <= 0 or len(items) <= count:
        return items
    positions = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[position] for position in sorted(set(positions.tolist()))]
