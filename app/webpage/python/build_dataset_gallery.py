#!/usr/bin/env python3
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Build the Overview page's dataset gallery: one thumbnail per dataset plus ``dataset_gallery.json``.

Thumbnails are cropped from a real frame of each dataset under the benchmark root, so
they only need to be regenerated when the catalog or the chosen scenes change::

    python app/webpage/python/build_dataset_gallery.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.webpage.python.dataset_catalog import DATASET_CATALOG, DatasetCard  # noqa: E402
from spatialarena.eval.dataset_registry import (  # noqa: E402
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_DATASET_SPECS,
    DatasetSpec,
)

APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THUMBNAIL_DIR = APP_ROOT / "assets" / "datasets"
DEFAULT_OUTPUT = APP_ROOT / "data" / "dataset_gallery.json"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
NON_RGB_MARKERS = ("depth", "mask", "normal", "disp")
THUMBNAIL_SIZE = (640, 400)
TASK_ORDER = ("MonoDepth", "VideoDepth", "MVRecon", "SfM", "SLAM")


def tasks_for_dataset(dataset: str, specs: Sequence[DatasetSpec] = DEFAULT_DATASET_SPECS) -> list[str]:
    tasks = {spec.task for spec in specs if spec.dataset == dataset}
    return sorted(tasks, key=lambda task: (TASK_ORDER.index(task) if task in TASK_ORDER else len(TASK_ORDER), task))


def candidate_images(image_dir: Path, hint: str = "") -> list[Path]:
    """RGB frames under ``image_dir``, preferring names containing ``hint`` when any do."""
    if not image_dir.is_dir():
        return []
    images = sorted(
        path
        for path in image_dir.rglob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES
        and not any(marker in path.name.lower() for marker in NON_RGB_MARKERS)
    )
    if hint:
        hinted = [path for path in images if hint in path.name.lower()]
        if hinted:
            return hinted
    return images


def center_crop_thumbnail(source: Path, destination: Path, size: tuple[int, int] = THUMBNAIL_SIZE) -> None:
    """Center-crop ``source`` to the thumbnail aspect ratio, resize, and save as JPEG."""
    with Image.open(source) as image:
        image = image.convert("RGB")
        target_ratio = size[0] / size[1]
        width, height = image.size
        if width / height > target_ratio:
            new_width = round(height * target_ratio)
            left = (width - new_width) // 2
            image = image.crop((left, 0, left + new_width, height))
        else:
            new_height = round(width / target_ratio)
            top = (height - new_height) // 2
            image = image.crop((0, top, width, top + new_height))
        image = image.resize(size, Image.Resampling.LANCZOS)
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, format="JPEG", quality=82, optimize=True, progressive=True)


def pick_sample(card: DatasetCard, benchmark_root: Path) -> tuple[str, Path] | None:
    """Return ``(scene, frame_path)`` for the card's thumbnail, or None when no frame is found."""
    for spec in DEFAULT_DATASET_SPECS:
        if spec.dataset != card.key:
            continue
        try:
            scenes = spec.create_dataset(benchmark_root).list_scenes()
        except Exception:  # noqa: BLE001 -- a missing dataset only drops its thumbnail
            continue
        ordered = sorted(scenes, key=lambda scene: scene.scene != card.thumbnail_scene)
        for scene in ordered:
            images = candidate_images(Path(scene.image_path), card.thumbnail_image_hint)
            if images:
                return scene.scene, images[len(images) // 3]
    return None


def build_gallery(
    *,
    benchmark_root: Path = DEFAULT_BENCHMARK_ROOT,
    thumbnail_dir: Path = DEFAULT_THUMBNAIL_DIR,
    catalog: Sequence[DatasetCard] = DATASET_CATALOG,
    reuse_thumbnails: bool = False,
) -> dict:
    entries = []
    for card in catalog:
        thumbnail_path = thumbnail_dir / f"{card.key}.jpg"
        sample_scene = ""
        if not (reuse_thumbnails and thumbnail_path.is_file()):
            sample = pick_sample(card, benchmark_root)
            if sample is not None:
                sample_scene, frame = sample
                center_crop_thumbnail(frame, thumbnail_path)
        entry = card.as_dict()
        entry["tasks"] = tasks_for_dataset(card.key)
        entry["thumbnail"] = (
            thumbnail_path.relative_to(APP_ROOT).as_posix()
            if thumbnail_path.is_file() and thumbnail_path.is_relative_to(APP_ROOT)
            else (str(thumbnail_path) if thumbnail_path.is_file() else "")
        )
        entry["thumbnail_source_scene"] = sample_scene or card.thumbnail_scene
        entries.append(entry)
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "app/webpage/python/dataset_catalog.py",
        "datasets": entries,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--thumbnail-dir", type=Path, default=DEFAULT_THUMBNAIL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reuse-thumbnails",
        action="store_true",
        help="Keep existing thumbnails and only rewrite dataset_gallery.json (e.g. after editing intros).",
    )
    args = parser.parse_args(argv)
    gallery = build_gallery(
        benchmark_root=args.benchmark_root,
        thumbnail_dir=args.thumbnail_dir,
        reuse_thumbnails=args.reuse_thumbnails,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gallery, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    missing = [entry["key"] for entry in gallery["datasets"] if not entry["thumbnail"]]
    print(f"Wrote {args.output} ({len(gallery['datasets'])} datasets)")
    if missing:
        print(f"  no thumbnail found for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
