#!/usr/bin/env python3
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Builds the Overview page's dataset gallery.

That is one thumbnail per dataset plus dataset_gallery.json. Thumbnails are
cropped from a real frame of each dataset under the benchmark root, so they
only need to be regenerated when the catalog or the chosen scenes change:

    python app/webpage/python/build_dataset_gallery.py
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys
from collections.abc import Sequence

from PIL import Image

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from app.webpage.python import dataset_catalog  # noqa: E402
from spatialarena.eval import dataset_registry  # noqa: E402
# pylint: enable=wrong-import-position

APP_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_THUMBNAIL_DIR = APP_ROOT / "assets" / "datasets"
DEFAULT_OUTPUT = APP_ROOT / "data" / "dataset_gallery.json"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
NON_RGB_MARKERS = ("depth", "mask", "normal", "disp")
THUMBNAIL_SIZE = (640, 400)
TASK_ORDER = ("MonoDepth", "VideoDepth", "MVRecon", "SfM", "SLAM")


def _task_sort_key(task: str) -> tuple[int, str]:
    """Sorts tasks in TASK_ORDER, then any other task by name."""
    if task in TASK_ORDER:
        return TASK_ORDER.index(task), task
    return len(TASK_ORDER), task


def tasks_for_dataset(
    dataset: str,
    specs: Sequence[
        dataset_registry.DatasetSpec] = dataset_registry.DEFAULT_DATASET_SPECS
) -> list[str]:
    """Returns the tasks a dataset is registered for, in TASK_ORDER."""
    tasks = {spec.task for spec in specs if spec.dataset == dataset}
    return sorted(tasks, key=_task_sort_key)


def candidate_images(image_dir: pathlib.Path,
                     hint: str = "") -> list[pathlib.Path]:
    """Returns the RGB frames under a directory, sorted.

    Args:
        image_dir: The directory to search recursively.
        hint: When some frame names contain it, only those frames are
            returned.

    Returns:
        The frames; [] when the directory does not exist.
    """
    if not image_dir.is_dir():
        return []
    images = sorted(
        path for path in image_dir.rglob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES and not any(
            marker in path.name.lower() for marker in NON_RGB_MARKERS))
    if hint:
        hinted = [path for path in images if hint in path.name.lower()]
        if hinted:
            return hinted
    return images


def center_crop_thumbnail(source: pathlib.Path,
                          destination: pathlib.Path,
                          size: tuple[int, int] = THUMBNAIL_SIZE) -> None:
    """Center-crops an image to the thumbnail aspect ratio and saves a JPEG.

    Args:
        source: The source image.
        destination: The JPEG to write; its directory is created.
        size: The thumbnail (width, height).
    """
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
        image.save(destination,
                   format="JPEG",
                   quality=82,
                   optimize=True,
                   progressive=True)


def pick_sample(
        card: dataset_catalog.DatasetCard,
        benchmark_root: pathlib.Path) -> tuple[str, pathlib.Path] | None:
    """Picks the frame for a card's thumbnail.

    The card's thumbnail scene is tried first, then the other scenes; the
    frame is a third of the way through the scene's frames.

    Args:
        card: The dataset card.
        benchmark_root: The benchmark dataset root.

    Returns:
        The (scene, frame path), or None when no frame is found.
    """
    for spec in dataset_registry.DEFAULT_DATASET_SPECS:
        if spec.dataset != card.key:
            continue
        try:
            scenes = spec.create_dataset(benchmark_root).list_scenes()
        # A missing dataset only drops its thumbnail.
        # pylint: disable-next=broad-exception-caught
        except Exception:  # noqa: BLE001
            continue
        ordered = sorted(scenes,
                         key=lambda scene: scene.scene != card.thumbnail_scene)
        for scene in ordered:
            images = candidate_images(pathlib.Path(scene.image_path),
                                      card.thumbnail_image_hint)
            if images:
                return scene.scene, images[len(images) // 3]
    return None


def _thumbnail_reference(thumbnail_path: pathlib.Path) -> str:
    """Returns how the gallery refers to a thumbnail.

    That is the path relative to the app root when the thumbnail is under
    it, the full path otherwise, and "" when the thumbnail does not exist.
    """
    if thumbnail_path.is_file() and thumbnail_path.is_relative_to(APP_ROOT):
        return thumbnail_path.relative_to(APP_ROOT).as_posix()
    if thumbnail_path.is_file():
        return str(thumbnail_path)
    return ""


def build_gallery(
    *,
    benchmark_root: pathlib.Path = dataset_registry.DEFAULT_BENCHMARK_ROOT,
    thumbnail_dir: pathlib.Path = DEFAULT_THUMBNAIL_DIR,
    catalog: Sequence[
        dataset_catalog.DatasetCard] = dataset_catalog.DATASET_CATALOG,
    reuse_thumbnails: bool = False,
) -> dict:
    """Writes the thumbnails and returns the gallery payload.

    Args:
        benchmark_root: The benchmark dataset root to crop frames from.
        thumbnail_dir: The directory the `<dataset>.jpg` thumbnails go to.
        catalog: The dataset cards.
        reuse_thumbnails: Whether to keep existing thumbnails.

    Returns:
        The dataset_gallery.json payload.
    """
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
        entry["thumbnail"] = _thumbnail_reference(thumbnail_path)
        entry["thumbnail_source_scene"] = sample_scene or card.thumbnail_scene
        entries.append(entry)
    generated_at = datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "generated_at": generated_at,
        "source": "app/webpage/python/dataset_catalog.py",
        "datasets": entries,
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Builds the gallery and writes dataset_gallery.json."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark-root",
                        type=pathlib.Path,
                        default=dataset_registry.DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--thumbnail-dir",
                        type=pathlib.Path,
                        default=DEFAULT_THUMBNAIL_DIR)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reuse-thumbnails",
        action="store_true",
        help=("Keep existing thumbnails and only rewrite dataset_gallery.json "
              "(e.g. after editing intros)."),
    )
    args = parser.parse_args(argv)
    gallery = build_gallery(
        benchmark_root=args.benchmark_root,
        thumbnail_dir=args.thumbnail_dir,
        reuse_thumbnails=args.reuse_thumbnails,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gallery, indent=2, ensure_ascii=False) +
                           "\n",
                           encoding="utf-8")
    missing = [
        entry["key"] for entry in gallery["datasets"] if not entry["thumbnail"]
    ]
    print(f"Wrote {args.output} ({len(gallery['datasets'])} datasets)")
    if missing:
        print(f"  no thumbnail found for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
