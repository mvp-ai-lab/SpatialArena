#!/usr/bin/env python
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Check real benchmark predictions for output-resolution and intrinsics consistency.

run_resolution_consistency_smoke.py drives the adapters on a synthetic 3-image scene; this script instead inspects
what a real run left in the prediction cache (``<cache_root>/<METHOD>/<key>/<dataset>/<scene>/``), against the
scene's real source images under ``<benchmark_root>/<Task>/<dataset>/<scene>/``:

  - every depth/confidence map has its source image's (height, width);
  - spatial_transforms.json records each image's true source size;
  - a COLMAP model under sparse/0 has, per image, its source image's width/height, a principal point near the image
    centre (an intrinsic left on the model grid puts it at the grid's centre instead), square-ish pixels (fx ~ fy),
    and, when the scene ships a ground-truth sparse/0, a focal length within a factor of two of the ground truth
    (a missed rescale is off by about model size / source size).

Usage:
    python scripts/test/check_prediction_resolution.py [--methods VGGT,PI3_X] [--datasets marigold,imc]
        [--since "2026-10-23 09:30"] [--output report.tsv]
"""

from __future__ import annotations

import os
import argparse
import csv
import datetime
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatialarena.geometry.read_write_model import read_model  # noqa: E402
from scripts.inference.adapter_utils import canonical_image_root, find_image_paths  # noqa: E402

# see spatialarena/paths.py
SPATIALARENA_ROOT = os.path.expanduser(os.environ.get("SPATIALARENA_ROOT", "~/spatialarena"))
DEFAULT_BENCHMARK_ROOT = Path(f"{SPATIALARENA_ROOT}/benchmarks")
TASKS = ("MonoDepth", "VideoDepth", "MVRecon", "SfM")
PRINCIPAL_POINT_TOLERANCE = 0.1  # fraction of the image size
ASPECT_TOLERANCE = 0.05
FOCAL_RATIO_RANGE = (0.5, 2.0)


def pinhole_params(camera) -> tuple[float, float, float, float]:
    """(fx, fy, cx, cy) of a COLMAP camera: the SIMPLE_*/RADIAL models share one focal, the others store two."""
    params = [float(value) for value in camera.params]
    if camera.model.startswith(("SIMPLE_", "RADIAL")):
        return params[0], params[0], params[1], params[2]
    return params[0], params[1], params[2], params[3]


def source_images(benchmark_root: Path, dataset: str, scene: str) -> dict[str, tuple[int, int]]:
    """{image stem: (width, height)} for the scene's source images, from the first task that has the scene."""
    for task in TASKS:
        scene_dir = benchmark_root / task / dataset / scene
        if scene_dir.is_dir():
            sizes = {}
            for path in find_image_paths(canonical_image_root(scene_dir)):
                with Image.open(path) as image:
                    sizes[Path(path).stem] = image.size
            if sizes:
                return sizes
    return {}


def ground_truth_cameras(benchmark_root: Path, dataset: str, scene: str) -> dict[str, tuple[float, float]]:
    """{image stem: (fx, fy)} from the scene's own sparse/0, when it has one."""
    for task in TASKS:
        sparse = benchmark_root / task / dataset / scene / "sparse" / "0"
        if sparse.is_dir():
            try:
                cameras, images, _ = read_model(str(sparse))
            except (OSError, ValueError, KeyError):
                return {}
            focal = {}
            for image in images.values():
                fx, fy, _, _ = pinhole_params(cameras[image.camera_id])
                focal[Path(image.name).stem] = (fx, fy)
            return focal
    return {}


def check_scene(scene_dir: Path, sizes: dict[str, tuple[int, int]], gt_focal: dict) -> list[str]:
    """Problems found in one cached scene prediction (empty when consistent)."""
    problems = []
    for folder in ("depths", "confidences"):
        for path in sorted(p for p in (scene_dir / folder).glob("*") if p.suffix in (".npz", ".npy")):
            expected = sizes.get(path.stem)
            if path.suffix == ".npz":  # the current export format: one compressed array per file
                with np.load(path) as data:
                    shape = data[data.files[0]].shape
            else:
                shape = np.load(path, mmap_mode="r").shape
            if expected is None:
                problems.append(f"{folder}/{path.name}: no source image with this name")
            elif tuple(shape) != (expected[1], expected[0]):
                problems.append(f"{folder}/{path.name}: shape {tuple(shape)} != source (H, W) {expected[::-1]}")
    records_path = scene_dir / "spatial_transforms.json"
    if records_path.is_file():
        for name, record in json.loads(records_path.read_text(encoding="utf-8")).items():
            expected = sizes.get(Path(name).stem)
            if expected is not None and tuple(record["source_wh"]) != expected:
                problems.append(f"spatial_transforms.json[{name}]: source_wh {record['source_wh']} != {expected}")
    sparse = scene_dir / "sparse" / "0"
    if sparse.is_dir():
        problems.extend(check_cameras(sparse, sizes, gt_focal))
    return problems


def check_cameras(sparse: Path, sizes: dict[str, tuple[int, int]], gt_focal: dict) -> list[str]:
    """Problems in an exported COLMAP model's per-image camera size and intrinsics."""
    try:
        cameras, images, _ = read_model(str(sparse))
    except (OSError, ValueError, KeyError) as exc:
        return [f"sparse/0: unreadable ({exc})"]
    problems = []
    for image in images.values():
        stem = Path(image.name).stem
        camera = cameras[image.camera_id]
        expected = sizes.get(stem)
        if expected is None:
            problems.append(f"sparse/0 {image.name}: no source image with this name")
            continue
        width, height = expected
        fx, fy, cx, cy = pinhole_params(camera)
        if (camera.width, camera.height) != expected:
            problems.append(f"sparse/0 {image.name}: camera {camera.width}x{camera.height} != source {width}x{height}")
        if abs(cx / width - 0.5) > PRINCIPAL_POINT_TOLERANCE or abs(cy / height - 0.5) > PRINCIPAL_POINT_TOLERANCE:
            problems.append(f"sparse/0 {image.name}: principal point ({cx:.1f}, {cy:.1f}) off-centre for "
                            f"{width}x{height}")
        if fy <= 0 or abs(fx / fy - 1) > ASPECT_TOLERANCE:
            problems.append(f"sparse/0 {image.name}: fx/fy = {fx:.1f}/{fy:.1f}")
        if stem in gt_focal:
            ratio = fx / gt_focal[stem][0]
            if not FOCAL_RATIO_RANGE[0] <= ratio <= FOCAL_RATIO_RANGE[1]:
                problems.append(f"sparse/0 {image.name}: fx {fx:.1f} is {ratio:.2f}x the ground truth")
    return problems


def scene_dirs(cache_root: Path, methods: set[str] | None, datasets: set[str] | None, since: float | None):
    """(method, dataset, scene, path) for each finished cached scene (has depths/ or sparse/0)."""
    for method_dir in sorted(cache_root.iterdir()):
        if not method_dir.is_dir() or (methods and method_dir.name not in methods):
            continue
        for dataset_dir in sorted(method_dir.glob("*/*")):
            if not dataset_dir.is_dir() or (datasets and dataset_dir.name not in datasets):
                continue
            for scene_dir in sorted(dataset_dir.iterdir()):
                ready = (scene_dir / "depths").is_dir() or (scene_dir / "sparse" / "0").is_dir()
                if not ready or (since is not None and scene_dir.stat().st_mtime < since):
                    continue
                yield method_dir.name, dataset_dir.name, scene_dir.name, scene_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark_root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--cache_root", type=Path, default=None,
                        help="default: <benchmark_root>/_evaluation/artifacts/_prediction_cache")
    parser.add_argument("--methods", default="", help="comma-separated; default: every cached method")
    parser.add_argument("--datasets", default="", help="comma-separated; default: every cached dataset")
    parser.add_argument("--since", default=None, help="only scenes written after this time (YYYY-MM-DD HH:MM)")
    parser.add_argument("--output", type=Path, default=None, help="write method/dataset/scene/problem rows as TSV")
    return parser.parse_args()


def main() -> int:
    """Check every selected cached scene; exit 1 if any is inconsistent."""
    args = parse_args()
    cache_root = args.cache_root or args.benchmark_root / "_evaluation" / "artifacts" / "_prediction_cache"
    methods = {m for m in args.methods.split(",") if m} or None
    datasets = {d for d in args.datasets.split(",") if d} or None
    since = datetime.datetime.strptime(args.since, "%Y-%m-%d %H:%M").timestamp() if args.since else None
    sources: dict[tuple[str, str], tuple[dict, dict]] = {}
    rows = []
    checked = 0
    for method, dataset, scene, scene_dir in scene_dirs(cache_root, methods, datasets, since):
        if (dataset, scene) not in sources:
            sources[dataset, scene] = (source_images(args.benchmark_root, dataset, scene),
                                       ground_truth_cameras(args.benchmark_root, dataset, scene))
        sizes, gt_focal = sources[dataset, scene]
        problems = check_scene(scene_dir, sizes, gt_focal) if sizes else ["no source images found"]
        checked += 1
        has_cameras = (scene_dir / "sparse" / "0").is_dir()
        print(f"{'OK  ' if not problems else 'FAIL'} {method} {dataset}/{scene}"
              f"{' (+cameras)' if has_cameras else ''}{'' if not problems else f': {len(problems)} problem(s)'}")
        for problem in problems[:5]:
            print(f"       {problem}")
        rows.extend((method, dataset, scene, problem) for problem in problems)
    if args.output:
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(("method", "dataset", "scene", "problem"))
            writer.writerows(rows)
    print(f"# {checked} scene(s) checked, {len({row[:3] for row in rows})} with problems")
    return 1 if rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
