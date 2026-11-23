#!/usr/bin/env python
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Checks real benchmark predictions for resolution and intrinsics consistency.

run_resolution_consistency_smoke.py drives the adapters on a synthetic 3-image
scene; this script instead inspects what a real run left in the prediction
cache (`<cache_root>/<METHOD>/<key>/<dataset>/<scene>/`), against the scene's
real source images under `<benchmark_root>/<Task>/<dataset>/<scene>/`:

  - every depth/confidence map has its source image's (height, width);
  - spatial_transforms.json records each image's true source size;
  - a COLMAP model under sparse/0 has, per image, its source image's
    width/height, a principal point near the image centre (an intrinsic left
    on the model grid puts it at the grid's centre instead), square-ish pixels
    (fx ~ fy), and, when the scene ships a ground-truth sparse/0, a focal
    length within a factor of two of the ground truth (a missed rescale is off
    by about model size / source size).

Usage:
    python scripts/test/check_prediction_resolution.py \\
        [--methods VGGT,PI3_X] [--datasets marigold,imc] \\
        [--since "2026-10-23 09:30"] [--output report.tsv]
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import pathlib
import sys

import numpy as np
from PIL import Image

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from scripts.inference import adapter_utils  # noqa: E402
from spatialarena.geometry import read_write_model  # noqa: E402

# pylint: enable=wrong-import-position

# See spatialarena/paths.py.
SPATIALARENA_ROOT = os.path.expanduser(
    os.environ.get("SPATIALARENA_ROOT", "~/spatialarena"))
DEFAULT_BENCHMARK_ROOT = pathlib.Path(f"{SPATIALARENA_ROOT}/benchmarks")
TASKS = ("MonoDepth", "VideoDepth", "MVRecon", "SfM")
# A fraction of the image size.
PRINCIPAL_POINT_TOLERANCE = 0.1
ASPECT_TOLERANCE = 0.05
FOCAL_RATIO_RANGE = (0.5, 2.0)


def pinhole_params(camera) -> tuple[float, float, float, float]:
    """Returns (fx, fy, cx, cy) of a COLMAP camera.

    The SIMPLE_* and RADIAL models share one focal length, the others store
    two.

    Args:
        camera: The COLMAP camera.

    Returns:
        The focal lengths and the principal point.
    """
    params = [float(value) for value in camera.params]
    if camera.model.startswith(("SIMPLE_", "RADIAL")):
        return params[0], params[0], params[1], params[2]
    return params[0], params[1], params[2], params[3]


def source_images(benchmark_root: pathlib.Path, dataset: str,
                  scene: str) -> dict[str, tuple[int, int]]:
    """Reads the sizes of a scene's source images.

    The images come from the first task that has the scene.

    Args:
        benchmark_root: The benchmark root.
        dataset: The dataset name.
        scene: The scene name.

    Returns:
        {image stem: (width, height)}, empty if no task has the scene.
    """
    for task in TASKS:
        scene_dir = benchmark_root / task / dataset / scene
        if scene_dir.is_dir():
            sizes = {}
            image_root = adapter_utils.canonical_image_root(scene_dir)
            for path in adapter_utils.find_image_paths(image_root):
                with Image.open(path) as image:
                    sizes[pathlib.Path(path).stem] = image.size
            if sizes:
                return sizes
    return {}


def ground_truth_cameras(benchmark_root: pathlib.Path, dataset: str,
                         scene: str) -> dict[str, tuple[float, float]]:
    """Reads the ground-truth focal lengths of a scene's images.

    Args:
        benchmark_root: The benchmark root.
        dataset: The dataset name.
        scene: The scene name.

    Returns:
        {image stem: (fx, fy)} from the scene's own sparse/0; empty if it has
        none or it cannot be read.
    """
    for task in TASKS:
        sparse = benchmark_root / task / dataset / scene / "sparse" / "0"
        if sparse.is_dir():
            try:
                cameras, images, _ = read_write_model.read_model(str(sparse))
            except (OSError, ValueError, KeyError):
                return {}
            focal = {}
            for image in images.values():
                fx, fy, _, _ = pinhole_params(cameras[image.camera_id])
                focal[pathlib.Path(image.name).stem] = (fx, fy)
            return focal
    return {}


def check_scene(scene_dir: pathlib.Path, sizes: dict[str, tuple[int, int]],
                gt_focal: dict) -> list[str]:
    """Checks one cached scene prediction.

    Args:
        scene_dir: The cached scene folder.
        sizes: The source image sizes of `source_images`.
        gt_focal: The ground-truth focal lengths of `ground_truth_cameras`.

    Returns:
        The problems found, empty when the prediction is consistent.
    """
    problems = []
    for folder in ("depths", "confidences"):
        map_paths = sorted(path for path in (scene_dir / folder).glob("*")
                           if path.suffix in (".npz", ".npy"))
        for path in map_paths:
            expected = sizes.get(path.stem)
            # .npz is the current export format: one compressed array per file.
            if path.suffix == ".npz":
                with np.load(path) as data:
                    shape = data[data.files[0]].shape
            else:
                shape = np.load(path, mmap_mode="r").shape
            if expected is None:
                problems.append(
                    f"{folder}/{path.name}: no source image with this name")
            elif tuple(shape) != (expected[1], expected[0]):
                problems.append(f"{folder}/{path.name}: shape {tuple(shape)} "
                                f"!= source (H, W) {expected[::-1]}")
    records_path = scene_dir / "spatial_transforms.json"
    if records_path.is_file():
        records = json.loads(records_path.read_text(encoding="utf-8"))
        for name, record in records.items():
            expected = sizes.get(pathlib.Path(name).stem)
            if expected is not None and tuple(record["source_wh"]) != expected:
                problems.append(f"spatial_transforms.json[{name}]: source_wh "
                                f"{record['source_wh']} != {expected}")
    sparse = scene_dir / "sparse" / "0"
    if sparse.is_dir():
        problems.extend(check_cameras(sparse, sizes, gt_focal))
    return problems


def check_cameras(sparse: pathlib.Path, sizes: dict[str, tuple[int, int]],
                  gt_focal: dict) -> list[str]:
    """Checks an exported COLMAP model's per-image camera size and intrinsics.

    Args:
        sparse: The model folder (sparse/0).
        sizes: The source image sizes of `source_images`.
        gt_focal: The ground-truth focal lengths of `ground_truth_cameras`.

    Returns:
        The problems found.
    """
    try:
        cameras, images, _ = read_write_model.read_model(str(sparse))
    except (OSError, ValueError, KeyError) as exc:
        return [f"sparse/0: unreadable ({exc})"]
    problems = []
    for image in images.values():
        stem = pathlib.Path(image.name).stem
        camera = cameras[image.camera_id]
        expected = sizes.get(stem)
        if expected is None:
            problems.append(
                f"sparse/0 {image.name}: no source image with this name")
            continue
        width, height = expected
        fx, fy, cx, cy = pinhole_params(camera)
        if (camera.width, camera.height) != expected:
            problems.append(
                f"sparse/0 {image.name}: camera "
                f"{camera.width}x{camera.height} != source {width}x{height}")
        if (abs(cx / width - 0.5) > PRINCIPAL_POINT_TOLERANCE or
                abs(cy / height - 0.5) > PRINCIPAL_POINT_TOLERANCE):
            problems.append(f"sparse/0 {image.name}: principal point "
                            f"({cx:.1f}, {cy:.1f}) off-centre for "
                            f"{width}x{height}")
        if fy <= 0 or abs(fx / fy - 1) > ASPECT_TOLERANCE:
            problems.append(f"sparse/0 {image.name}: fx/fy = {fx:.1f}/{fy:.1f}")
        if stem in gt_focal:
            ratio = fx / gt_focal[stem][0]
            if not FOCAL_RATIO_RANGE[0] <= ratio <= FOCAL_RATIO_RANGE[1]:
                problems.append(f"sparse/0 {image.name}: fx {fx:.1f} is "
                                f"{ratio:.2f}x the ground truth")
    return problems


def scene_dirs(cache_root: pathlib.Path, methods: set[str] | None,
               datasets: set[str] | None, since: float | None):
    """Yields each finished cached scene (it has depths/ or sparse/0).

    Args:
        cache_root: The prediction cache root.
        methods: The methods to keep; None keeps all.
        datasets: The datasets to keep; None keeps all.
        since: If set, only scenes modified after this timestamp.

    Yields:
        (method, dataset, scene, scene folder).
    """
    for method_dir in sorted(cache_root.iterdir()):
        if not method_dir.is_dir():
            continue
        if methods and method_dir.name not in methods:
            continue
        for dataset_dir in sorted(method_dir.glob("*/*")):
            if not dataset_dir.is_dir():
                continue
            if datasets and dataset_dir.name not in datasets:
                continue
            for scene_dir in sorted(dataset_dir.iterdir()):
                ready = (scene_dir / "depths").is_dir() or (
                    scene_dir / "sparse" / "0").is_dir()
                if not ready:
                    continue
                if since is not None and scene_dir.stat().st_mtime < since:
                    continue
                yield (method_dir.name, dataset_dir.name, scene_dir.name,
                       scene_dir)


def parse_args() -> argparse.Namespace:
    """Parses the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark_root",
                        type=pathlib.Path,
                        default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument(
        "--cache_root",
        type=pathlib.Path,
        default=None,
        help=("default: "
              "<benchmark_root>/_evaluation/artifacts/_prediction_cache"))
    parser.add_argument("--methods",
                        default="",
                        help="comma-separated; default: every cached method")
    parser.add_argument("--datasets",
                        default="",
                        help="comma-separated; default: every cached dataset")
    parser.add_argument(
        "--since",
        default=None,
        help="only scenes written after this time (YYYY-MM-DD HH:MM)")
    parser.add_argument("--output",
                        type=pathlib.Path,
                        default=None,
                        help="write method/dataset/scene/problem rows as TSV")
    return parser.parse_args()


def _status_line(method: str, dataset: str, scene: str, has_cameras: bool,
                 problems: list[str]) -> str:
    """Returns the OK/FAIL line printed for one scene."""
    if problems:
        status = "FAIL"
        summary = f": {len(problems)} problem(s)"
    else:
        status = "OK  "
        summary = ""
    if has_cameras:
        cameras = " (+cameras)"
    else:
        cameras = ""
    return f"{status} {method} {dataset}/{scene}{cameras}{summary}"


def main() -> int:
    """Checks every selected cached scene.

    Returns:
        The exit status: 1 if any scene is inconsistent, else 0.
    """
    args = parse_args()
    cache_root = (args.cache_root or args.benchmark_root / "_evaluation" /
                  "artifacts" / "_prediction_cache")
    methods = {method for method in args.methods.split(",") if method} or None
    datasets = {dataset for dataset in args.datasets.split(",") if dataset
               } or None
    if args.since:
        since = datetime.datetime.strptime(args.since,
                                           "%Y-%m-%d %H:%M").timestamp()
    else:
        since = None
    sources: dict[tuple[str, str], tuple[dict, dict]] = {}
    rows = []
    checked = 0
    for method, dataset, scene, scene_dir in scene_dirs(cache_root, methods,
                                                        datasets, since):
        if (dataset, scene) not in sources:
            sizes = source_images(args.benchmark_root, dataset, scene)
            focal = ground_truth_cameras(args.benchmark_root, dataset, scene)
            sources[dataset, scene] = (sizes, focal)
        sizes, gt_focal = sources[dataset, scene]
        if sizes:
            problems = check_scene(scene_dir, sizes, gt_focal)
        else:
            problems = ["no source images found"]
        checked += 1
        has_cameras = (scene_dir / "sparse" / "0").is_dir()
        print(_status_line(method, dataset, scene, has_cameras, problems))
        for problem in problems[:5]:
            print(f"       {problem}")
        rows.extend((method, dataset, scene, problem) for problem in problems)
    if args.output:
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(("method", "dataset", "scene", "problem"))
            writer.writerows(rows)
    print(f"# {checked} scene(s) checked, "
          f"{len({row[:3] for row in rows})} with problems")
    if rows:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
