#!/usr/bin/env python
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Checks one scene's prediction the way the SpatialArena evaluator reads it.

Run it on the --output folder of a submission's inference script (see
scripts/inference/submission_template.py and
docs/benchmark/submitting_a_model.md):

    python scripts/inference/check_submission_output.py \
        --images /path/to/scene/images --output /path/to/output \
        --task MVRecon

It reports, per task, whether the files the evaluator needs are there and
readable: a depth map per input image (depths/), readable as depth and
mostly valid, and a COLMAP model (sparse/0) whose image names are the input
images. It exits with 1 when a required check fails.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from spatialarena.eval import depth_prediction_evaluator
from spatialarena.eval import prediction_artifacts
from spatialarena.eval import result_layout
from spatialarena.geometry import read_write_model
# pylint: enable=wrong-import-position

TASKS = ("MonoDepth", "VideoDepth", "MVRecon", "SfM", "SLAM")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
# Below this fraction of finite, positive pixels a depth map is reported.
MIN_VALID_FRACTION = 0.5


def list_images(images_dir: pathlib.Path) -> list[pathlib.Path]:
    """Returns the images of a folder, sorted by name."""
    return sorted(path for path in images_dir.iterdir()
                  if path.suffix.lower() in IMAGE_SUFFIXES)


def check_depths(images: list[pathlib.Path],
                 output: pathlib.Path) -> tuple[list[str], list[str]]:
    """Checks the depth maps against the input images.

    Args:
        images: The input images.
        output: The prediction folder.

    Returns:
        The errors and the warnings.
    """
    errors = []
    warnings = []
    depth_files = {
        path.stem: path
        for path in prediction_artifacts.find_depth_files(output)
    }
    missing = [path.name for path in images if path.stem not in depth_files]
    if missing:
        errors.append(f"{len(missing)} of {len(images)} images have no depth "
                      f"map (named <image stem>.npz), e.g. {missing[0]}")
    low = []
    for path in images:
        depth_file = depth_files.get(path.stem)
        if depth_file is None:
            continue
        try:
            depth = depth_prediction_evaluator.load_depth_map(depth_file)
        except (ValueError, OSError) as error:
            errors.append(f"{depth_file.name} is not readable: {error}")
            continue
        if depth.ndim != 2:
            errors.append(f"{depth_file.name} has shape {depth.shape}, "
                          "not (H, W)")
            continue
        valid = np.isfinite(depth) & (depth > 0)
        if valid.mean() < MIN_VALID_FRACTION:
            low.append(depth_file.name)
    if low:
        warnings.append(f"{len(low)} depth maps have under "
                        f"{MIN_VALID_FRACTION:.0%} valid (finite, positive) "
                        f"pixels, e.g. {low[0]}")
    return errors, warnings


def check_cameras(images: list[pathlib.Path],
                  output: pathlib.Path) -> tuple[list[str], list[str]]:
    """Checks the COLMAP model against the input images.

    Args:
        images: The input images.
        output: The prediction folder.

    Returns:
        The errors and the warnings.
    """
    model_path = result_layout.resolve_sparse_model_path(output)
    if model_path is None:
        return ["no COLMAP model (cameras, images and points3D in sparse/0)"
               ], []
    model = read_write_model.read_model(str(model_path))
    if model is None:
        return [f"the COLMAP model in {model_path} is not readable"], []
    _cameras, colmap_images, _points = model
    names = {image.name for image in colmap_images.values()}
    input_names = {path.name for path in images}
    unknown = sorted(names - input_names)
    errors = []
    warnings = []
    if unknown:
        errors.append(f"{len(unknown)} COLMAP image names are not input file "
                      f"names, e.g. {unknown[0]}")
    missing = sorted(input_names - names)
    if missing:
        warnings.append(f"{len(missing)} of {len(images)} images have no "
                        f"camera (the pose metrics count them as missing), "
                        f"e.g. {missing[0]}")
    return errors, warnings


def main() -> int:
    """Checks a prediction folder and prints the result per task."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--images",
                        type=pathlib.Path,
                        required=True,
                        help="the scene's input images")
    parser.add_argument("--output",
                        type=pathlib.Path,
                        required=True,
                        help="the inference script's output folder")
    parser.add_argument("--task",
                        choices=TASKS,
                        action="append",
                        help="the task(s) to check for; all by default")
    args = parser.parse_args()

    images = list_images(args.images)
    if not images:
        print(f"no images in {args.images}")
        return 1
    depth_errors, depth_warnings = check_depths(images, args.output)
    camera_errors, camera_warnings = check_cameras(images, args.output)
    failed = False
    for task in args.task or TASKS:
        needs = prediction_artifacts.requirements_for_task(task)
        errors = []
        warnings = []
        if needs.require_depths:
            errors += depth_errors
            warnings += depth_warnings
        if needs.require_sparse:
            errors += camera_errors
            warnings += camera_warnings
        print(f"{task}: {'FAIL' if errors else 'ok'}")
        for message in errors:
            print(f"  error: {message}")
        for message in warnings:
            print(f"  warning: {message}")
        failed = failed or bool(errors)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
