#!/usr/bin/env python
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Generates full-standard benchmark case lists from canonical metadata."""

from __future__ import annotations

import argparse
import sys
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from spatialarena import slam_protocol  # noqa: E402
from spatialarena.eval import dataset_registry  # noqa: E402

# pylint: enable=wrong-import-position

DEPTH_TASKS = {"MonoDepth", "VideoDepth"}
SPARSE_TASKS = {"SfM", "WeakTextureSfM", "SLAM"}
DEFAULT_SPARSE_ORDER = ""


def parse_args() -> argparse.Namespace:
    """Parses the command line; `--slam_range` sets the tasks and datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark_root",
                        type=pathlib.Path,
                        default=dataset_registry.DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["MonoDepth", "VideoDepth", "MVRecon"],
        help="Tasks to include in the case list.",
    )
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument(
        "--include_mvrecon_depth_compat",
        action="store_true",
        help=("Include MVRecon scenes for the current depth-compatibility "
              "evaluator."),
    )
    parser.add_argument(
        "--exclude_scannet",
        action="store_true",
        help=("Exclude ScanNet/ScanNet-derived scenes from the generated "
              "case list."),
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Image interval. Full standard evaluation should keep this at 1.",
    )
    parser.add_argument(
        "--slam_range",
        choices=("short", "long", "large"),
        default=None,
        help=("Write a full-slam case list for this SLAM range track: its SLAM "
              "datasets only, each at its stride in the track "
              "(spatialarena/slam_protocol.py). Overrides --tasks, --datasets "
              "and --interval."),
    )
    args = parser.parse_args()
    if args.slam_range:
        args.tasks = ["SLAM"]
        args.datasets = slam_protocol.range_datasets(args.slam_range)
    elif "SLAM" in {
            dataset_registry.canonical_task_name(task) for task in args.tasks
    }:
        parser.error("SLAM case lists are per range track: use --slam_range "
                     "short|long|large")
    return args


def scene_is_evaluable(task: str, scene: dict,
                       include_mvrecon_depth_compat: bool) -> bool:
    """Returns whether a scene has the data its task's evaluator needs.

    Args:
        task: The task name (aliases are accepted).
        scene: One scene record of `describe_benchmark_scenes`.
        include_mvrecon_depth_compat: Whether MVRecon scenes without depth
            but with a sparse model count, for the depth-compatibility
            evaluator.

    Returns:
        True if the scene can be evaluated.
    """
    task = dataset_registry.canonical_task_name(task)
    if task == "MVRecon":
        if not scene["has_images"]:
            return False
        if not scene.get("has_gt_geometry", True):
            # MVRecon scores against a GT point cloud or mesh; e.g. ETH3D
            # meadow and terrace (not in DA3-BENCH) have none.
            return False
        if scene["has_depths"]:
            return True
        return include_mvrecon_depth_compat and bool(scene["has_sparse"])
    if task in DEPTH_TASKS:
        return bool(scene["has_images"] and scene["has_depths"])
    if task in SPARSE_TASKS:
        return bool(scene["has_images"] and scene["has_sparse"])
    return False


def interval_for_case(task: str,
                      dataset: str,
                      base_interval: int,
                      slam_range: str | None = None) -> int:
    """Returns a case's frame interval.

    Args:
        task: The task name (unused; kept for the call signature).
        dataset: The dataset name.
        base_interval: The interval from `--interval`.
        slam_range: The SLAM range track, if any; its per-dataset stride then
            replaces `base_interval`.

    Returns:
        The interval.
    """
    if slam_range:
        return slam_protocol.range_stride(slam_range, dataset)
    return base_interval


def orders_for_task(task: str) -> tuple[str, ...]:
    """Returns the image orders to write a case for (empty: the default)."""
    task = dataset_registry.canonical_task_name(task)
    if task in {"SfM", "WeakTextureSfM", "SLAM"}:
        return (DEFAULT_SPARSE_ORDER,)
    return ("",)


def is_scannet_record(dataset: str, scene: str) -> bool:
    """Returns whether a dataset or scene is ScanNet."""
    return dataset.lower() == "scannet" or scene.lower() == "scannet"


def _header_lines(slam_range: str | None) -> list[str]:
    """Returns the comment lines at the top of a case list."""
    if slam_range:
        interval_note = (
            f"# full-slam {slam_range}-range track: each interval is its "
            "dataset's temporal stride in the track; whole recordings, no "
            "frame cap (spatialarena/slam_protocol.py).")
    else:
        interval_note = ("# interval=1 preserves canonical ordering before any "
                         "task-specific frame cap.")
    return [
        "# Full-standard case list generated from canonical metadata.",
        "# Format: task|dataset|scene|interval|order",
        interval_note,
    ]


def main() -> None:
    """Writes the case list and prints the case and skip counts."""
    args = parse_args()
    records = dataset_registry.describe_benchmark_scenes(
        benchmark_root=args.benchmark_root,
        tasks=args.tasks,
        datasets=args.datasets,
    )
    lines = _header_lines(args.slam_range)
    skipped = []
    for record in records:
        task = dataset_registry.canonical_task_name(record["task"])
        dataset = record["dataset"]
        if record["metadata_required"] and not record["metadata_exists"]:
            skipped.append((task, dataset, "metadata missing"))
            continue
        for scene in record["scenes"]:
            if args.exclude_scannet and is_scannet_record(
                    dataset, scene["scene"]):
                skipped.append((task, dataset, scene["scene"]))
                continue
            if args.slam_range and scene.get("slam_range") not in (
                    None, args.slam_range):
                # A sequence that names its range track (SpatialArena's)
                # runs only in it.
                continue
            if scene_is_evaluable(task, scene,
                                  args.include_mvrecon_depth_compat):
                case_interval = interval_for_case(task, dataset, args.interval,
                                                  args.slam_range)
                case = f"{task}|{dataset}|{scene['scene']}|{case_interval}"
                for order_name in orders_for_task(task):
                    if order_name:
                        lines.append(f"{case}|{order_name}")
                    else:
                        lines.append(case)
            else:
                skipped.append((task, dataset, scene["scene"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Cases: {len(lines) - 3}")
    print(f"Skipped: {len(skipped)}")


if __name__ == "__main__":
    main()
