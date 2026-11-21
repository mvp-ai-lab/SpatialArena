#!/usr/bin/env python
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Generate full-standard benchmark case lists from canonical metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatialarena.eval.dataset_registry import (  # noqa: E402
    DEFAULT_BENCHMARK_ROOT,
    canonical_task_name,
    describe_benchmark_scenes,
)
from spatialarena.slam_protocol import range_datasets, range_stride  # noqa: E402

DEPTH_TASKS = {"MonoDepth", "VideoDepth"}
SPARSE_TASKS = {"SfM", "WeakTextureSfM", "SLAM"}
DEFAULT_SPARSE_ORDER = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark_root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output", type=Path, required=True)
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
        help="Include MVRecon scenes for the current depth-compatibility evaluator.",
    )
    parser.add_argument(
        "--exclude_scannet",
        action="store_true",
        help="Exclude ScanNet/ScanNet-derived scenes from the generated case list.",
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
        help="Write a full-slam case list for this SLAM range track: its SLAM datasets only, each at its stride in the "
        "track (spatialarena/slam_protocol.py). Overrides --tasks, --datasets and --interval.",
    )
    args = parser.parse_args()
    if args.slam_range:
        args.tasks, args.datasets = ["SLAM"], range_datasets(args.slam_range)
    elif "SLAM" in {canonical_task_name(task) for task in args.tasks}:
        parser.error("SLAM case lists are per range track: use --slam_range short|long|large")
    return args


def scene_is_evaluable(task: str, scene: dict, include_mvrecon_depth_compat: bool) -> bool:
    task = canonical_task_name(task)
    if task == "MVRecon":
        if not scene["has_images"]:
            return False
        if not scene.get("has_gt_geometry", True):
            # MVRecon scores against a GT point cloud / mesh: e.g. ETH3D meadow/terrace (not in DA3-BENCH) have none.
            return False
        if scene["has_depths"]:
            return True
        return include_mvrecon_depth_compat and bool(scene["has_sparse"])
    if task in DEPTH_TASKS:
        return bool(scene["has_images"] and scene["has_depths"])
    if task in SPARSE_TASKS:
        return bool(scene["has_images"] and scene["has_sparse"])
    return False


def interval_for_case(task: str, dataset: str, base_interval: int, slam_range: str | None = None) -> int:
    if slam_range:
        return range_stride(slam_range, dataset)
    return base_interval


def orders_for_task(task: str) -> tuple[str, ...]:
    task = canonical_task_name(task)
    if task in {"SfM", "WeakTextureSfM", "SLAM"}:
        return (DEFAULT_SPARSE_ORDER,)
    return ("",)


def is_scannet_record(dataset: str, scene: str) -> bool:
    return dataset.lower() == "scannet" or scene.lower() == "scannet"


def main() -> None:
    args = parse_args()
    records = describe_benchmark_scenes(
        benchmark_root=args.benchmark_root,
        tasks=args.tasks,
        datasets=args.datasets,
    )
    lines = [
        "# Full-standard case list generated from canonical metadata.",
        "# Format: task|dataset|scene|interval|order",
        (
            f"# full-slam {args.slam_range}-range track: each interval is its dataset's temporal stride in the track; "
            "whole recordings, no frame cap (spatialarena/slam_protocol.py)."
            if args.slam_range
            else "# interval=1 preserves canonical ordering before any task-specific frame cap."
        ),
    ]
    skipped = []
    for record in records:
        task = canonical_task_name(record["task"])
        dataset = record["dataset"]
        if record["metadata_required"] and not record["metadata_exists"]:
            skipped.append((task, dataset, "metadata missing"))
            continue
        for scene in record["scenes"]:
            if args.exclude_scannet and is_scannet_record(dataset, scene["scene"]):
                skipped.append((task, dataset, scene["scene"]))
                continue
            if args.slam_range and scene.get("slam_range") not in (None, args.slam_range):
                continue  # a sequence that names its range track (SpatialArena's) runs only in it
            if scene_is_evaluable(task, scene, args.include_mvrecon_depth_compat):
                case_interval = interval_for_case(task, dataset, args.interval, args.slam_range)
                for order_name in orders_for_task(task):
                    if order_name:
                        lines.append(f"{task}|{dataset}|{scene['scene']}|{case_interval}|{order_name}")
                    else:
                        lines.append(f"{task}|{dataset}|{scene['scene']}|{case_interval}")
            else:
                skipped.append((task, dataset, scene["scene"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Cases: {len(lines) - 3}")
    print(f"Skipped: {len(skipped)}")


if __name__ == "__main__":
    main()
