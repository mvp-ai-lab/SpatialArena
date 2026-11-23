#!/usr/bin/env python
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Verify standardized external-method outputs for benchmark scenes.

Checks, without running any model, that every selected method wrote the
prediction artifacts each selected scene needs, and prints (or writes) a JSON
report. With `--strict` the script exits with status 1 when any check fails.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from spatialarena.eval import dataset_registry  # noqa: E402
from spatialarena.eval import prediction_artifacts  # noqa: E402
from spatialarena.eval import result_layout  # noqa: E402
# pylint: enable=wrong-import-position


def parse_args() -> argparse.Namespace:
    """Parses the command-line arguments."""
    parser = argparse.ArgumentParser(
        description=
        "Verify benchmark prediction artifacts without running models.")
    parser.add_argument("--benchmark_root",
                        type=pathlib.Path,
                        default=dataset_registry.DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--experiment_root", type=pathlib.Path, required=True)
    parser.add_argument("--experiment_name", type=str, default="exp")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--scenes", nargs="*", default=[])
    parser.add_argument("--order_name", default="")
    parser.add_argument("--include_scannet", action="store_true")
    parser.add_argument("--require_sparse", action="store_true")
    parser.add_argument("--require_depths", action="store_true")
    parser.add_argument("--require_points", action="store_true")
    parser.add_argument("--require_metrics", action="store_true")
    parser.add_argument("--output_path", type=pathlib.Path, default=None)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def _requirements_override(
    args: argparse.Namespace,
) -> prediction_artifacts.PredictionRequirements | None:
    """Returns the requirements forced by `--require_*`, or None for none."""
    if not any((
            args.require_sparse,
            args.require_depths,
            args.require_points,
            args.require_metrics,
    )):
        return None
    return prediction_artifacts.PredictionRequirements(
        require_sparse=args.require_sparse,
        require_depths=args.require_depths,
        require_points=args.require_points,
        require_metrics=args.require_metrics,
    )


def main() -> None:
    """Runs the script with the command-line arguments."""
    args = parse_args()
    result_layout.validate_output_path_outside_benchmark_root(
        args.output_path,
        args.benchmark_root,
        label="--output_path",
    )
    requirements = _requirements_override(args)
    statuses = []
    warnings = []
    scene_filter = list(args.scenes)

    for spec in dataset_registry.select_dataset_specs(
            tasks=args.tasks,
            datasets=args.datasets,
            include_scannet=args.include_scannet,
    ):
        metadata_path = spec.data_path(args.benchmark_root) / "metadata.json"
        if spec.metadata_required and not metadata_path.is_file():
            warnings.append({
                "task": spec.task,
                "dataset": spec.dataset,
                "warning": f"metadata is missing: {metadata_path}",
            })
            continue

        dataset = spec.create_dataset(args.benchmark_root, scenes=scene_filter)
        for scene_info in dataset.list_scenes():
            for method in args.methods:
                statuses.append(
                    prediction_artifacts.validate_method_scene_output(
                        experiment_root=args.experiment_root,
                        experiment_name=args.experiment_name,
                        task=scene_info.task,
                        dataset=scene_info.dataset,
                        method=method,
                        scene=scene_info.scene,
                        order_name=args.order_name or None,
                        requirements=requirements,
                    ))

    failed = [status for status in statuses if not status.ok]
    result = {
        "checked": len(statuses),
        "failed": len(failed),
        "warnings": warnings,
        "statuses": [status.to_dict() for status in statuses],
    }
    text = json.dumps(result, indent=2)
    if args.output_path:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)

    if args.strict and failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
