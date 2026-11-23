# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Benchmark entrypoint.

The default mode preserves the existing COLMAP reconstruction evaluation for
ETH3D, DTU, and Tanks-and-Temples. The lightweight listing mode is used to
validate the broader benchmark dataset layout without importing optional COLMAP
runtime dependencies.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pylint: disable-next=wrong-import-position
from spatialarena.eval import dataset_registry  # noqa: E402


def parse_list_scenes_args(argv: list[str]) -> argparse.Namespace:
    """Parses the arguments of the scene listing mode."""
    parser = argparse.ArgumentParser(
        description="List canonical benchmark scenes.")
    parser.add_argument(
        "--list_scenes",
        "--list-scenes",
        action="store_true",
        help="List metadata-backed benchmark scenes and exit.",
    )
    parser.add_argument(
        "--benchmark_root",
        type=pathlib.Path,
        default=dataset_registry.DEFAULT_BENCHMARK_ROOT,
        help=(
            "Root containing task folders such as MonoDepth, VideoDepth, SfM, "
            "WeakTextureSfM, and MVRecon."),
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=[],
        help=("Optional task filter, e.g. MonoDepth VideoDepth SfM "
              "WeakTextureSfM MVRecon."),
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Optional dataset filter, e.g. bonn kitti imc.",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=[],
        help="Optional scene filter applied within each selected dataset.",
    )
    parser.add_argument(
        "--include_scannet",
        action="store_true",
        help="Include ScanNet in listing mode. It is skipped by default.",
    )
    return parser.parse_args(argv)


def _is_list_scenes_request(argv: list[str]) -> bool:
    return "--list_scenes" in argv or "--list-scenes" in argv


def _run_list_scenes(argv: list[str]) -> None:
    args = parse_list_scenes_args(argv)
    records = dataset_registry.describe_benchmark_scenes(
        benchmark_root=args.benchmark_root,
        tasks=args.tasks,
        datasets=args.datasets,
        scenes=args.scenes,
        include_scannet=args.include_scannet,
    )
    print(json.dumps(records, indent=2))


def _run_reconstruction_eval() -> None:
    """Runs the COLMAP reconstruction evaluation of ETH3D, DTU and TnT."""
    # Imported here: the listing mode must not need pycolmap.
    import pycolmap

    from spatialarena.eval import dtu_dataset
    from spatialarena.eval import eth3d_dataset
    from spatialarena.eval import tnt_dataset
    from spatialarena.eval import utils

    args = utils.parse_args()

    datasets = {
        "eth3d": eth3d_dataset.DatasetETH3D,
        "dtu": dtu_dataset.DatasetDTU,
        "tnt": tnt_dataset.DatasetTanksAndTemples,
    }

    for dataset_name in args.datasets:
        if dataset_name not in datasets:
            pycolmap.logging.error(f"Unknown dataset: {dataset_name}")
            return

        pycolmap.logging.info(f"Evaluating dataset: {dataset_name}")

        for eval_method in args.eval_methods:
            pycolmap.logging.info(f"Using evaluation method: {eval_method}")

            dataset = datasets[dataset_name](
                data_path=args.data_path,
                categories=args.categories,
                scenes=args.scenes,
                method=eval_method,
                run_path=args.run_path,
                run_name=args.run_name,
            )

            scene_infos = dataset.list_scenes()

            if not scene_infos:
                pycolmap.logging.warning("No scenes found")
                return

            utils.process_scenes(
                args=args,
                scene_infos=scene_infos,
                prepare_scene=dataset.prepare_scene,
                position_accuracy_gt=dataset.position_accuracy_gt,
                rotation_accuracy_gt=dataset.rotation_accuracy_gt,
            )


def main(argv: list[str] | None = None) -> None:
    """Lists scenes with --list_scenes, else runs the COLMAP evaluation."""
    if argv is None:
        argv = sys.argv[1:]
    if _is_list_scenes_request(argv):
        _run_list_scenes(argv)
        return
    _run_reconstruction_eval()


if __name__ == "__main__":
    main()
