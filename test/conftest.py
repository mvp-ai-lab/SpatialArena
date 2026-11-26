# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Shared pytest fixtures."""

from __future__ import annotations

import json
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# The datasets the planner tests select (test_evaluator.py, test_run_config.py).
STAND_IN_DATASETS = (
    ("MonoDepth", "marigold"),
    ("SfM", "eth3d"),
    ("SLAM", "7scenes"),
    ("SLAM", "tum_dynamics"),
)


def _case_list_scenes(task: str, dataset: str) -> list[str]:
    """Returns the scenes of `dataset` in the canonical case lists."""
    scenes = set()
    for case_file in REPO_ROOT.glob("scripts/test/full_standard_*cases*.txt"):
        for line in case_file.read_text().splitlines():
            fields = line.split("|")
            if len(fields) > 2 and fields[:2] == [task, dataset]:
                scenes.add(fields[2])
    return sorted(scenes)


@pytest.fixture
def benchmark_root(tmp_path: pathlib.Path) -> pathlib.Path:
    """Returns a stand-in benchmark root for planning and script generation.

    Validation, planning and script generation read only a dataset's
    metadata.json and its scene folders, never the frames, so each dataset in
    STAND_IN_DATASETS gets a metadata.json listing the scenes of the canonical
    case lists and an empty folder per scene. Tests that plan runs use it
    instead of the real benchmark data, which only the cluster has.
    """
    root = tmp_path / "benchmarks"
    for task, dataset in STAND_IN_DATASETS:
        dataset_root = root / task / dataset
        scenes = _case_list_scenes(task, dataset)
        for scene in scenes:
            (dataset_root / scene).mkdir(parents=True, exist_ok=True)
        dataset_root.mkdir(parents=True, exist_ok=True)
        metadata = {
            "dataset_name": dataset,
            "scenes": [{
                "scene_name": scene
            } for scene in scenes],
        }
        (dataset_root / "metadata.json").write_text(json.dumps(metadata))
    return root
