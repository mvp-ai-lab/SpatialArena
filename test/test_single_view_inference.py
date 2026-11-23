# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Single-view (MonoDepth) inference helpers: chunking, [B, 1] output folding, and the engine's cache key."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import scripts.inference.adapter_utils as adapter_utils
import scripts.inference.single_view_groups as single_view_groups

REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE = REPO_ROOT / "spatialarena" / "templates" / "_engine" / "run_method_case_list.sh"
PATHS = [f"/images/{index:02d}.png" for index in range(5)]


def test_batched_image_paths_returns_single_images_in_single_view_mode(
        monkeypatch):
    """Single-view mode makes every chunk one image unless the caller batches independent samples."""
    monkeypatch.delenv(adapter_utils.SINGLE_VIEW_ENV, raising=False)
    assert adapter_utils.batched_image_paths(
        PATHS, 2) == [PATHS[:2], PATHS[2:4], PATHS[4:]]
    monkeypatch.setenv(adapter_utils.SINGLE_VIEW_ENV, "1")
    assert adapter_utils.single_view_mode()
    assert adapter_utils.batched_image_paths(PATHS,
                                             2) == [[path] for path in PATHS]
    assert adapter_utils.batched_image_paths(
        PATHS, 2, independent_batch=True) == [PATHS[:2], PATHS[2:4], PATHS[4:]]
    monkeypatch.setenv(adapter_utils.SINGLE_VIEW_ENV, "0")
    assert not adapter_utils.single_view_mode()


def test_fold_single_view_outputs_matches_sequence_layout():
    """A [B, 1, ...] forward's outputs come back as [1, B, ...], recursively, leaving other values alone."""
    depth = torch.arange(3 * 1 * 4 * 5, dtype=torch.float32).reshape(3, 1, 4, 5)
    outputs = {
        "depth": depth,
        "pose_enc": [torch.zeros(3, 1, 9),
                     torch.ones(3, 1, 9)],
        "conf": np.zeros((3, 1, 4, 5)),
        "scale": torch.tensor(2.0),
        "name": "x",
    }
    folded = adapter_utils.fold_single_view_outputs(outputs, 3)
    assert folded["depth"].shape == (1, 3, 4, 5)
    assert torch.equal(folded["depth"][0, 2], depth[2, 0])
    assert [tensor.shape for tensor in folded["pose_enc"]] == [(1, 3, 9),
                                                               (1, 3, 9)]
    assert folded["conf"].shape == (1, 3, 4, 5)
    assert folded["scale"].item() == 2.0 and folded["name"] == "x"


class AttrDict(dict):
    """Stand-in for addict.Dict (DA3's output type)."""


def test_fold_single_view_outputs_keeps_dict_subclass():
    """DA3 returns an addict.Dict; folding keeps the type so attribute-style access still works downstream."""
    folded = adapter_utils.fold_single_view_outputs(
        AttrDict(depth=torch.zeros(2, 1, 3)), 2)
    assert isinstance(folded, AttrDict) and folded["depth"].shape == (1, 2, 3)


@pytest.mark.parametrize("layout", [(1, 4, 5, 1), (1, 1, 4, 5), (1, 4, 5)])
def test_normalize_confidence_keeps_the_frame_axis_for_one_image(layout):
    """One-image confidence in any layout becomes 1xHxW (a 1xHxWx1 map used to lose its frame axis)."""
    confidence = np.arange(20, dtype=np.float32).reshape(layout)
    normalized = adapter_utils.normalize_confidence(confidence, (1, 4, 5))
    assert normalized.shape == (1, 4, 5)
    np.testing.assert_array_equal(normalized.ravel(),
                                  np.arange(20, dtype=np.float32))


def engine_cache_key(*args: str) -> str:
    """Run the engine's prediction_group_key bash function in isolation."""
    source = ENGINE.read_text(encoding="utf-8")
    start = source.index("prediction_group_key() {")
    end = source.index("\nPY\n}\n",
                       start) + 6  # the function ends after its Python heredoc
    quoted = " ".join(f"'{arg}'" for arg in args)
    script = f'PYTHON=python3\n{source[start:end]}\nprediction_group_key {quoted}\n'
    return subprocess.run(["bash", "-c", script],
                          check=True,
                          capture_output=True,
                          text=True).stdout.strip()


def test_cache_key_changes_only_for_single_view_cases():
    """Non-single-view keys are unchanged (existing cached predictions stay valid); single-view gets its own key."""
    common = [
        "DA3-L", "rgb", "504", "504", "0", "100", "10000", "0.0", "0.0", "100",
        "1", "", "/data/s", "/c", "/k"
    ]
    payload = {
        "method": "DA3-L",
        "track": "rgb",
        "resolution": 504,
        "image_load_resolution": 504,
        "pixel_limit": 0,
        "max_images_per_run": 100,
        "max_points": 10000,
        "conf_thresh": 0.0,
        "depth_conf_thresh": 0.0,
        "max_input_frames": 100,
        "interval": 1,
        "order_name": "",
        "data_path": "/data/s",
        "code_path": "/c",
        "ckpt_path": "/k",
    }
    legacy = hashlib.sha1(
        json.dumps(payload, sort_keys=True,
                   separators=(",", ":")).encode()).hexdigest()[:20]
    assert engine_cache_key(*common) == legacy
    assert engine_cache_key(*common, "0") == legacy
    assert engine_cache_key(*common, "1") != legacy


def write_scene(root: Path, sizes: list[tuple[int, int]]) -> list[str]:
    """A scene with one blank image per size, named 00.png, 01.png, ...; returns their paths."""
    images = root / "images"
    images.mkdir(parents=True)
    for index, size in enumerate(sizes):
        Image.new("RGB", size).save(images / f"{index:02d}.png")
    return sorted(str(path) for path in images.glob("*.png"))


def test_plan_groups_selected_frames_by_size(tmp_path, monkeypatch):
    """A mixed-size scene gets one image list per size; a uniform scene gets none (it runs as before)."""
    monkeypatch.delenv("BENCHMARK_MAX_INPUT_FRAMES", raising=False)
    paths = write_scene(tmp_path / "mixed", [(8, 6), (6, 6), (8, 6), (6, 6),
                                             (4, 2)])
    groups_dir = tmp_path / "groups"
    assert single_view_groups.plan(tmp_path / "mixed", 1, 100, groups_dir) == 3
    lists = [
        path.read_text(encoding="utf-8").split()
        for path in sorted(groups_dir.glob("group_*.txt"))
    ]
    assert lists == [[paths[0], paths[2]], [paths[1], paths[3]], [paths[4]]]
    write_scene(tmp_path / "uniform", [(8, 6)] * 3)
    assert single_view_groups.plan(tmp_path / "uniform", 1, 100,
                                   tmp_path / "none") == 1
    assert not (tmp_path / "none").exists()


def test_plan_splits_a_large_scene_into_frame_budget_calls(
        tmp_path, monkeypatch):
    """Single-view selection is uncapped; each call gets at most max_images_per_run frames of one size."""
    monkeypatch.setenv("BENCHMARK_MAX_INPUT_FRAMES", "2")
    monkeypatch.setenv(adapter_utils.SINGLE_VIEW_ENV, "1")
    paths = write_scene(tmp_path / "large", [(8, 6)] * 5 + [(4, 2)])
    groups_dir = tmp_path / "groups"
    assert single_view_groups.plan(tmp_path / "large", 1, 2, groups_dir) == 4
    lists = [
        path.read_text(encoding="utf-8").split()
        for path in sorted(groups_dir.glob("group_*.txt"))
    ]
    assert lists == [paths[0:2], paths[2:4], [paths[4]], [paths[5]]]


def test_single_view_selection_ignores_the_frame_caps(monkeypatch):
    """MonoDepth predicts every frame: neither BENCHMARK_MAX_INPUT_FRAMES nor max_images_per_run drops any."""
    monkeypatch.setenv("BENCHMARK_MAX_INPUT_FRAMES", "2")
    monkeypatch.setenv(adapter_utils.SINGLE_VIEW_ENV, "1")
    assert adapter_utils.resolve_capped_image_indices(250, 1,
                                                      100) == list(range(250))
    assert adapter_utils.resolve_capped_image_indices(250, 2, 100) == list(
        range(0, 250, 2))
    monkeypatch.setenv(adapter_utils.SINGLE_VIEW_ENV, "0")
    assert len(adapter_utils.resolve_capped_image_indices(250, 1, 100)) == 2


def test_select_interval_restricts_to_the_image_subset(tmp_path, monkeypatch):
    """The subset applies after the frame cap, so every group sees the frames of the uncapped selection."""
    monkeypatch.delenv("BENCHMARK_MAX_INPUT_FRAMES", raising=False)
    monkeypatch.delenv("BENCHMARK_IMAGE_ORDER", raising=False)
    subset = tmp_path / "group_00.txt"
    subset.write_text(f"{PATHS[3]}\n{PATHS[1]}\n", encoding="utf-8")
    monkeypatch.setenv(adapter_utils.IMAGE_SUBSET_ENV, str(subset))
    assert adapter_utils.select_interval(PATHS, 1) == [PATHS[1], PATHS[3]]
    with pytest.raises(ValueError):
        adapter_utils.select_interval(
            PATHS, 2)  # 01 and 03 are dropped by the interval


def test_merge_moves_frames_and_combines_metrics(tmp_path):
    """Per-frame files land in one output, per-image transform records are united; times and frame counts add up,
    the peak memory is the largest."""
    groups_dir = tmp_path / "out" / "_size_groups"
    for index, (frames, seconds, memory) in enumerate([(2, 1.0, 100.0),
                                                       (1, 0.5, 300.0)]):
        group = groups_dir / f"group_{index:02d}"
        (group / "depths").mkdir(parents=True)
        (group / "raw_disparities").mkdir()
        for frame in range(frames):
            np.save(group / "depths" / f"g{index}_{frame}.npy",
                    np.zeros((2, 2), dtype=np.float32))
            np.save(group / "raw_disparities" / f"g{index}_{frame}.npy",
                    np.ones((2, 2), dtype=np.float32))
        group.with_suffix(".txt").write_text("".join(
            f"/images/g{index}_{frame}.png\n" for frame in range(frames)),
                                             encoding="utf-8")
        (group / "point_query_outputs.jsonl").write_text(
            f'{{"group": {index}}}\n', encoding="utf-8")
        (group / "spatial_transforms.json").write_text(json.dumps({
            f"g{index}_{frame}.png": {
                "source_wh": [8 - index, 6]
            } for frame in range(frames)
        }),
                                                       encoding="utf-8")
        (group / "metrics.json").write_text(json.dumps({
            "Preprocessed Images": frames,
            "inference_seconds": seconds,
            "inference time": seconds,
            "model_load_seconds": 10.0 + index,
            "peak_gpu_memory_mib": memory,
            "model_input_size": [8 - index, 6],
        }),
                                            encoding="utf-8")
    single_view_groups.merge(groups_dir, tmp_path / "out")
    assert sorted(
        path.name for path in (tmp_path / "out" / "depths").iterdir()) == [
            "g0_0.npy", "g0_1.npy", "g1_0.npy"
        ]
    assert sorted(path.name for path in (tmp_path / "out" /
                                         "raw_disparities").iterdir()) == [
                                             "g0_0.npy", "g0_1.npy", "g1_0.npy"
                                         ]
    assert (tmp_path / "out" / "point_query_outputs.jsonl").read_text(
        encoding="utf-8").count("group") == 2
    transforms = json.loads(
        (tmp_path / "out" /
         "spatial_transforms.json").read_text(encoding="utf-8"))
    assert transforms == {
        "g0_0.png": {
            "source_wh": [8, 6]
        },
        "g0_1.png": {
            "source_wh": [8, 6]
        },
        "g1_0.png": {
            "source_wh": [7, 6]
        }
    }
    metrics = json.loads(
        (tmp_path / "out" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["Preprocessed Images"] == 3 and metrics[
        "inference_seconds"] == 1.5
    assert metrics["model_load_seconds"] == 10.5 and metrics[
        "peak_gpu_memory_mib"] == 300.0
    assert metrics["model_input_size"] == [8, 6] and len(
        metrics["size_groups"]) == 2
    assert not groups_dir.exists()


def test_merge_combines_openrouter_logs_and_sums_api_counters(tmp_path):
    """The OpenRouter adapter's raw-answer log is concatenated and its per-group counters (cost, calls, tokens) add up
    instead of the scene reporting only its first group."""
    groups_dir = tmp_path / "out" / "_size_groups"
    for index, (cost, calls) in enumerate([(4.5, 99), (0.25, 5)]):
        group = groups_dir / f"group_{index:02d}"
        (group / "depths").mkdir(parents=True)
        np.save(group / "depths" / f"g{index}.npy",
                np.zeros((2, 2), dtype=np.float32))
        group.with_suffix(".txt").write_text(f"/images/g{index}.png\n",
                                             encoding="utf-8")
        (group / "openrouter_outputs.jsonl").write_text(
            f'{{"group": {index}}}\n', encoding="utf-8")
        (group / "metrics.json").write_text(json.dumps({
            "Preprocessed Images": 1,
            "api_cost_usd": cost,
            "api_calls": calls,
            "sky_points": index,
            "openrouter_model": "m",
        }),
                                            encoding="utf-8")
    single_view_groups.merge(groups_dir, tmp_path / "out")
    assert (tmp_path / "out" / "openrouter_outputs.jsonl").read_text(
        encoding="utf-8").count("group") == 2
    metrics = json.loads(
        (tmp_path / "out" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["api_cost_usd"] == 4.75 and metrics[
        "api_calls"] == 104 and metrics["sky_points"] == 1
    assert metrics["openrouter_model"] == "m"


def test_merge_refuses_a_group_that_ignored_its_list(tmp_path):
    """An adapter that predicted every frame in each group fails the merge, and nothing reaches the output."""
    groups_dir = tmp_path / "out" / "_size_groups"
    for index in range(2):
        group = groups_dir / f"group_{index:02d}"
        (group / "depths").mkdir(parents=True)
        for frame in range(2):
            np.save(group / "depths" / f"f{frame}.npy",
                    np.zeros((2, 2), dtype=np.float32))
        (group / "metrics.json").write_text("{}", encoding="utf-8")
        group.with_suffix(".txt").write_text(f"/images/f{index}.png\n",
                                             encoding="utf-8")
    with pytest.raises(ValueError):
        single_view_groups.merge(groups_dir, tmp_path / "out")
    assert not (tmp_path / "out" / "depths").exists()


def run_prediction_ready_check(scene: Path, output: Path) -> int:
    """Run the Python body of the engine's prediction_output_ready on a MonoDepth case; returns its exit code."""
    source = ENGINE.read_text(encoding="utf-8")
    start = source.index(
        "<<'PY'\n",
        source.index("prediction_output_ready() {")) + len("<<'PY'\n")
    body = source[start:source.index("\nPY\n", start)]
    args = [
        "MonoDepth", "toy", "scene",
        str(output), "true", "1", "inference", "0",
        str(scene), "1"
    ]
    # Fed on stdin like the engine's heredoc, so the repository root (cwd) is importable.
    result = subprocess.run([sys.executable, "-", *args],
                            input=body,
                            text=True,
                            cwd=REPO_ROOT,
                            capture_output=True,
                            check=False)
    assert "Traceback" not in result.stderr, result.stderr
    return result.returncode


def test_monodepth_prediction_is_ready_only_when_it_covers_every_frame(
        tmp_path):
    """A prediction under the former 100-frame cap (some frames missing) is not reused; a complete one is."""
    write_scene(tmp_path / "scene", [(8, 6)] * 3)
    depths = tmp_path / "out" / "depths"
    depths.mkdir(parents=True)
    for stem in ("00", "02"):
        np.save(depths / f"{stem}.npy", np.ones((6, 8), dtype=np.float32))
    assert run_prediction_ready_check(tmp_path / "scene", tmp_path / "out") == 1
    np.save(depths / "01.npy", np.ones((6, 8), dtype=np.float32))
    assert run_prediction_ready_check(tmp_path / "scene", tmp_path / "out") == 0
