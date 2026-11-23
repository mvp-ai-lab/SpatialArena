# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import argparse
import json
import re
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import scripts.inference.openrouter_vlm_depth as adapter
from spatialarena.methods import resolve_method
from spatialarena.eval.depth_prediction_evaluator import load_depth_map


def test_parse_grid_answer_reads_json_object_in_answer_tags():
    text = 'thinking... {"1": 9} <answer>{"1": 1.5, "2": "2.0", "3": -1, "4": null}</answer>'
    depths = adapter.parse_grid_answer(text, 4)
    assert depths[:2] == [1.5, 2.0]
    assert math.isnan(depths[2]) and math.isnan(depths[3])


def test_parse_grid_answer_accepts_list_and_loose_pairs():
    assert adapter.parse_grid_answer("<answer>[1, 2, 3]</answer>",
                                     3) == [1.0, 2.0, 3.0]
    assert adapter.parse_grid_answer("1: 4.0, 2 = 5.5", 2) == [4.0, 5.5]
    assert all(
        math.isnan(d) for d in adapter.parse_grid_answer("I cannot tell.", 2))


def test_markers_stay_inside_the_image():
    image = Image.new("RGB", (320, 200), (90, 90, 90))
    points = adapter.grid_points(image.width, image.height, 8)
    marked = adapter.draw_numbered_markers(image, points)
    assert marked.size == image.size
    assert np.asarray(marked).max(
        axis=(0, 1))[0] == 255  # red marker pixels were drawn


@pytest.mark.parametrize(
    "name,slug",
    [
        ("CLAUDE_OPUS_5_5", "anthropic/claude-opus-5.5"),
        ("GPT_6_ASTRA", "openai/gpt-6-astra"),
        ("GEMINI_4_ARGON", "google/gemini-4-argon"),
    ],
)
def test_registry_entries_match_adapter_defaults(name, slug):
    spec = resolve_method(name)
    assert spec.adapter == "openrouter_vlm_depth.py"
    assert spec.ckpt_default == slug == adapter.OPENROUTER_MODELS[name]
    assert spec.tasks == ("MonoDepth",)


def _scene(tmp_path: Path, count: int = 3) -> Path:
    images = tmp_path / "scene" / "images"
    images.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for index in range(count):
        Image.fromarray(rng.integers(0, 255, (60, 90, 3), dtype=np.uint8)).save(
            images / f"{index:04d}.png")
    return tmp_path / "scene"


def _run(monkeypatch, tmp_path, scene, out_name, extra=()):
    calls = []

    def fake_post(payload, api_key, timeout):  # pylint: disable=unused-argument
        calls.append(payload)
        assert api_key == "test-key"
        assert payload["provider"] == {"data_collection": "deny"}
        content = payload["messages"][0]["content"]
        assert content[0]["image_url"]["url"].startswith(
            "data:image/jpeg;base64,")
        # Depth increases with the marker index, i.e. down the image rows.
        answer = {str(i): 1.0 + i * 0.1 for i in range(1, 5)}
        return {
            "id": "gen-1",
            "choices": [{
                "message": {
                    "content": f"<answer>{json.dumps(answer)}</answer>"
                }
            }],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 50,
                "cost": 0.01
            }
        }

    monkeypatch.setattr(adapter, "post_json", fake_post)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(sys, "argv", [
        "openrouter_vlm_depth.py",
        "--method_name",
        "GEMINI_4_ARGON",
        "--data_path",
        str(scene),
        "--output_path",
        str(tmp_path / out_name),
        "--export_depth",
        "--depth_only",
        "--point_grid_size",
        "2",
        "--cache_dir",
        str(tmp_path / "cache"),
        *extra,
    ])
    adapter.run()
    return calls


def test_run_exports_dense_depths_and_reuses_cache(monkeypatch, tmp_path):
    scene = _scene(tmp_path)
    calls = _run(monkeypatch, tmp_path, scene, "out1")
    assert len(calls) == 3 and calls[0]["model"] == "google/gemini-4-argon"
    depth = load_depth_map(tmp_path / "out1" / "depths" / "0000.npz")
    assert depth.shape == (60, 90)
    assert depth[0].mean() < depth[-1].mean(
    )  # top-row markers (1, 2) are nearer than bottom-row (3, 4)
    metrics = json.loads((tmp_path / "out1" / "metrics.json").read_text())
    blob = json.dumps(metrics)
    assert '"api_calls": 3' in blob and '"parse_failures": 0' in blob and "test-key" not in blob

    calls = _run(monkeypatch, tmp_path, scene, "out2")
    assert calls == []  # every image answered from the cache
    np.testing.assert_array_equal(
        load_depth_map(tmp_path / "out2" / "depths" / "0000.npz"), depth)


def test_budget_cap_stops_the_run(monkeypatch, tmp_path):
    scene = _scene(tmp_path, count=4)
    with pytest.raises(adapter.BudgetExceeded):
        _run(monkeypatch,
             tmp_path,
             scene,
             "out",
             extra=("--max_cost_usd", "0.015", "--concurrency", "1"))


def test_scene_depth_limits_reads_the_dataset_metadata_entry(tmp_path):
    scene = tmp_path / "dataset" / "sceneA"
    scene.mkdir(parents=True)
    (tmp_path / "dataset" / "metadata.json").write_text(
        json.dumps({
            "scenes": [
                {
                    "scene_name": "sceneB",
                    "min_depth": 0.5,
                    "max_depth": 5.0
                },
                {
                    "scene_name": "sceneA",
                    "min_depth": 3.0,
                    "max_depth": 150.0
                },
            ]
        }))
    assert adapter.scene_depth_limits(scene) == (3.0, 150.0)
    assert adapter.scene_depth_limits(tmp_path / "elsewhere") == (None, None)


def test_clamp_samples_bounds_outliers_and_keeps_missing():
    samples, moved = adapter.clamp_samples([(0, 0, 2664.0), (1, 0, 1.0),
                                            (0, 1, 40.0), (1, 1, float("nan"))],
                                           3.0, 150.0)
    assert [d for _x, _y, d in samples[:3]] == [150.0, 3.0, 40.0
                                               ] and math.isnan(samples[3][2])
    assert moved == 2
    unchanged, moved = adapter.clamp_samples([(0, 0, 2664.0)], None, None)
    assert unchanged == [(0, 0, 2664.0)] and moved == 0


def test_run_clamps_answers_to_the_scene_range(monkeypatch, tmp_path):
    scene = _scene(tmp_path, count=1)
    (scene / "metadata.json").write_text(
        json.dumps({
            "min_depth": 0.1,
            "max_depth": 1.25
        }))
    _run(monkeypatch, tmp_path, scene, "out")  # fake answers are 1.1 .. 1.4 m
    depth = load_depth_map(tmp_path / "out" / "depths" / "0000.npz")
    assert depth.max() <= 1.25 + 1e-6
    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert '"clamped_points": 2' in json.dumps(metrics)


def _two_tone(width=96, height=64):
    """Left half dark and near (1 m), right half bright and far (4 m)."""
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[:, width // 2:] = 230
    return Image.fromarray(array)


def _edge_samples(image, grid):
    return [(x, y, 1.0 if x < image.width // 2 else 4.0)
            for x, y in adapter.grid_points(image.width, image.height, grid)]


def test_joint_bilateral_keeps_a_colour_edge_sharp():
    image = _two_tone()
    samples = _edge_samples(image, 4)
    joint = adapter.joint_bilateral_depth(samples, image, 4)
    bilinear = adapter.interpolate_sparse_depth(samples, image.width,
                                                image.height, 4)
    # Just left / right of the edge: joint bilateral stays near the true depths, bilinear blends them.
    left, right = image.width // 2 - 2, image.width // 2 + 1
    assert abs(joint[:, left].mean() -
               1.0) < 0.1 and abs(joint[:, right].mean() - 4.0) < 0.3
    assert abs(bilinear[:, left].mean() - 1.0) > abs(joint[:, left].mean() -
                                                     1.0)


def test_refinement_points_straddle_the_depth_edge():
    image = _two_tone()
    samples = _edge_samples(image, 4)
    points = adapter.refinement_points(samples, image, 4, max_points=8)
    assert points and len(points) <= 8
    edge = image.width // 2
    assert any(x < edge for x, _y in points) and any(
        x >= edge for x, _y in points)
    assert all(abs(x - edge) < image.width // 4 for x, _y in points)
    assert adapter.refinement_points([(x, y, 2.0) for x, y, _d in samples],
                                     image, 4) == []  # no edge, no points


def test_grid_v2_sends_clean_and_marked_images_and_accepts_sky():
    args = argparse.Namespace(max_tokens=10,
                              reasoning_effort="low",
                              allow_data_collection=False)
    payload = adapter.build_payload("m", ["data:a", "data:b"],
                                    adapter.build_prompt(4, "grid-v2"), args)
    content = payload["messages"][0]["content"]
    assert [part["type"] for part in content
           ] == ["image_url", "image_url", "text"]
    assert '"sky"' in content[-1]["text"]
    text = '<answer>{"1": 2.0, "2": "sky", "3": "SKY", "4": 5}</answer>'
    depths = adapter.parse_grid_answer(text, 4)
    assert depths[0] == 2.0 and depths[3] == 5.0 and math.isnan(
        depths[1]) and adapter.count_sky_answers(text) == 2


def test_two_pass_refinement_run(monkeypatch, tmp_path):
    images = tmp_path / "scene" / "images"
    images.mkdir(parents=True)
    _two_tone(160, 120).save(images / "0000.png")
    calls = []

    def fake_post(payload, api_key, timeout):  # pylint: disable=unused-argument
        calls.append(payload)
        text = payload["messages"][-1]["content"][-1]["text"]
        count = int(re.search(r"(\d+) (?:new )?red dots", text).group(1))
        answer = {str(i): (1.0 if i % 2 else 4.0) for i in range(1, count + 1)}
        return {
            "choices": [{
                "message": {
                    "content": f"<answer>{json.dumps(answer)}</answer>"
                }
            }],
            "usage": {
                "cost": 0.01
            }
        }

    monkeypatch.setattr(adapter, "post_json", fake_post)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(sys, "argv", [
        "openrouter_vlm_depth.py",
        "--method_name",
        "GEMINI_4_ARGON",
        "--data_path",
        str(tmp_path / "scene"),
        "--output_path",
        str(tmp_path / "out"),
        "--export_depth",
        "--depth_only",
        "--point_grid_size",
        "4",
        "--cache_dir",
        str(tmp_path / "cache"),
        "--prompt_version",
        "grid-v2",
        "--interpolation",
        "joint_bilateral",
        "--refine_points",
        "8",
    ])
    adapter.run()
    assert len(calls) == 2
    second = calls[1]["messages"]
    assert [m["role"] for m in second
           ] == ["user", "assistant",
                 "user"]  # pass one is the conversation history
    assert len(second[0]["content"]
              ) == 3 and "new red dots" in second[-1]["content"][-1]["text"]
    metrics = json.dumps(
        json.loads((tmp_path / "out" / "metrics.json").read_text()))
    assert '"api_calls": 2' in metrics and '"interpolation": "joint_bilateral"' in metrics
    assert load_depth_map(tmp_path / "out" / "depths" /
                          "0000.npz").shape == (120, 160)


def test_system_prompt_is_optional_and_keyed_separately():
    base = dict(max_tokens=10,
                reasoning_effort="low",
                allow_data_collection=False,
                point_grid_size=8,
                image_load_resolution=1024)
    plain = argparse.Namespace(**base, system_prompt="none")
    expert = argparse.Namespace(**base, system_prompt="depth-expert")
    assert adapter.build_payload("m", "data:a", "q",
                                 plain)["messages"][0]["role"] == "user"
    messages = adapter.build_payload("m", "data:a", "q", expert)["messages"]
    assert messages[0]["role"] == "system" and "metric depth" in messages[0][
        "content"]
    # "none" keeps the pre-existing cache key (no system_prompt field); the expert prompt gets its own.
    legacy = argparse.Namespace(**base)
    assert adapter.cache_key("m", plain,
                             "h") == adapter.cache_key("m", legacy, "h")
    assert adapter.cache_key("m", expert,
                             "h") != adapter.cache_key("m", plain, "h")


def test_missing_grid_cells_are_filled_from_neighbours_not_the_median():
    grid = np.array([[1.0, 1.0, 9.0], [1.0, np.nan, 9.0], [np.nan, np.nan,
                                                           9.0]])
    filled = adapter.fill_missing_grid(grid)
    assert np.isfinite(filled).all()
    np.testing.assert_allclose(filled[[0, 0, 0, 1, 1], [0, 1, 2, 0, 2]],
                               [1, 1, 9, 1, 9])  # answers unchanged
    assert filled[
        2,
        0] < 3.0  # bottom-left gap borders only 1 m answers, so it stays near 1 m
    with pytest.raises(RuntimeError):
        adapter.fill_missing_grid(np.full((2, 2), np.nan))


def test_bilinear_densify_has_no_flat_block_at_a_sky_gap():
    image = Image.new("RGB", (80, 80))
    points = adapter.grid_points(80, 80, 4)
    depths = [2.0 + 0.5 * index for index in range(16)]
    depths[5] = float("nan")  # one "sky" answer inside the grid
    dense = adapter.densify([(x, y, d) for (x, y), d in zip(points, depths)],
                            image, 4, "bilinear")
    gap_x, gap_y = points[5]
    patch = dense[gap_y - 4:gap_y + 5, gap_x - 4:gap_x + 5]
    assert patch.std() > 1e-3  # varies smoothly instead of a constant block
    assert 2.0 < dense[gap_y, gap_x] < 9.5


ENGINE = Path(__file__).resolve().parents[
    1] / "spatialarena" / "templates" / "_engine" / "run_method_case_list.sh"


def _engine_cache_key(method: str, env: dict[str, str]) -> str:
    """The engine's prediction_group_key bash function, run in isolation with ``env``."""
    import os  # pylint: disable=import-outside-toplevel
    import subprocess  # pylint: disable=import-outside-toplevel

    source = ENGINE.read_text(encoding="utf-8")
    start = source.index("prediction_group_key() {")
    end = source.index("\nPY\n}\n", start) + 6
    args = [
        method, "rgb", "1024", "1024", "0", "100", "0", "0.0", "0.0", "0", "1",
        "", "/data/s", "", "slug", "1"
    ]
    script = f"PYTHON={sys.executable}\n{source[start:end]}\nprediction_group_key " + " ".join(
        f"'{a}'" for a in args)
    clean = {
        k: v for k, v in os.environ.items() if not k.startswith("OPENROUTER_")
    }
    return subprocess.run(["bash", "-c", script],
                          check=True,
                          capture_output=True,
                          text=True,
                          env={
                              **clean,
                              **env
                          }).stdout.strip()


def test_engine_cache_key_separates_openrouter_settings():
    grid8 = _engine_cache_key("GEMINI_4_ARGON",
                              {"OPENROUTER_DEPTH_GRID_SIZE": "8"})
    grid16 = _engine_cache_key("GEMINI_4_ARGON",
                               {"OPENROUTER_DEPTH_GRID_SIZE": "16"})
    v2 = _engine_cache_key("GEMINI_4_ARGON", {
        "OPENROUTER_DEPTH_GRID_SIZE": "16",
        "OPENROUTER_PROMPT_VERSION": "grid-v2"
    })
    assert len({grid8, grid16, v2}) == 3
    # Other methods ignore the OpenRouter variables, so their cached predictions keep their keys.
    assert _engine_cache_key(
        "DA3-L",
        {"OPENROUTER_DEPTH_GRID_SIZE": "8"}) == _engine_cache_key("DA3-L", {})


def test_format_dot_ranges():
    assert adapter.format_dot_ranges([9, 1, 2, 3, 7, 10]) == "1-3, 7, 9-10"
    assert adapter.format_dot_ranges([5]) == "5"


def test_incomplete_answer_gets_a_completion_followup(monkeypatch, tmp_path):
    """An answer that stops partway is asked, in the same conversation, for exactly the missing dots (not the dots
    answered "sky"); the follow-up's depths fill the gaps and the exchange is recorded."""
    scene = _scene(tmp_path, count=1)
    calls = []

    def fake_post(payload, api_key, timeout):  # pylint: disable=unused-argument
        calls.append(payload)
        if len(calls) == 1:  # stops after dot 1, dot 2 is sky, 3 and 4 missing
            text = '<answer>{"1": 1.5, "2": "sky"}</answer> Only dots 1-2 are covered.'
        else:
            text = '<answer>{"3": 2.5, "4": 3.5}</answer>'
        return {
            "choices": [{
                "message": {
                    "content": text
                }
            }],
            "usage": {
                "cost": 0.01
            }
        }

    monkeypatch.setattr(adapter, "post_json", fake_post)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(sys, "argv", [
        "openrouter_vlm_depth.py",
        "--method_name",
        "CLAUDE_OPUS_5_5",
        "--data_path",
        str(scene),
        "--output_path",
        str(tmp_path / "out"),
        "--export_depth",
        "--depth_only",
        "--point_grid_size",
        "2",
        "--cache_dir",
        str(tmp_path / "cache"),
        "--prompt_version",
        "grid-v2",
    ])
    adapter.run()
    assert len(calls) == 2
    roles = [m["role"] for m in calls[1]["messages"]]
    assert roles == ["user", "assistant", "user"]
    follow_text = calls[1]["messages"][-1]["content"][-1]["text"]
    assert "dots 3-4" in follow_text and "image_url" not in json.dumps(
        calls[1]["messages"][-1])
    record = json.loads(
        (tmp_path / "out" /
         "openrouter_outputs.jsonl").read_text().splitlines()[0])
    assert record["depths"] == [1.5, None, 2.5, 3.5
                               ] and record["sky_points"] == 1
    assert record["followups"][0]["filled"] == 2
    metrics = json.dumps(
        json.loads((tmp_path / "out" / "metrics.json").read_text()))
    assert '"followup_calls": 1' in metrics and '"followup_filled_points": 2' in metrics
    # A rerun replays both calls from the cache.
    calls.clear()
    argv = list(sys.argv)
    argv[argv.index("--output_path") + 1] = str(tmp_path / "out2")
    monkeypatch.setattr(sys, "argv", argv)
    adapter.run()
    assert calls == []


def test_followups_stop_when_the_model_adds_nothing(monkeypatch, tmp_path):
    scene = _scene(tmp_path, count=1)
    calls = []

    def fake_post(payload, api_key, timeout):  # pylint: disable=unused-argument
        calls.append(payload)
        text = '<answer>{"1": 1.5}</answer>' if len(
            calls) == 1 else "I cannot see those dots."
        return {
            "choices": [{
                "message": {
                    "content": text
                }
            }],
            "usage": {
                "cost": 0.01
            }
        }

    monkeypatch.setattr(adapter, "post_json", fake_post)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(sys, "argv", [
        "openrouter_vlm_depth.py",
        "--method_name",
        "CLAUDE_OPUS_5_5",
        "--data_path",
        str(scene),
        "--output_path",
        str(tmp_path / "out"),
        "--export_depth",
        "--depth_only",
        "--point_grid_size",
        "2",
        "--cache_dir",
        str(tmp_path / "cache"),
    ])
    adapter.run()
    assert len(
        calls
    ) == 2  # one unproductive follow-up, then it stops (not the full 2 rounds)
