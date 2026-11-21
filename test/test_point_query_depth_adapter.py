# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.inference import point_query_depth_adapter as adapter
from scripts.inference.point_query_depth_adapter import (
    build_message,
    parse_depth,
)
from spatialarena.eval.depth_prediction_evaluator import load_depth_map

# see spatialarena/paths.py
SPATIALARENA_ROOT = os.path.expanduser(os.environ.get("SPATIALARENA_ROOT", "~/spatialarena"))
DEPTHLM_SNAPSHOTS = Path(
    f"{SPATIALARENA_ROOT}/benchmarks/_checkpoints/huggingface/hub/"
    "models--facebook--DepthLM/snapshots"
)


def _blank_image(size=(64, 64)):
    return Image.new("RGB", size, color=(200, 200, 200))


def test_depthlm_message_matches_the_official_pixtral_eval():
    """DepthLM_Official eval.py (convert_example_pixtral(..., True)): one user turn, the image before the text."""
    messages = build_message("DEPTHLM", _blank_image(), 10, 10)
    assert [m["role"] for m in messages] == ["user"]
    assert [part["type"] for part in messages[0]["content"]] == ["image", "text"]


def test_depthlm_message_renders_with_the_model_chat_template():
    """The released Pixtral chat template renders the message (it raised a TypeError on a list system prompt)."""
    templates = sorted(DEPTHLM_SNAPSHOTS.glob("*/chat_template.json"))
    if not templates:
        pytest.skip("facebook/DepthLM is not in the local Hugging Face cache")
    import jinja2  # pylint: disable=import-outside-toplevel

    template = jinja2.Environment().from_string(json.loads(templates[0].read_text())["chat_template"])
    text = template.render(messages=build_message("DEPTHLM", _blank_image(), 10, 10), bos_token="<s>",
                           eos_token="</s>", raise_exception=lambda message: (_ for _ in ()).throw(ValueError(message)))
    assert text.startswith("<s>[INST][IMG]Given this image") and text.endswith("[/INST]")


def test_depthlm_prompt_asks_for_think_before_answer():
    messages = build_message("DEPTHLM", _blank_image(), 10, 10)
    user_text = next(part["text"] for part in messages[0]["content"] if part["type"] == "text")
    assert "<think>" in user_text and "</think>" in user_text
    assert "<answer>" in user_text and "</answer>" in user_text
    # The old prompt explicitly told the model to skip thinking -- must be gone.
    assert "Do not include any other numbers" not in user_text


def test_vlm3_and_qwen3_vl_messages_have_no_system_turn():
    # Scope discipline: the DepthLM fix must not change VLM3/Qwen3-VL's
    # message shape (VLM3 already asked for <think>/<answer> on its own;
    # Qwen3-VL is kept as a naive point-query baseline, untouched).
    vlm3_messages = build_message("VLM3", _blank_image(), 10, 10)
    qwen_messages = build_message("QWEN3_VL", _blank_image(), 10, 10)
    assert [m["role"] for m in vlm3_messages] == ["user"]
    assert [m["role"] for m in qwen_messages] == ["user"]


def test_parse_depth_extracts_answer_tag_even_with_a_preceding_think_block():
    text = "<think> The point is around 2.35 meters away from the camera. </think><answer> 2.35 </answer>"
    assert parse_depth(text) == 2.35


def test_parse_depth_falls_back_to_meter_pattern_without_answer_tag():
    assert parse_depth("The object is approximately 3.1 meters away.") == 3.1




@pytest.fixture(name="point_query_run")
def fixture_point_query_run(tmp_path, monkeypatch):
    """Exercise the real image/query/export path without loading a model."""
    for name in list(os.environ):
        if name.startswith("POINT_QUERY_"):
            monkeypatch.delenv(name)
    images = tmp_path / "images"
    images.mkdir()
    output = tmp_path / "output"
    metrics = {}
    monkeypatch.setattr(adapter, "load_model_and_processor", lambda args: (None, None))
    monkeypatch.setattr(adapter, "mark_model_loaded", lambda: 0.0)
    monkeypatch.setattr(adapter, "write_adapter_metrics", lambda **kwargs: metrics.update(kwargs))

    def prepare(answers, image_count=1, strict=True, grid_size=16, method="QWEN3_VL"):
        for index in range(image_count):
            _blank_image((32, 32)).save(images / f"{index:03d}.png")
        argv = [
            "point_query_depth_adapter", "--method_name", method,
            "--data_path", str(tmp_path), "--output_path", str(output), "--ckpt_path", "unused",
            "--image_load_resolution", "32", "--max_images_per_run", str(image_count),
            "--point_grid_size", str(grid_size), "--query_batch_size", "7",
            "--export_depth", "--depth_only",
        ]
        if strict:
            argv.append("--require_spatial_queries")
        monkeypatch.setattr(sys, "argv", argv)
        responses = iter(answers)

        def generate(_model, _processor, messages, _max_new_tokens):
            return [next(responses) for _ in messages]

        monkeypatch.setattr(adapter, "generate_batch", generate)
        return output, metrics

    return prepare


def _read_query_evidence(output):
    def reject_non_json(value):
        raise AssertionError(f"Non-JSON number in query evidence: {value}")

    records = [json.loads(line, parse_constant=reject_non_json)
               for line in (output / "point_query_outputs.jsonl").read_text().splitlines()]
    coverage = json.loads((output / "point_query_coverage.json").read_text())
    return records, coverage


def test_strict_queries_reject_center_only_before_model_load(point_query_run, monkeypatch):
    point_query_run([], grid_size=1)

    def unexpected_load(_args):
        pytest.fail("Invalid strict query configuration must fail before model loading")

    monkeypatch.setattr(adapter, "load_model_and_processor", unexpected_load)
    with pytest.raises(ValueError, match="point_grid_size >= 2"):
        adapter.run()


@pytest.mark.parametrize("method", ["DEPTHLM", "QWEN3_VL"])
def test_strict_16_grid_exports_partial_answers_and_coverage(point_query_run, method):
    answers = ["cannot estimate"] * 256
    answers[0], answers[-1] = "<answer>2</answer>", "<answer>6</answer>"
    output, metrics = point_query_run(answers, method=method)
    adapter.run()
    records, coverage = _read_query_evidence(output)
    assert len(records) == 256
    assert len({(record["x"], record["y"]) for record in records}) == 256
    assert sum(record["depth"] is None for record in records) == 254
    assert coverage == [{"image": records[0]["image"], "queries": 256, "valid": 2}]
    depth = load_depth_map(output / "depths" / "000.npz")
    assert depth.shape == (32, 32)
    assert np.isfinite(depth).all()
    assert depth[0, 0] == 2 and depth[-1, -1] == 6
    assert depth[16, 16] == 4  # Missing queries use this image's valid median.
    assert 2 < depth[1, 1] < 4  # Actual bilinear interpolation, not nearest-neighbor.
    extra = metrics["metrics_extra"]
    assert extra["total_point_queries"] == 256
    assert extra["parse_failures"] == 254
    assert extra["fallback_imputations"] == 0
    assert extra["require_spatial_queries"] is True
    assert extra["interpolation"] == "bilinear"
    assert extra["image_query_coverage"] == coverage


@pytest.mark.parametrize("preceding_valid_image", [False, True])
def test_strict_all_invalid_preserves_evidence_without_imputation(point_query_run, preceding_valid_image):
    answers = (["<answer>9</answer>"] * 256 if preceding_valid_image else []) + ["unknown"] * 256
    image_count = 2 if preceding_valid_image else 1
    output, metrics = point_query_run(answers, image_count=image_count)
    with pytest.raises(RuntimeError, match="constant imputation disabled"):
        adapter.run()
    records, coverage = _read_query_evidence(output)
    assert len(records) == image_count * 256
    assert [item["valid"] for item in coverage] == ([256, 0] if preceding_valid_image else [0])
    assert all(item["queries"] == 256 for item in coverage)
    assert all(record["depth"] is None for record in records[-256:])
    assert not any(record.get("fallback_imputed") for record in records)
    assert not list(output.rglob("*.npy")) and not list(output.rglob("*.npz"))
    assert not metrics  # A failed run does not write successful completion metadata.


def test_legacy_fallback_retains_records_and_previous_image_depth(point_query_run):
    output, metrics = point_query_run(
        ["unknown", "<answer>7</answer>", "unknown"], image_count=3, strict=False, grid_size=1,
    )
    adapter.run()
    records, coverage = _read_query_evidence(output)
    assert len(records) == 5  # Three query records plus two explicit imputations.
    assert [item["valid"] for item in coverage] == [0, 1, 0]
    for index, expected in enumerate([1, 7, 7]):
        np.testing.assert_array_equal(load_depth_map(output / "depths" / f"{index:03d}.npz"), expected)
    assert metrics["metrics_extra"]["fallback_imputations"] == 2
    assert metrics["metrics_extra"]["require_spatial_queries"] is False


def test_strict_valid_constant_answers_are_not_reclassified(point_query_run):
    output, _metrics = point_query_run(["<answer>100</answer>"] * 256)
    adapter.run()
    _records, coverage = _read_query_evidence(output)
    assert coverage[0]["valid"] == 256
    np.testing.assert_array_equal(load_depth_map(output / "depths" / "000.npz"), 100)


def test_strict_grid_environment_configuration_is_opt_in(point_query_run, monkeypatch):
    point_query_run([], strict=False)
    argv = list(sys.argv)
    index = argv.index("--point_grid_size")
    del argv[index:index + 2]
    monkeypatch.setattr(sys, "argv", argv)
    args = adapter.parse_args()
    assert args.point_grid_size == 1
    assert args.require_spatial_queries is False
    monkeypatch.setenv("POINT_QUERY_DEPTH_GRID_SIZE", "16")
    monkeypatch.setenv("POINT_QUERY_REQUIRE_SPATIAL", "1")
    args = adapter.parse_args()
    assert args.point_grid_size == 16
    assert args.require_spatial_queries is True
def test_parse_depth_returns_none_for_unparseable_text():
    assert parse_depth("I cannot determine the distance.") is None


def test_parse_depth_rejects_non_positive_values():
    assert parse_depth("<answer>-1.0</answer>") is None
    assert parse_depth("<answer>0</answer>") is None


def test_run_exports_depth_at_true_source_resolution_not_the_contained_size(tmp_path, monkeypatch):
    """load_rgb_image's ImageOps.contain downscales to fit --image_load_resolution; the exported
    depth map must be restored back to the image's true source resolution (every other patched
    adapter's convention), not left at that smaller, contained size."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    source_w, source_h = 300, 200
    Image.fromarray((np.random.rand(source_h, source_w, 3) * 255).astype("uint8")).save(images_dir / "a.png")

    monkeypatch.setattr(adapter, "load_model_and_processor", lambda args: (object(), object()))
    monkeypatch.setattr(adapter, "mark_model_loaded", lambda: 0.0)
    monkeypatch.setattr(adapter, "generate_batch", lambda *a, **k: ["<answer>2.5</answer>"])

    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "point_query_depth_adapter.py",
            "--method_name", "VLM3",
            "--data_path", str(tmp_path),
            "--output_path", str(out),
            "--ckpt_path", "unused",
            "--image_load_resolution", "64",
            "--export_depth",
            "--depth_only",
        ],
    )
    adapter.run()

    depth = load_depth_map(next((out / "depths").glob("*.npz")))
    assert depth.shape == (source_h, source_w)
    np.testing.assert_allclose(depth, 2.5, atol=1e-4)
