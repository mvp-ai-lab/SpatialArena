# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""What adapters write: prediction-map format and conversion, depth normalization, runtime metrics, prior scale.

prediction_map_format: Prediction maps are float32 compressed .npz (adapter_utils.save_prediction_map); older .npy
ones are still read.

convert_prediction_maps: Converting existing .npy prediction maps to .npz
(scripts/test/convert_prediction_maps_to_npz.py).

normalize_depths: normalize_depths() layouts, in particular single-frame (MonoDepth) predictions.

adapter_runtime_metrics: Standard adapter runtime metrics: written by adapter_utils, read back by the evaluators.

depth_prior_scale: Depth priors use the dataset's declared PNG scale (TUM: 5000 per metre), not the millimetre guess.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import os
from scripts.test import convert_prediction_maps_to_npz as convert
import json
import pytest
import scripts.inference.adapter_utils as adapter_utils
from spatialarena.eval.depth_prediction_evaluator import load_depth_map
from spatialarena.eval.prediction_artifacts import read_adapter_runtime
from PIL import Image


# ---- prediction_map_format ---------------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "inference"))

# pylint: disable-next=wrong-import-position
from adapter_utils import export_depth_predictions, save_prediction_map  # noqa: E402
from spatialarena.eval.prediction_artifacts import (  # noqa: E402
    confidence_file_for,
    find_depth_files,
    frame_file,
    load_confidence_map,
)


def test_export_writes_float32_compressed_npz_for_depth_and_confidence(tmp_path):
    depth = np.linspace(1.0, 2.0, num=12).reshape((1, 3, 4))
    confidence = np.full((1, 3, 4), 70000.0)  # beyond float16: clipped to its largest value
    export_depth_predictions(tmp_path, ["/images/frame_0.png"], depth, confidence=confidence)

    depth_file = tmp_path / "depths" / "frame_0.npz"
    with np.load(depth_file) as data:
        assert data.files == ["depth"] and data["depth"].dtype == np.float32
    with np.load(tmp_path / "confidences" / "frame_0.npz") as data:
        assert data.files == ["confidence"] and data["confidence"].dtype == np.float16
    assert not list(tmp_path.rglob("*.npy"))
    np.testing.assert_allclose(load_depth_map(depth_file), depth[0], rtol=1e-6)
    assert confidence_file_for(depth_file) == tmp_path / "confidences" / "frame_0.npz"
    np.testing.assert_array_equal(load_confidence_map(confidence_file_for(depth_file)), np.finfo(np.float16).max)


def test_a_new_npz_replaces_the_frames_old_npy(tmp_path):
    (tmp_path / "depths").mkdir()
    np.save(tmp_path / "depths" / "a.npy", np.zeros((2, 2), dtype=np.float32))
    save_prediction_map(tmp_path / "depths", "a", np.ones((2, 2)))
    assert [path.name for path in find_depth_files(tmp_path)] == ["a.npz"]


def test_older_npy_predictions_are_still_read(tmp_path):
    for folder in ("depths", "confidences", "raw_disparities"):
        (tmp_path / folder).mkdir()
    np.save(tmp_path / "depths" / "a.npy", np.full((2, 2), 3.0, dtype=np.float32))
    np.save(tmp_path / "confidences" / "a.npy", np.full((2, 2), 0.5, dtype=np.float16))
    save_prediction_map(tmp_path / "raw_disparities", "a", np.full((2, 2), 0.25))
    depth_file = tmp_path / "depths" / "a.npy"
    assert load_depth_map(depth_file)[0, 0] == 3.0
    assert load_confidence_map(confidence_file_for(depth_file)).dtype == np.float32
    # The raw disparity of a .npy depth is found by its stem, whichever format it is in.
    assert frame_file(tmp_path / "raw_disparities", "a") == tmp_path / "raw_disparities" / "a.npz"
    assert frame_file(tmp_path / "raw_disparities", "missing") is None


# ---- convert_prediction_maps -------------------------------------------------------------------------------

def _npz(path):
    with np.load(path) as data:
        return data.files, data[data.files[0]]


def test_prediction_maps_become_npz_and_ground_truth_is_left_alone(tmp_path):
    case = tmp_path / "batch" / "artifacts" / "full_standard" / "M" / "MonoDepth" / "ds" / "M" / "scene"
    for folder in ("depths", "confidences"):
        (case / folder).mkdir(parents=True)
    depth = np.random.default_rng(0).random((4, 5)).astype(np.float16)  # imported maps were stored as float16
    np.save(case / "depths" / "a.npy", depth)
    np.save(case / "confidences" / "a.npy", np.full((4, 5), 0.5, dtype=np.float32))
    # The same map hard-linked into a second batch (releases reuse identical files).
    other = tmp_path / "batch2" / "artifacts" / "x" / "depths"
    other.mkdir(parents=True)
    os.link(case / "depths" / "a.npy", other / "a.npy")
    # A frame already converted with the same values, and one whose .npz disagrees.
    np.save(case / "depths" / "b.npy", np.ones((2, 2), dtype=np.float32))
    np.savez_compressed(case / "depths" / "b.npz", depth=np.ones((2, 2), dtype=np.float32))
    np.save(case / "depths" / "c.npy", np.ones((2, 2), dtype=np.float32))
    np.savez_compressed(case / "depths" / "c.npz", depth=np.zeros((2, 2), dtype=np.float32))
    gt = tmp_path / "MonoDepth" / "ds" / "scene" / "depths"
    gt.mkdir(parents=True)
    np.save(gt / "a.npy", depth)

    assert convert.main([str(tmp_path), "--workers", "2"]) == 0

    files, values = _npz(case / "depths" / "a.npz")
    assert files == ["depth"] and values.dtype == np.float32
    np.testing.assert_array_equal(values, depth.astype(np.float32))
    assert os.path.samefile(case / "depths" / "a.npz", other / "a.npz")
    files, values = _npz(case / "confidences" / "a.npz")
    assert files == ["confidence"] and values.dtype == np.float16
    assert not (case / "depths" / "a.npy").exists() and not (other / "a.npy").exists()
    assert not (case / "depths" / "b.npy").exists()
    assert (case / "depths" / "c.npy").exists() and (case / "depths" / "c.npz").exists()  # conflict: both kept
    assert (gt / "a.npy").exists() and not (gt / "a.npz").exists()
    assert convert.map_files([tmp_path]) == [case / "depths" / "c.npy"]  # a rerun has only the conflict left


# ---- normalize_depths --------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "shape",
    [
        (1, 6, 8, 1),  # VGGT-style depth head, one frame: the MonoDepth case
        (3, 6, 8, 1),
        (1, 3, 6, 8),  # Pi3-style leading batch dim
        (3, 1, 6, 8),
        (1, 1, 6, 8),
        (1, 3, 6, 8, 1),
        (3, 6, 8),
    ],
)
def test_normalize_depths_layouts(shape):
    frames = 1 if shape in ((1, 6, 8, 1), (1, 1, 6, 8)) else 3
    assert adapter_utils.normalize_depths(np.zeros(shape)).shape == (frames, 6, 8)


def test_single_frame_spatial_export_restores_source_resolution(tmp_path):
    # Source 16x12 (WxH) resized to the model's 8x6 grid; the export must come back at 16x12.
    transform = adapter_utils.SpatialTransform(source_wh=(16, 12), resized_wh=(8, 6))
    depth = np.full((1, 6, 8, 1), 2.0, dtype=np.float32)
    adapter_utils.export_spatial_depth_predictions(tmp_path, ["img0.png"], depth, depth[..., 0], [transform])

    restored = load_depth_map(tmp_path / "depths" / "img0.npz")
    assert restored.shape == (12, 16)
    assert np.allclose(restored, 2.0)
    records = json.loads((tmp_path / "spatial_transforms.json").read_text())
    assert records["img0.png"]["source_wh"] == [16, 12]


def test_single_frame_legacy_export_keeps_full_map(tmp_path):
    adapter_utils.export_depth_predictions(tmp_path, ["img0.png"], np.ones((1, 6, 8, 1), dtype=np.float32))
    assert load_depth_map(tmp_path / "depths" / "img0.npz").shape == (6, 8)


# ---- adapter_runtime_metrics -------------------------------------------------------------------------------

def _write_metrics(tmp_path, monkeypatch, *, loaded: bool):
    monkeypatch.setattr(adapter_utils, "_MODEL_LOADED_AT", None)
    monkeypatch.setattr(adapter_utils, "_PROCESS_STARTED_AT", 100.0)
    monkeypatch.setattr(adapter_utils, "_peak_gpu_memory_mib", lambda: 512.0)
    clock = iter([110.0, 125.0, 125.0])  # mark_model_loaded, runtime_metrics, legacy "inference time"
    monkeypatch.setattr(adapter_utils.time, "time", lambda: next(clock))
    started_at = adapter_utils.mark_model_loaded() if loaded else next(clock)
    adapter_utils.write_adapter_metrics(
        output_path=tmp_path,
        method="TEST",
        image_paths=["a.png", "b.png", "c.png"],
        started_at=started_at,
        checkpoint=None,
        depth_only=False,
    )
    return json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))


def test_runtime_excludes_model_load_after_mark(tmp_path, monkeypatch):
    metrics = _write_metrics(tmp_path, monkeypatch, loaded=True)

    assert metrics["inference_seconds"] == 15.0
    assert metrics["model_load_seconds"] == 10.0
    assert metrics["peak_gpu_memory_mib"] == 512.0
    assert metrics["runtime_scope"] == "after_model_load"
    assert read_adapter_runtime(tmp_path) == {
        "inference_seconds": 15.0,
        "model_load_seconds": 10.0,
        "peak_gpu_memory_mib": 512.0,
        "runtime_scope": "after_model_load",
        "num_input_images": 3,
    }


def test_runtime_without_mark_falls_back_to_adapter_start(tmp_path, monkeypatch):
    metrics = _write_metrics(tmp_path, monkeypatch, loaded=False)

    assert metrics["inference_seconds"] == 15.0
    assert metrics["model_load_seconds"] is None
    assert metrics["runtime_scope"] == "adapter_started_at"


def test_read_adapter_runtime_marks_legacy_metrics(tmp_path):
    (tmp_path / "metrics.json").write_text(
        json.dumps({"inference time": 4.5, "Peak GPU Memory": 2048.0, "Preprocessed Images": 9}),
        encoding="utf-8",
    )

    assert read_adapter_runtime(tmp_path) == {
        "inference_seconds": 4.5,
        "model_load_seconds": None,
        "peak_gpu_memory_mib": 2048.0,
        "runtime_scope": "legacy",
        "num_input_images": 9,
    }
    assert read_adapter_runtime(tmp_path / "missing") is None


# ---- depth_prior_scale -------------------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "inference"))
from adapter_utils import dataset_depth_png_scale, load_depth_priors  # noqa: E402


def _scene(root: Path, depth_png_scale: float | None) -> Path:
    scene = root / "dataset" / "scene"
    (scene / "images").mkdir(parents=True)
    (scene / "depths").mkdir(parents=True)
    Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8)).save(scene / "images" / "000001.png")
    Image.fromarray(np.full((4, 6), 5000, dtype=np.uint16)).save(scene / "depths" / "000001.png")
    metadata = {"dataset_name": "dataset", "scenes": []}
    if depth_png_scale is not None:
        metadata["depth_png_scale"] = depth_png_scale
    (root / "dataset" / "metadata.json").write_text(json.dumps(metadata))
    return scene


def test_declared_tum_scale_gives_metres(tmp_path):
    scene = _scene(tmp_path, 5000.0)
    assert dataset_depth_png_scale(scene) == 5000.0
    [depth] = load_depth_priors(scene, [str(scene / "images" / "000001.png")])
    np.testing.assert_allclose(depth, 1.0)  # 5000 units = 1 m, not 5 m


def test_undeclared_scale_keeps_millimetre_guess(tmp_path):
    scene = _scene(tmp_path, None)
    assert dataset_depth_png_scale(scene) is None
    [depth] = load_depth_priors(scene, [str(scene / "images" / "000001.png")])
    np.testing.assert_allclose(depth, 5.0)
