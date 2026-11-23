# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""scripts/inference/streaming_slam.py: pose readers, conventions and wiring of the streaming SLAM methods."""

import json
import os
from pathlib import Path

import numpy as np

from spatialarena.download import HF_LOCAL_FILE_CHECKPOINTS, METHOD_CODE_COMMITS, METHOD_CODE_REPOS, _code_leaf_dir
from spatialarena.methods import resolve_method
from scripts.inference.long_sequence_slam import FIXED_INPUT_LAUNCHER, PHASE_LAUNCHER, GpuMemorySampler, phase_command
from scripts.inference.streaming_slam import (
    CODE_DIRS,
    METHODS,
    _lingbot_resize,
    LOOP_INFO,
    _poses_by_index,
    read_r3_poses,
    run_abot_recon,
    read_scal3r_poses,
    weight_paths,
)

ENGINE = Path(__file__).resolve().parents[
    1] / "spatialarena" / "templates" / "_engine" / "run_method_case_list.sh"


def _c2w(index: int) -> np.ndarray:
    pose = np.eye(4)
    angle = 0.1 * index
    pose[:3, :3] = [[np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    pose[:3, 3] = [index, 2.0 * index, -0.5 * index]
    return pose


def test_poses_by_index_pads_3x4_to_homogeneous():
    """A distinct camera-to-world pose per index."""
    poses = _poses_by_index(np.stack([_c2w(i)[:3] for i in range(3)]))
    assert sorted(poses) == [0, 1, 2]
    for index, pose in poses.items():
        np.testing.assert_allclose(pose, _c2w(index))


def test_read_scal3r_poses_reads_raveled_c2w_rows(tmp_path):
    """Scal3R mat.txt rows are raveled 4x4 camera-to-world poses."""
    np.savetxt(tmp_path / "mat.txt",
               np.stack([_c2w(i).ravel() for i in range(4)]))
    poses = read_scal3r_poses(tmp_path)
    assert sorted(poses) == [0, 1, 2, 3]
    np.testing.assert_allclose(poses[3], _c2w(3))


def test_read_r3_poses_maps_returned_frames_to_input_indices(tmp_path):
    """R3 may return a subset of the input frames; output k belongs to input frame output_frame_ids[k]."""
    frame_ids = [0, 2, 5]
    (tmp_path / "camera").mkdir()
    for k, frame_id in enumerate(frame_ids):
        np.savez(tmp_path / "camera" / f"{k:06d}.npz",
                 pose=_c2w(frame_id),
                 intrinsics=np.eye(3))
    (tmp_path / "run_params.json").write_text(json.dumps(
        {"output_frame_ids": frame_ids}),
                                              encoding="utf-8")
    poses = read_r3_poses(tmp_path)
    assert sorted(poses) == frame_ids
    np.testing.assert_allclose(poses[5], _c2w(5))


def test_lingbot_resize_matches_its_benchmark_area_budget():
    """lingbot-map benchmark (area budget 255,000, align 14): 640x480 -> 574x434."""
    assert _lingbot_resize(np.zeros((480, 640, 3),
                                    dtype=np.uint8)).shape == (434, 574, 3)


def test_streaming_methods_are_registered_slam_only_with_downloadable_weights(
        tmp_path):
    """Registry, engine dispatch, code pins and weight paths agree for all six methods."""
    weights = weight_paths(tmp_path)
    engine = ENGINE.read_text(encoding="utf-8")
    assert "LOGER|ZIPMAP|LINGBOT_MAP|R3|SCAL3R|ABOT_RECON)" in engine
    for method in METHODS:
        spec = resolve_method(method)
        assert spec.adapter == "streaming_slam.py"
        assert spec.tasks == ("SLAM",)
        assert spec.tracks == ("rgb",)
        leaf = _code_leaf_dir(spec.code_default)
        assert leaf == CODE_DIRS[method]
        assert leaf in METHOD_CODE_REPOS and leaf in METHOD_CODE_COMMITS
        # The adapter reads the weights where the download table puts them, and ckpt_default is the main file.
        _repo, subdir, files = HF_LOCAL_FILE_CHECKPOINTS[method]
        assert weights[method] == tmp_path / "local" / subdir / files[0]
        assert spec.ckpt_default.endswith(
            f"/_checkpoints/local/{subdir}/{files[0]}")


def test_gpu_memory_sampler_starts_and_stops():
    """Regression: the sampler's stop event used to shadow threading.Thread._stop, so join() raised."""
    # It kept the event in ``self._stop``, so join() raised "TypeError: 'Event' object is not callable" after every
    # subprocess method (R3, Scal3R, VGGT-Long).
    sampler = GpuMemorySampler(os.getpid())
    sampler.start()
    peak = sampler.stop()
    assert peak is None or peak >= 0
    assert not sampler.is_alive()


def test_gpu_phase_keeps_depth_arrays_when_exporting_depth(tmp_path):
    """Regression: a split VGGT-Long / Pi-Long run slimmed its chunks to the alignment's arrays, so the depth export
    after the align phase failed with KeyError: 'depth'."""
    system = [
        "python",
        str(FIXED_INPUT_LAUNCHER), "--patch", "loader", "--",
        "/code/vggt_long.py", "--image_dir", "x"
    ]
    command = phase_command(system, "gpu", tmp_path, export_depth=True)
    assert command[:2] == ["python", str(PHASE_LAUNCHER)]
    separator = command.index("--")
    assert "--export_depth" in command[:
                                       separator] and "--patch" in command[:
                                                                           separator]
    assert command[separator + 1:] == ["/code/vggt_long.py", "--image_dir", "x"]
    assert "--export_depth" not in phase_command(system, "gpu", tmp_path)


def test_load_weights_refuses_a_checkpoint_that_does_not_match_the_model():
    import pytest
    import torch

    from scripts.inference.streaming_slam import load_weights

    model = torch.nn.Linear(2, 2)
    load_weights(model, {
        "weight": torch.ones(2, 2),
        "bias": torch.zeros(2)
    }, "Toy")
    assert torch.equal(model.weight, torch.ones(2, 2))
    with pytest.raises(RuntimeError, match="1 missing keys"):
        load_weights(model, {"weight": torch.ones(2, 2)}, "Toy")
    with pytest.raises(RuntimeError, match="1 unexpected keys"):
        load_weights(
            model, {
                "weight": torch.ones(2, 2),
                "bias": torch.zeros(2),
                "extra": torch.zeros(1)
            }, "Toy")


_FAKE_ABOT = """
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

CALLS = {}


class _Poses:
    def __init__(self, array):
        self.array = array

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class ABotRecon:
    @classmethod
    def from_pretrained(cls, checkpoint, **kwargs):
        CALLS["from_pretrained"] = kwargs
        model = cls()
        model.kwargs = kwargs
        model.config = SimpleNamespace(width=504, height=280)
        return model

    def infer(self, paths, **kwargs):
        CALLS["infer"] = kwargs
        out = Path(self.kwargs["loop_output_dir"])
        out.mkdir(parents=True)
        (out / "loop_info.json").write_text(json.dumps({"num_loop_edges": 2, "num_candidates": 5}))
        return SimpleNamespace(camera_poses=_Poses(np.stack([np.eye(4)] * len(paths))))
"""


def test_abot_recon_runs_with_loop_closure(tmp_path, monkeypatch):
    """ABot-Recon runs its loop-closure backend with the shared SALAD/DINOv2 weights; its summary reaches LOOP_INFO."""
    import sys

    code = tmp_path / "code"
    (code / "abot_recon").mkdir(parents=True)
    (code / "abot_recon" / "__init__.py").write_text(_FAKE_ABOT)
    monkeypatch.delitem(sys.modules, "abot_recon", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    staging = tmp_path / "work" / "images"
    staging.mkdir(parents=True)
    staged = [staging / f"{i:06d}.png" for i in range(3)]
    weights = weight_paths(tmp_path / "ckpt")
    LOOP_INFO.clear()
    poses = run_abot_recon(staged, code, weights)
    calls = sys.modules["abot_recon"].CALLS
    assert calls["from_pretrained"]["loop_closure"] is True
    assert calls["infer"]["loop_closure"] is True
    assert calls["from_pretrained"]["loop_salad_checkpoint"] == weights["SALAD"]
    assert calls["from_pretrained"]["loop_dino_checkpoint"] == weights["DINO"]
    assert weights["DINO"].name == "dinov2_vitb14_pretrain.pth"
    assert Path(
        calls["from_pretrained"]["loop_output_dir"]).parent == tmp_path / "work"
    assert sorted(poses) == [0, 1, 2]
    assert LOOP_INFO == {"num_loop_edges": 2, "num_candidates": 5}
    LOOP_INFO.clear()
