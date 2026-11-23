# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import yaml
from spatialarena.methods import resolve_method, validate_track
from spatialarena.task_track_matrix import runnable_tracks

# ---- video_diffusion_depth ---------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/inference"))
from video_diffusion_depth import processing_size, restore_output


def test_official_resolution_rounding():
    assert processing_size(1242, 375, 1024) == (1024, 320)
    assert processing_size(640, 480, 1024) == (640, 512)
    with pytest.raises(ValueError):
        processing_size(10, 10, 1024)


def test_crafter_native_list_output_and_sequence_normalization():
    # diffusers returns a batch list of THWC arrays, not necessarily one ndarray.
    frames = np.stack([np.full((3, 4, 3), .2), np.full((3, 4, 3), .8)])
    raw, depth = restore_output([frames], "DEPTHCRAFTER", 2, (9, 7))
    assert depth.shape == (2, 7, 9)
    np.testing.assert_allclose(raw[0], 0)
    np.testing.assert_allclose(raw[1], 1)
    np.testing.assert_allclose(depth[0], 1000, rtol=1e-6)
    np.testing.assert_allclose(depth[1], 1)


def test_chrono_depth_is_not_inverted():
    frames = np.full((1, 1, 3, 4), .25, dtype=np.float32)
    _raw, depth = restore_output(frames, "CHRONODEPTH", 1, (9, 7))
    assert depth.shape == (1, 7, 9)
    np.testing.assert_allclose(depth, .25)


@pytest.mark.parametrize("frames,count", [(np.zeros((2, 1, 3, 4)), 3),
                                          (np.full((2, 1, 3, 4), np.nan), 2)])
def test_invalid_predictions_fail(frames, count):
    with pytest.raises(ValueError):
        restore_output(frames, "CHRONODEPTH", count, (9, 7))


def test_degenerate_disparity_fails():
    with pytest.raises(ValueError, match="Constant"):
        restore_output(np.ones((1, 2, 3, 4, 3)), "DEPTHCRAFTER", 2, (9, 7))


@pytest.mark.parametrize("method", ["DEPTHCRAFTER", "CHRONODEPTH"])
def test_real_cli_dry_run_symlink_and_subset(tmp_path, method):
    import json
    from PIL import Image
    scene = tmp_path / "scene"
    (scene / "images").mkdir(parents=True)
    source = tmp_path / "original.png"
    Image.new("RGB", (124, 80)).save(source)
    for i in range(3):
        (scene / "images" / f"frame{i}.png").symlink_to(source)
    env = dict(os.environ, BENCHMARK_MAX_INPUT_FRAMES="2")
    result = subprocess.run([
        sys.executable,
        str(ROOT / "scripts/inference/video_diffusion_depth.py"), "--method",
        method, "--data_path",
        str(scene), "--output_path",
        str(tmp_path / "out"), "--code_path",
        str(tmp_path), "--spatial_policy", "shared", "--export_depth",
        "--dry_run"
    ],
                            env=env,
                            capture_output=True,
                            text=True,
                            check=True)
    payload = json.loads(result.stdout)
    assert payload["method"] == method
    assert payload["num_images"] == 2


# ---- videodepth_diffusion_registry -------------------------------------------------------------------------


@pytest.mark.parametrize("name,alias", [("DEPTHCRAFTER", "depthcraft"),
                                        ("CHRONODEPTH", "ChronoDepth")])
def test_video_diffusion_registration(name, alias):
    spec = resolve_method(alias)
    assert spec.name == name
    assert spec.native
    assert spec.adapter == "video_diffusion_depth.py"
    assert spec.external_command_env == name + "_COMMAND"
    assert spec.default_depth_protocol == ("inverse_huber_irls_gt"
                                           if name == "DEPTHCRAFTER" else
                                           "depth_huber_irls_gt")
    assert runnable_tracks("VideoDepth", spec) == ("rgb",)
    for task in ("MonoDepth", "SfM", "SLAM", "MVRecon"):
        assert runnable_tracks(task, spec) == ()
    with pytest.raises(ValueError):
        validate_track(name, "rgb_cameras")
    path = Path(__file__).resolve(
    ).parents[1] / "configs/benchmark/full_videodepth.yaml"
    models = yaml.safe_load(path.read_text())["models"]
    assert sum(m["name"] == name and m["track"] == "rgb" for m in models) == 1


@pytest.mark.parametrize("method", ["DEPTHCRAFTER", "CHRONODEPTH"])
def test_actual_engine_inference_branch(tmp_path, method):
    import json

    root = Path(__file__).resolve().parents[1]
    engine = (
        root /
        "spatialarena/templates/_engine/run_method_case_list.sh").read_text()
    function = engine.split("run_inference_once() {",
                            1)[1].split("\nverify_case()", 1)[0]
    stub = tmp_path / "python-stub"
    stub.write_text("#!" + sys.executable +
                    "\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
    stub.chmod(0o755)
    (tmp_path / "scene").mkdir()
    script = '''set -eu
canonical_task_name() { printf '%s' "$1"; }
dataset_root_for_case() { printf '%s' "$TEST_ROOT"; }
env_value() { printf '%s' "$TEST_ROOT"; }
depth_only_exports_poses() { return 1; }
''' + "run_inference_once() {" + function + '''
run_inference_once VideoDepth bonn scene 1 '' "$TEST_ROOT/out"
'''
    env = dict(os.environ,
               TEST_ROOT=str(tmp_path),
               PYTHON=str(stub),
               METHOD=method,
               METHOD_ID=method,
               METHOD_NATIVE="1",
               METHOD_ADAPTER="video_diffusion_depth.py",
               METHOD_CODE_ENV="CODE",
               METHOD_CKPT_ENV="CKPT",
               BENCHMARK_TRACK="rgb",
               MAX_IMAGES_PER_RUN="100",
               BENCHMARK_TMP_ROOT=str(tmp_path),
               SPATIAL_POLICY="shared")
    result = subprocess.run(["bash", "-c", script],
                            env=env,
                            capture_output=True,
                            text=True,
                            check=True)
    args = json.loads(result.stdout.splitlines()[-1])
    assert args[0] == "scripts/inference/video_diffusion_depth.py"
    assert args[args.index("--method") + 1] == method
    assert args[args.index("--max_images_per_run") + 1] == "100"
    assert "--export_depth" in args and "--export_colmap" not in args
