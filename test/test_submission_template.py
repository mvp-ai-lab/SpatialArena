# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Tests for the submission template and the submission output checker."""

from __future__ import annotations

import pathlib
import subprocess
import sys

import numpy as np
from PIL import Image

from spatialarena.geometry import read_write_model

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "scripts" / "inference" / "submission_template.py"
CHECKER = REPO_ROOT / "scripts" / "inference" / "check_submission_output.py"
ADAPTER = REPO_ROOT / "scripts" / "inference" / "external_command_adapter.py"


def _scene(tmp_path: pathlib.Path, count: int = 3) -> pathlib.Path:
    """Writes a scene of `count` small random images; returns the scene."""
    images = tmp_path / "scene" / "images"
    images.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for index in range(count):
        pixels = rng.integers(0, 255, (24, 32, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(images / f"frame_{index:03d}.png")
    return tmp_path / "scene"


def _run(*args: object) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *map(str, args)],
                          cwd=REPO_ROOT,
                          capture_output=True,
                          text=True,
                          check=False)


def test_template_output_passes_the_checker_for_every_task(tmp_path):
    """The template's placeholder prediction is in the evaluator's layout."""
    scene = _scene(tmp_path)
    output = tmp_path / "out"
    result = _run(TEMPLATE, "--images", scene / "images", "--output", output)
    assert result.returncode == 0, result.stderr

    with np.load(output / "depths" / "frame_000.npz") as data:
        assert data["depth"].shape == (24, 32)
        assert data["depth"].dtype == np.float32
    _cameras, images, _points = read_write_model.read_model(
        str(output / "sparse" / "0"))
    assert sorted(image.name for image in images.values()) == [
        "frame_000.png", "frame_001.png", "frame_002.png"
    ]
    checked = _run(CHECKER, "--images", scene / "images", "--output", output)
    assert checked.returncode == 0, checked.stdout
    assert checked.stdout.count(": ok") == 5


def test_template_runs_through_the_external_command_adapter(tmp_path):
    """The command template the guide shows works in the adapter."""
    scene = _scene(tmp_path, count=2)
    code = tmp_path / "code"
    code.mkdir()
    (code / "spatialarena_infer.py").write_text(TEMPLATE.read_text())
    output = tmp_path / "out"
    command = ("python {code_path}/spatialarena_infer.py --images {images_dir} "
               "--output {output_path} --checkpoint {ckpt_path}")
    result = _run(ADAPTER, "--data_path", scene, "--output_path", output,
                  "--method_name", "MY_MODEL", "--code_path", code,
                  "--ckpt_path", tmp_path / "weights.pt", "--command", command)
    assert result.returncode == 0, result.stderr
    assert (output / "depths" / "frame_001.npz").is_file()
    assert (output / "sparse" / "0" / "images.txt").is_file()


def test_checker_reports_missing_depths_and_wrong_image_names(tmp_path):
    """A missing depth map and a renamed COLMAP image fail the check."""
    scene = _scene(tmp_path)
    output = tmp_path / "out"
    assert _run(TEMPLATE, "--images", scene / "images", "--output",
                output).returncode == 0
    (output / "depths" / "frame_001.npz").unlink()
    images_txt = output / "sparse" / "0" / "images.txt"
    images_txt.write_text(images_txt.read_text().replace(
        "frame_002.png", "002.png"))

    checked = _run(CHECKER, "--images", scene / "images", "--output", output,
                   "--task", "MonoDepth", "--task", "SfM")
    assert checked.returncode == 1
    assert "MonoDepth: FAIL" in checked.stdout
    assert "1 of 3 images have no depth map" in checked.stdout
    assert "SfM: FAIL" in checked.stdout
    assert "not input file names, e.g. 002.png" in checked.stdout
