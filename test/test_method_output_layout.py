# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from spatialarena.eval.prediction_artifacts import (
    PredictionRequirements,
    validate_prediction_output,
)
from spatialarena.eval.result_layout import (
    has_colmap_model_files,
    method_scene_output_candidates,
    method_scene_output_path,
    resolve_method_scene_output_path,
    resolve_sparse_model_path,
    resolve_sparse_model_path_from_any,
)


def write_colmap_stub(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (path / name).write_bytes(b"stub")


def test_resolve_sparse_model_prefers_canonical_sparse_zero(tmp_path):
    scene_output = tmp_path / "scene_a"
    write_colmap_stub(scene_output / "sparse")
    write_colmap_stub(scene_output / "sparse" / "0")

    assert resolve_sparse_model_path(scene_output) == scene_output / "sparse" / "0"


def test_resolve_sparse_model_accepts_legacy_sparse_root(tmp_path):
    scene_output = tmp_path / "scene_a"
    write_colmap_stub(scene_output / "sparse")

    assert resolve_sparse_model_path(scene_output) == scene_output / "sparse"
    assert has_colmap_model_files(scene_output / "sparse")


def test_resolve_sparse_model_from_any_accepts_scene_sparse_and_sparse_zero(tmp_path):
    scene_output = tmp_path / "scene_a"
    write_colmap_stub(scene_output / "sparse" / "0")

    assert resolve_sparse_model_path_from_any(scene_output) == scene_output / "sparse" / "0"
    assert resolve_sparse_model_path_from_any(scene_output / "sparse") == scene_output / "sparse" / "0"
    assert resolve_sparse_model_path_from_any(scene_output / "sparse" / "0") == scene_output / "sparse" / "0"


def test_method_scene_output_path_keeps_legacy_contract_without_task(tmp_path):
    path = method_scene_output_path(
        tmp_path,
        "exp",
        "bonn",
        "VGGT",
        "rgbd_bonn_static",
    )

    assert path == tmp_path / "exp" / "bonn" / "VGGT" / "rgbd_bonn_static"


def test_method_scene_output_candidates_prefer_task_namespace(tmp_path):
    candidates = method_scene_output_candidates(
        tmp_path,
        "exp",
        "eth3d",
        "DA3",
        "courtyard",
        task_name="RMVD",
    )

    assert candidates == [
        tmp_path / "exp" / "RMVD" / "eth3d" / "DA3" / "courtyard",
        tmp_path / "exp" / "eth3d" / "DA3" / "courtyard",
    ]


def test_resolve_method_scene_output_falls_back_to_legacy_layout(tmp_path):
    legacy_path = tmp_path / "exp" / "eth3d" / "DA3" / "courtyard"
    legacy_path.mkdir(parents=True)

    resolved = resolve_method_scene_output_path(
        tmp_path,
        "exp",
        "eth3d",
        "DA3",
        "courtyard",
        task_name="RMVD",
    )

    assert resolved == legacy_path


def test_validate_depth_task_requires_depth_predictions(tmp_path):
    scene_output = tmp_path / "scene"
    (scene_output / "depths").mkdir(parents=True)
    (scene_output / "depths" / "000.npy").write_bytes(b"stub")

    status = validate_prediction_output(
        task="VideoDepth",
        dataset="bonn",
        method="DA3",
        scene="rgbd_bonn_static",
        scene_output_path=scene_output,
    )

    assert status.ok
    assert len(status.depth_files) == 1


def test_validate_sfm_task_requires_sparse_model(tmp_path):
    scene_output = tmp_path / "scene"
    scene_output.mkdir()

    missing = validate_prediction_output(
        task="SfM",
        dataset="imc",
        method="VGGT",
        scene="scene_a",
        scene_output_path=scene_output,
    )
    assert missing.missing == ["sparse_colmap_model"]

    write_colmap_stub(scene_output / "sparse" / "0")
    present = validate_prediction_output(
        task="SfM",
        dataset="imc",
        method="VGGT",
        scene="scene_a",
        scene_output_path=scene_output,
    )
    assert present.ok


def test_validate_optional_point_cloud_requirement(tmp_path):
    scene_output = tmp_path / "scene"
    scene_output.mkdir()

    missing = validate_prediction_output(
        task="MVRecon",
        dataset="7scenes",
        method="method",
        scene="chess",
        scene_output_path=scene_output,
        requirements=PredictionRequirements(require_points=True),
    )
    assert missing.missing == ["point_cloud"]

    (scene_output / "points3D_dense.ply").write_text("ply\n", encoding="utf-8")
    present = validate_prediction_output(
        task="MVRecon",
        dataset="7scenes",
        method="method",
        scene="chess",
        scene_output_path=scene_output,
        requirements=PredictionRequirements(require_points=True),
    )
    assert present.ok


def test_vggt_and_da3_dry_runs_report_standard_artifact_layout(tmp_path):
    scene = tmp_path / "scene"
    images = scene / "images"
    images.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(images / "000.png")

    repo_root = Path(__file__).resolve().parents[1]
    for script_name, method in (
        ("vggt.py", "VGGT"),
        ("da3.py", "DA3"),
        ("pi3.py", "PI3"),
        ("fastvggt.py", "FASTVGGT"),
        ("come.py", "COME"),
        ("cut3r.py", "CUT3R"),
        ("fast3r.py", "FAST3R"),
        ("vggt_omega.py", "VGGT_OMEGA"),
    ):
        result = subprocess.run(
            [
                sys.executable,
                f"scripts/inference/{script_name}",
                "--data_path",
                str(scene),
                "--output_path",
                str(tmp_path / method),
                "--export_colmap",
                "--export_depth",
                "--dry_run",
            ],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        assert payload["method"] == method
        assert payload["num_images"] == 1
        assert payload["depths"].endswith("/depths")
        assert payload["sparse"].endswith("/sparse/0")
