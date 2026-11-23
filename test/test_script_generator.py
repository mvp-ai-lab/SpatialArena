# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from spatialarena.paths import SPATIALARENA_ROOT
from spatialarena.script_generator import (
    ENGINE_FILES,
    GeneratedCaseListEntry,
    ensure_engine_copied,
    generate_run_scripts,
    render_case_file,
)
from spatialarena.report_layout import canonical_report_dir


def test_generated_case_file_leaves_out_cases_the_track_does_not_run_on(
        tmp_path):
    """SLAM rgb_depth: a sequence without depth maps (TRACK_EXCLUDED_SCENES) is not in the generated case file."""
    work_item = {
        "model": "PI3_X",
        "suite": "full-slam",
        "track": "rgb_depth",
        "slam_range": "short",
        "task": "SLAM",
        "dataset": "spatialarena"
    }
    entries = [
        GeneratedCaseListEntry(task="SLAM",
                               dataset="spatialarena",
                               scene="EdinburghCastle1",
                               interval=3),
        GeneratedCaseListEntry(task="SLAM",
                               dataset="spatialarena",
                               scene="ArtHouse",
                               interval=1),
    ]
    generated = generate_run_scripts(
        work_item,
        entries,
        output_root=tmp_path,
        run_id="run1",
        benchmark_eval_root=Path("/eval"),
        checkpoint_root=Path("/ckpt"),
        method_code_root=Path("/code"),
        benchmark_aux_root=Path("/aux"),
    )
    lines = [
        line for line in generated.case_file.read_text(
            encoding="utf-8").splitlines() if line and not line.startswith("#")
    ]
    assert lines == ["SLAM|spatialarena|ArtHouse|1"]


def test_render_case_file_matches_engine_format():
    entries = [
        GeneratedCaseListEntry(task="MonoDepth",
                               dataset="marigold",
                               scene="nyuv2",
                               interval=1),
        GeneratedCaseListEntry(task="SfM",
                               dataset="eth3d",
                               scene="courtyard",
                               interval=2,
                               order="fps"),
    ]
    text = render_case_file(entries)
    lines = [
        line for line in text.splitlines() if line and not line.startswith("#")
    ]
    assert lines == ["MonoDepth|marigold|nyuv2|1", "SfM|eth3d|courtyard|2|fps"]


def test_ensure_engine_copied_is_byte_identical_and_idempotent(tmp_path):
    engine_dir = ensure_engine_copied(tmp_path)
    for name in ENGINE_FILES:
        assert (engine_dir /
                name).read_bytes() == (Path(__file__).resolve().parents[1] /
                                       "spatialarena" / "templates" /
                                       "_engine" / name).read_bytes()
    # Calling again must not fail or change content (idempotent).
    ensure_engine_copied(tmp_path)


def test_generate_run_scripts_writes_case_file_and_wrappers(tmp_path):
    work_item = {
        "task": "MonoDepth",
        "dataset": "marigold",
        "model": "VGGT",
        "track": "rgb",
        "suite": "full-depth"
    }
    entries = [
        GeneratedCaseListEntry(task="MonoDepth",
                               dataset="marigold",
                               scene="nyuv2")
    ]
    generated = generate_run_scripts(
        work_item,
        entries,
        output_root=tmp_path,
        run_id="20260101_000000",
        benchmark_eval_root=Path("/eval"),
        checkpoint_root=Path("/ckpt"),
        method_code_root=Path("/code"),
        benchmark_aux_root=Path("/aux"),
    )
    assert generated.case_file.exists()
    assert generated.infer_script.exists()
    assert generated.eval_script.exists()
    assert generated.infer_script.name == "run_VGGT_full-depth_MonoDepth_marigold_infer.sh"
    assert generated.eval_script.name == "run_VGGT_full-depth_MonoDepth_marigold_eval.sh"

    infer_text = generated.infer_script.read_text(encoding="utf-8")
    assert infer_text.startswith("#!/bin/bash\n#SBATCH")
    assert "--gres=gpu" in infer_text
    assert str(generated.case_file) in infer_text
    assert "exec " in infer_text

    eval_text = generated.eval_script.read_text(encoding="utf-8")
    assert "--partition=cpu" in eval_text

    assert generated.report_dir == Path(
        "/eval/reports/benchmark_full_reports/full-depth/VGGT")


def test_work_items_of_one_method_get_their_own_files(tmp_path):
    """Datasets and SLAM ranges of one method each get their own case file, wrappers, job name and log folder."""
    scripts = []
    for dataset, slam_range in (("7scenes", "short"), ("7scenes", "long"),
                                ("tum_dynamics", "short")):
        work_item = {
            "task": "SLAM",
            "dataset": dataset,
            "model": "PI3_X",
            "track": "rgb_depth",
            "suite": "full-slam",
            "slam_range": slam_range
        }
        entries = [
            GeneratedCaseListEntry(task="SLAM",
                                   dataset=dataset,
                                   scene=f"{dataset}_scene",
                                   interval=10)
        ]
        scripts.append(
            generate_run_scripts(
                work_item,
                entries,
                output_root=tmp_path,
                run_id="run1",
                benchmark_eval_root=Path("/eval"),
                checkpoint_root=Path("/ckpt"),
                method_code_root=Path("/code"),
                benchmark_aux_root=Path("/aux"),
                slurm_exclude="node1,node2",
                slurm_resources={"eval": {
                    "mem": "64G",
                    "qos": "low"
                }},
            ))
    assert len({generated.case_file for generated in scripts}) == 3
    assert len({generated.infer_script for generated in scripts}) == 3
    first = scripts[0]
    assert "7scenes_scene" in first.case_file.read_text(encoding="utf-8")
    infer_text = first.infer_script.read_text(encoding="utf-8")
    assert "#SBATCH --job-name=infer-PI3_X_rgb_depth_full-slam-short_SLAM_7scenes" in infer_text
    assert f"#SBATCH --output={tmp_path / 'logs'}/%x_%j.log" in infer_text
    assert "#SBATCH --exclude=node1,node2" in infer_text
    eval_text = first.eval_script.read_text(encoding="utf-8")
    assert "#SBATCH --mem=64G" in eval_text and "#SBATCH --qos=low" in eval_text
    assert eval_text.count(
        "#SBATCH --mem=") == 1 and "#SBATCH --mem=64G" not in infer_text
    assert "export SLAM_RANGE" not in infer_text  # set by the evaluator's extra_env, not by the generator
    assert scripts[1].report_dir == Path(
        "/eval/reports/benchmark_full_slam_reports/full-slam-long/PI3_X_rgb_depth"
    )


def test_generate_run_scripts_method_id_includes_track_for_non_rgb(tmp_path):
    work_item = {
        "task": "MVRecon",
        "dataset": "dtu",
        "model": "PI3_X",
        "track": "rgb_cameras",
        "suite": "full-depth",
    }
    entries = [
        GeneratedCaseListEntry(task="MVRecon", dataset="dtu", scene="scan1")
    ]
    generated = generate_run_scripts(
        work_item,
        entries,
        output_root=tmp_path,
        run_id="run1",
        benchmark_eval_root=Path("/eval"),
        checkpoint_root=Path("/ckpt"),
        method_code_root=Path("/code"),
        benchmark_aux_root=Path("/aux"),
    )
    assert generated.report_dir == Path(
        "/eval/reports/benchmark_full_reports/full-depth/PI3_X_rgb_cameras")
    assert generated.infer_script.name == "run_PI3_X_rgb_cameras_full-depth_MVRecon_dtu_infer.sh"


def test_generate_run_scripts_renders_extra_env_before_exec(tmp_path):
    work_item = {
        "task": "MonoDepth",
        "dataset": "marigold",
        "model": "MyCustomModel",
        "track": "rgb",
        "suite": "full-depth",
        "command": "python infer.py --data {data_path}",
    }
    entries = [
        GeneratedCaseListEntry(task="MonoDepth",
                               dataset="marigold",
                               scene="nyuv2")
    ]
    generated = generate_run_scripts(
        work_item,
        entries,
        output_root=tmp_path,
        run_id="run1",
        benchmark_eval_root=Path("/eval"),
        checkpoint_root=Path("/ckpt"),
        method_code_root=Path("/code"),
        benchmark_aux_root=Path("/aux"),
        extra_env={
            "METHOD_COMMAND": "python infer.py --data {data_path}",
            "RESOLUTION": "336"
        },
    )
    infer_text = generated.infer_script.read_text(encoding="utf-8")
    lines = infer_text.splitlines()
    export_lines = [i for i, l in enumerate(lines) if l.startswith("export ")]
    exec_lines = [i for i, l in enumerate(lines) if l.startswith("exec ")]
    assert export_lines and exec_lines
    assert max(export_lines) < min(exec_lines)  # exports precede the exec call
    assert "export METHOD_COMMAND=" in infer_text
    assert "export RESOLUTION=336" in infer_text


def test_generate_run_scripts_no_extra_env_by_default(tmp_path):
    work_item = {
        "task": "MonoDepth",
        "dataset": "marigold",
        "model": "VGGT",
        "track": "rgb",
        "suite": "full-depth"
    }
    entries = [
        GeneratedCaseListEntry(task="MonoDepth",
                               dataset="marigold",
                               scene="nyuv2")
    ]
    generated = generate_run_scripts(
        work_item,
        entries,
        output_root=tmp_path,
        run_id="run1",
        benchmark_eval_root=Path("/eval"),
        checkpoint_root=Path("/ckpt"),
        method_code_root=Path("/code"),
        benchmark_aux_root=Path("/aux"),
    )
    infer_text = generated.infer_script.read_text(encoding="utf-8")
    exports = [
        line for line in infer_text.splitlines() if line.startswith("export ")
    ]
    # Only the data root the script was generated with (sbatch --export=NIL drops the submitting environment).
    assert exports == [
        f'export SPATIALARENA_ROOT="${{SPATIALARENA_ROOT:-{SPATIALARENA_ROOT}}}"'
    ]


def test_canonical_report_dir_is_one_folder_per_suite_and_method():
    """Every full suite writes one report folder per method."""
    root = Path("/eval")
    assert canonical_report_dir(
        root, "full-depth",
        "VGGT") == root / "reports/benchmark_full_reports/full-depth/VGGT"
    assert canonical_report_dir(
        root, "full-pose",
        "PI3_X") == (root /
                     "reports/benchmark_full_pose_reports/full-pose/PI3_X")
    assert canonical_report_dir(
        root, "full-slam", "COME",
        "long") == (root /
                    "reports/benchmark_full_slam_reports/full-slam-long/COME")
    with pytest.raises(ValueError):
        canonical_report_dir(root, "sampled", "VGGT")


@pytest.mark.parametrize("phase", ["infer", "eval"])
@pytest.mark.parametrize("report_dir", [
    "{eval}/reports/pi3x_fix_20261013/full-depth/VGGT",
    "{eval}/reports/full-depth/VGGT"
])
def test_engine_refuses_full_suite_report_dir_outside_the_canonical_folder(
        tmp_path, phase, report_dir):
    """Reruns must not scatter results into retry/fix folders: the engine exits 8 before doing any work."""
    engine = Path(
        __file__).resolve().parents[1] / "spatialarena/templates/_engine"
    env = {
        "HOME": os.environ.get("HOME", "/tmp"),
        "SPATIALARENA_ROOT": str(
            tmp_path / "root"
        ),  # the engine creates its cache folders below the root
        "PATH": "/usr/bin:/bin",
        "METHOD": "VGGT",
        "BENCHMARK_SUITE": "full-depth",
        "RUN_ID": "retry_20261013",
        "BENCHMARK_AUX_ROOT": str(tmp_path / "aux"),
        "BENCHMARK_EVAL_ROOT": str(tmp_path / "eval"),
        "REPORT_DIR": report_dir.format(eval=tmp_path / "eval"),
    }
    result = subprocess.run(
        ["bash", str(engine / f"run_method_benchmark_{phase}.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False)
    assert result.returncode == 8, result.stderr[-2000:]
    assert "benchmark_full_reports/full-depth/VGGT" in result.stderr
