# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spatialarena import BenchmarkEvaluator, evaluator
from spatialarena.evaluator import EvaluationPlan, ValidationReport


def _bench(tmp_path, **kwargs):
    return BenchmarkEvaluator(
        experiment_root=tmp_path,
        experiment_name="demo",
        **kwargs,
    )


def _vggt_with_checkpoint(tmp_path):
    """VGGT with a stand-in checkpoint: validate() checks that the file exists."""
    checkpoint = tmp_path / "vggt.pt"
    checkpoint.touch()
    return {"name": "VGGT", "env": {"VGGT_CKPT_PATH": str(checkpoint)}}


def test_singleton_is_a_benchmark_evaluator():
    assert isinstance(evaluator, BenchmarkEvaluator)


def test_constructor_defaults_job_system_to_slurm():
    bench = BenchmarkEvaluator()
    assert bench.job_system == "slurm"


def test_constructor_rejects_unknown_job_system():
    with pytest.raises(ValueError):
        BenchmarkEvaluator(job_system="not-a-real-system")


def test_select_job_system_updates_state_and_legacy_alias_works():
    bench = BenchmarkEvaluator()
    bench.select_job_system("native")
    assert bench.job_system == "native"
    bench.set_job_sys("slurm")
    assert bench.job_system == "slurm"
    with pytest.raises(ValueError):
        bench.select_job_system("bogus")


def test_select_datasets_requires_task_and_dataset():
    bench = BenchmarkEvaluator()
    with pytest.raises(ValueError):
        bench.select_datasets([{"task": "MonoDepth"}])
    with pytest.raises(ValueError):
        bench.select_datasets([{"dataset": "marigold"}])


def test_select_datasets_replace_vs_append():
    bench = BenchmarkEvaluator()
    bench.select_datasets([{"task": "MonoDepth", "dataset": "marigold"}])
    bench.select_datasets([{"task": "SfM", "dataset": "eth3d"}])
    assert bench._selected_datasets == [{"task": "SfM", "dataset": "eth3d"}]  # pylint: disable=protected-access

    bench.select_datasets([{
        "task": "MonoDepth",
        "dataset": "marigold"
    }],
                          append=True)
    assert bench._selected_datasets == [  # pylint: disable=protected-access
        {
            "task": "SfM",
            "dataset": "eth3d"
        },
        {
            "task": "MonoDepth",
            "dataset": "marigold"
        },
    ]


def test_select_models_rejects_unknown_name():
    bench = BenchmarkEvaluator()
    with pytest.raises(KeyError):
        bench.select_models([{"name": "NotARealMethod"}])


def test_select_models_accepts_alias():
    bench = BenchmarkEvaluator()
    bench.select_models([{"name": "PI3"}])  # alias for PI3_X
    assert bench._selected_models == [{"name": "PI3"}]  # pylint: disable=protected-access


def test_select_models_custom_adapter_requires_name_track_command():
    bench = BenchmarkEvaluator()
    with pytest.raises(ValueError):
        bench.select_models([{
            "name": "Custom",
            "command": "python x.py"
        }])  # missing track
    bench.select_models([{
        "name": "Custom",
        "track": "rgb",
        "command": "python x.py"
    }])
    assert bench._selected_models[0]["command"] == "python x.py"  # pylint: disable=protected-access


def test_select_runs_requires_task_dataset_model():
    bench = BenchmarkEvaluator()
    with pytest.raises(ValueError):
        bench.select_runs([{"task": "MonoDepth", "dataset": "marigold"}])
    bench.select_runs([{
        "task": "MonoDepth",
        "dataset": "marigold",
        "model": "VGGT"
    }])
    # pylint: disable-next=protected-access
    assert bench._selected_runs == [{
        "task": "MonoDepth",
        "dataset": "marigold",
        "model": "VGGT"
    }]


def test_reset_clears_selection_but_keeps_object_reusable():
    bench = BenchmarkEvaluator(experiment_root="/tmp", experiment_name="x")
    bench.select_datasets([{"task": "MonoDepth", "dataset": "marigold"}])
    bench.select_models([{"name": "VGGT"}])
    bench.reset()
    assert bench._selected_datasets == []  # pylint: disable=protected-access
    assert bench._selected_models == []  # pylint: disable=protected-access
    assert bench._selected_runs is None  # pylint: disable=protected-access
    # constructor values are NOT wiped by reset() (matches doc: "keeps the object reusable")
    assert bench.experiment_name == "x"


def test_plan_requires_experiment_name():
    """plan() refuses to run without an experiment name (it names the scripts folder)."""
    bench = BenchmarkEvaluator()
    with pytest.raises(ValueError):
        bench.plan()


def test_job_scripts_default_to_the_benchmark_root():
    """Without experiment_root, generated scripts go to <benchmark_root>/_eval_job_scripts/<name>/."""
    bench = BenchmarkEvaluator(benchmark_root="/bench", experiment_name="run")
    assert bench._require_output_root() == Path("/bench/_eval_job_scripts/run")  # pylint: disable=protected-access


def test_plan_default_product_filters_by_task_track_matrix(tmp_path):
    bench = _bench(tmp_path)
    bench.select_datasets([
        {
            "task": "MonoDepth",
            "dataset": "marigold"
        },
        {
            "task": "MVRecon",
            "dataset": "dtu"
        },
    ])
    bench.select_models([{"name": "PI3_X", "track": "rgb_cameras"}])
    plan = bench.plan()
    assert isinstance(plan, EvaluationPlan)
    # MonoDepth+rgb_cameras is excluded by the matrix even though PI3_X supports rgb_cameras.
    assert plan.selected_runs == [{
        "task": "MVRecon",
        "dataset": "dtu",
        "model": "PI3_X",
        "track": "rgb_cameras"
    }]


def test_plan_runs_mvrecon_rgb_depth_only_on_datasets_with_sparse_gt_depth(
        tmp_path):
    """MVRecon rgb_depth is planned on DTU (sparse GT depth), not on HiRoom (dense GT depth: nothing left)."""
    bench = _bench(tmp_path)
    bench.select_datasets([{
        "task": "MVRecon",
        "dataset": "dtu"
    }, {
        "task": "MVRecon",
        "dataset": "hiroom"
    }])
    bench.select_models([{
        "name": "PI3_X",
        "track": "rgb_depth"
    }, {
        "name": "PI3_X",
        "track": "rgb"
    }])
    runs = [(run["dataset"], run["track"]) for run in bench.plan().selected_runs
           ]
    assert runs == [("dtu", "rgb_depth"), ("dtu", "rgb"), ("hiroom", "rgb")]


def test_plan_default_track_is_rgb_when_unspecified(tmp_path):
    bench = _bench(tmp_path)
    bench.select_datasets([{"task": "SfM", "dataset": "eth3d"}])
    bench.select_models([{"name": "VGGT"}])
    plan = bench.plan()
    assert plan.selected_runs == [{
        "task": "SfM",
        "dataset": "eth3d",
        "model": "VGGT",
        "track": "rgb"
    }]


def test_plan_expands_slam_into_its_range_tracks(tmp_path):
    bench = _bench(tmp_path)
    bench.select_datasets([{"task": "SLAM", "dataset": "tum_dynamics"}])
    bench.select_models([{
        "name": "PI3_X"
    }, {
        "name": "VGGT"
    }, {
        "name": "MAPANYTHING"
    }, {
        "name": "VGGT_SLAM"
    }, {
        "name": "GENCEPTION_POSE"
    }])
    runs = {
        (run["model"], run["slam_range"]) for run in bench.plan().selected_runs
    }
    # PI3_X and VGGT-SLAM fit both ranges, VGGT (400 frames) only the short one; MapAnything (budget below both) is
    # attempted on both; GenCeption-Pose is not on the SLAM track.
    assert runs == {("PI3_X", "short"), ("PI3_X", "long"), ("VGGT", "short"),
                    ("MAPANYTHING", "short"), ("MAPANYTHING", "long"),
                    ("VGGT_SLAM", "short"), ("VGGT_SLAM", "long")}

    bench.select_datasets([{
        "task": "SLAM",
        "dataset": "tum_dynamics",
        "slam_range": "long"
    }])
    assert {run["model"] for run in bench.plan().selected_runs
           } == {"PI3_X", "MAPANYTHING", "VGGT_SLAM"}


def test_plan_is_read_only_and_does_not_mutate_selection(tmp_path):
    bench = _bench(tmp_path)
    bench.select_datasets([{"task": "MonoDepth", "dataset": "marigold"}])
    bench.select_models([{"name": "VGGT"}])
    before = list(bench._selected_datasets)  # pylint: disable=protected-access
    bench.plan()
    bench.plan()
    assert bench._selected_datasets == before  # pylint: disable=protected-access


def test_plan_respects_explicit_select_runs_over_cartesian_product(tmp_path):
    bench = _bench(tmp_path)
    bench.select_datasets([{"task": "MonoDepth", "dataset": "marigold"}])
    bench.select_models([{"name": "VGGT"}, {"name": "DA3-G"}])
    bench.select_runs([{"task": "SfM", "dataset": "eth3d", "model": "DA3-G"}])
    plan = bench.plan()
    assert plan.selected_runs == [{
        "task": "SfM",
        "dataset": "eth3d",
        "model": "DA3-G"
    }]


def test_validate_returns_structured_report(tmp_path, benchmark_root):
    bench = _bench(tmp_path, benchmark_root=benchmark_root)
    bench.select_datasets([{"task": "MonoDepth", "dataset": "marigold"}])
    bench.select_models([_vggt_with_checkpoint(tmp_path)])
    report = bench.validate()
    assert isinstance(report, ValidationReport)
    assert report.ok is True
    assert report.summary["run_count"] == 1


def test_validate_errors_on_unsupported_track(tmp_path):
    bench = _bench(tmp_path)
    bench.select_models([{
        "name": "VGGT",
        "track": "rgb_cameras"
    }])  # VGGT is rgb-only
    report = bench.validate()
    assert report.ok is False
    assert any(
        "does not support track" in issue.message for issue in report.issues)


def test_validate_errors_when_nothing_survives_task_track_filtering(tmp_path):
    bench = _bench(tmp_path)
    bench.select_datasets([{"task": "MonoDepth", "dataset": "marigold"}])
    bench.select_models([{
        "name": "PI3_X",
        "track": "rgb_cameras"
    }])  # excluded by the matrix for MonoDepth
    report = bench.validate()
    assert report.ok is False
    assert any("nothing would be evaluated" in issue.message
               for issue in report.issues)


def test_validate_warns_on_empty_selection(tmp_path):
    bench = _bench(tmp_path)
    report = bench.validate()
    assert any(issue.scope == "dataset" and issue.level == "warning"
               for issue in report.issues)
    assert any(issue.scope == "model" and issue.level == "warning"
               for issue in report.issues)


def test_validate_skips_checkpoint_checks_for_custom_adapters(tmp_path):
    bench = _bench(tmp_path)
    bench.select_models([{
        "name": "Custom",
        "track": "rgb",
        "command": "python x.py"
    }])
    report = bench.validate()
    assert not any(
        issue.scope == "model" and "checkpoint" in issue.message.lower()
        for issue in report.issues)


def test_export_report_requires_a_finished_report(tmp_path):
    bench = _bench(tmp_path)
    with pytest.raises(ValueError):
        bench.export_report(tmp_path / "out")


def test_configure_raises_on_missing_file():
    bench = BenchmarkEvaluator()
    with pytest.raises(FileNotFoundError):
        bench.configure("/no/such/config.yaml")


def test_configure_raises_on_unsupported_extension(tmp_path):
    bad = tmp_path / "config.txt"
    bad.write_text("benchmark_root: /x", encoding="utf-8")
    bench = BenchmarkEvaluator()
    with pytest.raises(ValueError):
        bench.configure(bad)


def test_configure_loads_yaml_and_populates_state(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
job_system:
  type: native

benchmark_root: /tmp/benchmarks
experiment_root: /tmp/exp_root
experiment_name: benchmark_full

datasets:
  - task: MonoDepth
    dataset: marigold
  - task: SfM
    dataset: imc

models:
  - name: VGGT
  - name: PI3_X
    track: rgb_depth

report:
  output_path: ./report
""",
        encoding="utf-8",
    )
    bench = BenchmarkEvaluator()
    bench.configure(config_path)

    assert bench.job_system == "native"
    assert bench.benchmark_root == Path("/tmp/benchmarks")
    assert bench.experiment_root == Path("/tmp/exp_root")
    assert bench.experiment_name == "benchmark_full"
    assert bench._selected_datasets == [  # pylint: disable=protected-access
        {
            "task": "MonoDepth",
            "dataset": "marigold"
        },
        {
            "task": "SfM",
            "dataset": "imc"
        },
    ]
    # pylint: disable-next=protected-access
    assert bench._selected_models == [{
        "name": "VGGT"
    }, {
        "name": "PI3_X",
        "track": "rgb_depth"
    }]
    assert bench.report_config == {"output_path": "./report"}


def test_configure_loads_json(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({
            "experiment_name": "from_json",
            "models": [{
                "name": "VGGT"
            }]
        }),
        encoding="utf-8",
    )
    bench = BenchmarkEvaluator(experiment_root=tmp_path)
    bench.configure(config_path)
    assert bench.experiment_name == "from_json"
    assert bench._selected_models == [{"name": "VGGT"}]  # pylint: disable=protected-access


def test_configure_explicit_constructor_value_wins_over_file(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("experiment_name: from_file\n", encoding="utf-8")
    bench = BenchmarkEvaluator(experiment_name="from_constructor")
    bench.configure(config_path)
    assert bench.experiment_name == "from_constructor"


def test_configure_call_time_override_wins_over_file(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("experiment_name: from_file\n", encoding="utf-8")
    bench = BenchmarkEvaluator()
    bench.configure(config_path, experiment_name="from_override")
    assert bench.experiment_name == "from_override"


def test_configure_env_var_used_when_no_explicit_or_file_value(
        tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("experiment_name: demo\n", encoding="utf-8")
    monkeypatch.setenv("BENCHMARK_EVAL_ROOT", "/env/eval/root")
    bench = BenchmarkEvaluator()
    bench.configure(config_path)
    assert bench.benchmark_eval_root == Path("/env/eval/root")


@pytest.mark.parametrize("evaluate_only", [False, True])
# pylint: disable-next=unused-argument
def test_native_evaluate_runs_infer_then_eval_as_subprocesses(
        tmp_path, monkeypatch, evaluate_only):
    import sys
    from unittest.mock import MagicMock, patch

    from spatialarena.script_generator import GeneratedRunScripts

    # spatialarena/__init__.py binds the module-level `evaluator` singleton
    # over the `spatialarena.evaluator` submodule attribute, so
    # patch("spatialarena.evaluator.subprocess...")'s dotted lookup would
    # resolve onto the singleton instance instead of the module. Go through
    # sys.modules directly to get the real module object.
    evaluator_module = sys.modules["spatialarena.evaluator"]

    bench = _bench(tmp_path, job_system="native")
    bench.evaluate_only = evaluate_only
    bench.select_runs([{
        "task": "MonoDepth",
        "dataset": "marigold",
        "model": "VGGT",
        "scenes": ["nyuv2"]
    }])

    fake_generated = GeneratedRunScripts(
        work_item={
            "task": "MonoDepth",
            "dataset": "marigold",
            "model": "VGGT",
            "track": "rgb",
            "suite": "full-depth"
        },
        case_file=Path("/fake/cases.txt"),
        infer_script=Path("/fake/infer.sh"),
        eval_script=Path("/fake/eval.sh"),
        report_dir=tmp_path / "report",
        exp_name="benchmark_full-depth_VGGT",
    )
    (tmp_path / "report").mkdir(parents=True, exist_ok=True)
    # run_method_case_list.sh names this summary_<case_file_stem>.tsv, not
    # plain summary.tsv, whenever CASE_FILE isn't the suite's own canonical
    # default -- which is always true for a generated case file (see
    # evaluator._report_tsv_paths()).
    (tmp_path / "report" / "summary_cases.tsv").write_text(
        "method\ttask\tdataset\tscene\tinterval\torder\toutput_path\treport_dir\tprediction_retention\n"
        "VGGT\tMonoDepth\tmarigold\tnyuv2\t1\tdefault\t/out\t/report\tkept\n",
        encoding="utf-8",
    )

    calls = []

    # pylint: disable-next=unused-argument
    def fake_popen(
            cmd,
            env=None,
            **kwargs
    ):  # native mode polls Popen.wait() to update its progress bars
        calls.append(cmd)
        process = MagicMock(returncode=0)
        process.wait.return_value = 0
        return process

    with patch.object(BenchmarkEvaluator, "_generate_for_work_item",
                      return_value=(fake_generated.work_item, fake_generated)), \
         patch.object(evaluator_module.subprocess, "Popen", side_effect=fake_popen):
        report = bench.evaluate(progress="none")

    # infer before eval; evaluate_only (predictions already exist) runs only the evaluation phase
    assert [c[1] for c in calls] == (["/fake/eval.sh"] if evaluate_only else
                                     ["/fake/infer.sh", "/fake/eval.sh"])
    assert report.summary["status"] == "finished"
    assert report.ok is True
    assert report.scene_results == [{
        "method": "VGGT",
        "task": "MonoDepth",
        "dataset": "marigold",
        "scene": "nyuv2",
        "status": "passed",
        "report_dir": str(tmp_path / "report"),
    }]
    assert report.dataset_results == [{
        "task": "MonoDepth",
        "dataset": "marigold",
        "passed": 1,
        "failed": 0,
        "skipped": 0
    }]


def test_native_evaluate_skips_eval_when_infer_fails(tmp_path):
    import sys
    from unittest.mock import MagicMock, patch

    from spatialarena.script_generator import GeneratedRunScripts

    evaluator_module = sys.modules["spatialarena.evaluator"]
    bench = _bench(tmp_path, job_system="native")
    bench.select_runs([{
        "task": "MonoDepth",
        "dataset": "marigold",
        "model": "VGGT",
        "scenes": ["nyuv2"]
    }])

    fake_generated = GeneratedRunScripts(
        work_item={
            "task": "MonoDepth",
            "dataset": "marigold",
            "model": "VGGT",
            "track": "rgb",
            "suite": "full-depth"
        },
        case_file=Path("/fake/cases.txt"),
        infer_script=Path("/fake/infer.sh"),
        eval_script=Path("/fake/eval.sh"),
        report_dir=tmp_path / "report_missing",
        exp_name="benchmark_full-depth_VGGT",
    )

    calls = []

    # pylint: disable-next=unused-argument
    def fake_popen(
            cmd,
            env=None,
            **kwargs
    ):  # native mode polls Popen.wait() to update its progress bars
        calls.append(cmd)
        process = MagicMock(returncode=1)
        process.wait.return_value = 1
        return process

    with patch.object(BenchmarkEvaluator, "_generate_for_work_item",
                      return_value=(fake_generated.work_item, fake_generated)), \
         patch.object(evaluator_module.subprocess, "Popen", side_effect=fake_popen):
        report = bench.evaluate(progress="none")

    assert len(calls) == 1  # eval never invoked
    assert report.ok is False
    assert report.summary["failed"] == 1


def test_native_evaluate_rejects_max_in_flight(tmp_path):
    bench = _bench(tmp_path, job_system="native")
    bench.select_runs([{
        "task": "MonoDepth",
        "dataset": "marigold",
        "model": "VGGT"
    }])
    with pytest.raises(ValueError):
        bench.evaluate(max_in_flight=5)


def test_evaluate_refuses_when_validation_fails(tmp_path):
    bench = _bench(tmp_path)
    bench.select_models([{
        "name": "VGGT",
        "track": "rgb_cameras"
    }])  # VGGT is rgb-only -> validate() error
    with pytest.raises(ValueError):
        bench.evaluate()


def test_collect_without_a_report_raises():
    bench = BenchmarkEvaluator()
    with pytest.raises(ValueError):
        bench.collect()


def test_collect_with_no_jobs_returns_report_unchanged():
    from spatialarena.evaluator import EvaluationReport

    plan = EvaluationPlan(config={},
                          selected_datasets=[],
                          selected_models=[],
                          selected_runs=[],
                          output_root=Path("/tmp/x"))
    report = EvaluationReport(ok=True,
                              config={},
                              plan=plan,
                              summary={"jobs": []})
    bench = BenchmarkEvaluator()
    assert bench.collect(report) is report


def test_discover_models_enriches_with_task_tracks():
    bench = BenchmarkEvaluator()
    result = bench.discover_models(["PI3_X"])
    assert result["summary"]["method_count"] == 1
    method = result["methods"][0]
    assert method["name"] == "PI3_X"
    assert method["task_tracks"]["MonoDepth"] == ["rgb", "rgb_masked_depth"]
    assert set(method["task_tracks"]["MVRecon"]) == {
        "rgb", "rgb_cameras", "rgb_depth", "rgb_masked_depth",
        "rgb_masked_depth_cameras", "rgb_sparse_depth",
        "rgb_sparse_depth_cameras"
    }
    assert set(method["task_tracks"]["SfM"]) == {
        "rgb", "rgb_depth", "rgb_masked_depth", "rgb_masked_depth_cameras",
        "rgb_sparse_depth", "rgb_sparse_depth_cameras"
    }


def test_discover_models_filters_by_model_types():
    bench = BenchmarkEvaluator()
    result = bench.discover_models(["VGGT", "PI3_X"])
    assert {m["name"] for m in result["methods"]} == {"VGGT", "PI3_X"}


def test_custom_adapter_env_none_for_registered_method():
    from spatialarena.evaluator import _custom_adapter_env

    assert _custom_adapter_env({
        "task": "MonoDepth",
        "dataset": "marigold",
        "model": "VGGT",
        "track": "rgb"
    }) is None


def test_custom_adapter_env_builds_overrides():
    from spatialarena.evaluator import _custom_adapter_env

    work_item = {
        "task": "MonoDepth",
        "dataset": "marigold",
        "model": "MyCustomModel",
        "track": "rgb",
        "command": "python infer.py",
        "resolution": 336,
        "code_path": "/code/mymodel",
        "ckpt_path": "/ckpt/mymodel.pt",
        "env": {
            "FOO": "bar"
        },
    }
    env = _custom_adapter_env(work_item)
    assert env["METHOD_COMMAND"] == "python infer.py"
    assert env["RESOLUTION"] == "336"
    assert env["_CUSTOM_METHOD_CODE_PATH"] == "/code/mymodel"
    assert env["METHOD_CODE_ENV"] == "_CUSTOM_METHOD_CODE_PATH"
    assert env["_CUSTOM_METHOD_CKPT_PATH"] == "/ckpt/mymodel.pt"
    assert env["METHOD_CKPT_ENV"] == "_CUSTOM_METHOD_CKPT_PATH"
    assert env["FOO"] == "bar"


def test_report_tsv_paths_uses_case_file_stem_suffix():
    from spatialarena.evaluator import _report_tsv_paths

    job = {
        "report_dir": "/report",
        "case_file": "/scripts/cases/VGGT_full-depth_MonoDepth.txt",
        "work_item": {
            "suite": "full-depth"
        },
    }
    summary, failures = _report_tsv_paths(job)
    assert summary == Path("/report/summary_VGGT_full-depth_MonoDepth.tsv")
    assert failures == Path(
        "/report/case_failures_VGGT_full-depth_MonoDepth.tsv")


def test_report_tsv_paths_no_suffix_when_case_file_matches_suite_default():
    from spatialarena.evaluator import _report_tsv_paths

    job = {
        "report_dir": "/report",
        "case_file": "/scripts/test/full_standard_depth_cases.txt",
        "work_item": {
            "suite": "full-depth"
        },
    }
    summary, failures = _report_tsv_paths(job)
    assert summary == Path("/report/summary.tsv")
    assert failures == Path("/report/case_failures.tsv")


def test_report_tsv_paths_pose_suite_default():
    from spatialarena.evaluator import _report_tsv_paths

    job = {
        "report_dir": "/report",
        "case_file": "/scripts/test/full_standard_sfm_cases.txt",
        "work_item": {
            "suite": "full-pose"
        },
    }
    summary, _ = _report_tsv_paths(job)
    assert summary == Path("/report/summary.tsv")


def test_read_tsv_data_rows_skips_header(tmp_path):
    from spatialarena.evaluator import _read_tsv_data_rows

    path = tmp_path / "summary.tsv"
    path.write_text(
        "method\ttask\tdataset\tscene\tinterval\torder\toutput_path\treport_dir\tprediction_retention\n"
        "VGGT\tMonoDepth\tmarigold\tnyuv2\t1\tdefault\t/out\t/report\tkept\n"
        "VGGT\tMonoDepth\tmarigold\tkitti\t1\tdefault\t/out2\t/report\tkept\n",
        encoding="utf-8",
    )
    rows = _read_tsv_data_rows(path)
    assert len(rows) == 2
    assert rows[0][0] == "VGGT"
    assert rows[0][3] == "nyuv2"
    assert rows[1][3] == "kitti"


def test_read_tsv_data_rows_header_only_yields_nothing(tmp_path):
    from spatialarena.evaluator import _read_tsv_data_rows

    path = tmp_path / "summary.tsv"
    path.write_text(
        "method\ttask\tdataset\tscene\tinterval\torder\toutput_path\treport_dir\tprediction_retention\n",
        encoding="utf-8")
    assert _read_tsv_data_rows(path) == []


def _fake_report(tmp_path, plan, *, report_dirs=()):
    from spatialarena.evaluator import EvaluationReport

    jobs = [{
        "report_dir": str(tmp_path / d),
        "work_item": {
            "task": "MonoDepth",
            "dataset": "marigold"
        }
    } for d in report_dirs]
    return EvaluationReport(
        ok=True,
        config=plan.config,
        plan=plan,
        scene_results=[{
            "method": "VGGT",
            "task": "MonoDepth",
            "dataset": "marigold",
            "scene": "nyuv2",
            "status": "passed"
        }],
        dataset_results=[{
            "task": "MonoDepth",
            "dataset": "marigold",
            "passed": 1,
            "failed": 0
        }],
        summary={
            "status": "finished",
            "jobs": jobs,
            "passed": 1,
            "failed": 0
        },
        report_dir=plan.output_root,
    )


def test_export_report_writes_summary_plan_validation_json(
        tmp_path, benchmark_root):
    bench = _bench(tmp_path, benchmark_root=benchmark_root)
    bench.select_datasets([{"task": "MonoDepth", "dataset": "marigold"}])
    bench.select_models([_vggt_with_checkpoint(tmp_path)])
    bench.validate()
    plan = bench.plan()
    # pylint: disable-next=protected-access
    bench._last_report = _fake_report(
        tmp_path, plan)  # no report_dirs -> skip the aggregated report step

    out_dir = tmp_path / "exported"
    result = bench.export_report(out_dir)

    assert result == out_dir
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["ok"] is True
    assert summary["passed"] == 1
    assert summary["dataset_results"] == [{
        "task": "MonoDepth",
        "dataset": "marigold",
        "passed": 1,
        "failed": 0
    }]

    plan_json = json.loads((out_dir / "plan.json").read_text())
    assert plan_json["selected_runs"] == plan.selected_runs

    validation_json = json.loads((out_dir / "validation.json").read_text())
    assert validation_json["ok"] is True

    # No report roots -> the aggregated report/ subdir isn't created.
    assert not (out_dir / "report").exists()


def test_export_report_writes_gather_metrics_script(tmp_path):
    from unittest.mock import patch

    bench = _bench(tmp_path)
    bench.select_runs([{
        "task": "MonoDepth",
        "dataset": "marigold",
        "model": "VGGT"
    }])
    plan = bench.plan()
    report_dirs = ("reports/full-depth/VGGT",)
    bench._last_report = _fake_report(tmp_path, plan, report_dirs=report_dirs)  # pylint: disable=protected-access

    with patch.object(BenchmarkEvaluator,
                      "_write_aggregated_report") as mock_aggregate:
        bench.export_report(tmp_path / "exported")

    mock_aggregate.assert_called_once()
    called_roots = mock_aggregate.call_args.args[1]
    assert called_roots == [str(tmp_path / "reports/full-depth/VGGT")]

    script_path = plan.output_root / "scripts" / "gather_metrics.sh"
    assert script_path.exists()
    text = script_path.read_text()
    assert str(tmp_path / "reports/full-depth/VGGT") in text
    assert "summarize_benchmark_reports.py" in text
    assert "benchmark_analyzer.py" in text
    assert "generate_benchmark_pdf_report.py" in text


def test_write_gather_metrics_script_handles_no_report_roots(tmp_path):
    bench = _bench(tmp_path)
    output_root = bench._require_output_root()  # pylint: disable=protected-access
    script_path = bench._write_gather_metrics_script(output_root, [])  # pylint: disable=protected-access
    assert script_path.exists()
    assert "no report roots recorded yet" in script_path.read_text()


def test_list_supported_datasets_delegates_to_download_module():
    bench = BenchmarkEvaluator()
    result = bench.list_supported_datasets()
    assert isinstance(result, list)
    assert any(
        e["task"] == "MonoDepth" and e["dataset"] == "marigold" for e in result)


def test_list_supported_models_delegates_to_download_module():
    bench = BenchmarkEvaluator()
    result = bench.list_supported_models()
    assert isinstance(result, list)
    assert any(e["name"] == "PI3_X" for e in result)


def test_download_all_supported_datasets_uses_own_benchmark_root(tmp_path):
    from unittest.mock import patch

    bench = BenchmarkEvaluator(benchmark_root=tmp_path)
    with patch("spatialarena.download.download_all_supported_datasets"
              ) as mock_download:
        mock_download.return_value = [{
            "task": "VideoDepth",
            "dataset": "sintel",
            "status": "ok"
        }]
        result = bench.download_all_supported_datasets(datasets=[("VideoDepth",
                                                                  "sintel")])
    mock_download.assert_called_once()
    assert mock_download.call_args.kwargs["benchmark_root"] == tmp_path
    assert result == [{
        "task": "VideoDepth",
        "dataset": "sintel",
        "status": "ok"
    }]


def test_download_all_supported_models_forwards_methods_filter():
    from unittest.mock import patch

    bench = BenchmarkEvaluator()
    with patch("spatialarena.download.download_all_supported_models"
              ) as mock_download:
        mock_download.return_value = [{"name": "PI3_X", "status": "ok"}]
        result = bench.download_all_supported_models(methods=["PI3_X"],
                                                     progress="none")
    mock_download.assert_called_once()
    kwargs = dict(mock_download.call_args.kwargs)
    assert callable(kwargs.pop("on_event"))  # progress hook (bars/board/none)
    assert kwargs == {
        "methods": ["PI3_X"],
        "shared_checkpoint_root": bench.checkpoint_root,
        "shared_hf_home": bench.hf_home,
        "method_code_root": bench.method_code_root,
    }
    assert result == [{"name": "PI3_X", "status": "ok"}]


def test_list_supported_method_code_delegates_to_download_module():
    bench = BenchmarkEvaluator()
    result = bench.list_supported_method_code()
    assert isinstance(result, list)
    assert any(
        e["name"] == "PI3_X" and e["status"] == "automated" for e in result)


def test_download_all_supported_method_code_uses_own_method_code_root(tmp_path):
    from unittest.mock import patch

    bench = BenchmarkEvaluator(method_code_root=tmp_path)
    with patch("spatialarena.download.download_all_supported_method_code"
              ) as mock_download:
        mock_download.return_value = [{"name": "PI3_X", "status": "ok"}]
        result = bench.download_all_supported_method_code(methods=["PI3_X"])
    mock_download.assert_called_once()
    assert mock_download.call_args.kwargs["method_code_root"] == tmp_path
    assert mock_download.call_args.kwargs["methods"] == ["PI3_X"]
    assert result == [{"name": "PI3_X", "status": "ok"}]


def test_shipped_example_config_loads_and_plans(tmp_path):
    config_path = Path(__file__).resolve(
    ).parents[1] / "configs" / "benchmark" / "evaluation_example.yaml"
    bench = BenchmarkEvaluator()
    bench.configure(config_path, experiment_root=tmp_path)
    assert bench.job_system == "slurm"
    assert bench.report_config == {"output_path": "./report"}
    runs = bench.plan().selected_runs
    assert {(run["task"], run["model"], run["track"]) for run in runs} >= {
        ("MonoDepth", "VGGT", "rgb"),
        ("SfM", "PI3_X", "rgb_depth"),
    }
    assert all(
        run["track"] == "rgb" for run in runs if run["task"] == "MonoDepth")


def test_export_report_defaults_to_config_output_path(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"report:\n  output_path: {tmp_path / 'from_config'}\n",
        encoding="utf-8")
    bench = BenchmarkEvaluator()
    with pytest.raises(ValueError, match="output_path"):
        bench.export_report()
    bench.configure(config_path)
    # The path resolves from the config; without a finished report it then stops at the next check.
    with pytest.raises(ValueError, match="finished report"):
        bench.export_report()


def test_mvrecon_cases_skip_scenes_without_gt_geometry():
    """MVRecon scenes without GT geometry (ETH3D meadow/terrace) get no case; SfM keeps them."""
    import importlib  # pylint: disable=import-outside-toplevel
    from unittest.mock import patch

    scenes = [{
        "scene": "office",
        "has_gt_geometry": True
    }, {
        "scene": "meadow",
        "has_gt_geometry": False
    }]
    bench = BenchmarkEvaluator()
    # The module, not the package attribute: spatialarena.evaluator is also the package's evaluator instance.
    module = importlib.import_module("spatialarena.evaluator")
    with patch.object(module,
                      "describe_benchmark_scenes",
                      return_value=[{
                          "scenes": scenes
                      }]):
        # pylint: disable-next=protected-access
        mvrecon = bench._resolve_case_entries({
            "task": "MVRecon",
            "dataset": "eth3d"
        })
        sfm = bench._resolve_case_entries({"task": "SfM", "dataset": "eth3d"})  # pylint: disable=protected-access
    assert [entry.scene for entry in mvrecon] == ["office"]
    assert [entry.scene for entry in sfm] == ["office", "meadow"]


def test_model_slurm_resources_override_the_config_per_phase(
        tmp_path, benchmark_root):
    """slurm_resources stack config < dataset item < model item, option by option, in the matching runs only."""
    checkpoint = _vggt_with_checkpoint(tmp_path)["env"]["VGGT_CKPT_PATH"]
    config = tmp_path / "cfg.yaml"
    config.write_text(
        f"experiment_root: {tmp_path}\nexperiment_name: exp\njob_system: slurm\n"
        f"benchmark_root: {benchmark_root}\n"
        "slurm_resources: {eval: {mem: 64G, cpus-per-task: 8}}\n"
        "datasets:\n  - {task: SLAM, dataset: tum_dynamics, slam_range: short,"
        " slurm_resources: {eval: {mem: 32G}, infer: {time: '1:00:00'}}}\n"
        "  - {task: SfM, dataset: eth3d}\n"
        f"models:\n  - {{name: VGGT, env: {{VGGT_CKPT_PATH: {checkpoint}}}}}\n"
        "  - {name: VGGT_LONG, slurm_resources: {eval: {mem: 128G}}}\n",
        encoding="utf-8",
    )
    bench = BenchmarkEvaluator()
    bench.configure(config)
    scripts = {
        (item["task"], item["model"]): generated
        for item, generated in bench.generate()
    }
    sfm = scripts[("SfM", "VGGT")].eval_script.read_text(encoding="utf-8")
    slam = scripts[("SLAM", "VGGT")].eval_script.read_text(encoding="utf-8")
    long = scripts[("SLAM",
                    "VGGT_LONG")].eval_script.read_text(encoding="utf-8")
    assert "#SBATCH --mem=64G" in sfm and "#SBATCH --cpus-per-task=8" in sfm  # config level
    assert "#SBATCH --mem=32G" in slam and "#SBATCH --cpus-per-task=8" in slam  # the dataset item on top
    assert "#SBATCH --mem=128G" in long and "#SBATCH --cpus-per-task=8" in long  # the model item on top of both
    assert "#SBATCH --time=1:00:00" in scripts[(
        "SLAM", "VGGT_LONG")].infer_script.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="takes the phases"):
        bench.select_models([{
            "name": "VGGT",
            "slurm_resources": {
                "align": {
                    "mem": "1G"
                }
            }
        }])


_SUMMARY_HEADER = "method\ttask\tdataset\tscene\tinterval\torder\toutput_path\treport_dir\tprediction_retention\n"


def _summary_row(scene, status):
    return f"VGGT\tMonoDepth\tmarigold\t{scene}\t1\tdefault\t/out\t/report\t{status}\n"


def test_read_job_results_counts_only_scored_cases_as_passed(tmp_path):
    from spatialarena.evaluator import _read_job_results

    (tmp_path / "summary_cases.tsv").write_text(
        _SUMMARY_HEADER + _summary_row("a", "kept") +
        _summary_row("b", "removed_after_eval") +
        _summary_row("c", "reused_existing_eval") +
        _summary_row("d", "missing_prediction_for_eval_phase") +
        _summary_row("e", "skipped_not_applicable") +
        _summary_row("f", "inference_ready"),
        encoding="utf-8",
    )
    job = {
        "report_dir": str(tmp_path),
        "case_file": str(tmp_path / "cases.txt"),
        "work_item": {
            "task": "MonoDepth",
            "dataset": "marigold",
            "model": "VGGT",
            "suite": "full-depth"
        }
    }
    bucket = {"passed": 0, "failed": 0, "skipped": 0}
    rows = _read_job_results(job, True, bucket)
    assert {
        row["scene"]: row["status"] for row in rows
    } == {
        "a": "passed",
        "b": "passed",
        "c": "passed",
        "d": "failed",
        "e": "skipped",
        "f": "failed"
    }
    assert bucket == {"passed": 3, "failed": 2, "skipped": 1}


def test_read_job_results_reports_a_run_that_was_never_generated():
    from spatialarena.evaluator import _read_job_results

    job = {
        "report_dir": None,
        "case_file": None,
        "dispatch_error": "no scenes found",
        "work_item": {
            "task": "SLAM",
            "dataset": "vbr",
            "model": "VGGT_LONG"
        }
    }
    bucket = {"passed": 0, "failed": 0, "skipped": 0}
    rows = _read_job_results(job, False, bucket)
    assert rows == [{
        "method": "VGGT_LONG",
        "task": "SLAM",
        "dataset": "vbr",
        "scene": None,
        "status": "failed",
        "report_dir": None,
        "note": "no scenes found"
    }]
    assert bucket["failed"] == 1


def test_collect_passes_an_evaluate_only_run_without_an_inference_job(
        tmp_path, monkeypatch):
    from spatialarena.evaluator import EvaluationReport

    (tmp_path / "summary_cases.tsv").write_text(_SUMMARY_HEADER +
                                                _summary_row("a", "kept"),
                                                encoding="utf-8")
    job = {
        "report_dir": str(tmp_path),
        "case_file": str(tmp_path / "cases.txt"),
        "infer_job_id": None,
        "eval_job_id": "42",
        "work_item": {
            "task": "MonoDepth",
            "dataset": "marigold",
            "model": "VGGT",
            "suite": "full-depth"
        }
    }
    plan = EvaluationPlan(config={},
                          selected_datasets=[],
                          selected_models=[],
                          selected_runs=[],
                          output_root=tmp_path)
    report = EvaluationReport(ok=True,
                              config={},
                              plan=plan,
                              summary={
                                  "status": "submitted",
                                  "jobs": [job]
                              })
    import sys

    # spatialarena/__init__.py binds the `evaluator` singleton over the submodule name; patch the module itself.
    monkeypatch.setattr(sys.modules["spatialarena.evaluator"], "sacct_states",
                        lambda ids: {"42": "COMPLETED"})
    collected = BenchmarkEvaluator().collect(report)
    assert collected.ok is True
    assert collected.dataset_results == [{
        "task": "MonoDepth",
        "dataset": "marigold",
        "passed": 1,
        "failed": 0,
        "skipped": 0
    }]


def test_run_label_and_scripts_default_the_track_to_rgb():
    from spatialarena.script_generator import run_label

    assert run_label({"suite": "full-depth", "model": "VGGT", "task": "MonoDepth", "dataset": "marigold"}) == \
        run_label({"suite": "full-depth", "model": "VGGT", "task": "MonoDepth", "dataset": "marigold", "track": "rgb"})
