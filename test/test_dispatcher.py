# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch
from spatialarena import BenchmarkEvaluator
from spatialarena.dispatcher import run_dispatcher
from spatialarena.script_generator import GeneratedRunScripts
from unittest.mock import MagicMock
from spatialarena.slurm_utils import sbatch_submit

# ---- dispatcher --------------------------------------------------------------------------------------------


def _fake_generate_for_work_item(self,
                                 work_item,
                                 run_id,
                                 case_parallelism=None):  # pylint: disable=unused-argument
    del case_parallelism
    method_id = work_item["model"]
    item_with_suite = dict(work_item, suite="full-depth")
    generated = GeneratedRunScripts(
        work_item=item_with_suite,
        case_file=Path(f"/fake/{method_id}_cases.txt"),
        infer_script=Path(f"/fake/{method_id}_infer.sh"),
        eval_script=Path(f"/fake/{method_id}_eval.sh"),
        report_dir=Path(f"/fake/reports/{method_id}"),
        exp_name=f"benchmark_full-depth_{method_id}",
    )
    return item_with_suite, generated


class _FakeSlurm:
    """Simulates sbatch/sacct: every submitted job completes on the very
    next sacct poll, so a 2-item, max_in_flight=1 run exercises the full
    pending -> infer -> eval -> done state machine deterministically."""

    def __init__(self):
        self.next_id = 1000
        self.submitted: list[str] = []
        self.job_ages: dict[str, int] = {}

    def sbatch(self, script, *, dependency=None):  # pylint: disable=unused-argument
        job_id = str(self.next_id)
        self.next_id += 1
        self.submitted.append(job_id)
        self.job_ages[job_id] = 0
        return job_id

    def sacct(self, job_ids):
        states = {}
        for job_id in job_ids:
            age = self.job_ages.get(job_id, 0)
            self.job_ages[job_id] = age + 1
            states[job_id] = "COMPLETED" if age >= 1 else "RUNNING"
        return states


def test_dispatcher_processes_all_items_respecting_max_in_flight(tmp_path):
    runs = [
        {
            "task": "MonoDepth",
            "dataset": "marigold",
            "model": "VGGT",
            "track": "rgb"
        },
        {
            "task": "MonoDepth",
            "dataset": "marigold",
            "model": "DA3-L",
            "track": "rgb"
        },
        {
            "task": "MonoDepth",
            "dataset": "marigold",
            "model": "PI3_X",
            "track": "rgb"
        },
    ]
    bench = BenchmarkEvaluator(experiment_root=tmp_path, experiment_name="demo")
    fake = _FakeSlurm()
    state_path = tmp_path / "dispatcher_state.json"

    with patch.object(BenchmarkEvaluator, "_generate_for_work_item", _fake_generate_for_work_item), \
         patch("spatialarena.dispatcher.sbatch_submit", side_effect=fake.sbatch), \
         patch("spatialarena.dispatcher.sacct_states", side_effect=fake.sacct), \
         patch("spatialarena.dispatcher.time.sleep", return_value=None):
        result = run_dispatcher(
            bench,
            selected_runs=runs,
            run_id="test-run",
            max_in_flight=1,
            infer_parallelism=4,
            eval_parallelism=6,
            poll_interval=0,
            state_path=state_path,
        )

    assert result == {"total": 3, "passed": 3, "failed": 0}
    # 1 infer + 1 eval job per run, 3 runs -> 6 jobs total.
    assert len(fake.submitted) == 6

    final_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert final_state["status"] == "finished"
    assert final_state["passed"] == 3
    assert final_state["failed"] == 0
    assert all(item["phase"] == "done" for item in final_state["items"])


def test_dispatcher_never_exceeds_max_in_flight_concurrent_jobs(tmp_path):
    runs = [{
        "task": "MonoDepth",
        "dataset": "marigold",
        "model": name,
        "track": "rgb"
    } for name in ("VGGT", "DA3-L", "DA3-G", "PI3_X", "FASTVGGT")]
    bench = BenchmarkEvaluator(experiment_root=tmp_path, experiment_name="demo")
    fake = _FakeSlurm()
    max_seen_active = {"value": 0}
    real_sacct = fake.sacct

    def tracking_sacct(job_ids):
        max_seen_active["value"] = max(max_seen_active["value"], len(job_ids))
        return real_sacct(job_ids)

    with patch.object(BenchmarkEvaluator, "_generate_for_work_item", _fake_generate_for_work_item), \
         patch("spatialarena.dispatcher.sbatch_submit", side_effect=fake.sbatch), \
         patch("spatialarena.dispatcher.sacct_states", side_effect=tracking_sacct), \
         patch("spatialarena.dispatcher.time.sleep", return_value=None):
        result = run_dispatcher(
            bench,
            selected_runs=runs,
            run_id="test-run",
            max_in_flight=2,
            infer_parallelism=4,
            eval_parallelism=6,
            poll_interval=0,
            state_path=tmp_path / "state.json",
        )

    assert result == {"total": 5, "passed": 5, "failed": 0}
    assert max_seen_active["value"] <= 2


def test_dispatcher_records_failure_without_stopping_other_items(tmp_path):
    runs = [
        {
            "task": "MonoDepth",
            "dataset": "marigold",
            "model": "VGGT",
            "track": "rgb"
        },
        {
            "task": "MonoDepth",
            "dataset": "marigold",
            "model": "DA3-L",
            "track": "rgb"
        },
    ]
    bench = BenchmarkEvaluator(experiment_root=tmp_path, experiment_name="demo")

    class _FailFirstSlurm(_FakeSlurm):

        def sacct(self, job_ids):
            states = super().sacct(job_ids)
            if job_ids and job_ids[0] == "1000":
                states["1000"] = "FAILED"
            return states

    fake = _FailFirstSlurm()

    with patch.object(BenchmarkEvaluator, "_generate_for_work_item", _fake_generate_for_work_item), \
         patch("spatialarena.dispatcher.sbatch_submit", side_effect=fake.sbatch), \
         patch("spatialarena.dispatcher.sacct_states", side_effect=fake.sacct), \
         patch("spatialarena.dispatcher.time.sleep", return_value=None):
        result = run_dispatcher(
            bench,
            selected_runs=runs,
            run_id="test-run",
            max_in_flight=2,
            infer_parallelism=4,
            eval_parallelism=6,
            poll_interval=0,
            state_path=tmp_path / "state.json",
        )

    assert result == {"total": 2, "passed": 1, "failed": 1}


def test_evaluate_only_dispatch_submits_only_evaluation_jobs(tmp_path):
    """evaluate_only: one evaluation job per run, no inference job; progress counts them as submitted."""
    from spatialarena.progress import dispatcher_progress  # pylint: disable=import-outside-toplevel

    runs = [{
        "task": "SLAM",
        "dataset": "7scenes",
        "model": model,
        "track": "rgb"
    } for model in ("VGGT", "PI3_X")]
    bench = BenchmarkEvaluator(experiment_root=tmp_path, experiment_name="demo")
    bench.evaluate_only = True
    fake = _FakeSlurm()
    scripts = []
    state_path = tmp_path / "dispatcher_state.json"

    def sbatch(script, *, dependency=None):
        scripts.append(Path(script).name)
        return fake.sbatch(script, dependency=dependency)

    with patch.object(BenchmarkEvaluator, "_generate_for_work_item", _fake_generate_for_work_item), \
         patch("spatialarena.dispatcher.sbatch_submit", side_effect=sbatch), \
         patch("spatialarena.dispatcher.sacct_states", side_effect=fake.sacct), \
         patch("spatialarena.dispatcher.time.sleep", return_value=None):
        result = run_dispatcher(bench,
                                selected_runs=runs,
                                run_id="test-run",
                                max_in_flight=2,
                                infer_parallelism=4,
                                eval_parallelism=6,
                                poll_interval=0,
                                state_path=state_path)

    assert result == {"total": 2, "passed": 2, "failed": 0}
    assert scripts == ["VGGT_eval.sh", "PI3_X_eval.sh"]
    with patch("spatialarena.slurm_utils.sacct_states",
               return_value={
                   "1000": "COMPLETED",
                   "1001": "COMPLETED"
               }):
        progress = dispatcher_progress(state_path)
    assert [run.stage for run in progress.runs] != ["queued", "queued"]


def test_detached_dispatcher_keeps_the_evaluation_switches(tmp_path):
    """evaluate_only, reevaluate and force_reinfer reach the detached dispatcher's evaluator (they decide what its
    jobs do)."""
    from types import SimpleNamespace  # pylint: disable=import-outside-toplevel

    from spatialarena.dispatcher import (  # pylint: disable=import-outside-toplevel
        _build_evaluator_from_input, launch_detached_dispatcher,
    )

    bench = BenchmarkEvaluator(experiment_root=tmp_path, experiment_name="demo")
    bench.evaluate_only = True
    bench.reevaluate = True
    bench.force_reinfer = True
    plan = SimpleNamespace(output_root=tmp_path / "demo", selected_runs=[])
    with patch("spatialarena.dispatcher.subprocess.Popen") as popen:
        popen.return_value.pid = 1
        launch_detached_dispatcher(bench,
                                   plan=plan,
                                   run_id="r",
                                   max_in_flight=1,
                                   infer_parallelism=None,
                                   eval_parallelism=None)
    payload = json.loads((tmp_path / "demo" / "scripts" /
                          "dispatcher_input.json").read_text(encoding="utf-8"))
    rebuilt = _build_evaluator_from_input(payload)
    assert rebuilt.evaluate_only and rebuilt.reevaluate and rebuilt.force_reinfer


def test_dispatcher_survives_a_transient_sacct_failure(tmp_path):
    import subprocess

    runs = [{
        "task": "MonoDepth",
        "dataset": "marigold",
        "model": "VGGT",
        "track": "rgb"
    }]
    bench = BenchmarkEvaluator(experiment_root=tmp_path, experiment_name="demo")
    fake = _FakeSlurm()
    calls = {"n": 0}

    def flaky_sacct(job_ids):
        calls["n"] += 1
        if calls["n"] == 1:  # slurmdbd briefly unreachable
            raise subprocess.CalledProcessError(1, ["sacct"])
        return fake.sacct(job_ids)

    with patch.object(BenchmarkEvaluator, "_generate_for_work_item", _fake_generate_for_work_item), \
         patch("spatialarena.dispatcher.sbatch_submit", side_effect=fake.sbatch), \
         patch("spatialarena.dispatcher.sacct_states", side_effect=flaky_sacct), \
         patch("spatialarena.dispatcher.time.sleep", return_value=None):
        result = run_dispatcher(bench,
                                selected_runs=runs,
                                run_id="test-run",
                                max_in_flight=1,
                                infer_parallelism=None,
                                eval_parallelism=None,
                                poll_interval=0,
                                state_path=tmp_path / "state.json")

    assert result == {"total": 1, "passed": 1, "failed": 0}
    assert json.loads(
        (tmp_path /
         "state.json").read_text(encoding="utf-8"))["status"] == "finished"


# ---- slurm_utils -------------------------------------------------------------------------------------------


def _mock_run(stdout="12345"):
    result = MagicMock()
    result.stdout = stdout
    return result


def test_sbatch_submit_uses_plain_export_nil():
    # Regression guard: on this cluster, --export=NIL,VAR=value has
    # intermittently triggered user_env_retrieval_failed_requeued_held
    # (jobs 134724, 148183, 148184), so sbatch never carries variables: a
    # CASE_PARALLELISM override is exported in the generated script instead.
    with patch("spatialarena.slurm_utils.subprocess.run",
               return_value=_mock_run()) as run:
        sbatch_submit("/fake/script.sh")
    cmd = run.call_args.args[0]
    assert "--export=NIL" in cmd
    assert not any("CASE_PARALLELISM" in arg for arg in cmd)


def test_sbatch_submit_appends_dependency_when_given():
    with patch("spatialarena.slurm_utils.subprocess.run",
               return_value=_mock_run()) as run:
        sbatch_submit("/fake/script.sh", dependency="afterok:999")
    cmd = run.call_args.args[0]
    assert "--dependency=afterok:999" in cmd


def test_sbatch_submit_parses_parsable_jobid_with_cluster_suffix():
    with patch("spatialarena.slurm_utils.subprocess.run",
               return_value=_mock_run("12345;mycluster\n")):
        job_id = sbatch_submit("/fake/script.sh")
    assert job_id == "12345"
