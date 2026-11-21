# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Per-stage progress of an evaluation (spatialarena/progress.py, BenchmarkEvaluator.progress()/wait())."""

import json
import sys
from pathlib import Path


from spatialarena.evaluator import BenchmarkEvaluator, EvaluationPlan, EvaluationReport
from spatialarena.progress import ProgressBars, dispatcher_progress, slurm_run_progress

HEADER = "method\ttask\tdataset\tscene\tinterval\torder\toutput_path\treport_dir\tprediction_retention\n"
FAIL_HEADER = "method\ttask\tdataset\tscene\tinterval\torder\tstatus\n"


def _job(tmp_path: Path, name: str = "run", cases: int = 5, infer: str | None = "1",
         evaluation: str | None = "2") -> dict:
    case_file = tmp_path / name / f"VGGT_full-slam_{name}.txt"
    case_file.parent.mkdir(parents=True)
    case_file.write_text("# header\n" + "".join(f"SLAM|tum_dynamics|s{i}|10\n" for i in range(cases)))
    return {
        "work_item": {"task": "SLAM", "dataset": "tum_dynamics", "model": "VGGT", "track": "rgb", "suite": "full-slam",
                      "slam_range": "short"},
        "case_file": str(case_file),
        "report_dir": str(case_file.parent),
        "infer_job_id": infer,
        "eval_job_id": evaluation,
    }


def _write_rows(job: dict, rows: int, failures: int = 0, status: str = "kept") -> None:
    stem = Path(job["case_file"]).stem
    report_dir = Path(job["report_dir"])
    (report_dir / f"summary_{stem}.tsv").write_text(HEADER + f"VGGT\tSLAM\ttum\ts\t10\t\t/o\t/r\t{status}\n" * rows)
    (report_dir / f"case_failures_{stem}.tsv").write_text(FAIL_HEADER + "VGGT\tSLAM\ttum\ts\t10\t\t1\n" * failures)


def test_run_stage_and_case_counts_follow_job_states_and_summary_rows(tmp_path):
    job = _job(tmp_path)
    assert slurm_run_progress(job, "PENDING", "PENDING").stage == "queued"

    _write_rows(job, 2, status="inference_complete")
    running = slurm_run_progress(job, "RUNNING", "PENDING")
    assert (running.stage, running.inference_done, running.evaluation_done, running.total_cases) == (
        "inference", 2, 0, 5)

    waiting_eval = slurm_run_progress(job, "COMPLETED", "PENDING")
    assert (waiting_eval.stage, waiting_eval.inference_done, waiting_eval.evaluation_done) == ("evaluation", 5, 0)

    _write_rows(job, 3, failures=1)  # the evaluation job rewrote the TSVs
    evaluating = slurm_run_progress(job, "COMPLETED", "RUNNING")
    assert (evaluating.stage, evaluating.evaluation_done, evaluating.failed_cases) == ("evaluation", 4, 1)

    done = slurm_run_progress(job, "COMPLETED", "COMPLETED")
    assert (done.stage, done.inference_done, done.evaluation_done) == ("done", 5, 5)

    failed = slurm_run_progress(job, "OUT_OF_MEMORY", "PENDING")
    assert failed.stage == "failed" and "OUT_OF_MEMORY" in failed.note


def test_dispatcher_progress_includes_unsubmitted_runs(tmp_path, monkeypatch):
    submitted, pending = _job(tmp_path, "a"), _job(tmp_path, "b", cases=3, infer=None, evaluation=None)
    _write_rows(submitted, 4, status="inference_complete")
    state = tmp_path / "dispatcher_state.json"
    state.write_text(json.dumps({"status": "running", "items": [
        {**submitted, "phase": "infer"}, {**pending, "phase": "pending"},
    ]}))
    monkeypatch.setattr("spatialarena.slurm_utils.sacct_states", lambda ids: {"1": "RUNNING"})

    snapshot = dispatcher_progress(state)
    assert [run.stage for run in snapshot.runs] == ["inference", "queued"]
    assert snapshot.cases("inference") == (4, 8)
    assert snapshot.stage_counts()["queued"] == 1 and not snapshot.finished
    assert "inference cases: 4/8" in snapshot.render()
    bars = ProgressBars(disable=True)
    bars.update(snapshot)
    bars.close()


def _report(tmp_path, summary: dict) -> EvaluationReport:
    plan = EvaluationPlan(config={}, selected_datasets=[], selected_models=[], selected_runs=[], output_root=tmp_path)
    return EvaluationReport(ok=True, config={}, plan=plan, summary=summary, report_dir=tmp_path)


def test_wait_polls_until_every_run_finishes_then_collects(tmp_path, monkeypatch):
    job = _job(tmp_path)
    _write_rows(job, 5)
    states = iter([{"1": "RUNNING", "2": "PENDING"}, {"1": "COMPLETED", "2": "RUNNING"}]
                  + [{"1": "COMPLETED", "2": "COMPLETED"}] * 3)
    monkeypatch.setattr("spatialarena.slurm_utils.sacct_states", lambda ids: next(states))
    evaluator_module = sys.modules["spatialarena.evaluator"]
    monkeypatch.setattr(evaluator_module, "sacct_states", lambda ids: {"1": "COMPLETED", "2": "COMPLETED"})

    bench = BenchmarkEvaluator()
    report = _report(tmp_path, {"status": "submitted", "jobs": [job]})
    collected = bench.wait(report, poll_interval=0, progress="none")
    assert collected.summary["status"] == "finished"
    assert collected.summary["passed"] == 5
    assert bench.progress(report).finished


def test_collect_reads_a_finished_dispatcher_run(tmp_path, monkeypatch):
    job = _job(tmp_path)
    _write_rows(job, 5)
    state = tmp_path / "dispatcher_state.json"
    state.write_text(json.dumps({"status": "finished", "items": [{**job, "phase": "done"}]}))
    evaluator_module = sys.modules["spatialarena.evaluator"]
    monkeypatch.setattr(evaluator_module, "sacct_states", lambda ids: {"1": "COMPLETED", "2": "COMPLETED"})

    report = _report(tmp_path, {"status": "dispatching", "state_path": str(state)})
    collected = BenchmarkEvaluator().collect(report)
    assert collected.summary["status"] == "finished" and collected.summary["passed"] == 5

    state.write_text(json.dumps({"status": "running", "items": [{**job, "phase": "eval"}]}))
    assert BenchmarkEvaluator().collect(report) is report  # still dispatching: unchanged
