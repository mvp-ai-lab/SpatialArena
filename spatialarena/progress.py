# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Progress of a ``BenchmarkEvaluator`` run, per pipeline stage.

Every run (one method x task x dataset x track work item) goes through the same stages:

    queued -> inference -> evaluation -> done | failed

``queued``: not submitted yet (bounded dispatcher) or waiting in SLURM's queue. ``inference``: the GPU job is running
the adapter case by case. ``evaluation``: the CPU job is verifying and scoring the predictions case by case.

Nothing here is guessed: run stages come from the SLURM job states (``sacct``) -- or the dispatcher's state file,
which also knows the runs it has not submitted yet -- and case counts from the files ``run_method_case_list.sh``
already writes. Its summary TSV gains one row per finished case (failures go to the case-failures TSV) and each phase
(inference job, then evaluation job) truncates and rewrites it, so while a phase runs its rows count that phase's
finished cases. The total per run is the number of case lines in its generated case file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

STAGES = ("queued", "inference", "evaluation", "done", "failed")


@dataclass
class RunProgress:
    label: str
    stage: str
    total_cases: int
    inference_done: int = 0
    evaluation_done: int = 0
    failed_cases: int = 0
    infer_job_id: str | None = None
    eval_job_id: str | None = None
    note: str = ""
    # What the run is, for grouping (by track, method, dataset) in the board.
    method: str = ""
    track: str = ""
    task: str = ""
    dataset: str = ""
    slam_range: str = ""
    report_dir: str = ""

    @property
    def track_key(self) -> str:
        """Task|track[|range], e.g. "SLAM|rgb|short" -- one row of the board's Tracks view."""
        return "|".join(part for part in (self.task, self.track, self.slam_range) if part)


def new_run(work_item: dict, *, total_cases: int, stage: str = "queued", report_dir: str | Path | None = None,
            **fields) -> RunProgress:
    """A RunProgress for one plan work item."""
    return RunProgress(
        label=run_label(work_item),
        stage=stage,
        total_cases=total_cases,
        method=str(work_item.get("model", "")),
        track=str(work_item.get("track", "rgb")),
        task=str(work_item.get("task", "")),
        dataset=str(work_item.get("dataset", "")),
        slam_range=str(work_item.get("slam_range") or ""),
        report_dir=str(report_dir or ""),
        **fields,
    )


@dataclass
class PipelineProgress:
    runs: list[RunProgress] = field(default_factory=list)

    @property
    def finished(self) -> bool:
        return all(run.stage in ("done", "failed") for run in self.runs)

    def stage_counts(self) -> dict[str, int]:
        counts = dict.fromkeys(STAGES, 0)
        for run in self.runs:
            counts[run.stage] += 1
        return counts

    def cases(self, stage: str) -> tuple[int, int]:
        """(finished, total) cases of ``stage`` ("inference" or "evaluation") over all runs."""
        attribute = {"inference": "inference_done", "evaluation": "evaluation_done"}[stage]
        return sum(getattr(run, attribute) for run in self.runs), sum(run.total_cases for run in self.runs)

    def to_dict(self) -> dict:
        return {
            "finished": self.finished,
            "runs_by_stage": self.stage_counts(),
            "inference_cases": list(self.cases("inference")),
            "evaluation_cases": list(self.cases("evaluation")),
            "runs": [run.__dict__ for run in self.runs],
        }

    def render(self) -> str:
        """Plain-text table (one line per run), for logs and notebooks."""
        counts = self.stage_counts()
        lines = [
            "runs: " + ", ".join(f"{stage} {counts[stage]}" for stage in STAGES),
            "inference cases: {}/{}   evaluation cases: {}/{}".format(*self.cases("inference"),
                                                                       *self.cases("evaluation")),
        ]
        for run in self.runs:
            lines.append(
                f"  {run.label:50s} {run.stage:10s} infer {run.inference_done}/{run.total_cases}  "
                f"eval {run.evaluation_done}/{run.total_cases}"
                + (f"  failed {run.failed_cases}" if run.failed_cases else "")
                + (f"  ({run.note})" if run.note else "")
            )
        return "\n".join(lines)


def run_label(work_item: dict) -> str:
    label = (f"{work_item.get('model')}/{work_item.get('track', 'rgb')} "
             f"{work_item.get('task')}/{work_item.get('dataset')}")
    if work_item.get("slam_range"):
        label += f" ({work_item['slam_range']} range)"
    return label


def count_cases(case_file: str | Path | None) -> int:
    if not case_file or not Path(case_file).is_file():
        return 0
    return sum(
        1 for line in Path(case_file).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def finished_case_rows(job: dict) -> tuple[int, int]:
    """(rows in the run's summary TSV, rows in its case-failures TSV) -- the current phase's finished cases."""
    from spatialarena.evaluator import _read_tsv_data_rows, _report_tsv_paths

    summary_tsv, failures_tsv = _report_tsv_paths(job)
    rows = len(_read_tsv_data_rows(summary_tsv)) if summary_tsv.is_file() else 0
    failures = len(_read_tsv_data_rows(failures_tsv)) if failures_tsv.is_file() else 0
    return rows, failures


def _state_kind(state: str | None) -> str:
    """"none" (no job), "waiting" (submitted, not started or not indexed yet), "running", "ok", "failed"."""
    from spatialarena.slurm_utils import TERMINAL_MARKERS

    if state is None:
        return "none"
    if not state or state.startswith("PENDING") or state.startswith("REQUEUED"):
        return "waiting"
    if "COMPLETED" in state:
        return "ok"
    if any(marker in state for marker in TERMINAL_MARKERS):
        return "failed"
    return "running"


def slurm_run_progress(job: dict, infer_state: str | None, eval_state: str | None) -> RunProgress:
    """Progress of one submitted run from its two job states (None: that job was never submitted)."""
    total = count_cases(job.get("case_file"))
    progress = new_run(
        job["work_item"],
        total_cases=total,
        report_dir=job.get("report_dir"),
        infer_job_id=job.get("infer_job_id"),
        eval_job_id=job.get("eval_job_id"),
    )
    infer, evaluation = _state_kind(infer_state), _state_kind(eval_state)
    if infer in ("none", "waiting"):
        return progress
    if infer == "running":
        progress.stage = "inference"
        progress.inference_done, progress.failed_cases = _phase_rows(job, total)
        return progress
    if infer == "failed":
        progress.stage, progress.note = "failed", f"inference job {infer_state}"
        progress.inference_done, progress.failed_cases = _phase_rows(job, total)
        return progress
    progress.inference_done = total
    if evaluation in ("none", "waiting"):  # inference finished; evaluation job not submitted or not started yet
        progress.stage = "evaluation"
        return progress
    if evaluation == "running":
        progress.stage = "evaluation"
        progress.evaluation_done, progress.failed_cases = _phase_rows(job, total)
        return progress
    progress.evaluation_done, progress.failed_cases = _phase_rows(job, total)
    if evaluation == "ok":
        progress.stage, progress.evaluation_done = "done", total
    else:
        progress.stage, progress.note = "failed", f"evaluation job {eval_state}"
    return progress


def _phase_rows(job: dict, total: int) -> tuple[int, int]:
    rows, failures = finished_case_rows(job)
    return min(rows + failures, total), failures


def slurm_progress(jobs: list[dict]) -> PipelineProgress:
    from spatialarena.slurm_utils import sacct_states

    ids = [job_id for job in jobs for job_id in (job.get("infer_job_id"), job.get("eval_job_id")) if job_id]
    states = sacct_states(ids) if ids else {}
    return PipelineProgress(
        runs=[
            slurm_run_progress(
                job,
                # An evaluate-only run (BenchmarkEvaluator.evaluate_only) has no inference job: its predictions exist.
                states.get(job["infer_job_id"], "") if job.get("infer_job_id")
                else ("COMPLETED" if job.get("eval_job_id") else None),
                states.get(job["eval_job_id"], "") if job.get("eval_job_id") else None,
            )
            for job in jobs
        ]
    )


def dispatcher_jobs(state_path: str | Path) -> tuple[str, list[dict]]:
    """(dispatcher status, its items as collect()-style job dicts; unsubmitted items have no job ids)."""
    path = Path(state_path)
    if not path.is_file():
        return "starting", []
    payload = json.loads(path.read_text(encoding="utf-8"))
    jobs = []
    for item in payload.get("items", []):
        job = {key: item.get(key) for key in ("work_item", "infer_job_id", "eval_job_id", "case_file", "report_dir")}
        job["infer_script"], job["eval_script"] = item.get("infer_script"), item.get("eval_script")
        job["dispatch_phase"], job["dispatch_error"] = item.get("phase"), item.get("error")
        jobs.append(job)
    return payload.get("status", "running"), jobs


def dispatcher_progress(state_path: str | Path) -> PipelineProgress:
    _status, jobs = dispatcher_jobs(state_path)
    submitted = [job for job in jobs if job.get("infer_job_id") or job.get("eval_job_id")]
    submitted_runs = iter(slurm_progress(submitted).runs if submitted else [])
    runs = []
    for job in jobs:
        if job.get("infer_job_id") or job.get("eval_job_id"):
            runs.append(next(submitted_runs))
        elif job.get("dispatch_phase") == "failed":
            runs.append(new_run(job["work_item"], stage="failed", total_cases=count_cases(job.get("case_file")),
                                report_dir=job.get("report_dir"),
                                note=job.get("dispatch_error") or "submission failed"))
        else:
            runs.append(new_run(job["work_item"], total_cases=count_cases(job.get("case_file")),
                                report_dir=job.get("report_dir")))
    return PipelineProgress(runs=runs)


class ProgressBars:
    """Three tqdm bars for a pipeline: finished runs, inference cases, evaluation cases."""

    def __init__(self, *, disable: bool = False):
        from tqdm.auto import tqdm

        # Counts are set from snapshots rather than incremented, so tqdm's rate/ETA would be meaningless: show
        # counts and elapsed time only.
        bar_format = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {unit} [{elapsed}]{postfix}"
        self._bars = {
            name: tqdm(total=0, desc=desc, unit=unit, position=position, disable=disable, bar_format=bar_format)
            for position, (name, desc, unit) in enumerate(
                (("runs", "runs finished", "runs"), ("inference", "inference   ", "cases"),
                 ("evaluation", "evaluation  ", "cases"))
            )
        }

    def update(self, progress: PipelineProgress) -> None:
        counts = progress.stage_counts()
        values = {
            "runs": (counts["done"] + counts["failed"], len(progress.runs)),
            "inference": progress.cases("inference"),
            "evaluation": progress.cases("evaluation"),
        }
        for name, (done, total) in values.items():
            bar = self._bars[name]
            bar.total = total
            bar.n = done
            if name == "runs":
                bar.set_postfix_str(", ".join(f"{s} {counts[s]}" for s in STAGES if counts[s]), refresh=False)
            bar.refresh()

    def close(self) -> None:
        for bar in self._bars.values():
            bar.close()
