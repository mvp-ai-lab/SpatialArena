# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Progress of a `BenchmarkEvaluator` run, per pipeline stage.

Every run (one method x task x dataset x track work item) goes through the
same stages:

    queued -> inference -> evaluation -> done | failed

`queued`: not submitted yet (bounded dispatcher) or waiting in SLURM's queue.
`inference`: the GPU job is running the adapter case by case. `evaluation`:
the CPU job is verifying and scoring the predictions case by case.

Nothing here is guessed: run stages come from the SLURM job states (`sacct`)
-- or the dispatcher's state file, which also knows the runs it has not
submitted yet -- and case counts from the files `run_method_case_list.sh`
already writes. Its summary TSV gains one row per finished case (failures go
to the case-failures TSV) and each phase (inference job, then evaluation job)
truncates and rewrites it, so while a phase runs its rows count that phase's
finished cases. The total per run is the number of case lines in its
generated case file.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

from spatialarena import slurm_utils

STAGES = ("queued", "inference", "evaluation", "done", "failed")


@dataclasses.dataclass
class RunProgress:
    """Progress of one run.

    Attributes:
        label: Display label (`run_label`).
        stage: One of `STAGES`.
        total_cases: Cases in the run's case file.
        inference_done: Cases the inference job has finished.
        evaluation_done: Cases the evaluation job has finished.
        failed_cases: Cases of the current phase that failed.
        infer_job_id: SLURM id of the inference job.
        eval_job_id: SLURM id of the evaluation job.
        note: Why the run failed, or another remark.
        method: Method name.
        track: Input-modality track.
        task: Benchmark task.
        dataset: Dataset name.
        slam_range: SLAM range, or "".
        report_dir: Report folder of the run.
    """

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
        """Task|track[|range], e.g. "SLAM|rgb|short": a board Tracks row."""
        return "|".join(
            part for part in (self.task, self.track, self.slam_range) if part)


def new_run(work_item: dict,
            *,
            total_cases: int,
            stage: str = "queued",
            report_dir: str | pathlib.Path | None = None,
            **fields) -> RunProgress:
    """Returns a RunProgress for one plan work item.

    Args:
        work_item: The planned run.
        total_cases: Cases in its case file.
        stage: Initial stage.
        report_dir: Report folder of the run.
        **fields: Other RunProgress fields.

    Returns:
        The progress record.
    """
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


@dataclasses.dataclass
class PipelineProgress:
    """Progress of all runs of a pipeline.

    Attributes:
        runs: Progress of each run.
    """

    runs: list[RunProgress] = dataclasses.field(default_factory=list)

    @property
    def finished(self) -> bool:
        """Whether every run is done or failed."""
        return all(run.stage in ("done", "failed") for run in self.runs)

    def stage_counts(self) -> dict[str, int]:
        """Returns the number of runs in each stage."""
        counts = dict.fromkeys(STAGES, 0)
        for run in self.runs:
            counts[run.stage] += 1
        return counts

    def cases(self, stage: str) -> tuple[int, int]:
        """Returns (finished, total) cases of a stage over all runs.

        Args:
            stage: "inference" or "evaluation".

        Returns:
            The finished and total case counts.
        """
        attribute = {
            "inference": "inference_done",
            "evaluation": "evaluation_done"
        }[stage]
        finished = sum(getattr(run, attribute) for run in self.runs)
        total = sum(run.total_cases for run in self.runs)
        return finished, total

    def to_dict(self) -> dict:
        """Returns the progress as a JSON-serializable dict."""
        return {
            "finished": self.finished,
            "runs_by_stage": self.stage_counts(),
            "inference_cases": list(self.cases("inference")),
            "evaluation_cases": list(self.cases("evaluation")),
            "runs": [run.__dict__ for run in self.runs],
        }

    def render(self) -> str:
        """Returns a plain-text table, one line per run, for logs."""
        counts = self.stage_counts()
        infer_done, infer_total = self.cases("inference")
        eval_done, eval_total = self.cases("evaluation")
        lines = [
            "runs: " +
            ", ".join(f"{stage} {counts[stage]}" for stage in STAGES),
            f"inference cases: {infer_done}/{infer_total}   "
            f"evaluation cases: {eval_done}/{eval_total}",
        ]
        for run in self.runs:
            line = (f"  {run.label:50s} {run.stage:10s} infer "
                    f"{run.inference_done}/{run.total_cases}  eval "
                    f"{run.evaluation_done}/{run.total_cases}")
            if run.failed_cases:
                line += f"  failed {run.failed_cases}"
            if run.note:
                line += f"  ({run.note})"
            lines.append(line)
        return "\n".join(lines)


def run_label(work_item: dict) -> str:
    """Returns the display label of a work item, e.g. "VGGT/rgb SfM/eth3d"."""
    label = (f"{work_item.get('model')}/{work_item.get('track', 'rgb')} "
             f"{work_item.get('task')}/{work_item.get('dataset')}")
    if work_item.get("slam_range"):
        label += f" ({work_item['slam_range']} range)"
    return label


def count_cases(case_file: str | pathlib.Path | None) -> int:
    """Returns the number of case lines of a case file; 0 if it is missing."""
    if not case_file or not pathlib.Path(case_file).is_file():
        return 0
    text = pathlib.Path(case_file).read_text(encoding="utf-8")
    return sum(1 for line in text.splitlines()
               if line.strip() and not line.lstrip().startswith("#"))


def finished_case_rows(job: dict) -> tuple[int, int]:
    """Returns the current phase's finished cases of a run.

    Args:
        job: A collect()-style job dict.

    Returns:
        The rows in the run's summary TSV and in its case-failures TSV.
    """
    # Imported by name, here: `from spatialarena import evaluator` would give
    # the package's `evaluator` instance, and a top-level import is circular.
    # pylint: disable-next=import-outside-toplevel
    from spatialarena.evaluator import _read_tsv_data_rows
    # pylint: disable-next=import-outside-toplevel
    from spatialarena.evaluator import _report_tsv_paths

    summary_tsv, failures_tsv = _report_tsv_paths(job)
    rows = 0
    if summary_tsv.is_file():
        rows = len(_read_tsv_data_rows(summary_tsv))
    failures = 0
    if failures_tsv.is_file():
        failures = len(_read_tsv_data_rows(failures_tsv))
    return rows, failures


def _state_kind(state: str | None) -> str:
    """Classifies a SLURM job state.

    Args:
        state: The sacct state; None if the job was never submitted.

    Returns:
        "none" (no job), "waiting" (submitted, not started or not indexed
        yet), "running", "ok" or "failed".
    """
    if state is None:
        return "none"
    if not state or state.startswith("PENDING") or state.startswith("REQUEUED"):
        return "waiting"
    if "COMPLETED" in state:
        return "ok"
    if any(marker in state for marker in slurm_utils.TERMINAL_MARKERS):
        return "failed"
    return "running"


def slurm_run_progress(job: dict, infer_state: str | None,
                       eval_state: str | None) -> RunProgress:
    """Returns the progress of one submitted run from its two job states.

    Args:
        job: A collect()-style job dict.
        infer_state: State of the inference job; None if never submitted.
        eval_state: State of the evaluation job; None if never submitted.

    Returns:
        The run's progress.
    """
    total = count_cases(job.get("case_file"))
    progress = new_run(
        job["work_item"],
        total_cases=total,
        report_dir=job.get("report_dir"),
        infer_job_id=job.get("infer_job_id"),
        eval_job_id=job.get("eval_job_id"),
    )
    infer = _state_kind(infer_state)
    evaluation = _state_kind(eval_state)
    if infer in ("none", "waiting"):
        return progress
    if infer == "running":
        progress.stage = "inference"
        progress.inference_done, progress.failed_cases = _phase_rows(job, total)
        return progress
    if infer == "failed":
        progress.stage = "failed"
        progress.note = f"inference job {infer_state}"
        progress.inference_done, progress.failed_cases = _phase_rows(job, total)
        return progress
    progress.inference_done = total
    if evaluation in ("none", "waiting"):
        # Inference finished; the evaluation job is not submitted or not
        # started yet.
        progress.stage = "evaluation"
        return progress
    if evaluation == "running":
        progress.stage = "evaluation"
        progress.evaluation_done, progress.failed_cases = _phase_rows(
            job, total)
        return progress
    progress.evaluation_done, progress.failed_cases = _phase_rows(job, total)
    if evaluation == "ok":
        progress.stage = "done"
        progress.evaluation_done = total
    else:
        progress.stage = "failed"
        progress.note = f"evaluation job {eval_state}"
    return progress


def _phase_rows(job: dict, total: int) -> tuple[int, int]:
    """Returns (finished, failed) cases of the current phase, capped."""
    rows, failures = finished_case_rows(job)
    return min(rows + failures, total), failures


def slurm_progress(jobs: list[dict]) -> PipelineProgress:
    """Returns the progress of submitted runs from their sacct job states."""
    ids = [
        job_id for job in jobs
        for job_id in (job.get("infer_job_id"), job.get("eval_job_id"))
        if job_id
    ]
    states = slurm_utils.sacct_states(ids) if ids else {}
    runs = []
    for job in jobs:
        if job.get("infer_job_id"):
            infer_state = states.get(job["infer_job_id"], "")
        elif job.get("eval_job_id"):
            # An evaluate-only run (BenchmarkEvaluator.evaluate_only) has no
            # inference job: its predictions exist.
            infer_state = "COMPLETED"
        else:
            infer_state = None
        if job.get("eval_job_id"):
            eval_state = states.get(job["eval_job_id"], "")
        else:
            eval_state = None
        runs.append(slurm_run_progress(job, infer_state, eval_state))
    return PipelineProgress(runs=runs)


def dispatcher_jobs(state_path: str | pathlib.Path) -> tuple[str, list[dict]]:
    """Reads the dispatcher state file.

    Args:
        state_path: The dispatcher state file.

    Returns:
        The dispatcher status and its items as collect()-style job dicts;
        unsubmitted items have no job ids.
    """
    path = pathlib.Path(state_path)
    if not path.is_file():
        return "starting", []
    payload = json.loads(path.read_text(encoding="utf-8"))
    jobs = []
    for item in payload.get("items", []):
        job = {
            key: item.get(key)
            for key in ("work_item", "infer_job_id", "eval_job_id", "case_file",
                        "report_dir")
        }
        job["infer_script"] = item.get("infer_script")
        job["eval_script"] = item.get("eval_script")
        job["dispatch_phase"] = item.get("phase")
        job["dispatch_error"] = item.get("error")
        jobs.append(job)
    return payload.get("status", "running"), jobs


def dispatcher_progress(state_path: str | pathlib.Path) -> PipelineProgress:
    """Returns the progress of a dispatcher run, unsubmitted runs included."""
    _status, jobs = dispatcher_jobs(state_path)
    submitted = [
        job for job in jobs if job.get("infer_job_id") or job.get("eval_job_id")
    ]
    submitted_runs = iter(slurm_progress(submitted).runs if submitted else [])
    runs = []
    for job in jobs:
        if job.get("infer_job_id") or job.get("eval_job_id"):
            runs.append(next(submitted_runs))
        elif job.get("dispatch_phase") == "failed":
            runs.append(
                new_run(job["work_item"],
                        stage="failed",
                        total_cases=count_cases(job.get("case_file")),
                        report_dir=job.get("report_dir"),
                        note=job.get("dispatch_error") or "submission failed"))
        else:
            runs.append(
                new_run(job["work_item"],
                        total_cases=count_cases(job.get("case_file")),
                        report_dir=job.get("report_dir")))
    return PipelineProgress(runs=runs)


class ProgressBars:
    """Three tqdm bars for a pipeline.

    The bars show the finished runs, the inference cases and the evaluation
    cases.
    """

    def __init__(self, *, disable: bool = False):
        # pylint: disable-next=import-outside-toplevel
        from tqdm import auto as tqdm_auto

        # Counts are set from snapshots rather than incremented, so tqdm's
        # rate/ETA would be meaningless: show counts and elapsed time only.
        bar_format = (
            "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {unit} "
            "[{elapsed}]{postfix}")
        bar_specs = (
            ("runs", "runs finished", "runs"),
            ("inference", "inference   ", "cases"),
            ("evaluation", "evaluation  ", "cases"),
        )
        self._bars = {}
        for position, (name, desc, unit) in enumerate(bar_specs):
            self._bars[name] = tqdm_auto.tqdm(total=0,
                                              desc=desc,
                                              unit=unit,
                                              position=position,
                                              disable=disable,
                                              bar_format=bar_format)

    def update(self, progress: PipelineProgress) -> None:
        """Sets the bars from a progress snapshot."""
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
                postfix = ", ".join(
                    f"{stage} {counts[stage]}" for stage in STAGES
                    if counts[stage])
                bar.set_postfix_str(postfix, refresh=False)
            bar.refresh()

    def close(self) -> None:
        """Closes the bars."""
        for bar in self._bars.values():
            bar.close()
