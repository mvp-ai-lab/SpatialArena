# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Bounded-concurrency SLURM dispatcher.

`BenchmarkEvaluator.evaluate()` by default submits every run's infer and
eval job pair at once: fine for a handful of runs, but a large plan (dozens
or hundreds of runs) floods the scheduler's queue, which is unfriendly to
other users of a shared cluster. This module keeps at most `max_in_flight`
runs' jobs outstanding: it submits a run's inference job, waits (polling
sacct) for it to finish, submits that run's evaluation job, waits for that,
and only then takes the next run. The number of this dispatcher's jobs in
the queue is bounded by `max_in_flight`, never by the plan size.

It runs as a detached background process (`launch_detached_dispatcher`), so
it outlives the Python process that started it: a large plan can take many
hours to drain.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import subprocess
import sys
import time
from typing import TYPE_CHECKING

from spatialarena import slurm_utils
# Imported by name: the tests patch `dispatcher.sacct_states` and
# `dispatcher.sbatch_submit`.
from spatialarena.slurm_utils import sacct_states
from spatialarena.slurm_utils import sbatch_submit

if TYPE_CHECKING:
    # Only for annotations: importing the evaluator at load time is circular.
    from spatialarena.evaluator import BenchmarkEvaluator
    from spatialarena.evaluator import EvaluationPlan
    from spatialarena.evaluator import EvaluationReport

DEFAULT_POLL_INTERVAL = 30


def _job_terminal(state: str) -> bool:
    """Returns whether a job state is final; an empty state is not.

    Unlike slurm_utils.is_terminal_state (used by collect(), which polls well
    after submission), an empty state here just means sacct has not indexed a
    just-submitted job yet. It is treated as still active, not terminal, or
    the dispatcher would wrongly mark fresh submissions as failed.
    """
    return bool(state) and any(
        marker in state for marker in slurm_utils.TERMINAL_MARKERS)


@dataclasses.dataclass
class DispatchItem:
    """One run of the dispatch queue and its jobs.

    Attributes:
        index: Position of the run in the plan.
        work_item: The planned run.
        phase: pending -> infer -> eval -> done | failed.
        infer_job_id: SLURM id of the inference job.
        eval_job_id: SLURM id of the evaluation job.
        infer_script: Inference job script.
        eval_script: Evaluation job script.
        case_file: Case list of the run.
        report_dir: Report folder of the run.
        error: Why the run failed.
    """

    index: int
    work_item: dict
    phase: str = "pending"
    infer_job_id: str | None = None
    eval_job_id: str | None = None
    infer_script: str | None = None
    eval_script: str | None = None
    case_file: str | None = None
    report_dir: str | None = None
    error: str | None = None


def _current_job_id(item: DispatchItem) -> str | None:
    """Returns the job id of the phase the item is in."""
    if item.phase == "infer":
        return item.infer_job_id
    return item.eval_job_id


def _submit_first_job(bench: BenchmarkEvaluator, item: DispatchItem,
                      run_id: str, infer_parallelism: int | None) -> None:
    """Generates the scripts of a run and submits its first job."""
    # pylint: disable-next=protected-access
    item_with_suite, generated = bench._generate_for_work_item(
        item.work_item, run_id, case_parallelism=infer_parallelism)
    item.work_item = item_with_suite
    item.infer_script = str(generated.infer_script)
    item.eval_script = str(generated.eval_script)
    item.case_file = str(generated.case_file)
    item.report_dir = str(generated.report_dir)
    # The predictions already exist: only the evaluation job (no GPU).
    if bench.evaluate_only:
        item.eval_job_id = sbatch_submit(generated.eval_script)
        item.phase = "eval"
    else:
        item.infer_job_id = sbatch_submit(generated.infer_script)
        item.phase = "infer"


def _advance_finished_job(item: DispatchItem, job_id: str | None, state: str,
                          still_active: list[DispatchItem],
                          finished: list[DispatchItem]) -> None:
    """Moves a run whose current job ended to its next phase.

    A completed inference job is followed by the evaluation job; a completed
    evaluation job finishes the run; any other end state fails it.

    Args:
        item: The run.
        job_id: Its current job.
        state: The terminal sacct state of that job.
        still_active: Receives the run if it has a new job.
        finished: Receives the run if it is done or failed.
    """
    if "COMPLETED" in state:
        if item.phase == "infer":
            try:
                item.eval_job_id = sbatch_submit(item.eval_script)
                item.phase = "eval"
                still_active.append(item)
            # pylint: disable-next=broad-exception-caught
            except Exception as exc:  # noqa: BLE001
                item.phase = "failed"
                item.error = str(exc)
                finished.append(item)
        else:
            item.phase = "done"
            finished.append(item)
    else:
        item.phase = "failed"
        item.error = f"job {job_id} ended in state {state!r}"
        finished.append(item)


def run_dispatcher(
    bench: BenchmarkEvaluator,
    *,
    selected_runs: list[dict],
    run_id: str,
    max_in_flight: int,
    infer_parallelism: int | None,
    eval_parallelism: int | None,  # pylint: disable=unused-argument
    poll_interval: int = DEFAULT_POLL_INTERVAL,
    state_path: pathlib.Path,
) -> dict:
    """Runs the plan with at most `max_in_flight` runs in SLURM at a time.

    Each run's inference job is submitted, polled until it ends, then its
    evaluation job; the state file is rewritten after every change.

    Args:
        bench: The configured evaluator that generates the job scripts.
        selected_runs: The planned runs.
        run_id: Id of this evaluation run.
        max_in_flight: Maximum number of runs with a job in SLURM.
        infer_parallelism: Case parallelism of the inference jobs.
        eval_parallelism: Unused; kept for the launcher's interface.
        poll_interval: Seconds between sacct polls.
        state_path: JSON file the dispatcher state is written to.

    Returns:
        The `total`, `passed` and `failed` run counts.

    Raises:
        ValueError: If `max_in_flight` is less than 1.
    """
    if max_in_flight < 1:
        raise ValueError(f"max_in_flight must be >= 1, got {max_in_flight}")

    items = [
        DispatchItem(index=i, work_item=run)
        for i, run in enumerate(selected_runs)
    ]
    pending = list(items)
    active: list[DispatchItem] = []
    finished: list[DispatchItem] = []

    def write_state(status: str = "running") -> None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run_id,
            "status": status,
            "max_in_flight": max_in_flight,
            "total": len(items),
            "pending": len(pending),
            "active": len(active),
            "passed": sum(1 for item in finished if item.phase == "done"),
            "failed": sum(1 for item in finished if item.phase == "failed"),
            "items": [dataclasses.asdict(item) for item in items],
        }
        tmp_path = state_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, default=str),
                            encoding="utf-8")
        # Atomic, so a concurrent reader never sees a half-written file.
        tmp_path.replace(state_path)

    write_state()

    while pending or active:
        while len(active) < max_in_flight and pending:
            item = pending.pop(0)
            try:
                _submit_first_job(bench, item, run_id, infer_parallelism)
                active.append(item)
            # One bad run must not kill the dispatcher.
            # pylint: disable-next=broad-exception-caught
            except Exception as exc:  # noqa: BLE001
                item.phase = "failed"
                item.error = str(exc)
                finished.append(item)
        write_state()

        if not active:
            # Nothing outstanding and nothing pending: the loop exits.
            continue

        job_ids = [_current_job_id(item) for item in active]
        try:
            states = sacct_states([job_id for job_id in job_ids if job_id])
        except (subprocess.SubprocessError, OSError) as exc:
            # A transient sacct/slurmdbd error: poll again later.
            print(f"sacct failed ({exc}); retrying in {poll_interval}s",
                  flush=True)
            time.sleep(poll_interval)
            continue

        still_active: list[DispatchItem] = []
        for item in active:
            job_id = _current_job_id(item)
            state = states.get(job_id, "")
            if not _job_terminal(state):
                still_active.append(item)
                continue
            _advance_finished_job(item, job_id, state, still_active, finished)
        active = still_active
        write_state()

        if pending or active:
            time.sleep(poll_interval)

    write_state(status="finished")
    return {
        "total": len(items),
        "passed": sum(1 for item in finished if item.phase == "done"),
        "failed": sum(1 for item in finished if item.phase == "failed"),
    }


def launch_detached_dispatcher(
    bench: BenchmarkEvaluator,
    *,
    plan: EvaluationPlan,
    run_id: str,
    max_in_flight: int,
    infer_parallelism: int | None,
    eval_parallelism: int | None,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
) -> EvaluationReport:
    """Starts the dispatcher as a detached background process.

    The dispatcher input is written to `<output_root>/scripts/`, next to its
    state file and log; the process outlives the caller.

    Args:
        bench: The configured evaluator.
        plan: The evaluation plan to run.
        run_id: Id of this evaluation run.
        max_in_flight: Maximum number of runs with a job in SLURM.
        infer_parallelism: Case parallelism of the inference jobs.
        eval_parallelism: Case parallelism of the evaluation jobs.
        poll_interval: Seconds between sacct polls.

    Returns:
        A "dispatching" report with the dispatcher pid and file paths.
    """
    # Imported here to avoid a circular import.
    # pylint: disable-next=import-outside-toplevel
    from spatialarena.evaluator import EvaluationReport

    output_root = plan.output_root
    scripts_dir = output_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    input_path = scripts_dir / "dispatcher_input.json"
    state_path = scripts_dir / "dispatcher_state.json"
    log_path = scripts_dir / "dispatcher.log"

    input_payload = {
        "benchmark_root": str(bench.benchmark_root),
        "benchmark_eval_root": str(bench.benchmark_eval_root),
        "method_code_root": str(bench.method_code_root),
        "checkpoint_root": str(bench.checkpoint_root),
        "slurm_exclude": bench.slurm_exclude,
        "evaluate_only": bench.evaluate_only,
        "reevaluate": bench.reevaluate,
        "force_reinfer": bench.force_reinfer,
        "slurm_resources": bench.slurm_resources,
        "experiment_root": str(bench.experiment_root),
        "experiment_name": bench.experiment_name,
        "run_id": run_id,
        "max_in_flight": max_in_flight,
        "infer_parallelism": infer_parallelism,
        "eval_parallelism": eval_parallelism,
        "poll_interval": poll_interval,
        "selected_runs": plan.selected_runs,
    }
    input_path.write_text(json.dumps(input_payload, indent=2), encoding="utf-8")

    project_root = pathlib.Path(__file__).resolve().parents[1]
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "spatialarena.dispatcher",
                "--input",
                str(input_path),
                "--state",
                str(state_path),
            ],
            cwd=str(project_root),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            # Detach: survives the parent process exiting.
            start_new_session=True,
        )

    report = EvaluationReport(
        ok=True,
        config=bench._config_snapshot(),  # pylint: disable=protected-access
        plan=plan,
        scene_results=[],
        dataset_results=[],
        summary={
            "status": "dispatching",
            "run_id": run_id,
            "max_in_flight": max_in_flight,
            "total_runs": len(plan.selected_runs),
            "dispatcher_pid": process.pid,
            "input_path": str(input_path),
            "state_path": str(state_path),
            "log_path": str(log_path),
        },
        report_dir=output_root,
    )
    bench._last_report = report  # pylint: disable=protected-access
    return report


def _build_evaluator_from_input(payload: dict) -> BenchmarkEvaluator:
    """Rebuilds the evaluator from the input written by the launcher."""
    # pylint: disable-next=import-outside-toplevel
    from spatialarena.evaluator import BenchmarkEvaluator

    bench = BenchmarkEvaluator(
        benchmark_root=payload["benchmark_root"],
        benchmark_eval_root=payload["benchmark_eval_root"],
        method_code_root=payload["method_code_root"],
        checkpoint_root=payload.get("checkpoint_root"),
        experiment_root=payload["experiment_root"],
        experiment_name=payload["experiment_name"],
    )
    bench.slurm_exclude = payload.get("slurm_exclude")
    bench.evaluate_only = bool(payload.get("evaluate_only", False))
    bench.reevaluate = bool(payload.get("reevaluate", False))
    bench.force_reinfer = bool(payload.get("force_reinfer", False))
    bench.slurm_resources = payload.get("slurm_resources") or {}
    return bench


def main(argv: list[str] | None = None) -> int:
    """Runs the dispatcher of an input file written by the launcher."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=pathlib.Path)
    parser.add_argument("--state", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    bench = _build_evaluator_from_input(payload)

    result = run_dispatcher(
        bench,
        selected_runs=payload["selected_runs"],
        run_id=payload["run_id"],
        max_in_flight=payload["max_in_flight"],
        infer_parallelism=payload["infer_parallelism"],
        eval_parallelism=payload["eval_parallelism"],
        poll_interval=payload.get("poll_interval", DEFAULT_POLL_INTERVAL),
        state_path=args.state,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
