# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Bounded-concurrency SLURM dispatcher.

``BenchmarkEvaluator.evaluate()``'s default path submits every run's
infer+eval job pair immediately -- fine for a handful of runs, but a large
plan (dozens/hundreds of runs) floods the scheduler's queue all at once,
which is unfriendly to other users on a shared cluster and can draw admin
attention. This module keeps at most ``max_in_flight`` runs' worth of jobs
outstanding at any time: it submits a run's inference job, waits (polling
``sacct``) for it to finish, submits that run's evaluation job, waits for
that to finish, and only then pulls the next run off the queue -- so the
total number of jobs this dispatcher has in SLURM's queue at once is bounded
by ``max_in_flight``, never by the total plan size.

Runs as a detached background process (see ``launch_detached_dispatcher``)
so it outlives the Python process that started it -- necessary since a
large plan can take many hours to fully drain.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from spatialarena.slurm_utils import TERMINAL_MARKERS, sacct_states, sbatch_submit

DEFAULT_POLL_INTERVAL = 30


def _job_terminal(state: str) -> bool:
    """Unlike slurm_utils.is_terminal_state (used by collect(), which polls
    well after submission), an empty state here just means sacct hasn't
    indexed a just-submitted job yet -- treat that as still-active, not
    terminal, or the dispatcher would wrongly mark fresh submissions as
    failed."""
    return bool(state) and any(marker in state for marker in TERMINAL_MARKERS)


@dataclass
class DispatchItem:
    index: int
    work_item: dict
    phase: str = "pending"  # pending -> infer -> eval -> done | failed
    infer_job_id: str | None = None
    eval_job_id: str | None = None
    infer_script: str | None = None
    eval_script: str | None = None
    case_file: str | None = None
    report_dir: str | None = None
    error: str | None = None


def run_dispatcher(
    bench: "BenchmarkEvaluator",  # noqa: F821 (avoid a circular import at module load time)
    *,
    selected_runs: list[dict],
    run_id: str,
    max_in_flight: int,
    infer_parallelism: int | None,
    eval_parallelism: int | None,  # pylint: disable=unused-argument
    poll_interval: int = DEFAULT_POLL_INTERVAL,
    state_path: Path,
) -> dict:
    if max_in_flight < 1:
        raise ValueError(f"max_in_flight must be >= 1, got {max_in_flight}")

    items = [DispatchItem(index=i, work_item=run) for i, run in enumerate(selected_runs)]
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
            "passed": sum(1 for it in finished if it.phase == "done"),
            "failed": sum(1 for it in finished if it.phase == "failed"),
            "items": [asdict(it) for it in items],
        }
        tmp_path = state_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp_path.replace(state_path)  # atomic, so a concurrent reader never sees a half-written file

    write_state()

    while pending or active:
        while len(active) < max_in_flight and pending:
            item = pending.pop(0)
            try:
                item_with_suite, generated = bench._generate_for_work_item(  # pylint: disable=protected-access
                    item.work_item, run_id, case_parallelism=infer_parallelism)
                item.work_item = item_with_suite
                item.infer_script = str(generated.infer_script)
                item.eval_script = str(generated.eval_script)
                item.case_file = str(generated.case_file)
                item.report_dir = str(generated.report_dir)
                if bench.evaluate_only:  # predictions already exist: only the evaluation job (no GPU)
                    item.eval_job_id = sbatch_submit(generated.eval_script)
                    item.phase = "eval"
                else:
                    item.infer_job_id = sbatch_submit(generated.infer_script)
                    item.phase = "infer"
                active.append(item)
            # pylint: disable-next=broad-exception-caught
            except Exception as exc:  # noqa: BLE001 -- one bad run must not kill the dispatcher
                item.phase = "failed"
                item.error = str(exc)
                finished.append(item)
        write_state()

        if not active:
            continue  # nothing outstanding but pending remains empty too -> loop exits

        job_ids = [(it.infer_job_id if it.phase == "infer" else it.eval_job_id) for it in active]
        try:
            states = sacct_states([j for j in job_ids if j])
        except (subprocess.SubprocessError, OSError) as exc:  # a transient sacct/slurmdbd error: poll again later
            print(f"sacct failed ({exc}); retrying in {poll_interval}s", flush=True)
            time.sleep(poll_interval)
            continue

        still_active: list[DispatchItem] = []
        for it in active:
            job_id = it.infer_job_id if it.phase == "infer" else it.eval_job_id
            state = states.get(job_id, "")
            if not _job_terminal(state):
                still_active.append(it)
                continue
            if "COMPLETED" in state:
                if it.phase == "infer":
                    try:
                        it.eval_job_id = sbatch_submit(it.eval_script)
                        it.phase = "eval"
                        still_active.append(it)
                    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
                        it.phase = "failed"
                        it.error = str(exc)
                        finished.append(it)
                else:
                    it.phase = "done"
                    finished.append(it)
            else:
                it.phase = "failed"
                it.error = f"job {job_id} ended in state {state!r}"
                finished.append(it)
        active = still_active
        write_state()

        if pending or active:
            time.sleep(poll_interval)

    write_state(status="finished")
    return {
        "total": len(items),
        "passed": sum(1 for it in finished if it.phase == "done"),
        "failed": sum(1 for it in finished if it.phase == "failed"),
    }


def launch_detached_dispatcher(
    bench: "BenchmarkEvaluator",  # noqa: F821
    *,
    plan: "EvaluationPlan",  # noqa: F821
    run_id: str,
    max_in_flight: int,
    infer_parallelism: int | None,
    eval_parallelism: int | None,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
) -> "EvaluationReport":  # noqa: F821
    from spatialarena.evaluator import EvaluationReport  # local import: avoid a circular import

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

    project_root = Path(__file__).resolve().parents[1]
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [sys.executable, "-m", "spatialarena.dispatcher", "--input", str(input_path), "--state", str(state_path)],
            cwd=str(project_root),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # detach: survives the parent process exiting
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


def _build_evaluator_from_input(payload: dict) -> "BenchmarkEvaluator":  # noqa: F821
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
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
