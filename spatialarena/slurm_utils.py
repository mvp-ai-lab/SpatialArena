# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Thin `sbatch`/`sacct` wrappers.

They are shared by `BenchmarkEvaluator.evaluate()` (submit-everything mode)
and `spatialarena.dispatcher` (bounded concurrency mode) and kept in one place
so both submit jobs the same way.
"""

from __future__ import annotations

import pathlib
import subprocess

TERMINAL_MARKERS = (
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "DEADLINE",
    "PREEMPTED",
)


def sbatch_submit(script: pathlib.Path,
                  *,
                  dependency: str | None = None) -> str:
    """Submits a job script with `sbatch --export=NIL`.

    The script is expected to carry its own #SBATCH resource header (see
    script_generator.py). It is submitted with plain `--export=NIL`: a clean
    environment and no extra variables. Combining it with an explicit
    VAR=value (even `--export=NIL,VAR=value`) intermittently leaves the job
    held with `user_env_retrieval_failed_requeued_held` on this cluster (jobs
    134724, 148183, 148184), so a per-run setting such as CASE_PARALLELISM is
    written into the generated script instead
    (script_generator.generate_run_scripts).

    Args:
        script: The job script.
        dependency: Optional `--dependency` value, e.g. `afterok:123`.

    Returns:
        The job id.

    Raises:
        subprocess.CalledProcessError: If sbatch fails.
    """
    cmd = ["sbatch", "--parsable", "--export=NIL"]
    if dependency:
        cmd.append(f"--dependency={dependency}")
    cmd.append(str(script))
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # sbatch --parsable prints "jobid" or "jobid;cluster".
    return result.stdout.strip().split(";")[0]


def sacct_states(job_ids: list[str]) -> dict[str, str]:
    """Returns the sacct state of each job, keyed by job id."""
    if not job_ids:
        return {}
    cmd = [
        "sacct",
        "-j",
        ",".join(job_ids),
        "--format=JobID,State",
        "-X",
        "--noheader",
        "--parsable2",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    states: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) >= 2:
            states[parts[0].strip()] = parts[1].strip()
    return states


def is_terminal_state(state: str) -> bool:
    """Returns whether a job state is final; an empty state counts as final."""
    return state == "" or any(marker in state for marker in TERMINAL_MARKERS)


def cancel_jobs(job_ids: list[str]) -> None:
    """Cancels jobs with scancel, ignoring its exit status."""
    if not job_ids:
        return
    subprocess.run(["scancel", *job_ids], check=False)
