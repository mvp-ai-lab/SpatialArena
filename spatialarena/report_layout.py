# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""The one report folder of each full-benchmark (suite, method) pair.

Every full-suite run of a method writes its per-case results into the same folder, so reruns, retries and fixes
overwrite in place and all metrics stay in one tree. Must match the ``REPORT_DIR`` defaults (and the guard that
rejects any other ``REPORT_DIR``) in ``templates/_engine/run_method_benchmark_{infer,eval}.sh``.

    <eval root>/reports/benchmark_full_reports/full-depth/<METHOD_ID>
    <eval root>/reports/benchmark_full_pose_reports/full-pose/<METHOD_ID>
    <eval root>/reports/benchmark_full_slam_reports/full-slam-<range>/<METHOD_ID>

One-off tests write under ``<eval root>/reports/_<name>/`` (underscore prefix), never next to these folders.
"""

from __future__ import annotations

from pathlib import Path

FULL_REPORT_ROOTS = {
    "full-depth": "benchmark_full_reports",
    "full-pose": "benchmark_full_pose_reports",
    "full-slam": "benchmark_full_slam_reports",
}


def canonical_report_dir(benchmark_eval_root: Path, suite: str, method_id: str, slam_range: str = "short") -> Path:
    """Report folder of ``method_id`` in full suite ``suite`` (``full-depth``, ``full-pose`` or ``full-slam``)."""
    if suite not in FULL_REPORT_ROOTS:
        raise ValueError(f"no canonical report folder for suite {suite!r}; full suites: {sorted(FULL_REPORT_ROOTS)}")
    suite_dir = f"full-slam-{slam_range}" if suite == "full-slam" else suite
    return Path(benchmark_eval_root) / "reports" / FULL_REPORT_ROOTS[suite] / suite_dir / method_id
