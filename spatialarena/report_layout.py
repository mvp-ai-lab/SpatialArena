# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The one report folder of each full-benchmark (suite, method) pair.

Every full-suite run of a method writes its per-case results into the same
folder, so reruns, retries and fixes overwrite in place and all metrics stay
in one tree. Must match the `REPORT_DIR` defaults (and the guard that rejects
any other `REPORT_DIR`) in
`templates/_engine/run_method_benchmark_{infer,eval}.sh`:

  <eval root>/reports/benchmark_full_reports/full-depth/<METHOD_ID>
  <eval root>/reports/benchmark_full_pose_reports/full-pose/<METHOD_ID>
  <eval root>/reports/benchmark_full_slam_reports/full-slam-<range>/<METHOD_ID>

One-off tests write under `<eval root>/reports/_<name>/` (underscore prefix),
never next to these folders.
"""

from __future__ import annotations

import pathlib

FULL_REPORT_ROOTS = {
    "full-depth": "benchmark_full_reports",
    "full-pose": "benchmark_full_pose_reports",
    "full-slam": "benchmark_full_slam_reports",
}


def canonical_report_dir(benchmark_eval_root: pathlib.Path,
                         suite: str,
                         method_id: str,
                         slam_range: str = "short") -> pathlib.Path:
    """Returns the report folder of a method in a full suite.

    Args:
        benchmark_eval_root: Evaluation output root.
        suite: Full suite: `full-depth`, `full-pose` or `full-slam`.
        method_id: Method id, the last path component.
        slam_range: SLAM sequence range of the `full-slam` suite.

    Returns:
        The report folder.

    Raises:
        ValueError: If `suite` is not a full suite.
    """
    if suite not in FULL_REPORT_ROOTS:
        raise ValueError(
            f"no canonical report folder for suite {suite!r}; full suites: "
            f"{sorted(FULL_REPORT_ROOTS)}")
    if suite == "full-slam":
        suite_dir = f"full-slam-{slam_range}"
    else:
        suite_dir = suite
    reports_dir = pathlib.Path(benchmark_eval_root) / "reports"
    return reports_dir / FULL_REPORT_ROOTS[suite] / suite_dir / method_id
