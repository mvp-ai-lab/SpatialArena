# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Merges scattered full-depth and full-pose report folders.

The result is one folder per suite and method. Runs that set their own
`RUN_ID` / `REPORT_DIR` (retries, fixes, resumes) left copies of the same cases
in several trees under `reports/`. For every case (suite, METHOD_ID, Task,
Dataset, Scene[, order]) the copy whose `eval_*.json` is newest wins and is
copied into the canonical folder (`spatialarena/report_layout.py`), below
`reports/`:

    benchmark_full_reports/full-depth/<METHOD_ID>/<Task>/<Dataset>/<Scene>
    benchmark_full_pose_reports/full-pose/<METHOD_ID>/<Task>/<Dataset>/<Scene>

An older canonical copy it replaces moves to
`_archive/<tag>/replaced_canonical/`; afterwards every other top-level folder
of `reports/` (per-run trees, tests) moves to `_archive/<tag>/`. Nothing is
deleted, and `_archive/<tag>/manifest.tsv` records every action. Test (`_*`)
and `*verify*` folders are archived, never merged. SLAM reports already live in
their canonical tree and are not touched.

    # Plan:
    python scripts/benchmark/consolidate_report_folders.py \\
        --reports_root <eval root>/reports
    # Apply:
    python scripts/benchmark/consolidate_report_folders.py \\
        --reports_root <eval root>/reports --execute
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import pathlib
import re
import shutil

CANONICAL = {
    "full-depth": "benchmark_full_reports",
    "full-pose": "benchmark_full_pose_reports"
}
KEEP_TOP = {
    "benchmark_full_reports", "benchmark_full_pose_reports",
    "benchmark_full_slam_reports", "benchmark_sampled_reports", "_archive"
}
NOT_SOURCES = ("benchmark_full_slam_reports", "benchmark_sampled_reports", "_*",
               "*verify*")
CHUNK = re.compile(r"^chunk_\d+$")


def case_key(case_dir: pathlib.Path,
             reports: pathlib.Path) -> tuple[str, tuple[str, ...]] | None:
    """Returns the key of a case folder.

    Args:
        case_dir: The case folder.
        reports: The reports root.

    Returns:
        The suite and the case path below the suite folder, with retry
        `chunk_NN` levels dropped, or None if the folder is not a case of a
        full-depth or full-pose suite.
    """
    parts = case_dir.relative_to(reports).parts
    for index, part in enumerate(parts):
        if part in CANONICAL:
            rel = tuple(
                level for level in parts[index + 1:] if not CHUNK.match(level))
            if len(rel) >= 2:
                return (part, rel)
            return None
    return None


def newest_eval_mtime(case_dir: pathlib.Path) -> float:
    """Returns the latest mtime of the folder's non-empty `eval_*.json` files.

    Args:
        case_dir: The case folder.

    Returns:
        The mtime, or -1 if the folder has no such file.
    """
    times = [
        path.stat().st_mtime
        for path in case_dir.glob("eval_*.json")
        if path.stat().st_size > 0
    ]
    if times:
        return max(times)
    return -1.0


def find_case_copies(
    reports: pathlib.Path
) -> dict[tuple[str, tuple[str, ...]], list[tuple[float, pathlib.Path]]]:
    """Finds every copy of every full-depth and full-pose case under `reports`.

    Args:
        reports: The reports root.

    Returns:
        The (newest eval mtime, folder) of each copy, keyed by `case_key`.
    """
    copies: dict[tuple[str, tuple[str, ...]], list[tuple[float,
                                                         pathlib.Path]]] = {}
    for top in sorted(os.listdir(reports)):
        if any(fnmatch.fnmatch(top, pattern) for pattern in NOT_SOURCES):
            continue
        for current, _dirs, files in os.walk(reports / top):
            if not any(
                    name.startswith("eval_") and name.endswith(".json")
                    for name in files):
                continue
            case_dir = pathlib.Path(current)
            key = case_key(case_dir, reports)
            mtime = newest_eval_mtime(case_dir)
            if key is not None and mtime >= 0:
                copies.setdefault(key, []).append((mtime, case_dir))
    return copies


def copy_method_summaries(reports: pathlib.Path, copies: dict) -> int:
    """Copies a `summary.json` into each canonical method folder without one.

    The summary comes from the newest source run that has one.

    Args:
        reports: The reports root.
        copies: The case copies of `find_case_copies`.

    Returns:
        The number of summaries copied.
    """
    added = 0
    for (suite, rel), found in copies.items():
        method_dir = reports / CANONICAL[suite] / suite / rel[0]
        if (method_dir / "summary.json").exists():
            continue
        for _mtime, case_dir in sorted(found, reverse=True):
            source = case_dir
            while source != reports and source.name != rel[0]:
                source = source.parent
            if (source / "summary.json").exists():
                shutil.copy2(source / "summary.json",
                             method_dir / "summary.json")
                added += 1
                break
    return added


def _plan_rows(reports: pathlib.Path, copies: dict) -> list[tuple]:
    """Picks the winning copy of each case and the action that places it.

    Args:
        reports: The reports root.
        copies: The case copies of `find_case_copies`; each list is sorted in
            place.

    Returns:
        One (suite, case, action, winner, destination, copies found) row per
        case.
    """
    rows = []
    for (suite, rel), found in sorted(copies.items()):
        found.sort()
        winner = found[-1][1]
        dest = reports / CANONICAL[suite] / suite / pathlib.Path(*rel)
        if winner.resolve() == dest.resolve():
            action = "keep_canonical"
        elif dest.exists():
            action = "replace"
        else:
            action = "copy"
        rows.append((suite, "/".join(rel), action, winner, dest, len(found)))
    return rows


def _apply_rows(rows: list[tuple], archive: pathlib.Path) -> None:
    """Writes the manifest and copies each winner into its canonical folder."""
    archive.mkdir(parents=True, exist_ok=True)
    with open(archive / "manifest.tsv", "w", encoding="utf-8") as handle:
        handle.write("suite\tcase\taction\tsource\tdestination\tcopies_found\n")
        for row in rows:
            handle.write("\t".join(map(str, row)) + "\n")
    for suite, rel, action, winner, dest, _count in rows:
        if action == "keep_canonical":
            continue
        if action == "replace":
            old = archive / "replaced_canonical" / suite / rel
            old.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dest), str(old))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(winner, dest, symlinks=True)


def main() -> None:
    """Plans the merge; with `--execute` applies it and archives the rest.

    After applying, the result is checked: cases left without an eval file
    are printed.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reports_root", type=pathlib.Path, required=True)
    parser.add_argument("--tag", default="consolidate_20261013")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    reports = args.reports_root.resolve()
    archive = reports / "_archive" / args.tag

    copies = find_case_copies(reports)
    rows = _plan_rows(reports, copies)

    counts = {
        action: sum(row[2] == action for row in rows)
        for action in ("keep_canonical", "copy", "replace")
    }
    to_archive = [
        top for top in sorted(os.listdir(reports)) if top not in KEEP_TOP
    ]
    print(f"cases: {len(rows)} | {counts}")
    for top in to_archive:
        print(f"  archive: {top}")
    if not args.execute:
        return

    _apply_rows(rows, archive)
    print(f"method summaries added: {copy_method_summaries(reports, copies)}")
    for top in to_archive:
        shutil.move(str(reports / top), str(archive / top))
    missing = [row for row in rows if newest_eval_mtime(row[4]) < 0]
    print(f"archived {len(to_archive)} folders into {archive}; cases without "
          f"an eval file afterwards: {len(missing)}")
    for row in missing[:20]:
        print("  MISSING", row)


if __name__ == "__main__":
    main()
