# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Merge scattered full-depth / full-pose report folders into the one folder per suite and method.

Runs that set their own ``RUN_ID`` / ``REPORT_DIR`` (retries, fixes, resumes) left copies of the same cases in
several trees under ``reports/``. For every case (suite, METHOD_ID, Task, Dataset, Scene[, order]) the copy whose
``eval_*.json`` is newest wins and is copied into the canonical folder (``spatialarena/report_layout.py``)::

    reports/benchmark_full_reports/full-depth/<METHOD_ID>/<Task>/<Dataset>/<Scene>
    reports/benchmark_full_pose_reports/full-pose/<METHOD_ID>/<Task>/<Dataset>/<Scene>

An older canonical copy it replaces moves to ``_archive/<tag>/replaced_canonical/``; afterwards every other
top-level folder of ``reports/`` (per-run trees, tests) moves to ``_archive/<tag>/``. Nothing is deleted, and
``_archive/<tag>/manifest.tsv`` records every action. Test (``_*``) and ``*verify*`` folders are archived, never
merged. SLAM reports already live in their canonical tree and are not touched.

    python scripts/benchmark/consolidate_report_folders.py --reports_root <eval root>/reports            # plan
    python scripts/benchmark/consolidate_report_folders.py --reports_root <eval root>/reports --execute  # apply
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import shutil
from pathlib import Path

CANONICAL = {"full-depth": "benchmark_full_reports", "full-pose": "benchmark_full_pose_reports"}
KEEP_TOP = {"benchmark_full_reports", "benchmark_full_pose_reports", "benchmark_full_slam_reports",
            "benchmark_sampled_reports", "_archive"}
NOT_SOURCES = ("benchmark_full_slam_reports", "benchmark_sampled_reports", "_*", "*verify*")
CHUNK = re.compile(r"^chunk_\d+$")


def case_key(case_dir: Path, reports: Path) -> tuple[str, tuple[str, ...]] | None:
    """(suite, case path below the suite folder) of a case folder, with retry ``chunk_NN`` levels dropped."""
    parts = case_dir.relative_to(reports).parts
    for index, part in enumerate(parts):
        if part in CANONICAL:
            rel = tuple(p for p in parts[index + 1:] if not CHUNK.match(p))
            return (part, rel) if len(rel) >= 2 else None
    return None


def newest_eval_mtime(case_dir: Path) -> float:
    """Latest mtime of the folder's non-empty ``eval_*.json`` files, or -1 if it has none."""
    times = [path.stat().st_mtime for path in case_dir.glob("eval_*.json") if path.stat().st_size > 0]
    return max(times) if times else -1.0


def find_case_copies(reports: Path) -> dict[tuple[str, tuple[str, ...]], list[tuple[float, Path]]]:
    """Every copy of every full-depth / full-pose case under ``reports``, keyed by case."""
    copies: dict[tuple[str, tuple[str, ...]], list[tuple[float, Path]]] = {}
    for top in sorted(os.listdir(reports)):
        if any(fnmatch.fnmatch(top, pattern) for pattern in NOT_SOURCES):
            continue
        for current, _dirs, files in os.walk(reports / top):
            if not any(name.startswith("eval_") and name.endswith(".json") for name in files):
                continue
            case_dir = Path(current)
            key = case_key(case_dir, reports)
            mtime = newest_eval_mtime(case_dir)
            if key is not None and mtime >= 0:
                copies.setdefault(key, []).append((mtime, case_dir))
    return copies


def copy_method_summaries(reports: Path, copies: dict) -> int:
    """Give each canonical method folder without a ``summary.json`` the newest one of its source runs."""
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
                shutil.copy2(source / "summary.json", method_dir / "summary.json")
                added += 1
                break
    return added


def main() -> None:
    """Plan the merge; with ``--execute`` apply it, archive the other folders and check the result."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reports_root", type=Path, required=True)
    parser.add_argument("--tag", default="consolidate_20261013")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    reports = args.reports_root.resolve()
    archive = reports / "_archive" / args.tag

    copies = find_case_copies(reports)
    rows = []
    for (suite, rel), found in sorted(copies.items()):
        found.sort()
        winner = found[-1][1]
        dest = reports / CANONICAL[suite] / suite / Path(*rel)
        if winner.resolve() == dest.resolve():
            action = "keep_canonical"
        else:
            action = "replace" if dest.exists() else "copy"
        rows.append((suite, "/".join(rel), action, winner, dest, len(found)))

    counts = {action: sum(row[2] == action for row in rows) for action in ("keep_canonical", "copy", "replace")}
    to_archive = [top for top in sorted(os.listdir(reports)) if top not in KEEP_TOP]
    print(f"cases: {len(rows)} | {counts}")
    for top in to_archive:
        print(f"  archive: {top}")
    if not args.execute:
        return

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
    print(f"method summaries added: {copy_method_summaries(reports, copies)}")
    for top in to_archive:
        shutil.move(str(reports / top), str(archive / top))
    missing = [row for row in rows if newest_eval_mtime(row[4]) < 0]
    print(f"archived {len(to_archive)} folders into {archive}; cases without an eval file afterwards: {len(missing)}")
    for row in missing[:20]:
        print("  MISSING", row)


if __name__ == "__main__":
    main()
