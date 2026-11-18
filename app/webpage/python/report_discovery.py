# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Discover per-method benchmark report roots under one or more reports trees.

Benchmark jobs write evaluator outputs to
``<eval_root>/reports/[<batch>/]<suite>/<method_id>[/<chunk>]/<Task>/<dataset>/<scene>/eval_*.json``
and drop a ``summary.json`` (with ``method``, ``base_method`` and
``benchmark_track``) at the run root. A *report root* here is that run root:

1. the nearest ancestor of an ``eval_*.json`` that holds a ``summary.json``
   with a ``method`` key; otherwise
2. the directory just above the first task-named path component; otherwise
3. the eval file's own directory.

Batches whose top-level directory name matches an exclude glob (by default
private test/verification runs such as ``_parallel_eval_test`` or
``*_verify*``) are skipped entirely.
"""

from __future__ import annotations

import csv
import fnmatch
import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_EXCLUDE_PATTERNS: tuple[str, ...] = ("_*", "*verify*", "*smoke*", "*debug*")
TASK_DIRECTORY_NAMES = frozenset(
    {"MonoDepth", "VideoDepth", "MVRecon", "SfM", "SLAM", "WeakTextureSfM", "PoseEstimation"}
)
TRACK_SUFFIXES: tuple[str, ...] = (
    "rgb_sparse_depth_cameras",
    "rgb_sparse_depth",
    "rgb_masked_depth_cameras",
    "rgb_masked_depth",
    "rgb_depth_cameras",
    "rgb_cameras_depth",
    "rgb_cameras",
    "rgb_depth",
)


@dataclass(frozen=True)
class ReportRoot:
    """One method run root discovered under a reports tree."""

    path: Path
    reports_root: Path
    batch: str
    suite: str
    method: str
    base_method: str
    track: str
    has_summary: bool
    eval_files: int

    def as_dict(self) -> dict:
        return {
            "path": str(self.path),
            "reports_root": str(self.reports_root),
            "batch": self.batch,
            "suite": self.suite,
            "method": self.method,
            "base_method": self.base_method,
            "track": self.track,
            "has_summary": self.has_summary,
            "eval_files": self.eval_files,
        }


@dataclass(frozen=True)
class CaseFailure:
    method: str
    task: str
    dataset: str
    scene: str
    status: str
    source_path: Path


def split_method_track(method_id: str) -> tuple[str, str]:
    """Split ``PI3_X_rgb_cameras`` into ``("PI3_X", "rgb_cameras")``; plain IDs are the ``rgb`` track."""
    for track in TRACK_SUFFIXES:
        suffix = f"_{track}"
        if method_id.endswith(suffix) and len(method_id) > len(suffix):
            return method_id[: -len(suffix)], track
    return method_id, "rgb"


def is_excluded(relative_path: Path, patterns: Sequence[str]) -> bool:
    """True when the first component of ``relative_path`` matches any exclude glob."""
    if not relative_path.parts:
        return False
    head = relative_path.parts[0]
    return any(fnmatch.fnmatchcase(head, pattern) for pattern in patterns)


def _read_summary_method(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not payload.get("method"):
        return None
    return payload


def _fallback_root(eval_dir: Path, reports_root: Path) -> Path:
    relative_parts = eval_dir.relative_to(reports_root).parts
    for index, part in enumerate(relative_parts):
        if part in TASK_DIRECTORY_NAMES and index > 0:
            return reports_root.joinpath(*relative_parts[:index])
    return eval_dir


def _batch_and_suite(root: Path, reports_root: Path) -> tuple[str, str]:
    parts = root.relative_to(reports_root).parts
    suite_index = next((index for index, part in enumerate(parts) if part.startswith("full-")), None)
    if suite_index is None:
        return (parts[0] if len(parts) > 1 else ""), ""
    return "/".join(parts[:suite_index]), parts[suite_index]


def discover_report_roots(
    reports_roots: Iterable[Path],
    exclude_patterns: Sequence[str] = DEFAULT_EXCLUDE_PATTERNS,
) -> list[ReportRoot]:
    """Return every method report root that holds at least one ``eval_*.json`` or a ``summary.json``.

    Roots nested inside another discovered root are folded into the outermost one so
    ``eval_*.json`` files are never attributed twice.
    """
    discovered: dict[Path, ReportRoot] = {}
    for reports_root in reports_roots:
        reports_root = Path(reports_root).resolve()
        if not reports_root.is_dir():
            continue
        summary_roots: dict[Path, dict] = {}
        eval_counts: dict[Path, int] = {}
        for current, dirnames, filenames in os.walk(reports_root):
            current_path = Path(current)
            relative = current_path.relative_to(reports_root)
            if relative.parts and is_excluded(relative, exclude_patterns):
                dirnames[:] = []
                continue
            if current_path == reports_root:
                dirnames[:] = [name for name in dirnames if not is_excluded(Path(name), exclude_patterns)]
            if "summary.json" in filenames:
                payload = _read_summary_method(current_path / "summary.json")
                if payload is not None:
                    summary_roots[current_path] = payload
            count = sum(1 for name in filenames if name.startswith("eval_") and name.endswith(".json"))
            if count:
                eval_counts[current_path] = count

        root_evals: dict[Path, int] = {}
        for eval_dir, count in eval_counts.items():
            root = next((parent for parent in (eval_dir, *eval_dir.parents) if parent in summary_roots), None)
            if root is None:
                root = _fallback_root(eval_dir, reports_root)
            root_evals[root] = root_evals.get(root, 0) + count
        for root in summary_roots:
            root_evals.setdefault(root, 0)

        candidate_roots = sorted(root_evals, key=lambda path: len(path.parts))
        outermost: list[Path] = []
        for root in candidate_roots:
            parent = next((kept for kept in outermost if kept in root.parents), None)
            if parent is None:
                outermost.append(root)
            else:
                root_evals[parent] += root_evals[root]

        for root in outermost:
            payload = summary_roots.get(root)
            if payload is None:
                payload = next(
                    (summary_roots[nested] for nested in summary_roots if root in nested.parents), None
                )
            method = str(payload.get("method")) if payload else root.name
            fallback_base, fallback_track = split_method_track(method)
            base_method = str(payload.get("base_method") or fallback_base) if payload else fallback_base
            track = str(payload.get("benchmark_track") or fallback_track) if payload else fallback_track
            batch, suite = _batch_and_suite(root, reports_root)
            discovered[root] = ReportRoot(
                path=root,
                reports_root=reports_root,
                batch=batch,
                suite=suite,
                method=method,
                base_method=base_method,
                track=track,
                has_summary=root in summary_roots,
                eval_files=root_evals[root],
            )
    return sorted(discovered.values(), key=lambda item: (item.method, item.batch, item.suite, str(item.path)))


def read_case_failures(report_roots: Iterable[Path]) -> list[CaseFailure]:
    """Read every ``case_failures*.tsv`` written by the case-list runner under the given roots."""
    failures: list[CaseFailure] = []
    for root in report_roots:
        for path in sorted(Path(root).rglob("case_failures*.tsv")):
            try:
                with path.open("r", encoding="utf-8", newline="") as handle:
                    for row in csv.DictReader(handle, delimiter="\t"):
                        if not row.get("method") or not row.get("task"):
                            continue
                        failures.append(
                            CaseFailure(
                                method=row["method"].strip(),
                                task=row["task"].strip(),
                                dataset=(row.get("dataset") or "").strip(),
                                scene=(row.get("scene") or "").strip(),
                                status=(row.get("status") or "").strip(),
                                source_path=path,
                            )
                        )
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
    return failures


def read_case_lists(paths: Iterable[Path]) -> set[tuple[str, str, str]]:
    """Parse ``task|dataset|scene|interval|order`` case lists into ``(task, dataset, scene)`` keys."""
    cases: set[tuple[str, str, str]] = set()
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split("|")
                if len(fields) < 3:
                    continue
                cases.add((fields[0].strip(), fields[1].strip(), fields[2].strip()))
    return cases
