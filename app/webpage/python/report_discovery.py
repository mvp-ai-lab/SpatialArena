# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Discovery of per-method benchmark report roots and recorded case failures.

Benchmark jobs write evaluator outputs to
`<eval_root>/reports/[<batch>/]<suite>/<method_id>[/<chunk>]/<Task>/<dataset>/
<scene>/eval_*.json` and drop a `summary.json` (with `method`, `base_method`
and `benchmark_track`) at the run root. A report root here is that run root:

1. the nearest ancestor of an `eval_*.json` that holds a `summary.json` with
   a `method` key; otherwise
2. the directory just above the first task-named path component; otherwise
3. the eval file's own directory.

Batches whose top-level directory name matches an exclude glob (by default
private test/verification runs such as `_parallel_eval_test` or `*_verify*`)
are skipped entirely.

The module also reads the case failures the runs recorded
(`read_case_failures`) and the full-standard case lists (`read_case_lists`).
The dashboard collector (collect_benchmark_metrics.py) uses all three.
"""

from __future__ import annotations

import csv
import dataclasses
import fnmatch
import json
import os
import pathlib
from collections.abc import Iterable, Sequence

DEFAULT_EXCLUDE_PATTERNS: tuple[str,
                                ...] = ("_*", "*verify*", "*smoke*", "*debug*")
TASK_DIRECTORY_NAMES = frozenset({
    "MonoDepth", "VideoDepth", "MVRecon", "SfM", "SLAM", "WeakTextureSfM",
    "PoseEstimation"
})
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


@dataclasses.dataclass(frozen=True)
class ReportRoot:
    """One method run root discovered under a reports tree.

    Attributes:
        path: The run root.
        reports_root: The reports tree it was found under.
        batch: The batch directories above the suite, joined by "/"; "" when
            there are none.
        suite: The suite directory ("full-*"); "" when there is none.
        method: The raw method ID, e.g. "PI3_X_rgb_cameras".
        base_method: The method without its track, e.g. "PI3_X".
        track: The input track, e.g. "rgb_cameras".
        has_summary: Whether the root holds a summary.json.
        eval_files: The number of eval_*.json files under the root.
    """

    path: pathlib.Path
    reports_root: pathlib.Path
    batch: str
    suite: str
    method: str
    base_method: str
    track: str
    has_summary: bool
    eval_files: int

    def as_dict(self) -> dict:
        """Returns the root as a JSON-serializable dict."""
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


@dataclasses.dataclass(frozen=True)
class CaseFailure:
    """One case failure a run recorded.

    Attributes:
        method: The raw method ID.
        task: The task.
        dataset: The dataset; "" when the record names none.
        scene: The scene; "" when the record names none.
        status: The failure cause, e.g. "gpu_oom" or "error".
        source_path: The file the failure was read from.
    """

    method: str
    task: str
    dataset: str
    scene: str
    status: str
    source_path: pathlib.Path


def split_method_track(method_id: str) -> tuple[str, str]:
    """Splits a raw method ID into its base method and track.

    For example "PI3_X_rgb_cameras" becomes ("PI3_X", "rgb_cameras"); an ID
    without a track suffix is the "rgb" track.

    Args:
        method_id: The raw method ID.

    Returns:
        The (base method, track) pair.
    """
    for track in TRACK_SUFFIXES:
        suffix = f"_{track}"
        if method_id.endswith(suffix) and len(method_id) > len(suffix):
            return method_id[:-len(suffix)], track
    return method_id, "rgb"


def is_excluded(relative_path: pathlib.Path, patterns: Sequence[str]) -> bool:
    """Returns whether the first component of a path matches an exclude glob."""
    if not relative_path.parts:
        return False
    head = relative_path.parts[0]
    return any(fnmatch.fnmatchcase(head, pattern) for pattern in patterns)


def _read_summary_method(path: pathlib.Path) -> dict | None:
    """Returns a summary.json payload that names a method, else None."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not payload.get("method"):
        return None
    return payload


def _fallback_root(eval_dir: pathlib.Path,
                   reports_root: pathlib.Path) -> pathlib.Path:
    """Returns the run root of an eval directory that has no summary.json.

    That is the directory above the first task-named component below the
    reports tree's top level, or the eval directory itself.
    """
    relative_parts = eval_dir.relative_to(reports_root).parts
    for index, part in enumerate(relative_parts):
        if part in TASK_DIRECTORY_NAMES and index > 0:
            return reports_root.joinpath(*relative_parts[:index])
    return eval_dir


def _batch_and_suite(root: pathlib.Path,
                     reports_root: pathlib.Path) -> tuple[str, str]:
    """Returns the batch and suite of a run root (see `ReportRoot`)."""
    parts = root.relative_to(reports_root).parts
    suite_index = next(
        (index for index, part in enumerate(parts) if part.startswith("full-")),
        None)
    if suite_index is None:
        if len(parts) > 1:
            return parts[0], ""
        return "", ""
    return "/".join(parts[:suite_index]), parts[suite_index]


def _scan_reports_tree(
    reports_root: pathlib.Path, exclude_patterns: Sequence[str]
) -> tuple[dict[pathlib.Path, dict], dict[pathlib.Path, int]]:
    """Walks a reports tree, skipping excluded top-level batches.

    Args:
        reports_root: The resolved reports tree.
        exclude_patterns: Globs of top-level batch directories to skip.

    Returns:
        The summary.json payloads that name a method, by directory, and the
        number of eval_*.json files, by directory that has any.
    """
    summary_roots: dict[pathlib.Path, dict] = {}
    eval_counts: dict[pathlib.Path, int] = {}
    for current, dirnames, filenames in os.walk(reports_root):
        current_path = pathlib.Path(current)
        relative = current_path.relative_to(reports_root)
        if relative.parts and is_excluded(relative, exclude_patterns):
            dirnames[:] = []
            continue
        if current_path == reports_root:
            dirnames[:] = [
                name for name in dirnames
                if not is_excluded(pathlib.Path(name), exclude_patterns)
            ]
        if "summary.json" in filenames:
            payload = _read_summary_method(current_path / "summary.json")
            if payload is not None:
                summary_roots[current_path] = payload
        count = sum(1 for name in filenames
                    if name.startswith("eval_") and name.endswith(".json"))
        if count:
            eval_counts[current_path] = count
    return summary_roots, eval_counts


def _outermost_root_evals(
        summary_roots: dict[pathlib.Path, dict], eval_counts: dict[pathlib.Path,
                                                                   int],
        reports_root: pathlib.Path) -> tuple[list[pathlib.Path], dict]:
    """Attributes eval files to run roots and folds nested roots.

    Each eval directory goes to its nearest ancestor with a summary.json, or
    to its fallback root. Roots nested inside another root are folded into
    the outermost one, so eval_*.json files are never attributed twice.

    Args:
        summary_roots: The summary.json payloads by directory.
        eval_counts: The number of eval_*.json files by directory.
        reports_root: The resolved reports tree.

    Returns:
        The outermost roots, and the eval file counts by root (the counts of
        the outermost roots include those of their nested roots).
    """
    root_evals: dict[pathlib.Path, int] = {}
    for eval_dir, count in eval_counts.items():
        root = next((parent for parent in (eval_dir, *eval_dir.parents)
                     if parent in summary_roots), None)
        if root is None:
            root = _fallback_root(eval_dir, reports_root)
        root_evals[root] = root_evals.get(root, 0) + count
    for root in summary_roots:
        root_evals.setdefault(root, 0)

    candidate_roots = sorted(root_evals, key=lambda path: len(path.parts))
    outermost: list[pathlib.Path] = []
    for root in candidate_roots:
        parent = next((kept for kept in outermost if kept in root.parents),
                      None)
        if parent is None:
            outermost.append(root)
        else:
            root_evals[parent] += root_evals[root]
    return outermost, root_evals


def _report_root(root: pathlib.Path, reports_root: pathlib.Path,
                 summary_roots: dict[pathlib.Path,
                                     dict], eval_files: int) -> ReportRoot:
    """Builds the `ReportRoot` of an outermost run root.

    The method comes from the root's summary.json, else from that of a
    nested root, else from the root's directory name.
    """
    payload = summary_roots.get(root)
    if payload is None:
        payload = next((summary_roots[nested]
                        for nested in summary_roots
                        if root in nested.parents), None)
    if payload:
        method = str(payload.get("method"))
    else:
        method = root.name
    fallback_base, fallback_track = split_method_track(method)
    if payload:
        base_method = str(payload.get("base_method") or fallback_base)
        track = str(payload.get("benchmark_track") or fallback_track)
    else:
        base_method = fallback_base
        track = fallback_track
    batch, suite = _batch_and_suite(root, reports_root)
    return ReportRoot(
        path=root,
        reports_root=reports_root,
        batch=batch,
        suite=suite,
        method=method,
        base_method=base_method,
        track=track,
        has_summary=root in summary_roots,
        eval_files=eval_files,
    )


def discover_report_roots(
    reports_roots: Iterable[pathlib.Path],
    exclude_patterns: Sequence[str] = DEFAULT_EXCLUDE_PATTERNS,
) -> list[ReportRoot]:
    """Returns every method report root with an eval_*.json or a summary.json.

    Roots nested inside another discovered root are folded into the outermost
    one so eval_*.json files are never attributed twice.

    Args:
        reports_roots: The reports trees to search; missing ones are skipped.
        exclude_patterns: Globs of top-level batch directories to skip.

    Returns:
        The roots, sorted by method, batch, suite and path.
    """
    discovered: dict[pathlib.Path, ReportRoot] = {}
    for reports_root in reports_roots:
        reports_root = pathlib.Path(reports_root).resolve()
        if not reports_root.is_dir():
            continue
        summary_roots, eval_counts = _scan_reports_tree(reports_root,
                                                        exclude_patterns)
        outermost, root_evals = _outermost_root_evals(summary_roots,
                                                      eval_counts, reports_root)
        for root in outermost:
            discovered[root] = _report_root(root, reports_root, summary_roots,
                                            root_evals[root])
    return sorted(discovered.values(),
                  key=lambda item:
                  (item.method, item.batch, item.suite, str(item.path)))


# A failed eval_*.json's error_type -> case-failure status
# (benchmark_analyzer.CASE_FAILURE_LABELS).
EVAL_ERROR_STATUSES = {"CUDA_OOM": "gpu_oom", "HOST_MEMORY_OOM": "host_oom"}
# Without an error_type, a message fragment names the status; anything else
# is a plain "error".
EVAL_ERROR_MESSAGES = (
    ("constant prediction", "degenerate_constant_prediction"),
    ("point cloud is empty", "empty_point_cloud"),
    ("input frames have a predicted pose", "insufficient_predicted_poses"),
)
# A job watchdog's outcome (watchdog_execution_outcomes/*.json) -> status,
# for the scenes of the job it killed.
WATCHDOG_STATUSES = {"OUT_OF_MEMORY": "oom"}
TASK_DIRS = ("MonoDepth", "VideoDepth", "MVRecon", "SfM", "SLAM")


def _case_from_scene_dir(
        path: pathlib.Path) -> tuple[str, str, str, str] | None:
    """Returns the case of a per-scene file.

    Args:
        path: A file `<METHOD>/<Task>/<dataset>/<scene>/<file>`.

    Returns:
        The (method, task, dataset, scene) case, or None when the path does
        not have that layout.
    """
    parts = path.parts
    if len(parts) < 5 or parts[-4] not in TASK_DIRS:
        return None
    return parts[-5], parts[-4], parts[-3], parts[-2]


def eval_failure_status(payload: dict) -> str | None:
    """Returns the case-failure status of a failed eval_*.json payload.

    Args:
        payload: An eval_*.json payload.

    Returns:
        The status when the payload has no scene result and records an error
        (from its error_type, else from its message, else "error"); None
        otherwise.
    """
    if payload.get("num_scene_results") or payload.get(
            "scene_results") or not payload.get("errors"):
        return None
    if isinstance(payload["errors"][0], dict):
        error = payload["errors"][0]
    else:
        error = {}
    status = EVAL_ERROR_STATUSES.get(str(error.get("error_type") or ""))
    if status:
        return status
    message = str(error.get("error") or "").lower()
    return next((status for fragment, status in EVAL_ERROR_MESSAGES
                 if fragment in message), "error")


def _read_failure_tsv(path: pathlib.Path) -> list[CaseFailure]:
    """Reads a case-list runner's case_failures*.tsv."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            CaseFailure(
                method=row["method"].strip(),
                task=row["task"].strip(),
                dataset=(row.get("dataset") or "").strip(),
                scene=(row.get("scene") or "").strip(),
                # The runner's files put the exit code in "status" and the
                # cause (gpu_oom, ...) in "reason".
                status=(row.get("reason") or row.get("status") or "").strip(),
                source_path=path)
            for row in csv.DictReader(handle, delimiter="\t")
            if row.get("method") and row.get("task")
        ]


def _read_inference_failure(path: pathlib.Path) -> list[CaseFailure]:
    """Reads a scene folder's inference_failure.tsv.

    The file holds one "<exit code>\\t<reason>" line, e.g. "1\\tgpu_oom".
    """
    case = _case_from_scene_dir(path)
    fields = path.read_text(encoding="utf-8").strip().split("\t")
    if case is None or len(fields) < 2 or not fields[1].strip():
        return []
    return [CaseFailure(*case, status=fields[1].strip(), source_path=path)]


def _read_eval_failure(path: pathlib.Path) -> list[CaseFailure]:
    """Reads a scene's eval_*.json; a failure only if it failed."""
    case = _case_from_scene_dir(path)
    text = path.read_text(encoding="utf-8")
    if case is None or '"errors": []' in text:
        return []
    status = eval_failure_status(json.loads(text))
    if not status:
        return []
    return [CaseFailure(*case, status=status, source_path=path)]


def _read_watchdog_outcome(path: pathlib.Path) -> list[CaseFailure]:
    """Reads a job watchdog's outcome: one failure per scene of the job."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    status = WATCHDOG_STATUSES.get(str(payload.get("status") or ""))
    item = (payload.get("job") or {}).get("work_item") or {}
    if not status or not item.get("model") or not item.get("task"):
        return []
    track = str(item.get("track") or "rgb")
    if track == "rgb":
        method = item["model"]
    else:
        method = f"{item['model']}_{track}"
    return [
        CaseFailure(method=method,
                    task=item["task"],
                    dataset=str(item.get("dataset") or ""),
                    scene=str(scene),
                    status=status,
                    source_path=path) for scene in item.get("scenes") or ()
    ]


def read_case_failures(
        report_roots: Iterable[pathlib.Path]) -> list[CaseFailure]:
    """Returns every case failure the runs recorded under the given roots.

    The records are the case-list runner's case_failures*.tsv, a scene's
    inference_failure.tsv, a failed eval_*.json (no scene result, an error
    such as CUDA_OOM) and a job watchdog's watchdog_execution_outcomes/*.json
    (e.g. OUT_OF_MEMORY). The same case recorded twice keeps its first
    record, in that order. A SLAM sequence is one case per range. Unreadable
    files are skipped.

    Args:
        report_roots: The method report roots to search.

    Returns:
        The failures.
    """
    readers = (
        ("case_failures*.tsv", _read_failure_tsv),
        ("inference_failure.tsv", _read_inference_failure),
        ("eval_*.json", _read_eval_failure),
        ("watchdog_execution_outcomes/*.json", _read_watchdog_outcome),
    )
    failures: dict[tuple[str, ...], CaseFailure] = {}
    for root in report_roots:
        for pattern, reader in readers:
            for path in sorted(pathlib.Path(root).rglob(pattern)):
                try:
                    found = reader(path)
                except (OSError, UnicodeDecodeError, ValueError, KeyError,
                        csv.Error):
                    continue
                # A SLAM sequence is a case per range (its report folder
                # full-slam-<range>).
                slam_range = next((part for part in path.parts
                                   if part.startswith("full-slam-")), "")
                for failure in found:
                    key = (failure.method, failure.task, failure.dataset,
                           failure.scene, slam_range)
                    failures.setdefault(key, failure)
    return list(failures.values())


def read_case_lists(paths: Iterable[pathlib.Path]) -> set[tuple[str, str, str]]:
    """Parses case lists into (task, dataset, scene) keys.

    Args:
        paths: Case lists with "task|dataset|scene|interval|order" lines;
            blank lines, "#" comments and lines with fewer than three fields
            are skipped.

    Returns:
        The cases.
    """
    cases: set[tuple[str, str, str]] = set()
    for path in paths:
        with pathlib.Path(path).open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split("|")
                if len(fields) < 3:
                    continue
                cases.add(
                    (fields[0].strip(), fields[1].strip(), fields[2].strip()))
    return cases
