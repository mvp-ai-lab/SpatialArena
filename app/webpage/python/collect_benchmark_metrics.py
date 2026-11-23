#!/usr/bin/env python3
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Collects every evaluated method/track and exports the dashboard data.

Unlike benchmark_analyzer.py (which starts from a hand-curated summary TSV),
this discovers every method report root under the reports tree,
deduplicates per-scene results across batches (newest eval_*.json wins),
restricts them to the full-standard case lists, and joins the result with
the method registry (spatialarena.methods.METHOD_SPECS) and the task/track
matrix (spatialarena.task_track_matrix), so the dashboard can show which
supported (method, task, track) cells are complete, partial, failed or not
yet run.

Outputs, all under --output-dir:

    summary.tsv     Per (method, task, dataset) aggregated metrics.
    results.json    The dashboard payload (benchmark_analyzer's schema plus
                    the `collection` and `coverage` sections).

Example:

    python app/webpage/python/collect_benchmark_metrics.py \
        --reports-root $SPATIALARENA_ROOT/benchmarks/_evaluation/reports
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import pathlib
import sys
import time
from collections.abc import Iterable, Sequence
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from app.webpage.python import benchmark_analyzer  # noqa: E402
from app.webpage.python import report_discovery  # noqa: E402
from app.webpage.python import viewer_index  # noqa: E402
from scripts.test import summarize_benchmark_reports  # noqa: E402
from spatialarena import methods as method_registry  # noqa: E402
from spatialarena import paths  # noqa: E402
from spatialarena import task_track_matrix as track_matrix  # noqa: E402
from spatialarena.eval import dataset_registry  # noqa: E402
# pylint: enable=wrong-import-position

APP_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = APP_ROOT / "data"
DEFAULT_REPORTS_ROOT = pathlib.Path(
    os.environ.get(
        "BENCHMARK_REPORTS_ROOT",
        f"{paths.SPATIALARENA_ROOT}/benchmarks/_evaluation/reports",
    ))
DEFAULT_CASE_LISTS = (
    REPO_ROOT / "scripts" / "test" / "full_standard_depth_cases.txt",
    REPO_ROOT / "scripts" / "test" / "full_standard_sfm_cases.txt",
    REPO_ROOT / "scripts" / "test" / "full_standard_slam_cases_short.txt",
    REPO_ROOT / "scripts" / "test" / "full_standard_slam_cases_long.txt",
    REPO_ROOT / "scripts" / "test" / "full_standard_slam_cases_large.txt",
)
TRACK_ORDER = (
    "rgb",
    "rgb_cameras",
    "rgb_depth",
    "rgb_masked_depth",
    "rgb_masked_depth_cameras",
    "rgb_sparse_depth",
    "rgb_sparse_depth_cameras",
)

STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"
STATUS_REPORTED_UNSUPPORTED = "reported_unsupported"
STATUS_NOT_RUN = "not_run"
STATUS_UNSUPPORTED = "unsupported"
COVERAGE_STATUSES = (
    STATUS_COMPLETE,
    STATUS_PARTIAL,
    STATUS_FAILED,
    STATUS_REPORTED_UNSUPPORTED,
    STATUS_NOT_RUN,
    STATUS_UNSUPPORTED,
)

SCENES_TSV = "scenes.tsv"

# A (task, dataset, scene) case of a case list.
Case = tuple[str, str, str]


def method_id_for_track(base_method: str, track: str) -> str:
    """Returns the raw method ID that reports use for a track.

    Args:
        base_method: The registered method name, e.g. "PI3_X".
        track: The input track, e.g. "rgb_cameras".

    Returns:
        The base name for the "rgb" track, "<base>_<track>" otherwise.
    """
    if track == "rgb":
        return base_method
    return f"{base_method}_{track}"


def filter_to_case_list(
    rows: Iterable[summarize_benchmark_reports.SceneMetrics],
    cases: set[Case] | None,
) -> tuple[list[summarize_benchmark_reports.SceneMetrics], dict[str, int]]:
    """Keeps only the scenes in the case list.

    Args:
        rows: Per-scene metrics.
        cases: The (task, dataset, scene) cases to keep, or None to keep all.

    Returns:
        The kept rows, and the number of dropped scenes per
        "method|task|dataset", sorted by key.
    """
    rows = list(rows)
    if cases is None:
        return rows, {}
    kept: list[summarize_benchmark_reports.SceneMetrics] = []
    dropped: dict[str, int] = collections.defaultdict(int)
    for row in rows:
        if (row.task, row.dataset, row.scene) in cases:
            kept.append(row)
        else:
            dropped[f"{row.method}|{row.task}|{row.dataset}"] += 1
    return kept, dict(sorted(dropped.items()))


def expected_scenes_by_task(
        cases: set[Case] | None) -> dict[str, dict[str, set[str]]]:
    """Returns the expected scenes of the case list by task and dataset."""
    expected: dict[str, dict[str, set[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(set))
    for task, dataset, scene in cases or ():
        expected[task][dataset].add(scene)
    return expected


def slam_range_case_lists() -> dict[str, set[Case]]:
    """Returns the SLAM case list of each range track.

    The lists are read from scripts/test/full_standard_slam_cases_<range>.txt;
    a range without such a file is left out.
    """
    # pylint: disable=import-outside-toplevel
    from spatialarena import slam_protocol

    lists = {}
    for slam_range in slam_protocol.SLAM_RANGES:
        path = (REPO_ROOT / "scripts" / "test" /
                f"full_standard_slam_cases_{slam_range}.txt")
        if path.is_file():
            lists[slam_range] = report_discovery.read_case_lists([path])
    return lists


def slam_expected_scenes(
        method: str, track: str,
        range_cases: dict[str, set[Case]]) -> dict[str, set[str]]:
    """Returns the SLAM sequences a method and track are expected on.

    These are the sequences of the range tracks the protocol evaluates the
    method on (`slam_protocol.method_slam_ranges`).

    Args:
        method: The base method name.
        track: The input track.
        range_cases: The SLAM case list of each range track.

    Returns:
        The expected sequences by dataset.
    """
    # pylint: disable=import-outside-toplevel
    from spatialarena import slam_protocol

    expected: dict[str, set[str]] = collections.defaultdict(set)
    for slam_range in slam_protocol.method_slam_ranges(method, track):
        for _, dataset, scene in range_cases.get(slam_range, ()):
            expected[dataset].add(scene)
    return expected


def coverage_status(evaluated: int, expected: int, failed: int,
                    reported_unsupported: bool) -> str:
    """Returns the coverage status of one supported cell.

    Args:
        evaluated: The number of expected scenes with a result.
        expected: The number of expected scenes.
        failed: The number of recorded case failures.
        reported_unsupported: Whether the method wrote an
            unsupported_status.json.

    Returns:
        One of the COVERAGE_STATUSES other than STATUS_UNSUPPORTED.
    """
    if evaluated and expected and evaluated >= expected:
        return STATUS_COMPLETE
    if evaluated:
        return STATUS_PARTIAL
    if reported_unsupported:
        return STATUS_REPORTED_UNSUPPORTED
    if failed:
        return STATUS_FAILED
    return STATUS_NOT_RUN


def _supported_cell(
    *,
    task: str,
    track: str,
    base_method: str,
    method_id: str,
    evaluated_by_dataset: dict[str, set[str]],
    expected_by_dataset: dict[str, set[str]],
    range_cases: dict[str, set[Case]],
    cell_failures: Iterable[tuple[str, str]],
    restricted_to_cases: bool,
    unsupported: dict[str, Any] | None,
) -> dict[str, Any]:
    """Builds the coverage cell of a track the method supports.

    Args:
        task: The task.
        track: The input track.
        base_method: The registered method name.
        method_id: The raw method ID of the method on this track.
        evaluated_by_dataset: The evaluated scenes of the method ID on the
            task, by dataset.
        expected_by_dataset: The expected scenes of the task, by dataset.
        range_cases: The SLAM case list of each range track (empty when no
            SLAM cases are expected).
        cell_failures: The (dataset, scene) case failures of the method ID on
            the task.
        restricted_to_cases: Whether the scenes were restricted to case lists.
        unsupported: The method's unsupported status, or None.

    Returns:
        The cell, with its status, scene counts and per-dataset scenes.
    """
    # Only the datasets the track runs on (TRACK_DATASETS): e.g. no MVRecon
    # rgb_depth on dense-depth data; and only their scenes it runs on
    # (TRACK_EXCLUDED_SCENES: e.g. no SLAM rgb_depth on a sequence without
    # depth maps).
    evaluated_by_dataset = {
        dataset: scenes
        for dataset, scenes in evaluated_by_dataset.items()
        if track_matrix.track_runs_on(task, track, dataset)
    }
    expected_by_dataset = {
        dataset: {
            scene
            for scene in scenes
            if track_matrix.track_runs_on_case(task, track, dataset, scene)
        }
        for dataset, scenes in expected_by_dataset.items()
        if track_matrix.track_runs_on(task, track, dataset)
    }
    if task == "SLAM" and range_cases:
        # Only the sequences of the range tracks the method is evaluated on
        # (e.g. no large-range KITTI or VBR for the methods outside the large
        # range).
        in_ranges = slam_expected_scenes(base_method, track, range_cases)
        expected_by_dataset = {
            dataset: scenes & in_ranges[dataset]
            for dataset, scenes in expected_by_dataset.items()
            if in_ranges.get(dataset)
        }
    failed_by_dataset: dict[str, set[str]] = collections.defaultdict(set)
    cell_failures = [
        (dataset, scene)
        for dataset, scene in cell_failures
        if track_matrix.track_runs_on_case(task, track, dataset, scene)
    ]
    for dataset, scene in cell_failures:
        failed_by_dataset[dataset].add(scene)
    datasets = sorted(set(expected_by_dataset) | set(evaluated_by_dataset))
    # Scene names are kept per cell (finished/failed only); the frontend
    # derives missing scenes from the shared expected_scene_lists.
    dataset_cells = {}
    for dataset in datasets:
        evaluated_scenes = evaluated_by_dataset.get(dataset, set())
        failed_scenes = failed_by_dataset.get(dataset, set())
        dataset_cells[dataset] = {
            "evaluated": len(evaluated_by_dataset.get(dataset, ())),
            "expected": len(expected_by_dataset.get(dataset, ())),
            "evaluated_scenes": sorted(evaluated_by_dataset.get(dataset, ())),
            "failed_scenes": sorted(failed_scenes - evaluated_scenes),
        }
    evaluated_count = sum(
        len(scenes) for scenes in evaluated_by_dataset.values())
    expected_count = sum(len(scenes) for scenes in expected_by_dataset.values())
    if restricted_to_cases:
        covered = sum(
            len(evaluated_by_dataset.get(dataset, set()) & scenes)
            for dataset, scenes in expected_by_dataset.items())
    else:
        covered = evaluated_count
    failed_count = len(cell_failures)
    status = coverage_status(covered, expected_count, failed_count, unsupported
                             is not None)
    cell: dict[str, Any] = {
        "status": status,
        "method_id": method_id,
        "evaluated_scenes": covered,
        "expected_scenes": expected_count,
        "failed_cases": failed_count,
        "datasets": dataset_cells,
    }
    if unsupported is not None:
        cell["reason"] = unsupported.get("reason", "")
    return cell


def _method_entry(spec: method_registry.MethodSpec, cells: dict[str, Any],
                  method_totals: dict[str, int],
                  roots_by_method: dict[str, list[str]]) -> dict[str, Any]:
    """Returns the coverage entry of one registered method."""
    report_roots = []
    for track in TRACK_ORDER:
        method_id = method_id_for_track(spec.name, track)
        report_roots.extend(roots_by_method.get(method_id, []))
    return {
        "method": spec.name,
        "display": benchmark_analyzer.display_method_name(spec.name),
        "adapter": spec.adapter,
        "native": spec.native,
        "tracks": list(spec.tracks),
        "notes": spec.notes,
        "report_roots": sorted(report_roots),
        "cells": cells,
        "totals": method_totals,
    }


def _scenes_by_task(
    expected: dict[str, dict[str, set[str]]],
    summarize: Any,
) -> dict[str, dict[str, Any]]:
    """Applies `summarize` to the scenes of each task and dataset.

    Datasets are sorted within each task.
    """
    result = {}
    for task, datasets in expected.items():
        result[task] = {
            dataset: summarize(scenes)
            for dataset, scenes in sorted(datasets.items())
        }
    return result


def build_coverage(
    rows: Sequence[summarize_benchmark_reports.SceneMetrics],
    *,
    cases: set[Case] | None,
    failures_by_cell: dict[tuple[str, str], set[tuple[str, str]]],
    unsupported_statuses: Sequence[dict[str, Any]],
    report_roots: Sequence[report_discovery.ReportRoot],
    method_specs: Sequence[
        method_registry.MethodSpec] = method_registry.METHOD_SPECS,
    task_track_matrix: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """Joins the evaluated scenes with the method registry and track matrix.

    Every (registered method, task, matrix track) cell gets a status:
    "complete" (all expected scenes evaluated), "partial", "failed" (only case
    failures recorded), "reported_unsupported" (an unsupported_status.json
    exists), "not_run", or "unsupported" (the method does not declare that
    track).

    Args:
        rows: The evaluated scenes.
        cases: The expected (task, dataset, scene) cases, or None when the
            scenes were not restricted to case lists.
        failures_by_cell: The (dataset, scene) case failures by (method ID,
            task).
        unsupported_statuses: The methods' unsupported_status.json payloads.
        report_roots: The discovered report roots.
        method_specs: The registered methods.
        task_track_matrix: The tracks of each task; defaults to
            `task_track_matrix.TASK_TRACK_MATRIX`.

    Returns:
        The dashboard's "coverage" section.
    """
    if task_track_matrix is None:
        task_track_matrix = track_matrix.TASK_TRACK_MATRIX
    expected = expected_scenes_by_task(cases)
    if cases is not None and "SLAM" in expected:
        range_cases = slam_range_case_lists()
    else:
        range_cases = {}
    evaluated: dict[tuple[str, str],
                    dict[str, set[str]]] = collections.defaultdict(
                        lambda: collections.defaultdict(set))
    for row in rows:
        evaluated[(row.method, row.task)][row.dataset].add(row.scene)
    unsupported_by_method = {
        status["method"]: status for status in unsupported_statuses
    }
    roots_by_method: dict[str, list[str]] = collections.defaultdict(list)
    for root in report_roots:
        roots_by_method[root.method].append(str(root.path))

    tasks = list(task_track_matrix)
    registered_ids: set[str] = set()
    method_entries: list[dict[str, Any]] = []
    status_totals = {status: 0 for status in COVERAGE_STATUSES}
    for spec in method_specs:
        cells: dict[str, dict[str, Any]] = {}
        method_totals = {status: 0 for status in COVERAGE_STATUSES}
        for task in tasks:
            supported = set(track_matrix.runnable_tracks(task, spec))
            for track in task_track_matrix[task]:
                method_id = method_id_for_track(spec.name, track)
                registered_ids.add(method_id)
                if track not in supported:
                    cells[f"{task}|{track}"] = {
                        "status": STATUS_UNSUPPORTED,
                        "method_id": method_id
                    }
                    method_totals[STATUS_UNSUPPORTED] += 1
                    continue
                cell = _supported_cell(
                    task=task,
                    track=track,
                    base_method=spec.name,
                    method_id=method_id,
                    evaluated_by_dataset=evaluated.get((method_id, task), {}),
                    expected_by_dataset=expected.get(task, {}),
                    range_cases=range_cases,
                    cell_failures=failures_by_cell.get((method_id, task), ()),
                    restricted_to_cases=cases is not None,
                    unsupported=unsupported_by_method.get(method_id),
                )
                cells[f"{task}|{track}"] = cell
                method_totals[cell["status"]] += 1
        for status, count in method_totals.items():
            status_totals[status] += count
        method_entries.append(
            _method_entry(spec, cells, method_totals, roots_by_method))

    evaluated_ids = {row.method for row in rows}
    unregistered = sorted(evaluated_ids - registered_ids)
    supported_cells = sum(count for status, count in status_totals.items()
                          if status != STATUS_UNSUPPORTED)
    methods_with_results = {
        entry["method"]
        for entry in method_entries
        if entry["totals"][STATUS_COMPLETE] or entry["totals"][STATUS_PARTIAL]
    }
    summary = {
        "registered_methods": len(method_specs),
        "methods_with_results": len(methods_with_results),
        "supported_cells": supported_cells,
    }
    summary.update(status_totals)
    return {
        "tasks": tasks,
        "task_display_names": {
            task: benchmark_analyzer.display_task_name(task) for task in tasks
        },
        "tracks": [
            track for track in TRACK_ORDER
            if any(track in tracks for tracks in task_track_matrix.values())
        ],
        "task_tracks": {
            task: list(tracks) for task, tracks in task_track_matrix.items()
        },
        "expected_scenes": _scenes_by_task(expected, len),
        "expected_scene_lists": _scenes_by_task(expected, sorted),
        "statuses": list(COVERAGE_STATUSES),
        "methods": method_entries,
        "unregistered_methods": unregistered,
        "summary": summary,
    }


def failure_slam_range(failure) -> str:
    """Returns the SLAM range of a case failure.

    The range comes from the failure's report folder ("full-slam-<range>").

    Args:
        failure: A `report_discovery.CaseFailure`.

    Returns:
        The range, or "" when the folder names none.
    """
    for part in pathlib.Path(failure.source_path).parts:
        if part.startswith("full-slam-"):
            return part.removeprefix("full-slam-")
    return ""


def read_failures_by_cell(
    report_roots: Sequence[report_discovery.ReportRoot]
) -> dict[tuple[str, str], set[tuple[str, str]]]:
    """Returns the (dataset, scene) case failures by (method ID, task).

    Failures on a SLAM range the protocol does not evaluate the method on are
    left out.
    """
    cells: dict[tuple[str, str], set[tuple[str,
                                           str]]] = collections.defaultdict(set)
    failures = report_discovery.read_case_failures(
        root.path for root in report_roots)
    for failure in failures:
        if not benchmark_analyzer.on_slam_protocol(failure.method, failure.task,
                                                   failure_slam_range(failure)):
            continue
        cells[(failure.method, failure.task)].add(
            (failure.dataset, failure.scene))
    return cells


def render_scene_tsv(
        rows: Sequence[summarize_benchmark_reports.SceneMetrics]) -> str:
    """Renders one TSV line per evaluated scene with its metrics.

    Each line holds the method, task, dataset, SLAM range ("variant") and
    scene, then the metrics: what the leaderboard scores per scene
    (`benchmark_analyzer.add_scene_scores`).

    Args:
        rows: The evaluated scenes.

    Returns:
        The TSV text, with a header line and a trailing newline.
    """
    metric_keys = sorted({key for row in rows for key in row.metrics})
    lines = [
        "\t".join(
            ["method", "task", "dataset", "variant", "scene", *metric_keys])
    ]
    for row in rows:
        values = []
        for key in metric_keys:
            if key in row.metrics:
                values.append(f"{row.metrics[key]:.10g}")
            else:
                values.append("")
        lines.append("\t".join([
            row.method, row.task, row.dataset, row.variant, row.scene, *values
        ]))
    return "\n".join(lines) + "\n"


def _discover_roots(
    reports_roots: Sequence[pathlib.Path],
    extra_report_roots: Sequence[pathlib.Path],
    exclude_patterns: Sequence[str],
) -> list[report_discovery.ReportRoot]:
    """Returns the report roots of the reports trees plus the extra roots.

    Extra roots are not subject to the exclude patterns; one already found
    under a reports tree is not added again.
    """
    roots = report_discovery.discover_report_roots(reports_roots,
                                                   exclude_patterns)
    known_paths = {root.path for root in roots}
    for extra in extra_report_roots:
        extra_path = pathlib.Path(extra).resolve()
        if extra_path in known_paths:
            continue
        roots.extend(
            report_discovery.discover_report_roots([extra_path],
                                                   exclude_patterns=()) or [])
    return roots


def collect(
    *,
    reports_roots: Sequence[pathlib.Path],
    extra_report_roots: Sequence[pathlib.Path] = (),
    exclude_patterns: Sequence[str] = report_discovery.DEFAULT_EXCLUDE_PATTERNS,
    case_lists: Sequence[pathlib.Path] | None = DEFAULT_CASE_LISTS,
    methods: Sequence[str] = (),
    output_dir: pathlib.Path = DEFAULT_OUTPUT_DIR,
    benchmark_root: pathlib.Path = dataset_registry.DEFAULT_BENCHMARK_ROOT,
    recover_geometry: bool = True,
) -> dict[str, Any]:
    """Runs discovery, aggregation, ranking and coverage.

    Writes summary.tsv, scenes.tsv and results.json under `output_dir`.

    Args:
        reports_roots: The reports trees to discover method report roots in.
        extra_report_roots: Single method report roots to include as they
            are.
        exclude_patterns: Globs of top-level batch directories to skip.
        case_lists: The case lists that define the expected scenes, or None
            to aggregate every evaluated scene.
        methods: Raw method IDs to restrict to; empty for all.
        output_dir: The directory to write to.
        benchmark_root: The benchmark dataset root.
        recover_geometry: Whether to recompute normal/point-cloud metrics
            missing from older eval_*.json files.

    Returns:
        The dashboard payload written to results.json.
    """
    started = time.monotonic()
    roots = _discover_roots(reports_roots, extra_report_roots, exclude_patterns)
    root_paths = [root.path for root in roots]

    scene_rows = summarize_benchmark_reports.deduplicate_scene_metrics(
        summarize_benchmark_reports.collect_scene_metrics(
            root_paths, benchmark_root=benchmark_root))
    if methods:
        wanted = set(methods)
        scene_rows = [row for row in scene_rows if row.method in wanted]
    if case_lists:
        cases = report_discovery.read_case_lists(case_lists)
    else:
        cases = None
    scene_rows, dropped = filter_to_case_list(scene_rows, cases)
    # SLAM scenes on a range the protocol does not evaluate the method on are
    # not shown (on_slam_protocol).
    scene_rows = [
        row for row in scene_rows if benchmark_analyzer.on_slam_protocol(
            row.method, row.task, row.variant)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_tsv = output_dir / "summary.tsv"
    summary_tsv.write_text(summarize_benchmark_reports.render_tsv(
        summarize_benchmark_reports.aggregate_by_dataset(scene_rows)),
                           encoding="utf-8")
    scenes_tsv = output_dir / SCENES_TSV
    scenes_tsv.write_text(render_scene_tsv(scene_rows), encoding="utf-8")

    scene_sources = {
        (row.method, row.task, row.dataset, row.scene): row.source_path
        for row in scene_rows
    }
    data = benchmark_analyzer.build_benchmark_visualization(
        summary_tsv,
        root_paths,
        list(methods) or None,
        recover_geometry=recover_geometry,
        scene_sources=scene_sources,
        scene_rows=benchmark_analyzer.read_scene_rows(scenes_tsv),
    )
    data["coverage"] = build_coverage(
        scene_rows,
        cases=cases,
        failures_by_cell=read_failures_by_cell(roots),
        unsupported_statuses=data.get("method_statuses", []),
        report_roots=roots,
        method_specs=method_registry.METHOD_SPECS,
    )
    # Every method a task/track requires gets a row in its tables, "-" where
    # it has no result.
    benchmark_analyzer.add_missing_method_rows(data["task_tables"],
                                               data["coverage"])
    if cases is not None:
        scene_filter = "case_list"
    else:
        scene_filter = "all_scenes"
    generated_at = datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    data["collection"] = {
        "generated_at": generated_at,
        "reports_roots": [str(pathlib.Path(path)) for path in reports_roots],
        "exclude_patterns": list(exclude_patterns),
        "case_lists": [str(path) for path in case_lists or ()],
        "scene_filter": scene_filter,
        "scenes": len(scene_rows),
        "scenes_dropped_outside_case_list": dropped,
        "geometry_recovery": recover_geometry,
        "report_roots": [root.as_dict() for root in roots],
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(data, indent=2, default=str) + "\n",
                            encoding="utf-8")
    return data


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses the command line."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--reports-root",
        type=pathlib.Path,
        action="append",
        default=[],
        help=("Reports tree to discover method report roots in. Repeatable. "
              f"Default: {DEFAULT_REPORTS_ROOT} "
              "(override with BENCHMARK_REPORTS_ROOT)."),
    )
    parser.add_argument(
        "--report-root",
        type=pathlib.Path,
        action="append",
        default=[],
        help=("Extra single method report root to include as-is "
              "(not subject to exclude patterns). Repeatable."),
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Extra glob for top-level batch directories to skip. Repeatable.",
    )
    parser.add_argument(
        "--no-default-excludes",
        action="store_true",
        help=("Do not skip the default batch globs "
              f"{list(report_discovery.DEFAULT_EXCLUDE_PATTERNS)}."),
    )
    parser.add_argument(
        "--case-list",
        type=pathlib.Path,
        action="append",
        default=[],
        help=("task|dataset|scene case list defining the expected scenes. "
              "Repeatable. "
              "Default: the full-standard depth and SfM case lists."),
    )
    parser.add_argument(
        "--all-scenes",
        action="store_true",
        help=("Aggregate every evaluated scene instead of restricting to the "
              "case lists."),
    )
    parser.add_argument(
        "--method",
        dest="methods",
        action="append",
        default=[],
        help=("Restrict to this raw method ID (e.g. PI3_X_rgb_cameras). "
              "Repeatable."),
    )
    parser.add_argument("--benchmark-root",
                        type=pathlib.Path,
                        default=dataset_registry.DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-dir",
                        type=pathlib.Path,
                        default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-geometry-recovery",
        action="store_true",
        help=("Do not recompute normal/point-cloud metrics missing from older "
              "eval_*.json files."),
    )
    parser.add_argument(
        "--no-viewer-index",
        action="store_true",
        help=("Skip indexing the evaluators' viewer assets for the artifact "
              "viewer's comparison modes."),
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help=("Write results.json without the collection record, source paths "
              "and coverage matrix "
              "(python/public_results.py), as committed with the public "
              "dashboard."),
    )
    return parser.parse_args(argv)


def _print_summary(data: dict[str, Any], output_dir: pathlib.Path) -> None:
    """Prints what was written and the coverage totals."""
    coverage = data["coverage"]["summary"]
    print(f"Wrote {output_dir / 'summary.tsv'} and "
          f"{output_dir / 'results.json'}")
    print(f"  report roots: {len(data['collection']['report_roots'])}, "
          f"scenes: {data['collection']['scenes']}, "
          f"methods with results: {coverage['methods_with_results']}"
          f"/{coverage['registered_methods']}")
    status_counts = ", ".join(
        f"{status}={coverage[status]}" for status in COVERAGE_STATUSES
        if status != STATUS_UNSUPPORTED)
    print("  supported cells: " + status_counts +
          f" (of {coverage['supported_cells']})")
    unregistered = data["coverage"]["unregistered_methods"]
    if unregistered:
        print("  evaluated but not in the method registry: "
              f"{', '.join(unregistered)}")


def main(argv: Sequence[str] | None = None) -> None:
    """Collects the dashboard data and optionally indexes viewer assets."""
    args = parse_args(argv)
    reports_roots = args.reports_root or [DEFAULT_REPORTS_ROOT]
    if args.no_default_excludes:
        excludes = []
    else:
        excludes = list(report_discovery.DEFAULT_EXCLUDE_PATTERNS)
    excludes += list(args.exclude)
    if args.all_scenes:
        case_lists = None
    else:
        case_lists = args.case_list or list(DEFAULT_CASE_LISTS)
    data = collect(
        reports_roots=reports_roots,
        extra_report_roots=args.report_root,
        exclude_patterns=excludes,
        case_lists=case_lists,
        methods=args.methods,
        output_dir=args.output_dir,
        benchmark_root=args.benchmark_root,
        recover_geometry=not args.skip_geometry_recovery,
    )
    _print_summary(data, args.output_dir)
    if args.public:
        # pylint: disable-next=import-outside-toplevel
        from app.webpage.python import public_results  # noqa: PLC0415

        public_results.make_public(args.output_dir / "results.json")

    if not args.no_viewer_index:
        # The reports tree is <eval root>/reports.
        eval_root = pathlib.Path((args.reports_root or
                                  [DEFAULT_REPORTS_ROOT])[0]).resolve().parent
        counts = viewer_index.build_viewer_index(eval_root)
        print(f"Wrote {eval_root / 'viewer_assets' / 'index'}: {counts} "
              "scenes with viewer assets")


if __name__ == "__main__":
    main()
