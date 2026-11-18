#!/usr/bin/env python3
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Collect metrics for every evaluated method/track and export the benchmark dashboard data.

Unlike ``benchmark_analyzer.py`` (which starts from a hand-curated summary TSV),
this entry point discovers every method report root under the reports tree,
deduplicates per-scene results across batches (newest ``eval_*.json`` wins),
restricts them to the full-standard case lists, and joins the result against
the method registry (``spatialarena.methods.METHOD_SPECS``) and the
task/track matrix (``spatialarena.task_track_matrix``) so the dashboard can
show which supported (method, task, track) cells are complete, partial,
failed, or not yet run.

Outputs, all under ``--output-dir``:

- ``summary.tsv``: per (method, task, dataset) aggregated metrics;
- ``results.json``: dashboard payload (``benchmark_analyzer`` schema plus
  ``collection`` and ``coverage`` sections).

Example::

    python app/webpage/python/collect_benchmark_metrics.py \\
        --reports-root $SPATIALARENA_ROOT/benchmarks/_evaluation/reports
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.webpage.python.benchmark_analyzer import (  # noqa: E402
    UNLISTED_METHODS,
    add_missing_method_rows,
    build_benchmark_visualization,
    on_slam_protocol,
    read_scene_rows,
    display_method_name,
    display_task_name,
)
from app.webpage.python.report_discovery import (  # noqa: E402
    DEFAULT_EXCLUDE_PATTERNS,
    ReportRoot,
    discover_report_roots,
    read_case_failures,
    read_case_lists,
)
from app.webpage.python.viewer_index import build_viewer_index  # noqa: E402
from spatialarena.paths import SPATIALARENA_ROOT  # noqa: E402
from spatialarena.eval.dataset_registry import DEFAULT_BENCHMARK_ROOT  # noqa: E402
from spatialarena.methods import METHOD_SPECS, MethodSpec  # noqa: E402
from spatialarena.task_track_matrix import TASK_TRACK_MATRIX, runnable_tracks, track_runs_on  # noqa: E402
from scripts.test.summarize_benchmark_reports import (  # noqa: E402
    SceneMetrics,
    aggregate_by_dataset,
    collect_scene_metrics,
    deduplicate_scene_metrics,
    render_tsv,
)

APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = APP_ROOT / "data"
DEFAULT_REPORTS_ROOT = Path(
    os.environ.get(
        "BENCHMARK_REPORTS_ROOT",
        f"{SPATIALARENA_ROOT}/benchmarks/_evaluation/reports",
    )
)
DEFAULT_CASE_LISTS = (
    REPO_ROOT / "scripts" / "test" / "full_standard_depth_cases.txt",
    REPO_ROOT / "scripts" / "test" / "full_standard_sfm_cases.txt",
    REPO_ROOT / "scripts" / "test" / "full_standard_slam_cases_short.txt",
    REPO_ROOT / "scripts" / "test" / "full_standard_slam_cases_long.txt",
    REPO_ROOT / "scripts" / "test" / "full_standard_slam_cases_large.txt",
)
TRACK_ORDER = (
    "rgb", "rgb_cameras", "rgb_depth", "rgb_masked_depth", "rgb_masked_depth_cameras", "rgb_sparse_depth",
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


def method_id_for_track(base_method: str, track: str) -> str:
    """Raw method ID used in reports: the base name for ``rgb``, ``<base>_<track>`` otherwise."""
    return base_method if track == "rgb" else f"{base_method}_{track}"


def filter_to_case_list(
    rows: Iterable[SceneMetrics],
    cases: set[tuple[str, str, str]] | None,
) -> tuple[list[SceneMetrics], dict[str, int]]:
    """Keep only scenes in the case list; return the kept rows and dropped counts per method/task."""
    rows = list(rows)
    if cases is None:
        return rows, {}
    kept: list[SceneMetrics] = []
    dropped: dict[str, int] = defaultdict(int)
    for row in rows:
        if (row.task, row.dataset, row.scene) in cases:
            kept.append(row)
        else:
            dropped[f"{row.method}|{row.task}|{row.dataset}"] += 1
    return kept, dict(sorted(dropped.items()))


def expected_scenes_by_task(cases: set[tuple[str, str, str]] | None) -> dict[str, dict[str, set[str]]]:
    expected: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for task, dataset, scene in cases or ():
        expected[task][dataset].add(scene)
    return expected


def slam_range_case_lists() -> dict[str, set[tuple[str, str, str]]]:
    """The SLAM case list of each range track (``scripts/test/full_standard_slam_cases_<range>.txt``)."""
    # pylint: disable=import-outside-toplevel
    from spatialarena.slam_protocol import SLAM_RANGES

    lists = {}
    for slam_range in SLAM_RANGES:
        path = REPO_ROOT / "scripts" / "test" / f"full_standard_slam_cases_{slam_range}.txt"
        if path.is_file():
            lists[slam_range] = read_case_lists([path])
    return lists


def slam_expected_scenes(method: str, track: str, range_cases: dict[str, set[tuple[str, str, str]]]
                         ) -> dict[str, set[str]]:
    """The SLAM sequences a method / track is expected on: those of the range tracks the protocol evaluates it on
    (``slam_protocol.method_slam_ranges``), per dataset."""
    # pylint: disable=import-outside-toplevel
    from spatialarena.slam_protocol import method_slam_ranges

    expected: dict[str, set[str]] = defaultdict(set)
    for slam_range in method_slam_ranges(method, track):
        for _, dataset, scene in range_cases.get(slam_range, ()):
            expected[dataset].add(scene)
    return expected


def coverage_status(evaluated: int, expected: int, failed: int, reported_unsupported: bool) -> str:
    if evaluated and expected and evaluated >= expected:
        return STATUS_COMPLETE
    if evaluated:
        return STATUS_PARTIAL
    if reported_unsupported:
        return STATUS_REPORTED_UNSUPPORTED
    if failed:
        return STATUS_FAILED
    return STATUS_NOT_RUN


def build_coverage(
    rows: Sequence[SceneMetrics],
    *,
    cases: set[tuple[str, str, str]] | None,
    failures_by_cell: dict[tuple[str, str], set[tuple[str, str]]],
    unsupported_statuses: Sequence[dict[str, Any]],
    report_roots: Sequence[ReportRoot],
    method_specs: Sequence[MethodSpec] = METHOD_SPECS,
    task_track_matrix: dict[str, tuple[str, ...]] = TASK_TRACK_MATRIX,
) -> dict[str, Any]:
    """Join evaluated scenes with the method registry and the task/track matrix.

    Every (registered method, task, matrix track) cell gets a status:
    ``complete`` (all expected scenes evaluated), ``partial``, ``failed`` (only case
    failures recorded), ``reported_unsupported`` (an ``unsupported_status.json`` exists),
    ``not_run``, or ``unsupported`` (the method does not declare that track).
    """
    expected = expected_scenes_by_task(cases)
    range_cases = slam_range_case_lists() if cases is not None and "SLAM" in expected else {}
    evaluated: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        evaluated[(row.method, row.task)][row.dataset].add(row.scene)
    unsupported_by_method = {status["method"]: status for status in unsupported_statuses}
    roots_by_method: dict[str, list[str]] = defaultdict(list)
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
            supported = set(runnable_tracks(task, spec))
            for track in task_track_matrix[task]:
                method_id = method_id_for_track(spec.name, track)
                registered_ids.add(method_id)
                if track not in supported:
                    cells[f"{task}|{track}"] = {"status": STATUS_UNSUPPORTED, "method_id": method_id}
                    method_totals[STATUS_UNSUPPORTED] += 1
                    continue
                # Only the datasets the track runs on (TRACK_DATASETS): e.g. no MVRecon rgb_depth on dense-depth data.
                evaluated_by_dataset = {
                    dataset: scenes for dataset, scenes in evaluated.get((method_id, task), {}).items()
                    if track_runs_on(task, track, dataset)
                }
                expected_by_dataset = {
                    dataset: scenes for dataset, scenes in expected.get(task, {}).items()
                    if track_runs_on(task, track, dataset)
                }
                if task == "SLAM" and range_cases:
                    # Only the sequences of the range tracks the method is evaluated on (e.g. no large-range KITTI
                    # or VBR for the methods outside the large range).
                    in_ranges = slam_expected_scenes(spec.name, track, range_cases)
                    expected_by_dataset = {dataset: scenes & in_ranges[dataset]
                                           for dataset, scenes in expected_by_dataset.items() if in_ranges.get(dataset)}
                failed_by_dataset: dict[str, set[str]] = defaultdict(set)
                cell_failures = [
                    (dataset, scene) for dataset, scene in failures_by_cell.get((method_id, task), ())
                    if track_runs_on(task, track, dataset)
                ]
                for dataset, scene in cell_failures:
                    failed_by_dataset[dataset].add(scene)
                datasets = sorted(set(expected_by_dataset) | set(evaluated_by_dataset))
                # Scene names are kept per cell (finished/failed only); the frontend derives
                # missing scenes from the shared expected_scene_lists below.
                dataset_cells = {
                    dataset: {
                        "evaluated": len(evaluated_by_dataset.get(dataset, ())),
                        "expected": len(expected_by_dataset.get(dataset, ())),
                        "evaluated_scenes": sorted(evaluated_by_dataset.get(dataset, ())),
                        "failed_scenes": sorted(
                            failed_by_dataset.get(dataset, set()) - evaluated_by_dataset.get(dataset, set())
                        ),
                    }
                    for dataset in datasets
                }
                evaluated_count = sum(len(scenes) for scenes in evaluated_by_dataset.values())
                expected_count = sum(len(scenes) for scenes in expected_by_dataset.values())
                if cases is not None:
                    covered = sum(
                        len(evaluated_by_dataset.get(dataset, set()) & scenes)
                        for dataset, scenes in expected_by_dataset.items()
                    )
                else:
                    covered = evaluated_count
                failed_count = len(cell_failures)
                unsupported = unsupported_by_method.get(method_id)
                status = coverage_status(covered, expected_count, failed_count, unsupported is not None)
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
                cells[f"{task}|{track}"] = cell
                method_totals[status] += 1
        for status, count in method_totals.items():
            status_totals[status] += count
        method_entries.append(
            {
                "method": spec.name,
                "display": display_method_name(spec.name),
                "adapter": spec.adapter,
                "native": spec.native,
                "tracks": list(spec.tracks),
                "notes": spec.notes,
                "report_roots": sorted(
                    path
                    for track in TRACK_ORDER
                    for path in roots_by_method.get(method_id_for_track(spec.name, track), [])
                ),
                "cells": cells,
                "totals": method_totals,
            }
        )

    evaluated_ids = {row.method for row in rows}
    unregistered = sorted(evaluated_ids - registered_ids)
    supported_cells = sum(count for status, count in status_totals.items() if status != STATUS_UNSUPPORTED)
    return {
        "tasks": tasks,
        "task_display_names": {task: display_task_name(task) for task in tasks},
        "tracks": [track for track in TRACK_ORDER if any(track in tracks for tracks in task_track_matrix.values())],
        "task_tracks": {task: list(tracks) for task, tracks in task_track_matrix.items()},
        "expected_scenes": {
            task: {dataset: len(scenes) for dataset, scenes in sorted(datasets.items())}
            for task, datasets in expected.items()
        },
        "expected_scene_lists": {
            task: {dataset: sorted(scenes) for dataset, scenes in sorted(datasets.items())}
            for task, datasets in expected.items()
        },
        "statuses": list(COVERAGE_STATUSES),
        "methods": method_entries,
        "unregistered_methods": unregistered,
        "summary": {
            "registered_methods": len(method_specs),
            "methods_with_results": len(
                {entry["method"] for entry in method_entries if entry["totals"][STATUS_COMPLETE]
                 or entry["totals"][STATUS_PARTIAL]}
            ),
            "supported_cells": supported_cells,
            **status_totals,
        },
    }


def failure_slam_range(failure) -> str:
    """The SLAM range of a case failure, from its report folder (``full-slam-<range>``); empty when it has none."""
    for part in Path(failure.source_path).parts:
        if part.startswith("full-slam-"):
            return part.removeprefix("full-slam-")
    return ""


def failures_by_cell(report_roots: Sequence[ReportRoot]) -> dict[tuple[str, str], set[tuple[str, str]]]:
    cells: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for failure in read_case_failures(root.path for root in report_roots):
        if not on_slam_protocol(failure.method, failure.task, failure_slam_range(failure)):
            continue
        cells[(failure.method, failure.task)].add((failure.dataset, failure.scene))
    return cells


SCENES_TSV = "scenes.tsv"


def render_scene_tsv(rows: Sequence[SceneMetrics]) -> str:
    """One line per evaluated scene (method, task, dataset, SLAM range, scene) with its metrics: what the leaderboard
    scores per scene (benchmark_analyzer.add_scene_scores)."""
    metric_keys = sorted({key for row in rows for key in row.metrics})
    lines = ["\t".join(["method", "task", "dataset", "variant", "scene", *metric_keys])]
    for row in rows:
        values = [f"{row.metrics[key]:.10g}" if key in row.metrics else "" for key in metric_keys]
        lines.append("\t".join([row.method, row.task, row.dataset, row.variant, row.scene, *values]))
    return "\n".join(lines) + "\n"


def collect(
    *,
    reports_roots: Sequence[Path],
    extra_report_roots: Sequence[Path] = (),
    exclude_patterns: Sequence[str] = DEFAULT_EXCLUDE_PATTERNS,
    case_lists: Sequence[Path] | None = DEFAULT_CASE_LISTS,
    methods: Sequence[str] = (),
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    benchmark_root: Path = DEFAULT_BENCHMARK_ROOT,
    recover_geometry: bool = True,
) -> dict[str, Any]:
    """Run discovery, aggregation, ranking and coverage; write ``summary.tsv`` and ``results.json``."""
    started = time.monotonic()
    roots = discover_report_roots(reports_roots, exclude_patterns)
    known_paths = {root.path for root in roots}
    for extra in extra_report_roots:
        extra_path = Path(extra).resolve()
        if extra_path in known_paths:
            continue
        roots.extend(discover_report_roots([extra_path], exclude_patterns=()) or [])
    root_paths = [root.path for root in roots]

    scene_rows = deduplicate_scene_metrics(collect_scene_metrics(root_paths, benchmark_root=benchmark_root))
    if methods:
        wanted = set(methods)
        scene_rows = [row for row in scene_rows if row.method in wanted]
    cases = read_case_lists(case_lists) if case_lists else None
    scene_rows, dropped = filter_to_case_list(scene_rows, cases)
    # SLAM scenes on a range the protocol does not evaluate the method on are not shown (on_slam_protocol).
    scene_rows = [row for row in scene_rows if on_slam_protocol(row.method, row.task, row.variant)]

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_tsv = output_dir / "summary.tsv"
    summary_tsv.write_text(render_tsv(aggregate_by_dataset(scene_rows)), encoding="utf-8")
    scenes_tsv = output_dir / SCENES_TSV
    scenes_tsv.write_text(render_scene_tsv(scene_rows), encoding="utf-8")

    scene_sources = {(row.method, row.task, row.dataset, row.scene): row.source_path for row in scene_rows}
    data = build_benchmark_visualization(
        summary_tsv,
        root_paths,
        list(methods) or None,
        recover_geometry=recover_geometry,
        scene_sources=scene_sources,
        scene_rows=read_scene_rows(scenes_tsv),
    )
    data["coverage"] = build_coverage(
        scene_rows,
        cases=cases,
        failures_by_cell=failures_by_cell(roots),
        unsupported_statuses=data.get("method_statuses", []),
        report_roots=roots,
        # Our own unpublished models stay off the coverage matrix (and so out of the "missing" rows).
        method_specs=[spec for spec in METHOD_SPECS if spec.name not in UNLISTED_METHODS],
    )
    # Every method a task/track requires gets a row in its tables, "-" where it has no result.
    add_missing_method_rows(data["task_tables"], data["coverage"])
    data["collection"] = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reports_roots": [str(Path(path)) for path in reports_roots],
        "exclude_patterns": list(exclude_patterns),
        "case_lists": [str(path) for path in case_lists or ()],
        "scene_filter": "case_list" if cases is not None else "all_scenes",
        "scenes": len(scene_rows),
        "scenes_dropped_outside_case_list": dropped,
        "geometry_recovery": recover_geometry,
        "report_roots": [root.as_dict() for root in roots],
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    return data


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--reports-root",
        type=Path,
        action="append",
        default=[],
        help=f"Reports tree to discover method report roots in. Repeatable. Default: {DEFAULT_REPORTS_ROOT} "
        "(override with BENCHMARK_REPORTS_ROOT).",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        action="append",
        default=[],
        help="Extra single method report root to include as-is (not subject to exclude patterns). Repeatable.",
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
        help=f"Do not skip the default batch globs {list(DEFAULT_EXCLUDE_PATTERNS)}.",
    )
    parser.add_argument(
        "--case-list",
        type=Path,
        action="append",
        default=[],
        help="task|dataset|scene case list defining the expected scenes. Repeatable. "
        "Default: the full-standard depth and SfM case lists.",
    )
    parser.add_argument(
        "--all-scenes",
        action="store_true",
        help="Aggregate every evaluated scene instead of restricting to the case lists.",
    )
    parser.add_argument(
        "--method",
        dest="methods",
        action="append",
        default=[],
        help="Restrict to this raw method ID (e.g. PI3_X_rgb_cameras). Repeatable.",
    )
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-geometry-recovery",
        action="store_true",
        help="Do not recompute normal/point-cloud metrics missing from older eval_*.json files.",
    )
    parser.add_argument(
        "--no-viewer-index",
        action="store_true",
        help="Skip indexing the evaluators' viewer assets for the artifact viewer's comparison modes.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Write results.json without the collection record, source paths and coverage matrix "
             "(python/public_results.py), as committed with the public dashboard.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    reports_roots = args.reports_root or [DEFAULT_REPORTS_ROOT]
    excludes = ([] if args.no_default_excludes else list(DEFAULT_EXCLUDE_PATTERNS)) + list(args.exclude)
    case_lists = None if args.all_scenes else (args.case_list or list(DEFAULT_CASE_LISTS))
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
    coverage = data["coverage"]["summary"]
    print(f"Wrote {args.output_dir / 'summary.tsv'} and {args.output_dir / 'results.json'}")
    print(
        f"  report roots: {len(data['collection']['report_roots'])}, scenes: {data['collection']['scenes']}, "
        f"methods with results: {coverage['methods_with_results']}/{coverage['registered_methods']}"
    )
    print(
        "  supported cells: "
        + ", ".join(f"{status}={coverage[status]}" for status in COVERAGE_STATUSES if status != STATUS_UNSUPPORTED)
        + f" (of {coverage['supported_cells']})"
    )
    if data["coverage"]["unregistered_methods"]:
        print(f"  evaluated but not in the method registry: {', '.join(data['coverage']['unregistered_methods'])}")
    if args.public:
        from app.webpage.python.public_results import make_public  # noqa: PLC0415

        make_public(args.output_dir / "results.json")

    if not args.no_viewer_index:
        # The reports tree is <eval root>/reports.
        eval_root = Path((args.reports_root or [DEFAULT_REPORTS_ROOT])[0]).resolve().parent
        counts = build_viewer_index(eval_root)
        print(f"Wrote {eval_root / 'viewer_assets' / 'index'}: {counts} scenes with viewer assets")


if __name__ == "__main__":
    main()
