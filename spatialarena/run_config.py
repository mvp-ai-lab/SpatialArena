# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Runs a BenchmarkEvaluator config file (configs/*.yaml).

    python -m spatialarena.run_config <config.yaml> plan
        The planned runs, one per line.
    python -m spatialarena.run_config <config.yaml> generate
        Writes the scripts only, for review.
    python -m spatialarena.run_config <config.yaml> coverage
        Generates, then compares with the canonical cases.
    python -m spatialarena.run_config <config.yaml> submit
        Generates and submits (max_in_flight from the config).

Every job script comes from the pipeline (spatialarena/script_generator.py),
under <experiment_root>/<experiment_name>/scripts/; nothing is hand-written.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

from spatialarena import slam_protocol
from spatialarena import task_track_matrix
# Imported by name: `from spatialarena import evaluator` would give the
# package's `evaluator` instance, not this module.
from spatialarena.evaluator import BenchmarkEvaluator
from spatialarena.evaluator import TASK_SUITE

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
# The canonical case list of each suite (and SLAM range): what a full run of the
# suite must evaluate.
CANONICAL_CASE_FILES = {
    "full-depth": "scripts/test/full_standard_depth_cases.txt",
    "full-pose": "scripts/test/full_standard_sfm_cases.txt",
    "full-slam": "scripts/test/full_standard_slam_cases_{slam_range}.txt",
}


def describe_run(work_item: dict) -> str:
    """Returns one planned run as `task dataset model track [range]`."""
    parts = [
        work_item["task"],
        work_item["dataset"],
        work_item["model"],
        work_item.get("track", "rgb"),
    ]
    if work_item.get("slam_range"):
        parts.append(work_item["slam_range"])
    return " ".join(parts)


def read_cases(path: pathlib.Path) -> set[str]:
    """Returns the case lines (`task|dataset|scene|interval[|order]`)."""
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def _is_configured_case(
        line: str, task: str, slam_range: str, configured: dict[tuple[str, str],
                                                                str],
        configured_scenes: dict[tuple[str, str], set[str]]) -> bool:
    """Returns whether a canonical case line belongs to a configured dataset.

    Args:
        line: A `task|dataset|scene|...` case line.
        task: The task being expanded.
        slam_range: The SLAM range being expanded ("" outside SLAM).
        configured: (task, dataset) -> the SLAM range the dataset is
            restricted to ("" when it takes every range).
        configured_scenes: (task, dataset) -> the scenes the dataset is
            restricted to, for datasets that name any.

    Returns:
        True when the case is of `task`, its dataset is configured for
        `slam_range` and its scene is not filtered out.
    """
    fields = line.split("|")
    if fields[0] != task:
        return False
    key = (task, fields[1].lower())
    if key not in configured or configured[key] not in ("", slam_range):
        return False
    return fields[2] in configured_scenes.get(key, {fields[2]})


def expected_cases(datasets: list[dict],
                   models: list[dict]) -> dict[tuple, set[str]]:
    """Returns what the configured datasets and models must evaluate.

    The result comes from the protocol alone, not from the plan: for every
    task/track of the matrix (and every SLAM range), each configured method of
    `evaluated_methods` with that suite's canonical cases of the datasets.

    Args:
        datasets: The configured dataset items (`task`, `dataset`, optional
            `slam_range` and `scenes`).
        models: The configured model items (`name`, optional `track` and
            `slam_range`).

    Returns:
        The expected case lines, keyed by
        `(suite, task, method, track, slam_range)`.
    """
    # (task, dataset) -> the SLAM range the item is restricted to ("" when it
    # takes every range).
    configured = {
        (item["task"], item["dataset"].lower()): item.get("slam_range", "")
        for item in datasets
    }
    # (task, dataset) -> the scenes the item is restricted to (`scenes:`),
    # when it names any.
    configured_scenes = {
        (item["task"], item["dataset"].lower()): set(item["scenes"])
        for item in datasets
        if item.get("scenes")
    }
    configured_models = {
        (item["name"], item.get("track", "rgb")): item.get("slam_range", "")
        for item in models
    }
    expected: dict[tuple, set[str]] = {}
    for task, tracks in task_track_matrix.TASK_TRACK_MATRIX.items():
        suite = TASK_SUITE[task]
        if task == "SLAM":
            slam_ranges = slam_protocol.SLAM_RANGES
        else:
            slam_ranges = ("",)
        for slam_range in slam_ranges:
            path = PROJECT_ROOT / CANONICAL_CASE_FILES[suite].format(
                slam_range=slam_range)
            wanted = {
                line for line in read_cases(path) if _is_configured_case(
                    line, task, slam_range, configured, configured_scenes)
            }
            if not wanted:
                continue
            for track in tracks:
                track_wanted = {
                    line for line in wanted
                    if task_track_matrix.track_runs_on_case(
                        task, track,
                        *line.split("|")[1:3])
                }
                if not track_wanted:
                    continue
                methods = task_track_matrix.evaluated_methods(
                    task, track, slam_range or None)
                for method in methods:
                    if configured_models.get(
                        (method, track)) in ("", slam_range):
                        key = (suite, task, method, track, slam_range)
                        expected[key] = track_wanted
    return expected


def coverage_problems(bench: BenchmarkEvaluator, generated: list) -> list[str]:
    """Returns how the generated runs' case files differ from the protocol.

    Args:
        bench: The configured evaluator.
        generated: `(work item, scripts)` pairs of `bench.generate()`.

    Returns:
        One line per problem: a missing run, an unexpected run, or a run whose
        cases differ from `expected_cases`.
    """
    plan = bench.plan()
    expected = expected_cases(plan.selected_datasets, plan.selected_models)
    got: dict[tuple, set[str]] = {}
    for item, scripts in generated:
        key = (item["suite"], item["task"], item["model"],
               item.get("track", "rgb"), item.get("slam_range", ""))
        got.setdefault(key, set()).update(read_cases(scripts.case_file))
    problems = []
    for key in sorted(set(expected) | set(got)):
        label = " ".join(part for part in key if part)
        if key not in got:
            problems.append(
                f"missing run: {label} ({len(expected[key])} cases)")
        elif key not in expected:
            problems.append(f"unexpected run: {label} ({len(got[key])} cases)")
        elif got[key] != expected[key]:
            missing = expected[key] - got[key]
            extra = got[key] - expected[key]
            problems.append(
                f"case mismatch: {label}: {len(missing)} missing (e.g. "
                f"{sorted(missing)[:2]}), {len(extra)} extra (e.g. "
                f"{sorted(extra)[:2]})")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Plans, generates or submits the runs of one config file."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=pathlib.Path)
    parser.add_argument("action",
                        choices=("plan", "generate", "coverage", "submit"))
    args = parser.parse_args(argv)

    bench = BenchmarkEvaluator()
    bench.configure(args.config)
    if args.action == "plan":
        runs = bench.plan().selected_runs
        for work_item in runs:
            print(describe_run(work_item))
        by_task = collections.Counter(
            (run["task"], run.get("track", "rgb"), run.get("slam_range", ""))
            for run in runs)
        counts = []
        for (task, track, slam_range), count in sorted(by_task.items()):
            range_suffix = f"/{slam_range}" if slam_range else ""
            counts.append(f"{task}/{track}{range_suffix}={count}")
        print(f"# {len(runs)} runs: " + ", ".join(counts))
        return 0
    if args.action == "generate":
        generated = bench.generate()
        for item, scripts in generated:
            print(f"{describe_run(item):60s} {scripts.infer_script}")
        print(f"# {len(generated)} runs generated under "
              f"{bench.plan().output_root / 'scripts'}")
        return 0
    if args.action == "coverage":
        generated = bench.generate()
        problems = coverage_problems(bench, generated)
        cases = sum(
            len(read_cases(scripts.case_file)) for _, scripts in generated)
        for problem in problems:
            print(problem)
        print(f"# {len(generated)} runs, {cases} cases; {len(problems)} "
              "coverage problems")
        return 1 if problems else 0
    report = bench.evaluate()
    summary = {
        key: value for key, value in report.summary.items() if key != "jobs"
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
