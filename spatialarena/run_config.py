# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Run a BenchmarkEvaluator config file (configs/*.yaml) from the command line.

    python -m spatialarena.run_config <config.yaml> plan      # the planned runs, one per line
    python -m spatialarena.run_config <config.yaml> generate  # write the scripts only, for review
    python -m spatialarena.run_config <config.yaml> coverage  # generate, then compare with the canonical cases
    python -m spatialarena.run_config <config.yaml> submit    # generate and submit (max_in_flight from the config)

Every job script comes from the pipeline (spatialarena/script_generator.py) under
``<experiment_root>/<experiment_name>/scripts/``; nothing is hand-written.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from spatialarena.evaluator import TASK_SUITE, BenchmarkEvaluator
from spatialarena.slam_protocol import SLAM_RANGES
from spatialarena.task_track_matrix import TASK_TRACK_MATRIX, evaluated_methods, track_runs_on_case

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# The canonical case list of each suite (and SLAM range): what a full run of the suite must evaluate.
CANONICAL_CASE_FILES = {
    "full-depth": "scripts/test/full_standard_depth_cases.txt",
    "full-pose": "scripts/test/full_standard_sfm_cases.txt",
    "full-slam": "scripts/test/full_standard_slam_cases_{slam_range}.txt",
}


def describe_run(work_item: dict) -> str:
    """One planned run as ``task dataset model track [range]``."""
    parts = [work_item["task"], work_item["dataset"], work_item["model"], work_item.get("track", "rgb")]
    if work_item.get("slam_range"):
        parts.append(work_item["slam_range"])
    return " ".join(parts)


def read_cases(path: Path) -> set[str]:
    """The ``task|dataset|scene|interval[|order]`` lines of a case file."""
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")}


def expected_cases(datasets: list[dict], models: list[dict]) -> dict[tuple, set[str]]:
    """What the configured datasets and models must evaluate, from the protocol alone (not from the plan).

    Key ``(suite, task, method, track, slam_range)``: for every task/track of the matrix (and every SLAM range
    track), each configured method of ``evaluated_methods`` with that suite's canonical cases of the datasets.
    """
    # (task, dataset) -> the SLAM range the item is restricted to ("" when it takes every range)
    configured = {(item["task"], item["dataset"].lower()): item.get("slam_range", "") for item in datasets}
    # (task, dataset) -> the scenes the item is restricted to (``scenes:``), when it names any
    configured_scenes = {(item["task"], item["dataset"].lower()): set(item["scenes"])
                         for item in datasets if item.get("scenes")}
    configured_models = {(item["name"], item.get("track", "rgb")): item.get("slam_range", "") for item in models}
    expected: dict[tuple, set[str]] = {}
    for task, tracks in TASK_TRACK_MATRIX.items():
        suite = TASK_SUITE[task]
        for slam_range in SLAM_RANGES if task == "SLAM" else ("",):
            path = PROJECT_ROOT / CANONICAL_CASE_FILES[suite].format(slam_range=slam_range)
            wanted = {line for line in read_cases(path)
                      if line.split("|")[0] == task and (task, line.split("|")[1].lower()) in configured
                      and configured[(task, line.split("|")[1].lower())] in ("", slam_range)
                      and line.split("|")[2] in configured_scenes.get((task, line.split("|")[1].lower()),
                                                                     {line.split("|")[2]})}
            if not wanted:
                continue
            for track in tracks:
                track_wanted = {line for line in wanted if track_runs_on_case(task, track, *line.split("|")[1:3])}
                if not track_wanted:
                    continue
                for method in evaluated_methods(task, track, slam_range or None):
                    if configured_models.get((method, track)) in ("", slam_range):
                        expected[(suite, task, method, track, slam_range)] = track_wanted
    return expected


def coverage_problems(bench: BenchmarkEvaluator, generated: list) -> list[str]:
    """Differences between the generated runs' case files and :func:`expected_cases`, one line each."""
    plan = bench.plan()
    expected = expected_cases(plan.selected_datasets, plan.selected_models)
    got: dict[tuple, set[str]] = {}
    for item, scripts in generated:
        key = (item["suite"], item["task"], item["model"], item.get("track", "rgb"), item.get("slam_range", ""))
        got.setdefault(key, set()).update(read_cases(scripts.case_file))
    problems = []
    for key in sorted(set(expected) | set(got)):
        label = " ".join(part for part in key if part)
        if key not in got:
            problems.append(f"missing run: {label} ({len(expected[key])} cases)")
        elif key not in expected:
            problems.append(f"unexpected run: {label} ({len(got[key])} cases)")
        elif got[key] != expected[key]:
            missing, extra = expected[key] - got[key], got[key] - expected[key]
            problems.append(f"case mismatch: {label}: {len(missing)} missing (e.g. {sorted(missing)[:2]}), "
                            f"{len(extra)} extra (e.g. {sorted(extra)[:2]})")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Plan, generate or submit the runs of one config file."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("action", choices=("plan", "generate", "coverage", "submit"))
    args = parser.parse_args(argv)

    bench = BenchmarkEvaluator()
    bench.configure(args.config)
    if args.action == "plan":
        runs = bench.plan().selected_runs
        for work_item in runs:
            print(describe_run(work_item))
        by_task = Counter((run["task"], run.get("track", "rgb"), run.get("slam_range", "")) for run in runs)
        print(f"# {len(runs)} runs: " + ", ".join(f"{task}/{track}{'/' + rng if rng else ''}={count}"
                                                 for (task, track, rng), count in sorted(by_task.items())))
        return 0
    if args.action == "generate":
        generated = bench.generate()
        for item, scripts in generated:
            print(f"{describe_run(item):60s} {scripts.infer_script}")
        print(f"# {len(generated)} runs generated under {bench.plan().output_root / 'scripts'}")
        return 0
    if args.action == "coverage":
        generated = bench.generate()
        problems = coverage_problems(bench, generated)
        cases = sum(len(read_cases(scripts.case_file)) for _, scripts in generated)
        for problem in problems:
            print(problem)
        print(f"# {len(generated)} runs, {cases} cases; {len(problems)} coverage problems")
        return 1 if problems else 0
    report = bench.evaluate()
    print(json.dumps({key: value for key, value in report.summary.items() if key != "jobs"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
