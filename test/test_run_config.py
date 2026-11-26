# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Config-file runs (spatialarena/run_config.py): the coverage check of generated scripts."""

from __future__ import annotations

from pathlib import Path

import yaml

from spatialarena.evaluator import BenchmarkEvaluator
from spatialarena.run_config import coverage_problems, expected_cases, read_cases


def slam_bench(tmp_path: Path, benchmark_root: Path) -> BenchmarkEvaluator:
    """PI3_X rgb_depth on 7-Scenes SLAM: one short and one long range run."""
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump({
            "experiment_root": str(tmp_path / "experiments"),
            "benchmark_root": str(benchmark_root),
            "experiment_name": "check",
            "datasets": [{
                "task": "SLAM",
                "dataset": "7scenes"
            }],
            "models": [{
                "name": "PI3_X",
                "track": "rgb_depth"
            }],
        }))
    bench = BenchmarkEvaluator()
    bench.configure(config)
    return bench


def test_generated_slam_runs_cover_every_range_track(tmp_path, benchmark_root):
    """The planner's runs hold exactly the canonical cases of each range the method takes part in."""
    bench = slam_bench(tmp_path, benchmark_root)
    generated = bench.generate("check")
    assert sorted(
        item["slam_range"] for item, _ in generated) == ["long", "short"]
    assert coverage_problems(bench, generated) == []
    plan = bench.plan()
    expected = expected_cases(plan.selected_datasets, plan.selected_models)
    assert set(expected) == {
        ("full-slam", "SLAM", "PI3_X", "rgb_depth", "short"),
        ("full-slam", "SLAM", "PI3_X", "rgb_depth", "long")
    }


def test_coverage_reports_a_missing_run_and_a_wrong_case(
        tmp_path, benchmark_root):
    """Dropping a range's run and changing a case's frame interval are both reported."""
    bench = slam_bench(tmp_path, benchmark_root)
    generated = bench.generate("check")
    short = [(item, scripts)
             for item, scripts in generated
             if item["slam_range"] == "short"]
    case_file = short[0][1].case_file
    first = sorted(read_cases(case_file))[0]
    text = case_file.read_text(encoding="utf-8").replace(
        first,
        first.rsplit("|", 1)[0] + "|1")
    case_file.write_text(text, encoding="utf-8")
    problems = coverage_problems(bench, short)
    assert any(
        problem.startswith("missing run: full-slam SLAM PI3_X rgb_depth long")
        for problem in problems)
    assert any(
        problem.startswith(
            "case mismatch: full-slam SLAM PI3_X rgb_depth short: 1 missing")
        for problem in problems)


def test_a_slam_range_restriction_narrows_the_expected_runs(
        tmp_path, benchmark_root):
    """A dataset item restricted to one range expects (and plans) only that range."""
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump({
            "experiment_root": str(tmp_path / "experiments"),
            "benchmark_root": str(benchmark_root),
            "experiment_name": "check",
            "datasets": [{
                "task": "SLAM",
                "dataset": "7scenes",
                "slam_range": "short"
            }],
            "models": [{
                "name": "PI3_X",
                "track": "rgb_depth"
            }],
        }))
    bench = BenchmarkEvaluator()
    bench.configure(config)
    generated = bench.generate("check")
    assert [item["slam_range"] for item, _ in generated] == ["short"]
    assert coverage_problems(bench, generated) == []


def test_reevaluate_exports_the_engine_switch(tmp_path, benchmark_root):
    """`reevaluate: true` makes every generated script export REEVALUATE=1 (the engine then skips its reuse check)."""
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump({
            "experiment_root": str(tmp_path / "experiments"),
            "benchmark_root": str(benchmark_root),
            "experiment_name": "check",
            "reevaluate": True,
            "datasets": [{
                "task": "SLAM",
                "dataset": "7scenes"
            }],
            "models": [{
                "name": "PI3_X",
                "track": "rgb_depth"
            }],
        }))
    bench = BenchmarkEvaluator()
    bench.configure(config)
    generated = bench.generate("check")

    assert bench.reevaluate
    for _item, scripts in generated:
        assert "export REEVALUATE=1" in scripts.eval_script.read_text()
    plain = tmp_path / "plain"
    plain.mkdir()
    assert "export REEVALUATE" not in slam_bench(
        plain, benchmark_root).generate("check")[0][1].eval_script.read_text()


def test_force_reinfer_exports_the_engine_switch(tmp_path, benchmark_root):
    """`force_reinfer: true` makes every generated script export FORCE_REINFER=1 (the engine then re-runs
    inference although the content-addressed prediction cache already has a matching entry)."""
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump({
            "experiment_root": str(tmp_path / "experiments"),
            "benchmark_root": str(benchmark_root),
            "experiment_name": "check",
            "force_reinfer": True,
            "datasets": [{
                "task": "SLAM",
                "dataset": "7scenes"
            }],
            "models": [{
                "name": "PI3_X",
                "track": "rgb_depth"
            }],
        }))
    bench = BenchmarkEvaluator()
    bench.configure(config)
    generated = bench.generate("check")

    assert bench.force_reinfer
    for _item, scripts in generated:
        assert "export FORCE_REINFER=1" in scripts.infer_script.read_text()
        assert "export FORCE_REINFER=1" in scripts.eval_script.read_text()
    plain = tmp_path / "plain"
    plain.mkdir()
    assert "export FORCE_REINFER" not in slam_bench(
        plain, benchmark_root).generate("check")[0][1].infer_script.read_text()


def test_case_parallelism_is_exported_in_the_generated_scripts(
        tmp_path, benchmark_root):
    """`case_parallelism` is written into both scripts (sbatch then submits with plain --export=NIL)."""
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump({
            "experiment_root": str(tmp_path / "experiments"),
            "benchmark_root": str(benchmark_root),
            "experiment_name": "check",
            "case_parallelism": 1,
            "datasets": [{
                "task": "SLAM",
                "dataset": "7scenes"
            }],
            "models": [{
                "name": "PI3_X",
                "track": "rgb_depth"
            }],
        }))
    bench = BenchmarkEvaluator()
    bench.configure(config)
    for _item, scripts in bench.generate("check"):
        assert "export CASE_PARALLELISM=1" in scripts.infer_script.read_text()
        assert "export CASE_PARALLELISM=1" in scripts.eval_script.read_text()
    plain = tmp_path / "plain"
    plain.mkdir()
    assert "CASE_PARALLELISM" not in slam_bench(
        plain, benchmark_root).generate("check")[0][1].infer_script.read_text()
