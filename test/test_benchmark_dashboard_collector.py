# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Tests for the standalone benchmark dashboard's report discovery and metric collection."""

from __future__ import annotations

import json
import os
from pathlib import Path

from app.webpage.python.benchmark_artifact_analyzer import deduplicate_cases
from app.webpage.python.collect_benchmark_metrics import build_coverage, collect
from app.webpage.python.report_discovery import (
    discover_report_roots,
    read_case_failures,
    read_case_lists,
    split_method_track,
)
from spatialarena.methods import METHOD_SPECS, MethodSpec
from scripts.test.summarize_benchmark_reports import SceneMetrics

DEPTH_METRICS = {
    "abs_relative_difference": 0.1,
    "absrel_percent": 10.0,
    "squared_relative_difference": 0.2,
    "rmse_linear": 1.0,
    "rmse_log": 0.1,
    "log10": 0.05,
    "delta103_acc": 0.5,
    "delta125_acc": 0.9,
    "delta125_2_acc": 0.95,
    "delta125_3_acc": 0.99,
    "i_rmse": 1.0,
    "silog_rmse": 10.0,
    "mae_normal_deg": 20.0,
}
SPARSE_METRICS = {
    "num_pairs": 10,
    "ate_rmse": 0.05,
    "relative_rotation_error_deg_mean": 1.0,
    "relative_translation_error_deg_mean": 2.0,
    "auc_30": 0.8,
}


def write_summary(root: Path, method: str, base_method: str,
                  track: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": method,
        "base_method": base_method,
        "benchmark_track": track
    }
    (root / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def write_eval(
    path: Path,
    *,
    method: str,
    task: str,
    dataset: str,
    scene: str,
    metrics: dict,
    mtime: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scene_result = {
        "task": task,
        "dataset": dataset,
        "scene": scene,
        "method": method,
        "num_evaluated_frames": 4,
        "num_valid_pixels": 100,
        "num_matched_images": 4,
        "metrics": metrics,
    }
    payload = {
        "num_scene_results": 1,
        "num_errors": 0,
        "metrics": {},
        "scene_results": [scene_result],
        "errors": []
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def build_reports_tree(tmp_path: Path) -> Path:
    """Two batches of FOO depth results, a summary-less FOO_rgb_depth pose root, BAR failures, a test batch."""
    reports = tmp_path / "reports"

    foo_root = reports / "full-depth" / "FOO"
    write_summary(foo_root, "FOO", "FOO", "rgb")
    write_eval(
        foo_root / "MonoDepth" / "marigold" / "s1" / "eval_depth.json",
        method="FOO",
        task="MonoDepth",
        dataset="marigold",
        scene="s1",
        metrics={
            **DEPTH_METRICS, "abs_relative_difference": 0.5,
            "absrel_percent": 50.0
        },
        mtime=1_000_000,
    )
    write_eval(
        foo_root / "MonoDepth" / "marigold" / "outside" / "eval_depth.json",
        method="FOO",
        task="MonoDepth",
        dataset="marigold",
        scene="outside",
        metrics=DEPTH_METRICS,
    )

    retry_root = reports / "retry_batch" / "full-depth" / "FOO" / "chunk_00"
    write_summary(retry_root, "FOO", "FOO", "rgb")
    write_eval(
        retry_root / "MonoDepth" / "marigold" / "s1" / "eval_depth.json",
        method="FOO",
        task="MonoDepth",
        dataset="marigold",
        scene="s1",
        metrics={
            **DEPTH_METRICS, "abs_relative_difference": 0.25,
            "absrel_percent": 25.0
        },
        mtime=2_000_000,
    )

    write_eval(
        reports / "full-pose" / "FOO_rgb_depth" / "SfM" / "eth3d" / "s1" /
        "eval_sparse.json",
        method="FOO_rgb_depth",
        task="SfM",
        dataset="eth3d",
        scene="s1",
        metrics=SPARSE_METRICS,
    )

    bar_root = reports / "full-pose" / "BAR"
    write_summary(bar_root, "BAR", "BAR", "rgb")
    (bar_root / "case_failures_full_standard_sfm_cases_slam.tsv").write_text(
        "method\ttask\tdataset\tscene\tinterval\torder\tstatus\nBAR\tSLAM\t7scenes\ta\t1\t\tfailed\n",
        encoding="utf-8",
    )

    write_eval(
        reports / "_private_test" / "FOO" / "MonoDepth" / "marigold" / "s2" /
        "eval_depth.json",
        method="FOO",
        task="MonoDepth",
        dataset="marigold",
        scene="s2",
        metrics=DEPTH_METRICS,
    )
    return reports


def write_case_list(tmp_path: Path) -> Path:
    path = tmp_path / "cases.txt"
    path.write_text(
        "# Format: task|dataset|scene|interval|order\n"
        "MonoDepth|marigold|s1|1|\n"
        "MonoDepth|marigold|s2|1|\n"
        "SfM|eth3d|s1|1|\n"
        "SLAM|7scenes|a|1|\n",
        encoding="utf-8",
    )
    return path


def test_split_method_track():
    assert split_method_track("PI3_X_rgb_cameras") == ("PI3_X", "rgb_cameras")
    assert split_method_track("PI3_X_rgb_depth") == ("PI3_X", "rgb_depth")
    assert split_method_track("VGGT") == ("VGGT", "rgb")
    assert split_method_track("DA3-L") == ("DA3-L", "rgb")


def test_discover_report_roots_uses_summaries_fallbacks_and_excludes(tmp_path):
    reports = build_reports_tree(tmp_path)

    roots = discover_report_roots([reports])
    by_path = {
        root.path.relative_to(reports.resolve()).as_posix(): root
        for root in roots
    }

    assert set(by_path) == {
        "full-depth/FOO",
        "retry_batch/full-depth/FOO/chunk_00",
        "full-pose/FOO_rgb_depth",
        "full-pose/BAR",
    }
    assert by_path["full-depth/FOO"].eval_files == 2
    assert by_path["full-depth/FOO"].has_summary
    assert by_path["retry_batch/full-depth/FOO/chunk_00"].batch == "retry_batch"
    assert by_path["retry_batch/full-depth/FOO/chunk_00"].suite == "full-depth"
    fallback = by_path["full-pose/FOO_rgb_depth"]
    assert not fallback.has_summary
    assert (fallback.method, fallback.base_method,
            fallback.track) == ("FOO_rgb_depth", "FOO", "rgb_depth")
    assert by_path["full-pose/BAR"].eval_files == 0

    with_private = discover_report_roots([reports], exclude_patterns=())
    assert any("_private_test" in str(root.path) for root in with_private)


def test_discover_report_roots_folds_nested_roots(tmp_path):
    reports = tmp_path / "reports"
    outer = reports / "full-depth" / "FOO"
    write_summary(outer, "FOO", "FOO", "rgb")
    write_eval(
        outer / "nested" / "MonoDepth" / "marigold" / "s1" / "eval_depth.json",
        method="FOO",
        task="MonoDepth",
        dataset="marigold",
        scene="s1",
        metrics=DEPTH_METRICS,
    )
    write_summary(outer / "nested", "FOO", "FOO", "rgb")

    roots = discover_report_roots([reports])

    assert [root.path for root in roots] == [outer.resolve()]
    assert roots[0].eval_files == 1


def test_read_case_lists_and_failures(tmp_path):
    reports = build_reports_tree(tmp_path)
    cases = read_case_lists([write_case_list(tmp_path)])
    assert ("SfM", "eth3d", "s1") in cases
    assert len(cases) == 4

    failures = read_case_failures([reports / "full-pose" / "BAR"])
    assert [
        (item.method, item.task, item.dataset, item.scene) for item in failures
    ] == [("BAR", "SLAM", "7scenes", "a")]


def test_read_case_failures_reads_every_failure_record(tmp_path):
    root = tmp_path / "full-depth" / "FOO"
    scene_dir = root / "MonoDepth" / "hammer" / "hammer"
    scene_dir.mkdir(parents=True)
    (scene_dir / "inference_failure.tsv").write_text("1\tgpu_oom\n",
                                                     encoding="utf-8")
    for scene, error in (("s1", {
            "error_type": "CUDA_OOM",
            "error": "CUDA out of memory during inference"
    }), ("s2", {
            "error": "Degenerate constant prediction"
    }), ("s3", {
            "error": "something broke"
    })):
        path = root / "SfM" / "eth3d" / scene / "eval_sparse.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({
            "status": "failed",
            "num_scene_results": 0,
            "scene_results": [],
            "errors": [error]
        }),
                        encoding="utf-8")
    scored = root / "SfM" / "eth3d" / "s4" / "eval_sparse.json"
    scored.parent.mkdir(parents=True)
    scored.write_text(json.dumps({
        "num_scene_results": 1,
        "scene_results": [{}],
        "errors": []
    }),
                      encoding="utf-8")
    watchdog = root / "watchdog_execution_outcomes" / "job.json"
    watchdog.parent.mkdir(parents=True)
    watchdog.write_text(json.dumps({
        "status": "OUT_OF_MEMORY",
        "job": {
            "work_item": {
                "task": "VideoDepth",
                "dataset": "bonn",
                "model": "FOO",
                "track": "rgb",
                "scenes": ["b1"]
            }
        }
    }),
                        encoding="utf-8")

    failures = read_case_failures([root])

    assert sorted(
        (item.method, item.task, item.dataset, item.scene, item.status)
        for item in failures) == [
            ("FOO", "MonoDepth", "hammer", "hammer", "gpu_oom"),
            ("FOO", "SfM", "eth3d", "s1", "gpu_oom"),
            ("FOO", "SfM", "eth3d", "s2", "degenerate_constant_prediction"),
            ("FOO", "SfM", "eth3d", "s3", "error"),
            ("FOO", "VideoDepth", "bonn", "b1", "oom"),
        ]


def test_collect_dedupes_batches_and_filters_to_case_list(tmp_path):
    reports = build_reports_tree(tmp_path)
    output_dir = tmp_path / "out"

    data = collect(
        reports_roots=[reports],
        case_lists=[write_case_list(tmp_path)],
        output_dir=output_dir,
        recover_geometry=False,
    )

    assert (output_dir / "summary.tsv").is_file()
    assert json.loads((output_dir / "results.json").read_text(
        encoding="utf-8"))["methods"] == data["methods"]
    assert data["methods"] == ["FOO", "FOO_rgb_depth"]

    monodepth = data["task_tables"]["MonoDepth"]["dataset_tables"]["marigold"][
        "methods"]
    foo_row = next(row for row in monodepth if row["method"] == "FOO")
    # The newer retry batch wins for s1, and the out-of-list scene is not aggregated.
    assert foo_row["metrics"]["absrel_percent"]["value"] == 25.0
    assert foo_row["scenes"] == 1

    collection = data["collection"]
    assert collection["scene_filter"] == "case_list"
    assert collection["scenes"] == 2
    assert collection["scenes_dropped_outside_case_list"] == {
        "FOO|MonoDepth|marigold": 1
    }
    assert len(collection["report_roots"]) == 4
    # FOO/BAR are synthetic, so they are reported as outside the real method registry.
    assert data["coverage"]["unregistered_methods"] == ["FOO", "FOO_rgb_depth"]


def test_collect_all_scenes_keeps_out_of_list_results(tmp_path):
    reports = build_reports_tree(tmp_path)

    data = collect(reports_roots=[reports],
                   case_lists=None,
                   output_dir=tmp_path / "out",
                   recover_geometry=False)

    assert data["collection"]["scene_filter"] == "all_scenes"
    assert data["collection"]["scenes"] == 3


def scene_row(method: str, task: str, dataset: str, scene: str) -> SceneMetrics:
    kind = "depth" if task == "MonoDepth" else "sparse"
    return SceneMetrics(kind=kind,
                        method=method,
                        task=task,
                        dataset=dataset,
                        scene=scene,
                        metrics={},
                        source_path=Path("x"))


def test_build_coverage_statuses(monkeypatch):
    # The fake methods take part in the SLAM track (only range-track methods are offered SLAM).
    import spatialarena.slam_protocol as slam_protocol

    real_ranges = slam_protocol.method_slam_ranges
    monkeypatch.setattr(
        slam_protocol,
        "method_slam_ranges",
        lambda method, track="rgb":
        (["short"]
         if method in {"FOO", "BAR", "BAZ"} else real_ranges(method, track)))
    specs = [
        MethodSpec(name="FOO",
                   adapter="foo.py",
                   code_env="FOO_CODE",
                   code_default="",
                   tracks=("rgb", "rgb_depth")),
        MethodSpec(name="BAR",
                   adapter="bar.py",
                   code_env="BAR_CODE",
                   code_default=""),
        MethodSpec(name="BAZ",
                   adapter="baz.py",
                   code_env="BAZ_CODE",
                   code_default=""),
    ]
    matrix = {
        "MonoDepth": ("rgb",),
        "SfM": ("rgb", "rgb_depth"),
        "SLAM": ("rgb", "rgb_depth")
    }
    cases = {
        ("MonoDepth", "marigold", "s1"),
        ("MonoDepth", "marigold", "s2"),
        ("SfM", "eth3d", "s1"),
        ("SLAM", "7scenes", "a"),
    }
    rows = [
        scene_row("FOO", "MonoDepth", "marigold", "s1"),
        scene_row("FOO", "MonoDepth", "marigold", "s2"),
        scene_row("FOO_rgb_depth", "SfM", "eth3d", "s1"),
        scene_row("BAR", "MonoDepth", "marigold", "s1"),
        scene_row("LEGACY", "SfM", "eth3d", "s1"),
    ]

    coverage = build_coverage(
        rows,
        cases=cases,
        failures_by_cell={("BAR", "SLAM"): {("7scenes", "a")}},
        unsupported_statuses=[{
            "method": "BAZ",
            "reason": "no pose head"
        }],
        report_roots=[],
        method_specs=specs,
        task_track_matrix=matrix,
    )

    cells = {entry["method"]: entry["cells"] for entry in coverage["methods"]}
    assert cells["FOO"]["MonoDepth|rgb"]["status"] == "complete"
    assert cells["FOO"]["MonoDepth|rgb"]["datasets"] == {
        "marigold": {
            "evaluated": 2,
            "expected": 2,
            "evaluated_scenes": ["s1", "s2"],
            "failed_scenes": []
        }
    }
    assert cells["BAR"]["SLAM|rgb"]["datasets"]["7scenes"]["failed_scenes"] == [
        "a"
    ]
    assert coverage["expected_scene_lists"]["MonoDepth"] == {
        "marigold": ["s1", "s2"]
    }
    assert cells["FOO"]["SfM|rgb"]["status"] == "not_run"
    assert cells["FOO"]["SfM|rgb_depth"]["status"] == "complete"
    assert cells["FOO"]["SfM|rgb_depth"]["method_id"] == "FOO_rgb_depth"
    assert cells["BAR"]["MonoDepth|rgb"]["status"] == "partial"
    assert cells["BAR"]["SLAM|rgb"]["status"] == "failed"
    assert cells["BAR"]["SLAM|rgb"]["failed_cases"] == 1
    assert cells["BAR"]["SfM|rgb_depth"]["status"] == "unsupported"
    assert cells["BAZ"]["SfM|rgb"]["status"] == "reported_unsupported"
    assert cells["BAZ"]["SfM|rgb"]["reason"] == "no pose head"
    assert coverage["unregistered_methods"] == ["LEGACY"]

    summary = coverage["summary"]
    assert summary["registered_methods"] == 3
    assert summary["methods_with_results"] == 2
    # FOO: 5 supported cells, BAR and BAZ: 3 rgb cells each.
    assert summary["supported_cells"] == 11
    assert summary["complete"] == 2
    assert summary["partial"] == 1
    assert summary["failed"] == 1
    assert summary["reported_unsupported"] == 3
    assert summary["unsupported"] == 4


def test_deduplicate_artifact_cases_keeps_newest_report(tmp_path):
    old_report = tmp_path / "old.json"
    new_report = tmp_path / "new.json"
    old_report.write_text("{}", encoding="utf-8")
    new_report.write_text("{}", encoding="utf-8")
    os.utime(old_report, (1_000_000, 1_000_000))
    os.utime(new_report, (2_000_000, 2_000_000))
    base = {
        "kind": "depth",
        "task": "MonoDepth",
        "dataset": "marigold",
        "method": "FOO",
        "scene": "s1"
    }

    cases = deduplicate_cases([{
        **base, "report_path": str(new_report)
    }, {
        **base, "report_path": str(old_report)
    }])

    assert [case["report_path"] for case in cases] == [str(new_report)]


def test_geometry_recovery_only_reevaluates_deduplicated_winner(
        tmp_path, monkeypatch):
    from types import SimpleNamespace

    from app.webpage.python import benchmark_analyzer

    metrics_without_normals = {
        key: value
        for key, value in DEPTH_METRICS.items()
        if key != "mae_normal_deg"
    }
    old_eval = tmp_path / "old" / "eval_depth.json"
    new_eval = tmp_path / "new" / "eval_depth.json"
    for path, prediction in ((old_eval, "old_prediction"), (new_eval,
                                                            "new_prediction")):
        write_eval(path,
                   method="FOO",
                   task="MonoDepth",
                   dataset="marigold",
                   scene="s1",
                   metrics=metrics_without_normals)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["scene_results"][0]["prediction_scene_path"] = str(tmp_path /
                                                                   prediction)
        path.write_text(json.dumps(payload), encoding="utf-8")

    fake_dataset = SimpleNamespace(
        list_scenes=lambda: [SimpleNamespace(scene="s1")])
    monkeypatch.setattr(
        benchmark_analyzer,
        "find_dataset_spec",
        lambda task, dataset: SimpleNamespace(create_dataset=lambda root:
                                              fake_dataset),
    )
    evaluated_paths: list[str] = []

    def fake_evaluate_depth_scene(scene_info, prediction_scene_path, config):
        del scene_info, config  # the analyzer passes config by keyword: the names stay
        evaluated_paths.append(Path(prediction_scene_path).name)
        return {
            "metrics": {
                "mae_normal_deg":
                    3.0 if Path(prediction_scene_path).name == "new_prediction"
                    else 9.0
            }
        }

    monkeypatch.setattr(benchmark_analyzer, "evaluate_depth_scene",
                        fake_evaluate_depth_scene)

    recovered = benchmark_analyzer.recover_geometry_metrics_from_reports(
        [old_eval.parent, new_eval.parent],
        scene_sources={("FOO", "MonoDepth", "marigold", "s1"): new_eval},
    )

    assert evaluated_paths == ["new_prediction"]
    assert recovered == {
        ("FOO", "MonoDepth", "marigold"): {
            "mae_normal_deg": 3.0
        }
    }


def test_slam_rows_keep_their_range_track(tmp_path: Path) -> None:
    """Short / long / large SLAM rows of one method and dataset stay separate dataset groups, never averaged."""
    from app.webpage.python.benchmark_analyzer import display_dataset_name, read_summary_rows

    summary = tmp_path / "summary.tsv"
    summary.write_text(
        "kind\tmethod\ttask\tdataset\tvariant\tscenes\tate_rmse\n"
        "slam\tPI3_X\tSLAM\t7scenes\tshort\t18\t0.10\n"
        "slam\tPI3_X\tSLAM\t7scenes\tlong\t18\t0.20\n"
        "depth\tPI3_X\tMonoDepth\tsintel\t\t23\t\n",
        encoding="utf-8",
    )
    rows = read_summary_rows(summary)
    assert [row["dataset"] for row in rows
           ] == ["7scenes@short", "7scenes@long", "sintel"]
    assert [row["slam_range"] for row in rows] == ["short", "long", ""]
    assert rows[0]["dataset_base"] == "7scenes"
    assert display_dataset_name("7scenes@long") == "7Scenes · long"
    assert display_dataset_name("sintel") == "Sintel"


def test_slam_ranks_each_range_over_methods_that_cover_all_of_it(
        tmp_path: Path) -> None:
    """A SLAM range ranks only the methods complete on every dataset of it; the task score averages the ranges."""
    from app.webpage.python.benchmark_analyzer import (
        build_rankings,
        build_task_tables,
        read_summary_rows,
    )

    header = "kind\tmethod\ttask\tdataset\tvariant\tscenes\tsequences_total\tslam_ate_rmse\tslam_drift_pct\n"
    lines = [
        # short range: datasets x and y; C is the best on x but never ran y.
        ("A", "x", "short", 0.1, 1.0),
        ("B", "x", "short", 0.2, 2.0),
        ("C", "x", "short", 0.01, 0.1),
        ("A", "y", "short", 30.0, 3.0),
        ("B", "y", "short", 10.0, 1.0),
        # long range: dataset x only, all three complete.
        ("A", "x", "long", 0.1, 1.0),
        ("B", "x", "long", 0.2, 2.0),
        ("C", "x", "long", 0.05, 0.5),
    ]
    summary = tmp_path / "summary.tsv"
    summary.write_text(
        header + "".join(f"slam\t{m}\tSLAM\t{d}\t{r}\t4\t4\t{ate}\t{drift}\n"
                         for m, d, r, ate, drift in lines),
        encoding="utf-8",
    )
    rows = read_summary_rows(summary)

    short = build_task_tables(rows)["SLAM"]["range_tables"]["short"]
    by_method = {row["method"]: row for row in short["methods"]}
    assert short["metrics"][:2] == ["slam_ate_rmse::x", "slam_drift_pct::x"]
    assert by_method["A"]["ranked"] and by_method["B"][
        "ranked"] and not by_method["C"]["ranked"]
    assert by_method["C"]["score"] is None
    assert by_method["C"]["metrics"]["slam_ate_rmse::x"]["rank"] is None
    assert by_method["B"]["metrics"]["slam_ate_rmse::x"]["rank"] == 2
    assert by_method["A"]["score"] == by_method["B"]["score"] == 50.0

    # Within the single dataset@range cell C is complete, so it is ranked there.
    cell = {
        row["method"]: row for row in build_task_tables(rows)["SLAM"]
        ["dataset_tables"]["x@short"]["methods"]
    }
    assert cell["C"]["ranked"] and cell["C"]["score_rank"] == 1

    ranking = {
        row["method"]: row["task_scores"]["SLAM"]
        for row in build_rankings(rows)["rows"]
    }
    # Mean over the two ranges, a range not fully covered counting as 0.
    assert round(ranking["A"]["score"], 2) == round((50.0 + 100.0 * 2 / 3) / 2,
                                                    2)
    assert ranking["B"]["score"] == 25.0
    assert ranking["C"]["score"] == 50.0
    assert list(ranking["C"]["ranges"]) == ["long"]
    assert ranking["A"]["rank"] == 1


def test_slam_scene_scores_rank_every_method_and_score_failed_sequences_zero(
        tmp_path: Path) -> None:
    """With per-sequence rows SLAM is scored like the other tasks: every method that ran a cell is ranked, a failed
    (OOM) or missing sequence scores 0, and a method whose every sequence failed gets a row with the failure."""
    from app.webpage.python.benchmark_analyzer import (
        build_rankings,
        build_task_tables,
        read_case_failures as read_tsv_failures,
        read_scene_rows,
        read_summary_rows,
    )

    summary = tmp_path / "summary.tsv"
    summary.write_text(
        "kind\tmethod\ttask\tdataset\tvariant\tscenes\tsequences_total\tslam_ate_rmse\tslam_drift_pct\n"
        "slam\tA\tSLAM\tx\tlarge\t2\t2\t1.0\t1.0\n"
        "slam\tB\tSLAM\tx\tlarge\t1\t2\t2.0\t2.0\n",
        encoding="utf-8",
    )
    scenes = tmp_path / "scenes.tsv"
    scenes.write_text(  # the collector's scene table: SLAM metrics without the "slam_" prefix
        "method\ttask\tdataset\tvariant\tscene\tate_rmse\tdrift_pct\n"
        "A\tSLAM\tx\tlarge\ts0\t1.0\t1.0\nA\tSLAM\tx\tlarge\ts1\t3.0\t3.0\n"
        "B\tSLAM\tx\tlarge\ts0\t2.0\t2.0\n",
        encoding="utf-8",
    )
    failures = tmp_path / "case_failures.tsv"
    failures.write_text(
        "method\ttask\tdataset\tvariant\tscene\tstatus\tdetail\n"
        "B\tSLAM\tx\tlarge\ts1\tgpu_oom_recorded\t\n"
        "C\tSLAM\tx\tlarge\ts0\tgpu_oom_recorded\t\nC\tSLAM\tx\tlarge\ts1\tgpu_oom_recorded\t\n"
        "A\tSLAM\tx\tlarge\ts1\tgpu_oom_recorded\tsuperseded by the scored report\n",
        encoding="utf-8",
    )
    rows, scene_rows, case_failures = read_summary_rows(
        summary), read_scene_rows(scenes), read_tsv_failures(failures)
    assert scene_rows[0]["metrics"] == {
        "slam_ate_rmse": 1.0,
        "slam_drift_pct": 1.0
    }

    tables = build_task_tables(rows, scene_rows, case_failures)["SLAM"]
    cell = {
        row["method"]: row
        for row in tables["dataset_tables"]["x@large"]["methods"]
    }
    # s0: A best (100), B worst (0); s1: A alone (100), B and C failed (0).
    assert cell["A"]["score"] == 100.0 and cell["A"][
        "coverage_label"] == "" and cell["A"]["failed_scenes"] == {}
    assert cell["B"]["ranked"] and cell["B"]["score"] == 0.0
    assert cell["B"]["coverage_label"] == "1/2 sequences · 1 GPU OOM"
    assert cell["C"]["ranked"] and cell["C"][
        "failure_label"] == "2 GPU OOM" and cell["C"]["metrics"] == {}
    large = {
        row["method"]: row for row in tables["range_tables"]["large"]["methods"]
    }
    assert large["C"]["dataset_failures"] == {
        "x": "2 GPU OOM"
    } and large["C"]["score_rank"] == 2

    ranked_rows = build_rankings(rows, scene_rows, case_failures)["rows"]
    ranking = {row["method"]: row["task_scores"]["SLAM"] for row in ranked_rows}
    assert ranking["A"]["score"] == 100.0 and ranking["A"]["coverage"] == 2
    assert ranking["C"]["score"] == 0.0 and ranking["C"]["ranges"]["large"][
        "score"] == 0.0


def test_case_coverage_labels_rows_short_of_their_cases_and_keeps_them_out_of_metric_ranks(
):
    """A row with fewer evaluated scenes than expected is marked incomplete with its failures named; a leaderboard
    dataset the method never took on counts against its all-datasets row too (the task score scores it 0), while an
    excluded dataset (Spring) does not."""
    from app.webpage.python.benchmark_analyzer import attach_case_coverage

    def ds(evaluated, expected, failed=()):
        return {
            "evaluated": evaluated,
            "expected": expected,
            "evaluated_scenes": [],
            "failed_scenes": list(failed)
        }

    coverage = {
        "methods": [
            {
                "method": "A",
                "cells": {
                    "MonoDepth|rgb": {
                        "method_id": "A",
                        "datasets": {
                            "x": ds(2, 3),
                            "y": ds(0, 2, ["y0", "y1"]),
                            "z": ds(0, 4)
                        }
                    }
                }
            },
            {
                "method": "B",
                "cells": {
                    "MonoDepth|rgb": {
                        "method_id": "B",
                        "datasets": {
                            "x": ds(3, 3),
                            "y": ds(2, 2),
                            "z": ds(0, 4)
                        }
                    }
                }
            },
            {
                "method": "C",
                "cells": {
                    "MonoDepth|rgb": {
                        "method_id": "C",
                        "datasets": {
                            "x": ds(3, 3),
                            "y": ds(2, 2),
                            "z": ds(4, 4),
                            "spring": ds(0, 1)
                        }
                    }
                }
            },
        ]
    }
    tables = {
        "MonoDepth": {
            "methods": [{
                "method": "A",
                "metrics": {}
            }, {
                "method": "B",
                "metrics": {}
            }, {
                "method": "C",
                "metrics": {}
            }],
            "dataset_tables": {
                "x": {
                    "methods": [{
                        "method": "A",
                        "metrics": {}
                    }, {
                        "method": "B",
                        "metrics": {}
                    }]
                },
                "y": {
                    "methods": [{
                        "method": "A",
                        "missing": True,
                        "metrics": {}
                    }, {
                        "method": "B",
                        "metrics": {}
                    }]
                }
            },
        }
    }
    failures = [{
        "method": "A",
        "task": "MonoDepth",
        "dataset": "y",
        "scene": scene,
        "label": "GPU OOM"
    } for scene in ("y0", "y1")]
    attach_case_coverage(tables, coverage, failures)
    overall = {row["method"]: row for row in tables["MonoDepth"]["methods"]}
    assert overall["A"]["incomplete"] and overall["A"][
        "coverage_label"] == "2/9 scenes · 2 GPU OOM"
    # z was never run: it still counts, as the task score scores it 0.
    assert overall["B"]["incomplete"] and overall["B"][
        "coverage_label"] == "5/9 scenes"
    assert "incomplete" not in overall["C"]  # Spring is off the leaderboard
    y = {
        row["method"]: row
        for row in tables["MonoDepth"]["dataset_tables"]["y"]["methods"]
    }
    assert y["A"]["failure_label"] == "2 GPU OOM"
    x = {
        row["method"]: row
        for row in tables["MonoDepth"]["dataset_tables"]["x"]["methods"]
    }
    assert x["A"][
        "coverage_label"] == "2/3 scenes" and "failure_label" not in x["A"]


def test_case_coverage_labels_unregistered_methods_from_their_track():
    """A method outside the registry (no coverage cell, e.g. imported results) is held to its (task, track)'s
    expected scenes, counted from the dataset tables, so a dataset it never ran makes it incomplete."""
    from app.webpage.python.benchmark_analyzer import attach_case_coverage

    def ds(evaluated, expected):
        return {"evaluated": evaluated, "expected": expected, "failed_scenes": []}

    coverage = {
        "methods": [{
            "method": "A",
            "cells": {
                "MonoDepth|rgb": {
                    "method_id": "A",
                    "datasets": {"x": ds(3, 3), "y": ds(2, 2)}
                }
            }
        }]
    }
    tables = {
        "MonoDepth": {
            "methods": [{"method": "A", "metrics": {}}, {"method": "IMPORTED", "metrics": {}}],
            "dataset_tables": {
                "x": {"methods": [{"method": "A", "scenes": 3, "metrics": {}},
                                  {"method": "IMPORTED", "scenes": 3, "metrics": {}}]},
                "y": {"methods": [{"method": "A", "scenes": 2, "metrics": {}},
                                  {"method": "IMPORTED", "missing": True, "metrics": {}}]},
            },
        }
    }
    attach_case_coverage(tables, coverage)
    overall = {row["method"]: row for row in tables["MonoDepth"]["methods"]}
    assert "incomplete" not in overall["A"]
    assert overall["IMPORTED"]["incomplete"] and overall["IMPORTED"]["coverage_label"] == "3/5 scenes"
    x = {row["method"]: row for row in tables["MonoDepth"]["dataset_tables"]["x"]["methods"]}
    assert "incomplete" not in x["IMPORTED"]

def test_all_datasets_table_averages_the_leaderboard_scenes():
    """With per-scene rows the all-datasets table's metrics are scene means over the leaderboard datasets: each scene
    weighs the same and an excluded dataset (Spring) does not count."""
    from app.webpage.python.benchmark_analyzer import build_task_tables

    def row(method, dataset, absrel):
        return {
            "kind": "depth",
            "method": method,
            "track": "rgb",
            "has_auxiliary_input": False,
            "task": "MonoDepth",
            "dataset": dataset,
            "dataset_base": dataset,
            "slam_range": "",
            "scenes": 1,
            "frames": 0,
            "matched_images": 0,
            "pairs": 0,
            "metrics": {
                "absrel_percent": absrel
            }
        }

    rows = [
        row("A", "ibims1", 2.0),
        row("A", "spring", 50.0),
        row("B", "ibims1", 3.0),
        row("B", "spring", 5.0)
    ]
    scenes = [{
        "method": method,
        "task": "MonoDepth",
        "dataset": dataset,
        "scene": scene,
        "metrics": {
            "absrel_percent": value
        }
    } for method, dataset, scene, value in [(
        "A", "ibims1", "s0",
        1.0), ("A", "ibims1", "s1",
               4.0), ("A", "spring", "s", 50.0), ("B", "ibims1", "s0",
                                                  3.0), ("B", "spring", "s",
                                                         5.0)]]
    overall = {
        r["method"]: r
        for r in build_task_tables(rows, scenes)["MonoDepth"]["methods"]
    }
    assert overall["A"]["metrics"]["absrel_percent"][
        "value"] == 2.5 and overall["A"]["scenes"] == 2
    assert overall["B"]["metrics"]["absrel_percent"]["value"] == 3.0
    assert overall["A"]["metrics"]["absrel_percent"]["rank"] == 1 and overall[
        "A"]["datasets"] == ["ibims1"]


def test_slam_tables_list_the_methods_the_protocol_runs_on_each_range():
    """Range and dataset@range tables get a "-" row for a method the protocol evaluates there but that has none."""
    from app.webpage.python.benchmark_analyzer import add_missing_slam_rows

    table = {
        "range_tables": {
            "short": {
                "datasets": ["7scenes"],
                "methods": [{
                    "method": "PI3_X"
                }]
            },
            "large": {
                "datasets": ["vbr"],
                "methods": [{
                    "method": "PI_LONG"
                }]
            }
        },
        "dataset_tables": {
            "7scenes@short": {
                "methods": []
            },
            "vbr@large": {
                "methods": [{
                    "method": "PI_LONG"
                }]
            }
        },
    }
    # CUT3R runs short and long but not large; DA3-G only short.
    added = add_missing_slam_rows(table, [("CUT3R", "CUT3R", "rgb"),
                                          ("DA3-G", "DA3-G", "rgb")])
    short = {
        row["method"]: row for row in table["range_tables"]["short"]["methods"]
    }
    assert short["CUT3R"]["missing"] and short["DA3-G"]["missing"]
    assert [row["method"] for row in table["range_tables"]["large"]["methods"]
           ] == ["PI_LONG"]
    assert {
        row["method"]
        for row in table["dataset_tables"]["7scenes@short"]["methods"]
    } == {"DA3-G", "CUT3R", "PI3_X"}
    assert added == 5


def test_build_coverage_leaves_out_datasets_a_track_does_not_run_on():
    """MVRecon rgb_depth runs on DTU, not HiRoom: HiRoom is neither expected nor counted for that cell."""
    specs = [
        MethodSpec(name="FOO",
                   adapter="foo.py",
                   code_env="FOO_CODE",
                   code_default="",
                   tracks=("rgb", "rgb_depth"))
    ]
    cases = {("MVRecon", "dtu", "scan24"), ("MVRecon", "hiroom", "room1")}
    rows = [
        scene_row("FOO_rgb_depth", "MVRecon", "dtu", "scan24"),
        scene_row("FOO_rgb_depth", "MVRecon", "hiroom", "room1")
    ]
    coverage = build_coverage(
        rows,
        cases=cases,
        failures_by_cell={},
        unsupported_statuses=[],
        report_roots=[],
        method_specs=specs,
        task_track_matrix={"MVRecon": ("rgb", "rgb_depth")})
    cells = {
        entry["method"]: entry["cells"] for entry in coverage["methods"]
    }["FOO"]
    assert set(cells["MVRecon|rgb_depth"]["datasets"]) == {"dtu"}
    assert cells["MVRecon|rgb_depth"]["status"] == "complete"
    assert set(cells["MVRecon|rgb"]["datasets"]) == {"dtu", "hiroom"}


def test_off_matrix_rows_and_viewer_cases_are_left_out(tmp_path: Path) -> None:
    """MonoDepth given GT depth (PI3_X_rgb_depth) and VideoDepth [+C] on Sintel (no released GT cameras) are off the
    task/track matrix, so the dashboard drops them."""
    from app.webpage.python.benchmark_analyzer import build_benchmark_visualization
    from app.webpage.python.viewer_index import index_depth

    summary = tmp_path / "summary.tsv"
    summary.write_text(
        "kind\tmethod\ttask\tdataset\tvariant\tscenes\tabsrel_percent\tmae_normal_deg\tate_rmse\n"
        "depth\tPI3_X\tMonoDepth\tmarigold\t\t5\t5.0\t20.0\t\n"
        "depth\tPI3_X_rgb_depth\tMonoDepth\tmarigold\t\t5\t2.0\t20.0\t\n"
        "depth\tPI3_X_rgb_cameras\tVideoDepth\tsintel\t\t5\t9.0\t20.0\t\n"
        "sfm\tPI3_X_rgb_depth\tSfM\teth3d\t\t5\t\t\t0.5\n",
        encoding="utf-8",
    )
    data = build_benchmark_visualization(summary, recover_geometry=False)
    shown = {(row["method"], task)
             for task, table in data["task_tables"].items()
             for row in table["methods"]}
    assert ("PI3_X", "MonoDepth") in shown and ("PI3_X_rgb_cameras",
                                                "VideoDepth") not in shown
    assert ("PI3_X_rgb_depth", "SfM") in shown
    assert ("PI3_X_rgb_depth", "MonoDepth") not in shown
    assert data["summary"]["excluded_off_matrix_rows"] == {
        "PI3_X_rgb_depth|MonoDepth": 1,
        "PI3_X_rgb_cameras|VideoDepth": 1
    }

    root = tmp_path / "reports/benchmark_full_reports/full-depth"
    for method in ("PI3_X", "PI3_X_rgb_depth"):
        frames = root / method / "MonoDepth/marigold/scene0/viewer/frames.json"
        frames.parent.mkdir(parents=True)
        frames.write_text(json.dumps({
            "frames": [{
                "name": "0"
            }],
            "metrics": {}
        }),
                          encoding="utf-8")
    (scene,) = index_depth(tmp_path)["scenes"]
    assert [entry["method"] for entry in scene["methods"]] == ["PI3_X"]


def test_missing_method_rows_list_every_required_method_with_no_metrics():
    """Required methods without a result get a "missing" row in the tables of the datasets their cell expects."""
    from app.webpage.python.benchmark_analyzer import add_missing_method_rows

    result_row = {
        "method": "FOO",
        "metrics": {
            "ate_rmse": {
                "value": 1.0,
                "display": "1.000",
                "rank": 1
            }
        }
    }
    task_tables = {
        "SfM": {
            "methods": [dict(result_row)],
            "dataset_tables": {
                "eth3d": {
                    "methods": [dict(result_row)]
                },
                "hiroom": {
                    "methods": []
                }
            }
        },
        "SLAM": {
            "methods": [],
            "dataset_tables": {
                "tum": {
                    "methods": []
                }
            }
        },
    }
    cells = {
        "FOO": {
            "SfM|rgb": {
                "status": "partial",
                "method_id": "FOO",
                "datasets": {
                    "eth3d": {},
                    "hiroom": {}
                }
            }
        },
        "BAR": {
            "SfM|rgb": {
                "status": "not_run",
                "method_id": "BAR",
                "datasets": {
                    "eth3d": {}
                }
            },
            "SfM|rgb_depth": {
                "status": "unsupported",
                "method_id": "BAR_rgb_depth"
            },
            "MVRecon|rgb": {
                "status": "partial",
                "method_id": "BAR",
                "datasets": {}
            },
            "SLAM|rgb": {
                "status": "not_run",
                "method_id": "BAR",
                "datasets": {
                    "tum": {}
                }
            }
        },
        # Never run on any task (a track not run yet, or no released weights): no rows at all.
        "BAZ": {
            "SfM|rgb": {
                "status": "not_run",
                "method_id": "BAZ",
                "datasets": {
                    "eth3d": {}
                }
            }
        }
    }
    coverage = {
        "methods": [{
            "method": name,
            "cells": method_cells
        } for name, method_cells in cells.items()]
    }

    assert add_missing_method_rows(task_tables, coverage) == 3
    sfm = task_tables["SfM"]
    rows = {
        table: {
            row["method"]: row.get("missing", False)
            for row in sfm["dataset_tables"][table]["methods"]
        } for table in ("eth3d", "hiroom")
    }
    assert rows == {
        "eth3d": {
            "FOO": False,
            "BAR": True
        },
        "hiroom": {
            "FOO": True
        }
    }
    assert {
        row["method"]: row.get("missing", False) for row in sfm["methods"]
    } == {
        "FOO": False,
        "BAR": True
    }
    (bar,) = [row for row in sfm["methods"] if row["method"] == "BAR"]
    assert bar["metrics"] == {} and bar["scenes"] == 0
    assert task_tables["SLAM"]["dataset_tables"]["tum"]["methods"] == [
    ]  # range tables are left alone


def test_collect_lists_every_registered_method_on_the_coverage_matrix(tmp_path):
    data = collect(reports_roots=[tmp_path],
                   case_lists=None,
                   output_dir=tmp_path / "out",
                   recover_geometry=False)
    listed = {entry["method"] for entry in data["coverage"]["methods"]}
    assert listed == {spec.name for spec in METHOD_SPECS}
    assert data["coverage"]["summary"]["registered_methods"] == len(listed)


def _depth_scene(method, scene, absrel, **extra):
    metrics = {
        "absrel_percent": absrel,
        "delta103_percent": 50.0,
        "delta125_percent": 90.0,
        "delta125_2_percent": 95.0,
        "delta125_3_percent": 98.0,
        "rmse_linear": absrel
    } | extra
    return {
        "method": method,
        "task": "MonoDepth",
        "dataset": "ibims1",
        "scene": scene,
        "metrics": metrics
    }


def test_metric_case_scores_average_every_case_with_missing_ones_zero():
    """A metric column ranks by its normalized score over every case, a case the method skipped counting 0: BAR's
    better AbsRel on the two scenes it ran does not beat FOO, which ran all five."""
    from app.webpage.python.benchmark_analyzer import (attach_metric_case_scores, build_rankings)

    rows = [{
        "method": method,
        "task": "MonoDepth",
        "dataset": "ibims1",
        "slam_range": "",
        "metrics": {}
    } for method in ("FOO", "BAR")]
    scenes = [_depth_scene("FOO", f"s{i}", 10.0) for i in range(5)]
    scenes += [_depth_scene("BAR", f"s{i}", 5.0) for i in range(2)]
    ranking = build_rankings(rows, scenes)["rows"]
    by_method = {row["method"]: row["metric_case_scores"]["MonoDepth"] for row in ranking}
    # AbsRel: BAR best on s0, s1 (FOO 0 there); FOO alone on s2-s4 (100), BAR 0 there.
    assert by_method["FOO"]["all"]["absrel_percent"] == 60.0
    assert by_method["BAR"]["all"]["absrel_percent"] == 40.0
    assert by_method["BAR"]["datasets"]["ibims1"]["absrel_percent"] == 40.0
    assert "mae_normal_deg" not in by_method["FOO"]["all"]  # nobody reports it

    tables = {
        "MonoDepth": {
            "methods": [{"method": "FOO", "metrics": {}}, {"method": "BAR", "metrics": {}}],
            "dataset_tables": {
                "ibims1": {"methods": [{"method": "FOO", "metrics": {}}, {"method": "BAR", "metrics": {}}]}
            },
        }
    }
    attach_metric_case_scores(tables, ranking)
    overall = {row["method"]: row for row in tables["MonoDepth"]["methods"]}
    assert overall["BAR"]["metric_case_scores"]["absrel_percent"] == 40.0
    ibims1 = {row["method"]: row for row in tables["MonoDepth"]["dataset_tables"]["ibims1"]["methods"]}
    assert ibims1["FOO"]["metric_case_scores"]["absrel_percent"] == 60.0

def test_scene_scores_weigh_every_case_and_score_missing_ones_zero():
    """Per-scene min-max on the depth score metrics only; a case or metric a method lacks counts 0."""
    from app.webpage.python.benchmark_analyzer import build_rankings

    rows = [{
        "method": method,
        "task": "MonoDepth",
        "dataset": "ibims1",
        "slam_range": "",
        "metrics": {}
    } for method in ("FOO", "BAR")]
    scenes = [
        _depth_scene("FOO", "s0", 10.0, mae_normal_deg=20.0),
        _depth_scene(
            "BAR", "s0",
            20.0),  # lacks MAE-normal, which FOO reports: 0 for that metric
        _depth_scene("FOO", "s1", 10.0, rmse_linear=99.0
                    ),  # rmse is reference only: FOO's worse rmse is ignored
        _depth_scene("BAR", "s1", 20.0),
        _depth_scene("FOO", "s2",
                     10.0),  # BAR has no result on s2: 0 for that case
    ]
    ranking = {
        row["method"]: row for row in build_rankings(rows, scenes)["rows"]
    }
    # FOO: best on every metric reported on each scene -> 100 on all three (MAE-normal only counts on s0).
    assert ranking["FOO"]["task_scores"]["MonoDepth"]["score"] == 100.0
    # BAR: worst AbsRel (0), ties on the four deltas (100 each); s0 also scores MAE-normal, which it lacks (0):
    # s0 = 400/6, s1 = 400/5, s2 (no result) = 0.
    assert abs(ranking["BAR"]["task_scores"]["MonoDepth"]["score"] -
               (400 / 6 + 400 / 5) / 3) < 1e-9
    assert ranking["BAR"]["task_dataset_scores"]["MonoDepth"]["ibims1"][
        "coverage"] == 3
    assert "rmse_linear" not in ranking["FOO"]["metric_scores"]


def test_conditioned_tracks_are_averaged_over_the_tasks_they_can_run():
    """rgb averages over every task; [+C]/[+D] only over the tasks the matrix lets them run on a leaderboard dataset.

    VideoDepth [+C] runs only on SpatialArena, which is scored, so it counts once SpatialArena is evaluated; a task
    evaluated only on a leaderboard-excluded dataset (ScanNet++) does not.
    """
    from app.webpage.python.benchmark_analyzer import track_ranking_tasks

    tasks = ["MonoDepth", "VideoDepth", "MVRecon", "SfM", "SLAM"]
    datasets = {
        "MonoDepth": {"ibims1"},
        "VideoDepth": {"bonn", "sintel"},
        "MVRecon": {"dtu", "hiroom"},
        "SfM": {"eth3d"},
        "SLAM": {"7scenes"}
    }
    assert track_ranking_tasks("rgb", tasks, datasets) == tasks
    assert track_ranking_tasks("rgb_cameras", tasks, datasets) == ["MVRecon"]
    assert track_ranking_tasks("rgb_cameras", tasks, {
        **datasets, "VideoDepth": {"spatialarena"}
    }) == ["VideoDepth", "MVRecon"]
    assert track_ranking_tasks("rgb_cameras", tasks, {
        **datasets, "MVRecon": {"scannetpp"}
    }) == []
    assert track_ranking_tasks("rgb_depth", tasks,
                               datasets) == ["MVRecon", "SfM", "SLAM"]
    assert track_ranking_tasks("rgb_depth", tasks, {
        **datasets, "MVRecon": {"hiroom"}
    }) == ["SfM", "SLAM"]


def test_conditioned_tracks_stay_out_of_the_overall_ranking():
    """A [+C] track scored on MVRecon alone can out-score rgb methods averaged over every task; it follows them,
    without an overall rank, and keeps its MVRecon rank."""
    from app.webpage.python.benchmark_analyzer import build_rankings

    def metrics(task: str, error: float) -> dict[str, float]:
        if task == "MVRecon":
            return {"accuracy": error, "completion": error}
        return {"absrel_percent": error, "delta125_percent": 100.0 - error}

    rows = [{
        "method": method,
        "task": task,
        "dataset": dataset,
        "dataset_base": dataset,
        "slam_range": "",
        "metrics": metrics(task, error)
    } for method, task, dataset, error in (("FOO", "MVRecon", "dtu", 5.0),
                                           ("FOO", "MonoDepth", "ibims1", 9.0),
                                           ("BAR", "MVRecon", "dtu", 9.0),
                                           ("BAR", "MonoDepth", "ibims1",
                                            5.0), ("FOO_rgb_cameras", "MVRecon",
                                                   "dtu", 1.0))]
    ranking = build_rankings(rows)["rows"]
    assert [row["method"] for row in ranking][-1] == "FOO_rgb_cameras"
    conditioned = ranking[-1]
    assert conditioned["rank"] is None and conditioned["ranking_tasks"] == [
        "MVRecon"
    ]
    assert conditioned["task_scores"]["MVRecon"]["rank"] == 1
    assert sorted(row["rank"] for row in ranking[:-1]) == [1, 2]


def test_all_datasets_row_says_when_results_are_only_on_unscored_datasets():
    """A track with results only on a leaderboard-excluded dataset (here MVRecon [+C] on ScanNet++) gets a label
    in the all-datasets table instead of an empty row; a scored row gets none."""
    from app.webpage.python.benchmark_analyzer import summarize_method_scene_means

    rows = [{
        "method": method,
        "task": "MVRecon",
        "dataset": dataset,
        "dataset_base": dataset,
        "slam_range": "",
        "scenes": 1,
        "frames": 1,
        "matched_images": 0,
        "pairs": 0,
        "metrics": {}
    } for method, dataset in (("FOO", "dtu"), ("FOO_rgb_cameras", "scannetpp"))]
    scenes = [{
        "method": "FOO",
        "task": "MVRecon",
        "dataset": "dtu",
        "scene": "s0",
        "metrics": {
            "absrel_percent": 5.0
        }
    }]
    by_method = {
        row["method"]: row for row in summarize_method_scene_means(
            rows, scenes, ["absrel_percent"])
    }
    assert by_method["FOO_rgb_cameras"][
        "coverage_label"] == "ScanNet++ only · not scored"
    assert by_method["FOO_rgb_cameras"][
        "unscored_only"] and not by_method["FOO_rgb_cameras"]["metrics"]
    assert "coverage_label" not in by_method["FOO"] and by_method["FOO"][
        "scenes"] == 1


def test_summary_rows_derive_the_delta_percents(tmp_path):
    from app.webpage.python.benchmark_analyzer import read_summary_rows

    path = tmp_path / "summary.tsv"
    path.write_text(
        "kind\tmethod\ttask\tdataset\tscenes\tdelta125_2_acc\tdelta125_3_acc\n"
        "depth\tFOO\tMonoDepth\tibims1\t1\t0.95\t0.99\n")
    rows = read_summary_rows(path)
    assert len(rows) == 1
    row = rows[0]
    assert abs(row["metrics"]["delta125_2_percent"] - 95.0) < 1e-9
    assert abs(row["metrics"]["delta125_3_percent"] - 99.0) < 1e-9


def test_slam_records_off_the_protocol_range_are_left_out() -> None:
    """A SLAM record on a range the protocol does not evaluate the method on (the large range for the low-budget
    methods) is dropped; its other ranges, other tasks, range-less records and unknown methods are kept."""
    from app.webpage.python.benchmark_analyzer import on_slam_protocol

    for method in ("MAPANYTHING", "MAPANYTHING_rgb_depth", "MAST3R", "DEJAVIEW",
                   "DUST3R", "POW3R", "IGGT4D"):
        assert not on_slam_protocol(method, "SLAM", "large"), method
        assert on_slam_protocol(method, "SLAM", "short") and on_slam_protocol(
            method, "SLAM", "long"), method
    assert on_slam_protocol("LINGBOT_MAP", "SLAM", "large")
    assert not on_slam_protocol(
        "VGGT", "SLAM", "long")  # budget below the long range's longest input
    assert on_slam_protocol("MAST3R", "SfM", "large") and on_slam_protocol(
        "MAST3R", "SLAM", "")
    assert on_slam_protocol("CUSTOM_METHOD", "SLAM", "large")


def test_slam_rgb_depth_failures_on_sequences_without_depth_are_dropped(
        tmp_path):
    """A [+D] failure on a SLAM sequence without depth maps (TRACK_DATASETS / TRACK_EXCLUDED_SCENES) is not a case the
    track runs on: it is left out, and the method's score and coverage count only the sequences it runs on."""
    from app.webpage.python.benchmark_analyzer import read_case_failures as read_tsv_failures, slam_method_cases

    failures = tmp_path / "case_failures.tsv"
    failures.write_text(
        "method\ttask\tdataset\tvariant\tscene\tstatus\tdetail\n"
        "PI3_X_rgb_depth\tSLAM\toxford_spires\tshort\tkeble-college-02\tblocked_missing_depth_prior\t\n"
        "PI3_X_rgb_depth\tSLAM\tspatialarena\tshort\tEdinburghCastle1\tblocked_missing_depth_prior\t\n"
        "PI3_X_rgb_depth\tSLAM\tspatialarena\tlong\tArtHouse\tgpu_oom\t\n"
        "PI3_X\tSLAM\tspatialarena\tshort\tEdinburghCastle1\tgpu_oom\t\n",
        encoding="utf-8",
    )
    kept = [(failure["method"], failure["scene"])
            for failure in read_tsv_failures(failures)]
    assert kept == [("PI3_X_rgb_depth", "ArtHouse"),
                    ("PI3_X", "EdinburghCastle1")]
    cases = ["ArtHouse", "StPancrasOutside"]
    assert slam_method_cases("PI3_X_rgb_depth", "spatialarena",
                             cases) == ["ArtHouse"]
    assert slam_method_cases("PI3_X", "spatialarena", cases) == cases


def test_method_limitations_name_the_method_own_failures():
    """A row's limitations are its failures in the method's own code or outputs, with the record's cause when it
    gives one; out-of-memory and run failures are not limitations."""
    from app.webpage.python.benchmark_analyzer import method_limitations

    failures = {
        "a": {
            "status": "int32_index_limit",
            "label": "Int32 index limit",
            "detail": "Tensor too large for 32-bit math"
        },
        "b": {
            "status": "int32_index_limit",
            "label": "Int32 index limit",
            "detail": "~/reports/AMB3R/eval_depth.json"
        },
        "c": {
            "status": "gpu_oom",
            "label": "GPU OOM",
            "detail": ""
        },
        "d": {
            "status": "not_scored",
            "label": "Not scored",
            "detail": "Predictions exist"
        },
    }
    entries = method_limitations(failures)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["label"] == "Int32 index limit" and entry["count"] == 2
    assert entry["notes"][0] == "Tensor too large for 32-bit math"
    assert "32-bit indexing" in entry["notes"][
        1]  # a report path is not a cause: the generic note
    assert not method_limitations([failures["c"], failures["d"]])
