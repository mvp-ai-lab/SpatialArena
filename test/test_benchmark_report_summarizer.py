# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import importlib.util
import json
import sys
from pathlib import Path


def load_summarizer():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "test"
        / "summarize_benchmark_reports.py"
    )
    spec = importlib.util.spec_from_file_location("summarize_benchmark_reports", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_report_summarizer_weights_depth_by_metric_units_and_pose_metrics_by_pairs(tmp_path):
    summarizer = load_summarizer()
    report_root = tmp_path / "DA3"
    depth_report = report_root / "VideoDepth" / "kitti" / "scene" / "eval_depth.json"
    sparse_report = report_root / "SfM" / "imc" / "scene" / "eval_sparse.json"
    depth_report.parent.mkdir(parents=True)
    sparse_report.parent.mkdir(parents=True)
    depth_report.write_text(
        json.dumps(
            {
                "scene_results": [
                        {
                            "method": "DA3",
                            "task": "VideoDepth",
                            "dataset": "kitti",
                            "num_evaluated_frames": 1,
                            "num_metric_units": 1,
                            "num_valid_pixels": 2,
                            "metrics": {"abs_relative_difference": 1.0},
                        },
                        {
                            "method": "DA3",
                            "task": "VideoDepth",
                            "dataset": "kitti",
                            "num_evaluated_frames": 1,
                            "num_metric_units": 1,
                            "num_valid_pixels": 6,
                            "metrics": {"abs_relative_difference": 3.0},
                        },
                ]
            }
        ),
        encoding="utf-8",
    )
    sparse_report.write_text(
        json.dumps(
            {
                "scene_results": [
                    {
                        "method": "DA3",
                        "task": "SfM",
                        "dataset": "imc",
                        "num_matched_images": 2,
                        "metrics": {
                            "num_pairs": 4,
                            "ate_rmse": 1.0,
                            "relative_rotation_error_deg_mean": 2.0,
                            "relative_translation_error_deg_mean": 4.0,
                            "auc_3": 0.5,
                        },
                    },
                    {
                        "method": "DA3",
                        "task": "SfM",
                        "dataset": "imc",
                        "num_matched_images": 6,
                        "metrics": {
                            "num_pairs": 8,
                            "ate_rmse": 3.0,
                            "relative_rotation_error_deg_mean": 5.0,
                            "relative_translation_error_deg_mean": 7.0,
                            "auc_3": 1.0,
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = summarizer.collect_scene_metrics([report_root])
    summaries = summarizer.aggregate_by_dataset(rows)
    by_kind = {summary["kind"]: summary for summary in summaries}

    assert by_kind["depth"]["abs_relative_difference"] == 2.0
    assert by_kind["depth"]["valid_pixels"] == 8
    assert by_kind["depth"]["metric_units"] == 2
    assert by_kind["sparse"]["ate_rmse"] == 2.5
    assert by_kind["sparse"]["relative_rotation_error_deg_mean"] == 4.0
    assert by_kind["sparse"]["relative_translation_error_deg_mean"] == 6.0
    assert by_kind["sparse"]["auc_3"] == (0.5 * 4 + 1.0 * 8) / 12
    markdown = summarizer.render_markdown([report_root], summaries)
    assert "ATE RMSE" in markdown
    assert "Rel Rot Mean" in markdown
    assert "Rel Trans Mean" in markdown


def test_report_summarizer_can_exclude_scannet(tmp_path):
    summarizer = load_summarizer()
    report_root = tmp_path / "VGGT"
    depth_report = report_root / "MonoDepth" / "marigold" / "eval_depth.json"
    depth_report.parent.mkdir(parents=True)
    depth_report.write_text(
        json.dumps(
            {
                "scene_results": [
                    {
                        "method": "VGGT",
                        "task": "MonoDepth",
                        "dataset": "marigold",
                        "scene": "nyuv2",
                        "num_evaluated_frames": 1,
                        "num_metric_units": 1,
                        "num_valid_pixels": 2,
                        "metrics": {"abs_relative_difference": 1.0},
                    },
                    {
                        "method": "VGGT",
                        "task": "MonoDepth",
                        "dataset": "marigold",
                        "scene": "scannet",
                        "num_evaluated_frames": 1,
                        "num_metric_units": 1,
                        "num_valid_pixels": 2,
                        "metrics": {"abs_relative_difference": 9.0},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = summarizer.collect_scene_metrics([report_root], exclude_scannet=True)
    summaries = summarizer.aggregate_by_dataset(rows)
    markdown = summarizer.render_markdown([report_root], summaries, exclude_scannet=True)

    assert len(rows) == 1
    assert summaries[0]["abs_relative_difference"] == 1.0
    assert "Excluded datasets: ScanNet." in markdown


def test_report_summarizer_separates_depth_and_mvrecon_metrics(tmp_path):
    summarizer = load_summarizer()
    report_root = tmp_path / "PI3_X"
    depth_report = report_root / "MonoDepth" / "marigold" / "nyuv2" / "eval_depth.json"
    mvrecon_report = report_root / "MVRecon" / "dtu" / "scan1" / "eval_mvrecon.json"
    depth_report.parent.mkdir(parents=True)
    mvrecon_report.parent.mkdir(parents=True)
    depth_report.write_text(
        json.dumps(
            {
                "scene_results": [
                    {
                        "method": "PI3_X",
                        "task": "MonoDepth",
                        "dataset": "marigold",
                        "scene": "nyuv2",
                        "prediction_scene_path": "/tmp/does-not-matter",
                        "num_evaluated_frames": 1,
                        "num_metric_units": 1,
                        "num_valid_pixels": 2,
                        "metrics": {
                            "abs_relative_difference": 1.0,
                            "squared_relative_difference": 2.0,
                            "rmse_linear": 3.0,
                            "rmse_log": 4.0,
                            "log10": 5.0,
                            "delta103_acc": 6.0,
                            "delta125_acc": 7.0,
                            "delta125_2_acc": 8.0,
                            "delta125_3_acc": 9.0,
                            "i_rmse": 10.0,
                            "silog_rmse": 11.0,
                            "absrel_percent": 12.0,
                            "delta103_percent": 13.0,
                            "delta125_percent": 14.0,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    mvrecon_report.write_text(
        json.dumps(
            {
                "scene_results": [
                    {
                        "method": "PI3_X",
                        "task": "MVRecon",
                        "dataset": "dtu",
                        "scene": "scan1",
                        "prediction_scene_path": "/tmp/does-not-matter",
                        "num_evaluated_frames": 1,
                        "num_metric_units": 1,
                        "num_valid_pixels": 2,
                        "metrics": {
                            "accuracy": 0.25,
                            "completion": 0.5,
                            "overall": 0.375,
                            "precision": 0.6,
                            "recall": 0.7,
                            "fscore": 0.65,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = summarizer.collect_scene_metrics([report_root])
    summaries = summarizer.aggregate_by_dataset(rows)
    by_task = {(summary["task"], summary["dataset"]): summary for summary in summaries}

    assert by_task[("MonoDepth", "marigold")]["abs_relative_difference"] == 1.0
    assert by_task[("MonoDepth", "marigold")]["absrel_percent"] == 12.0
    assert by_task[("MVRecon", "dtu")]["accuracy"] == 0.25
    assert by_task[("MVRecon", "dtu")]["completion"] == 0.5
    assert "mae_normal_deg" not in by_task[("MVRecon", "dtu")]


def test_report_summarizer_aggregates_scene_runtime(tmp_path):
    summarizer = load_summarizer()
    report = tmp_path / "VGGT" / "SfM" / "imc" / "eval_sparse.json"
    report.parent.mkdir(parents=True)

    def scene(name, runtime):
        return {
            "method": "VGGT",
            "task": "SfM",
            "dataset": "imc",
            "scene": name,
            "num_matched_images": 10,
            "metrics": {"ate_rmse": 1.0, "num_pairs": 45},
            "runtime": runtime,
        }

    report.write_text(
        json.dumps(
            {
                "scene_results": [
                    scene("a", {"inference_seconds": 2.0, "peak_gpu_memory_mib": 100.0,
                                "runtime_scope": "after_model_load", "num_input_images": 10}),
                    scene("b", {"inference_seconds": 6.0, "peak_gpu_memory_mib": 300.0,
                                "runtime_scope": "after_model_load", "num_input_images": 30}),
                    scene("c", None),
                ]
            }
        ),
        encoding="utf-8",
    )
    rows = summarizer.collect_scene_metrics([tmp_path / "VGGT"])
    [summary] = summarizer.aggregate_by_dataset(rows)

    assert summary["scenes"] == 3
    assert summary["runtime_scenes"] == 2
    assert summary["inference_seconds_mean"] == 4.0
    assert summary["seconds_per_image"] == 0.2
    assert summary["peak_gpu_memory_mib_mean"] == 200.0
    assert summary["peak_gpu_memory_mib_max"] == 300.0
    assert summary["runtime_scope"] == "after_model_load"
    assert "| VGGT | sparse | SfM | imc | 2 |" in summarizer.render_markdown([tmp_path], [summary])
    assert "runtime_scope" in summarizer.render_tsv([summary]).splitlines()[0]


def test_report_summarizer_ranks_slam_by_ate_plus_drift_over_complete_methods(tmp_path):
    summarizer = load_summarizer()

    def write(method, sequences):
        report = tmp_path / method / "SLAM" / "tum_dynamics" / "eval_slam.json"
        report.parent.mkdir(parents=True)
        report.write_text(
            json.dumps(
                {
                    "scene_results": [
                        {"method": method, "task": "SLAM", "dataset": "tum_dynamics", "scene": scene, "stride": 10,
                         "slam_range": "short", "num_matched_images": 50,
                         "metrics": {"ate_rmse": ate, "drift_pct": drift, "completeness": 1.0}}
                        for scene, ate, drift in sequences
                    ]
                }
            ),
            encoding="utf-8",
        )

    write("A", [("s1", 0.10, 5.0), ("s2", 0.30, 7.0)])  # ATE 0.2, drift 6
    write("B", [("s1", 0.30, 1.0), ("s2", 0.30, 3.0)])  # ATE 0.3, drift 2
    write("C", [("s1", 0.25, 2.0), ("s2", 0.15, 4.0)])  # ATE 0.2, drift 3
    write("LONG_SKIPPED", [("s1", 0.01, 0.1)])  # skipped s2 (frame budget): reported, not ranked

    rows = summarizer.deduplicate_scene_metrics(summarizer.collect_scene_metrics([tmp_path]))
    summaries = {row["method"]: row for row in summarizer.aggregate_by_dataset(rows)}

    assert summaries["A"]["variant"] == "short"
    assert summaries["A"]["ate_rmse"] == 0.2 and summaries["A"]["drift_pct"] == 6.0
    assert (summaries["A"]["ate_rank"], summaries["A"]["drift_rank"]) == (1.5, 3.0)
    assert summaries["B"]["slam_rank_score"] == (3.0 + 1.0) / 2
    assert summaries["C"]["slam_rank_score"] == (1.5 + 2.0) / 2
    assert summaries["LONG_SKIPPED"]["sequences_total"] == 2
    assert summaries["LONG_SKIPPED"]["slam_rank_score"] is None
    markdown = summarizer.render_markdown([tmp_path], list(summaries.values()))
    assert "| C | tum_dynamics | short | 2/2 | 1.75" in markdown
    assert "| LONG_SKIPPED | tum_dynamics | short | 1/2 | - |" in markdown
