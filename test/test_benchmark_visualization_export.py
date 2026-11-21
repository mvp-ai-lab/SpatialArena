# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import json
import importlib.util
from pathlib import Path


def load_benchmark_analyzer():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "webpage"
        / "python"
        / "benchmark_analyzer.py"
    )
    spec = importlib.util.spec_from_file_location("benchmark_analyzer", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_benchmark_visualization_ranks_depth_and_pose_metrics(tmp_path):
    analyzer = load_benchmark_analyzer()
    summary_path = tmp_path / "summary.tsv"
    summary_path.write_text(
        "\t".join(
            [
                "kind",
                "method",
                "task",
                "dataset",
                "scenes",
                "frames",
                "valid_pixels",
                "matched_images",
                "pairs",
                "absrel_percent",
                "delta125_percent",
                "ate_rmse",
                "mAA_30",
            ]
        )
        + "\n"
        + "depth\tA\tMonoDepth\tnyu\t1\t2\t10\t0\t0\t10\t90\t\t\n"
        + "depth\tB\tMonoDepth\tnyu\t1\t2\t10\t0\t0\t20\t80\t\t\n"
        + "sparse\tA\tSfM\timc\t1\t0\t0\t10\t30\t\t\t2\t0.5\n"
        + "sparse\tB\tSfM\timc\t1\t0\t0\t10\t30\t\t\t1\t0.9\n"
        + "sparse\tPI3_X_rgb_depth\tSfM\timc\t1\t0\t0\t10\t30\t\t\t3\t0.4\n"
        + "sparse\tA\tSLAM\t7scenes\t1\t0\t0\t10\t30\t\t\t2\t0.5\n"
        + "sparse\tB\tSLAM\t7scenes\t1\t0\t0\t10\t30\t\t\t1\t0.9\n",
        encoding="utf-8",
    )

    data = analyzer.build_benchmark_visualization(summary_path)

    assert data["summary"]["methods"] == 3
    assert data["summary"]["tasks"] == 3
    assert data["task_tables"]["MonoDepth"]["methods"][0]["metrics"]["absrel_percent"]["rank"] == 1
    assert data["task_tables"]["SfM"]["methods"][1]["metrics"]["ate_rmse"]["rank"] == 1
    assert {row["method"] for row in data["ranking"]["rows"]} == {"A", "B"}
    assert data["method_display_names"]["PI3_X_rgb_depth"] == "Pi3X [+D]"
    assert analyzer.display_task_name("VideoDepth") == "Video Depth"
    assert data["radar"]["compact_axes"]
    assert data["radar"]["all_axes"]
    assert data["radar"]["task_axes"]
    assert data["radar"]["dataset_axes"]
    assert analyzer.TASK_ORDER[:5] == ["MonoDepth", "VideoDepth", "MVRecon", "SfM", "SLAM"]
    assert data["task_display_names"]["SLAM"] == "SLAM"


def test_benchmark_visualization_excludes_legacy_da3_when_explicit_da3g_exists(tmp_path):
    analyzer = load_benchmark_analyzer()
    summary_path = tmp_path / "summary.tsv"
    summary_path.write_text(
        "\t".join(
            [
                "kind",
                "method",
                "task",
                "dataset",
                "scenes",
                "frames",
                "valid_pixels",
                "matched_images",
                "pairs",
                "absrel_percent",
                "delta125_percent",
            ]
        )
        + "\n"
        + "depth\tDA3\tMonoDepth\tnyu\t1\t2\t10\t0\t0\t10\t90\n"
        + "depth\tDA3-G\tMonoDepth\tnyu\t1\t2\t10\t0\t0\t11\t89\n"
        + "depth\tDA3-L\tMonoDepth\tnyu\t1\t2\t10\t0\t0\t12\t88\n",
        encoding="utf-8",
    )

    data = analyzer.build_benchmark_visualization(summary_path)

    assert "DA3" not in data["methods"]
    assert {row["method"] for row in data["ranking"]["rows"]} == {"DA3-G", "DA3-L"}
    assert data["summary"]["raw_rows"] == 3
    assert data["summary"]["rows"] == 2
    assert data["summary"]["excluded_legacy_methods"]["DA3"]["replacement"] == "DA3-G"


def test_scannetpp_is_excluded_from_ranking_but_stays_in_task_tables(tmp_path):
    analyzer = load_benchmark_analyzer()
    summary_path = tmp_path / "summary.tsv"
    summary_path.write_text(
        "\t".join(
            [
                "kind",
                "method",
                "task",
                "dataset",
                "scenes",
                "frames",
                "valid_pixels",
                "matched_images",
                "pairs",
                "ate_rmse",
                "mAA_30",
            ]
        )
        + "\n"
        # A wins on scannetpp, B wins on 7scenes -- if scannetpp counted
        # toward the ranking, A and B would tie; excluded, B must win outright.
        + "sparse\tA\tSfM\tscannetpp\t1\t0\t0\t10\t30\t2\t0.9\n"
        + "sparse\tB\tSfM\tscannetpp\t1\t0\t0\t10\t30\t1\t0.5\n"
        + "sparse\tA\tSfM\t7scenes\t1\t0\t0\t10\t30\t2\t0.5\n"
        + "sparse\tB\tSfM\t7scenes\t1\t0\t0\t10\t30\t1\t0.9\n",
        encoding="utf-8",
    )

    data = analyzer.build_benchmark_visualization(summary_path)

    assert data["summary"]["datasets_excluded_from_ranking"] == ["scannetpp"]
    # scannetpp is still fully visible in the per-task/per-dataset tables.
    assert "scannetpp" in data["task_tables"]["SfM"]["dataset_tables"]
    assert data["task_tables"]["SfM"]["dataset_tables"]["scannetpp"]["methods"][0]["method"] == "A"
    # ...but does not count toward the overall/task ranking.
    ranked = {row["method"]: row for row in data["ranking"]["rows"]}
    assert ranked["B"]["rank"] == 1
    assert "scannetpp" not in ranked["A"]["dataset_scores"]
    assert "scannetpp" not in ranked["B"]["dataset_scores"]
    assert "scannetpp" not in data["ranking"]["datasets"]


def test_benchmark_visualization_recovers_geometry_and_mvrecon_metrics_from_reports(tmp_path):
    analyzer = load_benchmark_analyzer()
    summary_path = tmp_path / "summary.tsv"
    summary_path.write_text(
        "\t".join(
            [
                "kind",
                "method",
                "task",
                "dataset",
                "scenes",
                "frames",
                "valid_pixels",
                "matched_images",
                "pairs",
                "absrel_percent",
                "delta125_percent",
                "ate_rmse",
                "mAA_30",
                "accuracy",
                "completion",
                "overall",
                "precision",
                "recall",
                "fscore",
            ]
        )
        + "\n"
        + "depth\tPI3_X\tMonoDepth\tmarigold\t1\t2\t10\t0\t0\t10\t90\t\t\t\t\t\t\t\n"
        + "mvrecon\tPI3_X\tMVRecon\tdtu\t1\t2\t10\t0\t0\t\t\t\t\t\t0.25\t0.5\t0.375\t0.6\t0.7\t0.65\n",
        encoding="utf-8",
    )

    report_root = tmp_path / "reports"
    mono_scene = report_root / "full-depth" / "PI3_X" / "MonoDepth" / "marigold" / "nyuv2" / "eval_depth.json"
    mv_scene = report_root / "full-depth" / "PI3_X" / "MVRecon" / "dtu" / "scan1" / "eval_mvrecon.json"
    mono_scene.parent.mkdir(parents=True)
    mv_scene.parent.mkdir(parents=True)
    mono_scene.write_text(
        json.dumps(
            {
                "scene_results": [
                    {
                        "method": "PI3_X",
                        "task": "MonoDepth",
                        "dataset": "marigold",
                        "scene": "nyuv2",
                        "prediction_scene_path": "/tmp/mono",
                        "num_evaluated_frames": 1,
                        "num_metric_units": 1,
                        "num_valid_pixels": 2,
                        "metrics": {"abs_relative_difference": 1.0},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    mv_scene.write_text(
        json.dumps(
            {
                "scene_results": [
                    {
                        "method": "PI3_X",
                        "task": "RMVD",
                        "dataset": "dtu",
                        "scene": "scan1",
                        "prediction_scene_path": "/tmp/mvrecon",
                        "num_evaluated_frames": 1,
                        "num_metric_units": 1,
                        "num_valid_pixels": 2,
                        "metrics": {"abs_relative_difference": 1.0},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    calls = {"depth": 0, "mvrecon": 0}

    class DummyScene:
        def __init__(self, task: str, dataset: str, scene: str) -> None:
            self.task = task
            self.dataset = dataset
            self.scene = scene
            self.root_path = tmp_path / task / dataset / scene

    class DummyDataset:
        def __init__(self, scene: DummyScene) -> None:
            self._scene = scene

        def list_scenes(self):
            return [self._scene]

    class DummySpec:
        def __init__(self, task: str, dataset: str, scene: DummyScene) -> None:
            self.task = task
            self.dataset = dataset
            self._scene = scene

        def create_dataset(self, benchmark_root):  # pylint: disable=unused-argument
            return DummyDataset(self._scene)

    def fake_find_dataset_spec(task: str, dataset: str):
        scene = DummyScene(task, dataset, "nyuv2" if dataset == "marigold" else "scan1")
        return DummySpec(task, dataset, scene)

    def fake_evaluate_depth_scene(scene_info, prediction_path, config):  # pylint: disable=unused-argument
        assert scene_info.task == "MonoDepth"
        calls["depth"] += 1
        return {
            "metrics": {
                "mae_normal_deg": 1.0,
                "point_accuracy": 0.01,
                "point_completion": 0.02,
                "point_chamfer": 0.015,
                "point_f1": 0.5,
            }
        }

    def fake_evaluate_mvrecon_scene(scene_info, prediction_path, config):  # pylint: disable=unused-argument
        assert scene_info.task == "MVRecon"
        calls["mvrecon"] += 1
        return {
            "metrics": {
                "accuracy": 0.25,
                "completion": 0.5,
                "overall": 0.375,
                "precision": 0.6,
                "recall": 0.7,
                "fscore": 0.65,
            }
        }

    analyzer.find_dataset_spec = fake_find_dataset_spec
    analyzer.evaluate_depth_scene = fake_evaluate_depth_scene
    analyzer.evaluate_mvrecon_scene = fake_evaluate_mvrecon_scene

    data = analyzer.build_benchmark_visualization(summary_path, [report_root])

    mono_metrics = data["task_tables"]["MonoDepth"]["methods"][0]["metrics"]
    mv_metrics = data["task_tables"]["MVRecon"]["methods"][0]["metrics"]

    assert mono_metrics["mae_normal_deg"]["value"] == 1.0
    assert "mae_normal_deg" not in mv_metrics
    assert mv_metrics["accuracy"]["value"] == 0.25
    assert mv_metrics["completion"]["value"] == 0.5
    assert "point_f1" not in mv_metrics
    assert data["task_catalog"]["MVRecon"]["metrics_count"] == 6
    assert calls["depth"] == 1
    assert calls["mvrecon"] == 1


def test_mvrecon_required_metrics_skip_precision_recall_fscore_for_dtu():
    analyzer = load_benchmark_analyzer()

    assert analyzer.mvrecon_required_metrics_for_dataset("dtu") == ("accuracy", "completion", "overall")
    assert analyzer.mvrecon_required_metrics_for_dataset("eth3d") == (
        "accuracy",
        "completion",
        "overall",
        "precision",
        "recall",
        "fscore",
    )


def test_benchmark_visualization_does_not_recover_dtu_rows_with_only_chamfer_metrics(tmp_path):
    """evaluate_dtu_mvsnet (DTU's official protocol) never produces
    precision/recall/fscore -- a dtu row with only accuracy/completion/
    overall is already complete and must not trigger the expensive
    evaluate_mvrecon_scene recovery fallback."""
    analyzer = load_benchmark_analyzer()
    summary_path = tmp_path / "summary.tsv"
    summary_path.write_text(
        "\t".join(
            [
                "kind",
                "method",
                "task",
                "dataset",
                "scenes",
                "frames",
                "valid_pixels",
                "matched_images",
                "pairs",
                "accuracy",
                "completion",
                "overall",
            ]
        )
        + "\n"
        + "mvrecon\tPI3_X\tMVRecon\tdtu\t1\t2\t10\t0\t0\t0.25\t0.5\t0.375\n",
        encoding="utf-8",
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("recovery should not run for a complete dtu row")

    analyzer.find_dataset_spec = fail_if_called
    analyzer.evaluate_mvrecon_scene = fail_if_called

    data = analyzer.build_benchmark_visualization(summary_path, None)

    mv_metrics = data["task_tables"]["MVRecon"]["methods"][0]["metrics"]
    assert mv_metrics["accuracy"]["value"] == 0.25


def test_recovery_only_recomputes_the_scenes_that_are_actually_incomplete(tmp_path):
    """One incomplete row anywhere must not force every scene across every
    report root through evaluate_mvrecon_scene -- only the scenes whose own
    eval_mvrecon.json is actually missing something should be recomputed."""
    analyzer = load_benchmark_analyzer()
    summary_path = tmp_path / "summary.tsv"
    summary_path.write_text(
        "\t".join(
            [
                "kind", "method", "task", "dataset", "scenes", "frames", "valid_pixels",
                "matched_images", "pairs", "accuracy", "completion", "overall",
                "precision", "recall", "fscore",
            ]
        )
        + "\n"
        + "mvrecon\tPI3_X\tMVRecon\tdtu\t1\t2\t10\t0\t0\t0.25\t0.5\t0.375\t\t\t\n"
        + "mvrecon\tPI3_X\tMVRecon\teth3d\t1\t2\t10\t0\t0\t0.1\t0.2\t0.15\t\t\t\n",
        encoding="utf-8",
    )

    report_root = tmp_path / "reports"
    dtu_scene = report_root / "full-depth" / "PI3_X" / "MVRecon" / "dtu" / "scan1" / "eval_mvrecon.json"
    eth3d_scene = report_root / "full-depth" / "PI3_X" / "MVRecon" / "eth3d" / "office" / "eval_mvrecon.json"
    dtu_scene.parent.mkdir(parents=True)
    eth3d_scene.parent.mkdir(parents=True)
    dtu_scene.write_text(
        json.dumps(
            {
                "scene_results": [
                    {
                        "method": "PI3_X", "task": "MVRecon", "dataset": "dtu", "scene": "scan1",
                        "prediction_scene_path": "/tmp/dtu", "num_metric_units": 1,
                        "metrics": {"accuracy": 0.25, "completion": 0.5, "overall": 0.375},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    eth3d_scene.write_text(
        json.dumps(
            {
                "scene_results": [
                    {
                        "method": "PI3_X", "task": "MVRecon", "dataset": "eth3d", "scene": "office",
                        "prediction_scene_path": "/tmp/eth3d", "num_metric_units": 1,
                        "metrics": {"accuracy": 0.1, "completion": 0.2, "overall": 0.15},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    class DummyScene:
        def __init__(self, dataset: str, scene: str) -> None:
            self.dataset = dataset
            self.scene = scene
            self.root_path = tmp_path / dataset / scene

    class DummyDataset:
        def __init__(self, scene: DummyScene) -> None:
            self._scene = scene

        def list_scenes(self):
            return [self._scene]

    class DummySpec:
        def __init__(self, scene: DummyScene) -> None:
            self._scene = scene

        def create_dataset(self, benchmark_root):  # pylint: disable=unused-argument
            return DummyDataset(self._scene)

    def fake_find_dataset_spec(task: str, dataset: str):  # pylint: disable=unused-argument
        return DummySpec(DummyScene(dataset, "scan1" if dataset == "dtu" else "office"))

    recomputed_scenes: list[str] = []

    def fake_evaluate_mvrecon_scene(scene_info, prediction_path, config):  # pylint: disable=unused-argument
        recomputed_scenes.append(scene_info.dataset)
        return {"metrics": {
            "accuracy": 9.9, "completion": 9.9, "overall": 9.9, "precision": 1.0, "recall": 1.0, "fscore": 1.0
        }}

    analyzer.find_dataset_spec = fake_find_dataset_spec
    analyzer.evaluate_mvrecon_scene = fake_evaluate_mvrecon_scene

    analyzer.build_benchmark_visualization(summary_path, [report_root])

    assert recomputed_scenes == ["eth3d"]


def test_metric_keys_for_rows_stays_declared_even_when_unpopulated():
    """A metric column must not disappear just because no row currently has
    a value for it -- e.g. mae_normal_deg for VideoDepth (structurally never
    computed by evaluate_depth_scene, see depth_prediction_evaluator.py) must
    still show up as a column (rendered "-") rather than vanish, so it
    doesn't look like the metric was never part of the protocol."""
    analyzer = load_benchmark_analyzer()

    rows = [
        {"task": "VideoDepth", "metrics": {"abs_relative_difference": 0.1}},
    ]

    keys = analyzer.metric_keys_for_rows(rows)

    assert "mae_normal_deg" in keys
    assert keys == analyzer.metric_keys_for_task("VideoDepth")


def test_every_registered_method_has_a_model_type():
    from spatialarena.methods import METHOD_SPECS  # pylint: disable=import-outside-toplevel

    analyzer = load_benchmark_analyzer()
    untyped = [spec.name for spec in METHOD_SPECS if analyzer.method_type(spec.name) == "other"]
    assert not untyped, f"add these to METHOD_TYPES in benchmark_analyzer.py: {untyped}"
    assert analyzer.method_type("PI3_X_rgb_cameras") == "feedforward_3d"
    assert analyzer.method_type("UNKNOWN_METHOD") == "other"
    assert set(analyzer.METHOD_TYPES.values()) <= set(analyzer.METHOD_TYPE_LABELS)


def test_task_dataset_scores_reproduce_task_scores(tmp_path):
    """The per-(task, dataset) means weighted by comparison count give back each task score, also when a dataset
    name appears under two tasks."""
    analyzer = load_benchmark_analyzer()
    summary_path = tmp_path / "summary.tsv"
    lines = ["kind\tmethod\ttask\tdataset\tscenes\tabsrel_percent\tdelta125_percent\taccuracy\tcompletion"]
    for method, offset in (("A", 0.0), ("B", 1.0), ("C", 2.5)):
        lines += [
            f"depth\t{method}\tMonoDepth\tnyu\t1\t{10 + offset}\t{90 - offset}\t\t",
            f"depth\t{method}\tMonoDepth\tkitti\t1\t{12 - offset}\t{88 + 2 * offset}\t\t",
            f"recon\t{method}\tMVRecon\tdtu\t1\t\t\t{0.5 + offset}\t{0.4 + offset / 2}",
            f"recon\t{method}\tMVRecon\tkitti\t1\t\t\t{0.7 - offset / 10}\t{0.6}",
        ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    data = analyzer.build_benchmark_visualization(summary_path, recover_geometry=False)
    for row in data["task_ranking"]["rows"]:
        for task, by_dataset in row["task_dataset_scores"].items():
            total = sum(entry["coverage"] for entry in by_dataset.values())
            weighted = sum(entry["score"] * entry["coverage"] for entry in by_dataset.values()) / total
            assert abs(weighted - row["task_scores"][task]["score"]) < 1e-9
        assert set(row["task_dataset_scores"]["MVRecon"]) == {"dtu", "kitti"}
    assert data["method_types"] == {"A": "other", "B": "other", "C": "other"}
    assert data["method_type_labels"]["feedforward_3d"] == "Feed-forward 3D/4D"
