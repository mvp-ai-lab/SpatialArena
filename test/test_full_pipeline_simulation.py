# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-noncommercial-1.0.0

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatialarena.eval.dataset_registry import find_dataset_spec
from spatialarena.eval.prediction_artifacts import validate_method_scene_output
from spatialarena.eval.sparse_reconstruction_evaluator import evaluate_sparse_scene
from scripts.test.generate_benchmark_pdf_report import write_pdf


def write_colmap_text_model(model_dir: Path, image_names: list[str]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "cameras.txt").write_text(
        "\n".join(
            [
                "# Camera list with one line of data per camera:",
                "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
                "# Number of cameras: 1",
                "1 PINHOLE 4 4 4.0 4.0 2.0 2.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    image_lines: list[str] = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        "# Number of images: 2",
    ]
    for index, image_name in enumerate(image_names, start=1):
        image_lines.append(f"{index} 1 0 0 0 {index - 1}.0 0.0 0.0 1 {image_name}")
        image_lines.append("")
    (model_dir / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    (model_dir / "points3D.txt").write_text(
        "\n".join(
            [
                "# 3D point list with one line of data per point:",
                "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]",
                "# Number of points: 0, mean track length: 0",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_ply(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "0 0 0",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_png(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color).save(path)


def load_cases(case_file: Path) -> list[tuple[str, str, str, str, str]]:
    cases = []
    for line in case_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) == 4:
            task, dataset, scene, interval = parts
            order = ""
        elif len(parts) == 5:
            task, dataset, scene, interval, order = parts
        else:
            raise AssertionError(f"unexpected case format: {line}")
        cases.append((task, dataset, scene, interval, order))
    return cases


def write_depth_map(path: Path, value: float) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.full((4, 4), value, dtype=np.float32))


def materialize_synthetic_depth_root(root: Path, scene_names: list[str]) -> None:
    image_names = ["000000.png", "000001.png"]
    root.mkdir(parents=True, exist_ok=True)
    for scene_name in scene_names:
        scene_root = root / scene_name
        images_dir = scene_root / "images"
        depths_dir = scene_root / "depths"
        sparse_root = scene_root / "sparse" / "0"
        for index, image_name in enumerate(image_names):
            write_png(images_dir / image_name, (40 + index * 40, 80, 120))
            write_depth_map(depths_dir / f"{Path(image_name).stem}.npy", 1.0 + index)
        write_colmap_text_model(sparse_root, image_names)


def materialize_synthetic_pose_root(root: Path, scene_names: list[str]) -> None:
    image_names = ["000000.png", "000001.png"]
    root.mkdir(parents=True, exist_ok=True)
    for scene_name in scene_names:
        scene_root = root / scene_name
        images_dir = scene_root / "images"
        gt_sparse = scene_root / "sparse" / "0"
        for index, image_name in enumerate(image_names):
            write_png(images_dir / image_name, (40 + index * 40, 80, 120))
        write_colmap_text_model(gt_sparse, image_names)


def write_metadata(root: Path, dataset_name: str, scene_names: list[str]) -> None:
    metadata = {
        "dataset_name": dataset_name,
        "root_directory": str(root),
        "total_images": len(scene_names) * 2,
        "average_images_per_scene": 2,
        "scenes": [
            {
                "dataset_name": dataset_name,
                "scene_name": scene_name,
                "root_directory": str(root / scene_name),
                "total_images": 2,
            }
            for scene_name in scene_names
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def test_full_pipeline_simulation_generates_contract_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    benchmark_root = tmp_path / "benchmarks"
    eval_root = tmp_path / "evaluation"
    depth_case_file = repo_root / "scripts" / "test" / "full_standard_depth_cases.txt"
    case_file = repo_root / "scripts" / "test" / "full_standard_sfm_cases.txt"
    depth_cases = load_cases(depth_case_file)
    cases = load_cases(case_file)

    by_spec: dict[tuple[str, str], set[str]] = {}
    for task, dataset, scene, _interval, _order in depth_cases + cases:
        spec = find_dataset_spec(task, dataset)
        assert spec is not None, f"missing dataset spec for {task}/{dataset}"
        by_spec.setdefault((spec.task, spec.dataset), set()).add(scene)

    for (task, dataset), scene_names_set in sorted(by_spec.items()):
        spec = find_dataset_spec(task, dataset)
        assert spec is not None
        data_path = spec.data_path(benchmark_root)
        scene_names = sorted(scene_names_set)
        if task in {"MonoDepth", "VideoDepth", "MVRecon"}:
            materialize_synthetic_depth_root(data_path, scene_names)
        else:
            materialize_synthetic_pose_root(data_path, scene_names)
        if spec.metadata_required:
            write_metadata(data_path, dataset, scene_names)
    report_dir = eval_root / "reports" / "contract-sim"
    depth_report_dir = report_dir / "full-depth" / "PI3_X"
    pose_report_dir = report_dir / "full-pose" / "PI3_X"
    depth_artifact_root = eval_root / "artifacts" / "full_standard" / "benchmark_full-depth_PI3_X"
    pose_artifact_root = eval_root / "artifacts" / "full_standard" / "benchmark_full-pose_PI3_X"
    depth_summary_rows = [
        "method\ttask\tdataset\tscene\tinterval\torder\toutput_path\treport_dir\tprediction_retention"
    ]
    pose_summary_rows = [
        "method\ttask\tdataset\tscene\tinterval\torder\toutput_path\treport_dir\tprediction_retention"
    ]
    failure_rows = ["method\ttask\tdataset\tscene\tinterval\torder\tstatus"]

    scene_cache: dict[tuple[str, str], dict[str, object]] = {}
    for (task, dataset), scene_names_set in sorted(by_spec.items()):
        spec = find_dataset_spec(task, dataset)
        assert spec is not None
        dataset_obj = spec.create_dataset(benchmark_root)
        scene_cache[(task, dataset)] = {
            scene.scene: scene
            for scene in dataset_obj.list_scenes()
        }

    for task, dataset, scene, interval, order in depth_cases:
        scene_info = scene_cache[(task, dataset)][scene]
        scene_output = depth_artifact_root / task / dataset / "PI3_X" / scene
        case_report = depth_report_dir / task / dataset / scene
        if task == "MVRecon":
            prediction_sparse = scene_output / "sparse" / "0"
            prediction_sparse.mkdir(parents=True, exist_ok=True)
            shutil.copy2(scene_info.sparse_gt_path / "cameras.txt", prediction_sparse / "cameras.txt")
            shutil.copy2(scene_info.sparse_gt_path / "images.txt", prediction_sparse / "images.txt")
            shutil.copy2(scene_info.sparse_gt_path / "points3D.txt", prediction_sparse / "points3D.txt")
            write_ply(scene_output / "points3D_dense.ply")
            (scene_output / "metrics.json").write_text(
                json.dumps({"method": "PI3_X", "task": task, "dataset": dataset, "scene": scene, "num_images": 2},
                           indent=2)
                + "\n",
                encoding="utf-8",
            )
            verify_status = validate_method_scene_output(
                experiment_root=eval_root / "artifacts",
                experiment_name="full_standard/benchmark_full-depth_PI3_X",
                task=task,
                dataset=dataset,
                method="PI3_X",
                scene=scene,
                requirements=None,
            )
            case_report.mkdir(parents=True, exist_ok=True)
            (case_report / "verify.json").write_text(json.dumps(verify_status.to_dict(), indent=2) + "\n",
                                                     encoding="utf-8")
            eval_result = evaluate_sparse_scene(scene_info, scene_output)
            (case_report / "eval_mvrecon.json").write_text(
                json.dumps({"scene_results": [eval_result]}, indent=2) + "\n", encoding="utf-8")
            (case_report / "evaluation.log").write_text(
                f"simulated reconstruction evaluation for {task}/{dataset}/{scene}\n", encoding="utf-8")
            (case_report / "evaluation.err").write_text("", encoding="utf-8")
            (case_report / "depth_visualizations").mkdir(parents=True, exist_ok=True)
        else:
            images_dir = scene_output / "images"
            depths_dir = scene_output / "depths"
            images_dir.mkdir(parents=True, exist_ok=True)
            depths_dir.mkdir(parents=True, exist_ok=True)
            for index, image_name in enumerate(["000000.png", "000001.png"]):
                write_png(images_dir / image_name, (40 + index * 40, 80, 120))
                write_depth_map(depths_dir / f"{Path(image_name).stem}.npy", 1.0 + index)
            (scene_output / "metrics.json").write_text(
                json.dumps({"method": "PI3_X", "task": task, "dataset": dataset, "scene": scene, "num_images": 2},
                           indent=2)
                + "\n",
                encoding="utf-8",
            )
            case_report.mkdir(parents=True, exist_ok=True)
            depth_eval = {
                "task": task,
                "dataset": dataset,
                "scene": scene,
                "method": "PI3_X",
                "num_evaluated_frames": 2,
                "num_metric_units": 1,
                "num_valid_pixels": 8,
                "metrics": {
                    "abs_relative_difference": 0.1,
                    "squared_relative_difference": 0.2,
                    "rmse_linear": 0.3,
                    "rmse_log": 0.4,
                    "log10": 0.5,
                    "delta103_acc": 0.6,
                    "delta125_acc": 0.7,
                    "delta125_2_acc": 0.8,
                    "delta125_3_acc": 0.9,
                    "i_rmse": 1.0,
                    "silog_rmse": 2.0,
                    "mae_normal_deg": 3.0,
                    "point_accuracy": 0.01,
                    "point_completion": 0.02,
                    "point_chamfer": 0.015,
                    "point_f1": 0.5,
                    "accuracy": 0.01,
                    "completion": 0.02,
                    "overall": 0.015,
                    "precision": 0.5,
                    "recall": 0.5,
                    "fscore": 0.5,
                },
            }
            (case_report / "eval_depth.json").write_text(
                json.dumps({"scene_results": [depth_eval]}, indent=2) + "\n", encoding="utf-8")
            (case_report / "evaluation.log").write_text(
                f"simulated depth evaluation for {task}/{dataset}/{scene}\n", encoding="utf-8")
            (case_report / "evaluation.err").write_text("", encoding="utf-8")
            (case_report / "depth_visualizations").mkdir(parents=True, exist_ok=True)
        depth_summary_rows.append(
            "\t".join(
                [
                    "PI3_X",
                    task,
                    dataset,
                    scene,
                    interval,
                    order,
                    str(scene_output),
                    str(case_report),
                    "kept",
                ]
            )
        )

    for task, dataset, scene, interval, order in cases:
        scene_info = scene_cache[(task, dataset)][scene]
        scene_output = pose_artifact_root / task / dataset / "PI3_X" / scene
        case_report = pose_report_dir / task / dataset / scene
        prediction_sparse = scene_output / "sparse" / "0"
        prediction_sparse.mkdir(parents=True, exist_ok=True)
        shutil.copy2(scene_info.sparse_gt_path / "cameras.txt", prediction_sparse / "cameras.txt")
        shutil.copy2(scene_info.sparse_gt_path / "images.txt", prediction_sparse / "images.txt")
        shutil.copy2(scene_info.sparse_gt_path / "points3D.txt", prediction_sparse / "points3D.txt")
        write_ply(scene_output / "points3D_dense.ply")
        (scene_output / "metrics.json").write_text(
            json.dumps({"method": "PI3_X", "task": task, "dataset": dataset, "scene": scene, "num_images": 2}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        verify_status = validate_method_scene_output(
            experiment_root=eval_root / "artifacts",
            experiment_name="full_standard/benchmark_full-pose_PI3_X",
            task=task,
            dataset=dataset,
            method="PI3_X",
            scene=scene,
            requirements=None,
        )
        case_report.mkdir(parents=True, exist_ok=True)
        (case_report / "verify.json").write_text(json.dumps(verify_status.to_dict(), indent=2) + "\n", encoding="utf-8")
        eval_result = evaluate_sparse_scene(scene_info, scene_output)
        (case_report / "eval_sparse.json").write_text(json.dumps(eval_result, indent=2) + "\n", encoding="utf-8")
        (case_report / "evaluation.log").write_text(f"simulated evaluation for {task}/{dataset}/{scene}\n",
                                                    encoding="utf-8")
        (case_report / "evaluation.err").write_text("", encoding="utf-8")
        (case_report / "depth_visualizations").mkdir(parents=True, exist_ok=True)
        pose_summary_rows.append(
            "\t".join(
                [
                    "PI3_X",
                    task,
                    dataset,
                    scene,
                    interval,
                    order,
                    str(scene_output),
                    str(case_report),
                    "kept",
                ]
            )
        )
        assert verify_status.ok
        assert eval_result["scene"] == scene
        assert eval_result["num_matched_images"] == 2
        assert eval_result["metrics"]["num_pairs"] == 1
        assert eval_result["metrics"]["ate_rmse"] == 0.0

    texturepoorsfm_scene = next(scene for _task, dataset, scene, _interval, _order in cases
                                if dataset == "texturepoorsfm")
    depth_report_dir.mkdir(parents=True, exist_ok=True)
    pose_report_dir.mkdir(parents=True, exist_ok=True)
    (depth_report_dir / "summary.tsv").write_text("\n".join(depth_summary_rows) + "\n", encoding="utf-8")
    (depth_report_dir / "case_failures.tsv").write_text("\n".join(failure_rows) + "\n", encoding="utf-8")
    (depth_report_dir / "summary.json").write_text(
        json.dumps(
            {
                "method": "PI3_X",
                "base_method": "PI3_X",
                "benchmark_track": "rgb",
                "experiment_root": str(eval_root / "artifacts"),
                "experiment_name": "full_standard/benchmark_full-depth_PI3_X",
                "case_file": str(depth_case_file),
                "report_dir": str(depth_report_dir),
                "summary_tsv": str(depth_report_dir / "summary.tsv"),
                "case_failures_tsv": str(depth_report_dir / "case_failures.tsv"),
                "keep_predictions": "1",
                "keep_depth_visualizations": "1",
                "generate_pdf_report": "1",
                "allow_benchmark_root_output": "0",
                "allow_benchmark_eval_output": "1",
                "benchmark_aux_root": str(benchmark_root),
                "benchmark_eval_root": str(eval_root),
                "prediction_cache_root": str(eval_root / "artifacts" / "_prediction_cache"),
                "resolution_profile": {
                    "resolution": 512,
                    "image_load_resolution": 512,
                    "pixel_limit": 262144,
                    "conf_thresh": 0.0,
                    "depth_conf_thresh": 0.0,
                    "resolution_source": ("Pi3 evaluation branch uses load_img_size=512 for MonoDepth, VideoDepth, "
                                          "and relative pose; MVRecon is 518."),
                    "dense_depth_confidence_filtering": False,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (pose_report_dir / "summary.tsv").write_text("\n".join(pose_summary_rows) + "\n", encoding="utf-8")
    (pose_report_dir / "case_failures.tsv").write_text("\n".join(failure_rows) + "\n", encoding="utf-8")
    (pose_report_dir / "summary.json").write_text(
        json.dumps(
            {
                "method": "PI3_X",
                "base_method": "PI3_X",
                "benchmark_track": "rgb",
                "experiment_root": str(eval_root / "artifacts"),
                "experiment_name": "full_standard/benchmark_full-pose_PI3_X",
                "case_file": str(case_file),
                "report_dir": str(pose_report_dir),
                "summary_tsv": str(pose_report_dir / "summary.tsv"),
                "case_failures_tsv": str(pose_report_dir / "case_failures.tsv"),
                "keep_predictions": "1",
                "keep_depth_visualizations": "1",
                "generate_pdf_report": "1",
                "allow_benchmark_root_output": "0",
                "allow_benchmark_eval_output": "1",
                "benchmark_aux_root": str(benchmark_root),
                "benchmark_eval_root": str(eval_root),
                "prediction_cache_root": str(eval_root / "artifacts" / "_prediction_cache"),
                "resolution_profile": {
                    "resolution": 512,
                    "image_load_resolution": 512,
                    "pixel_limit": 262144,
                    "conf_thresh": 0.0,
                    "depth_conf_thresh": 0.0,
                    "resolution_source": ("Pi3 evaluation branch uses load_img_size=512 for MonoDepth, VideoDepth, "
                                          "and relative pose; MVRecon is 518."),
                    "dense_depth_confidence_filtering": False,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (report_dir / "case_failures.tsv").write_text("\n".join(failure_rows) + "\n", encoding="utf-8")
    write_pdf([depth_report_dir, pose_report_dir], report_dir / "report.pdf",
              title="FeedForward Benchmark: PI3_X (rgb)", exclude_scannet=False)

    assert (depth_artifact_root / "MVRecon" / "dtu" / "PI3_X" / "scan24" / "sparse" / "0" / "images.txt").is_file()
    assert (pose_artifact_root / "SfM" / "texturepoorsfm" / "PI3_X" / texturepoorsfm_scene / "sparse" / "0"
            / "images.txt").is_file()
    assert (depth_report_dir / "summary.tsv").is_file()
    assert (pose_report_dir / "summary.tsv").is_file()
    assert (depth_report_dir / "summary.json").is_file()
    assert (pose_report_dir / "summary.json").is_file()
    assert (depth_report_dir / "case_failures.tsv").is_file()
    assert (pose_report_dir / "case_failures.tsv").is_file()
    assert (report_dir / "report.pdf").is_file()
    assert (report_dir / "report.pdf").stat().st_size > 0

    depth_summary_tsv = (depth_report_dir / "summary.tsv").read_text(encoding="utf-8").splitlines()
    pose_summary_tsv = (pose_report_dir / "summary.tsv").read_text(encoding="utf-8").splitlines()
    summary_header = "method\ttask\tdataset\tscene\tinterval\torder\toutput_path\treport_dir\tprediction_retention"
    assert depth_summary_tsv[0] == summary_header
    assert pose_summary_tsv[0] == summary_header
    assert len(depth_summary_tsv) == len(depth_cases) + 1
    assert len(pose_summary_tsv) == len(cases) + 1
    assert all("\tkept" in line for line in depth_summary_tsv[1:])
    assert all("\tkept" in line for line in pose_summary_tsv[1:])
    case_failures = (report_dir / "case_failures.tsv").read_text(encoding="utf-8").splitlines()
    assert case_failures == ["method\ttask\tdataset\tscene\tinterval\torder\tstatus"]

    sample_case_report = pose_report_dir / "SfM" / "texturepoorsfm" / texturepoorsfm_scene
    eval_result = json.loads((sample_case_report / "eval_sparse.json").read_text(encoding="utf-8"))
    assert eval_result["num_matched_images"] == 2
    assert eval_result["metrics"]["num_pairs"] == 1
    assert eval_result["metrics"]["ate_rmse"] == 0.0
