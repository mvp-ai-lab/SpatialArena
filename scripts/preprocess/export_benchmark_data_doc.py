# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Export the benchmark scene inventory into docs/evaluation.md (between its BEGIN/END markers)."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatialarena.eval.dataset_registry import (  # noqa: E402
    DEFAULT_BENCHMARK_ROOT,
    describe_benchmark_scenes,
)


BEGIN_MARKER = "<!-- BEGIN BENCHMARK_DATASET_INVENTORY -->"
END_MARKER = "<!-- END BENCHMARK_DATASET_INVENTORY -->"

# The depth tasks' default protocol (inverse_huber_irls_gt, run_method_case_list.sh's MONO_DEPTH_PROTOCOL /
# VIDEO_DEPTH_PROTOCOL) is GT-only: it reports the normal error (`mae_normal_deg`) but no point-cloud metrics.
GT_ONLY_DEPTH_METRICS = (
    "`abs_relative_difference`, `squared_relative_difference`, `rmse_linear`, "
    "`rmse_log`, `log10`, `delta103_acc`, `delta125_acc`, "
    "`delta125_2_acc`, `delta125_3_acc`, `i_rmse`, `silog_rmse`, `mae_normal_deg`, "
    "`absrel_percent`, `delta103_percent`, and `delta125_percent`"
)

POSE_METRICS = (
    "`ate_rmse`, `ate_mean`, `ate_median`, `relative_rotation_error_deg_*`, "
    "`relative_translation_error_deg_*`, `relative_translation_axis_error_deg_*`, "
    "`relative_pose_error_deg_*`, `relative_pose_axis_error_deg_*`, "
    "`RRA_at_5/15/30`, axis-translation `RTA_at_5/15/30`, `mAA_30`, "
    "`auc_1`, `auc_3`, `auc_5`, `auc_10`, `auc_20`, and `auc_30`"
)

MVRECON_METRICS = (
    "`accuracy`, `completion`, `overall` (Chamfer), and `precision`, `recall`, "
    "`fscore` at the dataset threshold"
)

DTU_MVRECON_METRICS = (
    "`accuracy`, `completion`, and `overall` from the MVSNet-style "
    "ObsMask/Plane-filtered DTU evaluation"
)

SLAM_METRICS = (
    "`ate_rmse` (primary, with `drift_pct`), `ate_mean`, `ate_median`, `drift_pct`, `rpe_trans_step`, "
    "`rpe_rot_step`, `rpe_trans_1s`, `rpe_rot_1s`, `auc_1`, `auc_3`, `auc_5`, `auc_10`, `auc_20`, `auc_30` "
    "(pairwise pose AUC, as SfM), `completeness`; reported alongside: `ate_rmse_se3`, "
    "`alignment_scale`, `alignment_inlier_ratio`"
)

TASK_METRICS = {
    "MonoDepth": GT_ONLY_DEPTH_METRICS,
    "VideoDepth": GT_ONLY_DEPTH_METRICS,
    "SfM": POSE_METRICS,
    "SLAM": SLAM_METRICS,
    "MVRecon": MVRECON_METRICS,
}

DEPTH_METRIC_SUMMARY = (
    "AbsRel, SqRel, RMSE, RMSE-log, log10, `Delta@1.03`, `Delta@1.25`, "
    "`Delta@1.25^2`, `Delta@1.25^3`, iRMSE, SILog"
)

SPARSE_POSE_METRIC_SUMMARY = (
    "ATE, relative rotation error, direct relative translation-direction error, "
    "RPE, RRA@5/15/30, axis-translation RTA@5/15/30, mAA@30, "
    "AUC@1/3/5/10/20/30"
)

MVRECON_METRIC_SUMMARY = "Fused-geometry Acc, Comp, Overall, Precision, Recall, F-score"

DATASET_METRICS = {
    ("MVRecon", "dtu"): DTU_MVRECON_METRICS,
}

# Shared by every SfM row: sparse_reconstruction_evaluator.py.
SPARSE_POSE_PROTOCOL = (
    "Read COLMAP predictions from `sparse/0`, match images by filename, compute relative "
    "pose metrics over all matched image pairs, and compute ATE after a pose-aware "
    "Sim(3) alignment (LMedS RANSAC over the GTSAM pose fit + robust refit, "
    "`pose_alignment.py::ransac_irls_align_sim3`)."
)

# Every case is capped at 100 frames (BENCHMARK_MAX_INPUT_FRAMES, fixed-seed
# random subset kept in canonical order) by run_method_case_list.sh.
CAPPED_IMAGES = "Up to 100 canonical images per scene (fixed-seed random subset, canonical order kept)."


def _pose_row(ground_truth: str) -> dict:
    return {
        "images": CAPPED_IMAGES,
        "ground_truth": ground_truth,
        "metrics": SPARSE_POSE_METRIC_SUMMARY,
        "protocol": SPARSE_POSE_PROTOCOL,
    }


# SLAM rows: slam_trajectory_evaluator.py, full-slam suite (docs/benchmark/slam_protocol.md).
SLAM_IMAGES = (
    "Whole recording in temporal order at the range track's stride for the dataset, no frame cap, fed to every "
    "method at one fixed resolution (504x378; VBR 504x252; KITTI Odometry 504x154; `SLAM/`, see "
    "`benchmark/slam_protocol.md`); "
    "only the methods of that range track run."
)
SLAM_METRIC_SUMMARY = "ATE RMSE, drift %, RPE (consecutive frames and 1 s), completeness; ranked by ATE + drift"
SLAM_PROTOCOL = (
    "Match images by filename; Sim(3) alignment by `pose_alignment.py::ransac_irls_align_sim3` (LMedS RANSAC over "
    "the GTSAM pose-residual fit, then a Cauchy IRLS refit on all frames; no ICP) for both `rgb` and `rgb_depth`; "
    "ATE/RPE over every returned frame."
)


def _slam_row(ground_truth: str) -> dict:
    return {
        "images": SLAM_IMAGES,
        "ground_truth": ground_truth,
        "metrics": SLAM_METRIC_SUMMARY,
        "protocol": SLAM_PROTOCOL,
    }


def _mvrecon_row(ground_truth: str, protocol: str, metrics: str = MVRECON_METRIC_SUMMARY) -> dict:
    return {
        "images": CAPPED_IMAGES,
        "ground_truth": ground_truth,
        "metrics": metrics,
        "protocol": (
            "Unposed: Sim(3)-align predicted poses to GT (`ransac_irls_align_sim3`), scale depths by the "
            f"recovered scale, TSDF-fuse, sample 1M points, then score. {protocol}"
        ),
    }


# Mirrors depth_prediction_evaluator.py::resolve_depth_config ("standard"
# protocol) and mvrecon_evaluator.py::protocol_for_dataset.


def status_for_record(record: dict) -> str:
    if not record["exists"]:
        return "pending"
    if not record.get("metadata_required", True):
        return "metadata not required"
    if not record["metadata_exists"]:
        return "downloaded, metadata pending"
    return "metadata ready"


def format_bool(value: bool) -> str:
    return "yes" if value else "no"


def summarize_by_task(records: list[dict]) -> list[str]:
    grouped = defaultdict(lambda: {"datasets": 0, "ready": 0, "scenes": 0, "images": 0})
    for record in records:
        task = record["task"]
        grouped[task]["datasets"] += 1
        if record["metadata_exists"] or not record.get("metadata_required", True):
            grouped[task]["ready"] += 1
        grouped[task]["scenes"] += int(record["scene_count"])
        grouped[task]["images"] += int(record["total_images"])

    lines = [
        "| Task | Datasets | Metadata Ready | Scenes | Images |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for task in sorted(grouped):
        values = grouped[task]
        lines.append(
            f"| {task} | {values['datasets']} | {values['ready']} | "
            f"{values['scenes']} | {values['images']} |"
        )
    return lines


def summarize_datasets(records: list[dict]) -> list[str]:
    lines = [
        "| Task | Dataset | Status | Scenes | Images | Root |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for record in records:
        lines.append(
            f"| {record['task']} | {record['dataset']} | {status_for_record(record)} | "
            f"{record['scene_count']} | {record['total_images']} | "
            f"`{record['root_directory']}` |"
        )
    return lines


def label_summary_for_record(record: dict) -> str:
    labels = set()
    if record["metadata_exists"] or not record.get("metadata_required", True):
        if any(scene["has_images"] for scene in record["scenes"]):
            labels.add("images")
        if any(scene["has_depths"] for scene in record["scenes"]):
            labels.add("depth maps")
        if any(scene["has_sparse"] for scene in record["scenes"]):
            labels.add("sparse COLMAP")

    key = (record["task"], record["dataset"])
    if key == ("MonoDepth", "marigold"):
        labels.add("tar-backed image/depth archives")
    if record["dataset"] == "eth3d":
        labels.add("ETH3D calibration")
    if record["dataset"] == "dtu":
        labels.add("DTU scan geometry where present")
    if record["dataset"] == "tanks_and_temples":
        labels.add("Tanks-and-Temples depth/geometry where present")
    if record["task"] == "SLAM" and record["dataset"] == "7scenes":
        labels.add("per-sequence 7Scenes pose files exported to sparse COLMAP")

    if not labels:
        return "pending"
    return ", ".join(sorted(labels))


def summarize_labels_and_metrics(records: list[dict]) -> list[str]:
    lines = [
        "| Task | Dataset | Available Ground-Truth Labels | Evaluated Metrics |",
        "| --- | --- | --- | --- |",
    ]
    for record in records:
        metrics = DATASET_METRICS.get(
            (record["task"], record["dataset"]),
            TASK_METRICS.get(record["task"], "pending"),
        )
        lines.append(
            f"| {record['task']} | {record['dataset']} | "
            f"{label_summary_for_record(record)} | "
            f"{metrics} |"
        )
    return lines


SCENE_INVENTORY_NOTES = {
    ("SfM", "texturepoorsfm"): (
        "Split v2 (object-level, `scripts/preprocess/assets/texturepoorsfm_split_v2.json`): 30 bags from the "
        "held-out objects 1000, 1003, 1014, 1027, 1030 only. No frame of these objects is in the "
        "TexturePoorSfM training shards."
    ),
}


def scene_inventory(records: list[dict]) -> list[str]:
    lines = []
    for record in records:
        title = f"{record['task']} / {record['dataset']}"
        lines.append(f"#### {title}")
        lines.append("")
        if record["metadata_required"] and not record["metadata_exists"]:
            lines.append(
                f"Metadata not available yet. Expected metadata path: "
                f"`{record['metadata_path']}`."
            )
            lines.append("")
            continue

        note = SCENE_INVENTORY_NOTES.get((record["task"], record["dataset"]))
        if note:
            lines.extend([note, ""])
        lines.extend(
            [
                "| Scene | Images | Images Dir | Depths Dir | Sparse GT |",
                "| --- | ---: | --- | --- | --- |",
            ]
        )
        for scene in record["scenes"]:
            lines.append(
                f"| {scene['scene']} | {scene['total_images']} | "
                f"{format_bool(scene['has_images'])} | "
                f"{format_bool(scene['has_depths'])} | "
                f"{format_bool(scene['has_sparse'])} |"
            )
        lines.append("")
    return lines


def build_inventory_markdown(
    benchmark_root: Path,
    include_scannet: bool = False,
) -> str:
    records = describe_benchmark_scenes(
        benchmark_root=benchmark_root,
        include_scannet=include_scannet,
    )
    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        BEGIN_MARKER,
        "",
        "## Scene Inventory",
        "",
        f"Generated at: `{generated_at}`",
        "",
        f"Benchmark root: `{benchmark_root}`",
        "",
        "Scope: standalone ScanNet is skipped by default because it requires a manual terms-of-use download.",
        "",
        "This section is generated by `scripts/preprocess/export_benchmark_data_doc.py`.",
        "The Overview and Datasets sections above are the curated view (tracks, scene type,",
        "evaluated-vs-total image counts, and license per dataset) built on top of the raw counts below.",
        "",
        "### Task Summary",
        "",
        *summarize_by_task(records),
        "",
        "### Dataset Summary",
        "",
        *summarize_datasets(records),
        "",
        "### Ground-Truth Labels and Metrics",
        "",
        *summarize_labels_and_metrics(records),
        "",
        "### Per-Scene Inventory",
        "",
        *scene_inventory(records),
        END_MARKER,
        "",
    ]
    return "\n".join(lines)


def update_marked_section(existing: str, generated: str) -> str:
    if BEGIN_MARKER in existing and END_MARKER in existing:
        before = existing.split(BEGIN_MARKER, 1)[0].rstrip()
        after = existing.split(END_MARKER, 1)[1].lstrip()
        output = f"{before}\n\n{generated}"
        if after:
            output = f"{output}\n{after}"
        return output
    return f"{existing.rstrip()}\n\n{generated}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark_root",
        type=Path,
        default=DEFAULT_BENCHMARK_ROOT,
        help="Root containing benchmark task folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "docs" / "evaluation.md",
        help="Markdown file to update.",
    )
    parser.add_argument(
        "--include_scannet",
        action="store_true",
        help="Include ScanNet in the generated inventory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generated = build_inventory_markdown(
        benchmark_root=args.benchmark_root.expanduser(),
        include_scannet=args.include_scannet,
    )
    existing = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
    output_path.write_text(update_marked_section(existing, generated), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
