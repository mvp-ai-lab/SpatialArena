# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Summarize benchmark evaluator reports into per-dataset metric tables."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatialarena.eval.dataset_registry import (
    DEFAULT_BENCHMARK_ROOT,
    canonical_task_name,
    find_dataset_spec,
)
DEPTH_METRICS = (
    "abs_relative_difference",
    "squared_relative_difference",
    "rmse_linear",
    "rmse_log",
    "log10",
    "delta103_acc",
    "delta125_acc",
    "delta125_2_acc",
    "delta125_3_acc",
    "i_rmse",
    "silog_rmse",
    "mae_normal_deg",
)
MVRECON_METRICS = (
    "accuracy",
    "completion",
    "overall",
    "precision",
    "recall",
    "fscore",
)
DEPTH_DERIVED_METRICS = (
    "absrel_percent",
    "delta103_percent",
    "delta125_percent",
    "delta125_2_percent",
    "delta125_3_percent",
)
# The fraction each derived percent comes from (the evaluator writes only the first three itself).
DEPTH_PERCENT_SOURCES = dict(zip(DEPTH_DERIVED_METRICS, ("abs_relative_difference", "delta103_acc", "delta125_acc",
                                                         "delta125_2_acc", "delta125_3_acc")))
SPARSE_METRICS = (
    "ate_rmse",
    "ate_mean",
    "ate_median",
    "relative_rotation_error_deg_mean",
    "relative_rotation_error_deg_median",
    "relative_translation_error_deg_mean",
    "relative_translation_error_deg_median",
    "relative_translation_axis_error_deg_mean",
    "relative_translation_axis_error_deg_median",
    "relative_pose_error_deg_mean",
    "relative_pose_error_deg_median",
    "relative_pose_axis_error_deg_mean",
    "relative_pose_axis_error_deg_median",
    "RRA_at_5",
    "RRA_at_15",
    "RRA_at_30",
    "RTA_at_5",
    "RTA_at_15",
    "RTA_at_30",
    "mAA_30",
    "auc_1",
    "auc_3",
    "auc_5",
    "auc_10",
    "auc_20",
    "auc_30",
)
# SLAM track (eval_slam.json, spatialarena/eval/slam_trajectory_evaluator.py): unweighted means over sequences.
SLAM_METRICS = (
    "ate_rmse",
    "drift_pct",
    "rpe_trans_step",
    "rpe_rot_step",
    "rpe_trans_1s",
    "rpe_rot_1s",
    "auc_1",
    "auc_3",
    "auc_5",
    "auc_10",
    "auc_20",
    "auc_30",
    "completeness",
)
# Aggregated from each scene's "runtime" entry (the adapter's metrics.json, copied by the evaluators).
RUNTIME_COLUMNS = (
    "runtime_scenes",
    "inference_seconds_mean",
    "seconds_per_image",
    "peak_gpu_memory_mib_mean",
    "peak_gpu_memory_mib_max",
    "runtime_scope",
)


@dataclass(frozen=True, kw_only=True)
class SceneMetrics:
    kind: str
    method: str
    task: str
    dataset: str
    scene: str
    metrics: dict[str, float]
    frames: int = 0
    valid_pixels: int = 0
    metric_units: int = 0
    matched_images: int = 0
    pairs: int = 0
    source_path: Path
    runtime: dict | None = None
    variant: str = ""  # SLAM range track ("short" / "long"; "s<stride>" if unlabelled); empty for other tasks


def infer_method(report_root: Path) -> str:
    name = report_root.name
    if name:
        return name
    return "unknown"


def _is_scannet_scene(scene: dict) -> bool:
    return (
        str(scene.get("dataset", "")).lower() == "scannet"
        or str(scene.get("scene", "")).lower() == "scannet"
    )


def _prediction_scene_path_for_root(
    *,
    report_root: Path,
    scene: dict,
) -> Path | None:
    prediction_scene_path = scene.get("prediction_scene_path")
    if prediction_scene_path:
        return Path(prediction_scene_path)

    method = str(scene.get("method") or "").strip()
    task = canonical_task_name(str(scene.get("task") or ""))
    dataset = str(scene.get("dataset") or "").strip()
    scene_name = str(scene.get("scene") or scene.get("scene_name") or "").strip()
    if not method or not task or not dataset or not scene_name:
        return None

    eval_root = None
    for parent in report_root.parents:
        if parent.name == "reports":
            eval_root = parent.parent
            break
    if eval_root is None:
        return None

    suite = "full-depth" if task in {"MonoDepth", "VideoDepth", "MVRecon"} else "full-pose"
    return (eval_root / "artifacts" / "full_standard" / f"benchmark_{suite}_{method}" / task / dataset / method
            / scene_name)


def collect_scene_metrics(
    report_roots: Iterable[Path],
    *,
    benchmark_root: Path = DEFAULT_BENCHMARK_ROOT,
    exclude_scannet: bool = False,
) -> list[SceneMetrics]:
    rows: list[SceneMetrics] = []

    scene_info_cache: dict[tuple[str, str, str], object | None] = {}

    def resolve_scene_info(task: str, dataset: str, scene: str):
        cache_key = (task, dataset, scene)
        if cache_key in scene_info_cache:
            return scene_info_cache[cache_key]
        spec = find_dataset_spec(task, dataset)
        if spec is None:
            scene_info_cache[cache_key] = None
            return None
        try:
            dataset_obj = spec.create_dataset(benchmark_root)
            scene_info = next((item for item in dataset_obj.list_scenes() if item.scene == scene), None)
        except Exception:
            scene_info = None
        scene_info_cache[cache_key] = scene_info
        return scene_info

    for report_root in report_roots:
        report_root = Path(report_root)
        for path in sorted(report_root.rglob("eval_*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            for scene in payload.get("scene_results", []):
                if exclude_scannet and _is_scannet_scene(scene):
                    continue
                task = canonical_task_name(str(scene.get("task") or ""))
                metrics = {key: value for key, value in scene.get("metrics", {}).items() if value is not None}
                if path.name == "eval_slam.json":
                    kind = "slam"
                elif task == "MVRecon" or path.name == "eval_mvrecon.json":
                    kind = "mvrecon"
                elif "abs_relative_difference" in metrics:
                    kind = "depth"
                elif "rotation_error_deg_mean" in metrics or "ate_rmse" in metrics:
                    kind = "sparse"
                else:
                    continue
                if kind == "depth":
                    for percent_key, fraction_key in DEPTH_PERCENT_SOURCES.items():
                        if percent_key not in metrics and fraction_key in metrics:
                            metrics[percent_key] = float(metrics[fraction_key]) * 100.0
                if "relative_rotation_error_deg_mean" not in metrics and "rotation_error_deg_mean" in metrics:
                    metrics["relative_rotation_error_deg_mean"] = metrics["rotation_error_deg_mean"]
                if "relative_rotation_error_deg_median" not in metrics and "rotation_error_deg_median" in metrics:
                    metrics["relative_rotation_error_deg_median"] = metrics["rotation_error_deg_median"]
                if (
                    "relative_translation_error_deg_mean" not in metrics
                    and "translation_direction_error_deg_mean" in metrics
                ):
                    metrics["relative_translation_error_deg_mean"] = metrics[
                        "translation_direction_error_deg_mean"
                    ]
                if (
                    "relative_translation_error_deg_median" not in metrics
                    and "translation_direction_error_deg_median" in metrics
                ):
                    metrics["relative_translation_error_deg_median"] = metrics[
                        "translation_direction_error_deg_median"
                    ]
                rows.append(
                    SceneMetrics(
                        kind=kind,
                        method=scene.get("method") or infer_method(report_root),
                        task=task,
                        dataset=scene["dataset"],
                        scene=scene.get("scene") or scene.get("scene_name") or path.parent.name,
                        metrics={key: float(value) for key, value in metrics.items()},
                        frames=int(scene.get("num_evaluated_frames", 0)),
                        valid_pixels=int(scene.get("num_valid_pixels", 0)),
                        metric_units=int(
                            scene.get(
                                "num_metric_units",
                                scene.get("num_evaluated_frames", 0) or 1,
                            )
                        ),
                        matched_images=int(scene.get("num_matched_images", 0)),
                        pairs=int(metrics.get("num_pairs", 0)),
                        source_path=path,
                        runtime=scene.get("runtime") if isinstance(scene.get("runtime"), dict) else None,
                        variant=(
                            (scene.get("slam_range") or f"s{scene.get('stride', 1)}") if kind == "slam" else ""
                        ),
                    )
                )
    return rows


def deduplicate_scene_metrics(rows: Iterable[SceneMetrics]) -> list[SceneMetrics]:
    latest: dict[tuple[str, str, str, str, str, str], SceneMetrics] = {}
    for row in rows:
        key = (row.kind, row.method, row.task, row.dataset, row.scene, row.variant)
        current = latest.get(key)
        if current is None or row.source_path.stat().st_mtime > current.source_path.stat().st_mtime:
            latest[key] = row
    return sorted(
        latest.values(),
        key=lambda row: (row.kind, row.method, row.task, row.dataset, row.variant, row.scene),
    )


def weighted_average(rows: Iterable[SceneMetrics], metric: str, weight_name: str) -> float | None:
    weighted_sum = 0.0
    total_weight = 0
    for row in rows:
        if metric not in row.metrics:
            continue
        weight = int(getattr(row, weight_name))
        if weight <= 0:
            continue
        weighted_sum += row.metrics[metric] * weight
        total_weight += weight
    if total_weight <= 0:
        return None
    return weighted_sum / total_weight


def aggregate_runtime(rows: Iterable[SceneMetrics]) -> dict:
    """Mean inference time, time per input image and peak GPU memory over the scenes that report them.

    Scenes measured with different scopes (standard ``after_model_load`` vs per-adapter ``legacy``) are not
    comparable, so ``runtime_scope`` lists every scope present and readers should only compare rows that match.
    """
    timed = [row.runtime for row in rows if row.runtime and row.runtime.get("inference_seconds") is not None]
    per_image = [r for r in timed if r.get("num_input_images")]
    memory = [r["peak_gpu_memory_mib"] for r in timed if r.get("peak_gpu_memory_mib") is not None]
    return {
        "runtime_scenes": len(timed),
        "inference_seconds_mean": sum(r["inference_seconds"] for r in timed) / len(timed) if timed else None,
        "seconds_per_image": (
            sum(r["inference_seconds"] for r in per_image) / sum(r["num_input_images"] for r in per_image)
            if per_image
            else None
        ),
        "peak_gpu_memory_mib_mean": sum(memory) / len(memory) if memory else None,
        "peak_gpu_memory_mib_max": max(memory) if memory else None,
        "runtime_scope": ",".join(sorted({str(r.get("runtime_scope")) for r in timed})) or None,
    }


def aggregate_by_dataset(rows: Iterable[SceneMetrics]) -> list[dict]:
    rows = list(rows)
    groups: dict[tuple[str, str, str, str, str], list[SceneMetrics]] = {}
    for row in rows:
        groups.setdefault((row.kind, row.method, row.task, row.dataset, row.variant), []).append(row)
    # Every sequence any method evaluated, per SLAM dataset and stride: a method covering fewer (skipped for its
    # frame budget, or failed) is reported but not ranked.
    slam_sequences: dict[tuple[str, str, str], set[str]] = {}
    for row in rows:
        if row.kind == "slam":
            slam_sequences.setdefault((row.task, row.dataset, row.variant), set()).add(row.scene)

    summaries: list[dict] = []
    for (kind, method, task, dataset, variant), group in sorted(groups.items()):
        summary = {
            "kind": kind,
            "method": method,
            "task": task,
            "dataset": dataset,
            "variant": variant,
            "scenes": len(group),
            "frames": sum(row.frames for row in group),
            "valid_pixels": sum(row.valid_pixels for row in group),
            "metric_units": sum(row.metric_units for row in group),
            "matched_images": sum(row.matched_images for row in group),
            "pairs": sum(row.pairs for row in group),
            **aggregate_runtime(group),
        }
        if kind == "depth":
            for metric in DEPTH_METRICS:
                summary[metric] = weighted_average(group, metric, "metric_units")
            for metric in DEPTH_DERIVED_METRICS:
                summary[metric] = weighted_average(group, metric, "metric_units")
        elif kind == "slam":
            for metric in SLAM_METRICS:
                values = [row.metrics[metric] for row in group if metric in row.metrics]
                summary[metric] = sum(values) / len(values) if values else None
            summary["sequences_total"] = len(slam_sequences[(task, dataset, variant)])
        elif kind == "mvrecon":
            for metric in MVRECON_METRICS:
                summary[metric] = weighted_average(group, metric, "metric_units")
        else:
            for metric in SPARSE_METRICS:
                weight_name = "matched_images" if metric.startswith("ate_") else "pairs"
                summary[metric] = weighted_average(group, metric, weight_name)
        summaries.append(summary)
    rank_slam(summaries)
    return summaries


def _ranks(values: dict[str, float]) -> dict[str, float]:
    """Competition-style average ranks (1 = best, ties share the mean rank) of lower-is-better values."""
    ordered = sorted(values.items(), key=lambda item: item[1])
    ranks: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index
        while end + 1 < len(ordered) and ordered[end + 1][1] == ordered[index][1]:
            end += 1
        for position in range(index, end + 1):
            ranks[ordered[position][0]] = (index + end) / 2 + 1
        index = end + 1
    return ranks


def rank_slam(summaries: list[dict]) -> None:
    """SLAM ranking: mean of the ATE-RMSE rank and the drift rank within each (task, dataset, stride), over the
    methods that evaluated every sequence of it. Others get ``slam_rank_score = None``."""
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for summary in summaries:
        if summary["kind"] == "slam":
            groups.setdefault((summary["task"], summary["dataset"], summary["variant"]), []).append(summary)
    for group in groups.values():
        complete = [
            row for row in group
            if row["scenes"] == row["sequences_total"] and row.get("ate_rmse") is not None
            and row.get("drift_pct") is not None
        ]
        ate_rank = _ranks({row["method"]: row["ate_rmse"] for row in complete})
        drift_rank = _ranks({row["method"]: row["drift_pct"] for row in complete})
        for row in group:
            ranked = row["method"] in ate_rank
            row["ate_rank"] = ate_rank.get(row["method"])
            row["drift_rank"] = drift_rank.get(row["method"])
            row["slam_rank_score"] = (ate_rank[row["method"]] + drift_rank[row["method"]]) / 2 if ranked else None


def format_metric(value: object) -> str:
    if value is None:
        return "-"
    number = float(value)
    if abs(number) >= 1e6 or (0 < abs(number) < 1e-4):
        return f"{number:.4e}"
    if abs(number) >= 100:
        return f"{number:.3f}"
    if abs(number) >= 10:
        return f"{number:.4f}"
    if abs(number) >= 1:
        return f"{number:.5f}"
    return f"{number:.6f}"


def render_markdown(
    report_roots: list[Path],
    summaries: list[dict],
    *,
    exclude_scannet: bool = False,
) -> str:
    lines = [
        "Source reports:",
        *[f"- `{path}`" for path in report_roots],
        "",
        (
            "These metrics are computed by the artifact-only evaluators against "
            "canonical ground-truth labels. Depth rows are weighted by each "
            "protocol's metric unit, ATE by matched images, and relative pose "
            "metrics by image pairs."
        ),
        "",
    ]
    if exclude_scannet:
        lines.extend(["Excluded datasets: ScanNet.", ""])
    lines.extend(
        [
            "Depth metrics:",
            "",
            (
                "| Method | Task | Dataset | Scenes | Frames | Valid Pixels | AbsRel | "
                "SqRel | RMSE | RMSE log | log10 | Delta@1.03 | Delta@1.25 | "
                "Delta@1.25^2 | Delta@1.25^3 | iRMSE | SILog | MAE-normal | "
                "Point Acc. | Point Comp. | Point Chamfer | Point F1 | AbsRel% | "
                "Delta@1.03% | Delta@1.25% |"
            ),
            (
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
                "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
                "---: | ---: | ---: | ---: | ---: | ---: | ---: | "
                "---: | ---: |"
            ),
        ]
    )
    depth_rows = [summary for summary in summaries if summary["kind"] == "depth"]
    for row in depth_rows:
        values = [
            row["method"],
            row["task"],
            row["dataset"],
            str(row["scenes"]),
            str(row["frames"]),
            str(row["valid_pixels"]),
            *[format_metric(row.get(metric)) for metric in DEPTH_METRICS],
        ]
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "MVRecon metrics:",
            "",
            "| Method | Task | Dataset | Scenes | Frames | Valid Pixels | Accuracy | Completion | Overall | Precision "
            "| Recall | F-score |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    mvrecon_rows = [summary for summary in summaries if summary["kind"] == "mvrecon"]
    for row in mvrecon_rows:
        values = [
            row["method"],
            row["task"],
            row["dataset"],
            str(row["scenes"]),
            str(row["frames"]),
            str(row["valid_pixels"]),
            *[format_metric(row.get(metric)) for metric in MVRECON_METRICS],
        ]
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "Sparse pose metrics:",
            "",
            (
                "| Method | Task | Dataset | Scenes | Matched Images | Pairs | ATE RMSE | "
                "ATE Mean | ATE Median | Rel Rot Mean | Rel Rot Median | "
                "Rel Trans Mean | Rel Trans Median | Rel Trans Axis Mean | "
                "Rel Trans Axis Median | RPE Mean | RPE Median | RPE Axis Mean | "
                "RPE Axis Median | "
                "RRA@5 | RRA@15 | RRA@30 | RTA@5 | RTA@15 | RTA@30 | mAA@30 | "
                "AUC@1 | AUC@3 | AUC@5 | AUC@10 | AUC@20 | AUC@30 |"
            ),
            "| " + " | ".join(["---", "---", "---", "---:", "---:", "---:"] + ["---:"] * len(SPARSE_METRICS)) + " |",
        ]
    )
    sparse_rows = [summary for summary in summaries if summary["kind"] == "sparse"]
    for row in sparse_rows:
        values = [
            row["method"],
            row["task"],
            row["dataset"],
            str(row["scenes"]),
            str(row["matched_images"]),
            str(row["pairs"]),
            *[format_metric(row.get(metric)) for metric in SPARSE_METRICS],
        ]
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "SLAM trajectory metrics per range track (mean over sequences; ranked by the mean of the ATE and drift "
            "ranks among methods that ran every sequence of the track):",
            "",
            "| Method | Dataset | Range | Sequences | Rank score | ATE RMSE (m) | Drift % | RPE trans step (m) | "
            "RPE rot step (deg) | RPE trans 1s (m) | RPE rot 1s (deg) | AUC@1 | AUC@3 | AUC@5 | AUC@10 | AUC@20 | "
            "AUC@30 | Completeness |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
            "| ---: | ---: | ---: |",
        ]
    )
    slam_rows = sorted(
        (row for row in summaries if row["kind"] == "slam"),
        key=lambda row: (row["dataset"], row["variant"], row.get("slam_rank_score") is None,
                         row.get("slam_rank_score") or 0.0, row["method"]),
    )
    for row in slam_rows:
        values = [
            row["method"],
            row["dataset"],
            row["variant"] or "-",
            f"{row['scenes']}/{row['sequences_total']}",
            format_metric(row.get("slam_rank_score")),
            *[format_metric(row.get(metric)) for metric in SLAM_METRICS],
        ]
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "Runtime (from each scene's adapter metrics.json; compare only rows with the same scope):",
            "",
            "| Method | Kind | Task | Dataset | Timed Scenes | Inference s (mean) | s / image | "
            "Peak GPU MiB (mean) | Peak GPU MiB (max) | Scope |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summaries:
        if not row.get("runtime_scenes"):
            continue
        values = [
            row["method"],
            row["kind"],
            row["task"],
            row["dataset"],
            str(row["runtime_scenes"]),
            *[
                format_metric(row.get(metric))
                for metric in (
                    "inference_seconds_mean",
                    "seconds_per_image",
                    "peak_gpu_memory_mib_mean",
                    "peak_gpu_memory_mib_max",
                )
            ],
            str(row.get("runtime_scope") or "-"),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def render_tsv(summaries: list[dict]) -> str:
    columns = [
        "kind",
        "method",
        "task",
        "dataset",
        "variant",
        "scenes",
        "frames",
        "valid_pixels",
        "matched_images",
        "pairs",
        *DEPTH_METRICS,
        *DEPTH_DERIVED_METRICS,
        *MVRECON_METRICS,
        *SPARSE_METRICS,
        *[f"slam_{metric}" for metric in SLAM_METRICS],
        "sequences_total",
        "ate_rank",
        "drift_rank",
        "slam_rank_score",
        *RUNTIME_COLUMNS,
    ]
    lines = ["\t".join(columns)]
    for row in summaries:
        values = []
        for column in columns:
            value = row.get(column)
            if value is None and column.startswith("slam_") and row["kind"] == "slam":
                value = row.get(column[len("slam_"):])
            values.append("" if value is None else str(value))
        lines.append("\t".join(values))
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize benchmark eval_depth/eval_sparse reports."
    )
    parser.add_argument("report_roots", nargs="+", type=Path)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-markdown", type=Path, default=None)
    parser.add_argument("--output-tsv", type=Path, default=None)
    parser.add_argument("--exclude-scannet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_scene_metrics(
        args.report_roots,
        benchmark_root=args.benchmark_root,
        exclude_scannet=args.exclude_scannet,
    )
    rows = deduplicate_scene_metrics(rows)
    summaries = aggregate_by_dataset(rows)
    markdown = render_markdown(
        args.report_roots,
        summaries,
        exclude_scannet=args.exclude_scannet,
    )
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown, encoding="utf-8")
    else:
        print(markdown, end="")
    if args.output_tsv:
        args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
        args.output_tsv.write_text(render_tsv(summaries), encoding="utf-8")


if __name__ == "__main__":
    main()
