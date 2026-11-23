# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Summarizes benchmark evaluator reports into per-dataset metric tables.

Reads every `eval_*.json` under the given report roots, aggregates the scene
results per (kind, method, task, dataset, SLAM range) and renders them as
Markdown and TSV tables. The evaluator, the PDF report and the dashboard
collector reuse these functions, so all of them weigh metrics the same way.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
from typing import Iterable

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from spatialarena.eval import dataset_registry  # noqa: E402

# pylint: enable=wrong-import-position

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
# The fraction each derived percent comes from (the evaluator writes only the
# first three itself).
DEPTH_PERCENT_SOURCES = dict(
    zip(DEPTH_DERIVED_METRICS,
        ("abs_relative_difference", "delta103_acc", "delta125_acc",
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
# SLAM track (eval_slam.json, spatialarena/eval/slam_trajectory_evaluator.py):
# unweighted means over sequences.
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
# Aggregated from each scene's "runtime" entry (the adapter's metrics.json,
# copied by the evaluators).
RUNTIME_COLUMNS = (
    "runtime_scenes",
    "inference_seconds_mean",
    "seconds_per_image",
    "peak_gpu_memory_mib_mean",
    "peak_gpu_memory_mib_max",
    "runtime_scope",
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class SceneMetrics:
    """The metrics of one evaluated scene.

    Attributes:
        kind: "depth", "mvrecon", "sparse" or "slam".
        method: The method name.
        task: The canonical task name.
        dataset: The dataset name.
        scene: The scene name.
        metrics: The scene's metrics, including derived ones.
        frames: The number of evaluated frames.
        valid_pixels: The number of valid GT pixels.
        metric_units: The depth weighting unit count.
        matched_images: The number of matched images (ATE weight).
        pairs: The number of image pairs (relative pose weight).
        source_path: The `eval_*.json` the scene came from.
        runtime: The adapter's runtime entry, if any.
        variant: The SLAM range track ("short" / "long"; "s<stride>" if
            unlabelled); empty for other tasks.
    """
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
    source_path: pathlib.Path
    runtime: dict | None = None
    variant: str = ""


def infer_method(report_root: pathlib.Path) -> str:
    """Returns the method name a report root is named after."""
    name = report_root.name
    if name:
        return name
    return "unknown"


def _is_scannet_scene(scene: dict) -> bool:
    return (str(scene.get("dataset", "")).lower() == "scannet" or
            str(scene.get("scene", "")).lower() == "scannet")


def _prediction_scene_path_for_root(
    *,
    report_root: pathlib.Path,
    scene: dict,
) -> pathlib.Path | None:
    """Returns the prediction folder of a scene result, if it can be found."""
    prediction_scene_path = scene.get("prediction_scene_path")
    if prediction_scene_path:
        return pathlib.Path(prediction_scene_path)

    method = str(scene.get("method") or "").strip()
    task = dataset_registry.canonical_task_name(str(scene.get("task") or ""))
    dataset = str(scene.get("dataset") or "").strip()
    scene_name = str(scene.get("scene") or scene.get("scene_name") or
                     "").strip()
    if not method or not task or not dataset or not scene_name:
        return None

    eval_root = None
    for parent in report_root.parents:
        if parent.name == "reports":
            eval_root = parent.parent
            break
    if eval_root is None:
        return None

    if task in {"MonoDepth", "VideoDepth", "MVRecon"}:
        suite = "full-depth"
    else:
        suite = "full-pose"
    return (eval_root / "artifacts" / "full_standard" /
            f"benchmark_{suite}_{method}" / task / dataset / method /
            scene_name)


def _scene_kind(path: pathlib.Path, task: str, metrics: dict) -> str | None:
    """Returns the kind of a scene result, or None for an unknown report."""
    if path.name == "eval_slam.json":
        return "slam"
    if task == "MVRecon" or path.name == "eval_mvrecon.json":
        return "mvrecon"
    if "abs_relative_difference" in metrics:
        return "depth"
    if "rotation_error_deg_mean" in metrics or "ate_rmse" in metrics:
        return "sparse"
    return None


def _add_derived_metrics(kind: str, metrics: dict) -> None:
    """Adds the depth percents and the current relative pose metric names.

    Args:
        kind: The scene kind.
        metrics: The scene's metrics, updated in place.
    """
    if kind == "depth":
        for percent_key, fraction_key in DEPTH_PERCENT_SOURCES.items():
            if percent_key not in metrics and fraction_key in metrics:
                metrics[percent_key] = float(metrics[fraction_key]) * 100.0
    # Older reports use the pre-rename metric names.
    renamed = (
        ("relative_rotation_error_deg_mean", "rotation_error_deg_mean"),
        ("relative_rotation_error_deg_median", "rotation_error_deg_median"),
        ("relative_translation_error_deg_mean",
         "translation_direction_error_deg_mean"),
        ("relative_translation_error_deg_median",
         "translation_direction_error_deg_median"),
    )
    for new_name, old_name in renamed:
        if new_name not in metrics and old_name in metrics:
            metrics[new_name] = metrics[old_name]


def _scene_row(scene: dict, kind: str, task: str, metrics: dict,
               path: pathlib.Path, report_root: pathlib.Path) -> SceneMetrics:
    """Builds the `SceneMetrics` of one scene result.

    Args:
        scene: The scene result of the report.
        kind: The scene kind.
        task: The canonical task name.
        metrics: The scene's metrics, with derived ones.
        path: The report file.
        report_root: The report root (names the method if the scene does
            not).

    Returns:
        The scene's metrics.
    """
    if isinstance(scene.get("runtime"), dict):
        runtime = scene.get("runtime")
    else:
        runtime = None
    if kind == "slam":
        variant = scene.get("slam_range") or f"s{scene.get('stride', 1)}"
    else:
        variant = ""
    return SceneMetrics(
        kind=kind,
        method=scene.get("method") or infer_method(report_root),
        task=task,
        dataset=scene["dataset"],
        scene=scene.get("scene") or scene.get("scene_name") or path.parent.name,
        metrics={
            key: float(value) for key, value in metrics.items()
        },
        frames=int(scene.get("num_evaluated_frames", 0)),
        valid_pixels=int(scene.get("num_valid_pixels", 0)),
        metric_units=int(
            scene.get(
                "num_metric_units",
                scene.get("num_evaluated_frames", 0) or 1,
            )),
        matched_images=int(scene.get("num_matched_images", 0)),
        pairs=int(metrics.get("num_pairs", 0)),
        source_path=path,
        runtime=runtime,
        variant=variant,
    )


def collect_scene_metrics(
    report_roots: Iterable[pathlib.Path],
    *,
    benchmark_root: pathlib.Path = dataset_registry.DEFAULT_BENCHMARK_ROOT,
    exclude_scannet: bool = False,
) -> list[SceneMetrics]:
    """Reads the scene results of every `eval_*.json` under the report roots.

    Args:
        report_roots: The report folders to search.
        benchmark_root: The benchmark root (for scene lookups).
        exclude_scannet: Whether to leave out ScanNet scenes.

    Returns:
        One row per scene result of a known kind.
    """
    rows: list[SceneMetrics] = []

    scene_info_cache: dict[tuple[str, str, str], object | None] = {}

    def resolve_scene_info(task: str, dataset: str, scene: str):
        cache_key = (task, dataset, scene)
        if cache_key in scene_info_cache:
            return scene_info_cache[cache_key]
        spec = dataset_registry.find_dataset_spec(task, dataset)
        if spec is None:
            scene_info_cache[cache_key] = None
            return None
        try:
            dataset_obj = spec.create_dataset(benchmark_root)
            scene_info = next((item for item in dataset_obj.list_scenes()
                               if item.scene == scene), None)
        except Exception:  # pylint: disable=broad-exception-caught
            scene_info = None
        scene_info_cache[cache_key] = scene_info
        return scene_info

    for report_root in report_roots:
        report_root = pathlib.Path(report_root)
        for path in sorted(report_root.rglob("eval_*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            for scene in payload.get("scene_results", []):
                if exclude_scannet and _is_scannet_scene(scene):
                    continue
                task = dataset_registry.canonical_task_name(
                    str(scene.get("task") or ""))
                metrics = {
                    key: value
                    for key, value in scene.get("metrics", {}).items()
                    if value is not None
                }
                kind = _scene_kind(path, task, metrics)
                if kind is None:
                    continue
                _add_derived_metrics(kind, metrics)
                rows.append(
                    _scene_row(scene, kind, task, metrics, path, report_root))
    return rows


def deduplicate_scene_metrics(
        rows: Iterable[SceneMetrics]) -> list[SceneMetrics]:
    """Keeps the newest report of each scene.

    Args:
        rows: The scene rows, possibly with several reports per scene.

    Returns:
        One row per scene, sorted by kind, method, task, dataset, variant and
        scene.
    """
    latest: dict[tuple[str, str, str, str, str, str], SceneMetrics] = {}
    for row in rows:
        key = (row.kind, row.method, row.task, row.dataset, row.scene,
               row.variant)
        current = latest.get(key)
        if current is None:
            latest[key] = row
        elif (row.source_path.stat().st_mtime
              > current.source_path.stat().st_mtime):
            latest[key] = row
    return sorted(
        latest.values(),
        key=lambda row:
        (row.kind, row.method, row.task, row.dataset, row.variant, row.scene),
    )


def weighted_average(rows: Iterable[SceneMetrics], metric: str,
                     weight_name: str) -> float | None:
    """Averages a metric over the rows, weighted by a row attribute.

    Args:
        rows: The scene rows.
        metric: The metric name.
        weight_name: The `SceneMetrics` attribute to weigh by.

    Returns:
        The weighted mean, or None if no row has the metric and a weight.
    """
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
    """Aggregates the runtime of the scenes that report one.

    Scenes measured with different scopes (standard `after_model_load` vs
    per-adapter `legacy`) are not comparable, so `runtime_scope` lists every
    scope present and readers should only compare rows that match.

    Args:
        rows: The scene rows.

    Returns:
        The `RUNTIME_COLUMNS`: the mean inference time, the time per input
        image and the peak GPU memory (None where no scene reports it).
    """
    timed = [
        row.runtime
        for row in rows
        if row.runtime and row.runtime.get("inference_seconds") is not None
    ]
    per_image = [
        runtime for runtime in timed if runtime.get("num_input_images")
    ]
    memory = [
        runtime["peak_gpu_memory_mib"]
        for runtime in timed
        if runtime.get("peak_gpu_memory_mib") is not None
    ]
    inference_seconds_mean = None
    if timed:
        inference_seconds_mean = sum(
            runtime["inference_seconds"] for runtime in timed) / len(timed)
    seconds_per_image = None
    if per_image:
        seconds_per_image = (
            sum(runtime["inference_seconds"] for runtime in per_image) /
            sum(runtime["num_input_images"] for runtime in per_image))
    memory_mean = None
    memory_max = None
    if memory:
        memory_mean = sum(memory) / len(memory)
        memory_max = max(memory)
    scopes = sorted({str(runtime.get("runtime_scope")) for runtime in timed})
    return {
        "runtime_scenes": len(timed),
        "inference_seconds_mean": inference_seconds_mean,
        "seconds_per_image": seconds_per_image,
        "peak_gpu_memory_mib_mean": memory_mean,
        "peak_gpu_memory_mib_max": memory_max,
        "runtime_scope": ",".join(scopes) or None,
    }


def aggregate_by_dataset(rows: Iterable[SceneMetrics]) -> list[dict]:
    """Aggregates the scene rows per kind, method, task, dataset and variant.

    Depth and MVRecon metrics are weighted by metric units, ATE by matched
    images and relative pose metrics by image pairs; SLAM metrics are
    unweighted means over sequences, and SLAM rows are ranked
    (`rank_slam`).

    Args:
        rows: The scene rows.

    Returns:
        One summary dict per group, sorted by the group key.
    """
    rows = list(rows)
    groups: dict[tuple[str, str, str, str, str], list[SceneMetrics]] = {}
    for row in rows:
        groups.setdefault(
            (row.kind, row.method, row.task, row.dataset, row.variant),
            []).append(row)
    # Every sequence any method evaluated, per SLAM dataset and stride: a method
    # covering fewer (skipped for its frame budget, or failed) is reported but
    # not ranked.
    slam_sequences: dict[tuple[str, str, str], set[str]] = {}
    for row in rows:
        if row.kind == "slam":
            slam_sequences.setdefault((row.task, row.dataset, row.variant),
                                      set()).add(row.scene)

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
        }
        summary.update(aggregate_runtime(group))
        if kind == "depth":
            for metric in DEPTH_METRICS:
                summary[metric] = weighted_average(group, metric,
                                                   "metric_units")
            for metric in DEPTH_DERIVED_METRICS:
                summary[metric] = weighted_average(group, metric,
                                                   "metric_units")
        elif kind == "slam":
            for metric in SLAM_METRICS:
                values = [
                    row.metrics[metric]
                    for row in group
                    if metric in row.metrics
                ]
                if values:
                    summary[metric] = sum(values) / len(values)
                else:
                    summary[metric] = None
            summary["sequences_total"] = len(slam_sequences[(task, dataset,
                                                             variant)])
        elif kind == "mvrecon":
            for metric in MVRECON_METRICS:
                summary[metric] = weighted_average(group, metric,
                                                   "metric_units")
        else:
            for metric in SPARSE_METRICS:
                if metric.startswith("ate_"):
                    weight_name = "matched_images"
                else:
                    weight_name = "pairs"
                summary[metric] = weighted_average(group, metric, weight_name)
        summaries.append(summary)
    rank_slam(summaries)
    return summaries


def _ranks(values: dict[str, float]) -> dict[str, float]:
    """Ranks lower-is-better values with competition-style average ranks.

    Args:
        values: The value of each method.

    Returns:
        The rank of each method (1 = best; ties share the mean rank).
    """
    ordered = sorted(values.items(), key=lambda item: item[1])
    ranks: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index
        while (end + 1 < len(ordered) and
               ordered[end + 1][1] == ordered[index][1]):
            end += 1
        for position in range(index, end + 1):
            ranks[ordered[position][0]] = (index + end) / 2 + 1
        index = end + 1
    return ranks


def rank_slam(summaries: list[dict]) -> None:
    """Ranks the SLAM summaries in place.

    The score is the mean of the ATE-RMSE rank and the drift rank within each
    (task, dataset, stride), over the methods that evaluated every sequence of
    it. Others get `slam_rank_score = None`.

    Args:
        summaries: The dataset summaries; SLAM rows get `ate_rank`,
            `drift_rank` and `slam_rank_score`.
    """
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for summary in summaries:
        if summary["kind"] == "slam":
            groups.setdefault(
                (summary["task"], summary["dataset"], summary["variant"]),
                []).append(summary)
    for group in groups.values():
        complete = [
            row for row in group if row["scenes"] == row["sequences_total"] and
            row.get("ate_rmse") is not None and row.get("drift_pct") is not None
        ]
        ate_rank = _ranks({row["method"]: row["ate_rmse"] for row in complete})
        drift_rank = _ranks(
            {row["method"]: row["drift_pct"] for row in complete})
        for row in group:
            ranked = row["method"] in ate_rank
            row["ate_rank"] = ate_rank.get(row["method"])
            row["drift_rank"] = drift_rank.get(row["method"])
            if ranked:
                row["slam_rank_score"] = (ate_rank[row["method"]] +
                                          drift_rank[row["method"]]) / 2
            else:
                row["slam_rank_score"] = None


def format_metric(value: object) -> str:
    """Formats a metric with a precision that depends on its magnitude."""
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
    report_roots: list[pathlib.Path],
    summaries: list[dict],
    *,
    exclude_scannet: bool = False,
) -> str:
    """Renders the summaries as Markdown tables, one per kind plus runtime.

    Args:
        report_roots: The report folders (listed as sources).
        summaries: The dataset summaries.
        exclude_scannet: Whether ScanNet was excluded (noted in the text).

    Returns:
        The Markdown text.
    """
    lines = [
        "Source reports:",
        *[f"- `{path}`" for path in report_roots],
        "",
        ("These metrics are computed by the artifact-only evaluators against "
         "canonical ground-truth labels. Depth rows are weighted by each "
         "protocol's metric unit, ATE by matched images, and relative pose "
         "metrics by image pairs."),
        "",
    ]
    if exclude_scannet:
        lines.extend(["Excluded datasets: ScanNet.", ""])
    lines.extend([
        "Depth metrics:",
        "",
        ("| Method | Task | Dataset | Scenes | Frames | Valid Pixels | "
         "AbsRel | SqRel | RMSE | RMSE log | log10 | Delta@1.03 | Delta@1.25 | "
         "Delta@1.25^2 | Delta@1.25^3 | iRMSE | SILog | MAE-normal | "
         "Point Acc. | Point Comp. | Point Chamfer | Point F1 | AbsRel% | "
         "Delta@1.03% | Delta@1.25% |"),
        ("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
         "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
         "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
         "---: | ---: |"),
    ])
    depth_rows = [
        summary for summary in summaries if summary["kind"] == "depth"
    ]
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

    lines.extend([
        "",
        "MVRecon metrics:",
        "",
        ("| Method | Task | Dataset | Scenes | Frames | Valid Pixels | "
         "Accuracy | Completion | Overall | Precision | Recall | F-score |"),
        ("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
         "---: | ---: |"),
    ])
    mvrecon_rows = [
        summary for summary in summaries if summary["kind"] == "mvrecon"
    ]
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

    lines.extend([
        "",
        "Sparse pose metrics:",
        "",
        ("| Method | Task | Dataset | Scenes | Matched Images | Pairs | "
         "ATE RMSE | ATE Mean | ATE Median | Rel Rot Mean | Rel Rot Median | "
         "Rel Trans Mean | Rel Trans Median | Rel Trans Axis Mean | "
         "Rel Trans Axis Median | RPE Mean | RPE Median | RPE Axis Mean | "
         "RPE Axis Median | "
         "RRA@5 | RRA@15 | RRA@30 | RTA@5 | RTA@15 | RTA@30 | mAA@30 | "
         "AUC@1 | AUC@3 | AUC@5 | AUC@10 | AUC@20 | AUC@30 |"),
        "| " + " | ".join(["---", "---", "---", "---:", "---:", "---:"] +
                          ["---:"] * len(SPARSE_METRICS)) + " |",
    ])
    sparse_rows = [
        summary for summary in summaries if summary["kind"] == "sparse"
    ]
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

    lines.extend([
        "",
        ("SLAM trajectory metrics per range track (mean over sequences; ranked "
         "by the mean of the ATE and drift ranks among methods that ran every "
         "sequence of the track):"),
        "",
        ("| Method | Dataset | Range | Sequences | Rank score | ATE RMSE (m) | "
         "Drift % | RPE trans step (m) | RPE rot step (deg) | "
         "RPE trans 1s (m) | RPE rot 1s (deg) | AUC@1 | AUC@3 | AUC@5 | "
         "AUC@10 | AUC@20 | AUC@30 | Completeness |"),
        ("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
         "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"),
    ])
    slam_rows = sorted(
        (row for row in summaries if row["kind"] == "slam"),
        key=lambda row:
        (row["dataset"], row["variant"], row.get("slam_rank_score") is None,
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

    lines.extend([
        "",
        ("Runtime (from each scene's adapter metrics.json; compare only rows "
         "with the same scope):"),
        "",
        ("| Method | Kind | Task | Dataset | Timed Scenes | Inference s (mean) "
         "| s / image | Peak GPU MiB (mean) | Peak GPU MiB (max) | Scope |"),
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
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
                format_metric(row.get(metric)) for metric in (
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
    """Renders the summaries as one TSV table (SLAM metrics as `slam_*`)."""
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
            if (value is None and column.startswith("slam_") and
                    row["kind"] == "slam"):
                value = row.get(column[len("slam_"):])
            if value is None:
                values.append("")
            else:
                values.append(str(value))
        lines.append("\t".join(values))
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    """Parses the command line arguments."""
    parser = argparse.ArgumentParser(
        description="Summarize benchmark eval_depth/eval_sparse reports.")
    parser.add_argument("report_roots", nargs="+", type=pathlib.Path)
    parser.add_argument("--benchmark-root",
                        type=pathlib.Path,
                        default=dataset_registry.DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-markdown", type=pathlib.Path, default=None)
    parser.add_argument("--output-tsv", type=pathlib.Path, default=None)
    parser.add_argument("--exclude-scannet", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Prints or writes the Markdown and TSV summaries."""
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
