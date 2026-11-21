# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Generate per-method or consolidated benchmark PDF reports.

This intentionally reuses ``summarize_benchmark_reports.py`` for collection and
aggregation so PDF reports follow the same weighting protocol as Markdown/TSV
summaries.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.test.summarize_benchmark_reports import (
    DEPTH_METRICS,
    SPARSE_METRICS,
    aggregate_by_dataset,
    collect_scene_metrics,
    format_metric,
)

PRIMARY_DEPTH_METRICS = (
    "abs_relative_difference",
    "rmse_linear",
    "delta125_acc",
    "overall",
    "fscore",
)
PRIMARY_SPARSE_METRICS = ("ate_rmse", "relative_rotation_error_deg_mean", "mAA_30")
LOWER_IS_BETTER = {
    "abs_relative_difference",
    "squared_relative_difference",
    "rmse_linear",
    "rmse_log",
    "log10",
    "i_rmse",
    "silog_rmse",
    "mae_normal_deg",
    "point_accuracy",
    "point_completion",
    "point_chamfer",
    "accuracy",
    "completion",
    "overall",
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
}


def short_path(path: Path) -> str:
    text = str(path)
    return text if len(text) <= 96 else "..." + text[-93:]


def collect_unsupported(report_roots: list[Path]) -> list[dict]:
    unsupported = []
    for root in report_roots:
        summary_path = root / "summary.json"
        if not summary_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not summary.get("unsupported"):
            continue
        unsupported.append(
            {
                "method": summary.get("method", root.name),
                "track": summary.get("benchmark_track", ""),
                "status": summary.get("unsupported_status", "unsupported"),
                "reason": summary.get("unsupported_reason", ""),
                "root": str(root),
            }
        )
    return unsupported


def add_text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.93, title, fontsize=20, weight="bold", ha="left", va="top")
    y = 0.86
    for raw_line in lines:
        wrapped = wrap(raw_line, width=115) or [""]
        for line in wrapped:
            fig.text(0.06, y, line, fontsize=9.5, ha="left", va="top", family="monospace")
            y -= 0.032
            if y < 0.08:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig = plt.figure(figsize=(11, 8.5))
                fig.patch.set_facecolor("white")
                y = 0.92
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_table_page(pdf: PdfPages, title: str, headers: list[str], rows: list[list[str]]) -> None:
    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=15, weight="bold", pad=18)
    if not rows:
        ax.text(0.02, 0.85, "No rows.", fontsize=11, transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return
    max_rows = 28
    for page_start in range(0, len(rows), max_rows):
        if page_start:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(11.7, 8.3))
            fig.patch.set_facecolor("white")
            ax.axis("off")
            ax.set_title(f"{title} (continued)", loc="left", fontsize=15, weight="bold", pad=18)
        page_rows = rows[page_start:page_start + max_rows]
        table = ax.table(cellText=page_rows, colLabels=headers, loc="upper left", cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        table.scale(1.0, 1.2)
        for (row_idx, _col_idx), cell in table.get_celld().items():
            cell.set_edgecolor("#d0d7de")
            if row_idx == 0:
                cell.set_facecolor("#f6f8fa")
                cell.set_text_props(weight="bold")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def rank_rows(summaries: list[dict], metrics: tuple[str, ...]) -> list[list[str]]:
    rows: list[list[str]] = []
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for summary in summaries:
        groups[(summary["kind"], summary["task"], summary["dataset"])].append(summary)
    for (kind, task, dataset), group in sorted(groups.items()):
        for metric in metrics:
            scored = [row for row in group if row.get(metric) is not None]
            if not scored:
                continue
            scored.sort(key=lambda row: row[metric], reverse=metric not in LOWER_IS_BETTER)
            for rank, row in enumerate(scored[:10], 1):
                rows.append([kind, task, dataset, metric, str(rank), row["method"], format_metric(row[metric])])
    return rows


def depth_table_rows(summaries: list[dict]) -> list[list[str]]:
    rows = []
    for row in summaries:
        if row["kind"] != "depth":
            continue
        rows.append([
            row["method"], row["task"], row["dataset"], str(row["scenes"]), str(row["frames"]),
            *[format_metric(row.get(metric)) for metric in DEPTH_METRICS],
        ])
    return rows


def sparse_table_rows(summaries: list[dict]) -> list[list[str]]:
    rows = []
    for row in summaries:
        if row["kind"] != "sparse":
            continue
        rows.append([
            row["method"], row["task"], row["dataset"], str(row["scenes"]), str(row["matched_images"]),
            str(row["pairs"]),
            *[format_metric(row.get(metric)) for metric in SPARSE_METRICS],
        ])
    return rows


def write_pdf(report_roots: list[Path], output: Path, *, title: str, exclude_scannet: bool) -> None:
    rows = collect_scene_metrics(report_roots, exclude_scannet=exclude_scannet)
    summaries = aggregate_by_dataset(rows)
    unsupported = collect_unsupported(report_roots)
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        add_text_page(
            pdf,
            title,
            [
                f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
                f"Scene metric rows: {len(rows)}",
                f"Dataset summary rows: {len(summaries)}",
                f"Excluded ScanNet: {exclude_scannet}",
                "",
                "Source report roots:",
                *[f"- {short_path(path)}" for path in report_roots],
                "",
                "Protocol: depth metrics are weighted by metric units; ATE metrics are weighted by matched images; "
                "relative pose metrics are weighted by image pairs.",
            ],
        )
        if unsupported:
            add_table_page(
                pdf,
                "Unsupported / not runnable methods",
                ["Method", "Track", "Status", "Reason"],
                [[row["method"], row["track"], row["status"], row["reason"]] for row in unsupported],
            )
        add_table_page(
            pdf,
            "Primary rankings",
            ["Kind", "Task", "Dataset", "Metric", "Rank", "Method", "Value"],
            rank_rows(summaries, PRIMARY_DEPTH_METRICS + PRIMARY_SPARSE_METRICS),
        )
        add_table_page(
            pdf,
            "Depth dataset summaries",
            ["Method", "Task", "Dataset", "Scenes", "Frames", *DEPTH_METRICS],
            depth_table_rows(summaries),
        )
        add_table_page(
            pdf,
            "Sparse pose dataset summaries",
            ["Method", "Task", "Dataset", "Scenes", "Images", "Pairs", *SPARSE_METRICS],
            sparse_table_rows(summaries),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="FeedForward Benchmark Report")
    parser.add_argument("--exclude-scannet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_pdf(args.report_roots, args.output, title=args.title, exclude_scannet=args.exclude_scannet)
    print(f"Wrote PDF report to {args.output}")


if __name__ == "__main__":
    main()
