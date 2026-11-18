#!/usr/bin/env python3
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Export full-standard benchmark summaries for the benchmark dashboard UI."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatialarena.eval.dataset_registry import DEFAULT_BENCHMARK_ROOT, find_dataset_spec
from spatialarena.eval.depth_prediction_evaluator import DepthEvaluationConfig, evaluate_depth_scene
from spatialarena.eval.mvrecon_evaluator import MVReconEvaluationConfig, evaluate_mvrecon_scene
from spatialarena.task_track_matrix import on_matrix


DEFAULT_OUTPUT = REPO_ROOT / "app" / "webpage" / "data" / "results.json"

TASK_ALIASES = {
    "RMVD": "MVRecon",
    "CameraPose": "SfM",
}
TASK_ORDER = ["MonoDepth", "VideoDepth", "MVRecon", "SfM", "SLAM", "WeakTextureSfM"]
LEGACY_METHOD_REPLACEMENTS = {
    "DA3": "DA3-G",
}
TASK_DISPLAY_NAMES = {
    "MonoDepth": "MonoDepth",
    "VideoDepth": "Video Depth",
    "MVRecon": "MVRecon",
    "SfM": "SfM",
    "WeakTextureSfM": "Weak-Texture SfM",
}
DATASET_DISPLAY_NAMES = {
    "7scenes": "7Scenes",
    "aerialmetric": "AerialMetric",
    "bonn": "Bonn",
    "diode": "DIODE",
    "dtu": "DTU",
    "eth3d": "ETH3D",
    "hammer": "HAMMER",
    "ibims1": "iBims-1",
    "imc": "IMC",
    "kitti": "KITTI",
    "marigold": "Marigold",
    "nyuv2": "NYUv2",
    "nrgbd": "NRGBD",
    "scannetpp": "ScanNet++",
    "sintel": "Sintel",
    "spring": "Spring",
    "synth4k": "Synth4K",
    "tanks_and_temples": "Tanks and Temples",
    "hiroom": "HiRoom",
    "texturepoorsfm": "Weak-Texture SfM",
    "tum_dynamics": "TUM Dynamics",
    "oxford_spires": "Oxford Spires",
    "vbr": "VBR",
    "kitti_odometry": "KITTI Odometry",
    "spatialarena": "SpatialArena",
    "dycheck": "DyCheck",
}
# SLAM results are kept per range track (short / long / large stride): each (dataset, range) pair is its own table
# row group, keyed "<dataset>@<range>", so the ranges are never averaged together.
SLAM_RANGE_SEPARATOR = "@"

TASK_CATALOG = {
    "MonoDepth": {
        "label": "Monocular Depth",
        "group": "Depth",
        "description": "Single-image dense depth prediction.",
    },
    "VideoDepth": {
        "label": "Video Depth",
        "group": "Depth",
        "description": "Dense depth prediction on temporally ordered image sequences.",
    },
    "MVRecon": {
        "label": "Multi-View Reconstruction",
        "group": "Geometry",
        "description": "Reconstruction-only evaluation over shared multi-view inputs.",
    },
    "SfM": {
        "label": "Structure from Motion",
        "group": "Pose",
        "description": "Sparse pose and relative camera-geometry evaluation.",
    },
    "SLAM": {
        "label": "SLAM",
        "group": "Pose",
        "description": "Sequential sparse pose and relative camera-geometry evaluation.",
    },
    "WeakTextureSfM": {
        "label": "Weak-Texture SfM",
        "group": "Pose",
        "description": "Sparse pose evaluation on the DetectorFreeSfM texture-poor benchmark dataset.",
    },
}

# Column order of the depth tables: AbsRel, the delta accuracies, then the other errors.
DEPTH_METRICS = [
    {
        "key": "absrel_percent",
        "label": "AbsRel%",
        "family": "depth_error",
        "direction": "lower",
        "kind": "depth",
        "precision": 2,
    },
    {
        "key": "delta103_percent",
        "label": "Delta@1.03%",
        "family": "depth_accuracy",
        "direction": "higher",
        "kind": "depth",
        "precision": 2,
    },
    {
        "key": "delta125_percent",
        "label": "Delta@1.25%",
        "family": "depth_accuracy",
        "direction": "higher",
        "kind": "depth",
        "precision": 2,
    },
    {
        "key": "delta125_2_percent",
        "label": "Delta@1.25^2%",
        "family": "depth_accuracy",
        "direction": "higher",
        "kind": "depth",
        "precision": 2,
    },
    {
        "key": "delta125_3_percent",
        "label": "Delta@1.25^3%",
        "family": "depth_accuracy",
        "direction": "higher",
        "kind": "depth",
        "precision": 2,
    },
    {
        "key": "squared_relative_difference",
        "label": "SqRel",
        "family": "depth_error",
        "direction": "lower",
        "kind": "depth",
        "precision": 3,
    },
    {
        "key": "rmse_linear",
        "label": "RMSE",
        "family": "depth_error",
        "direction": "lower",
        "kind": "depth",
        "precision": 3,
    },
    {
        "key": "rmse_log",
        "label": "RMSE-log",
        "family": "depth_error",
        "direction": "lower",
        "kind": "depth",
        "precision": 3,
    },
    {
        "key": "log10",
        "label": "log10",
        "family": "depth_error",
        "direction": "lower",
        "kind": "depth",
        "precision": 3,
    },
    {
        "key": "i_rmse",
        "label": "iRMSE",
        "family": "depth_error",
        "direction": "lower",
        "kind": "depth",
        "precision": 3,
    },
    {
        "key": "silog_rmse",
        "label": "SILog",
        "family": "depth_error",
        "direction": "lower",
        "kind": "depth",
        "precision": 2,
    },
    {
        "key": "mae_normal_deg",
        "label": "MAE-normal",
        "family": "depth_geometry",
        "direction": "lower",
        "kind": "depth",
        "precision": 2,
    },
]

MVRECON_METRICS = [
    {
        "key": "accuracy",
        "label": "Accuracy",
        "family": "reconstruction",
        "direction": "lower",
        "kind": "mvrecon",
        "precision": 4,
    },
    {
        "key": "completion",
        "label": "Completion",
        "family": "reconstruction",
        "direction": "lower",
        "kind": "mvrecon",
        "precision": 4,
    },
    {
        "key": "overall",
        "label": "Overall",
        "family": "reconstruction",
        "direction": "lower",
        "kind": "mvrecon",
        "precision": 4,
    },
    {
        "key": "precision",
        "label": "Precision",
        "family": "reconstruction",
        "direction": "higher",
        "kind": "mvrecon",
        "precision": 3,
    },
    {
        "key": "recall",
        "label": "Recall",
        "family": "reconstruction",
        "direction": "higher",
        "kind": "mvrecon",
        "precision": 3,
    },
    {
        "key": "fscore",
        "label": "F-score",
        "family": "reconstruction",
        "direction": "higher",
        "kind": "mvrecon",
        "precision": 3,
    },
]

POSE_METRICS = [
    {
        "key": "ate_rmse",
        "label": "ATE RMSE",
        "family": "pose_trajectory",
        "direction": "lower",
        "kind": "sparse",
        "precision": 3,
    },
    {
        "key": "ate_mean",
        "label": "ATE Mean",
        "family": "pose_trajectory",
        "direction": "lower",
        "kind": "sparse",
        "precision": 3,
    },
    {
        "key": "ate_median",
        "label": "ATE Median",
        "family": "pose_trajectory",
        "direction": "lower",
        "kind": "sparse",
        "precision": 3,
    },
    {
        "key": "relative_rotation_error_deg_mean",
        "label": "RRE Mean",
        "family": "pose_error",
        "direction": "lower",
        "kind": "sparse",
        "precision": 2,
    },
    {
        "key": "relative_rotation_error_deg_median",
        "label": "RRE Median",
        "family": "pose_error",
        "direction": "lower",
        "kind": "sparse",
        "precision": 2,
    },
    {
        "key": "relative_translation_error_deg_mean",
        "label": "RTE Mean",
        "family": "pose_error",
        "direction": "lower",
        "kind": "sparse",
        "precision": 2,
    },
    {
        "key": "relative_translation_error_deg_median",
        "label": "RTE Median",
        "family": "pose_error",
        "direction": "lower",
        "kind": "sparse",
        "precision": 2,
    },
    {
        "key": "relative_translation_axis_error_deg_mean",
        "label": "RTAxis Mean",
        "family": "pose_error",
        "direction": "lower",
        "kind": "sparse",
        "precision": 2,
    },
    {
        "key": "relative_translation_axis_error_deg_median",
        "label": "RTAxis Median",
        "family": "pose_error",
        "direction": "lower",
        "kind": "sparse",
        "precision": 2,
    },
    {
        "key": "relative_pose_error_deg_mean",
        "label": "RPE Mean",
        "family": "pose_error",
        "direction": "lower",
        "kind": "sparse",
        "precision": 2,
    },
    {
        "key": "relative_pose_axis_error_deg_mean",
        "label": "RPE-Axis Mean",
        "family": "pose_error",
        "direction": "lower",
        "kind": "sparse",
        "precision": 2,
    },
    {
        "key": "RRA_at_5",
        "label": "RRA@5",
        "family": "pose_accuracy",
        "direction": "higher",
        "kind": "sparse",
        "precision": 3,
    },
    {
        "key": "RRA_at_15",
        "label": "RRA@15",
        "family": "pose_accuracy",
        "direction": "higher",
        "kind": "sparse",
        "precision": 3,
    },
    {
        "key": "RRA_at_30",
        "label": "RRA@30",
        "family": "pose_accuracy",
        "direction": "higher",
        "kind": "sparse",
        "precision": 3,
    },
    {
        "key": "RTA_at_5",
        "label": "RTA@5",
        "family": "pose_accuracy",
        "direction": "higher",
        "kind": "sparse",
        "precision": 3,
    },
    {
        "key": "RTA_at_15",
        "label": "RTA@15",
        "family": "pose_accuracy",
        "direction": "higher",
        "kind": "sparse",
        "precision": 3,
    },
    {
        "key": "RTA_at_30",
        "label": "RTA@30",
        "family": "pose_accuracy",
        "direction": "higher",
        "kind": "sparse",
        "precision": 3,
    },
    {
        "key": "mAA_30",
        "label": "mAA@30",
        "family": "pose_accuracy",
        "direction": "higher",
        "kind": "sparse",
        "precision": 3,
    },
    {
        "key": "auc_1",
        "label": "AUC@1",
        "family": "pose_auc",
        "direction": "higher",
        "kind": "sparse",
        "precision": 3,
    },
    {
        "key": "auc_3",
        "label": "AUC@3",
        "family": "pose_auc",
        "direction": "higher",
        "kind": "sparse",
        "precision": 3,
    },
    {
        "key": "auc_5",
        "label": "AUC@5",
        "family": "pose_auc",
        "direction": "higher",
        "kind": "sparse",
        "precision": 3,
    },
    {
        "key": "auc_10",
        "label": "AUC@10",
        "family": "pose_auc",
        "direction": "higher",
        "kind": "sparse",
        "precision": 3,
    },
    {
        "key": "auc_20",
        "label": "AUC@20",
        "family": "pose_auc",
        "direction": "higher",
        "kind": "sparse",
        "precision": 3,
    },
    {
        "key": "auc_30",
        "label": "AUC@30",
        "family": "pose_auc",
        "direction": "higher",
        "kind": "sparse",
        "precision": 3,
    },
]

# SLAM trajectory metrics (docs/benchmark/slam_protocol.md "Metrics"); ATE is in metres of the GT frame, so values
# are only comparable within one dataset@range cell, never averaged across datasets.
SLAM_METRICS = [
    {"key": "slam_ate_rmse", "label": "ATE RMSE (m)", "family": "slam_trajectory", "direction": "lower",
     "kind": "slam", "precision": 3},
    {"key": "slam_drift_pct", "label": "Drift %", "family": "slam_trajectory", "direction": "lower",
     "kind": "slam", "precision": 2},
    {"key": "slam_rpe_trans_1s", "label": "RPE-t 1s (m)", "family": "slam_rpe", "direction": "lower",
     "kind": "slam", "precision": 3},
    {"key": "slam_rpe_rot_1s", "label": "RPE-r 1s (deg)", "family": "slam_rpe", "direction": "lower",
     "kind": "slam", "precision": 3},
    {"key": "slam_rpe_trans_step", "label": "RPE-t step (m)", "family": "slam_rpe", "direction": "lower",
     "kind": "slam", "precision": 4},
    {"key": "slam_rpe_rot_step", "label": "RPE-r step (deg)", "family": "slam_rpe", "direction": "lower",
     "kind": "slam", "precision": 3},
    {"key": "slam_auc_1", "label": "AUC@1", "family": "slam_pose_auc", "direction": "higher",
     "kind": "slam", "precision": 3},
    {"key": "slam_auc_3", "label": "AUC@3", "family": "slam_pose_auc", "direction": "higher",
     "kind": "slam", "precision": 3},
    {"key": "slam_auc_5", "label": "AUC@5", "family": "slam_pose_auc", "direction": "higher",
     "kind": "slam", "precision": 3},
    {"key": "slam_auc_10", "label": "AUC@10", "family": "slam_pose_auc", "direction": "higher",
     "kind": "slam", "precision": 3},
    {"key": "slam_auc_20", "label": "AUC@20", "family": "slam_pose_auc", "direction": "higher",
     "kind": "slam", "precision": 3},
    {"key": "slam_auc_30", "label": "AUC@30", "family": "slam_pose_auc", "direction": "higher",
     "kind": "slam", "precision": 3},
    {"key": "slam_completeness", "label": "Completeness", "family": "slam_coverage", "direction": "higher",
     "kind": "slam", "precision": 3},
]
# The protocol ranks a SLAM cell by ATE RMSE and drift; the dashboard score normalizes the same two metrics.
SLAM_SCORE_METRICS = ("slam_ate_rmse", "slam_drift_pct")
# Columns of the per-range overview table (one group per dataset); the step RPEs stay in the per-dataset tables.
SLAM_RANGE_TABLE_METRICS = ("slam_ate_rmse", "slam_drift_pct", "slam_rpe_trans_1s", "slam_rpe_rot_1s",
                            "slam_auc_30", "slam_completeness")
SLAM_RANGE_ORDER = ("short", "long", "large")
# Range-table column key: "<metric>::<dataset>".
SLAM_COLUMN_SEPARATOR = "::"
# Reader-facing labels of recorded case failures (read_case_failures); a failed case scores 0 like a missing one.
CASE_FAILURE_LABELS = {
    "gpu_oom_recorded": "GPU OOM",
    "gpu_oom": "GPU OOM",
    "oom_job_affected": "RAM OOM",
    "cpu_memory_oom": "RAM OOM",
    "host_oom": "RAM OOM",
    # The FFB release's name for a job killed with exit 137 and no OOM line in its log: a host-RAM OOM.
    "killed_unknown": "RAM OOM",
    "insufficient_predicted_poses": "Too few poses",
    "blocked_missing_depth_prior": "No depth prior",
    # The release's GenCeption-Pose records: the sequence has more frames than the model accepts (81), not an OOM.
    "unsupported_sequence_length": "Too many frames",
    # The run's cache disk filled up: an infrastructure failure, not the method's.
    "cache_disk_capacity": "Disk full",
    "error_or_empty": "Failed",
    "error": "Failed",
}

METRICS = DEPTH_METRICS + MVRECON_METRICS + POSE_METRICS + SLAM_METRICS
METRIC_BY_KEY = {metric["key"]: metric for metric in METRICS}
METRIC_KEYS = [metric["key"] for metric in METRICS]

# Datasets kept in the registry and in every per-dataset/task table (still
# fully evaluated and visible as a diagnostic), but excluded from the
# cross-dataset ranking/leaderboard score -- see
# docs/benchmark/data_leakage_notes.md. As of this writing, a majority of
# the benchmarked methods disclose training on ScanNet++ (some with no
# stated train/test split at all), so including it in an aggregate score
# would systematically flatter those methods over ones that never saw it.
# Spring (MonoDepth) is excluded for the same reason: its evaluation frames
# come from the Spring train split, which DA3, MapAnything, CUT3R, OmniVGGT
# and Deja View train on.
# Datasets shown in the tables but left out of the leaderboard scores. SpatialArena: its evaluation is still running
# and the release has none of it, so only a few local runs have results (every other method would score 0 on it).
LEADERBOARD_EXCLUDED_DATASETS = {"scannetpp", "spring", "spatialarena"}

# Percent forms of the depth evaluator's fractions (it writes the percent form only for AbsRel, delta 1.03 and 1.25).
DEPTH_PERCENT_SOURCES = {
    "absrel_percent": "abs_relative_difference",
    "delta103_percent": "delta103_acc",
    "delta125_percent": "delta125_acc",
    "delta125_2_percent": "delta125_2_acc",
    "delta125_3_percent": "delta125_3_acc",
}
# The depth tracks' scores use these metrics only (MAE-normal where the dataset has it); the others are reference.
DEPTH_SCORE_METRICS = ("absrel_percent", "delta103_percent", "delta125_percent", "delta125_2_percent",
                       "delta125_3_percent", "mae_normal_deg")

DEPTH_TASKS = {"MonoDepth", "VideoDepth"}
MVRECON_TASKS = {"MVRecon"}
POSE_TASKS = {"SfM", "WeakTextureSfM"}
SLAM_TASKS = {"SLAM"}


def metric_keys_for_task(task: str) -> list[str]:
    task = canonical_task_name(task)
    if task in DEPTH_TASKS:
        metrics = DEPTH_METRICS
    elif task in MVRECON_TASKS:
        metrics = MVRECON_METRICS
    elif task in POSE_TASKS:
        metrics = POSE_METRICS
    elif task in SLAM_TASKS:
        metrics = SLAM_METRICS
    else:
        metrics = METRICS
    return [metric["key"] for metric in metrics]


def kind_for_task(task: str, default: str = "") -> str:
    task = canonical_task_name(task)
    if task in DEPTH_TASKS:
        return "depth"
    if task in MVRECON_TASKS:
        return "mvrecon"
    if task in POSE_TASKS:
        return "sparse"
    if task in SLAM_TASKS:
        return "slam"
    return default

COMPACT_AXES = [
    {
        "key": "depth_error",
        "label": "Depth Error",
        "metric_keys": [
            "absrel_percent",
            "squared_relative_difference",
            "rmse_linear",
            "rmse_log",
            "log10",
            "i_rmse",
            "silog_rmse",
        ],
    },
    {
        "key": "depth_geometry",
        "label": "Depth Geometry",
        "metric_keys": [
            "mae_normal_deg",
        ],
    },
    {
        "key": "mvrecon_reconstruction",
        "label": "MVRecon Reconstruction",
        "metric_keys": [
            "accuracy",
            "completion",
            "overall",
            "precision",
            "recall",
            "fscore",
        ],
    },
    {
        "key": "depth_accuracy",
        "label": "Depth Accuracy",
        "metric_keys": [
            "delta103_percent",
            "delta125_percent",
            "delta125_2_percent",
            "delta125_3_percent",
        ],
    },
    {
        "key": "pose_trajectory",
        "label": "Pose Trajectory",
        "metric_keys": ["ate_rmse", "ate_mean", "ate_median"],
    },
    {
        "key": "pose_error",
        "label": "Pose Error",
        "metric_keys": [
            "relative_rotation_error_deg_mean",
            "relative_translation_error_deg_mean",
            "relative_translation_axis_error_deg_mean",
            "relative_pose_error_deg_mean",
            "relative_pose_axis_error_deg_mean",
        ],
    },
    {
        "key": "pose_accuracy",
        "label": "Pose Accuracy",
        "metric_keys": [
            "RRA_at_5",
            "RRA_at_15",
            "RRA_at_30",
            "RTA_at_5",
            "RTA_at_15",
            "RTA_at_30",
            "mAA_30",
        ],
    },
    {
        "key": "pose_auc",
        "label": "Pose AUC",
        "metric_keys": ["auc_1", "auc_3", "auc_5", "auc_10", "auc_20", "auc_30"],
    },
]


METHOD_DISPLAY_BASE_NAMES = {
    "AMB3R": "AMB3R",
    "COME": "COME",
    "CUT3R": "CUT3R",
    "DA3": "DA3",
    "DA3-G": "DA3-G",
    "DA3-L": "DA3-L",
    "DA3-NGL": "DA3-NGL",
    "DEJAVIEW": "Déjà View",
    "DEPTHLM": "DepthLM",
    "DEPTHVLM": "DepthVLM",
    "DUST3R": "DUSt3R",
    "FAST3R": "Fast3R",
    "FASTVGGT": "FastVGGT",
    "GENCEPTION_DEPTH": "GenCeption-Depth",
    "GENCEPTION_POSE": "GenCeption-Pose",
    "IGGT": "IGGT",
    "IGGT4D": "IGGT4D",
    "MAPANYTHING": "MapAnything",
    "MAST3R": "MASt3R",
    "OMNIVGGT": "OmniVGGT",
    "PI3": "Pi3",
    "PI3_X": "Pi3X",
    "POW3R": "Pow3R",
    "QWEN3_VL": "Qwen3-VL",
    "CLAUDE_OPUS_5_5": "Claude Opus 5.5",
    "GPT_6_ASTRA": "GPT-6 Astra",
    "GEMINI_4_ARGON": "Gemini 4 Argon",
    "ROBUSTVGGT": "RobustVGGT",
    "SENSENOVA_VISION": "SenseNova-Vision",
    "VGGT": "VGGT",
    "VGGT_OMEGA": "VGGT-Ω",
    "VLM3": "VLM3",
    "MOGE": "MoGe",
    "MOGE2": "MoGe-2",
    "MOGE3": "MoGe-3",
    "MARIGOLD": "Marigold",
    "MARIGOLD_V2": "Marigold V2",
    "MARIGOLD_V1_1": "Marigold v1.1",
    "LOTUS_D": "Lotus-D",
    "DEPTH_PRO": "Depth Pro",
    "UNIDEPTH_V2": "UniDepthV2",
    "INFINIDEPTH": "InfiniDepth",
    "DVD": "DVD",
    "DEPTHCRAFTER": "DepthCrafter",
    "CHRONODEPTH": "ChronoDepth",
    "GEONEXT_OFFICIAL_SVD": "GeoNeXt",
    "ABOT_RECON": "ABot-Recon",
    "LINGBOT_MAP": "LingBot-Map",
    "LOGER": "LoGeR",
    "PI_LONG": "Pi-Long",
    "SCAL3R": "Scal3R",
    "VGGT_LONG": "VGGT-Long",
    "VGGT_SLAM": "VGGT-SLAM",
    "ZIPMAP": "ZipMap",
    "COME": "Co-Me",
}

# Model type of each base method, for the dashboard's model-type filter (unlisted methods are "other").
METHOD_TYPE_LABELS = {
    "feedforward_3d": "Feed-forward 3D/4D",
    "streaming": "Long-sequence / streaming",
    "mono_depth": "Monocular depth",
    "video_generative": "Video generative",
    "vlm": "Open VLM",
    "api_vlm": "Commercial VLM (API)",
    "other": "Other",
}
METHOD_TYPES = {
    **dict.fromkeys(("DUST3R", "MAST3R", "POW3R", "CUT3R", "FAST3R", "VGGT", "VGGT_OMEGA", "PI3_X", "AMB3R",
                     "FASTVGGT", "COME", "MAPANYTHING", "OMNIVGGT", "DEJAVIEW", "IGGT", "IGGT4D", "ROBUSTVGGT",
                     "DA3-G", "DA3-L", "DA3-NGL", "DA3", "TRIDENT"), "feedforward_3d"),
    **dict.fromkeys(("VGGT_LONG", "PI_LONG", "VGGT_SLAM", "LOGER", "ZIPMAP", "LINGBOT_MAP", "R3", "SCAL3R",
                     "ABOT_RECON"), "streaming"),
    **dict.fromkeys(("MOGE", "MOGE2", "MOGE3", "DEPTH_PRO", "UNIDEPTH_V2", "INFINIDEPTH", "MARIGOLD", "MARIGOLD_V2",
                     "MARIGOLD_V1_1", "LOTUS_D"), "mono_depth"),
    **dict.fromkeys(("DEPTHCRAFTER", "CHRONODEPTH", "DVD", "GEONEXT_OFFICIAL_SVD",
                     "GENCEPTION_DEPTH", "GENCEPTION_POSE"), "video_generative"),
    **dict.fromkeys(("DEPTHVLM", "DEPTHLM", "VLM3", "SENSENOVA_VISION", "QWEN3_VL"), "vlm"),
    **dict.fromkeys(("CLAUDE_OPUS_5_5", "GPT_6_ASTRA", "GEMINI_4_ARGON"), "api_vlm"),
}

TRACK_DISPLAY_SUFFIXES = {
    "rgb": "",
    "rgb_depth": " [+D]",
    "rgb_cameras": " [+C]",
    "rgb_depth_cameras": " [+DC]",
    "rgb_cameras_depth": " [+DC]",
    "rgb_masked_depth": " [+mD]",
    "rgb_masked_depth_cameras": " [+mDC]",
    "rgb_sparse_depth": " [+sD]",
    "rgb_sparse_depth_cameras": " [+sDC]",
}

AUXILIARY_TRACKS = {
    "rgb_depth",
    "rgb_cameras",
    "rgb_depth_cameras",
    "rgb_cameras_depth",
    "rgb_masked_depth",
    "rgb_masked_depth_cameras",
    "rgb_sparse_depth",
    "rgb_sparse_depth_cameras",
}


def method_track(method: str) -> str:
    raw = str(method or "")
    for track in sorted(TRACK_DISPLAY_SUFFIXES, key=len, reverse=True):
        if raw.endswith(f"_{track}"):
            return track
    return "rgb"


def has_auxiliary_input(method: str) -> bool:
    return method_track(method) in AUXILIARY_TRACKS


def method_type(method: str) -> str:
    """The model type of a method ID (its track suffix ignored), a METHOD_TYPE_LABELS key."""
    raw = str(method or "")
    track = method_track(raw)
    base = raw[: -len(track) - 1] if track != "rgb" and raw.endswith(f"_{track}") else raw
    return METHOD_TYPES.get(base, "other")


def display_method_name(method: str) -> str:
    """Return a reader-facing method label while keeping raw IDs stable."""
    raw = str(method or "")
    for track, suffix in sorted(TRACK_DISPLAY_SUFFIXES.items(), key=lambda item: -len(item[0])):
        marker = f"_{track}"
        if raw.endswith(marker):
            base = raw[: -len(marker)]
            return f"{METHOD_DISPLAY_BASE_NAMES.get(base, base)}{suffix}"
    return METHOD_DISPLAY_BASE_NAMES.get(raw, raw)


def parse_number(value: str | None) -> float | int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    if number.is_integer() and len(text) < 12:
        return int(number)
    return number


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def average(values: list[float]) -> float | None:
    valid = [float(value) for value in values if finite(value)]
    return mean(valid) if valid else None


def track_ranking_tasks(track: str, tasks: list[str], task_datasets: Mapping[str, set[str]]) -> list[str]:
    """The tasks a track's overall score and average rank are taken over: every task for ``rgb``; for a conditioned
    track ([+C], [+D]) only the tasks the task/track matrix lets it run on a leaderboard dataset. A task it cannot
    run (MonoDepth [+C], or VideoDepth [+C], which runs only on the unscored SpatialArena) is not a missing result."""
    if track == "rgb":
        return tasks
    return [task for task in tasks
            if any(on_matrix(task, track, dataset)
                   for dataset in task_datasets.get(task, set()) - LEADERBOARD_EXCLUDED_DATASETS)]


def average_with_missing_zero(values_by_key: dict[str, float], keys: list[str]) -> float | None:
    if not keys:
        return None
    values = [float(values_by_key[key]) if finite(values_by_key.get(key)) else 0.0 for key in keys]
    return mean(values)


def format_metric(value: float | None, metric_key: str) -> str:
    if value is None:
        return "-"
    precision = int(METRIC_BY_KEY[metric_key].get("precision", 3))
    return f"{float(value):.{precision}f}"


def task_sort_key(task: str) -> tuple[int, str]:
    try:
        return (TASK_ORDER.index(task), task)
    except ValueError:
        return (len(TASK_ORDER), task)


def display_task_name(task: str) -> str:
    return TASK_DISPLAY_NAMES.get(str(task or ""), str(task or ""))


def canonical_task_name(task: str) -> str:
    return TASK_ALIASES.get(str(task or ""), str(task or ""))


def display_dataset_name(dataset: str) -> str:
    """Reader-facing dataset name; a SLAM ``<dataset>@<range>`` key reads "<Dataset> · <range>"."""
    base, _, slam_range = str(dataset or "").partition(SLAM_RANGE_SEPARATOR)
    name = DATASET_DISPLAY_NAMES.get(base, base)
    return f"{name} · {slam_range}" if slam_range else name


def row_dataset_key(task: str, dataset: str, variant: str) -> str:
    """Dataset key of a summary row: SLAM rows carry their range track (``7scenes@short``), others the dataset."""
    return f"{dataset}{SLAM_RANGE_SEPARATOR}{variant}" if task == "SLAM" and variant else dataset


def with_depth_percents(metrics: dict[str, Any]) -> dict[str, Any]:
    """``metrics`` with each missing depth percent metric derived from its fraction (summaries and reports written
    before the dashboard showed every delta as a percent)."""
    derived = dict(metrics)
    for percent_key, fraction_key in DEPTH_PERCENT_SOURCES.items():
        if derived.get(percent_key) is None and finite(derived.get(fraction_key)):
            derived[percent_key] = float(derived[fraction_key]) * 100.0
    return derived


def read_summary_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for raw_row in reader:
            task = canonical_task_name(raw_row.get("task", ""))
            allowed_metrics = set(metric_keys_for_task(task))
            values = with_depth_percents({key: parse_number(value) for key, value in raw_row.items()})
            metrics = {
                metric_key: values[metric_key]
                for metric_key in METRIC_KEYS
                if metric_key in allowed_metrics and values.get(metric_key) is not None
            }
            rows.append(
                {
                    "kind": kind_for_task(task, raw_row.get("kind", "")),
                    "method": raw_row.get("method", ""),
                    "track": method_track(raw_row.get("method", "")),
                    "has_auxiliary_input": has_auxiliary_input(raw_row.get("method", "")),
                    "task": task,
                    "task_source": raw_row.get("task", ""),
                    "dataset": row_dataset_key(task, raw_row.get("dataset", ""), raw_row.get("variant", "")),
                    "dataset_base": raw_row.get("dataset", ""),
                    "slam_range": raw_row.get("variant", "") if task == "SLAM" else "",
                    "scenes": parse_number(raw_row.get("scenes")) or 0,
                    "frames": parse_number(raw_row.get("frames")) or 0,
                    "valid_pixels": parse_number(raw_row.get("valid_pixels")) or 0,
                    "matched_images": parse_number(raw_row.get("matched_images")) or 0,
                    "pairs": parse_number(raw_row.get("pairs")) or 0,
                    "sequences_total": parse_number(raw_row.get("sequences_total")) or 0,
                    "metrics": metrics,
                }
            )
    return rows


def on_slam_protocol(method: str, task: str, slam_range: str) -> bool:
    """False for a SLAM result or failure on a range the protocol does not evaluate the method on
    (``slam_protocol.in_range_track``), e.g. a large-range run of a method taken out of the large range; such records
    are left off the dashboard. Every non-SLAM record, every SLAM record without a range and every method the
    protocol has no budget for (custom or unregistered methods) is kept."""
    if task not in SLAM_TASKS or not slam_range:
        return True
    # pylint: disable=import-outside-toplevel
    from spatialarena.slam_protocol import H200_FRAME_BUDGET, SLAM_RANGES, in_range_track

    track = method_track(method)
    base = method[: -len(track) - 1] if track != "rgb" and method.endswith(f"_{track}") else method
    if slam_range not in SLAM_RANGES or base not in H200_FRAME_BUDGET:
        return True
    return in_range_track(base, slam_range, track)


def filter_off_matrix_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Drop rows whose (task, track, dataset) the task/track matrix rules out, e.g. MonoDepth given GT depth.

    The old shell pipeline produced some of these (``PI3_X_rgb_depth`` on MonoDepth); they score the method on
    the input it was handed, so they are not shown. SLAM rows on a range the protocol does not evaluate the method
    on (``on_slam_protocol``) are dropped too. Returns the kept rows and the dropped row count per
    ``<method>|<task>``.
    """
    kept, dropped = [], Counter()
    for row in rows:
        if (on_matrix(row["task"], row["track"], row["dataset_base"])
                and on_slam_protocol(row["method"], row["task"], row["slam_range"])):
            kept.append(row)
        else:
            dropped[f"{row['method']}|{row['task']}"] += 1
    return kept, dict(sorted(dropped.items()))


def filter_legacy_method_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drop obsolete alias rows when an explicit replacement method is present.

    The full-standard TSV intentionally preserves historical result rows. The
    benchmark dashboard ranking view should not show both a legacy alias and the
    replacement method when they represent different benchmark runs.
    """
    available_methods = {row["method"] for row in rows}
    excluded_counts: Counter[str] = Counter()
    filtered_rows = []
    for row in rows:
        method = row["method"]
        replacement = LEGACY_METHOD_REPLACEMENTS.get(method)
        if replacement and replacement in available_methods:
            excluded_counts[method] += 1
            continue
        filtered_rows.append(row)
    return filtered_rows, {
        "excluded_legacy_methods": {
            method: {
                "replacement": LEGACY_METHOD_REPLACEMENTS[method],
                "rows": count,
                "reason": "explicit replacement results are available",
            }
            for method, count in sorted(excluded_counts.items())
        }
    }


def collect_unsupported_method_statuses(report_roots: list[Path] | None) -> list[dict[str, Any]]:
    if not report_roots:
        return []

    statuses_by_key: dict[tuple[str, str, str], tuple[float, dict[str, Any]]] = {}
    for report_root in report_roots:
        for path in sorted(Path(report_root).rglob("unsupported_status.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            method = str(payload.get("method") or payload.get("base_method") or path.parent.name)
            base_method = str(payload.get("base_method") or method)
            track = str(payload.get("benchmark_track") or "rgb")
            key = (method, base_method, track)
            status = {
                "method": method,
                "method_display": display_method_name(method),
                "base_method": base_method,
                "base_method_display": display_method_name(base_method),
                "track": track,
                "status": str(payload.get("unsupported_status") or "unsupported"),
                "reason": str(payload.get("unsupported_reason") or ""),
                "report_dir": str(payload.get("report_dir") or path.parent),
                "source_path": str(path),
            }
            mtime = path.stat().st_mtime
            current = statuses_by_key.get(key)
            if current is None or mtime > current[0]:
                statuses_by_key[key] = (mtime, status)
    return sorted(
        (status for _mtime, status in statuses_by_key.values()),
        key=lambda row: (row["method_display"].lower(), row["track"], row["status"]),
    )


def mvrecon_required_metrics_for_dataset(dataset: str) -> tuple[str, ...]:
    """Which MVRecon metric keys a *complete* row must have, before triggering
    the expensive recover_geometry_metrics_from_reports fallback (which
    re-runs evaluate_mvrecon_scene -- for dtu, a full TSDF fusion + Sim(3)
    alignment + eval_dtu_mesh nearest-neighbor pass per scene).

    dtu rows written before 2026-09-17 (evaluate_dtu_mvsnet's own protocol
    only had accuracy/completion/overall; see eval_dtu_mesh.eval's
    fscore_threshold comment for why precision/recall/fscore were added)
    legitimately lack precision/recall/fscore and would otherwise force a
    full recompute of every MVRecon scene across every report root just to
    backfill 3 fields on scenes that already have everything else correct.
    Accept those stale-but-otherwise-complete dtu rows as-is; a fresh dtu
    eval always writes all 6 now, so this only matters for old report data.
    Every other MVRecon dataset's compute_reconstruction_metrics has always
    produced all 6, so a real gap there should still trigger recovery."""
    if dataset.lower() == "dtu":
        return ("accuracy", "completion", "overall")
    return ("accuracy", "completion", "overall", "precision", "recall", "fscore")


def recover_geometry_metrics_from_reports(
    report_roots: list[Path] | None,
    scene_sources: Mapping[tuple[str, str, str, str], Path] | None = None,
) -> dict[tuple[str, str, str], dict[str, float]]:
    """Recompute normal/point-cloud metrics missing from older ``eval_*.json`` files.

    When ``scene_sources`` maps ``(method, task, dataset, scene)`` to the eval file that
    won deduplication, only those files are recomputed, so superseded duplicates in
    other report roots are neither re-evaluated nor double-weighted.
    """
    if not report_roots:
        return {}
    resolved_sources = (
        {key: Path(path).resolve() for key, path in scene_sources.items()} if scene_sources is not None else None
    )

    recovered: dict[tuple[str, str, str], dict[str, dict[str, float]]] = {}
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
            dataset_obj = spec.create_dataset(DEFAULT_BENCHMARK_ROOT)
            scene_info = next((item for item in dataset_obj.list_scenes() if item.scene == scene), None)
        except Exception:
            scene_info = None
        scene_info_cache[cache_key] = scene_info
        return scene_info

    for report_root in report_roots:
        for path in sorted(Path(report_root).rglob("eval_*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for scene in payload.get("scene_results", []):
                task = canonical_task_name(str(scene.get("task") or ""))
                dataset = str(scene.get("dataset") or "")
                scene_name = str(scene.get("scene") or scene.get("scene_name") or "")
                method = str(scene.get("method") or "")
                if not task or not dataset or not scene_name or not method:
                    continue
                if resolved_sources is not None and (
                    resolved_sources.get((method, task, dataset, scene_name)) != path.resolve()
                ):
                    continue
                if path.name == "eval_depth.json" and task not in {"MonoDepth", "VideoDepth"}:
                    continue
                if path.name == "eval_mvrecon.json" and task != "MVRecon":
                    continue
                if path.name == "eval_sparse.json" and task not in {"SfM", "SLAM", "WeakTextureSfM"}:
                    continue
                # needs_geometry_recovery (the caller) only means *some* row
                # in the summary is incomplete -- it says nothing about
                # *this* scene. Without this per-scene check, one stale row
                # anywhere forced every MonoDepth/VideoDepth/MVRecon scene
                # across every report root to recompute from scratch,
                # including scenes whose own eval_*.json already had
                # everything (hours-long regeneration instead of seconds,
                # 2026-09-17).
                existing_metrics = scene.get("metrics") or {}
                if task in {"MonoDepth", "VideoDepth"} and "mae_normal_deg" in existing_metrics:
                    continue
                if task == "MVRecon" and all(
                    metric_key in existing_metrics
                    for metric_key in mvrecon_required_metrics_for_dataset(dataset)
                ):
                    continue
                scene_info = resolve_scene_info(task, dataset, scene_name)
                if scene_info is None:
                    continue
                weight = int(scene.get("num_metric_units") or scene.get("num_evaluated_frames") or 0) or 1
                recovered_metrics: dict[str, float] = {}
                if task in {"MonoDepth", "VideoDepth"}:
                    try:
                        prediction_scene_path = Path(scene.get("prediction_scene_path") or "")
                        depth_result = evaluate_depth_scene(
                            scene_info,
                            prediction_scene_path,
                            config=DepthEvaluationConfig(),
                        )
                    except Exception:
                        depth_result = {}
                    for metric_key in (
                        "mae_normal_deg",
                        "accuracy",
                        "completion",
                        "overall",
                        "precision",
                        "recall",
                        "fscore",
                    ):
                        value = depth_result.get("metrics", {}).get(metric_key)
                        if value is not None:
                            recovered_metrics[metric_key] = float(value)
                if task == "MVRecon":
                    try:
                        mvrecon_result = evaluate_mvrecon_scene(
                            scene_info,
                            Path(scene.get("prediction_scene_path") or ""),
                            config=MVReconEvaluationConfig(),
                        )
                    except Exception:
                        mvrecon_result = {}
                    for metric_key in ("accuracy", "completion", "overall", "precision", "recall", "fscore"):
                        value = mvrecon_result.get("metrics", {}).get(metric_key)
                        if value is not None:
                            recovered_metrics[metric_key] = float(value)
                if not recovered_metrics:
                    continue
                key = (method, task, dataset)
                metric_sums = recovered.setdefault(key, {})
                for metric_key, value in recovered_metrics.items():
                    bucket = metric_sums.setdefault(metric_key, {"sum": 0.0, "weight": 0.0})
                    bucket["sum"] += float(value) * weight
                    bucket["weight"] += weight

    return {
        key: {
            metric_key: bucket["sum"] / bucket["weight"]
            for metric_key, bucket in metric_sums.items()
            if bucket["weight"] > 0
        }
        for key, metric_sums in recovered.items()
    }


def metric_keys_for_rows(rows: list[dict[str, Any]]) -> list[str]:
    """Declared metric columns for whichever task(s) `rows` belong to --
    NOT filtered by whether any row actually has a value yet. A column
    should stay visible (rendered as "-" per method) even when nothing
    populates it, so it doesn't silently vanish and look like the metric
    was never part of the protocol (e.g. mae_normal_deg for a task/dataset
    combination where no method's evaluation has produced it)."""
    declared = {
        metric_key
        for row in rows
        for metric_key in metric_keys_for_task(row["task"])
    }
    return [metric_key for metric_key in METRIC_KEYS if metric_key in declared]


def rank_metric_entries(entries: list[dict[str, Any]], metric_key: str) -> dict[str, int]:
    direction = METRIC_BY_KEY[metric_key]["direction"]
    reverse = direction == "higher"
    sorted_entries = sorted(
        entries,
        key=lambda entry: (
            -float(entry["value"]) if reverse else float(entry["value"]),
            entry["method"],
        ),
    )
    ranks: dict[str, int] = {}
    previous_value: float | None = None
    previous_rank = 0
    for index, entry in enumerate(sorted_entries, start=1):
        value = float(entry["value"])
        if previous_value is not None and math.isclose(value, previous_value, rel_tol=1e-12, abs_tol=1e-12):
            rank = previous_rank
        else:
            rank = index
            previous_rank = rank
            previous_value = value
        ranks[entry["method"]] = rank
    return ranks


def normalized_scores(entries: list[dict[str, Any]], metric_key: str) -> dict[str, float]:
    values = [float(entry["value"]) for entry in entries]
    low = min(values)
    high = max(values)
    if math.isclose(low, high, rel_tol=1e-12, abs_tol=1e-12):
        return {entry["method"]: 100.0 for entry in entries}
    direction = METRIC_BY_KEY[metric_key]["direction"]
    scores = {}
    for entry in entries:
        value = float(entry["value"])
        if direction == "higher":
            score = 100.0 * (value - low) / (high - low)
        else:
            score = 100.0 * (high - value) / (high - low)
        scores[entry["method"]] = max(0.0, min(100.0, score))
    return scores


def slam_range_sort_key(slam_range: str) -> tuple[int, str]:
    """Order SLAM ranges short, long, large, then any other range alphabetically."""
    order = SLAM_RANGE_ORDER.index(slam_range) if slam_range in SLAM_RANGE_ORDER else len(SLAM_RANGE_ORDER)
    return order, slam_range


def slam_row_complete(row: dict[str, Any]) -> bool:
    """A SLAM cell is rankable when the method evaluated every sequence of it with ATE and drift (protocol rule)."""
    total = int(row.get("sequences_total") or 0)
    return (
        total > 0
        and int(row.get("scenes") or 0) >= total
        and all(finite(row["metrics"].get(metric_key)) for metric_key in SLAM_SCORE_METRICS)
    )


def slam_range_ranked_methods(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Per range: methods with a complete cell on every dataset of that range, i.e. on identical case sets."""
    slam_rows = [row for row in rows if row["task"] in SLAM_TASKS]
    ranked: dict[str, set[str]] = {}
    for slam_range in {row["slam_range"] for row in slam_rows}:
        range_rows = [row for row in slam_rows if row["slam_range"] == slam_range]
        datasets = {row["dataset_base"] for row in range_rows}
        complete: dict[str, set[str]] = defaultdict(set)
        for row in range_rows:
            if slam_row_complete(row):
                complete[row["method"]].add(row["dataset_base"])
        ranked[slam_range] = {method for method, covered in complete.items() if covered == datasets}
    return ranked


def slam_cell_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    """0-100 score of one dataset@range cell: mean of the normalized ATE and drift scores among complete rows."""
    complete_rows = [row for row in rows if slam_row_complete(row)]
    if not complete_rows:
        return {}
    per_metric = [
        normalized_scores(
            [{"method": row["method"], "value": row["metrics"][metric_key]} for row in complete_rows], metric_key
        )
        for metric_key in SLAM_SCORE_METRICS
    ]
    return {row["method"]: average([scores[row["method"]] for scores in per_metric]) for row in complete_rows}


def rank_scores(scores: dict[str, float]) -> dict[str, int]:
    """Competition ranks of 0-100 scores, higher first."""
    entries = [{"method": method, "value": score} for method, score in scores.items() if finite(score)]
    return rank_metric_entries(entries, "delta125_percent") if entries else {}


def failed_slam_method_row(method: str, failures: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """A SLAM table row for a method whose every sequence of the cell failed (e.g. all GPU OOM)."""
    return {
        "method": method,
        "method_display": display_method_name(method),
        "track": method_track(method),
        "has_auxiliary_input": has_auxiliary_input(method),
        "scenes": 0,
        "metrics": {},
        "failure_label": slam_failure_label(failures),
    }


def attach_slam_scene_cell_scores(table: dict[str, Any], cell: dict[str, Any]) -> None:
    """A dataset@range table scored per sequence (slam_case_scores): every participant is ranked by its cell score,
    the mean over the cell's sequences with failed or missing ones counting 0; rows that fell short get a coverage
    label and the failures, and a method whose every sequence failed gets a row with only its failure label."""
    present = {row["method"] for row in table["methods"]}
    for method in sorted(set(cell["scores"]) - present):
        table["methods"].append(failed_slam_method_row(method, cell["failures"].get(method, {})))
    total = len(cell["cases"])
    scores = {method: average([by_scene.get(scene, 0.0) for scene in cell["cases"]]) or 0.0
              for method, by_scene in cell["scores"].items()}
    ranks = rank_scores(scores)
    for method_row in table["methods"]:
        method = method_row["method"]
        failures = cell["failures"].get(method, {})
        method_row["ranked"] = method in scores
        method_row["score"] = scores.get(method)
        method_row["score_rank"] = ranks.get(method)
        method_row["coverage_label"] = slam_coverage_label(len(cell["evaluated"].get(method, ())), total, failures)
        method_row["incomplete"] = bool(method_row["coverage_label"])  # metric means over a subset: not ranked
        method_row["failed_scenes"] = {scene: failure["label"] for scene, failure in sorted(failures.items())}


def attach_slam_cell_scores(table: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Mark a dataset@range table's methods as ranked or not and give the ranked ones the cell score and ranks."""
    scores = slam_cell_scores(rows)
    complete = {row["method"] for row in rows if slam_row_complete(row)}
    complete_rows = summarize_method_rows([row for row in rows if row["method"] in complete], table["metrics"])
    metric_ranks = {
        method_row["method"]: {key: cell["rank"] for key, cell in method_row["metrics"].items()}
        for method_row in complete_rows
    }
    sequences_total = max((int(row.get("sequences_total") or 0) for row in rows), default=0)
    ranks = rank_scores(scores)
    for method_row in table["methods"]:
        method = method_row["method"]
        method_row["ranked"] = method in complete
        method_row["score"] = scores.get(method)
        method_row["score_rank"] = ranks.get(method)
        method_row["coverage_label"] = f"{method_row['scenes']}/{sequences_total} sequences"
        for metric_key, cell in method_row["metrics"].items():
            cell["rank"] = metric_ranks.get(method, {}).get(metric_key)


def build_slam_range_tables(task_rows: list[dict[str, Any]],
                            cells: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """One table per SLAM range: a column group per dataset. Without ``cells`` it ranks only the methods that cover
    the whole range; with them (slam_case_scores) every method that ran the range, by the mean over all its
    sequences (failed or missing ones counting 0)."""
    ranked_by_range = slam_range_ranked_methods(task_rows)
    tables = {}
    ranges = {row["slam_range"] for row in task_rows} | {cell["range"] for cell in (cells or {}).values()}
    for slam_range in sorted(ranges, key=slam_range_sort_key):
        range_rows = [row for row in task_rows if row["slam_range"] == slam_range]
        range_cells = {cell["dataset_base"]: cell for cell in (cells or {}).values() if cell["range"] == slam_range}
        datasets = sorted({row["dataset_base"] for row in range_rows} | set(range_cells))
        failures_by_method: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        if cells is not None:
            ranked = {method for cell in range_cells.values() for method in cell["scores"]}
            cases = [(dataset, scene) for dataset in datasets for scene in range_cells[dataset]["cases"]]
            range_scores = {
                method: average([range_cells[dataset]["scores"].get(method, {}).get(scene, 0.0)
                                 for dataset, scene in cases]) or 0.0
                for method in ranked
            }
            for dataset, cell in range_cells.items():
                for method, failures in cell["failures"].items():
                    failures_by_method[method][dataset] = failures
        else:
            ranked = ranked_by_range.get(slam_range, set())
            cell_scores = {
                dataset: slam_cell_scores(
                    [row for row in range_rows if row["dataset_base"] == dataset and row["method"] in ranked]
                )
                for dataset in datasets
            }
            range_scores = {
                method: average([cell_scores[dataset][method] for dataset in datasets]) for method in ranked
            }
        score_ranks = rank_scores(range_scores)
        columns = [
            f"{metric_key}{SLAM_COLUMN_SEPARATOR}{dataset}"
            for dataset in datasets
            for metric_key in SLAM_RANGE_TABLE_METRICS
        ]
        methods = []
        for method in sorted({row["method"] for row in range_rows} | set(failures_by_method)):
            method_rows = {row["dataset_base"]: row for row in range_rows if row["method"] == method}
            metrics = {}
            for dataset, row in method_rows.items():
                for metric_key in SLAM_RANGE_TABLE_METRICS:
                    value = row["metrics"].get(metric_key)
                    if finite(value):
                        metrics[f"{metric_key}{SLAM_COLUMN_SEPARATOR}{dataset}"] = {
                            "value": value,
                            "display": format_metric(value, metric_key),
                            "rank": None,
                            "complete": slam_row_complete(row),
                        }
            covered = sum(1 for row in method_rows.values() if slam_row_complete(row))
            if cells is not None:
                evaluated = sum(len(range_cells[dataset]["evaluated"].get(method, ())) for dataset in datasets)
                failures = {f"{dataset}/{scene}": failure for dataset, by_scene in failures_by_method[method].items()
                            for scene, failure in by_scene.items()}
                coverage_label = slam_coverage_label(evaluated, len(cases), failures)
            else:
                coverage_label = f"{covered}/{len(datasets)} datasets complete"
            methods.append(
                {
                    "method": method,
                    "method_display": display_method_name(method),
                    "track": method_track(method),
                    "has_auxiliary_input": has_auxiliary_input(method),
                    "datasets": sorted(method_rows),
                    "datasets_count": len(method_rows),
                    "ranked": method in ranked,
                    "score": range_scores.get(method),
                    "score_rank": score_ranks.get(method),
                    "coverage_label": coverage_label,
                    # Datasets where every sequence the method ran failed: their columns show the failure.
                    "dataset_failures": {dataset: slam_failure_label(by_scene)
                                         for dataset, by_scene in sorted(failures_by_method[method].items())
                                         if dataset not in method_rows},
                    "metrics": metrics,
                }
            )
        for column in columns:
            metric_key = column.split(SLAM_COLUMN_SEPARATOR)[0]
            entries = [
                {"method": row["method"], "value": row["metrics"][column]["value"]}
                for row in methods
                if row["ranked"] and column in row["metrics"]
            ]
            ranks = rank_metric_entries(entries, metric_key) if len(entries) >= 2 else {}
            for row in methods:
                if column in row["metrics"]:
                    row["metrics"][column]["rank"] = ranks.get(row["method"])
        tables[slam_range] = {
            "range": slam_range,
            "datasets": datasets,
            "dataset_keys": [f"{dataset}{SLAM_RANGE_SEPARATOR}{slam_range}" for dataset in datasets],
            "metrics": columns,
            "ranked_methods": len(ranked),
            "methods": methods,
        }
    return tables


def summarize_method_rows(rows: list[dict[str, Any]], metric_keys: list[str]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)

    method_rows = []
    for method in sorted(by_method):
        method_input_rows = by_method[method]
        metrics = {}
        for metric_key in metric_keys:
            value = average([row["metrics"].get(metric_key) for row in method_input_rows])
            if value is not None:
                metrics[metric_key] = {
                    "value": value,
                    "display": format_metric(value, metric_key),
                    "rank": None,
                }
        method_rows.append(
            {
                "method": method,
                "method_display": display_method_name(method),
                "track": method_track(method),
                "has_auxiliary_input": has_auxiliary_input(method),
                "datasets": sorted({row["dataset"] for row in method_input_rows}),
                "datasets_count": len({row["dataset"] for row in method_input_rows}),
                "scenes": sum(int(row["scenes"]) for row in method_input_rows),
                "frames": sum(int(row["frames"]) for row in method_input_rows),
                "matched_images": sum(int(row["matched_images"]) for row in method_input_rows),
                "pairs": sum(int(row["pairs"]) for row in method_input_rows),
                "metrics": metrics,
            }
        )

    for metric_key in metric_keys:
        entries = [
            {"method": method_row["method"], "value": method_row["metrics"][metric_key]["value"]}
            for method_row in method_rows
            if metric_key in method_row["metrics"]
        ]
        if len(entries) < 2:
            continue
        ranks = rank_metric_entries(entries, metric_key)
        for method_row in method_rows:
            if metric_key in method_row["metrics"]:
                method_row["metrics"][metric_key]["rank"] = ranks[method_row["method"]]

    return method_rows


def summarize_method_scene_means(rows: list[dict[str, Any]], scene_rows: list[dict[str, Any]],
                                 metric_keys: list[str]) -> list[dict[str, Any]]:
    """A task's all-datasets rows on the leaderboard's cases: every method with a row is listed, and its metrics are
    means over its evaluated scenes of the leaderboard datasets (each scene weighs the same, as in the task score;
    LEADERBOARD_EXCLUDED_DATASETS are left to their own tables). A row without per-scene results there shows none."""
    method_rows = summarize_method_rows(rows, [])
    eligible = {(row["method"], row["task"], row["dataset"]) for row in rows
                if row["dataset_base"] not in LEADERBOARD_EXCLUDED_DATASETS}
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scene_rows:
        if (row["method"], row["task"], row["dataset"]) in eligible:
            by_method[row["method"]].append(row)
    unscored: dict[str, set[str]] = defaultdict(set)  # results only on datasets left out of the leaderboard
    for row in rows:
        if row["dataset_base"] in LEADERBOARD_EXCLUDED_DATASETS:
            unscored[row["method"]].add(row["dataset_base"])
    for method_row in method_rows:
        scenes = by_method.get(method_row["method"], [])
        if not scenes and unscored.get(method_row["method"]):
            # e.g. VideoDepth [+C], run on SpatialArena only: say why the row is empty instead of leaving it blank.
            names = sorted(DATASET_DISPLAY_NAMES.get(name, name) for name in unscored[method_row["method"]])
            method_row["coverage_label"] = f"{', '.join(names)} only · not scored"
            method_row["unscored_only"] = True
        method_row["datasets"] = sorted({row["dataset"] for row in scenes})
        method_row["datasets_count"] = len(method_row["datasets"])
        method_row["scenes"] = len(scenes)
        method_row["metrics"] = {}
        for metric_key in metric_keys:
            value = average([row["metrics"].get(metric_key) for row in scenes])
            if value is not None:
                method_row["metrics"][metric_key] = {"value": value, "display": format_metric(value, metric_key),
                                                     "rank": None}
    for metric_key in metric_keys:
        entries = [{"method": row["method"], "value": row["metrics"][metric_key]["value"]}
                   for row in method_rows if metric_key in row["metrics"]]
        if len(entries) >= 2:
            ranks = rank_metric_entries(entries, metric_key)
            for row in method_rows:
                if metric_key in row["metrics"]:
                    row["metrics"][metric_key]["rank"] = ranks[row["method"]]
    return method_rows


def missing_method_row(method: str) -> dict[str, Any]:
    """A table row for a method the task/track requires but that has no result (every cell shows "-")."""
    return {
        "method": method,
        "method_display": display_method_name(method),
        "track": method_track(method),
        "has_auxiliary_input": has_auxiliary_input(method),
        "datasets": [],
        "datasets_count": 0,
        "scenes": 0,
        "frames": 0,
        "matched_images": 0,
        "pairs": 0,
        "metrics": {},
        "missing": True,
    }


# Registered methods (with every track variant) kept off the dashboard's coverage matrix and "missing" rows until
# they are published: our own models. Their real results, once collected, are listed as usual.
UNLISTED_METHODS = frozenset({"TRIDENT"})


def add_missing_method_rows(task_tables: dict[str, Any], coverage: dict[str, Any]) -> int:
    """Give every task table a row for each method its tracks require, ``missing`` where there is no result.

    "Required" is the coverage matrix's view (build_coverage): a registered method whose ``<task>|<track>`` cell is
    not ``unsupported``, on the datasets that cell expects. Each existing dataset table gets the required methods
    expected on that dataset, and the all-datasets table every required method; rows already there are kept.
    SLAM gets add_missing_slam_rows, and ``UNLISTED_METHODS`` get none. Returns the number of rows added.
    """
    expected: dict[str, dict[str, set[str]]] = {}
    slam_methods: list[tuple[str, str, str]] = []  # (method id, registered method, track)
    for entry in coverage.get("methods", []):
        if entry.get("method") in UNLISTED_METHODS:
            continue
        for key, cell in (entry.get("cells") or {}).items():
            task, _, track = key.partition("|")
            method = cell.get("method_id")
            if not method or cell.get("status", "unsupported") == "unsupported":
                continue
            expected.setdefault(task, {}).setdefault(method, set()).update(cell.get("datasets") or {})
            if task in SLAM_TASKS:
                slam_methods.append((method, entry["method"], track))
    added = 0
    for task, table in task_tables.items():
        if task in SLAM_TASKS:
            added += add_missing_slam_rows(table, slam_methods)
            continue
        required = expected.get(task, {})
        tables = [(table, set(required))] + [
            (dataset_table, {method for method, datasets in required.items() if dataset in datasets})
            for dataset, dataset_table in table.get("dataset_tables", {}).items()
        ]
        for target, methods in tables:
            present = {row["method"] for row in target.get("methods", [])}
            for method in sorted(methods - present):
                target.setdefault("methods", []).append(missing_method_row(method))
                added += 1
    return added


def attach_case_coverage(task_tables: dict[str, Any], coverage: dict[str, Any],
                         case_failures: list[dict[str, Any]] | None = None) -> None:
    """Label the non-SLAM rows that fell short of their cases, from the coverage matrix (build_coverage).

    A row (a dataset table's, or the all-datasets table's over the leaderboard datasets it ran) whose method evaluated
    fewer scenes than expected gets ``coverage_label`` ("2/14 scenes · 12 GPU OOM", failures named by
    ``case_failures``) and ``incomplete`` -- its metric means cover a subset, so the page does not rank them -- and one
    with no scene at all but recorded failures a ``failure_label`` shown in its cells.
    """
    labels = {(failure["method"], failure["task"], failure["dataset"], failure["scene"]): failure["label"]
              for failure in case_failures or ()}
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in coverage.get("methods", []):
        for key, cell in (entry.get("cells") or {}).items():
            if cell.get("method_id") and cell.get("datasets"):
                cells[(key.split("|", 1)[0], cell["method_id"])] = cell["datasets"]

    def label(method: str, task: str, by_dataset: dict[str, Any]) -> tuple[str, str, bool]:
        evaluated = sum(entry["evaluated"] for entry in by_dataset.values())
        expected = sum(entry["expected"] for entry in by_dataset.values())
        if not expected or evaluated >= expected:
            return "", "", False
        failed = {f"{dataset}/{scene}": {"label": labels.get((method, task, dataset, scene), "Failed")}
                  for dataset, entry in by_dataset.items() for scene in entry.get("failed_scenes", ())}
        failure = slam_failure_label(failed)
        return (f"{evaluated}/{expected} scenes" + (f" · {failure}" if failure else ""),
                failure if not evaluated else "", True)

    for task, table in task_tables.items():
        if task in SLAM_TASKS:
            continue
        targets = [(table, None)] + list((dataset_table, dataset)
                                         for dataset, dataset_table in table.get("dataset_tables", {}).items())
        for target, dataset in targets:
            for row in target.get("methods", []):
                by_dataset = cells.get((task, row["method"]), {})
                if dataset is not None:
                    by_dataset = {dataset: by_dataset[dataset]} if dataset in by_dataset else {}
                else:  # the task score's cases: leaderboard datasets the method ran (results or failures)
                    by_dataset = {name: entry for name, entry in by_dataset.items()
                                  if name not in LEADERBOARD_EXCLUDED_DATASETS
                                  and (entry["evaluated"] or entry.get("failed_scenes"))}
                coverage_label, failure_label, incomplete = label(row["method"], task, by_dataset)
                if row.get("missing") and not failure_label:
                    continue  # a plain "-" row: nothing to add
                if incomplete:
                    row["coverage_label"] = coverage_label
                    row["incomplete"] = True
                if failure_label:
                    row["failure_label"] = failure_label


def add_missing_slam_rows(table: dict[str, Any], slam_methods: list[tuple[str, str, str]]) -> int:
    """SLAM's required rows: each range table gets the methods the protocol evaluates on that range
    (slam_protocol.in_range_track) and each dataset@range table those plus the range table's other methods, a row
    ``missing`` where the method has no result or recorded failure there. Returns the number of rows added."""
    # pylint: disable=import-outside-toplevel
    from spatialarena.slam_protocol import SLAM_RANGES, in_range_track

    added = 0
    for slam_range, range_table in (table.get("range_tables") or {}).items():
        required = {method for method, base, track in slam_methods
                    if slam_range in SLAM_RANGES and in_range_track(base, slam_range, track)}
        present = {row["method"] for row in range_table.get("methods", [])}
        for method in sorted(required - present):
            range_table.setdefault("methods", []).append(missing_method_row(method))
            added += 1
        required |= present
        for dataset in range_table.get("datasets", []):
            dataset_table = table.get("dataset_tables", {}).get(f"{dataset}{SLAM_RANGE_SEPARATOR}{slam_range}")
            if dataset_table is None:
                continue
            in_table = {row["method"] for row in dataset_table.get("methods", [])}
            for method in sorted(required - in_table):
                dataset_table.setdefault("methods", []).append(missing_method_row(method))
                added += 1
    return added


def build_task_tables(rows: list[dict[str, Any]], scene_rows: list[dict[str, Any]] | None = None,
                      failures: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Per task: the all-datasets and per-dataset method tables. With SLAM ``scene_rows`` the SLAM tables are scored
    per sequence (slam_case_scores), ``failures`` naming the failed sequences."""
    slam_cells = (slam_case_scores(rows, scene_rows, failures)
                  if scene_rows and any(row["task"] in SLAM_TASKS for row in scene_rows) else None)
    task_tables: dict[str, Any] = {}
    tasks = sorted({row["task"] for row in rows}, key=task_sort_key)
    for task in tasks:
        task_rows = [row for row in rows if row["task"] == task]
        metric_keys = metric_keys_for_task(task)
        dataset_tables = {}
        for dataset in sorted({row["dataset"] for row in task_rows}):
            dataset_rows = [row for row in task_rows if row["dataset"] == dataset]
            dataset_metric_keys = metric_keys_for_rows(dataset_rows)
            dataset_tables[dataset] = {
                "dataset": dataset,
                "metrics": dataset_metric_keys,
                "methods": summarize_method_rows(dataset_rows, dataset_metric_keys),
            }
            if task in SLAM_TASKS:
                if slam_cells is not None:
                    attach_slam_scene_cell_scores(dataset_tables[dataset], slam_cells[dataset])
                else:
                    attach_slam_cell_scores(dataset_tables[dataset], dataset_rows)
        task_tables[task] = {
            "task": task,
            "kind": task_rows[0]["kind"] if task_rows else "",
            "datasets": sorted(dataset_tables),
            "metrics": metric_keys,
            "methods": (summarize_method_scene_means(task_rows, scene_rows, metric_keys)
                        if scene_rows and task not in SLAM_TASKS and any(row["task"] == task for row in scene_rows)
                        else summarize_method_rows(task_rows, metric_keys)),
            "dataset_tables": dataset_tables,
        }
        if task in SLAM_TASKS:
            # Averaging raw SLAM metrics over datasets (cm indoors, tens of metres outdoors) is meaningless, so the
            # all-datasets view is one table per range instead of the cross-dataset method summary.
            task_tables[task]["methods"] = []
            task_tables[task]["range_tables"] = build_slam_range_tables(task_rows, slam_cells)
            task_tables[task]["ranges"] = list(task_tables[task]["range_tables"])
    return task_tables


def add_comparison_scores(
    *,
    method_stats: dict[str, Any],
    task: str,
    dataset: str,
    metric_key: str,
    entries: list[dict[str, Any]],
) -> None:
    ranks = rank_metric_entries(entries, metric_key)
    scores = normalized_scores(entries, metric_key)
    for entry in entries:
        method = entry["method"]
        rank = ranks[method]
        score = scores[method]
        method_stats[method]["scores"].append(score)
        method_stats[method]["ranks"].append(rank)
        method_stats[method]["metric_scores"][metric_key].append(score)
        method_stats[method]["task_scores"][task].append(score)
        method_stats[method]["task_ranks"][task].append(rank)
        method_stats[method]["dataset_scores"][dataset].append(score)
        method_stats[method]["dataset_ranks"][dataset].append(rank)
        method_stats[method]["task_dataset_scores"][task][dataset].append(score)
        method_stats[method]["comparisons"].append(
            {
                "task": task,
                "dataset": dataset,
                "metric": metric_key,
                "rank": rank,
                "score": score,
                "value": float(entry["value"]),
            }
        )
        if rank == 1:
            method_stats[method]["first_place_wins"] += 1


def slam_range_scores_by_method(method_stats: dict[str, Any], methods: list[str]) -> dict[str, dict[str, float]]:
    """Per method and range: mean of its SLAM comparison scores in that range (each dataset weighs the same)."""
    by_method: dict[str, dict[str, float]] = {}
    for method in methods:
        per_range: dict[str, list[float]] = defaultdict(list)
        for comparison in method_stats[method]["comparisons"]:
            if comparison["task"] in SLAM_TASKS:
                per_range[comparison["dataset"].partition(SLAM_RANGE_SEPARATOR)[2]].append(comparison["score"])
        if per_range:
            by_method[method] = {slam_range: average(scores) for slam_range, scores in per_range.items()}
    return by_method


def method_task_score_values(
    stats: dict[str, Any], slam_ranges: dict[str, float] | None, slam_range_count: int
) -> dict[str, float]:
    """Task scores of one method. SLAM is the mean over the ranked ranges, counting a range the method does not
    cover in full as 0 -- the same rule the overall score applies to missing tasks."""
    scores = {
        task: average(task_scores)
        for task, task_scores in stats["task_scores"].items()
        if average(task_scores) is not None
    }
    if slam_ranges and slam_range_count:
        scores["SLAM"] = sum(slam_ranges.values()) / slam_range_count
    return scores


def read_scene_rows(path: Path) -> list[dict[str, Any]]:
    """Per-scene metrics (the collector's ``scenes.tsv``): ``method, task, dataset, scene`` and the metrics; SLAM
    scenes on a range the protocol does not evaluate the method on are left out (``on_slam_protocol``)."""
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        for raw_row in csv.DictReader(file, delimiter="\t"):
            task = canonical_task_name(raw_row.get("task", ""))
            values = with_depth_percents({key: parse_number(value) for key, value in raw_row.items()})
            if task in SLAM_TASKS:  # the collector's scene table names SLAM metrics without the "slam_" prefix
                values = {**{f"slam_{key}": value for key, value in values.items()}, **values}
            if not on_slam_protocol(raw_row.get("method", ""), task, raw_row.get("variant", "")):
                continue
            rows.append({
                "method": raw_row.get("method", ""),
                "task": task,
                "dataset": row_dataset_key(task, raw_row.get("dataset", ""), raw_row.get("variant", "")),
                "scene": raw_row.get("scene", ""),
                "metrics": {key: float(values[key]) for key in metric_keys_for_task(task) if finite(values.get(key))},
            })
    return rows


def read_case_failures(path: Path) -> list[dict[str, Any]]:
    """Recorded case failures (``method, task, dataset, variant, scene, status[, detail]``), e.g. a SLAM run that ran
    out of GPU memory; ``label`` is the reader-facing status (CASE_FAILURE_LABELS). Failures on a SLAM range the
    protocol does not evaluate the method on are left out (``on_slam_protocol``)."""
    failures = []
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        for raw_row in csv.DictReader(file, delimiter="\t"):
            task = canonical_task_name(raw_row.get("task", ""))
            status = raw_row.get("status", "")
            if not on_slam_protocol(raw_row.get("method", ""), task, raw_row.get("variant", "")):
                continue
            failures.append({
                "method": raw_row.get("method", ""),
                "task": task,
                "dataset": row_dataset_key(task, raw_row.get("dataset", ""), raw_row.get("variant", "")),
                "dataset_base": raw_row.get("dataset", ""),
                "slam_range": raw_row.get("variant", "") if task in SLAM_TASKS else "",
                "scene": raw_row.get("scene", ""),
                "status": status,
                "label": CASE_FAILURE_LABELS.get(status, status.replace("_", " ") or "Failed"),
                "detail": raw_row.get("detail", ""),
            })
    return failures


def slam_case_scores(rows: list[dict[str, Any]], scene_rows: list[dict[str, Any]],
                     failures: list[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    """Per SLAM ``dataset@range`` cell: its cases, each participant's per-sequence scores and its failures.

    A cell's participants are the methods with a summary row among ``rows`` or a recorded failure there; its cases
    are the sequences any method evaluated or failed. A sequence's score is the mean over SLAM_SCORE_METRICS (those
    some participant reports there) of the 0-100 min-max normalized value among the participants reporting it, a
    metric the method lacks counting 0; a sequence the method did not evaluate (failed or missing) scores 0.
    Returns ``{dataset: {"cases": [scene], "range": r, "dataset_base": d, "scores": {method: {scene: score}},
    "failures": {method: {scene: failure}}, "evaluated": {method: set(scene)}}}``.
    """
    cells: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["task"] in SLAM_TASKS:
            cell = cells.setdefault(row["dataset"], {"range": row["slam_range"], "dataset_base": row["dataset_base"],
                                                     "methods": set(), "cases": set(), "failures": defaultdict(dict)})
            cell["methods"].add(row["method"])
    for failure in failures or ():
        if failure["task"] in SLAM_TASKS:
            cell = cells.setdefault(failure["dataset"], {"range": failure["slam_range"],
                                                         "dataset_base": failure["dataset_base"], "methods": set(),
                                                         "cases": set(), "failures": defaultdict(dict)})
            cell["methods"].add(failure["method"])
    by_scene: dict[tuple[str, str], dict[str, dict[str, float]]] = defaultdict(dict)
    for row in scene_rows:
        if row["task"] in SLAM_TASKS and row["dataset"] in cells:
            cells[row["dataset"]]["cases"].add(row["scene"])
            if row["method"] in cells[row["dataset"]]["methods"]:
                by_scene[(row["dataset"], row["scene"])][row["method"]] = row["metrics"]
    for failure in failures or ():
        if failure["task"] in SLAM_TASKS:
            cell = cells[failure["dataset"]]
            cell["cases"].add(failure["scene"])
            if failure["method"] not in by_scene.get((failure["dataset"], failure["scene"]), {}):
                cell["failures"][failure["method"]][failure["scene"]] = failure  # not superseded by a scored report
    out = {}
    for dataset, cell in cells.items():
        scores: dict[str, dict[str, float]] = {method: {} for method in cell["methods"]}
        evaluated: dict[str, set[str]] = {method: set() for method in cell["methods"]}
        for scene in sorted(cell["cases"]):
            methods = by_scene.get((dataset, scene), {})
            for method in methods:
                evaluated[method].add(scene)
            per_metric = []
            for metric_key in SLAM_SCORE_METRICS:
                entries = [{"method": method, "value": metrics[metric_key]}
                           for method, metrics in methods.items() if finite(metrics.get(metric_key))]
                if entries:
                    per_metric.append(normalized_scores(entries, metric_key))
            for method in cell["methods"]:
                scores[method][scene] = (
                    average([metric_scores.get(method, 0.0) for metric_scores in per_metric]) or 0.0
                    if method in methods and per_metric else 0.0
                )
        out[dataset] = {"cases": sorted(cell["cases"]), "range": cell["range"], "dataset_base": cell["dataset_base"],
                        "scores": scores, "evaluated": evaluated,
                        "failures": {method: dict(by) for method, by in cell["failures"].items() if by}}
    return out


def slam_failure_label(failures: dict[str, dict[str, Any]] | None) -> str:
    """"2 GPU OOM, 1 Too few poses" for one method's failed sequences in a cell (or several cells)."""
    counts = Counter(failure["label"] for failure in (failures or {}).values())
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(f"{count} {label}" for label, count in ordered)


def slam_coverage_label(evaluated: int, total: int, failures: dict[str, dict[str, Any]] | None) -> str:
    """"5/7 sequences · 2 GPU OOM"; empty when the method evaluated every sequence."""
    if total and evaluated >= total:
        return ""
    failed = slam_failure_label(failures)
    return f"{evaluated}/{total} sequences" + (f" · {failed}" if failed else "")


def add_slam_scene_scores(method_stats: dict[str, Any], cells: dict[str, dict[str, Any]]) -> None:
    """Score SLAM per sequence like the other tasks per scene: every (range, dataset, sequence) case of the task weighs
    the same and one a method did not evaluate -- a recorded failure (OOM, too few poses) or not run -- scores 0. A
    method's task score is the mean over every case of the task, its range score the mean over that range's cases."""
    methods = {method for cell in cells.values() for method in cell["scores"]}
    for dataset, cell in sorted(cells.items()):
        for scene in cell["cases"]:
            for method in sorted(methods):
                score = cell["scores"].get(method, {}).get(scene, 0.0)
                stats = method_stats[method]
                stats["scores"].append(score)
                stats["task_scores"]["SLAM"].append(score)
                stats["dataset_scores"][dataset].append(score)
                stats["task_dataset_scores"]["SLAM"][dataset].append(score)
                stats["slam_range_scores"][cell["range"]].append(score)
        if len(cell["scores"]) < 2:
            continue
        for scene in cell["cases"]:  # per-sequence ranks (average rank) among the methods that evaluated it
            evaluated = [method for method in cell["scores"] if scene in cell["evaluated"][method]]
            ranks = rank_scores({method: cell["scores"][method][scene] for method in evaluated})
            for method, rank in ranks.items():
                stats = method_stats[method]
                stats["ranks"].append(rank)
                stats["task_ranks"]["SLAM"].append(rank)
                stats["dataset_ranks"][dataset].append(rank)
                stats["first_place_wins"] += rank == 1


def scene_score_metric_keys(task: str, scene_rows: list[dict[str, Any]]) -> list[str]:
    """The metrics a dataset's scenes are scored on: for the depth tasks ``DEPTH_SCORE_METRICS``, for the others all
    the task's metrics; in both cases those some method reports on the dataset (MAE-normal is not everywhere)."""
    available = {key for row in scene_rows for key in row["metrics"]}
    keys = DEPTH_SCORE_METRICS if task in DEPTH_TASKS else metric_keys_for_task(task)
    return [key for key in keys if key in available]


def add_scene_scores(method_stats: dict[str, Any], rows: list[dict[str, Any]],
                     scene_rows: list[dict[str, Any]]) -> None:
    """Score the non-SLAM tasks per scene: every scene weighs the same, and a missing one scores 0.

    A scene's metric scores are min-max normalized among the methods that evaluated it (one method alone gets 100);
    a method's scene score is the mean over the score metrics some method reports on that scene, one it lacks
    counting 0. Its task
    score is the mean over the task's cases -- the full-standard cases on its track's datasets that any method
    evaluated -- a case it did not evaluate counting 0. Only scenes whose (method, task, dataset) row is among
    ``rows`` count, so the leaderboard's filters (off-matrix tracks, excluded datasets, auxiliary inputs) apply, and
    a method is scored on a task only if it has per-scene results there.
    """
    # pylint: disable=import-outside-toplevel
    from spatialarena.task_track_matrix import on_matrix

    eligible = {(row["method"], row["task"], row["dataset"]) for row in rows if row["task"] not in SLAM_TASKS}
    by_scene: dict[tuple[str, str, str], dict[str, dict[str, float]]] = defaultdict(dict)
    for row in scene_rows:
        if (row["method"], row["task"], row["dataset"]) in eligible:
            by_scene[(row["task"], row["dataset"], row["scene"])][row["method"]] = row["metrics"]
    task_methods: dict[str, set[str]] = defaultdict(set)
    dataset_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (task, dataset, _scene), methods in by_scene.items():
        task_methods[task].update(methods)
        dataset_rows[(task, dataset)].extend({"metrics": metrics} for metrics in methods.values())
    score_keys = {key: scene_score_metric_keys(key[0], scene_list) for key, scene_list in dataset_rows.items()}

    for (task, dataset, scene), methods in sorted(by_scene.items()):
        keys = score_keys[(task, dataset)]
        metric_results = {}
        for metric_key in keys:
            entries = [{"method": method, "value": metrics[metric_key]}
                       for method, metrics in methods.items() if metric_key in metrics]
            if entries:
                metric_results[metric_key] = (entries, rank_metric_entries(entries, metric_key),
                                              normalized_scores(entries, metric_key))
        keys = [metric_key for metric_key in keys if metric_key in metric_results]  # metrics reported on this scene
        for method in sorted(task_methods[task]):
            if not on_matrix(task, method_track(method), dataset):  # (a non-SLAM key is the dataset)
                continue
            stats = method_stats[method]
            metric_scores = []
            for metric_key in keys:
                entries, ranks, scores = metric_results[metric_key]
                if method not in scores:
                    metric_scores.append(0.0)
                    continue
                score, rank = scores[method], ranks[method]
                metric_scores.append(score)
                stats["metric_scores"][metric_key].append(score)
                stats["ranks"].append(rank)
                stats["task_ranks"][task].append(rank)
                stats["dataset_ranks"][dataset].append(rank)
                value = next(entry["value"] for entry in entries if entry["method"] == method)
                stats["comparisons"].append({"task": task, "dataset": dataset, "scene": scene, "metric": metric_key,
                                             "rank": rank, "score": score, "value": float(value)})
                stats["first_place_wins"] += rank == 1
            scene_score = average(metric_scores) if method in methods and metric_scores else 0.0
            stats["scores"].append(scene_score)
            stats["task_scores"][task].append(scene_score)
            stats["dataset_scores"][dataset].append(scene_score)
            stats["task_dataset_scores"][task][dataset].append(scene_score)


def attach_excluded_dataset_scores(final_rows: list[dict[str, Any]], task_rows: list[dict[str, Any]],
                                   rows: list[dict[str, Any]], scene_rows: list[dict[str, Any]] | None,
                                   failures: list[dict[str, Any]] | None) -> None:
    """Scores on the LEADERBOARD_EXCLUDED_DATASETS, for display only (the dataset radar): scored like any dataset,
    each against the methods that ran it, as ``excluded_dataset_scores`` {task: {dataset: {score, rank, ...}}} on
    the ranking rows -- never in task_dataset_scores, so they reach no leaderboard score."""
    excluded = [row for row in rows if row["dataset_base"] in LEADERBOARD_EXCLUDED_DATASETS]
    if not excluded:
        return
    excluded_failures = [failure for failure in failures or ()
                         if failure["dataset_base"] in LEADERBOARD_EXCLUDED_DATASETS]
    for ranking_rows, keep_auxiliary in ((final_rows, False), (task_rows, True)):
        scored = build_rankings(
            [row for row in excluded if keep_auxiliary or not row["has_auxiliary_input"]], scene_rows,
            [failure for failure in excluded_failures if keep_auxiliary or not has_auxiliary_input(failure["method"])])
        by_method = {row["method"]: row.get("task_dataset_scores", {}) for row in scored["rows"]}
        for row in ranking_rows:
            if by_method.get(row["method"]):
                row["excluded_dataset_scores"] = by_method[row["method"]]


def build_rankings(rows: list[dict[str, Any]], scene_rows: list[dict[str, Any]] | None = None,
                   failures: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Method scores and ranks. With ``scene_rows`` every task is scored per scene (add_scene_scores; SLAM per
    sequence with add_slam_scene_scores, ``failures`` naming the failed ones); every task without them per dataset
    row: each metric min-max normalized across the methods."""
    method_stats: dict[str, Any] = defaultdict(
        lambda: {
            "scores": [],
            "ranks": [],
            "metric_scores": defaultdict(list),
            "task_scores": defaultdict(list),
            "task_ranks": defaultdict(list),
            "dataset_scores": defaultdict(list),
            "dataset_ranks": defaultdict(list),
            # task -> dataset -> scores: a dataset name can appear under several tasks (DTU in MVRecon and SfM).
            "task_dataset_scores": defaultdict(lambda: defaultdict(list)),
            "comparisons": [],
            "first_place_wins": 0,
            "slam_range_scores": defaultdict(list),
        }
    )

    slam_ranked = slam_range_ranked_methods(rows)
    slam_by_scene = scene_rows is not None and any(row["task"] in SLAM_TASKS for row in scene_rows)
    if scene_rows is not None:
        add_scene_scores(method_stats, rows, scene_rows)
    if slam_by_scene:
        add_slam_scene_scores(method_stats, slam_case_scores(rows, scene_rows, failures))
    for task in sorted({row["task"] for row in rows}, key=task_sort_key):
        if scene_rows is not None and (task not in SLAM_TASKS or slam_by_scene):
            continue
        for dataset in sorted({row["dataset"] for row in rows if row["task"] == task}):
            group_rows = [row for row in rows if row["task"] == task and row["dataset"] == dataset]
            group_metric_keys = metric_keys_for_rows(group_rows)
            if task in SLAM_TASKS:
                # Only methods that ran the whole range compete, on the protocol's ranking metrics.
                ranked = slam_ranked.get(group_rows[0]["slam_range"], set())
                group_rows = [row for row in group_rows if row["method"] in ranked]
                group_metric_keys = list(SLAM_SCORE_METRICS)
            for metric_key in group_metric_keys:
                entries = [
                    {"method": row["method"], "value": row["metrics"][metric_key]}
                    for row in group_rows
                    if finite(row["metrics"].get(metric_key))
                ]
                if len(entries) < 2:
                    continue
                add_comparison_scores(
                    method_stats=method_stats,
                    task=task,
                    dataset=dataset,
                    metric_key=metric_key,
                    entries=entries,
                )

    methods = sorted({row["method"] for row in rows} | {failure["method"] for failure in failures or ()
                                                        if failure["task"] in SLAM_TASKS and slam_by_scene})
    if slam_by_scene:  # range and task scores are plain means over the cases (already in task_scores)
        slam_range_scores = {
            method: {slam_range: average(scores)
                     for slam_range, scores in method_stats[method]["slam_range_scores"].items()}
            for method in methods if method_stats[method]["slam_range_scores"]
        }
        slam_range_count = 0
    else:
        slam_range_scores = slam_range_scores_by_method(method_stats, methods)
        slam_range_count = len({slam_range for scores in slam_range_scores.values() for slam_range in scores})
    task_score_values: dict[str, dict[str, float]] = defaultdict(dict)
    for method in methods:
        scores = method_task_score_values(method_stats[method],
                                          None if slam_by_scene else slam_range_scores.get(method), slam_range_count)
        for task, score in scores.items():
            task_score_values[task][method] = score
    slam_range_ranks = {
        slam_range: rank_scores({method: scores[slam_range] for method, scores in slam_range_scores.items()
                                 if slam_range in scores})
        for slam_range in {slam_range for scores in slam_range_scores.values() for slam_range in scores}
    }
    dataset_score_values: dict[str, dict[str, float]] = defaultdict(dict)
    for method in methods:
        stats = method_stats[method]
        for dataset, scores in stats["dataset_scores"].items():
            score = average(scores)
            if score is not None:
                dataset_score_values[dataset][method] = score

    task_ranks: dict[str, dict[str, int]] = {}
    for task, method_scores in task_score_values.items():
        entries = [
            {"method": method, "value": score}
            for method, score in method_scores.items()
            if finite(score)
        ]
        task_ranks[task] = rank_metric_entries(entries, "delta125_percent") if entries else {}
    dataset_ranks: dict[str, dict[str, int]] = {}
    for dataset, method_scores in dataset_score_values.items():
        entries = [
            {"method": method, "value": score}
            for method, score in method_scores.items()
            if finite(score)
        ]
        dataset_ranks[dataset] = rank_metric_entries(entries, "delta125_percent") if entries else {}

    ranking_tasks = sorted(task_score_values, key=task_sort_key)
    task_datasets: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        task_datasets[row["task"]].add(row.get("dataset_base") or row.get("dataset", ""))
    ranking_rows = []
    for method in methods:
        stats = method_stats[method]
        method_task_scores = method_task_score_values(stats, None if slam_by_scene else slam_range_scores.get(method),
                                                      slam_range_count)
        method_tasks = track_ranking_tasks(method_track(method), ranking_tasks, task_datasets)
        missing_task_rank = max((len(ranks) for ranks in task_ranks.values()), default=0) + 1
        method_task_ranks = [
            task_ranks.get(task, {}).get(method, missing_task_rank)
            for task in method_tasks
        ]
        overall_score = average_with_missing_zero(method_task_scores, method_tasks)
        average_rank = average(method_task_ranks)
        task_scores = {
            task: {
                "score": method_task_scores.get(task),
                "rank": task_ranks.get(task, {}).get(method),
                "average_rank": average(stats["task_ranks"][task]),
                "coverage": len(scores),
            }
            for task, scores in stats["task_scores"].items()
        }
        if method in slam_range_scores and "SLAM" in task_scores:
            task_scores["SLAM"]["ranges"] = {
                slam_range: {"score": score, "rank": slam_range_ranks[slam_range].get(method)}
                for slam_range, score in sorted(slam_range_scores[method].items(),
                                                key=lambda item: slam_range_sort_key(item[0]))
            }
        metric_scores = {
            metric_key: average(scores)
            for metric_key, scores in stats["metric_scores"].items()
            if average(scores) is not None
        }
        ranking_rows.append(
            {
                "method": method,
                "method_display": display_method_name(method),
                "track": method_track(method),
                "has_auxiliary_input": has_auxiliary_input(method),
                "score": overall_score,
                "score_display": f"{overall_score:.2f}" if overall_score is not None else "-",
                "ranking_tasks": method_tasks,  # the tasks its overall score averages over (track_ranking_tasks)
                "average_rank": average_rank,
                "average_rank_display": f"{average_rank:.2f}" if average_rank is not None else "-",
                "coverage": len(stats["scores"]),
                "first_place_wins": stats["first_place_wins"],
                "task_scores": task_scores,
                "dataset_scores": {
                    dataset: {
                        "score": average(scores),
                        "rank": dataset_ranks.get(dataset, {}).get(method),
                        "average_rank": average(stats["dataset_ranks"][dataset]),
                        "coverage": len(scores),
                    }
                    for dataset, scores in stats["dataset_scores"].items()
                },
                "metric_scores": metric_scores,
                # Mean score and comparison count per (task, dataset): the dashboard filters recompute task and
                # overall scores from these for a subset of datasets (a task score is the mean over its comparisons).
                "task_dataset_scores": {
                    task: {dataset: {"score": average(scores), "coverage": len(scores)}
                           for dataset, scores in sorted(by_dataset.items())}
                    for task, by_dataset in stats["task_dataset_scores"].items()
                },
            }
        )

    # Conditioned tracks ([+C], [+D]) are kept out of the overall ranking: their score covers only the tasks they run
    # (track_ranking_tasks), so it is not comparable with an rgb method's over every task. They follow the rgb rows,
    # unranked overall; their task ranks stay.
    ranking_rows.sort(
        key=lambda row: (
            row["track"] != "rgb",
            -(row["score"] if row["score"] is not None else -1.0),
            row["average_rank"] if row["average_rank"] is not None else 1e9,
            row["method"],
        )
    )
    ranked_overall = [row for row in ranking_rows if row["track"] == "rgb"]
    for index, row in enumerate(ranked_overall, start=1):
        row["rank"] = index
    for row in ranking_rows[len(ranked_overall):]:
        row["rank"] = None

    return {
        "rows": ranking_rows,
        "tasks": sorted(task_score_values, key=task_sort_key),
        "datasets": sorted(dataset_score_values),
    }


def average_metric_group(metric_scores: dict[str, float], metric_keys: list[str]) -> float | None:
    values = [metric_scores[key] for key in metric_keys if key in metric_scores and finite(metric_scores[key])]
    return average(values)


def build_radar_data(ranking_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_has_data = {
        metric_key
        for row in ranking_rows
        for metric_key, value in row.get("metric_scores", {}).items()
        if finite(value)
    }
    all_axes = [
        {
            "key": metric["key"],
            "label": metric["label"],
            "family": metric["family"],
            "direction": metric["direction"],
        }
        for metric in METRICS
        if metric["key"] in metric_has_data
    ]
    compact_axes = [
        {
            "key": axis["key"],
            "label": axis["label"],
            "metric_keys": [key for key in axis["metric_keys"] if key in metric_has_data],
        }
        for axis in COMPACT_AXES
        if any(key in metric_has_data for key in axis["metric_keys"])
    ]
    task_axes = [
        {"key": task, "label": display_task_name(task)}
        for task in sorted(
            {task for row in ranking_rows for task in row.get("task_scores", {})},
            key=task_sort_key,
        )
    ]
    dataset_axes = [
        {"key": dataset, "label": display_dataset_name(dataset)}
        for dataset in sorted({dataset for row in ranking_rows for dataset in row.get("dataset_scores", {})})
    ]

    method_scores = {}
    for row in ranking_rows:
        metric_scores = row.get("metric_scores", {})
        method_scores[row["method"]] = {
            "compact": [
                average_metric_group(metric_scores, axis["metric_keys"])
                for axis in compact_axes
            ],
            "all": [
                metric_scores.get(axis["key"])
                for axis in all_axes
            ],
            "tasks": [
                row.get("task_scores", {}).get(axis["key"], {}).get("score")
                for axis in task_axes
            ],
            "datasets": [
                row.get("dataset_scores", {}).get(axis["key"], {}).get("score")
                for axis in dataset_axes
            ],
        }

    return {
        "compact_axes": compact_axes,
        "all_axes": all_axes,
        "task_axes": task_axes,
        "dataset_axes": dataset_axes,
        "method_scores": method_scores,
    }


def build_benchmark_visualization(
    input_path: Path,
    report_roots: list[Path] | None = None,
    include_methods: list[str] | None = None,
    *,
    recover_geometry: bool = True,
    scene_sources: Mapping[tuple[str, str, str, str], Path] | None = None,
    scene_rows: list[dict[str, Any]] | None = None,
    case_failures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The dashboard payload from a ``summary.tsv``; with ``scene_rows`` (``read_scene_rows``) every task is scored
    per scene (add_scene_scores, add_slam_scene_scores), ``case_failures`` (``read_case_failures``) naming the
    failed SLAM sequences."""
    raw_rows, off_matrix_rows = filter_off_matrix_rows(read_summary_rows(input_path))
    needs_geometry_recovery = recover_geometry and any(
        (
            row["task"] in {"MonoDepth", "VideoDepth"}
            and "mae_normal_deg" not in row["metrics"]
        )
        or (
            row["task"] == "MVRecon"
            and any(
                metric_key not in row["metrics"]
                for metric_key in mvrecon_required_metrics_for_dataset(row["dataset"])
            )
        )
        for row in raw_rows
    )
    if needs_geometry_recovery:
        recovered_metrics = recover_geometry_metrics_from_reports(report_roots, scene_sources)
        for row in raw_rows:
            metrics = recovered_metrics.get((row["method"], row["task"], row["dataset"]))
            if metrics:
                row["metrics"].update(metrics)
    if include_methods:
        allowed_methods = {str(method) for method in include_methods if str(method).strip()}
        raw_rows = [row for row in raw_rows if row["method"] in allowed_methods]
    rows, row_filter_metadata = filter_legacy_method_rows(raw_rows)
    methods_with_results = {row["method"] for row in rows}
    method_statuses = [
        status
        for status in collect_unsupported_method_statuses(report_roots)
        if status["method"] not in methods_with_results
    ]
    task_tables = build_task_tables(rows, scene_rows, case_failures)
    # SLAM rows are keyed "<dataset>@<range>"; the exclusion is by the dataset itself.
    ranking_eligible_rows = [row for row in rows if row["dataset_base"] not in LEADERBOARD_EXCLUDED_DATASETS]
    final_ranking_rows = [row for row in ranking_eligible_rows if not row["has_auxiliary_input"]]
    eligible_failures = [failure for failure in case_failures or ()
                         if failure["dataset_base"] not in LEADERBOARD_EXCLUDED_DATASETS]
    rankings = build_rankings(final_ranking_rows, scene_rows,
                              [failure for failure in eligible_failures if not has_auxiliary_input(failure["method"])])
    task_rankings = build_rankings(ranking_eligible_rows, scene_rows, eligible_failures)
    attach_excluded_dataset_scores(rankings["rows"], task_rankings["rows"], rows, scene_rows, case_failures)
    radar = build_radar_data(rankings["rows"])
    methods = sorted({row["method"] for row in rows})
    tasks = sorted({row["task"] for row in rows}, key=task_sort_key)
    datasets = sorted({row["dataset"] for row in rows})
    metric_keys = metric_keys_for_rows(rows)
    best_method = rankings["rows"][0]["method"] if rankings["rows"] else None
    method_display_names = {method: display_method_name(method) for method in methods}
    task_catalog = {}
    for task in tasks:
        task_rows = [row for row in rows if row["task"] == task]
        score_rows = [
            row
            for row in task_rankings["rows"]
            if row["task_scores"].get(task, {}).get("score") is not None
        ]
        best_row = max(
            score_rows,
            key=lambda row: (
                row["task_scores"][task]["score"],
                # None: a method scored only from failed cases (all 0) has no average rank
                -(row["task_scores"][task].get("average_rank") or float("inf")),
                row["method"],
            ),
            default=None,
        )
        task_catalog[task] = {
            "task": task,
            "label": TASK_CATALOG.get(task, {}).get("label", display_task_name(task)),
            "group": TASK_CATALOG.get(task, {}).get("group", "Task"),
            "description": TASK_CATALOG.get(task, {}).get("description", ""),
            "datasets": sorted({row["dataset"] for row in task_rows}),
            "datasets_count": len({row["dataset"] for row in task_rows}),
            "methods_count": len({row["method"] for row in task_rows}),
            "metrics_count": len(metric_keys_for_rows(task_rows)),
            "best_method": best_row["method"] if best_row else None,
            "best_method_display": display_method_name(best_row["method"]) if best_row else None,
            "best_score": best_row["task_scores"][task]["score"] if best_row else None,
        }
    for status in method_statuses:
        method_display_names.setdefault(status["method"], status["method_display"])
        method_display_names.setdefault(status["base_method"], status["base_method_display"])

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_tsv": str(Path(input_path)),
        "source_report_roots": [str(Path(path)) for path in report_roots or []],
        "summary": {
            "rows": len(rows),
            "raw_rows": len(raw_rows),
            "methods": len(methods),
            "unsupported_methods": len(method_statuses),
            "tasks": len(tasks),
            "datasets": len(datasets),
            "metrics": len(metric_keys),
            "best_method": best_method,
            "best_method_display": method_display_names.get(best_method, best_method) if best_method else None,
            "ranked_methods": len(rankings["rows"]),
            "auxiliary_methods_excluded_from_final_ranking": sorted(
                {row["method"] for row in rows if row["has_auxiliary_input"]}
            ),
            "datasets_excluded_from_ranking": sorted(
                LEADERBOARD_EXCLUDED_DATASETS & {row["dataset"] for row in rows}
            ),
            **row_filter_metadata,
            "excluded_off_matrix_rows": off_matrix_rows,
        },
        "methods": methods,
        "method_display_names": method_display_names,
        "method_types": {method: method_type(method) for method in methods},
        "method_type_labels": METHOD_TYPE_LABELS,
        "tasks": tasks,
        "task_display_names": {task: display_task_name(task) for task in tasks},
        "task_catalog": task_catalog,
        "datasets": datasets,
        # SLAM range tables label their column groups by the bare dataset, so those names are included too.
        "dataset_display_names": {
            dataset: display_dataset_name(dataset)
            for dataset in sorted(set(datasets) | {row["dataset_base"] for row in rows if row["dataset_base"]})
        },
        "metric_catalog": {
            metric["key"]: {
                "label": metric["label"],
                "family": metric["family"],
                "direction": metric["direction"],
                "kind": metric["kind"],
                "precision": metric["precision"],
            }
            for metric in METRICS
            if metric["key"] in metric_keys
        },
        "task_tables": task_tables,
        "ranking": rankings,
        "task_ranking": task_rankings,
        "radar": radar,
        "method_statuses": method_statuses,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input benchmark summary TSV (from scripts/test/summarize_benchmark_reports.py).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON path for the benchmark dashboard.",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        action="append",
        default=[],
        help="Report root to scan for unsupported_status.json files.",
    )
    parser.add_argument(
        "--method",
        dest="methods",
        action="append",
        default=[],
        help="Restrict the exported benchmark payload to this raw method ID. Repeatable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = build_benchmark_visualization(args.input, args.report_root, args.methods)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote benchmark visualization data to {args.output}")


if __name__ == "__main__":
    main()
