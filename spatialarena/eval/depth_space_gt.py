# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Sequence depth-space alignment and explicitly oracle GT-bounds diagnostics.

Temporary port of FeedForward-Benchmark PR #21 (eval_refactor_hsy, commit
d88025f4) -- ported directly rather than merging the full PR since it still
has other things to correct; remove/replace once the PR itself lands.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spatialarena.eval.depth_prediction_evaluator import DepthEvaluationConfig, DepthPairData

import numpy as np

from spatialarena.eval.huber_irls import fit_huber_irls
from spatialarena.eval.inverse_l1_gt import NONPOSITIVE_DEPTH_FLOOR, floor_nonpositive_depth


def aligned_depth_space_map(prediction: np.ndarray, scale: float, shift: float, config: DepthEvaluationConfig,
                            bounds: list[float] | None = None) -> np.ndarray:
    """One frame's depth under a depth-space fit ``scale * depth + shift`` (depth: the floored prediction, or the
    GT-bounds decode of a raw disparity), clipped to the config's depth range, as the scorer aligns it."""
    depth = prediction.astype(np.float64)
    depth = decode_with_bounds(depth, *bounds) if bounds is not None else floor_nonpositive_depth(depth)
    max_depth = config.max_depth if config.max_depth is not None else np.inf
    return np.clip(scale * depth + shift, config.min_depth, max_depth)


def decode_with_bounds(code: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Normalized disparity code -> depth between the inverse-depth bounds ``lo``/``hi``."""
    return 1.0 / np.maximum(lo + np.clip(code, 0, 1) * (hi - lo + 1e-5), 1e-8)


def decode_gt_bounds(code: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Use sequence-wide GT-valid q02/q98; not per inference-window bounds."""
    lo, hi = np.quantile(1.0 / gt, [0.02, 0.98])
    hi = max(hi, lo + 1e-8)
    return decode_with_bounds(code, lo, hi), [float(lo), float(hi)]


def score_sequence(
    pairs: list[DepthPairData], config: DepthEvaluationConfig, diagnostic_suffix: str | None = None,
    artifacts=None,
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("Cannot score an empty sequence")
    from spatialarena.eval.depth_prediction_evaluator import (
        compute_depth_metrics_from_values,
        mean_normal_error,
    )
    from spatialarena.eval.inverse_l1_gt import write_frame_artifacts

    raw = [p.pred_path.parent.name == "raw_disparities" for p in pairs]
    oracle = config.protocol.startswith("gt_bounds_")
    if any(raw) != all(raw) or all(raw) != oracle:
        raise ValueError("Protocol requires raw disparity for GT bounds, native depth otherwise")
    pred = np.concatenate([p.prediction[p.valid_mask] for p in pairs]).astype(np.float64)
    gt = np.concatenate([p.ground_truth[p.valid_mask] for p in pairs]).astype(np.float64)
    if not np.isfinite(pred).all() or not np.isfinite(gt).all() or (gt <= 0).any():
        raise ValueError("Invalid values on GT-only pixel domain")
    diagnostic = dict(
        protocol=config.protocol,
        oracle_gt_bounds=oracle,
        fit_pixels=int(len(gt)),
        score_pixels=int(len(gt)),
        representation="depth",
    )
    bounds = None
    a, b = 1.0, 0.0
    if oracle:
        pred, bounds = decode_gt_bounds(pred, gt)
        diagnostic.update(
            gt_inverse_bounds=bounds,
            bounds_domain="selected sequence GT-valid evaluation pixels",
            quantiles=[0.02, 0.98],
            code_clamp=[0, 1],
            decode_epsilon=1e-5,
            note="Sequence oracle applied to cached stitched disparity; not per-window training codec",
        )
    else:
        diagnostic.update(nonpositive_depth_floor=NONPOSITIVE_DEPTH_FLOOR,
                          floored_nonpositive_pixels=int((pred <= 0).sum()))
        pred = floor_nonpositive_depth(pred)
    if config.protocol != "gt_bounds_direct":
        a, b, fit = fit_huber_irls(pred, gt)
        # Solver is shared with inverse-space protocol; rename its unit-specific metadata.
        fit["normalization"] = "residual / median(GT depth)"
        fit["gt_depth_median"] = fit.pop("gt_inverse_median")
        fit["effective_depth_delta"] = fit.pop("effective_inverse_delta")
        for key in list(fit):
            if "inverse_residual" in key:
                fit[key.replace("inverse_residual", "depth_residual")] = fit.pop(key)
        diagnostic.update(scale=a, shift=b, optimizer=fit)
        pred = a * pred + b
    diagnostic["nonpositive_before_clip"] = int((pred <= 0).sum())
    max_depth = config.max_depth if config.max_depth is not None else np.inf
    pred = np.clip(pred, config.min_depth, max_depth)

    def aligned_depth_maps():
        """Each frame's full map under the same decode, fit and clip."""
        for pair in pairs:
            yield aligned_depth_space_map(pair.prediction, a, b, config, bounds=bounds)

    # The LiDAR-GT pass (SpatialArena) keeps its own fit records beside the dense pass's.
    stem = config.protocol + ("_lidar" if getattr(config, "gt_source", "dense") == "lidar" else "")
    diagnostic_path = pairs[0].pred_path.parent.parent / f"{stem}_fit.json"
    if diagnostic_suffix is not None:
        diagnostic_path = diagnostic_path.parent / f"{stem}_fits" / f"{diagnostic_suffix}.json"
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_path.write_text(json.dumps(diagnostic, indent=2))
    metrics = compute_depth_metrics_from_values(pred, gt, min_depth=config.min_depth) | mean_normal_error(
        pairs, aligned_depth_maps(), config)
    if artifacts is not None:
        write_frame_artifacts(artifacts, pairs, aligned_depth_maps(), a, b, metrics, config)
    return metrics
