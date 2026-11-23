# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Sequence depth-space alignment and explicitly oracle GT-bounds diagnostics.

Temporary port of FeedForward-Benchmark PR #21 (eval_refactor_hsy, commit
d88025f4) -- ported directly rather than merging the full PR since it still
has other things to correct; remove/replace once the PR itself lands.

`inverse_l1_gt.score_sequence` delegates the depth-space protocols
(protocol names without "inverse") to `score_sequence` here.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spatialarena.eval import depth_prediction_evaluator

import numpy as np

from spatialarena.eval import huber_irls
from spatialarena.eval import inverse_l1_gt


def aligned_depth_space_map(
        prediction: np.ndarray,
        scale: float,
        shift: float,
        config: depth_prediction_evaluator.DepthEvaluationConfig,
        bounds: list[float] | None = None) -> np.ndarray:
    """Returns one frame's depth under a depth-space fit, as the scorer has it.

    The fit is `scale * depth + shift`, clipped to the config's depth range.

    Args:
        prediction: The frame's prediction: depth, or a raw disparity code
            when bounds is given.
        scale: Fitted scale.
        shift: Fitted shift.
        config: Depth evaluation config (min_depth, max_depth).
        bounds: The GT inverse-depth bounds that decode a raw disparity;
            None floors the non-positive depths of the prediction instead.

    Returns:
        The aligned depth map.
    """
    depth = prediction.astype(np.float64)
    if bounds is not None:
        depth = decode_with_bounds(depth, *bounds)
    else:
        depth = inverse_l1_gt.floor_nonpositive_depth(depth)
    max_depth = config.max_depth if config.max_depth is not None else np.inf
    return np.clip(scale * depth + shift, config.min_depth, max_depth)


def decode_with_bounds(code: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Decodes a normalized disparity code to depth between `lo` and `hi`.

    Args:
        code: Normalized disparity code; clamped to [0, 1].
        lo: Lower inverse-depth bound.
        hi: Upper inverse-depth bound.

    Returns:
        The depth.
    """
    return 1.0 / np.maximum(lo + np.clip(code, 0, 1) * (hi - lo + 1e-5), 1e-8)


def decode_gt_bounds(code: np.ndarray,
                     gt: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Decodes a disparity code with the GT inverse-depth q02/q98 bounds.

    Uses sequence-wide GT-valid q02/q98; not per inference-window bounds.

    Args:
        code: Normalized disparity code of the GT-valid pixels.
        gt: GT depth of the same pixels.

    Returns:
        The decoded depth and the [lo, hi] inverse-depth bounds.
    """
    lo, hi = np.quantile(1.0 / gt, [0.02, 0.98])
    hi = max(hi, lo + 1e-8)
    return decode_with_bounds(code, lo, hi), [float(lo), float(hi)]


def score_sequence(
    pairs: list[depth_prediction_evaluator.DepthPairData],
    config: depth_prediction_evaluator.DepthEvaluationConfig,
    diagnostic_suffix: str | None = None,
    artifacts=None,
) -> dict[str, Any]:
    """Fits and scores one sequence with a depth-space protocol.

    The fit record is written to <protocol>[_lidar]_fit.json next to the
    prediction directory, or to <protocol>[_lidar]_fits/<suffix>.json.

    Args:
        pairs: The sequence's prediction/GT pairs.
        config: Depth evaluation config; config.protocol selects the
            protocol ("gt_bounds_*" decodes raw disparities with GT bounds;
            "gt_bounds_direct" skips the Huber fit).
        diagnostic_suffix: Optional name of a per-window fit record.
        artifacts: Optional per-frame artifact writer.

    Returns:
        The depth metrics and the mean normal error.

    Raises:
        ValueError: The sequence is empty, the prediction kind does not match
            the protocol, or the GT-valid values are invalid.
    """
    if not pairs:
        raise ValueError("Cannot score an empty sequence")
    # Imported here: depth_prediction_evaluator imports this module.
    # pylint: disable-next=import-outside-toplevel
    from spatialarena.eval import depth_prediction_evaluator

    raw = [pair.pred_path.parent.name == "raw_disparities" for pair in pairs]
    oracle = config.protocol.startswith("gt_bounds_")
    if any(raw) != all(raw) or all(raw) != oracle:
        raise ValueError("Protocol requires raw disparity for GT bounds, "
                         "native depth otherwise")
    pred = np.concatenate([pair.prediction[pair.valid_mask] for pair in pairs
                          ]).astype(np.float64)
    gt = np.concatenate([pair.ground_truth[pair.valid_mask] for pair in pairs
                        ]).astype(np.float64)
    if (not np.isfinite(pred).all() or not np.isfinite(gt).all() or
        (gt <= 0).any()):
        raise ValueError("Invalid values on GT-only pixel domain")
    diagnostic = dict(
        protocol=config.protocol,
        oracle_gt_bounds=oracle,
        fit_pixels=int(len(gt)),
        score_pixels=int(len(gt)),
        representation="depth",
    )
    bounds = None
    scale, shift = 1.0, 0.0
    if oracle:
        pred, bounds = decode_gt_bounds(pred, gt)
        diagnostic.update(
            gt_inverse_bounds=bounds,
            bounds_domain="selected sequence GT-valid evaluation pixels",
            quantiles=[0.02, 0.98],
            code_clamp=[0, 1],
            decode_epsilon=1e-5,
            note=("Sequence oracle applied to cached stitched disparity; "
                  "not per-window training codec"),
        )
    else:
        diagnostic.update(
            nonpositive_depth_floor=inverse_l1_gt.NONPOSITIVE_DEPTH_FLOOR,
            floored_nonpositive_pixels=int((pred <= 0).sum()))
        pred = inverse_l1_gt.floor_nonpositive_depth(pred)
    if config.protocol != "gt_bounds_direct":
        scale, shift, fit = huber_irls.fit_huber_irls(pred, gt)
        # The solver is shared with the inverse-space protocol; rename its
        # unit-specific metadata.
        fit["normalization"] = "residual / median(GT depth)"
        fit["gt_depth_median"] = fit.pop("gt_inverse_median")
        fit["effective_depth_delta"] = fit.pop("effective_inverse_delta")
        for key in list(fit):
            if "inverse_residual" in key:
                fit[key.replace("inverse_residual",
                                "depth_residual")] = fit.pop(key)
        diagnostic.update(scale=scale, shift=shift, optimizer=fit)
        pred = scale * pred + shift
    diagnostic["nonpositive_before_clip"] = int((pred <= 0).sum())
    max_depth = config.max_depth if config.max_depth is not None else np.inf
    pred = np.clip(pred, config.min_depth, max_depth)

    def aligned_depth_maps():
        """Each frame's full map under the same decode, fit and clip."""
        for pair in pairs:
            yield aligned_depth_space_map(pair.prediction,
                                          scale,
                                          shift,
                                          config,
                                          bounds=bounds)

    # The LiDAR-GT pass (SpatialArena) keeps its own fit records beside the
    # dense pass's.
    stem = config.protocol
    if getattr(config, "gt_source", "dense") == "lidar":
        stem = stem + "_lidar"
    diagnostic_path = pairs[0].pred_path.parent.parent / f"{stem}_fit.json"
    if diagnostic_suffix is not None:
        diagnostic_path = (diagnostic_path.parent / f"{stem}_fits" /
                           f"{diagnostic_suffix}.json")
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_path.write_text(json.dumps(diagnostic, indent=2))
    depth_metrics = (
        depth_prediction_evaluator.compute_depth_metrics_from_values(
            pred, gt, min_depth=config.min_depth))
    normal_metrics = depth_prediction_evaluator.mean_normal_error(
        pairs, aligned_depth_maps(), config)
    metrics = depth_metrics | normal_metrics
    if artifacts is not None:
        inverse_l1_gt.write_frame_artifacts(artifacts, pairs,
                                            aligned_depth_maps(), scale, shift,
                                            metrics, config)
    return metrics
