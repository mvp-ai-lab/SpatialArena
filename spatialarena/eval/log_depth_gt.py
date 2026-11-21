# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""``log_huber_irls_gt``: GT-only Huber IRLS alignment of a normalized log-depth prediction.

For methods that export normalized log depth ``q`` (``raw_log_depths/<stem>.npz`` (``.npy`` before 2026-11-17), e.g.
Marigold V2's [-1, 1] code),
as the FFB releases score them (their ``log_huber_irls_gt_fits/*.json`` record this exact recipe): one affine fit
``a * q + b`` to ``log(GT depth)`` over the GT-valid pixels of the image (MonoDepth) or sequence (VideoDepth), by
Huber IRLS with delta 0.05 on the absolute log-depth residual (unit 1, not a median normalization -- log depth is
not a positive scale); decoded as ``exp(a * q + b)`` and clipped to the configured depth range. Metrics are in depth.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

from spatialarena.eval.huber_irls import fit_huber_irls

if TYPE_CHECKING:
    from spatialarena.eval.depth_prediction_evaluator import DepthEvaluationConfig, DepthPairData

PROTOCOL = "log_huber_irls_gt"
RAW_DIR = "raw_log_depths"
RESIDUAL_UNIT = 1.0


def score_sequence(
    pairs: list[DepthPairData], config: DepthEvaluationConfig, diagnostic_suffix: str | None = None,
    artifacts=None,
) -> dict[str, Any]:
    """Fit, decode and score ``pairs`` (one fit for all of them); writes the fit record next to the predictions."""
    if not pairs:
        raise ValueError("Cannot score an empty sequence")
    from spatialarena.eval.depth_prediction_evaluator import (  # pylint: disable=import-outside-toplevel
        compute_depth_metrics_from_values,
        mean_normal_error,
    )
    from spatialarena.eval.inverse_l1_gt import write_frame_artifacts  # pylint: disable=import-outside-toplevel

    if any(pair.pred_path.parent.name != RAW_DIR for pair in pairs):
        raise ValueError(f"{PROTOCOL} needs the normalized log depths in {RAW_DIR}/")
    q = np.concatenate([p.prediction[p.valid_mask] for p in pairs]).astype(np.float64)
    gt = np.concatenate([p.ground_truth[p.valid_mask] for p in pairs]).astype(np.float64)
    if gt.size < 2 or not np.isfinite(gt).all() or (gt <= 0).any():
        raise ValueError("Alignment requires at least two finite positive GT values")
    if not np.isfinite(q).all():
        raise ValueError("Nonfinite predictions on GT-valid pixels")
    a, b, fit = fit_huber_irls(q, np.log(gt), unit=RESIDUAL_UNIT)
    fit.pop("gt_inverse_median", None)
    fit.pop("effective_inverse_delta", None)
    for key in list(fit):
        if "inverse_residual" in key:
            fit[key.replace("inverse_residual", "log_depth_residual")] = fit.pop(key)
    max_depth = config.max_depth if config.max_depth is not None else np.inf

    def decode(code: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore"):
            return np.clip(np.exp(a * code.astype(np.float64) + b), config.min_depth, max_depth)

    depth = decode(q)
    diagnostic = dict(
        fit, protocol=PROTOCOL, representation="raw_normalized_log_depth",
        alignment_scope="sequence", residual_unit=RESIDUAL_UNIT, effective_delta=fit["delta"] * RESIDUAL_UNIT,
        scale=a, shift=b, fit_pixels=int(len(q)), score_pixels=int(len(q)),
        decode="exp(a*q+b), clipped to configured depth range",
    )

    def aligned_depth_maps():
        """Each frame's full map under the same fit and clip (pixels off the GT domain may be nonfinite)."""
        for pair in pairs:
            yield decode(pair.prediction)

    # The LiDAR-GT pass (SpatialArena) keeps its own fit records beside the dense pass's.
    stem = PROTOCOL + ("_lidar" if getattr(config, "gt_source", "dense") == "lidar" else "")
    diagnostic_path = pairs[0].pred_path.parent.parent / f"{stem}_fit.json"
    if diagnostic_suffix is not None:
        diagnostic_path = diagnostic_path.parent / f"{stem}_fits" / f"{diagnostic_suffix}.json"
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_path.write_text(json.dumps(diagnostic, indent=2))
    metrics = compute_depth_metrics_from_values(depth, gt, min_depth=config.min_depth) | mean_normal_error(
        pairs, aligned_depth_maps(), config)
    if artifacts is not None:
        write_frame_artifacts(artifacts, pairs, aligned_depth_maps(), a, b, metrics, config)
    return metrics
