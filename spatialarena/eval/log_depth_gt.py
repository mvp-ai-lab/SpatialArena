# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""`log_huber_irls_gt`: GT-only Huber IRLS alignment of normalized log depth.

For methods that export normalized log depth `q` (`raw_log_depths/<stem>.npz`,
`.npy` before 2026-11-17; e.g. Marigold V2's [-1, 1] code), as the FFB
releases score them (their `log_huber_irls_gt_fits/*.json` record this exact
recipe): one affine fit `a * q + b` to `log(GT depth)` over the GT-valid pixels
of the image (MonoDepth) or sequence (VideoDepth), by Huber IRLS with delta
0.05 on the absolute log-depth residual (unit 1, not a median normalization --
log depth is not a positive scale); decoded as `exp(a * q + b)` and clipped to
the configured depth range. Metrics are in depth.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

from spatialarena.eval import huber_irls

if TYPE_CHECKING:
    from spatialarena.eval import depth_prediction_evaluator

PROTOCOL = "log_huber_irls_gt"
RAW_DIR = "raw_log_depths"
RESIDUAL_UNIT = 1.0


def score_sequence(
    pairs: list[depth_prediction_evaluator.DepthPairData],
    config: depth_prediction_evaluator.DepthEvaluationConfig,
    diagnostic_suffix: str | None = None,
    artifacts=None,
) -> dict[str, Any]:
    """Fits, decodes and scores `pairs`, with one fit for all of them.

    Writes the fit record next to the predictions.

    Args:
        pairs: The prediction/GT pairs to score together.
        config: Evaluation config; sets the depth range and GT source.
        diagnostic_suffix: Per-image fit record name, or None for one record
            per sequence.
        artifacts: Optional per-frame artifact writer.

    Returns:
        The depth metrics, including the mean normal error.

    Raises:
        ValueError: On empty pairs, predictions outside raw_log_depths/ or
            invalid values.
    """
    if not pairs:
        raise ValueError("Cannot score an empty sequence")
    # Imported here: both modules import this one.
    from spatialarena.eval import depth_prediction_evaluator  # pylint: disable=import-outside-toplevel
    from spatialarena.eval import inverse_l1_gt  # pylint: disable=import-outside-toplevel

    if any(pair.pred_path.parent.name != RAW_DIR for pair in pairs):
        raise ValueError(
            f"{PROTOCOL} needs the normalized log depths in {RAW_DIR}/")
    q = np.concatenate([p.prediction[p.valid_mask] for p in pairs
                       ]).astype(np.float64)
    gt = np.concatenate([p.ground_truth[p.valid_mask] for p in pairs
                        ]).astype(np.float64)
    if gt.size < 2 or not np.isfinite(gt).all() or (gt <= 0).any():
        raise ValueError(
            "Alignment requires at least two finite positive GT values")
    if not np.isfinite(q).all():
        raise ValueError("Nonfinite predictions on GT-valid pixels")
    a, b, fit = huber_irls.fit_huber_irls(q, np.log(gt), unit=RESIDUAL_UNIT)
    fit.pop("gt_inverse_median", None)
    fit.pop("effective_inverse_delta", None)
    for key in list(fit):
        if "inverse_residual" in key:
            fit[key.replace("inverse_residual",
                            "log_depth_residual")] = fit.pop(key)
    if config.max_depth is not None:
        max_depth = config.max_depth
    else:
        max_depth = np.inf

    def decode(code: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore"):
            return np.clip(np.exp(a * code.astype(np.float64) + b),
                           config.min_depth, max_depth)

    depth = decode(q)
    diagnostic = dict(
        fit,
        protocol=PROTOCOL,
        representation="raw_normalized_log_depth",
        alignment_scope="sequence",
        residual_unit=RESIDUAL_UNIT,
        effective_delta=fit["delta"] * RESIDUAL_UNIT,
        scale=a,
        shift=b,
        fit_pixels=int(len(q)),
        score_pixels=int(len(q)),
        decode="exp(a*q+b), clipped to configured depth range",
    )

    def aligned_depth_maps():
        """Yields each frame's full map under the same fit and clip.

        Pixels off the GT domain may be nonfinite.
        """
        for pair in pairs:
            yield decode(pair.prediction)

    # The LiDAR-GT pass (SpatialArena) keeps its own fit records beside the
    # dense pass's.
    stem = PROTOCOL
    if getattr(config, "gt_source", "dense") == "lidar":
        stem += "_lidar"
    diagnostic_path = pairs[0].pred_path.parent.parent / f"{stem}_fit.json"
    if diagnostic_suffix is not None:
        diagnostic_path = (diagnostic_path.parent / f"{stem}_fits" /
                           f"{diagnostic_suffix}.json")
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_path.write_text(json.dumps(diagnostic, indent=2))
    evaluator = depth_prediction_evaluator
    depth_metrics = evaluator.compute_depth_metrics_from_values(
        depth, gt, min_depth=config.min_depth)
    normal_metrics = evaluator.mean_normal_error(pairs, aligned_depth_maps(),
                                                 config)
    metrics = depth_metrics | normal_metrics
    if artifacts is not None:
        inverse_l1_gt.write_frame_artifacts(artifacts, pairs,
                                            aligned_depth_maps(), a, b, metrics,
                                            config)
    return metrics
