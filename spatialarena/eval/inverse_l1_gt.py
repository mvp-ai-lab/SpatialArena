# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Opt-in sequence inverse-depth L1 Adam, using the GT-only pixel domain.

Temporary port of FeedForward-Benchmark PR #21 (eval_refactor_hsy, commit
d88025f4) -- ported directly rather than merging the full PR since it still
has other things to correct; remove/replace once the PR itself lands.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from spatialarena.eval import depth_prediction_evaluator

# Native depth at or below zero (e.g. MapAnything's ray-depth on some ScanNet
# pixels) is floored here before any reciprocal or fit, rather than failing the
# scene; the count lands in the fit diagnostic.
NONPOSITIVE_DEPTH_FLOOR = 1e-5


def floor_nonpositive_depth(depth: np.ndarray) -> np.ndarray:
    """Raises values of `depth` at or below zero to NONPOSITIVE_DEPTH_FLOOR.

    NaN and inf pass through.
    """
    return np.where(depth <= 0, NONPOSITIVE_DEPTH_FLOOR, depth)


def fit_adam(
        x: np.ndarray,
        y: np.ndarray,
        device: str | torch.device | None = None,
        initialization: str = "median") -> tuple[float, float, dict[str, Any]]:
    """Fits scale * x + shift to y under an L1 loss with Adam.

    Args:
        x: Predicted inverse-depth values.
        y: GT inverse-depth values, paired with `x`.
        device: Torch device; defaults to CUDA when available.
        initialization: "median" (scale = median(y) / median(x), shift = 0)
            or "l2" (the least-squares fit).

    Returns:
        (scale, shift, diagnostic dict).

    Raises:
        ValueError: On invalid inputs, an unknown initialization or a
            degenerate median.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 2 or x.shape != y.shape or not np.isfinite(
            x).all() or not np.isfinite(y).all():
        raise ValueError("Alignment requires at least two finite paired values")
    initial_ab = None
    if initialization == "l2":
        xn = np.asarray(x, dtype=np.float64).reshape(-1)
        yn = np.asarray(y, dtype=np.float64).reshape(-1)
        if len(xn) < 2 or xn.shape != yn.shape or not np.isfinite(
                xn).all() or not np.isfinite(yn).all():
            raise ValueError("Invalid L2 initialization inputs")
        initial_ab = np.linalg.lstsq(np.column_stack([xn, np.ones_like(xn)]),
                                     yn,
                                     rcond=None)[0]
    elif initialization != "median":
        raise ValueError("Unknown Adam initialization")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.float32, device=device)
    if initial_ab is None:
        median_x = torch.median(x)
        if median_x.abs() < 1e-12:
            raise ValueError("Degenerate median for Adam initialization")
        scale_param = (torch.median(y) /
                       median_x).detach().reshape(1).requires_grad_(True)
        shift_param = torch.zeros(1,
                                  dtype=x.dtype,
                                  device=device,
                                  requires_grad=True)
    else:
        scale_param = torch.tensor([initial_ab[0]],
                                   dtype=x.dtype,
                                   device=device,
                                   requires_grad=True)
        shift_param = torch.tensor([initial_ab[1]],
                                   dtype=x.dtype,
                                   device=device,
                                   requires_grad=True)
    initial_scale, initial_shift = scale_param.item(), shift_param.item()
    optimizer = torch.optim.Adam([scale_param, shift_param], lr=1e-4)
    initial = float((scale_param * x + shift_param - y).abs().mean().detach())
    prev = None
    stop = "max_iters"
    trace = []
    for i in range(1000):
        optimizer.zero_grad()
        loss = (scale_param * x + shift_param - y).abs().sum()
        value = loss.item()
        if i % 100 == 0:
            trace.append(
                dict(step=i, mean_absolute_inverse_residual=value / x.numel()))
        loss.backward()
        optimizer.step()
        if prev is not None and abs(prev - value) < 1e-6:
            stop = "absolute_loss_tolerance"
            break
        prev = value
    final = float((scale_param * x + shift_param - y).abs().mean().detach())
    return (
        scale_param.item(),
        shift_param.item(),
        dict(
            name="Adam",
            initialization=initialization,
            initial_scale=initial_scale,
            initial_shift=initial_shift,
            lr=1e-4,
            max_iters=1000,
            tol=1e-6,
            steps=i + 1,
            stop=stop,
            initial_mean_absolute_inverse_residual=initial,
            final_mean_absolute_inverse_residual=final,
            trace=trace,
        ),
    )


def aligned_inverse_depth_map(
        prediction: np.ndarray, scale: float, shift: float, *, raw: bool,
        inverse_floor: float,
        config: depth_prediction_evaluator.DepthEvaluationConfig) -> np.ndarray:
    """Returns one frame's depth under an inverse-space fit.

    The fit is `scale * q + shift` (q: raw disparity, or 1 / floored depth),
    floored at `inverse_floor` and clipped to the config's depth range, as the
    scorer aligns it.
    """
    if config.max_depth is not None:
        max_depth = config.max_depth
    else:
        max_depth = np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        if raw:
            q_map = prediction
        else:
            q_map = 1.0 / floor_nonpositive_depth(prediction)
        return np.clip(1.0 / np.maximum(scale * q_map + shift, inverse_floor),
                       config.min_depth, max_depth)


def write_frame_artifacts(
        artifacts, pairs: list[depth_prediction_evaluator.DepthPairData],
        aligned_maps, scale: float, shift: float, metrics: dict[str, Any],
        config: depth_prediction_evaluator.DepthEvaluationConfig) -> None:
    """Hands each frame's aligned map to the per-frame writers.

    The writers produce error maps and dashboard viewer images. `scale` and
    `shift` are the protocol's own fit (inverse-depth space for the inverse
    protocols). A single frame reports the scored metrics; a sequence's
    per-frame metrics are for inspection only, as on the standard path.
    """
    # Imported here: depth_prediction_evaluator imports this module.
    from spatialarena.eval import depth_prediction_evaluator

    for pair, aligned in zip(pairs, aligned_maps, strict=True):
        if len(pairs) == 1:
            frame_metrics = metrics
        else:
            frame_metrics = depth_prediction_evaluator.compute_depth_metrics(
                aligned,
                pair.ground_truth,
                pair.valid_mask,
                min_depth=config.min_depth)
        artifacts.write(pair, aligned, scale, shift, frame_metrics)


def score_sequence(
    pairs: list[depth_prediction_evaluator.DepthPairData],
    config: depth_prediction_evaluator.DepthEvaluationConfig,
    diagnostic_suffix: str | None = None,
    artifacts=None,
) -> dict[str, Any]:
    """Fits, aligns and scores `pairs` under `config.protocol`.

    One fit covers all pairs (the image for MonoDepth, the sequence for
    VideoDepth). The log-depth and depth-space protocols are delegated to
    `log_depth_gt` and `depth_space_gt`. The fit record is written next to the
    predictions.

    Args:
        pairs: The prediction/GT pairs to score together.
        config: Evaluation config; selects the protocol and depth range.
        diagnostic_suffix: Per-image fit record name, or None for one record
            per sequence.
        artifacts: Optional per-frame artifact writer.

    Returns:
        The depth metrics, including the mean normal error.

    Raises:
        ValueError: On empty or invalid inputs or settings.
    """
    if not pairs:
        raise ValueError("Cannot score an empty sequence")
    if config.protocol == "log_huber_irls_gt":
        from spatialarena.eval import log_depth_gt

        return log_depth_gt.score_sequence(pairs,
                                           config,
                                           diagnostic_suffix=diagnostic_suffix,
                                           artifacts=artifacts)
    if config.protocol in {
            "depth_huber_irls_gt", "gt_bounds_direct",
            "gt_bounds_depth_huber_irls"
    }:
        from spatialarena.eval import depth_space_gt

        return depth_space_gt.score_sequence(
            pairs,
            config,
            diagnostic_suffix=diagnostic_suffix,
            artifacts=artifacts)
    # Imported here: depth_prediction_evaluator imports this module.
    from spatialarena.eval import depth_prediction_evaluator

    raw = [p.pred_path.parent.name == "raw_disparities" for p in pairs]
    if any(raw) and not all(raw):
        raise ValueError("Mixed raw disparity and depth representations")
    pred = np.concatenate([p.prediction[p.valid_mask] for p in pairs
                          ]).astype(np.float64)
    gt = np.concatenate([p.ground_truth[p.valid_mask] for p in pairs
                        ]).astype(np.float64)
    if gt.size < 2 or not np.isfinite(gt).all() or (gt <= 0).any():
        raise ValueError(
            "Alignment requires at least two finite positive GT values")
    if not np.isfinite(pred).all():
        raise ValueError("Nonfinite predictions on GT-valid pixels")
    if all(raw):
        floored_pixels = 0
        q = pred
    else:
        floored_pixels = int((pred <= 0).sum())
        q = 1.0 / floor_nonpositive_depth(pred)
    if config.protocol == "inverse_huber_irls_gt":
        from spatialarena.eval import huber_irls

        a, b, diagnostic = huber_irls.fit_huber_irls(q, 1.0 / gt)
    else:
        if config.protocol == "inverse_l1_l2init_gt":
            initialization = "l2"
        else:
            initialization = "median"
        a, b, diagnostic = fit_adam(q, 1.0 / gt, initialization=initialization)
    aligned_q = a * q + b
    policy = getattr(config, "inverse_floor_policy", "fixed")
    if policy not in {"fixed", "gt_max"}:
        raise ValueError(f"Unknown inverse floor policy: {policy}")
    if policy == "gt_max" and (len(pairs) != 1 or diagnostic_suffix is None):
        raise ValueError(
            "gt_max floor currently supports per-image MonoDepth only")
    if policy == "gt_max":
        inverse_floor = 1.0 / float(gt.max())
    else:
        inverse_floor = 1e-3
    if config.max_depth is not None:
        max_depth = config.max_depth
    else:
        max_depth = np.inf
    depth = np.clip(1.0 / np.maximum(aligned_q, inverse_floor),
                    config.min_depth, max_depth)

    def aligned_depth_maps():
        """Yields each frame's full map under the same fit, floor and clip.

        Pixels off the GT domain may be nonfinite.
        """
        for pair in pairs:
            yield aligned_inverse_depth_map(pair.prediction,
                                            a,
                                            b,
                                            raw=all(raw),
                                            inverse_floor=inverse_floor,
                                            config=config)

    if all(raw):
        representation = "raw_disparity"
    else:
        representation = "reciprocal_depth"
    diagnostic.update(
        scale=a,
        shift=b,
        fit_pixels=int(len(q)),
        score_pixels=int(len(q)),
        representation=representation,
        nonpositive_depth_floor=NONPOSITIVE_DEPTH_FLOOR,
        floored_nonpositive_pixels=floored_pixels,
        inverse_floor=inverse_floor,
        inverse_floor_policy=policy,
        clipped_aligned_pixels=int((aligned_q < inverse_floor).sum()),
        nonpositive_aligned_pixels=int((aligned_q <= 0).sum()),
        protocol=config.protocol,
    )
    # The LiDAR-GT pass (SpatialArena) keeps its own fit records beside the
    # dense pass's.
    stem = config.protocol
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
        write_frame_artifacts(artifacts, pairs, aligned_depth_maps(), a, b,
                              metrics, config)
    return metrics
