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

if TYPE_CHECKING:
    from spatialarena.eval.depth_prediction_evaluator import DepthEvaluationConfig, DepthPairData

import numpy as np
import torch

# Native depth at or below zero (e.g. MapAnything's ray-depth on some ScanNet pixels) is floored here before any
# reciprocal or fit, rather than failing the scene; the count lands in the fit diagnostic.
NONPOSITIVE_DEPTH_FLOOR = 1e-5


def floor_nonpositive_depth(depth: np.ndarray) -> np.ndarray:
    """``depth`` with values at or below zero raised to NONPOSITIVE_DEPTH_FLOOR (NaN/inf pass through)."""
    return np.where(depth <= 0, NONPOSITIVE_DEPTH_FLOOR, depth)


def fit_adam(
    x: np.ndarray, y: np.ndarray, device: str | torch.device | None = None, initialization: str = "median"
) -> tuple[float, float, dict[str, Any]]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 2 or x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Alignment requires at least two finite paired values")
    initial_ab = None
    if initialization == "l2":
        xn = np.asarray(x, dtype=np.float64).reshape(-1)
        yn = np.asarray(y, dtype=np.float64).reshape(-1)
        if len(xn) < 2 or xn.shape != yn.shape or not np.isfinite(xn).all() or not np.isfinite(yn).all():
            raise ValueError("Invalid L2 initialization inputs")
        initial_ab = np.linalg.lstsq(np.column_stack([xn, np.ones_like(xn)]), yn, rcond=None)[0]
    elif initialization != "median":
        raise ValueError("Unknown Adam initialization")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.float32, device=device)
    if initial_ab is None:
        med = torch.median(x)
        if med.abs() < 1e-12:
            raise ValueError("Degenerate median for Adam initialization")
        s = (torch.median(y) / med).detach().reshape(1).requires_grad_(True)
        t = torch.zeros(1, dtype=x.dtype, device=device, requires_grad=True)
    else:
        s = torch.tensor([initial_ab[0]], dtype=x.dtype, device=device, requires_grad=True)
        t = torch.tensor([initial_ab[1]], dtype=x.dtype, device=device, requires_grad=True)
    initial_scale, initial_shift = s.item(), t.item()
    opt = torch.optim.Adam([s, t], lr=1e-4)
    initial = float((s * x + t - y).abs().mean().detach())
    prev = None
    stop = "max_iters"
    trace = []
    for i in range(1000):
        opt.zero_grad()
        loss = (s * x + t - y).abs().sum()
        value = loss.item()
        if i % 100 == 0:
            trace.append(dict(step=i, mean_absolute_inverse_residual=value / x.numel()))
        loss.backward()
        opt.step()
        if prev is not None and abs(prev - value) < 1e-6:
            stop = "absolute_loss_tolerance"
            break
        prev = value
    final = float((s * x + t - y).abs().mean().detach())
    return (
        s.item(), t.item(),
        dict(
            name="Adam", initialization=initialization, initial_scale=initial_scale, initial_shift=initial_shift,
            lr=1e-4, max_iters=1000, tol=1e-6, steps=i + 1, stop=stop,
            initial_mean_absolute_inverse_residual=initial, final_mean_absolute_inverse_residual=final, trace=trace,
        ),
    )


def aligned_inverse_depth_map(prediction: np.ndarray, scale: float, shift: float, *, raw: bool,
                              inverse_floor: float, config: DepthEvaluationConfig) -> np.ndarray:
    """One frame's depth under an inverse-space fit ``scale * q + shift`` (q: raw disparity, or 1 / floored depth),
    floored at ``inverse_floor`` and clipped to the config's depth range, as the scorer aligns it."""
    max_depth = config.max_depth if config.max_depth is not None else np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        q_map = prediction if raw else 1.0 / floor_nonpositive_depth(prediction)
        return np.clip(1.0 / np.maximum(scale * q_map + shift, inverse_floor), config.min_depth, max_depth)


def write_frame_artifacts(artifacts, pairs: list[DepthPairData], aligned_maps, scale: float, shift: float,
                          metrics: dict[str, Any], config: DepthEvaluationConfig) -> None:
    """Hand each frame's aligned map to the per-frame writers (error maps, dashboard viewer images).

    ``scale``/``shift`` are the protocol's own fit (inverse-depth space for the inverse protocols). A single frame
    reports the scored metrics; a sequence's per-frame metrics are for inspection only, as on the standard path.
    """
    from spatialarena.eval.depth_prediction_evaluator import compute_depth_metrics

    for pair, aligned in zip(pairs, aligned_maps, strict=True):
        frame_metrics = metrics if len(pairs) == 1 else compute_depth_metrics(
            aligned, pair.ground_truth, pair.valid_mask, min_depth=config.min_depth)
        artifacts.write(pair, aligned, scale, shift, frame_metrics)


def score_sequence(
    pairs: list[DepthPairData], config: DepthEvaluationConfig, diagnostic_suffix: str | None = None,
    artifacts=None,
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("Cannot score an empty sequence")
    if config.protocol == "log_huber_irls_gt":
        from spatialarena.eval.log_depth_gt import score_sequence as score_log_depth

        return score_log_depth(pairs, config, diagnostic_suffix=diagnostic_suffix, artifacts=artifacts)
    if config.protocol in {"depth_huber_irls_gt", "gt_bounds_direct", "gt_bounds_depth_huber_irls"}:
        from spatialarena.eval.depth_space_gt import score_sequence as score_depth

        return score_depth(pairs, config, diagnostic_suffix=diagnostic_suffix, artifacts=artifacts)
    from spatialarena.eval.depth_prediction_evaluator import (
        compute_depth_metrics_from_values,
        mean_normal_error,
    )

    raw = [p.pred_path.parent.name == "raw_disparities" for p in pairs]
    if any(raw) and not all(raw):
        raise ValueError("Mixed raw disparity and depth representations")
    pred = np.concatenate([p.prediction[p.valid_mask] for p in pairs]).astype(np.float64)
    gt = np.concatenate([p.ground_truth[p.valid_mask] for p in pairs]).astype(np.float64)
    if gt.size < 2 or not np.isfinite(gt).all() or (gt <= 0).any():
        raise ValueError("Alignment requires at least two finite positive GT values")
    if not np.isfinite(pred).all():
        raise ValueError("Nonfinite predictions on GT-valid pixels")
    floored_pixels = 0 if all(raw) else int((pred <= 0).sum())
    q = pred if all(raw) else 1.0 / floor_nonpositive_depth(pred)
    if config.protocol == "inverse_huber_irls_gt":
        from spatialarena.eval.huber_irls import fit_huber_irls

        a, b, diagnostic = fit_huber_irls(q, 1.0 / gt)
    else:
        a, b, diagnostic = fit_adam(
            q, 1.0 / gt, initialization="l2" if config.protocol == "inverse_l1_l2init_gt" else "median"
        )
    aligned_q = a * q + b
    policy = getattr(config, "inverse_floor_policy", "fixed")
    if policy not in {"fixed", "gt_max"}:
        raise ValueError(f"Unknown inverse floor policy: {policy}")
    if policy == "gt_max" and (len(pairs) != 1 or diagnostic_suffix is None):
        raise ValueError("gt_max floor currently supports per-image MonoDepth only")
    inverse_floor = 1.0 / float(gt.max()) if policy == "gt_max" else 1e-3
    max_depth = config.max_depth if config.max_depth is not None else np.inf
    depth = np.clip(1.0 / np.maximum(aligned_q, inverse_floor), config.min_depth, max_depth)

    def aligned_depth_maps():
        """Each frame's full map under the same fit, floor and clip (pixels off the GT domain may be nonfinite)."""
        for pair in pairs:
            yield aligned_inverse_depth_map(pair.prediction, a, b, raw=all(raw), inverse_floor=inverse_floor,
                                            config=config)

    diagnostic.update(
        scale=a, shift=b, fit_pixels=int(len(q)), score_pixels=int(len(q)),
        representation="raw_disparity" if all(raw) else "reciprocal_depth",
        nonpositive_depth_floor=NONPOSITIVE_DEPTH_FLOOR, floored_nonpositive_pixels=floored_pixels,
        inverse_floor=inverse_floor, inverse_floor_policy=policy,
        clipped_aligned_pixels=int((aligned_q < inverse_floor).sum()),
        nonpositive_aligned_pixels=int((aligned_q <= 0).sum()),
        protocol=config.protocol,
    )
    # The LiDAR-GT pass (SpatialArena) keeps its own fit records beside the dense pass's.
    stem = config.protocol + ("_lidar" if getattr(config, "gt_source", "dense") == "lidar" else "")
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
