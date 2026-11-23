# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Full-pixel affine Huber fit by IRLS, without prediction-dependent exclusion.

Temporary port of FeedForward-Benchmark PR #21 (eval_refactor_hsy, commit
d88025f4) -- ported directly rather than merging the full PR since it still
has other things to correct; remove/replace once the PR itself lands.
"""

from typing import Any

import numpy as np
import torch


def _fit_constant_offset(target: torch.Tensor, unit: torch.Tensor, delta: float,
                         max_iters: int,
                         tol: float) -> tuple[float, float, dict[str, Any]]:
    """Estimates the Huber-IRLS location of `target` (GT / median(GT)).

    Returns:
        (0, offset in GT units, diagnostic dict).
    """

    def objective(residual):
        e = residual.abs()
        return torch.where(e <= delta, 0.5 * e.square(),
                           delta * (e - 0.5 * delta)).mean()

    shift = target.mean()
    initial = objective(shift - target).item()
    trace = [dict(step=0, normalized_huber=initial)]
    stop = "max_iters"
    for step in range(1, max_iters + 1):
        residual = shift - target
        weight = (delta / residual.abs().clamp_min(
            torch.finfo(target.dtype).tiny)).clamp_max(1.0)
        next_shift = (weight * target).sum() / weight.sum()
        change = (next_shift - shift).abs()
        shift = next_shift
        trace.append(
            dict(step=step, normalized_huber=objective(shift - target).item()))
        if change <= tol * (1 + shift.abs()):
            stop = "parameter_tolerance"
            break
    diagnostic = dict(
        name="IRLS",
        objective="Huber",
        initialization="mean",
        delta=delta,
        constant_prediction=True,
        note="constant prediction: offset-only fit (scale fixed at 0)",
        normalization="residual / median(GT inverse depth)",
        gt_inverse_median=unit.item(),
        effective_inverse_delta=(unit * delta).item(),
        max_iters=max_iters,
        tol=tol,
        steps=len(trace) - 1,
        stop=stop,
        dtype="float64",
        trace=trace,
        initial_normalized_huber=initial,
        final_normalized_huber=trace[-1]["normalized_huber"],
        initial_mean_absolute_inverse_residual=(
            (target.mean() - target).abs().mean() * unit).item(),
        final_mean_absolute_inverse_residual=((shift - target).abs().mean() *
                                              unit).item(),
    )
    return 0.0, (unit * shift).item(), diagnostic


def fit_huber_irls(
    x: np.ndarray,
    y: np.ndarray,
    device: str | torch.device | None = None,
    delta: float = 0.05,
    max_iters: int = 100,
    tol: float = 1e-7,
    unit: float | None = None,
) -> tuple[float, float, dict[str, Any]]:
    """Fits a * x + b to y with a Huber loss, by IRLS.

    Huber residuals are divided by median(y), or by `unit` when given. An
    explicit unit is for targets that are not positive scales, e.g. log depth
    (`log_huber_irls_gt`: unit 1, so delta is an absolute log-depth residual).
    x is centered and standardized internally for numerical conditioning only.
    There are no constraints on a and b. The initial parameters are the
    full-pixel L2 solution.

    Args:
        x: Predicted values.
        y: Target values, paired with `x`.
        device: Torch device; defaults to CUDA when available.
        delta: Huber threshold on the normalized residual.
        max_iters: Maximum number of IRLS iterations.
        tol: Relative parameter-change tolerance for stopping.
        unit: Residual normalization; defaults to median(y).

    Returns:
        (a, b, diagnostic dict).

    Raises:
        ValueError: On non-finite or too few values, invalid settings, a
            non-positive unit or a degenerate weighted fit.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.as_tensor(x, dtype=torch.float64, device=device).flatten()
    y = torch.as_tensor(y, dtype=torch.float64, device=device).flatten()
    if x.numel() < 2 or x.shape != y.shape or not torch.isfinite(
            x).all() or not torch.isfinite(y).all():
        raise ValueError("Need finite paired fitting values")
    if delta <= 0 or max_iters < 1 or tol <= 0:
        raise ValueError("Invalid IRLS settings")
    explicit_unit = unit is not None
    if explicit_unit:
        unit = torch.as_tensor(float(unit), dtype=torch.float64, device=device)
    else:
        unit = torch.median(y)
    if unit <= 0:
        if not explicit_unit:
            raise ValueError("GT inverse-depth median must be positive")
        raise ValueError("unit must be positive")
    center = x.mean()
    spread = x.std(unbiased=False)
    target = y / unit
    # A spread of a few ulp is rounding in mean/std, not structure: identical
    # values (max == min) also count.
    if x.max() == x.min() or spread <= torch.finfo(x.dtype).eps * torch.maximum(
            center.abs(), center.new_tensor(1.0)):
        # A constant prediction carries no scale information: fit the offset
        # alone (a = 0) under the same Huber objective, i.e. score the frame as
        # the best single value for it. The method keeps the full penalty for a
        # flat answer, and the rest of its frames are still scored instead of
        # the whole scene failing.
        return _fit_constant_offset(target, unit, delta, max_iters, tol)
    z = (x - center) / spread
    slope = (z * target).mean()
    shift = target.mean()

    def objective(residual):
        e = residual.abs()
        return torch.where(e <= delta, 0.5 * e.square(),
                           delta * (e - 0.5 * delta)).mean()

    residual = slope * z + shift - target
    initial = objective(residual).item()
    initial_mae = (residual.abs().mean() * unit).item()
    trace = [dict(step=0, normalized_huber=initial)]
    stop = "max_iters"
    for step in range(1, max_iters + 1):
        weight = (
            delta /
            residual.abs().clamp_min(torch.finfo(x.dtype).tiny)).clamp_max(1.0)
        weight_sum = weight.sum()
        zm = (weight * z).sum() / weight_sum
        ym = (weight * target).sum() / weight_sum
        dz = z - zm
        denominator = (weight * dz.square()).sum()
        if denominator <= torch.finfo(x.dtype).tiny:
            raise ValueError("Degenerate weighted affine fit")
        next_slope = (weight * dz * (target - ym)).sum() / denominator
        next_shift = ym - next_slope * zm
        change = torch.maximum((next_slope - slope).abs(),
                               (next_shift - shift).abs())
        magnitude = torch.maximum(slope.abs(), shift.abs())
        slope, shift = next_slope, next_shift
        residual = slope * z + shift - target
        value = objective(residual).item()
        trace.append(dict(step=step, normalized_huber=value))
        if change <= tol * (1 + magnitude):
            stop = "parameter_tolerance"
            break
    a = unit * slope / spread
    b = unit * shift - a * center
    if explicit_unit:
        normalization = "residual / explicit unit"
    else:
        normalization = "residual / median(GT inverse depth)"
    diagnostic = dict(
        name="IRLS",
        objective="Huber",
        initialization="full-pixel L2",
        delta=delta,
        normalization=normalization,
        gt_inverse_median=unit.item(),
        effective_inverse_delta=(unit * delta).item(),
        max_iters=max_iters,
        tol=tol,
        steps=step,
        stop=stop,
        dtype="float64",
        trace=trace,
        initial_normalized_huber=initial,
        final_normalized_huber=value,
        initial_mean_absolute_inverse_residual=initial_mae,
        final_mean_absolute_inverse_residual=(residual.abs().mean() *
                                              unit).item(),
    )
    return a.item(), b.item(), diagnostic
