# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Depth alignment and error metrics adapted from AMB3R's depth evaluation.

Adapted from
https://github.com/HengyiWang/amb3r/blob/main/benchmark/tools/depth_eval.py
"""

from typing import Union

import numpy as np
from scipy import optimize
import torch


def depth2disparity(depth: Union[torch.Tensor, np.ndarray],
                    return_mask: bool = False):
    """Returns 1 / depth where depth > 0 and 0 elsewhere.

    Args:
        depth: Depth map, as a tensor or array.
        return_mask: Also return the mask of positive depths.

    Returns:
        The disparity, or (disparity, positive-depth mask) with `return_mask`.

    Raises:
        TypeError: If `depth` is neither a tensor nor an array.
    """
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    else:
        raise TypeError(f"Unsupported depth type: {type(depth)!r}")

    non_negative_mask = depth > 0
    disparity[non_negative_mask] = 1.0 / depth[non_negative_mask]

    if return_mask:
        return disparity, non_negative_mask
    else:
        return disparity


def absolute_error_loss(params, predicted_depth, ground_truth_depth):
    """Sum of |s * predicted_depth + t - ground_truth_depth| for (s, t)."""
    s, t = params

    predicted_aligned = s * predicted_depth + t

    abs_error = np.abs(predicted_aligned - ground_truth_depth)
    return np.sum(abs_error)


def absolute_value_scaling(predicted_depth: torch.Tensor,
                           ground_truth_depth: torch.Tensor,
                           s: float = 1,
                           t: float = 0):
    """Fits scale s and shift t under an L1 loss with scipy's minimize.

    Args:
        predicted_depth: Predicted values.
        ground_truth_depth: GT values, paired with `predicted_depth`.
        s: Initial scale.
        t: Initial shift.

    Returns:
        (s, t).
    """
    predicted_depth_np = predicted_depth.cpu().numpy().reshape(-1)
    ground_truth_depth_np = ground_truth_depth.cpu().numpy().reshape(-1)

    initial_params = [s, t]

    result = optimize.minimize(
        absolute_error_loss,
        initial_params,
        args=(predicted_depth_np, ground_truth_depth_np),
    )

    s, t = result.x
    return s, t


def absolute_value_scaling2(
    predicted_depth: torch.Tensor,
    ground_truth_depth: torch.Tensor,
    s_init: float = 1.0,
    t_init: float = 0.0,
    lr: float = 1e-4,
    max_iters: int = 1000,
    tol: float = 1e-6,
):
    """Fits scale s and shift t under an L1 loss with Adam.

    Args:
        predicted_depth: Predicted values.
        ground_truth_depth: GT values, paired with `predicted_depth`.
        s_init: Initial scale.
        t_init: Initial shift.
        lr: Adam learning rate.
        max_iters: Maximum number of steps.
        tol: Stop when the loss changes by less than this.

    Returns:
        (s, t) as floats.
    """
    s = torch.tensor(
        [s_init],
        requires_grad=True,
        device=predicted_depth.device,
        dtype=predicted_depth.dtype,
    )
    t = torch.tensor(
        [t_init],
        requires_grad=True,
        device=predicted_depth.device,
        dtype=predicted_depth.dtype,
    )

    optimizer = torch.optim.Adam([s, t], lr=lr)

    prev_loss = None

    for _ in range(max_iters):
        optimizer.zero_grad()

        predicted_aligned = s * predicted_depth + t

        abs_error = torch.abs(predicted_aligned - ground_truth_depth)

        loss = torch.sum(abs_error)
        loss.backward()

        optimizer.step()

        if prev_loss is not None and torch.abs(prev_loss - loss) < tol:
            break

        prev_loss = loss.item()

    return s.detach().item(), t.detach().item()


def depth_evaluation(
    predicted_depth_original,
    ground_truth_depth_original,
    max_depth=80,
    custom_mask=None,
    post_clip_min=None,
    post_clip_max=None,
    pre_clip_min=None,
    pre_clip_max=None,
    align_with_lstsq=False,
    align_with_lad=False,
    align_with_lad2=False,
    metric_scale=False,
    lr=1e-4,
    max_iters=1000,
    use_gpu=False,
    align_with_scale=False,
    disp_input=False,
):
    """Aligns a depth map to the ground truth and computes depth metrics.

    Without an alignment flag the prediction is aligned by median scaling.

    Args:
        predicted_depth_original: The predicted depth map (array or tensor).
        ground_truth_depth_original: The ground truth depth map (array or
            tensor).
        max_depth: The maximum GT depth to consider, or None for no limit.
        custom_mask: Optional extra mask of pixels to score.
        post_clip_min: Clip the aligned prediction from below.
        post_clip_max: Clip the aligned prediction from above.
        pre_clip_min: Clip the prediction from below before alignment.
        pre_clip_max: Clip the prediction from above before alignment.
        align_with_lstsq: Align scale and shift by least squares.
        align_with_lad: Align scale and shift by L1 (scipy).
        align_with_lad2: Align scale and shift by L1 (Adam).
        metric_scale: Score the prediction as it is.
        lr: Adam learning rate for `align_with_lad2`.
        max_iters: Adam steps for `align_with_lad2`.
        use_gpu: Move the maps to CUDA.
        align_with_scale: Align the scale alone by IRLS.
        disp_input: The prediction is a disparity; align in disparity space.

    Returns:
        (metrics dict, depth error parity map, predicted depth map, GT depth
        map), the maps at the full image size.
    """
    if isinstance(predicted_depth_original, np.ndarray):
        predicted_depth_original = torch.from_numpy(predicted_depth_original)
    if isinstance(ground_truth_depth_original, np.ndarray):
        ground_truth_depth_original = torch.from_numpy(
            ground_truth_depth_original)
    if custom_mask is not None and isinstance(custom_mask, np.ndarray):
        custom_mask = torch.from_numpy(custom_mask)

    # If the dimension is 3, flatten to 2D along the batch dimension.
    if predicted_depth_original.dim() == 3:
        _, _, w = predicted_depth_original.shape
        predicted_depth_original = predicted_depth_original.view(-1, w)
        ground_truth_depth_original = ground_truth_depth_original.view(-1, w)
        if custom_mask is not None:
            custom_mask = custom_mask.view(-1, w)

    # Put to device.
    if use_gpu:
        predicted_depth_original = predicted_depth_original.cuda()
        ground_truth_depth_original = ground_truth_depth_original.cuda()

    # Filter out depths greater than max_depth.
    if max_depth is not None:
        mask = (ground_truth_depth_original > 0) & (ground_truth_depth_original
                                                    < max_depth)
    else:
        mask = ground_truth_depth_original > 0
    predicted_depth = predicted_depth_original[mask]
    ground_truth_depth = ground_truth_depth_original[mask]

    # Clip the depth values.
    if pre_clip_min is not None:
        predicted_depth = torch.clamp(predicted_depth, min=pre_clip_min)
    if pre_clip_max is not None:
        predicted_depth = torch.clamp(predicted_depth, max=pre_clip_max)

    if disp_input:  # Align the pred to GT in the disparity space.
        real_gt = ground_truth_depth.clone()
        ground_truth_depth = 1 / (ground_truth_depth + 1e-8)

    # Alignment methods.
    if metric_scale:
        pass  # Metric-scale predictions are scored as they are.
    elif align_with_lstsq:
        # Convert to numpy for lstsq.
        predicted_depth_np = predicted_depth.cpu().numpy().reshape(-1, 1)
        ground_truth_depth_np = ground_truth_depth.cpu().numpy().reshape(-1, 1)

        # Add a column of ones for the shift term.
        design_matrix = np.hstack(
            [predicted_depth_np,
             np.ones_like(predicted_depth_np)])

        # Solve for scale (s) and shift (t) using least squares.
        result = np.linalg.lstsq(design_matrix,
                                 ground_truth_depth_np,
                                 rcond=None)
        s, t = result[0][0], result[0][1]

        # Convert to torch tensors.
        s = torch.tensor(s, device=predicted_depth_original.device)
        t = torch.tensor(t, device=predicted_depth_original.device)

        # Apply scale and shift.
        predicted_depth = s * predicted_depth + t
    elif align_with_lad:
        s, t = absolute_value_scaling(
            predicted_depth,
            ground_truth_depth,
            s=torch.median(ground_truth_depth) / torch.median(predicted_depth),
        )
        predicted_depth = s * predicted_depth + t
    elif align_with_lad2:
        s_init = (torch.median(ground_truth_depth) /
                  torch.median(predicted_depth)).item()
        s, t = absolute_value_scaling2(
            predicted_depth,
            ground_truth_depth,
            s_init=s_init,
            lr=lr,
            max_iters=max_iters,
        )
        predicted_depth = s * predicted_depth + t
    elif align_with_scale:
        # Compute the initial scale factor 's' using the closed-form solution
        # (L2 norm).
        dot_pred_gt = torch.nanmean(ground_truth_depth)
        dot_pred_pred = torch.nanmean(predicted_depth)
        s = dot_pred_gt / dot_pred_pred

        # Iterative reweighted least squares using the Weiszfeld method.
        for _ in range(10):
            # Compute residuals between scaled predictions and ground truth.
            residuals = s * predicted_depth - ground_truth_depth
            # Add a small constant to avoid division by zero.
            abs_residuals = residuals.abs() + 1e-8

            # Compute weights inversely proportional to the residuals.
            weights = 1.0 / abs_residuals

            # Update 's' using weighted sums.
            weighted_dot_pred_gt = torch.sum(weights * predicted_depth *
                                             ground_truth_depth)
            weighted_dot_pred_pred = torch.sum(weights * predicted_depth**2)
            s = weighted_dot_pred_gt / weighted_dot_pred_pred

        # Clip 's' to prevent extreme scaling.
        s = s.clamp(min=1e-3)

        # Detach 's' to stop gradients from flowing through it.
        s = s.detach()

        # Apply the scale factor to the predicted depth.
        predicted_depth = s * predicted_depth

    else:
        # Align the predicted depth with the ground truth by median scaling.
        scale_factor = torch.median(ground_truth_depth) / torch.median(
            predicted_depth)
        predicted_depth *= scale_factor

    if disp_input:
        # Convert back to depth.
        ground_truth_depth = real_gt
        predicted_depth = depth2disparity(predicted_depth)

    # Clip the predicted depth values.
    if post_clip_min is not None:
        predicted_depth = torch.clamp(predicted_depth, min=post_clip_min)
    if post_clip_max is not None:
        predicted_depth = torch.clamp(predicted_depth, max=post_clip_max)

    if custom_mask is not None:
        assert custom_mask.shape == ground_truth_depth_original.shape
        mask_within_mask = custom_mask.cpu()[mask]
        predicted_depth = predicted_depth[mask_within_mask]
        ground_truth_depth = ground_truth_depth[mask_within_mask]

    # Calculate the metrics.
    abs_rel = torch.mean(
        torch.abs(predicted_depth - ground_truth_depth) /
        ground_truth_depth).item()
    sq_rel = torch.mean(((predicted_depth - ground_truth_depth)**2) /
                        ground_truth_depth).item()

    # Correct RMSE calculation.
    rmse = torch.sqrt(torch.mean(
        (predicted_depth - ground_truth_depth)**2)).item()

    # Clip the depth values to avoid log(0).
    predicted_depth = torch.clamp(predicted_depth, min=1e-5)
    log_rmse = torch.sqrt(
        torch.mean((torch.log(predicted_depth) -
                    torch.log(ground_truth_depth))**2)).item()

    # Calculate the accuracy thresholds.
    max_ratio = torch.maximum(predicted_depth / ground_truth_depth,
                              ground_truth_depth / predicted_depth)
    threshold_0 = torch.mean((max_ratio < 1.0).float()).item()
    threshold_103 = torch.mean((max_ratio < 1.03).float()).item()
    threshold_1 = torch.mean((max_ratio < 1.25).float()).item()
    threshold_2 = torch.mean((max_ratio < 1.25**2).float()).item()
    threshold_3 = torch.mean((max_ratio < 1.25**3).float()).item()

    # Compute the depth error parity map.
    if metric_scale:
        pass
    elif align_with_lstsq or align_with_lad or align_with_lad2:
        predicted_depth_original = predicted_depth_original * s + t
    elif align_with_scale:
        predicted_depth_original = predicted_depth_original * s
    else:
        predicted_depth_original = predicted_depth_original * scale_factor
    if disp_input:
        predicted_depth_original = depth2disparity(predicted_depth_original)
    depth_error_parity_map = (
        torch.abs(predicted_depth_original - ground_truth_depth_original) /
        ground_truth_depth_original)

    # Reshape the depth_error_parity_map back to the original image size.
    depth_error_parity_map_full = torch.zeros_like(ground_truth_depth_original)
    depth_error_parity_map_full = torch.where(mask, depth_error_parity_map,
                                              depth_error_parity_map_full)

    predict_depth_map_full = predicted_depth_original
    gt_depth_map_full = torch.zeros_like(ground_truth_depth_original)
    gt_depth_map_full = torch.where(mask, ground_truth_depth_original,
                                    gt_depth_map_full)

    if custom_mask is None:
        num_valid_pixels = torch.sum(mask).item()
    else:
        num_valid_pixels = torch.sum(mask_within_mask).item()
    if num_valid_pixels == 0:
        (
            abs_rel,
            sq_rel,
            rmse,
            log_rmse,
            threshold_0,
            threshold_103,
            threshold_1,
            threshold_2,
            threshold_3,
        ) = (0, 0, 0, 0, 0, 0, 0, 0, 0)

    results = {
        "Abs Rel": abs_rel,
        "Sq Rel": sq_rel,
        "RMSE": rmse,
        "Log RMSE": log_rmse,
        "δ < 1.": threshold_0,
        "δ < 1.03": threshold_103,
        "δ < 1.25": threshold_1,
        "δ < 1.25^2": threshold_2,
        "δ < 1.25^3": threshold_3,
        "valid_pixels": num_valid_pixels,
    }

    return (
        results,
        depth_error_parity_map_full,
        predict_depth_map_full,
        gt_depth_map_full,
    )
