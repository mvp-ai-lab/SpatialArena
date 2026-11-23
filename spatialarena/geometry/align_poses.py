# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Similarity (Sim(3)) alignment of camera trajectories.

`align_ate_c2b_use_a2b` estimates the Sim(3) that maps the camera centres of
trajectory a onto those of trajectory b (Umeyama) and applies it to a third
trajectory c; `spatialarena.eval.utils` uses it to align predicted poses to
the ground truth.
"""

import copy

# The module itself is bound to `edict`, as in the original code.
import easydict as edict
import numpy as np
import torch


def convert3x4_4x4(input):
    """Makes poses homogeneous by appending the row [0, 0, 0, 1].

    Args:
        input: (N, 3, 4) or (3, 4) poses, a torch tensor or a numpy array.

    Returns:
        The (N, 4, 4) or (4, 4) poses, of the same type as `input`.
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat([input, torch.zeros_like(input[:, 0:1])],
                               dim=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat([
                input,
                torch.tensor(
                    [[0, 0, 0, 1]], dtype=input.dtype, device=input.device)
            ],
                               dim=0)  # (4, 4)
    else:
        if len(input.shape) == 3:
            output = np.concatenate([input, np.zeros_like(input[:, 0:1])],
                                    axis=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate(
                [input, np.array([[0, 0, 0, 1]], dtype=input.dtype)],
                axis=0)  # (4, 4)
            output[3, 3] = 1.0
    return output


def procrustes_analysis(X0, X1):
    """Returns the similarity that aligns the (N, 3) points X1 to X0.

    X1 maps onto X0 as `(X1 - t1) / s1 @ R.t() * s0 + t0`.

    Args:
        X0: (N, 3) torch tensor of target points.
        X1: (N, 3) torch tensor of source points.

    Returns:
        The translations t0, t1, scales s0, s1 and rotation R.
    """
    # Translation.
    t0 = X0.mean(dim=0, keepdim=True)
    t1 = X1.mean(dim=0, keepdim=True)
    X0c = X0 - t0
    X1c = X1 - t1

    # Scale.
    s0 = (X0c**2).sum(dim=-1).mean().sqrt()
    s1 = (X1c**2).sum(dim=-1).mean().sqrt()
    X0cs = X0c / s0
    X1cs = X1c / s1

    # Rotation (SVD in double precision, float loses precision).
    U, S, V = (X0cs.t() @ X1cs).double().svd(some=True)
    R = (U @ V.t()).float()
    if R.det() < 0:
        R[2] *= -1

    # Align X1 to X0: X1to0 = (X1-t1)/s1@R.t()*s0+t0
    sim3 = edict(t0=t0[0], t1=t1[0], s0=s0, s1=s1, R=R)
    return sim3


def get_best_yaw(C):
    """Returns the yaw theta that maximizes trace(Rz(theta) * C)."""
    assert C.shape == (3, 3)

    A = C[0, 1] - C[1, 0]
    B = C[0, 0] + C[1, 1]
    theta = np.pi / 2 - np.arctan2(B, A)

    return theta


def align_umeyama(model, data, known_scale=False):
    """Estimates the similarity `model = s * R * data + t` (Umeyama).

    Implements S. Umeyama, Least-Squares Estimation of Transformation
    Parameters Between Two Point Patterns, IEEE Trans. Pattern Anal. Mach.
    Intell., vol. 13, no. 4, 1991.

    Args:
        model: (n, 3) numpy array, the first trajectory.
        data: (n, 3) numpy array, the second trajectory.
        known_scale: Whether to fix the scale to 1.

    Returns:
        The scale factor s, the (3, 3) rotation R and the (3,) translation t.
    """
    # Subtract the means.
    mu_M = model.mean(0)
    mu_D = data.mean(0)
    model_zerocentered = model - mu_M
    data_zerocentered = data - mu_D
    n = np.shape(model)[0]

    # Correlation.
    C = 1.0 / n * np.dot(model_zerocentered.transpose(), data_zerocentered)
    sigma2 = 1.0 / n * np.multiply(data_zerocentered, data_zerocentered).sum()
    U_svd, D_svd, V_svd = np.linalg.linalg.svd(C)

    D_svd = np.diag(D_svd)
    V_svd = np.transpose(V_svd)

    S = np.eye(3)
    if (np.linalg.det(U_svd) * np.linalg.det(V_svd) < 0):
        S[2, 2] = -1

    R = np.dot(U_svd, np.dot(S, np.transpose(V_svd)))

    if known_scale:
        s = 1
    else:
        s = 1.0 / sigma2 * np.trace(np.dot(D_svd, S))

    t = mu_M - s * np.dot(R, mu_D)

    return s, R, t


def _get_indices(n_aligned, total_n):
    """Returns the indices of the first n_aligned poses (-1: all of them)."""
    if n_aligned == -1:
        idxs = np.arange(0, total_n)
    else:
        assert n_aligned <= total_n and n_aligned >= 1
        idxs = np.arange(0, n_aligned)
    return idxs


def align_sim3(p_es, p_gt, n_aligned=-1):
    """Returns the similarity s, R, t with `gt = R * s * est + t`.

    Falls back to the identity when the SVD does not converge.

    Args:
        p_es: (N, >=3) estimated positions.
        p_gt: (N, >=3) ground-truth positions.
        n_aligned: The number of leading positions to align on, -1 for all.
    """
    idxs = _get_indices(n_aligned, p_es.shape[0])
    est_pos = p_es[idxs, 0:3]
    gt_pos = p_gt[idxs, 0:3]
    try:
        s, R, t = align_umeyama(gt_pos, est_pos)  # note the order
    except BaseException:  # pylint: disable=broad-exception-caught
        print('[WARNING] align_poses.py: SVD did not converge!')
        s, R, t = 1.0, np.eye(3), np.zeros(3)
    return s, R, t


def align_ate_c2b_use_a2b(traj_a, traj_b, traj_c=None):
    """Aligns trajectory c to b with the Sim(3) estimated from a to b.

    Only the camera centres of a and b enter the estimate.

    Args:
        traj_a: (N0, 3 or 4, 4) poses, a torch tensor or a numpy array.
        traj_b: (N0, 3 or 4, 4) poses, a torch tensor or a numpy array.
        traj_c: None (align a itself) or (N1, 3 or 4, 4) poses.

    Returns:
        The aligned (N1, 4, 4) poses of c and the scale s, the (1, 3, 3)
        rotation R and the (1, 3, 1) translation t.
    """

    if isinstance(traj_a, torch.Tensor):
        traj_a = traj_a.float().cpu().numpy()

    if isinstance(traj_a, torch.Tensor):
        traj_b = traj_b.float().cpu().numpy()

    if traj_c is None:
        traj_c = copy.deepcopy(traj_a)
    # traj_c = traj_c.float().cpu().numpy()

    # R_a = traj_a[:, :3, :3]  # (N0, 3, 3)
    t_a = traj_a[:, :3, 3]  # (N0, 3)

    # R_b = traj_b[:, :3, :3]  # (N0, 3, 3)
    t_b = traj_b[:, :3, 3]  # (N0, 3)

    # Scalar, (3, 3), (3,) with gt = R * s * est + t.
    s, R, t = align_sim3(t_a, t_b)

    # Reshape for batched products.
    R = R[None, :, :].astype(np.float32)  # (1, 3, 3)
    t = t[None, :, None].astype(np.float32)  # (1, 3, 1)
    s = float(s)

    R_c = traj_c[:, :3, :3]  # (N1, 3, 3)
    t_c = traj_c[:, :3, 3:4]  # (N1, 3, 1)

    R_c_aligned = R @ R_c  # (N1, 3, 3)
    t_c_aligned = s * (R @ t_c) + t  # (N1, 3, 1)
    traj_c_aligned = np.concatenate([R_c_aligned, t_c_aligned],
                                    axis=2)  # (N1, 3, 4)

    # Append the last row.
    traj_c_aligned = convert3x4_4x4(traj_c_aligned)  # (N1, 4, 4)

    return traj_c_aligned, s, R, t  # (N1, 4, 4)
