# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Compare trajectory alignment estimators for the SLAM ATE: evo's Umeyama Sim(3) vs our RANSAC Sim(3).

Estimators (all map a predicted camera-to-world trajectory onto the ground truth):

- ``evo_umeyama``: ``evo.core.trajectory.PoseTrajectory3D.align(ref, correct_scale=True)`` -- closed-form
  least-squares Sim(3) on all camera centers (what LingBot-Map's / ABot-Recon's evaluations use).
- ``ransac_dagsfm``: ``slam_trajectory_evaluator.fit_alignment(estimator="ransac_dagsfm")``, the SLAM evaluator's
  previous estimator (LMedS RANSAC over the pose-aware DAGSfM fit, minimal sample 2, 25 iterations, refit on
  inliers).
- ``ransac_dagsfm_200``: the same with 200 iterations (is 25 enough?).
- ``ransac_umeyama``: the same harness around the center-only Umeyama fit (minimal sample 3), separating "robust
  selection" from "pose-aware fit".
- ``gtsam_pose``: ``pose_alignment.fit_sim3_pose_residuals`` (the cost of GTSAM's ``TrajectoryAlignerSim3``: a joint
  rotation-geodesic + translation residual per pose pair, nonlinear least squares) on all frames, no RANSAC.
- ``ransac_gtsam_pose``: that fit inside the same RANSAC harness and settings as ``ransac_dagsfm`` (minimal sample
  2, 25 iterations, LMedS), so only the fit kernel differs.
- ``ransac_gtsam_irls``: ``pose_alignment.ransac_irls_align_sim3``, the SLAM evaluator's default: ``ransac_gtsam_pose``
  followed by ``irls_refine_sim3`` on all frames (the same pose residual, Cauchy-weighted IRLS with per-iteration
  robust scales for rotation and translation).
- ``gtsam_irls``: the same refit started from ``gtsam_pose`` instead of RANSAC (deterministic by construction).

Synthetic mode (``--mode synthetic``): ground-truth trajectories of four shapes are mapped through a known random
Sim(3) and corrupted in controlled ways (noise, odometry drift, scale drift, a tracking-loss jump, gross outliers,
orientation-only noise), plus a near-collinear family (a straight 100 m path with lateral deviation eps x length,
eps = 0 ... 0.05) where rotation about the motion axis is barely observable from camera centers. For each trial it
records, per estimator:

- ``align_err``: RMS distance, over the ground-truth camera centers, between the estimated and the true alignment,
  divided by the trajectory extent -- how far the estimated frame is from the true one, independent of the
  corruption;
- ``rot_err_deg``: angle between the estimated and the true alignment rotation (ATE alone hides roll about a line);
- ``ate``, ``ate_true``: ATE RMSE (in extents) after the estimated vs the true alignment;
- ``ate_clean``: ATE RMSE over the uncorrupted frames only (jump / outlier scenarios);
- ``seed_cv``: for the RANSAC estimators, the coefficient of variation of the ATE over 10 RANSAC seeds.

Real-GT mode (``--mode real_gt``): the same corruptions applied to the real ground-truth trajectories of the SLAM
datasets at each range-track stride, so the estimators are compared on real camera motion (CPU only).

Usage (CPU job): ``python scripts/test/compare_trajectory_alignment.py --mode synthetic --output out.json``.
"""

from __future__ import annotations

import os
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# pylint: disable=wrong-import-position
from spatialarena.eval.dataset_registry import select_dataset_specs  # noqa: E402
from spatialarena.eval.pose_alignment import (  # noqa: E402
    Sim3Estimate,
    fit_sim3_dagsfm,
    fit_sim3_pose_residuals,
    fit_sim3_umeyama,
    irls_refine_sim3,
    ransac_align_sim3,
    ransac_irls_align_sim3,
)
from spatialarena.eval.slam_trajectory_evaluator import (  # noqa: E402
    RANSAC_ITERATIONS,
    apply_alignment,
    camera_to_world,
    expected_input_names,
    fit_alignment,
)
from spatialarena.eval.sparse_reconstruction_evaluator import read_colmap_poses  # noqa: E402

# pylint: enable=wrong-import-position

ESTIMATORS = ("evo_umeyama", "ransac_dagsfm", "ransac_dagsfm_200", "ransac_umeyama", "gtsam_pose",
              "ransac_gtsam_pose", "ransac_gtsam_irls", "gtsam_irls")


# ---------------------------------------------------------------------------------------------------- estimators
# Every estimator takes (gt, pred, seed) and returns (scale, rotation, translation); deterministic ones ignore the seed.
def align_evo(gt: np.ndarray, pred: np.ndarray, seed: int = 42):
    """evo's ``PoseTrajectory3D.align(correct_scale=True)``: Umeyama Sim(3) on all camera centers."""
    del seed
    from evo.core.trajectory import PoseTrajectory3D  # pylint: disable=import-outside-toplevel  # optional dependency

    stamps = np.arange(len(gt), dtype=float)
    ref = PoseTrajectory3D(poses_se3=list(gt), timestamps=stamps)
    est = PoseTrajectory3D(poses_se3=list(pred), timestamps=stamps)
    rotation, translation, scale = est.align(ref, correct_scale=True)
    return float(scale), np.asarray(rotation), np.asarray(translation).reshape(3)


def _ransac(gt, pred, *, fit_fn, sample_size, iterations, seed):
    estimate = ransac_align_sim3(
        np.swapaxes(pred[:, :3, :3], 1, 2), pred[:, :3, 3], np.swapaxes(gt[:, :3, :3], 1, 2), gt[:, :3, 3],
        fit_fn=fit_fn, sample_size=sample_size, num_iterations=iterations, random_state=seed,
    )
    return estimate.scale, estimate.rotation, estimate.translation


def align_ransac_dagsfm(gt, pred, seed: int = 42):
    """The SLAM evaluator's previous estimator (``fit_alignment(estimator="ransac_dagsfm")`` at seed 42)."""
    if seed == 42:  # the evaluator's own call, unchanged
        estimate = fit_alignment(gt, pred, with_scale=True, estimator="ransac_dagsfm")
        return estimate.scale, estimate.rotation, estimate.translation
    return _ransac(gt, pred, fit_fn=fit_sim3_dagsfm, sample_size=2, iterations=RANSAC_ITERATIONS, seed=seed)


def align_ransac_dagsfm_200(gt, pred, seed: int = 42):
    """``align_ransac_dagsfm`` with 200 RANSAC iterations."""
    return _ransac(gt, pred, fit_fn=fit_sim3_dagsfm, sample_size=2, iterations=200, seed=seed)


def align_ransac_umeyama(gt, pred, seed: int = 42):
    """The same RANSAC harness around the center-only Umeyama fit (minimal sample 3)."""
    return _ransac(gt, pred, fit_fn=fit_sim3_umeyama, sample_size=3, iterations=RANSAC_ITERATIONS, seed=seed)


def align_gtsam(gt, pred, seed: int = 42):
    """GTSAM TrajectoryAlignerSim3-style joint pose-residual least squares (fit_sim3_pose_residuals) on all frames."""
    del seed
    return fit_sim3_pose_residuals(np.swapaxes(pred[:, :3, :3], 1, 2), pred[:, :3, 3], np.swapaxes(gt[:, :3, :3], 1, 2),
                                   gt[:, :3, 3], np.arange(len(gt)))


def align_ransac_gtsam(gt, pred, seed: int = 42):
    """The same RANSAC harness and settings as ``align_ransac_dagsfm`` around ``fit_sim3_pose_residuals``."""
    return _ransac(gt, pred, fit_fn=fit_sim3_pose_residuals, sample_size=2, iterations=RANSAC_ITERATIONS, seed=seed)


def _pose_args(gt, pred):
    """(pred w2c rotations, pred centers, gt w2c rotations, gt centers) from camera-to-world trajectories."""
    return np.swapaxes(pred[:, :3, :3], 1, 2), pred[:, :3, 3], np.swapaxes(gt[:, :3, :3], 1, 2), gt[:, :3, 3]


def align_ransac_gtsam_irls(gt, pred, seed: int = 42):
    """``pose_alignment.ransac_irls_align_sim3``: the SLAM evaluator's default (``fit_alignment`` at seed 42)."""
    estimate = ransac_irls_align_sim3(*_pose_args(gt, pred), num_iterations=RANSAC_ITERATIONS, random_state=seed)
    return estimate.scale, estimate.rotation, estimate.translation


def align_gtsam_irls(gt, pred, seed: int = 42):
    """``irls_refine_sim3`` started from the all-frames GTSAM fit instead of RANSAC (deterministic)."""
    del seed
    scale, rotation, translation = align_gtsam(gt, pred)
    initial = Sim3Estimate(scale=scale, rotation=rotation, translation=translation, inliers=np.ones(len(gt), bool))
    estimate = irls_refine_sim3(*_pose_args(gt, pred), initial)
    return estimate.scale, estimate.rotation, estimate.translation


ALIGN = {
    "evo_umeyama": align_evo,
    "ransac_dagsfm": align_ransac_dagsfm,
    "ransac_dagsfm_200": align_ransac_dagsfm_200,
    "ransac_umeyama": align_ransac_umeyama,
    "gtsam_pose": align_gtsam,
    "ransac_gtsam_pose": align_ransac_gtsam,
    "ransac_gtsam_irls": align_ransac_gtsam_irls,
    "gtsam_irls": align_gtsam_irls,
}


def ate(gt: np.ndarray, pred: np.ndarray, sim3, mask: np.ndarray | None = None) -> float:
    """ATE RMSE of ``pred`` after the Sim(3) ``sim3`` = (scale, rotation, translation), over ``mask`` if given."""
    aligned = apply_alignment(pred, *sim3)
    err = np.linalg.norm(aligned[:, :3, 3] - gt[:, :3, 3], axis=1)
    if mask is not None:
        err = err[mask]
    return float(np.sqrt(np.mean(err**2)))


# ---------------------------------------------------------------------------------------------------- synthetic data
def _look_at(center: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - center
    forward /= np.linalg.norm(forward) + 1e-12
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward], axis=1)  # OpenCV camera-to-world rotation


def make_trajectory(shape: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """[n, 4, 4] camera-to-world ground truth of one of four shapes."""
    t = np.linspace(0.0, 1.0, n)
    if shape == "handheld":  # TUM / 7-Scenes-like: smooth wandering in a room, looking around
        knots = rng.uniform(-1.5, 1.5, size=(6, 3)) * [1, 1, 0.3]
        centers = np.stack([np.interp(t, np.linspace(0, 1, 6), knots[:, k]) for k in range(3)], axis=1)
        centers += 0.05 * np.sin(2 * np.pi * 7 * t)[:, None]
        targets = centers + np.stack([np.cos(4 * t), np.sin(4 * t), 0.2 * np.sin(9 * t)], axis=1) * 2.0
    elif shape == "near_static":  # TUM sitting_*: centers barely move, the view sweeps
        centers = 0.08 * np.stack([np.sin(3 * t), np.cos(5 * t), 0.3 * np.sin(7 * t)], axis=1)
        targets = centers + np.stack([np.cos(1.5 * t), np.sin(1.5 * t), 0.1 * np.ones_like(t)], axis=1)
    elif shape == "driving":  # VBR / Oxford / KITTI-like: long, near-collinear forward motion with a few turns
        heading = np.cumsum(rng.normal(0.0, 0.004, n))
        heading += np.interp(t, [0, 0.3, 0.35, 0.7, 0.75, 1], [0, 0, np.pi / 2, np.pi / 2, np.pi / 3, np.pi / 3])
        step = 500.0 / n
        centers = np.cumsum(np.stack([np.cos(heading), np.sin(heading), 0.002 * rng.normal(size=n)], axis=1) * step,
                            axis=0)
        targets = centers + np.stack([np.cos(heading), np.sin(heading), np.zeros(n)], axis=1)
    elif shape == "loop":  # a closed outdoor loop
        angle = 2 * np.pi * t
        centers = np.stack([20 * np.cos(angle), 12 * np.sin(angle), 0.5 * np.sin(3 * angle)], axis=1)
        targets = centers + np.stack([-np.sin(angle), np.cos(angle), np.zeros(n)], axis=1)
    else:
        raise ValueError(shape)
    poses = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        poses[i, :3, :3] = _look_at(centers[i], targets[i])
        poses[i, :3, 3] = centers[i]
    return poses


def extent(poses: np.ndarray) -> float:
    """Diagonal of the camera centers' bounding box."""
    centers = poses[:, :3, 3]
    return float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)))


def _random_rotation(rng, sigma_deg: float) -> np.ndarray:
    return Rotation.from_rotvec(np.deg2rad(sigma_deg) * rng.normal(size=3)).as_matrix()


def corrupt(gt: np.ndarray, scenario: str, level: float, rng: np.random.Generator):
    """Corrupted copy of ``gt`` (still in the GT frame) and the mask of uncorrupted frames."""
    n, ext = len(gt), extent(gt)
    out = gt.copy()
    clean = np.ones(n, dtype=bool)
    if scenario == "noise":  # i.i.d. center noise (level = sigma / extent) and 1 deg orientation noise
        out[:, :3, 3] += level * ext * rng.normal(size=(n, 3)) / np.sqrt(3)
        for i in range(n):
            out[i, :3, :3] = _random_rotation(rng, 1.0) @ out[i, :3, :3]
    elif scenario == "drift":  # odometry random walk: noisy relative motions integrated (level = rel. step noise)
        rel = np.linalg.inv(gt[:-1]) @ gt[1:]
        step = np.linalg.norm(rel[:, :3, 3], axis=1).mean()
        for i in range(n - 1):
            noisy = rel[i].copy()
            noisy[:3, :3] = _random_rotation(rng, 10 * level) @ noisy[:3, :3]
            noisy[:3, 3] += level * step * rng.normal(size=3)
            out[i + 1] = out[i] @ noisy
    elif scenario == "scale_drift":  # relative translations scaled from 1 to 1 + level along the sequence
        rel = np.linalg.inv(gt[:-1]) @ gt[1:]
        for i in range(n - 1):
            scaled = rel[i].copy()
            scaled[:3, 3] *= 1.0 + level * i / (n - 1)
            out[i + 1] = out[i] @ scaled
    elif scenario == "jump":  # tracking loss: the last `level` fraction is offset by a rigid jump
        start = int(round((1.0 - level) * n))
        jump = np.eye(4)
        jump[:3, :3] = Rotation.from_rotvec(np.deg2rad(20) * np.array([0.0, 0.0, 1.0])).as_matrix()
        jump[:3, 3] = 0.3 * ext * np.array([1.0, -0.5, 0.2])
        anchor = gt[start]
        out[start:] = anchor @ jump @ np.linalg.inv(anchor) @ gt[start:]
        clean[start:] = False
    elif scenario == "outliers":  # a `level` fraction of frames replaced by random poses inside 2x the extent
        bad = rng.choice(n, size=int(round(level * n)), replace=False)
        center = gt[:, :3, 3].mean(axis=0)
        for i in bad:
            out[i, :3, :3] = Rotation.random(random_state=int(rng.integers(1 << 31))).as_matrix()
            out[i, :3, 3] = center + ext * rng.uniform(-1, 1, size=3)
        clean[bad] = False
        # mild noise on the rest, so the clean part is not exact
        good = np.flatnonzero(clean)
        out[good, :3, 3] += 0.005 * ext * rng.normal(size=(len(good), 3))
    elif scenario == "rot_noise":  # accurate centers (0.5 % noise), orientations perturbed by `level` degrees
        out[:, :3, 3] += 0.005 * ext * rng.normal(size=(n, 3)) / np.sqrt(3)
        for i in range(n):
            out[i, :3, :3] = _random_rotation(rng, level) @ out[i, :3, :3]
    else:
        raise ValueError(scenario)
    return out, clean


SCENARIOS = {
    "noise": (0.005, 0.02, 0.05),
    "drift": (0.02, 0.05, 0.1),
    "scale_drift": (0.1, 0.3, 0.6),
    "jump": (0.05, 0.15, 0.3, 0.45),
    "outliers": (0.05, 0.15, 0.3, 0.45),
    "rot_noise": (2.0, 10.0, 30.0),
}
SHAPES = ("handheld", "near_static", "driving", "loop")
# Near-collinear forward motion: a straight 100 m path whose lateral deviation is `eps` x its length (eps = 0 is an
# exactly straight line). Rotation about the motion axis is then (nearly) unobservable from camera centers alone.
COLLINEAR_EPS = (0.0, 1e-4, 1e-3, 1e-2, 5e-2)
COLLINEAR_SCENARIOS = {"noise": (0.001, 0.01), "jump": (0.15, 0.3), "outliers": (0.15, 0.3), "rot_noise": (10.0,)}


def make_collinear(n: int, eps: float, rng: np.random.Generator) -> np.ndarray:
    """[n, 4, 4] camera-to-world on a 100 m straight path with lateral deviation ``eps`` x its length."""
    t = np.linspace(0.0, 1.0, n)
    length = 100.0
    phase = rng.uniform(0, 2 * np.pi, size=2)
    lateral = eps * length * np.stack(
        [np.sin(2 * np.pi * 3 * t + phase[0]), 0.5 * np.sin(2 * np.pi * 5 * t + phase[1])], axis=1)
    centers = np.stack([length * t, lateral[:, 0], lateral[:, 1]], axis=1)
    yaw = 0.05 * np.sin(2 * np.pi * 2 * t)  # the camera looks roughly forward
    poses = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        poses[i, :3, :3] = _look_at(centers[i], centers[i] + [np.cos(yaw[i]), np.sin(yaw[i]), 0.0])
        poses[i, :3, 3] = centers[i]
    return poses


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two rotation matrices, in degrees."""
    cos = (np.trace(a.T @ b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def run_trial(gt: np.ndarray, scenario: str, level: float, rng: np.random.Generator, *, seed_trials: bool) -> dict:
    """Corrupt ``gt``, map it into a random frame (the "prediction"), and score every estimator against the truth."""
    ext = extent(gt)
    corrupted, clean = corrupt(gt, scenario, level, rng)
    # The prediction lives in its own frame: pred = T_true^-1 (corrupted), T_true a random Sim(3).
    true_scale = float(np.exp(rng.uniform(np.log(0.05), np.log(20.0))))
    true_rotation = Rotation.random(random_state=int(rng.integers(1 << 31))).as_matrix()
    true_translation = rng.normal(size=3) * 10.0
    pred = corrupted.copy()
    pred[:, :3, :3] = true_rotation.T @ corrupted[:, :3, :3]
    # center c = s R p + t  =>  p = (c - t) R / s  (row vectors)
    pred[:, :3, 3] = ((corrupted[:, :3, 3] - true_translation) @ true_rotation) / true_scale
    true_sim3 = (true_scale, true_rotation, true_translation)
    clean_pred = gt.copy()
    clean_pred[:, :3, 3] = ((gt[:, :3, 3] - true_translation) @ true_rotation) / true_scale
    row = {"scenario": scenario, "level": level, "frames": len(gt), "extent_m": ext,
           "ate_true": ate(gt, pred, true_sim3) / ext}
    if not clean.all():
        row["ate_true_clean"] = ate(gt, pred, true_sim3, clean) / ext
    for name, fn in ALIGN.items():
        try:
            est = fn(gt, pred)
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught  # a failure is a result
            row[f"{name}_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            continue
        # How far the estimated alignment is from the true one, over the GT centers (in extents).
        mapped = apply_alignment(clean_pred, *est)[:, :3, 3]
        row[f"{name}_align_err"] = float(np.sqrt(np.mean(np.linalg.norm(mapped - gt[:, :3, 3], axis=1) ** 2)) / ext)
        row[f"{name}_rot_err_deg"] = rotation_error_deg(est[1], true_rotation)
        row[f"{name}_scale_err"] = float(abs(np.log(est[0] / true_scale)))
        row[f"{name}_ate"] = ate(gt, pred, est) / ext
        if not clean.all():
            row[f"{name}_ate_clean"] = ate(gt, pred, est, clean) / ext
        if name.startswith("ransac") and seed_trials:
            seeds = [ate(gt, pred, fn(gt, pred, seed=s)) for s in range(10)]
            row[f"{name}_seed_cv"] = float(np.std(seeds) / max(np.mean(seeds), 1e-12))
    return row


def run_synthetic(trials: int, n: int, seed: int) -> list[dict]:
    """``trials`` trials of every shape x scenario x level, then of the near-collinear family."""
    rows = []
    rng = np.random.default_rng(seed)
    for shape in SHAPES:
        for scenario, levels in SCENARIOS.items():
            for level in levels:
                for trial in range(trials):
                    gt = make_trajectory(shape, n, rng)
                    rows.append({"source": shape, "trial": trial}
                                | run_trial(gt, scenario, level, rng, seed_trials=trial < 5))
        print(f"done {shape}", flush=True)
    for eps in COLLINEAR_EPS:
        for scenario, levels in COLLINEAR_SCENARIOS.items():
            for level in levels:
                for trial in range(trials):
                    gt = make_collinear(n, eps, rng)
                    rows.append({"source": f"collinear_eps{eps:g}", "trial": trial}
                                | run_trial(gt, scenario, level, rng, seed_trials=trial < 5))
        print(f"done collinear eps={eps:g}", flush=True)
    return rows


def load_real_gt(benchmark_root: Path, datasets: list[str], stride: int) -> list[tuple[str, np.ndarray]]:
    """GT camera-to-world of every SLAM sequence at ``stride`` (the frames a method is given in that range track)."""
    out = []
    for spec in select_dataset_specs(tasks=["SLAM"], datasets=datasets, include_scannet=False):
        dataset = spec.create_dataset(benchmark_root, scenes=[])
        if not dataset.metadata_path.is_file():  # registered but not materialized yet
            print(f"skip {spec.dataset}: no {dataset.metadata_path}", flush=True)
            continue
        for scene_info in dataset.list_scenes():
            if scene_info.sparse_gt_path is None:
                continue
            gt_poses = read_colmap_poses(scene_info.sparse_gt_path)
            names = expected_input_names(list(gt_poses), stride)
            out.append((f"{scene_info.dataset}/{scene_info.scene}", camera_to_world(gt_poses, names)))
    return out


def run_real_gt(benchmark_root: Path, datasets: list[str], strides: list[int], trials: int, seed: int) -> list[dict]:
    """``trials`` trials of every scenario x level on each real GT trajectory at each stride."""
    rows = []
    rng = np.random.default_rng(seed)
    for stride in strides:
        for case, gt in load_real_gt(benchmark_root, datasets, stride):
            for scenario, levels in SCENARIOS.items():
                for level in levels:
                    for trial in range(trials):
                        rows.append({"source": case, "stride": stride, "trial": trial}
                                    | run_trial(gt, scenario, level, rng, seed_trials=trial < 3))
            print(f"done stride {stride} {case} ({len(gt)} frames)", flush=True)
    return rows


def summarize(rows: list[dict], group_key) -> str:
    """Markdown table per (scenario, level, group): median / p90 alignment error (extents), rotation error, ATE
    relative to the truth, clean-frame ATE, RANSAC seed variation and estimator failures."""
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row["scenario"], row["level"], group_key(row))].append(row)
    header = ("| scenario | level | group | n | "
              + " | ".join(f"{e} align_err med/p90 · rot_err med (deg)" for e in ESTIMATORS)
              + " | ATE/ATE_true med (evo, ours, r-umeyama) | clean ATE med (true, evo, ours, r-umeyama) "
                "| seed CV p90 (ours, r-umeyama) | failures |")
    lines = [header, "|" + "---|" * (header.count("|") - 1)]

    def med(group, key):
        values = [r[key] for r in group if key in r and np.isfinite(r[key])]
        return float(np.median(values)) if values else float("nan")

    def p90(group, key):
        values = [r[key] for r in group if key in r and np.isfinite(r[key])]
        return float(np.percentile(values, 90)) if values else float("nan")

    for (scenario, level, group_name), group in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        cells = [f"{med(group, f'{e}_align_err'):.4f}/{p90(group, f'{e}_align_err'):.4f} · "
                 f"{med(group, f'{e}_rot_err_deg'):.2f}" for e in ESTIMATORS]
        ratio = ", ".join(
            f"{np.nanmedian([r[f'{e}_ate'] / max(r['ate_true'], 1e-12) for r in group if f'{e}_ate' in r]):.2f}"
            if any(f"{e}_ate" in r for r in group) else "nan"
            for e in ("evo_umeyama", "ransac_dagsfm", "ransac_umeyama"))
        clean = ", ".join(f"{med(group, k):.4f}" for k in ("ate_true_clean", "evo_umeyama_ate_clean",
                                                           "ransac_dagsfm_ate_clean", "ransac_umeyama_ate_clean")) \
            if any("ate_true_clean" in r for r in group) else "-"
        cv = f"{p90(group, 'ransac_dagsfm_seed_cv'):.3f}, {p90(group, 'ransac_umeyama_seed_cv'):.3f}"
        failures = collections.Counter(e for r in group for e in ESTIMATORS if f"{e}_error" in r)
        fail_cell = ", ".join(f"{e}: {c}" for e, c in failures.items()) or "-"
        lines.append(f"| {scenario} | {level} | {group_name} | {len(group)} | " + " | ".join(cells)
                     + f" | {ratio} | {clean} | {cv} | {fail_cell} |")
    return "\n".join(lines) + "\n"


# see spatialarena/paths.py
SPATIALARENA_ROOT = os.path.expanduser(os.environ.get("SPATIALARENA_ROOT", "~/spatialarena"))
def main() -> None:
    """Run the selected study and write its rows (JSON) and summary (Markdown)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("synthetic", "real_gt"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--frames", type=int, default=300, help="Synthetic trajectory length.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--benchmark_root", type=Path,
                        default=Path(f"{SPATIALARENA_ROOT}/benchmarks"))
    parser.add_argument("--datasets", nargs="*", default=[], help="SLAM datasets for real_gt (default: all).")
    parser.add_argument("--strides", nargs="+", type=int, default=[10, 3], help="real_gt strides (range tracks).")
    args = parser.parse_args()
    if args.mode == "synthetic":
        rows = run_synthetic(args.trials, args.frames, args.seed)
        table = summarize(rows, lambda r: r["source"])
    else:
        rows = run_real_gt(args.benchmark_root, args.datasets, args.strides, args.trials, args.seed)
        table = summarize(rows, lambda r: f"{r['source'].split('/')[0]} stride {r['stride']}")
    args.output.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(table, encoding="utf-8")


if __name__ == "__main__":
    main()
