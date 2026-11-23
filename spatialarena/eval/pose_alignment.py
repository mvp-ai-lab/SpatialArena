# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Sim(3) camera-trajectory alignment strategies and a shared RANSAC harness.

Both Pi3's and DA3's own reference evaluation code estimate the Sim(3)
aligning a predicted trajectory to ground truth from camera *centers* only
(Umeyama/Kabsch on 3D points -- see `align_points_similarity` in
`sparse_reconstruction_evaluator.py`, which this module reuses as one of
three interchangeable "fit" strategies). That is ill-conditioned whenever
the centers are close to collinear or coplanar: the cross-covariance matrix
becomes rank-deficient and the SVD can't pin down rotation about the
degenerate axis.

Because we have full SE(3) poses (rotation *and* translation) for every
frame, not just positions, two better-conditioned strategies are available,
both verified against public reference implementations rather than
reimplemented from memory:

- `fit_sim3_dagsfm` -- ports the rotation/translation step of
  `ComputeSimilarityByCameraMotions` in
  https://github.com/AIBluefisher/DAGSfM/blob/dev/src/controllers/sfm_aligner.cpp
  and its `EuclideanEstimator` (src/estimators/ransac_similarity.h). Scale
  comes from the *median* of pairwise camera-center distance ratios (Eq. 5
  of "Divide and Conquer: Efficient Large-Scale Structure from Motion Using
  Graph Partitioning"), which needs no rotation and is unaffected by
  collinearity. Rotation for a *single* pose pair is exact --
  `R = gt_R[i].T @ pred_R[i]` -- because one full pose already fixes the
  relative orientation between the two coordinate frames; no point spread
  is involved at all. The reference's own minimal sample size is 1
  (`EuclideanEstimator::SampleSize()`), but `ransac_align_sim3` below
  defaults to 2 instead: see its docstring for why (a size-1 sample needs
  scale from a *global*, not per-sample, statistic, which is measurably
  less outlier-robust than every other strategy here).
- `fit_sim3_pose_residuals` -- ports the cost function of GTSAM's
  `gtsam/sfm/TrajectoryAlignerSim3` (fetched from
  https://github.com/borglab/gtsam), simplified to our single-parent
  (fixed ground truth) / single-child (predicted trajectory) case: jointly
  minimize a 6D SE(3) residual (rotation geodesic error + translation
  error) per matched pose pair via nonlinear least squares
  (`scipy.optimize.least_squares`), instead of GTSAM's own factor-graph
  formulation. Note GTSAM's own robustness mechanism is GNC (Graduated
  Non-Convexity), not RANSAC; RANSAC wrapping here is this module's own
  addition, applied uniformly to all three strategies via
  `ransac_align_sim3`.

`ransac_align_sim3` is the shared harness: given any `fit_fn` (of the
three above) and a per-strategy *minimal* sample size, it runs
`num_iterations` random-subsample fits and selects the one minimizing the
**median** squared residual over all matched points -- Least Median of
Squares (LMedS; Rousseeuw & Leroy, "Robust Regression and Outlier
Detection", 1987), which tolerates up to 50% outlier contamination without
ever needing a pre-chosen inlier threshold. A threshold is derived only
*after* selection, from the winning candidate's own residual distribution
(the same `1.4826 * MAD`-style robust-sigma estimate LMedS uses), and used
once to pick the final inlier set for a last refit.

This deliberately does *not* fit once on the full (possibly outlier-
contaminated) sample first to derive a threshold up front, the way
`mvrecon_evaluator.py::_align_da3_poses` does for the Umeyama-only case --
that full-sample fit can itself be dragged arbitrarily far off by a
minority of gross outliers (e.g. `fit_sim3_umeyama` on centers is not
robust at all: a single very-large-magnitude outlier center can dominate
the SVD), which then poisons the very threshold meant to reject it. Using
minimal samples throughout keeps each candidate's chance of being
outlier-free high enough for LMedS to work in practice.

`fit_sim3_dagsfm` is this module's default fit strategy (for both
`ransac_align_sim3` and `ransac_icp_align_sim3` below): a seed sweep
comparing it against `fit_sim3_pose_residuals` across outlier ratios from
20-45%, with and without the collinear case, found them equally robust once
`fit_sim3_dagsfm` is wrapped at `sample_size=2`, at a fraction of the
per-candidate cost (closed-form vs. a nonlinear solve).
`fit_sim3_pose_residuals` and `fit_sim3_umeyama` remain fully implemented and
are kept available via `fit_fn` for comparison and for callers with different
tradeoffs.

`ransac_irls_align_sim3` is the repo-wide pose alignment, used by every
evaluator that aligns predicted to ground-truth poses (SfM ATE, MVRecon pose
alignment, SLAM): RANSAC over `fit_sim3_pose_residuals` followed by
`irls_refine_sim3`, a Cauchy-weighted refit on all poses. A 3,450-trial
simulation (`scripts/test/compare_trajectory_alignment.py`, results in
docs/benchmark/slam_protocol.md, "Choice of estimator") found the refit
removes the RANSAC seed dependence on drifting trajectories and the
orientation-noise bias of the DAGSfM rotation average, while staying exact
under outliers, tracking-loss jumps and near-collinear motion.
`ransac_icp_align_sim3` (RANSAC over `fit_sim3_dagsfm` + ICP on the
inlier camera centers) was the previous default; it is kept for comparison.
Its ICP on camera centers is a center-only Umeyama fit on nearest-neighbor
matches, which reintroduces the collinear degeneracy the pose-aware fit
avoids. `icp_refine_sim3` on dense point clouds (MVRecon mesh alignment) is
a separate stage and unaffected.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Callable

import numpy as np
from scipy import optimize
from scipy.spatial import transform

from spatialarena.eval import sparse_reconstruction_evaluator

FitFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                 tuple]


@dataclasses.dataclass(frozen=True, kw_only=True)
class Sim3Estimate:
    """A Sim(3) alignment x -> scale * rotation @ x + translation.

    Attributes:
        scale: Scale factor.
        rotation: (3, 3) rotation.
        translation: (3,) translation.
        inliers: Boolean mask, length == num matches.
        num_iterations: RANSAC iterations run.
        scale_collapsed: ransac_irls_align_sim3 only: the pose fit collapsed
            the scale; this is the center-only fallback instead.
    """

    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    inliers: np.ndarray = dataclasses.field(repr=False)
    num_iterations: int = 0
    scale_collapsed: bool = False

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Returns the (N, 3) points mapped by this Sim(3)."""
        points = np.asarray(points, dtype=np.float64)
        return self.scale * (points @ self.rotation.T) + self.translation

    def transform_poses(self, rotations_w2c: np.ndarray,
                        centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Applies this Sim(3) to full poses.

        Rotates the orientations and scales and moves the centers. Matches
        `mvrecon_evaluator.py::_apply_similarity_to_c2w`'s convention: the
        similarity transform composes into the *rotation* (no scale there --
        scale only ever affects position), and scales + translates the
        *center*.

        Args:
            rotations_w2c: (N, 3, 3) world-to-camera rotations.
            centers: (N, 3) camera centers.

        Returns:
            The transformed rotations and centers.
        """
        rotations_w2c = np.asarray(rotations_w2c, dtype=np.float64)
        centers = np.asarray(centers, dtype=np.float64)
        # rotations_w2c[i] maps world->camera; a Sim(3) applied to world
        # coordinates composes on the world side, i.e.
        # new_R_w2c = R_w2c @ rotation.T (equivalent to rotating the c2w
        # orientation by `rotation` on the left).
        new_rotations = rotations_w2c @ self.rotation.T
        new_centers = self.scale * (
            centers @ self.rotation.T) + self.translation
        return new_rotations, new_centers


def _relative_rotation_deg(rotation_a: np.ndarray,
                           rotation_b: np.ndarray) -> float:
    """Returns the angle between two rotations in degrees."""
    relative = rotation_a.T @ rotation_b
    trace = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(trace)))


def median_pairwise_scale(
    pred_centers: np.ndarray,
    gt_centers: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
    max_pairs: int = 20_000,
) -> float:
    """DAGSfM's scale estimate: median of all pairwise center-distance ratios.

    Needs no rotation and is unaffected by collinear centers (a distance
    ratio between two points is well-defined regardless of how many other
    points share their line). Matches `ComputeSimilarityByCameraMotions`'s
    scale computation exactly (Eq. 5 of "Divide and Conquer: Efficient
    Large-Scale Structure from Motion Using Graph Partitioning").

    Args:
        pred_centers: (N, 3) predicted camera centers.
        gt_centers: (N, 3) ground-truth camera centers.
        rng: Generator that subsamples the pairs; None uses seed 42.
        max_pairs: Maximum number of pairs used.

    Returns:
        The scale from predicted to ground-truth distances.

    Raises:
        ValueError: There are fewer than two poses, or all pairs coincide.
    """
    num_poses = len(pred_centers)
    if num_poses < 2:
        raise ValueError("Need at least two poses to estimate scale")
    idx_i, idx_j = np.triu_indices(num_poses, k=1)
    if len(idx_i) > max_pairs:
        rng = rng or np.random.default_rng(42)
        choice = rng.choice(len(idx_i), size=max_pairs, replace=False)
        idx_i, idx_j = idx_i[choice], idx_j[choice]
    pred_dist = np.linalg.norm(pred_centers[idx_i] - pred_centers[idx_j],
                               axis=1)
    gt_dist = np.linalg.norm(gt_centers[idx_i] - gt_centers[idx_j], axis=1)
    valid = pred_dist > 1e-9
    if not np.any(valid):
        raise ValueError("All pose pairs are coincident; cannot estimate scale")
    return float(np.median(gt_dist[valid] / pred_dist[valid]))


def fit_sim3_umeyama(
    pred_rotations: np.ndarray,
    pred_centers: np.ndarray,
    gt_rotations: np.ndarray,
    gt_centers: np.ndarray,
    indices: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Baseline: closed-form Umeyama/Kabsch on camera centers only.

    Ignores rotation entirely for the fit itself; this is what
    `mvrecon_evaluator.py::_align_da3_poses` and
    `sparse_reconstruction_evaluator.py::compute_absolute_trajectory_metrics`
    already do. Included here so the comparison unit tests can show,
    concretely, where it degrades on near-collinear trajectories.
    """
    del pred_rotations, gt_rotations
    _, scale, rotation, translation = (
        sparse_reconstruction_evaluator.align_points_similarity(
            pred_centers[indices], gt_centers[indices]))
    return scale, rotation, translation


def fit_sim3_dagsfm(
    pred_rotations: np.ndarray,
    pred_centers: np.ndarray,
    gt_rotations: np.ndarray,
    gt_centers: np.ndarray,
    indices: np.ndarray,
    *,
    scale: float | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Ports DAGSfM's `ComputeSimilarityByCameraMotions` rotation step.

    The port covers the rotation/translation step. Rotation for a single
    pose pair `i` is exact and needs no point spread:
    `R = gt_R[i].T @ pred_R[i]` (both world-to-camera rotations of the *same
    physical camera*, so this recovers the fixed relative
    orientation between the two coordinate frames directly from one
    sample). When called with more than one index (e.g. the final refit on
    a RANSAC inlier set), rotations are averaged with scipy's built-in
    chordal-L2 mean (`Rotation.mean()`) and translation is the
    least-squares-optimal mean given that fixed rotation -- both harmless,
    well-posed generalizations of the original single-sample algorithm,
    which computed neither.

    `scale` is normally supplied once, globally, via
    `median_pairwise_scale` (matching the reference, which estimates
    scale before the RANSAC loop over rotation/translation) -- computing it
    is undefined for a single pair, so it must be passed in for
    `len(indices) == 1`.

    Raises:
        ValueError: scale is None for a single sample.
    """
    if scale is None:
        if len(indices) < 2:
            raise ValueError(
                "scale must be supplied when fitting a single sample")
        scale = median_pairwise_scale(pred_centers[indices],
                                      gt_centers[indices])

    if len(indices) == 1:
        i = int(indices[0])
        rotation = gt_rotations[i].T @ pred_rotations[i]
    else:
        relative = np.einsum("nij,njk->nik",
                             np.swapaxes(gt_rotations[indices], 1, 2),
                             pred_rotations[indices])
        rotation = transform.Rotation.from_matrix(relative).mean().as_matrix()

    translation = np.mean(gt_centers[indices] - scale *
                          (pred_centers[indices] @ rotation.T),
                          axis=0)
    return float(scale), rotation, translation


def _pose_residuals(
    x: np.ndarray,
    pred_r: np.ndarray,
    pred_c: np.ndarray,
    gt_r: np.ndarray,
    gt_c: np.ndarray,
    rotation_weight: float = 1.0,
) -> np.ndarray:
    """Returns the [N, 6] per-pose residual of a Sim(3).

    The residual is (rotation geodesic, translation) of the Sim(3) `x` =
    (rotvec, translation, log-scale) mapping the predicted poses onto the
    ground truth (`fit_sim3_pose_residuals`'s cost).
    """
    rotation = transform.Rotation.from_rotvec(x[:3]).as_matrix()
    rotation_residual = transform.Rotation.from_matrix(
        np.swapaxes(pred_r @ rotation.T, 1, 2) @ gt_r).as_rotvec()
    translation_residual = np.exp(x[6]) * (pred_c @ rotation.T) + x[3:6] - gt_c
    return np.concatenate(
        [rotation_residual * rotation_weight, translation_residual], axis=1)


def fit_sim3_pose_residuals(
    pred_rotations: np.ndarray,
    pred_centers: np.ndarray,
    gt_rotations: np.ndarray,
    gt_centers: np.ndarray,
    indices: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    rotation_weight: float = 1.0,
    scale: float | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fits a Sim(3): a simplified port of GTSAM's TrajectoryAlignerSim3.

    The reference is gtsam/sfm/TrajectoryAlignerSim3.

    GTSAM jointly optimizes a factor graph over multiple child trajectories
    and (lightly-priored) parent poses. We only ever have one parent (fixed
    ground truth) and one child (the predicted trajectory), so this reduces
    to a plain nonlinear least squares over the 7 Sim(3) parameters
    (rotation vector, translation, log-scale), minimizing a 6D residual per
    matched pose pair: 3D rotation-geodesic error + 3D translation error --
    the same residual shape as GTSAM's `Similarity3::transformFrom`
    expression factor, solved here with `scipy.optimize.least_squares`
    instead of GTSAM's factor graph / Levenberg-Marquardt machinery.

    `scale` fixes the scale (e.g. 1.0 for an SE(3) alignment) and solves
    for rotation and translation only.
    """
    if scale is not None:
        _, init_rotation, init_translation = fit_sim3_dagsfm(pred_rotations,
                                                             pred_centers,
                                                             gt_rotations,
                                                             gt_centers,
                                                             indices,
                                                             scale=scale)
        fixed = np.log(scale)
        result = optimize.least_squares(
            lambda x: _pose_residuals(
                np.append(x, fixed),
                pred_rotations[indices],
                pred_centers[indices],
                gt_rotations[indices],
                gt_centers[indices],
                rotation_weight,
            ).ravel(),
            np.concatenate([
                transform.Rotation.from_matrix(init_rotation).as_rotvec(),
                init_translation
            ]),
            method="lm",
        )
        return float(scale), transform.Rotation.from_rotvec(
            result.x[:3]).as_matrix(), result.x[3:6]

    pred_r = pred_rotations[indices]
    pred_c = pred_centers[indices]
    gt_r = gt_rotations[indices]
    gt_c = gt_centers[indices]

    if x0 is None:
        init_scale, init_rotation, init_translation = fit_sim3_dagsfm(
            pred_rotations, pred_centers, gt_rotations, gt_centers, indices)
        x0 = np.concatenate([
            transform.Rotation.from_matrix(init_rotation).as_rotvec(),
            init_translation,
            [np.log(max(init_scale, 1e-8))],
        ])

    def residuals(x: np.ndarray) -> np.ndarray:
        rotvec, translation, log_scale = x[:3], x[3:6], x[6]
        rotation = transform.Rotation.from_rotvec(rotvec).as_matrix()
        scale = np.exp(log_scale)
        # Predicted world-to-camera rotation composed with the Sim(3)'s
        # world-side rotation (see Sim3Estimate.transform_poses).
        predicted_gt_r = pred_r @ rotation.T
        rotation_residual = transform.Rotation.from_matrix(
            np.swapaxes(predicted_gt_r, 1, 2)
            @ gt_r).as_rotvec() * rotation_weight
        predicted_gt_c = scale * (pred_c @ rotation.T) + translation
        translation_residual = predicted_gt_c - gt_c
        return np.concatenate([rotation_residual, translation_residual],
                              axis=1).ravel()

    result = optimize.least_squares(residuals, x0, method="lm")
    rotvec, translation, log_scale = result.x[:3], result.x[3:6], result.x[6]
    rotation = transform.Rotation.from_rotvec(rotvec).as_matrix()
    return float(np.exp(log_scale)), rotation, translation


def _pose_alignment_errors(
    pred_centers: np.ndarray,
    gt_centers: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Returns the distance of each aligned predicted center to its GT."""
    predicted = scale * (pred_centers @ rotation.T) + translation
    return np.linalg.norm(predicted - gt_centers, axis=1)


def ransac_align_sim3(
    pred_rotations: np.ndarray,
    pred_centers: np.ndarray,
    gt_rotations: np.ndarray,
    gt_centers: np.ndarray,
    *,
    fit_fn: FitFn = fit_sim3_dagsfm,
    sample_size: int | None = None,
    num_iterations: int = 10,
    random_state: int = 42,
) -> Sim3Estimate:
    """Runs the generic LMedS-style RANSAC harness with a `fit_*` strategy.

    `sample_size` defaults to 2 for `fit_sim3_dagsfm` and to 3 otherwise
    (the minimal size `fit_sim3_umeyama`'s SVD needs to be
    well-conditioned; also enough for `fit_sim3_pose_residuals`'s
    7-parameter least squares). Deliberately *not* a large fraction of
    `n`: a minimal sample is far more likely to be entirely outlier-free,
    which is what LMedS selection below relies on.

    `fit_sim3_dagsfm` alone supports a minimal sample of size 1 (matching
    `EuclideanEstimator::SampleSize()` in the reference -- one full pose
    pair already fixes rotation exactly), but this harness does not default
    to it: a size-1 sample can't estimate its own scale, so it would need a
    scale precomputed once from *all* n points via `median_pairwise_scale`
    -- and while that estimator tolerates up to 50% outlier *points* in
    isolation, the fraction of *pairs* touching at least one outlier grows
    combinatorially faster than the point fraction (at 30% point-outliers,
    ~51% of all pairs already touch one), so a globally-precomputed scale
    can break down well before the point-outlier rate does. Measured via a
    seed sweep (see the accompanying hard-case tests): with the size-1
    default, `fit_sim3_dagsfm` reliably recovers scale only up to ~20%
    outliers and its success rate collapses to 0% by 40%; with
    `sample_size=2`, scale is instead estimated locally from each
    minimal sample -- exactly like every other strategy here -- and full
    robustness (matching `fit_sim3_pose_residuals`) is restored at the
    same outlier rates, for a fraction of the cost (no nonlinear solve).
    Callers who explicitly pass `sample_size=1` still get the size-1,
    globally-scaled path via `median_pairwise_scale` below.

    Raises:
        ValueError: The pose arrays are empty or of different lengths.
    """
    pred_rotations = np.asarray(pred_rotations, dtype=np.float64)
    pred_centers = np.asarray(pred_centers, dtype=np.float64)
    gt_rotations = np.asarray(gt_rotations, dtype=np.float64)
    gt_centers = np.asarray(gt_centers, dtype=np.float64)
    num_poses = len(pred_centers)
    if num_poses == 0 or len(gt_centers) != num_poses:
        raise ValueError(
            "pred and gt pose arrays must have matching non-empty lengths")

    all_indices = np.arange(num_poses)
    if sample_size is None:
        sample_size = 2 if fit_fn is fit_sim3_dagsfm else 3
    sample_size = min(sample_size, num_poses)

    if num_poses < 3:
        # Below every strategy's minimal sample size (2 for fit_sim3_dagsfm,
        # 3 otherwise) RANSAC can't meaningfully subsample at all, so always
        # fall back to plain closed-form Umeyama regardless of `fit_fn`:
        # fit_sim3_dagsfm in particular cannot determine scale from a single
        # pose pair (n == 1) without an externally supplied scale, whereas
        # fit_sim3_umeyama degrades gracefully (identity rotation, scale 1)
        # when there's no point spread to estimate from at all.
        scale, rotation, translation = fit_sim3_umeyama(pred_rotations,
                                                        pred_centers,
                                                        gt_rotations,
                                                        gt_centers, all_indices)
        return Sim3Estimate(
            scale=scale,
            rotation=rotation,
            translation=translation,
            inliers=np.ones(num_poses, dtype=bool),
            num_iterations=0,
        )

    # Only reached when a caller explicitly passes sample_size=1 for
    # fit_sim3_dagsfm (see the docstring above for why that isn't the
    # default): a lone pose pair fixes rotation but not scale, so scale
    # must come from somewhere else -- computed once, globally, here.
    if fit_fn is fit_sim3_dagsfm and sample_size == 1:
        global_scale = median_pairwise_scale(pred_centers, gt_centers)
    else:
        global_scale = None

    def _fit(indices: np.ndarray):
        if fit_fn is fit_sim3_dagsfm and len(indices) == 1:
            return fit_fn(pred_rotations,
                          pred_centers,
                          gt_rotations,
                          gt_centers,
                          indices,
                          scale=global_scale)
        return fit_fn(pred_rotations, pred_centers, gt_rotations, gt_centers,
                      indices)

    rng = np.random.default_rng(random_state)
    best_median_sq_error = np.inf
    best = None
    for _ in range(num_iterations):
        sample = rng.choice(num_poses, size=sample_size, replace=False)
        try:
            candidate = _fit(sample)
        except (ValueError, np.linalg.LinAlgError):
            continue
        errors = _pose_alignment_errors(pred_centers, gt_centers, *candidate)
        median_sq_error = float(np.median(errors**2))
        if median_sq_error < best_median_sq_error:
            best_median_sq_error = median_sq_error
            best = candidate

    if best is None:
        best = _fit(all_indices)
        best_median_sq_error = float(
            np.median(
                _pose_alignment_errors(pred_centers, gt_centers, *best)**2))

    # Rousseeuw & Leroy's robust-sigma estimate from the winning (minimal
    # median residual) model, used only now to classify inliers -- never to
    # select among candidates.
    robust_sigma = 1.4826 * (1.0 + 5.0 / max(num_poses - sample_size, 1)
                            ) * np.sqrt(best_median_sq_error)
    threshold = max(2.5 * robust_sigma, 1e-9)

    scale, rotation, translation = best
    final_errors = _pose_alignment_errors(pred_centers, gt_centers, scale,
                                          rotation, translation)
    inliers = final_errors <= threshold
    if int(inliers.sum()) >= sample_size:
        scale, rotation, translation = _fit(np.flatnonzero(inliers))
    else:
        inliers = np.ones(num_poses, dtype=bool)

    return Sim3Estimate(
        scale=scale,
        rotation=rotation,
        translation=translation,
        inliers=inliers,
        num_iterations=num_iterations,
    )


# Cauchy constant for 95 % asymptotic efficiency under Gaussian noise (on
# unit-scale residuals).
CAUCHY_C = 2.3849
# ransac_irls_align_sim3 treats a pose-fit scale below this fraction of the
# center-only fit's as collapsed.
SCALE_COLLAPSE_RATIO = 1e-2


def _robust_sigma(norms: np.ndarray) -> float:
    """Returns 1.4826 * MAD of per-frame residual norms, with a floor.

    The floor is 1.4826 * their median (norms are >= 0, so their MAD alone
    collapses when most frames share one error level).
    """
    median = float(np.median(norms))
    return max(1.4826 * float(np.median(np.abs(norms - median))),
               1.4826 * median, 1e-12)


def _weighted_pose_residuals(
    params: np.ndarray,
    log_scale: float | None,
    sqrt_weights: np.ndarray,
    unit: np.ndarray,
    pred_r: np.ndarray,
    pred_c: np.ndarray,
    gt_r: np.ndarray,
    gt_c: np.ndarray,
) -> np.ndarray:
    """Returns one IRLS step's residual vector.

    That is `_pose_residuals` scaled per pose by `sqrt_weights` and per
    component by `unit` (the robust rotation/translation scales).
    `log_scale` is appended to `params` when the scale is fixed.
    """
    if log_scale is None:
        full = params
    else:
        full = np.append(params, log_scale)
    return (sqrt_weights * _pose_residuals(full, pred_r, pred_c, gt_r, gt_c) /
            unit).ravel()


def irls_refine_sim3(
    pred_rotations: np.ndarray,
    pred_centers: np.ndarray,
    gt_rotations: np.ndarray,
    gt_centers: np.ndarray,
    initial: Sim3Estimate,
    *,
    scale: float | None = None,
    max_iterations: int = 20,
    tol: float = 1e-9,
) -> Sim3Estimate:
    """Robustly refits `initial` on *all* matched poses.

    Minimizes `fit_sim3_pose_residuals`'s cost (rotation geodesic +
    translation per pose) by iteratively reweighted least squares with a
    Cauchy weight on each pose's normalized error.

    Each iteration divides the rotation (rad) and translation residual norms
    by their own robust scale (`_robust_sigma`), so neither unit dominates,
    weights pose `i` by `1 / (1 + (e_i / CAUCHY_C)^2)` with `e_i` the hypot
    of the two normalized norms, and solves the weighted problem with
    Levenberg-Marquardt from the current estimate; it stops when the
    parameters move by less than `tol`. Started from a RANSAC estimate
    (`ransac_irls_align_sim3`), it converges to the same answer from every
    seed whose sample lies in the same basin -- removing the seed dependence
    a minimal-sample LMedS fit has on smoothly drifting trajectories, where
    no subset is exactly right -- while frames with gross errors keep a
    near-zero weight. `scale` fixes the scale (SE(3)). `inliers` marks the
    poses with final weight >= 1/2 (`e_i <= CAUCHY_C`).
    """
    pred_r = np.asarray(pred_rotations, dtype=np.float64)
    pred_c = np.asarray(pred_centers, dtype=np.float64)
    gt_r = np.asarray(gt_rotations, dtype=np.float64)
    gt_c = np.asarray(gt_centers, dtype=np.float64)
    fixed_scale = scale is not None
    x = np.concatenate([
        transform.Rotation.from_matrix(initial.rotation).as_rotvec(),
        initial.translation,
        [np.log(max(scale if fixed_scale else initial.scale, 1e-12))],
    ])

    def normalized_error(params: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Returns the normalized per-pose errors and the two robust scales."""
        residual = _pose_residuals(params, pred_r, pred_c, gt_r, gt_c)
        rot_norm = np.linalg.norm(residual[:, :3], axis=1)
        trans_norm = np.linalg.norm(residual[:, 3:], axis=1)
        sigma_r, sigma_t = _robust_sigma(rot_norm), _robust_sigma(trans_norm)
        return np.hypot(rot_norm / sigma_r,
                        trans_norm / sigma_t), sigma_r, sigma_t

    for _ in range(max_iterations):
        error, sigma_r, sigma_t = normalized_error(x)
        sqrt_w = np.sqrt(1.0 / (1.0 + (error / CAUCHY_C)**2))[:, None]
        unit = np.array([sigma_r] * 3 + [sigma_t] * 3)

        start = x[:6] if fixed_scale else x
        log_scale = x[6] if fixed_scale else None
        solved = optimize.least_squares(_weighted_pose_residuals,
                                        start,
                                        method="lm",
                                        args=(log_scale, sqrt_w, unit, pred_r,
                                              pred_c, gt_r, gt_c)).x
        new_x = np.append(solved, x[6]) if fixed_scale else solved
        step = float(np.linalg.norm(new_x - x))
        x = new_x
        if step < tol:
            break
    error, _, _ = normalized_error(x)
    return Sim3Estimate(
        scale=float(np.exp(x[6])),
        rotation=transform.Rotation.from_rotvec(x[:3]).as_matrix(),
        translation=x[3:6].copy(),
        inliers=error <= CAUCHY_C,
        num_iterations=initial.num_iterations,
    )


def ransac_irls_align_sim3(
    pred_rotations: np.ndarray,
    pred_centers: np.ndarray,
    gt_rotations: np.ndarray,
    gt_centers: np.ndarray,
    *,
    scale: float | None = None,
    num_iterations: int = 25,
    random_state: int = 42,
) -> Sim3Estimate:
    """Aligns predicted to ground-truth poses: the repo-wide pose alignment.

    LMedS RANSAC over `fit_sim3_pose_residuals` (minimal sample 2) followed
    by `irls_refine_sim3` on all poses: the SLAM track's trajectory alignment
    (docs/benchmark/slam_protocol.md, "Trajectory alignment").

    The RANSAC stage escapes tracking-loss jumps and gross outliers a
    least-squares fit is dragged by; the refit makes the result independent
    of the RANSAC seed and as precise as a full least-squares fit on clean
    data. `scale` fixes the scale (SE(3), e.g. 1.0).
    """
    if scale is None:
        fit_fn = fit_sim3_pose_residuals
    else:
        fit_fn = functools.partial(fit_sim3_pose_residuals, scale=scale)
    initial = ransac_align_sim3(
        pred_rotations,
        pred_centers,
        gt_rotations,
        gt_centers,
        fit_fn=fit_fn,
        sample_size=2,
        num_iterations=num_iterations,
        random_state=random_state,
    )
    if len(pred_centers) < 3:
        return initial
    refined = irls_refine_sim3(pred_rotations,
                               pred_centers,
                               gt_rotations,
                               gt_centers,
                               initial,
                               scale=scale)
    if scale is not None:
        return refined
    # When the predicted centers disagree with the ground truth under the
    # rotation the orientations choose, the pose cost falls monotonically as
    # the scale shrinks (every frame mapped to the ground-truth centroid): a
    # degenerate optimum that says nothing about the prediction. Such
    # trajectories are aligned on their camera centers alone (Umeyama in the
    # same RANSAC harness, minimal sample 3), as evo does, and flagged.
    centers_only = ransac_align_sim3(
        pred_rotations,
        pred_centers,
        gt_rotations,
        gt_centers,
        fit_fn=fit_sim3_umeyama,
        sample_size=3,
        num_iterations=num_iterations,
        random_state=random_state,
    )
    if refined.scale < SCALE_COLLAPSE_RATIO * centers_only.scale:
        return dataclasses.replace(centers_only, scale_collapsed=True)
    return refined


def icp_refine_sim3(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    initial: Sim3Estimate,
    *,
    max_iterations: int = 20,
    tolerance: float = 1e-9,
    max_correspondence_distance: float | None = None,
) -> Sim3Estimate:
    """Refines a Sim(3) estimate with point-to-point ICP (Besl & McKay, 1992).

    `max_correspondence_distance` (like Open3D's `registration_icp`
    threshold) keeps only nearest-neighbor pairs closer than that: without it
    every predicted point -- background the GT does not cover included --
    pulls on the fit, and a scaled ICP on partial overlap slides and rotates
    away from a correct start.

    Both Pi3's and DA3's own reference evaluation code follow their coarse
    Umeyama/RANSAC alignment with an ICP refinement pass on the dense
    reconstructed point cloud (Pi3's `mv_recon/eval.py`, credited there to
    CUT3R's evaluation code). This repo's Tanks-and-Temples/DTU mesh
    alignment previously used a `refine_icp` helper wrapping Open3D's
    `registration_icp`, which is *rigid*-only -- it has no scale degree of
    freedom, so it could not correct any residual scale error left over from
    the pose-based Sim(3) fit that ran before fusion. This implements the
    scaled generalization directly instead of adding a dependency: each
    iteration finds nearest-neighbor correspondences under the current
    Sim(3) via a KD-tree, then re-solves the closed-form similarity fit
    (`align_points_similarity`) on those correspondences.

    Works on any point set -- camera centers (small, exact correspondence
    already known from RANSAC; ICP still helps when per-frame position
    noise means the best *rigid-up-to-scale* fit isn't a perfect Umeyama
    solution) or dense fused point clouds (large, no known correspondence
    at all -- the original Pi3/DA3 use case). Falls back to returning
    `initial` unchanged if either point set is empty.
    """
    # pylint: disable-next=import-outside-toplevel
    from scipy import spatial

    pred_points = np.asarray(pred_points, dtype=np.float64)
    gt_points = np.asarray(gt_points, dtype=np.float64)
    if len(pred_points) == 0 or len(gt_points) == 0:
        return initial

    scale = initial.scale
    rotation = initial.rotation
    translation = initial.translation
    gt_tree = spatial.cKDTree(gt_points)
    prev_error = np.inf
    for _ in range(max_iterations):
        transformed = scale * (pred_points @ rotation.T) + translation
        distances, nn_indices = gt_tree.query(transformed)
        keep = np.ones(len(distances), dtype=bool)
        if max_correspondence_distance is not None:
            keep = distances < max_correspondence_distance
            if keep.sum() < 3:
                break
        matched_gt = gt_points[nn_indices[keep]]
        try:
            _, scale, rotation, translation = (
                sparse_reconstruction_evaluator.align_points_similarity(
                    pred_points[keep], matched_gt))
        except (ValueError, np.linalg.LinAlgError):
            break
        mean_error = float(np.mean(distances[keep]))
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    return Sim3Estimate(
        scale=scale,
        rotation=rotation,
        translation=translation,
        inliers=initial.inliers,
        num_iterations=initial.num_iterations,
    )


def ransac_icp_align_sim3(
    pred_rotations: np.ndarray,
    pred_centers: np.ndarray,
    gt_rotations: np.ndarray,
    gt_centers: np.ndarray,
    *,
    fit_fn: FitFn = fit_sim3_dagsfm,
    sample_size: int | None = None,
    num_iterations: int = 25,
    random_state: int = 42,
    icp_points: tuple[np.ndarray, np.ndarray] | None = None,
    icp_max_iterations: int = 20,
) -> Sim3Estimate:
    """Runs the repo's previous default Sim(3) alignment pipeline.

    Superseded by `ransac_irls_align_sim3`; kept for comparison: pose-aware
    RANSAC (`fit_sim3_dagsfm` by default) followed by scaled ICP refinement
    (`icp_refine_sim3`).

    This is exactly the combination originally proposed against the
    degenerate camera-center-only Umeyama fit both Pi3's and DA3's own
    reference evaluation code use: RANSAC for outlier rejection (Pi3/DA3
    both already do *some* form of this, but on centers only) plus ICP
    refinement (Pi3's own eval code, credited there to CUT3R, unconditionally
    ICP-refines after its coarse fit) -- except the coarse fit here is
    pose-aware, so it isn't degenerate on near-collinear trajectories in the
    first place, and RANSAC now uses LMedS selection so a contaminated
    full-sample fit can't poison its own inlier threshold (see
    `ransac_align_sim3`'s docstring).

    `fit_sim3_dagsfm` is the default fit strategy: a seed sweep across
    outlier ratios from 20-45%, with and without the original degenerate
    (collinear) case, showed it matches `fit_sim3_pose_residuals`'s
    robustness exactly once wrapped with `sample_size=2` (the harness's
    own default -- see `ransac_align_sim3`), at roughly 1/30th the cost
    per RANSAC candidate (closed-form vs. a nonlinear least-squares solve
    per candidate). `fit_sim3_pose_residuals` (the GTSAM-style port) and
    `fit_sim3_umeyama` (the Pi3/DA3 baseline) remain available via
    `fit_fn` for comparison or for callers with different tradeoffs.

    `icp_points`, if given, is a `(pred_points, gt_points)` pair of
    dense reconstructed point clouds with *no known correspondence* --
    matching Pi3's/DA3's original dense-point ICP use case, where ICP's
    nearest-neighbor matching does real work. If omitted (the default),
    ICP instead refines against the RANSAC inlier *camera centers*, whose
    correspondence is already known; this still helps when per-frame
    position noise means the pose-aware fit isn't already the exact
    closed-form optimum for those centers, but is a much smaller
    correction than the dense-point case (see
    `test_ransac_then_icp_pipeline_recovers_sim3_on_noisy_trajectory` in
    `test/test_pose_alignment.py`).
    """
    estimate = ransac_align_sim3(
        pred_rotations,
        pred_centers,
        gt_rotations,
        gt_centers,
        fit_fn=fit_fn,
        sample_size=sample_size,
        num_iterations=num_iterations,
        random_state=random_state,
    )
    if icp_points is not None:
        pred_points, gt_points = icp_points
    else:
        pred_points = np.asarray(pred_centers,
                                 dtype=np.float64)[estimate.inliers]
        gt_points = np.asarray(gt_centers, dtype=np.float64)[estimate.inliers]
    return icp_refine_sim3(pred_points,
                           gt_points,
                           estimate,
                           max_iterations=icp_max_iterations)
