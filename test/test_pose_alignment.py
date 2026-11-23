# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Compare Sim(3) trajectory-alignment strategies, including a case
(near-collinear camera centers) that is known to be ill-conditioned for the
camera-center-only Umeyama fit that both Pi3's and DA3's own reference
evaluation code use.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from spatialarena.eval.pose_alignment import (
    Sim3Estimate,
    fit_sim3_dagsfm,
    fit_sim3_pose_residuals,
    fit_sim3_umeyama,
    icp_refine_sim3,
    irls_refine_sim3,
    median_pairwise_scale,
    ransac_align_sim3,
    ransac_icp_align_sim3,
    ransac_irls_align_sim3,
)

FIT_FUNCTIONS = {
    "umeyama_points": fit_sim3_umeyama,
    "dagsfm": fit_sim3_dagsfm,
    "pose_residuals": fit_sim3_pose_residuals,
}


def _make_gt_trajectory(num_poses: int, *, collinear: bool,
                        rng: np.random.Generator):
    if collinear:
        gt_centers = np.stack(
            [np.array([float(i), 0.0, 0.0]) for i in range(num_poses)])
    else:
        gt_centers = rng.normal(scale=5.0, size=(num_poses, 3))
    # Independent random orientation per frame regardless of position --
    # this is what makes rotation recoverable from a single pose even when
    # centers are collinear: orientation carries information position alone
    # does not.
    gt_rotations = Rotation.random(num_poses, random_state=rng).as_matrix()
    return gt_rotations, gt_centers


def _apply_inverse_sim3(
    gt_rotations: np.ndarray,
    gt_centers: np.ndarray,
    *,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert Sim3Estimate.transform_poses to synthesize a "predicted"
    trajectory whose true alignment to `gt` is exactly (scale, rotation,
    translation)."""
    pred_rotations = gt_rotations @ rotation
    pred_centers = ((gt_centers - translation) / scale) @ rotation
    return pred_rotations, pred_centers


def _random_sim3(
        rng: np.random.Generator) -> tuple[float, np.ndarray, np.ndarray]:
    scale = float(rng.uniform(0.3, 3.0))
    rotation = Rotation.random(random_state=rng).as_matrix()
    translation = rng.normal(scale=2.0, size=3)
    return scale, rotation, translation


def _rotation_error_deg(estimated: np.ndarray, truth: np.ndarray) -> float:
    relative = estimated.T @ truth
    trace = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(trace)))


@pytest.mark.parametrize("fit_name", list(FIT_FUNCTIONS))
def test_recovers_known_sim3_on_well_conditioned_trajectory(fit_name):
    rng = np.random.default_rng(0)
    gt_rotations, gt_centers = _make_gt_trajectory(20, collinear=False, rng=rng)
    true_scale, true_rotation, true_translation = _random_sim3(rng)
    pred_rotations, pred_centers = _apply_inverse_sim3(
        gt_rotations,
        gt_centers,
        scale=true_scale,
        rotation=true_rotation,
        translation=true_translation)

    scale, rotation, translation = FIT_FUNCTIONS[fit_name](pred_rotations,
                                                           pred_centers,
                                                           gt_rotations,
                                                           gt_centers,
                                                           np.arange(20))

    assert scale == pytest.approx(true_scale, rel=1e-3)
    assert _rotation_error_deg(rotation, true_rotation) < 0.1
    np.testing.assert_allclose(translation, true_translation, atol=1e-3)


def test_umeyama_on_points_degrades_on_collinear_centers_but_pose_aware_fits_do_not(
):
    """The actual comparison the plan called for: on a degenerate (collinear
    camera-center) trajectory, camera-center-only Umeyama -- what Pi3's and
    DA3's own reference evaluation code, and this repo's
    mvrecon_evaluator.py::_align_da3_poses, use -- should recover rotation
    far worse than the two pose-aware strategies, which use each frame's
    orientation and are unaffected by center collinearity.
    """
    rng = np.random.default_rng(1)
    gt_rotations, gt_centers = _make_gt_trajectory(20, collinear=True, rng=rng)
    true_scale, true_rotation, true_translation = _random_sim3(rng)
    pred_rotations, pred_centers = _apply_inverse_sim3(
        gt_rotations,
        gt_centers,
        scale=true_scale,
        rotation=true_rotation,
        translation=true_translation)
    indices = np.arange(20)

    errors = {}
    for name, fit_fn in FIT_FUNCTIONS.items():
        _, rotation, _ = fit_fn(pred_rotations, pred_centers, gt_rotations,
                                gt_centers, indices)
        errors[name] = _rotation_error_deg(rotation, true_rotation)

    # Umeyama-on-points must be markedly worse than both pose-aware fits on
    # this degenerate trajectory -- the whole point of the comparison.
    assert errors["umeyama_points"] > 1.0
    assert errors["dagsfm"] < 0.1
    assert errors["pose_residuals"] < 0.1
    assert errors["umeyama_points"] > 10 * max(errors["dagsfm"],
                                               errors["pose_residuals"])


def test_dagsfm_single_sample_matches_analytic_relative_rotation():
    """Sanity-checks the ported formula against the derivation directly:
    for one pose pair, R = gt_R.T @ pred_R exactly, no fitting involved."""
    rng = np.random.default_rng(2)
    gt_rotations, gt_centers = _make_gt_trajectory(5, collinear=False, rng=rng)
    true_scale, true_rotation, true_translation = _random_sim3(rng)
    pred_rotations, pred_centers = _apply_inverse_sim3(
        gt_rotations,
        gt_centers,
        scale=true_scale,
        rotation=true_rotation,
        translation=true_translation)

    _scale, rotation, translation = fit_sim3_dagsfm(pred_rotations,
                                                    pred_centers,
                                                    gt_rotations,
                                                    gt_centers,
                                                    np.array([0]),
                                                    scale=true_scale)
    np.testing.assert_allclose(rotation, true_rotation, atol=1e-8)
    np.testing.assert_allclose(translation, true_translation, atol=1e-8)


def test_median_pairwise_scale_matches_known_scale():
    rng = np.random.default_rng(3)
    gt_rotations, gt_centers = _make_gt_trajectory(15, collinear=False, rng=rng)
    true_scale, true_rotation, true_translation = _random_sim3(rng)
    _, pred_centers = _apply_inverse_sim3(gt_rotations,
                                          gt_centers,
                                          scale=true_scale,
                                          rotation=true_rotation,
                                          translation=true_translation)
    assert median_pairwise_scale(pred_centers,
                                 gt_centers) == pytest.approx(true_scale,
                                                              rel=1e-6)


@pytest.mark.parametrize("fit_name", list(FIT_FUNCTIONS))
def test_ransac_rejects_outlier_poses(fit_name):
    rng = np.random.default_rng(4)
    gt_rotations, gt_centers = _make_gt_trajectory(40, collinear=False, rng=rng)
    true_scale, true_rotation, true_translation = _random_sim3(rng)
    pred_rotations, pred_centers = _apply_inverse_sim3(
        gt_rotations,
        gt_centers,
        scale=true_scale,
        rotation=true_rotation,
        translation=true_translation)

    # Corrupt a minority of predicted poses with unrelated random poses.
    outlier_count = 8
    outlier_indices = rng.choice(40, size=outlier_count, replace=False)
    pred_centers = pred_centers.copy()
    pred_rotations = pred_rotations.copy()
    pred_centers[outlier_indices] = rng.normal(scale=50.0,
                                               size=(outlier_count, 3))
    pred_rotations[outlier_indices] = Rotation.random(
        outlier_count, random_state=rng).as_matrix()

    estimate = ransac_align_sim3(
        pred_rotations,
        pred_centers,
        gt_rotations,
        gt_centers,
        fit_fn=FIT_FUNCTIONS[fit_name],
        num_iterations=25,
        random_state=42,
    )

    assert estimate.scale == pytest.approx(true_scale, rel=1e-2)
    assert _rotation_error_deg(estimate.rotation, true_rotation) < 1.0
    np.testing.assert_allclose(estimate.translation, true_translation, atol=0.1)
    assert not np.any(estimate.inliers[outlier_indices])


def _inject_outliers(
    pred_rotations: np.ndarray,
    pred_centers: np.ndarray,
    *,
    outlier_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Corrupt a fraction of poses in place; returns the outlier indices."""
    n = len(pred_centers)
    outlier_count = int(round(outlier_ratio * n))
    outlier_indices = rng.choice(n, size=outlier_count, replace=False)
    pred_centers[outlier_indices] = rng.normal(scale=50.0,
                                               size=(outlier_count, 3))
    pred_rotations[outlier_indices] = Rotation.random(
        outlier_count, random_state=rng).as_matrix()
    return outlier_indices


@pytest.mark.parametrize("collinear", [False, True],
                         ids=["well_conditioned", "collinear"])
@pytest.mark.parametrize("outlier_ratio", [0.3, 0.4])
@pytest.mark.parametrize("fit_name", ["dagsfm", "pose_residuals"])
def test_ransac_survives_high_outlier_ratios(fit_name, outlier_ratio,
                                             collinear):
    """Harder than test_ransac_rejects_outlier_poses's 20%: pushes to 30% and
    40% outliers, and combines that with the original degenerate (collinear)
    case. A seed sweep (30 seeds, not run in CI) showed fit_sim3_dagsfm's
    *default* single-pose minimal sample (sample_size=1) collapses to 0%
    success by 40% outliers -- its scale comes from one global
    median-of-all-pairwise-ratios statistic, and the fraction of *pairs*
    touching an outlier grows combinatorially faster than the point
    fraction (30% point-outliers already contaminates ~51% of all pairs).
    ``ransac_align_sim3`` now defaults fit_sim3_dagsfm to sample_size=2
    instead (scale estimated locally per minimal sample, like every other
    strategy), which the same sweep confirmed restores 100% success at
    these same outlier rates, matching fit_sim3_pose_residuals -- this test
    locks that fix in place.
    """
    rng = np.random.default_rng(40)
    gt_rotations, gt_centers = _make_gt_trajectory(40,
                                                   collinear=collinear,
                                                   rng=rng)
    true_scale, true_rotation, true_translation = _random_sim3(rng)
    pred_rotations, pred_centers = _apply_inverse_sim3(
        gt_rotations,
        gt_centers,
        scale=true_scale,
        rotation=true_rotation,
        translation=true_translation)
    outlier_indices = _inject_outliers(pred_rotations,
                                       pred_centers,
                                       outlier_ratio=outlier_ratio,
                                       rng=rng)

    estimate = ransac_align_sim3(
        pred_rotations,
        pred_centers,
        gt_rotations,
        gt_centers,
        fit_fn=FIT_FUNCTIONS[fit_name],
        num_iterations=40,
        random_state=42,
    )

    assert estimate.scale == pytest.approx(true_scale, rel=0.05)
    assert _rotation_error_deg(estimate.rotation, true_rotation) < 1.0
    assert not np.any(estimate.inliers[outlier_indices])


def test_dagsfm_default_sample_size_beats_minimal_single_pose_sample_at_high_outlier_ratio(
):
    """Directly demonstrates the diagnosed root cause fixed above: with the
    *same* 40%-outlier trajectory, fit_sim3_dagsfm wrapped at sample_size=1
    (a global, less-robust scale statistic) performs far worse than at the
    harness's actual default sample_size=2 (a local, per-sample scale).
    """
    rng = np.random.default_rng(41)
    gt_rotations, gt_centers = _make_gt_trajectory(40, collinear=False, rng=rng)
    true_scale, true_rotation, true_translation = _random_sim3(rng)
    pred_rotations, pred_centers = _apply_inverse_sim3(
        gt_rotations,
        gt_centers,
        scale=true_scale,
        rotation=true_rotation,
        translation=true_translation)
    _inject_outliers(pred_rotations, pred_centers, outlier_ratio=0.4, rng=rng)

    degenerate = ransac_align_sim3(
        pred_rotations,
        pred_centers,
        gt_rotations,
        gt_centers,
        fit_fn=fit_sim3_dagsfm,
        sample_size=1,
        num_iterations=40,
        random_state=42,
    )
    fixed_default = ransac_align_sim3(
        pred_rotations,
        pred_centers,
        gt_rotations,
        gt_centers,
        fit_fn=fit_sim3_dagsfm,
        num_iterations=40,
        random_state=42,
    )

    degenerate_scale_err = abs(degenerate.scale - true_scale) / true_scale
    fixed_scale_err = abs(fixed_default.scale - true_scale) / true_scale
    assert fixed_scale_err < 0.05
    assert degenerate_scale_err > 10 * fixed_scale_err


def test_sim3_estimate_transform_poses_round_trips_apply_inverse():
    rng = np.random.default_rng(5)
    gt_rotations, gt_centers = _make_gt_trajectory(10, collinear=False, rng=rng)
    true_scale, true_rotation, true_translation = _random_sim3(rng)
    pred_rotations, pred_centers = _apply_inverse_sim3(
        gt_rotations,
        gt_centers,
        scale=true_scale,
        rotation=true_rotation,
        translation=true_translation)

    estimate = Sim3Estimate(
        scale=true_scale,
        rotation=true_rotation,
        translation=true_translation,
        inliers=np.ones(10, dtype=bool),
    )
    aligned_rotations, aligned_centers = estimate.transform_poses(
        pred_rotations, pred_centers)
    np.testing.assert_allclose(aligned_centers, gt_centers, atol=1e-8)
    np.testing.assert_allclose(aligned_rotations, gt_rotations, atol=1e-8)


def test_icp_refine_sim3_improves_alignment_on_noisy_dense_point_cloud():
    """The step-3 use case: a dense point cloud with *no known correspondence*
    between predicted and GT points (unlike camera centers, where index i in
    `pred` always corresponds to index i in `gt`). RANSAC on camera poses
    gives a coarse initial Sim(3); ICP's nearest-neighbor correspondence loop
    should then reduce the residual further, since the coarse fit alone
    cannot account for the independent per-point noise added below.
    """
    rng = np.random.default_rng(6)
    num_points = 500
    gt_points = rng.normal(scale=5.0, size=(num_points, 3))
    true_scale, true_rotation, true_translation = _random_sim3(rng)

    # Predicted points are the true inverse-Sim3-mapped GT points, each
    # perturbed by independent noise -- so the "coarse" alignment (fit on
    # the noiseless correspondence) will not perfectly match the noisy
    # dense cloud, giving ICP something real to improve.
    pred_points = ((gt_points - true_translation) / true_scale) @ true_rotation
    pred_points += rng.normal(scale=0.05, size=pred_points.shape)
    # Shuffle so there is no assumed index correspondence, matching the
    # real dense-point-cloud scenario (unlike camera poses).
    rng.shuffle(pred_points)

    initial = Sim3Estimate(
        scale=true_scale,
        rotation=true_rotation,
        translation=true_translation,
        inliers=np.ones(num_points, dtype=bool),
    )
    initial_errors = np.linalg.norm(
        initial.transform_points(pred_points)[:, None] - gt_points[None, :],
        axis=-1).min(axis=1)

    refined = icp_refine_sim3(pred_points, gt_points, initial)
    refined_errors = np.linalg.norm(
        refined.transform_points(pred_points)[:, None] - gt_points[None, :],
        axis=-1).min(axis=1)

    assert float(np.mean(refined_errors)) <= float(np.mean(initial_errors))
    assert refined.scale == pytest.approx(true_scale, rel=0.05)
    assert _rotation_error_deg(refined.rotation, true_rotation) < 1.0


def test_ransac_then_icp_pipeline_recovers_sim3_on_noisy_trajectory():
    """End-to-end step 1 -> 2 -> 3: pose-aware fit, wrapped in RANSAC, refined
    by ICP -- run for both pose-aware strategies (DAGSfM- and GTSAM-style) on
    a trajectory with outlier poses *and* small per-frame position noise, to
    confirm ICP refinement is a strict (or neutral) improvement on top of
    RANSAC alone once outliers are already rejected.
    """
    for fit_name in ("dagsfm", "pose_residuals"):
        rng = np.random.default_rng(7)
        gt_rotations, gt_centers = _make_gt_trajectory(40,
                                                       collinear=False,
                                                       rng=rng)
        true_scale, true_rotation, true_translation = _random_sim3(rng)
        pred_rotations, pred_centers = _apply_inverse_sim3(
            gt_rotations,
            gt_centers,
            scale=true_scale,
            rotation=true_rotation,
            translation=true_translation)
        pred_centers = pred_centers + rng.normal(scale=0.02,
                                                 size=pred_centers.shape)

        outlier_count = 8
        outlier_indices = rng.choice(40, size=outlier_count, replace=False)
        pred_centers[outlier_indices] = rng.normal(scale=50.0,
                                                   size=(outlier_count, 3))
        pred_rotations[outlier_indices] = Rotation.random(
            outlier_count, random_state=rng).as_matrix()

        ransac_estimate = ransac_align_sim3(
            pred_rotations,
            pred_centers,
            gt_rotations,
            gt_centers,
            fit_fn={
                "dagsfm": fit_sim3_dagsfm,
                "pose_residuals": fit_sim3_pose_residuals
            }[fit_name],
            num_iterations=25,
            random_state=42,
        )
        inlier_mask = ransac_estimate.inliers
        refined = icp_refine_sim3(pred_centers[inlier_mask],
                                  gt_centers[inlier_mask], ransac_estimate)

        ransac_error = float(
            np.mean(
                np.linalg.norm(
                    ransac_estimate.transform_points(pred_centers[inlier_mask])
                    - gt_centers[inlier_mask],
                    axis=1,
                )))
        refined_error = float(
            np.mean(
                np.linalg.norm(
                    refined.transform_points(pred_centers[inlier_mask]) -
                    gt_centers[inlier_mask],
                    axis=1)))
        # Here correspondence is already exact (index i in pred is index i in
        # gt), unlike the dense-point-cloud case above -- so ICP's point-only
        # Umeyama refit isn't guaranteed to beat the pose-aware fit, which
        # also used orientation info ICP discards. It should stay close
        # (no meaningful degradation), not necessarily improve.
        assert refined_error <= ransac_error * 1.05, fit_name


def test_ransac_align_sim3_defaults_to_dagsfm():
    """fit_sim3_dagsfm is this repo's default fit strategy (see the module
    docstring): both explicit and default calls must agree exactly.
    """
    rng = np.random.default_rng(8)
    gt_rotations, gt_centers = _make_gt_trajectory(20, collinear=False, rng=rng)
    true_scale, true_rotation, true_translation = _random_sim3(rng)
    pred_rotations, pred_centers = _apply_inverse_sim3(
        gt_rotations,
        gt_centers,
        scale=true_scale,
        rotation=true_rotation,
        translation=true_translation)

    default_estimate = ransac_align_sim3(pred_rotations, pred_centers,
                                         gt_rotations, gt_centers)
    explicit_estimate = ransac_align_sim3(pred_rotations,
                                          pred_centers,
                                          gt_rotations,
                                          gt_centers,
                                          fit_fn=fit_sim3_dagsfm)

    assert default_estimate.scale == explicit_estimate.scale
    np.testing.assert_array_equal(default_estimate.rotation,
                                  explicit_estimate.rotation)
    np.testing.assert_array_equal(default_estimate.translation,
                                  explicit_estimate.translation)


@pytest.mark.parametrize("num_poses", [1, 2])
def test_ransac_align_sim3_handles_fewer_than_minimal_sample_size_with_dagsfm_default(
        num_poses):
    """Regression test: a scene with only 1-2 matched views (below every
    strategy's minimal sample size -- 2 for fit_sim3_dagsfm, 3 otherwise) must
    not crash under the dagsfm default. fit_sim3_dagsfm alone cannot
    determine scale from a single pose pair without an externally supplied
    scale (this broke ``mvrecon_evaluator.py::_align_da3_poses`` on a
    single-view scene before ``ransac_align_sim3`` was fixed to always fall
    back to plain closed-form Umeyama below the minimal sample size).
    """
    rng = np.random.default_rng(11)
    gt_rotations, gt_centers = _make_gt_trajectory(num_poses,
                                                   collinear=False,
                                                   rng=rng)
    true_scale, true_rotation, true_translation = _random_sim3(rng)
    pred_rotations, pred_centers = _apply_inverse_sim3(
        gt_rotations,
        gt_centers,
        scale=true_scale,
        rotation=true_rotation,
        translation=true_translation)

    estimate = ransac_align_sim3(pred_rotations, pred_centers, gt_rotations,
                                 gt_centers)

    assert np.isfinite(estimate.scale)
    assert np.all(np.isfinite(estimate.rotation))
    assert np.all(np.isfinite(estimate.translation))
    assert estimate.inliers.sum() == num_poses


def test_ransac_icp_align_sim3_recovers_sim3_with_outliers_and_noise():
    """End-to-end default pipeline (dagsfm + RANSAC + ICP on camera centers):
    outlier poses plus small per-frame noise, no dense point cloud supplied.
    """
    rng = np.random.default_rng(9)
    gt_rotations, gt_centers = _make_gt_trajectory(40, collinear=False, rng=rng)
    true_scale, true_rotation, true_translation = _random_sim3(rng)
    pred_rotations, pred_centers = _apply_inverse_sim3(
        gt_rotations,
        gt_centers,
        scale=true_scale,
        rotation=true_rotation,
        translation=true_translation)
    pred_centers = pred_centers + rng.normal(scale=0.02,
                                             size=pred_centers.shape)
    outlier_indices = _inject_outliers(pred_rotations,
                                       pred_centers,
                                       outlier_ratio=0.2,
                                       rng=rng)

    estimate = ransac_icp_align_sim3(pred_rotations,
                                     pred_centers,
                                     gt_rotations,
                                     gt_centers,
                                     random_state=42)

    assert estimate.scale == pytest.approx(true_scale, rel=0.05)
    assert _rotation_error_deg(estimate.rotation, true_rotation) < 1.0
    assert not np.any(estimate.inliers[outlier_indices])


def test_ransac_icp_align_sim3_uses_supplied_dense_point_cloud():
    """When `icp_points` is given, ICP refines against it instead of the
    camera centers -- the real Pi3/DA3 use case (dense points, unknown
    correspondence), with RANSAC's pose-based outlier rejection running
    first on the (separate) camera poses.
    """
    rng = np.random.default_rng(10)
    gt_rotations, gt_centers = _make_gt_trajectory(40, collinear=False, rng=rng)
    true_scale, true_rotation, true_translation = _random_sim3(rng)
    pred_rotations, pred_centers = _apply_inverse_sim3(
        gt_rotations,
        gt_centers,
        scale=true_scale,
        rotation=true_rotation,
        translation=true_translation)
    _inject_outliers(pred_rotations, pred_centers, outlier_ratio=0.2, rng=rng)

    num_points = 300
    gt_points = rng.normal(scale=5.0, size=(num_points, 3))
    pred_points = ((gt_points - true_translation) / true_scale) @ true_rotation
    pred_points += rng.normal(scale=0.05, size=pred_points.shape)
    rng.shuffle(pred_points)

    estimate = ransac_icp_align_sim3(
        pred_rotations,
        pred_centers,
        gt_rotations,
        gt_centers,
        random_state=42,
        icp_points=(pred_points, gt_points),
    )

    assert estimate.scale == pytest.approx(true_scale, rel=0.05)
    assert _rotation_error_deg(estimate.rotation, true_rotation) < 1.0


def _drifting_trajectory(num_poses: int = 120, seed: int = 4):
    """GT on a curve plus a prediction = known Sim(3) of the GT with random-walk drift on centers and orientations
    (no frame is exactly right, so a minimal-sample RANSAC fit depends on which frames it samples)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2.0 * np.pi, num_poses)
    gt_centers = np.stack(
        [3.0 * np.cos(t), 2.0 * np.sin(t), 0.3 * np.sin(3 * t)], axis=1)
    gt_rotations = Rotation.from_rotvec(
        np.stack([0.1 * np.sin(t), 0.2 * np.cos(t), t], axis=1)).as_matrix()
    drifted = gt_centers + np.cumsum(
        rng.normal(scale=0.05, size=gt_centers.shape), axis=0)
    drifted_rotations = Rotation.from_rotvec(
        np.cumsum(rng.normal(scale=0.02, size=(num_poses, 3)),
                  axis=0)).as_matrix()
    true = Sim3Estimate(scale=0.4,
                        rotation=Rotation.from_rotvec([0.3, -1.0,
                                                       0.5]).as_matrix(),
                        translation=np.array([2.0, 0.0, -1.0]),
                        inliers=np.ones(num_poses, bool))
    # pred = true^-1 applied to the drifted GT (world-to-camera rotations compose on the world side).
    pred_centers = ((drifted - true.translation) @ true.rotation) / true.scale
    pred_rotations = drifted_rotations @ gt_rotations @ true.rotation
    return pred_rotations, pred_centers, gt_rotations, gt_centers


def test_ransac_irls_is_independent_of_the_ransac_seed_on_a_drifting_trajectory(
):
    """RANSAC alone moves with the seed under drift; the robust refit converges to one answer."""
    args = _drifting_trajectory()
    plain = [
        ransac_align_sim3(*args,
                          fit_fn=fit_sim3_pose_residuals,
                          sample_size=2,
                          num_iterations=25,
                          random_state=seed) for seed in range(6)
    ]
    refit = [
        ransac_irls_align_sim3(*args, random_state=seed) for seed in range(6)
    ]
    # Without the refit, the answer moves with the seed; with it, every seed lands on the same fit.
    assert np.ptp([estimate.scale for estimate in plain]) > 1e-2
    assert np.ptp([estimate.scale for estimate in refit]) < 1e-6
    for estimate in refit[1:]:
        np.testing.assert_allclose(estimate.rotation,
                                   refit[0].rotation,
                                   atol=1e-6)
        np.testing.assert_allclose(estimate.translation,
                                   refit[0].translation,
                                   atol=1e-6)


@pytest.mark.parametrize("collinear", [False, True])
def test_ransac_irls_recovers_exact_similarity_despite_gross_outliers(
        collinear):
    """30 % gross outliers: exact Sim(3) and the outliers flagged, also on a straight path."""
    rng = np.random.default_rng(7)
    num_poses = 60
    gt_rotations = Rotation.random(num_poses, random_state=1).as_matrix()
    if collinear:
        gt_centers = np.stack([
            np.linspace(0.0, 30.0, num_poses),
            np.zeros(num_poses),
            np.zeros(num_poses)
        ],
                              axis=1)
    else:
        gt_centers = rng.normal(scale=5.0, size=(num_poses, 3))
    rotation = Rotation.from_rotvec([0.4, 0.2, -1.3]).as_matrix()
    scale, translation = 2.5, np.array([1.0, -2.0, 3.0])
    pred_centers = ((gt_centers - translation) @ rotation) / scale
    pred_rotations = gt_rotations @ rotation
    outliers = rng.choice(num_poses, size=int(0.3 * num_poses), replace=False)
    pred_centers[outliers] += rng.normal(scale=20.0, size=(len(outliers), 3))
    estimate = ransac_irls_align_sim3(pred_rotations, pred_centers,
                                      gt_rotations, gt_centers)
    assert estimate.scale == pytest.approx(scale, rel=1e-6)
    np.testing.assert_allclose(estimate.rotation, rotation, atol=1e-6)
    np.testing.assert_allclose(estimate.translation, translation, atol=1e-5)
    expected = np.ones(num_poses, bool)
    expected[outliers] = False
    np.testing.assert_array_equal(estimate.inliers, expected)


def test_ransac_irls_with_fixed_scale_solves_se3_only():
    """``scale=1.0`` keeps the scale fixed and returns a proper rotation."""
    pred_rotations, pred_centers, gt_rotations, gt_centers = _drifting_trajectory(
    )
    estimate = ransac_irls_align_sim3(pred_rotations,
                                      pred_centers,
                                      gt_rotations,
                                      gt_centers,
                                      scale=1.0)
    assert estimate.scale == 1.0
    np.testing.assert_allclose(estimate.rotation @ estimate.rotation.T,
                               np.eye(3),
                               atol=1e-9)


def test_irls_refine_converges_from_a_perturbed_start_to_the_exact_answer():
    """Noise-free poses: the refit reaches the identity from a perturbed start."""
    rng = np.random.default_rng(0)
    gt_rotations = Rotation.random(40, random_state=2).as_matrix()
    gt_centers = rng.normal(size=(40, 3))
    estimate = irls_refine_sim3(
        gt_rotations,
        gt_centers,
        gt_rotations,
        gt_centers,
        Sim3Estimate(scale=1.3,
                     rotation=Rotation.from_rotvec([0.1, 0.0, 0.2]).as_matrix(),
                     translation=np.array([0.2, 0.1, 0.0]),
                     inliers=np.ones(40, bool)),
    )
    assert estimate.scale == pytest.approx(1.0, abs=1e-8)
    np.testing.assert_allclose(estimate.rotation, np.eye(3), atol=1e-8)
    np.testing.assert_allclose(estimate.translation, np.zeros(3), atol=1e-8)


def test_ransac_irls_falls_back_to_centers_when_the_pose_fit_collapses_the_scale(
):
    """Orientations that agree with the ground truth while the centers run backwards: the pose cost is lowest at
    scale -> 0, so the alignment falls back to the center-only fit and flags it; a normal case is not flagged."""
    rng = np.random.default_rng(0)
    gt_centers = np.c_[np.linspace(0.0, 10.0, 60),
                       rng.normal(0.0, 0.3, (60, 2))]
    gt_rotations = np.tile(np.eye(3), (60, 1, 1))
    pred_centers = -gt_centers + rng.normal(0.0, 0.05, gt_centers.shape)
    estimate = ransac_irls_align_sim3(gt_rotations.copy(), pred_centers,
                                      gt_rotations, gt_centers)
    assert estimate.scale_collapsed
    assert estimate.scale == pytest.approx(1.0, rel=0.05)
    normal = ransac_irls_align_sim3(*_drifting_trajectory())
    assert not normal.scale_collapsed
