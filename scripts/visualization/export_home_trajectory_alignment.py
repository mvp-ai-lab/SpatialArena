# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Exports the trajectory-alignment cases of the dashboard home page.

The cases feed `js/home_align.js` ("Trajectory alignment"). Each case is one
trial of scripts/test/compare_trajectory_alignment.py (`run_trial`: corrupt a
GT trajectory with `corrupt`, map it into a random Sim(3) frame) on a real GT
trajectory (TUM RGB-D, SpatialArena) or the synthetic near-collinear path. The
SLAM evaluator's estimator (`ransac_irls_align_sim3`) is stepped through with
the module's own building blocks, recording every RANSAC sample, the inlier
refit and every IRLS iteration, and the final estimate is asserted equal to
`ransac_irls_align_sim3`'s. Umeyama on all centers (evo's
`align(correct_scale=True)`) and the previous default (R-DAGSfM) are exported
as references, with the study's seed spread (ATE over 10 RANSAC seeds).

Six scenarios, three cases each: noise 5 %, outliers 45 %, a 45 % tracking
jump, 30° rotation noise and 10 % drift on real GT (TUM RGB-D, CC BY 4.0, and
SpatialArena, CC BY-NC 4.0, at stride 10), and 30 % outliers on the study's
near-collinear path (eps = 1e-3). Outputs, in `--output` (default
`app/webpage/assets/align/trajectory`): `manifest.json` (scenarios, notes, the
study's medians) and one `<scenario>_<n>.json` per case.

Usage (CPU, a few seconds):
`python scripts/visualization/export_home_trajectory_alignment.py`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from scipy import optimize
from scipy.spatial import transform

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
for _path in (ROOT_DIR, ROOT_DIR / "scripts/test"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
# pylint: disable=wrong-import-position
import compare_trajectory_alignment as study  # noqa: E402
from spatialarena.eval import pose_alignment as pa  # noqa: E402
from spatialarena.eval import slam_trajectory_evaluator  # noqa: E402

# pylint: enable=wrong-import-position

STRIDE = 10
RANSAC_ITERATIONS = slam_trajectory_evaluator.RANSAC_ITERATIONS
SEED = 42

# Key, label, experiment scenario, level, sources, note.
SCENARIOS = [
    ("noise", "Noise", "noise", 0.05, "real",
     "Gaussian noise on every camera center (σ = 5 % of the extent), 1° on "
     "every orientation. Nothing is wrong, so every estimator should agree."),
    ("outliers", "Outliers", "outliers", 0.45, "real",
     "45 % of the frames replaced by random poses within twice the "
     "extent. A least-squares fit is dragged towards them and hides the "
     "error."),
    ("jump", "Tracking jump", "jump", 0.45, "real",
     "Tracking lost: the last 45 % of the frames are offset by a rigid "
     "20° / 0.3-extent jump. Fitting all frames splits the difference."),
    ("rot_noise", "Rotation noise", "rot_noise", 30.0, "real",
     "Accurate centers, orientations off by up to 30°. "
     "Fits that trust single-frame rotations (the previous default) inflate "
     "the ATE."),
    ("drift", "Drift", "drift", 0.10, "real",
     "Odometry drift: every relative motion is perturbed and integrated. No "
     "subset of frames is right, so the answer must not depend on the RANSAC "
     "seed."),
    ("collinear", "Near-collinear", "outliers", 0.30, "collinear",
     "A 100 m straight path (lateral deviation 0.1 % of "
     "its length) with 30 % outliers: roll about the path is invisible to "
     "camera centers."),
]
REAL_SOURCES = [
    ("tum_dynamics", "rgbd_dataset_freiburg3_walking_halfsphere",
     "TUM RGB-D fr3/walking_halfsphere"),
    ("tum_dynamics", "rgbd_dataset_freiburg3_sitting_xyz",
     "TUM RGB-D fr3/sitting_xyz"),
    ("spatialarena", "ArtHouse", "SpatialArena ArtHouse"),
    ("tum_dynamics", "rgbd_dataset_freiburg2_desk_with_person",
     "TUM RGB-D fr2/desk_with_person"),
    ("spatialarena", "StPancrasOutside", "SpatialArena StPancrasOutside"),
]
VARIANTS = 3
# Experiment medians shown beside each case:
# (source, {estimator: (ATE ratio, rotation error deg)}).
PAPER = {
    "noise": ("real TUM RGB-D & 7-Scenes GT, 540 trials", {
        "umeyama": (0.97, None),
        "dagsfm": (0.98, None),
        "ours": (0.98, None)
    }),
    "outliers": ("real TUM RGB-D & 7-Scenes GT, 540 trials", {
        "umeyama": (0.38, None),
        "dagsfm": (1.00, None),
        "ours": (1.00, None)
    }),
    "jump": ("real TUM RGB-D & 7-Scenes GT, 540 trials", {
        "umeyama": (0.51, None),
        "dagsfm": (1.00, None),
        "ours": (1.00, None)
    }),
    "rot_noise": ("real TUM RGB-D & 7-Scenes GT, 540 trials", {
        "umeyama": (0.99, None),
        "dagsfm": (2.89, None),
        "ours": (0.99, None)
    }),
    "drift": ("synthetic, 120 trials", {
        "umeyama": (0.34, None),
        "dagsfm": (0.45, None),
        "ours": (0.35, None)
    }),
    "collinear": ("synthetic ε = 1e-3, 30 trials", {
        "umeyama": (None, 86.3),
        "dagsfm": (None, 0.0),
        "ours": (None, 0.0)
    }),
}


def load_real(benchmark_root: pathlib.Path, dataset: str,
              scene: str) -> np.ndarray:
    """Loads one SLAM sequence's GT at `STRIDE`, via the study's loader.

    Args:
        benchmark_root: The benchmark root.
        dataset: The SLAM dataset.
        scene: The sequence.

    Returns:
        The [N, 4, 4] camera-to-world poses.

    Raises:
        KeyError: If the sequence is not found.
    """
    for name, gt in study.load_real_gt(benchmark_root, [dataset], STRIDE):
        if name == f"{dataset}/{scene}":
            return gt
    raise KeyError(f"{dataset}/{scene}")


def make_case(gt: np.ndarray, scenario: str, level: float,
              rng: np.random.Generator):
    """Generates one case exactly as `run_trial` does.

    Args:
        gt: The [N, 4, 4] camera-to-world ground truth.
        scenario: The study's corruption scenario.
        level: The corruption strength.
        rng: The random generator.

    Returns:
        The prediction (the corrupted copy in a random frame), the mask of
        uncorrupted frames and the true (scale, rotation, translation).
    """
    corrupted, clean = study.corrupt(gt, scenario, level, rng)
    true_scale = float(np.exp(rng.uniform(np.log(0.05), np.log(20.0))))
    true_rotation = transform.Rotation.random(
        random_state=int(rng.integers(1 << 31))).as_matrix()
    true_translation = rng.normal(size=3) * 10.0
    pred = corrupted.copy()
    pred[:, :3, :3] = true_rotation.T @ corrupted[:, :3, :3]
    pred[:, :3, 3] = (
        (corrupted[:, :3, 3] - true_translation) @ true_rotation) / true_scale
    return pred, clean, (true_scale, true_rotation, true_translation)


def sim_dict(scale, rotation, translation) -> dict:
    """Returns a Sim(3) as JSON: scale `s`, quaternion `q`, translation `t`."""
    return {
        "s":
            float(scale),
        "q":
            transform.Rotation.from_matrix(rotation).as_quat().round(9
                                                                    ).tolist(),
        "t":
            np.asarray(translation, float).round(9).tolist()
    }


def stats(gt, pred, clean, truth, sim) -> dict:
    """Scores an alignment against the true one.

    Args:
        gt: The [N, 4, 4] camera-to-world ground truth.
        pred: The prediction.
        clean: The mask of uncorrupted frames.
        truth: The true (scale, rotation, translation).
        sim: The estimated (scale, rotation, translation).

    Returns:
        The ATE, its ratio to the true alignment's ATE, the rotation and
        scale errors and, if some frames are corrupted, the clean-frame ATE.
    """
    ate_true = study.ate(gt, pred, truth)
    out = {
        "ate": study.ate(gt, pred, sim),
        "ateRatio": study.ate(gt, pred, sim) / max(ate_true, 1e-12),
        "rotErr": study.rotation_error_deg(sim[1], truth[1]),
        "scaleErr": float(abs(np.log(sim[0] / truth[0])))
    }
    if not clean.all():
        out["ateClean"] = study.ate(gt, pred, sim, clean)
    return {key: round(value, 6) for key, value in out.items()}


def _ransac_samples(pred_r, pred_c, gt_r, gt_c):
    """Runs the RANSAC stage of `ransac_irls_align_sim3`, recording samples.

    Returns:
        The samples (pair, candidate, LMedS score, best so far) and the best
        LMedS score and candidate.
    """
    n = len(pred_c)
    fit = pa.fit_sim3_pose_residuals
    rng = np.random.default_rng(SEED)
    best_med, best, samples = np.inf, None, []
    for _ in range(RANSAC_ITERATIONS):
        sample = rng.choice(n, size=2, replace=False)
        try:
            candidate = fit(pred_r, pred_c, gt_r, gt_c, sample)
        except (ValueError, np.linalg.LinAlgError):
            continue
        errors = pa._pose_alignment_errors(pred_c, gt_c, *candidate)
        med = float(np.median(errors**2))
        if med < best_med:
            best_med, best = med, candidate
        samples.append({
            "pair": [int(index) for index in sample],
            "sim": candidate,
            "lmeds": med,
            "best": best
        })
    return samples, best_med, best


def _irls_steps(refit, pred_r, pred_c, gt_r, gt_c) -> list[dict]:
    """Runs `irls_refine_sim3` from `refit`, recording every iteration.

    Returns:
        Per iteration the Sim(3), the Cauchy weights and the robust scales.
    """
    x = np.concatenate([
        transform.Rotation.from_matrix(refit[1]).as_rotvec(), refit[2],
        [np.log(max(refit[0], 1e-12))]
    ])
    irls = []
    for _ in range(20):
        residual = pa._pose_residuals(x, pred_r, pred_c, gt_r, gt_c)
        rot_norm = np.linalg.norm(residual[:, :3], axis=1)
        trans_norm = np.linalg.norm(residual[:, 3:], axis=1)
        sigma_r = pa._robust_sigma(rot_norm)
        sigma_t = pa._robust_sigma(trans_norm)
        error = np.hypot(rot_norm / sigma_r, trans_norm / sigma_t)
        weights = 1.0 / (1.0 + (error / pa.CAUCHY_C)**2)
        unit = np.array([sigma_r] * 3 + [sigma_t] * 3)
        new_x = optimize.least_squares(pa._weighted_pose_residuals,
                                       x,
                                       method="lm",
                                       args=(None, np.sqrt(weights)[:,
                                                                    None], unit,
                                             pred_r, pred_c, gt_r, gt_c)).x
        step = float(np.linalg.norm(new_x - x))
        x = new_x
        irls.append({
            "sim": (float(np.exp(x[6])),
                    transform.Rotation.from_rotvec(x[:3]).as_matrix(),
                    x[3:6].copy()),
            "weights": weights,
            "sigmaR": sigma_r,
            "sigmaT": sigma_t
        })
        if step < 1e-9:
            break
    return irls


def step_ours(gt: np.ndarray, pred: np.ndarray):
    """Runs `ransac_irls_align_sim3` with every intermediate state recorded.

    It makes the same calls with the same random stream, and checks that the
    final estimate equals `ransac_irls_align_sim3`'s.

    Args:
        gt: The [N, 4, 4] camera-to-world ground truth.
        pred: The prediction.

    Returns:
        (samples, best, threshold, inliers, refit, irls, final, collapsed):
        the RANSAC samples, the best sample's fit, the inlier threshold and
        mask, the inlier refit, the IRLS iterations, the final Sim(3) and
        whether the scale collapsed.
    """
    pred_r, pred_c, gt_r, gt_c = study._pose_args(gt, pred)
    n = len(pred_c)
    samples, best_med, best = _ransac_samples(pred_r, pred_c, gt_r, gt_c)
    robust_sigma = 1.4826 * (1.0 + 5.0 / max(n - 2, 1)) * np.sqrt(best_med)
    threshold = max(2.5 * robust_sigma, 1e-9)
    errors = pa._pose_alignment_errors(pred_c, gt_c, *best)
    inliers = errors <= threshold
    if inliers.sum() >= 2:
        refit = pa.fit_sim3_pose_residuals(pred_r, pred_c, gt_r, gt_c,
                                           np.flatnonzero(inliers))
    else:
        refit = best
    if inliers.sum() < 2:
        inliers = np.ones(n, bool)
    irls = _irls_steps(refit, pred_r, pred_c, gt_r, gt_c)
    final = irls[-1]["sim"]
    reference = pa.ransac_irls_align_sim3(pred_r,
                                          pred_c,
                                          gt_r,
                                          gt_c,
                                          num_iterations=RANSAC_ITERATIONS,
                                          random_state=SEED)
    collapsed = bool(reference.scale_collapsed)
    if collapsed:
        final = (reference.scale, reference.rotation, reference.translation)
    else:
        assert abs(np.log(reference.scale / final[0])) < 1e-9, (reference.scale,
                                                                final[0])
        assert np.abs(reference.rotation - final[1]).max() < 1e-8
    return samples, best, threshold, inliers, refit, irls, final, collapsed


def quat_list(rotations: np.ndarray) -> list:
    """Returns rotation matrices as quaternions rounded to 6 digits."""
    quaternions = transform.Rotation.from_matrix(rotations).as_quat()
    return [
        [round(value, 6) for value in quaternion] for quaternion in quaternions
    ]


def pose_block(poses: np.ndarray, digits: int) -> dict:
    """Returns poses as JSON: centers `c` and quaternions `q`."""
    return {
        "c": poses[:, :3, 3].round(digits).tolist(),
        "q": quat_list(poses[:, :3, :3])
    }


def export_case(key, scenario, level, gt, source, variant) -> dict:
    """Builds one case: the data, every step of our fit and the references.

    Args:
        key: The scenario key.
        scenario: The study's corruption scenario.
        level: The corruption strength.
        gt: The [N, 4, 4] camera-to-world ground truth.
        source: The trajectory's label.
        variant: The case number within the scenario (seeds the case).

    Returns:
        The case JSON.
    """
    rng = np.random.default_rng(1000 * variant + 17 + sum(map(ord, key)))
    pred, clean, truth = make_case(gt, scenario, level, rng)
    ext = study.extent(gt)
    gt_digits = max(3, int(np.ceil(-np.log10(ext))) + 4)
    (samples, best, threshold, inliers, refit, irls, final,
     collapsed) = step_ours(gt, pred)
    umeyama = pa.fit_sim3_umeyama(None, pred[:, :3, 3], None, gt[:, :3, 3],
                                  np.arange(len(gt)))
    dagsfm = study._ransac(gt,
                           pred,
                           fit_fn=pa.fit_sim3_dagsfm,
                           sample_size=2,
                           iterations=RANSAC_ITERATIONS,
                           seed=SEED)

    def case_stats(sim) -> dict:
        return stats(gt, pred, clean, truth, sim)

    def seed_cv(align) -> float:
        """Returns the study's seed CV: the ATE spread over 10 RANSAC seeds."""
        values = [
            study.ate(gt, pred, align(gt, pred, seed=seed))
            for seed in range(10)
        ]
        return round(float(np.std(values) / max(np.mean(values), 1e-12)), 8)

    seed_spread = {
        "dagsfm": seed_cv(study.align_ransac_dagsfm),
        "ours": seed_cv(study.align_ransac_gtsam_irls)
    }
    pred_scale = study.extent(pred)
    pred_digits = max(3, int(np.ceil(-np.log10(pred_scale))) + 4)
    return {
        "scenario": key,
        "source": source,
        "variant": variant,
        "frames": len(gt),
        "extent": round(ext, 4),
        "gt": pose_block(gt, gt_digits),
        "pred": pose_block(pred, pred_digits),
        "clean": [int(value) for value in clean],
        "truth": sim_dict(*truth),
        "truthStats": {
            "ate": round(study.ate(gt, pred, truth), 6)
        },
        "samples": [{
            "pair": sample["pair"],
            "sim": sim_dict(*sample["sim"]),
            "lmeds": sample["lmeds"],
            "stats": case_stats(sample["sim"]),
            "bestStats": case_stats(sample["best"])
        } for sample in samples],
        "best": {
            "sim": sim_dict(*best),
            "stats": case_stats(best),
            "threshold": threshold,
            "inliers": [int(value) for value in inliers]
        },
        "refit": {
            "sim": sim_dict(*refit),
            "stats": case_stats(refit)
        },
        "irls": [{
            "sim": sim_dict(*iteration["sim"]),
            "stats": case_stats(iteration["sim"]),
            "weights": iteration["weights"].round(3).tolist()
        } for iteration in irls],
        "final": {
            "sim": sim_dict(*final),
            "stats": case_stats(final),
            "scaleCollapsed": collapsed
        },
        "umeyama": {
            "sim": sim_dict(*umeyama),
            "stats": case_stats(umeyama)
        },
        "dagsfm": {
            "sim": sim_dict(*dagsfm),
            "stats": case_stats(dagsfm)
        },
        "seedCV": seed_spread,
    }


def _case_line(key: str, variant: int, source: str, case: dict) -> str:
    """Returns the summary line printed for one case."""
    final = case["final"]["stats"]
    umeyama = case["umeyama"]["stats"]
    dagsfm = case["dagsfm"]["stats"]
    if case["final"]["scaleCollapsed"]:
        collapsed = "  COLLAPSED"
    else:
        collapsed = ""
    return (f"{key:10s} v{variant} {source:36s} n={case['frames']:3d}  "
            f"ATE ratio / rot err: "
            f"umeyama {umeyama['ateRatio']:.2f}/{umeyama['rotErr']:6.2f}°  "
            f"dagsfm {dagsfm['ateRatio']:.2f}/{dagsfm['rotErr']:6.2f}°  "
            f"ours {final['ateRatio']:.2f}/{final['rotErr']:6.2f}°  "
            f"irls {len(case['irls'])}"
            f"  seedCV {case['seedCV']['dagsfm']:.3f}/"
            f"{case['seedCV']['ours']:.1e}"
            f"{collapsed}")


def main() -> None:
    """Exports every case and the manifest."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark_root",
                        type=pathlib.Path,
                        default=pathlib.Path.home() / "datasets/benchmark")
    parser.add_argument("--output",
                        type=pathlib.Path,
                        default=ROOT_DIR /
                        "app/webpage/assets/align/trajectory")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    real = {
        label: load_real(args.benchmark_root, dataset, scene)
        for dataset, scene, label in REAL_SOURCES
    }
    manifest = {
        "stride": STRIDE,
        "ransacIterations": RANSAC_ITERATIONS,
        "seed": SEED,
        "cauchyC": pa.CAUCHY_C,
        "credits": {
            "TUM RGB-D": "Sturm et al., IROS 2012, CC BY 4.0",
            "SpatialArena": "SpatialArena captures, CC BY-NC 4.0"
        },
        "scenarios": []
    }
    for index, (key, label, scenario, level, sources,
                note) in enumerate(SCENARIOS):
        cases = []
        for variant in range(VARIANTS):
            if sources == "collinear":
                gt = study.make_collinear(200, 1e-3,
                                          np.random.default_rng(500 + variant))
                source = "synthetic 100 m straight path (ε = 1e-3)"
            else:
                source = list(real)[(variant + index) % len(real)]
                gt = real[source]
            case = export_case(key, scenario, level, gt, source, variant)
            name = f"{key}_{variant}.json"
            (out / name).write_text(
                json.dumps(case, separators=(",", ":"), ensure_ascii=False))
            cases.append(name)
            print(_case_line(key, variant, source, case))
        paper_source, paper = PAPER[key]
        paper_entry = {"source": paper_source}
        paper_entry.update(paper)
        manifest["scenarios"].append({
            "key": key,
            "label": label,
            "scenario": scenario,
            "level": level,
            "note": note,
            "cases": cases,
            "paper": paper_entry
        })
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
