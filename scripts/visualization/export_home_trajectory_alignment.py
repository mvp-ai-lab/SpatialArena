# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Export the trajectory-alignment cases of the dashboard home page (``js/home_align.js``, "Trajectory alignment").

Each case is one trial of scripts/test/compare_trajectory_alignment.py (``run_trial``: corrupt a GT trajectory with
``corrupt``, map it into a random Sim(3) frame) on a real GT trajectory (TUM RGB-D, SpatialArena) or the synthetic
near-collinear path. The SLAM evaluator's estimator (``ransac_irls_align_sim3``) is stepped through with the module's
own building blocks, recording every RANSAC sample, the inlier refit and every IRLS iteration, and the final estimate is
asserted equal to ``ransac_irls_align_sim3``'s. Umeyama on all centers (evo's ``align(correct_scale=True)``) and the
previous default (R-DAGSfM) are exported as references, with the study's seed spread (ATE over 10 RANSAC seeds).

Six scenarios, three cases each: noise 5 %, outliers 45 %, a 45 % tracking jump, 30° rotation noise and 10 % drift on
real GT (TUM RGB-D, CC BY 4.0, and SpatialArena, CC BY-NC 4.0, at stride 10), and 30 % outliers on the study's
near-collinear path (eps = 1e-3). Outputs, in ``--output`` (default ``app/webpage/assets/align/
trajectory``): ``manifest.json`` (scenarios, notes, the study's medians) and one ``<scenario>_<n>.json`` per case.

Usage: ``python scripts/visualization/export_home_trajectory_alignment.py`` (CPU, a few seconds).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT_DIR = Path(__file__).resolve().parents[2]
for _path in (ROOT_DIR, ROOT_DIR / "scripts/test"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
# pylint: disable=wrong-import-position
import compare_trajectory_alignment as study  # noqa: E402
from spatialarena.eval import pose_alignment as pa  # noqa: E402

# pylint: enable=wrong-import-position

STRIDE = 10
RANSAC_ITERATIONS = study.RANSAC_ITERATIONS
SEED = 42

SCENARIOS = [  # key, label, experiment scenario, level, sources, note
    ("noise", "Noise", "noise", 0.05, "real", "Gaussian noise on every camera center (σ = 5 % of the extent), 1° on "
     "every orientation. Nothing is wrong, so every estimator should agree."),
    ("outliers", "Outliers", "outliers", 0.45, "real", "45 % of the frames replaced by random poses within twice the "
     "extent. A least-squares fit is dragged towards them and hides the error."),
    ("jump", "Tracking jump", "jump", 0.45, "real", "Tracking lost: the last 45 % of the frames are offset by a rigid "
     "20° / 0.3-extent jump. Fitting all frames splits the difference."),
    ("rot_noise", "Rotation noise", "rot_noise", 30.0, "real", "Accurate centers, orientations off by up to 30°. "
     "Fits that trust single-frame rotations (the previous default) inflate the ATE."),
    ("drift", "Drift", "drift", 0.10, "real", "Odometry drift: every relative motion is perturbed and integrated. No "
     "subset of frames is right, so the answer must not depend on the RANSAC seed."),
    ("collinear", "Near-collinear", "outliers", 0.30, "collinear", "A 100 m straight path (lateral deviation 0.1 % of "
     "its length) with 30 % outliers: roll about the path is invisible to camera centers."),
]
REAL_SOURCES = [
    ("tum_dynamics", "rgbd_dataset_freiburg3_walking_halfsphere", "TUM RGB-D fr3/walking_halfsphere"),
    ("tum_dynamics", "rgbd_dataset_freiburg3_sitting_xyz", "TUM RGB-D fr3/sitting_xyz"),
    ("spatialarena", "ArtHouse", "SpatialArena ArtHouse"),
    ("tum_dynamics", "rgbd_dataset_freiburg2_desk_with_person", "TUM RGB-D fr2/desk_with_person"),
    ("spatialarena", "StPancrasOutside", "SpatialArena StPancrasOutside"),
]
VARIANTS = 3
# experiment medians shown beside each case: (source, {estimator: (ATE ratio, rotation error deg)})
PAPER = {
    "noise": ("real TUM RGB-D & 7-Scenes GT, 540 trials", {"umeyama": (0.97, None), "dagsfm": (0.98, None),
                                                          "ours": (0.98, None)}),
    "outliers": ("real TUM RGB-D & 7-Scenes GT, 540 trials", {"umeyama": (0.38, None), "dagsfm": (1.00, None),
                                                             "ours": (1.00, None)}),
    "jump": ("real TUM RGB-D & 7-Scenes GT, 540 trials", {"umeyama": (0.51, None), "dagsfm": (1.00, None),
                                                         "ours": (1.00, None)}),
    "rot_noise": ("real TUM RGB-D & 7-Scenes GT, 540 trials", {"umeyama": (0.99, None), "dagsfm": (2.89, None),
                                                              "ours": (0.99, None)}),
    "drift": ("synthetic, 120 trials", {"umeyama": (0.34, None), "dagsfm": (0.45, None), "ours": (0.35, None)}),
    "collinear": ("synthetic ε = 1e-3, 30 trials", {"umeyama": (None, 86.3), "dagsfm": (None, 0.0),
                                                     "ours": (None, 0.0)}),
}


def load_real(benchmark_root: Path, dataset: str, scene: str) -> np.ndarray:
    """GT camera-to-world of one SLAM sequence at ``STRIDE``, via the study's loader."""
    for name, gt in study.load_real_gt(benchmark_root, [dataset], STRIDE):
        if name == f"{dataset}/{scene}":
            return gt
    raise KeyError(f"{dataset}/{scene}")


def make_case(gt: np.ndarray, scenario: str, level: float, rng: np.random.Generator):
    """``run_trial``'s data generation, verbatim: corrupted copy, random frame, prediction."""
    corrupted, clean = study.corrupt(gt, scenario, level, rng)
    true_scale = float(np.exp(rng.uniform(np.log(0.05), np.log(20.0))))
    true_rotation = Rotation.random(random_state=int(rng.integers(1 << 31))).as_matrix()
    true_translation = rng.normal(size=3) * 10.0
    pred = corrupted.copy()
    pred[:, :3, :3] = true_rotation.T @ corrupted[:, :3, :3]
    pred[:, :3, 3] = ((corrupted[:, :3, 3] - true_translation) @ true_rotation) / true_scale
    return pred, clean, (true_scale, true_rotation, true_translation)


def sim_dict(scale, rotation, translation) -> dict:
    return {"s": float(scale), "q": Rotation.from_matrix(rotation).as_quat().round(9).tolist(),
            "t": np.asarray(translation, float).round(9).tolist()}


def stats(gt, pred, clean, truth, sim) -> dict:
    ate_true = study.ate(gt, pred, truth)
    out = {"ate": study.ate(gt, pred, sim), "ateRatio": study.ate(gt, pred, sim) / max(ate_true, 1e-12),
           "rotErr": study.rotation_error_deg(sim[1], truth[1]),
           "scaleErr": float(abs(np.log(sim[0] / truth[0])))}
    if not clean.all():
        out["ateClean"] = study.ate(gt, pred, sim, clean)
    return {k: round(v, 6) for k, v in out.items()}


def step_ours(gt: np.ndarray, pred: np.ndarray):
    """``ransac_irls_align_sim3`` with every intermediate state recorded (same calls, same random stream)."""
    pred_r, pred_c, gt_r, gt_c = study._pose_args(gt, pred)  # pylint: disable=protected-access
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
        errors = pa._pose_alignment_errors(pred_c, gt_c, *candidate)  # pylint: disable=protected-access
        med = float(np.median(errors ** 2))
        if med < best_med:
            best_med, best = med, candidate
        samples.append({"pair": [int(v) for v in sample], "sim": candidate, "lmeds": med, "best": best})
    robust_sigma = 1.4826 * (1.0 + 5.0 / max(n - 2, 1)) * np.sqrt(best_med)
    threshold = max(2.5 * robust_sigma, 1e-9)
    errors = pa._pose_alignment_errors(pred_c, gt_c, *best)  # pylint: disable=protected-access
    inliers = errors <= threshold
    refit = fit(pred_r, pred_c, gt_r, gt_c, np.flatnonzero(inliers)) if inliers.sum() >= 2 else best
    if inliers.sum() < 2:
        inliers = np.ones(n, bool)
    # irls_refine_sim3, iteration by iteration
    x = np.concatenate([Rotation.from_matrix(refit[1]).as_rotvec(), refit[2], [np.log(max(refit[0], 1e-12))]])
    irls = []
    for _ in range(20):
        residual = pa._pose_residuals(x, pred_r, pred_c, gt_r, gt_c)  # pylint: disable=protected-access
        rot_norm = np.linalg.norm(residual[:, :3], axis=1)
        trans_norm = np.linalg.norm(residual[:, 3:], axis=1)
        sigma_r, sigma_t = pa._robust_sigma(rot_norm), pa._robust_sigma(trans_norm)  # pylint: disable=protected-access
        error = np.hypot(rot_norm / sigma_r, trans_norm / sigma_t)
        weights = 1.0 / (1.0 + (error / pa.CAUCHY_C) ** 2)
        unit = np.array([sigma_r] * 3 + [sigma_t] * 3)
        new_x = least_squares(pa._weighted_pose_residuals, x, method="lm",  # pylint: disable=protected-access
                              args=(None, np.sqrt(weights)[:, None], unit, pred_r, pred_c, gt_r, gt_c)).x
        step = float(np.linalg.norm(new_x - x))
        x = new_x
        irls.append({"sim": (float(np.exp(x[6])), Rotation.from_rotvec(x[:3]).as_matrix(), x[3:6].copy()),
                     "weights": weights, "sigmaR": sigma_r, "sigmaT": sigma_t})
        if step < 1e-9:
            break
    final = irls[-1]["sim"]
    reference = pa.ransac_irls_align_sim3(pred_r, pred_c, gt_r, gt_c, num_iterations=RANSAC_ITERATIONS,
                                          random_state=SEED)
    collapsed = bool(reference.scale_collapsed)
    if collapsed:
        final = (reference.scale, reference.rotation, reference.translation)
    else:
        assert abs(np.log(reference.scale / final[0])) < 1e-9, (reference.scale, final[0])
        assert np.abs(reference.rotation - final[1]).max() < 1e-8
    return samples, best, threshold, inliers, refit, irls, final, collapsed


def quat_list(rotations: np.ndarray) -> list:
    return [[round(v, 6) for v in q] for q in Rotation.from_matrix(rotations).as_quat()]


def pose_block(poses: np.ndarray, digits: int) -> dict:
    return {"c": poses[:, :3, 3].round(digits).tolist(), "q": quat_list(poses[:, :3, :3])}


def export_case(key, scenario, level, gt, source, variant) -> dict:
    rng = np.random.default_rng(1000 * variant + 17 + sum(map(ord, key)))
    pred, clean, truth = make_case(gt, scenario, level, rng)
    ext = study.extent(gt)
    gt_digits = max(3, int(np.ceil(-np.log10(ext))) + 4)
    samples, best, threshold, inliers, refit, irls, final, collapsed = step_ours(gt, pred)
    umeyama = pa.fit_sim3_umeyama(None, pred[:, :3, 3], None, gt[:, :3, 3], np.arange(len(gt)))
    dagsfm = study._ransac(gt, pred, fit_fn=pa.fit_sim3_dagsfm, sample_size=2,  # pylint: disable=protected-access
                           iterations=RANSAC_ITERATIONS, seed=SEED)

    def st(sim) -> dict:
        return stats(gt, pred, clean, truth, sim)

    def seed_cv(fn) -> float:  # the study's seed CV: ATE spread over 10 RANSAC seeds
        values = [study.ate(gt, pred, fn(gt, pred, seed=s)) for s in range(10)]
        return round(float(np.std(values) / max(np.mean(values), 1e-12)), 8)

    seed_spread = {"dagsfm": seed_cv(study.align_ransac_dagsfm), "ours": seed_cv(study.align_ransac_gtsam_irls)}
    pred_scale = study.extent(pred)
    return {
        "scenario": key, "source": source, "variant": variant, "frames": len(gt), "extent": round(ext, 4),
        "gt": pose_block(gt, gt_digits),
        "pred": pose_block(pred, max(3, int(np.ceil(-np.log10(pred_scale))) + 4)),
        "clean": [int(v) for v in clean],
        "truth": sim_dict(*truth), "truthStats": {"ate": round(study.ate(gt, pred, truth), 6)},
        "samples": [{"pair": s["pair"], "sim": sim_dict(*s["sim"]), "lmeds": s["lmeds"],
                     "stats": st(s["sim"]), "bestStats": st(s["best"])} for s in samples],
        "best": {"sim": sim_dict(*best), "stats": st(best), "threshold": threshold,
                 "inliers": [int(v) for v in inliers]},
        "refit": {"sim": sim_dict(*refit), "stats": st(refit)},
        "irls": [{"sim": sim_dict(*s["sim"]), "stats": st(s["sim"]),
                  "weights": s["weights"].round(3).tolist()} for s in irls],
        "final": {"sim": sim_dict(*final), "stats": st(final), "scaleCollapsed": collapsed},
        "umeyama": {"sim": sim_dict(*umeyama), "stats": st(umeyama)},
        "dagsfm": {"sim": sim_dict(*dagsfm), "stats": st(dagsfm)},
        "seedCV": seed_spread,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark_root", type=Path, default=Path.home() / "datasets/benchmark")
    parser.add_argument("--output", type=Path,
                        default=ROOT_DIR / "app/webpage/assets/align/trajectory")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    real = {label: load_real(args.benchmark_root, ds, scene) for ds, scene, label in REAL_SOURCES}
    manifest = {"stride": STRIDE, "ransacIterations": RANSAC_ITERATIONS, "seed": SEED, "cauchyC": pa.CAUCHY_C,
                "credits": {"TUM RGB-D": "Sturm et al., IROS 2012, CC BY 4.0",
                            "SpatialArena": "SpatialArena captures, CC BY-NC 4.0"},
                "scenarios": []}
    for index, (key, label, scenario, level, sources, note) in enumerate(SCENARIOS):
        cases = []
        for variant in range(VARIANTS):
            if sources == "collinear":
                gt = study.make_collinear(200, 1e-3, np.random.default_rng(500 + variant))
                source = "synthetic 100 m straight path (ε = 1e-3)"
            else:
                source = list(real)[(variant + index) % len(real)]
                gt = real[source]
            case = export_case(key, scenario, level, gt, source, variant)
            name = f"{key}_{variant}.json"
            (out / name).write_text(json.dumps(case, separators=(",", ":"), ensure_ascii=False))
            cases.append(name)
            f, u, d = case["final"]["stats"], case["umeyama"]["stats"], case["dagsfm"]["stats"]
            print(f"{key:10s} v{variant} {source:36s} n={case['frames']:3d}  ATE ratio / rot err: "
                  f"umeyama {u['ateRatio']:.2f}/{u['rotErr']:6.2f}°  dagsfm {d['ateRatio']:.2f}/{d['rotErr']:6.2f}°  "
                  f"ours {f['ateRatio']:.2f}/{f['rotErr']:6.2f}°  irls {len(case['irls'])}"
                  f"  seedCV {case['seedCV']['dagsfm']:.3f}/{case['seedCV']['ours']:.1e}"
                  f"{'  COLLAPSED' if case['final']['scaleCollapsed'] else ''}")
        paper_source, paper = PAPER[key]
        manifest["scenarios"].append({"key": key, "label": label, "scenario": scenario, "level": level,
                                      "note": note, "cases": cases, "paper": {"source": paper_source, **paper}})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
