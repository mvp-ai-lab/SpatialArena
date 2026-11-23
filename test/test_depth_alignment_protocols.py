# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""GT-only depth alignment protocols: Huber IRLS, inverse/log/depth-space fits and their edge cases.

gt_only_nonpositive: Nonpositive native depth under the GT-only protocols: floored to NONPOSITIVE_DEPTH_FLOOR, not a
scene failure.

gt_only_normals: mae_normal_deg under the GT-only protocols: computed on each frame's full aligned depth map.

log_depth_gt: log_huber_irls_gt: affine fit of a normalized log-depth code to log(GT), decoded with exp.

stored_sequence_fit: The GT-only sequence scorers' saved fit (``<prediction>/<protocol>_fit.json``) rebuilds exactly
the per-frame aligned maps the scorer hands its writers -- what ``write_sequence_viewer_from_fit`` relies on.

diode_prediction_clip: DIODE post-alignment clipping must use only each image's valid GT.
"""

import numpy as np
from scipy.optimize import minimize
from spatialarena.eval.canonical_dataset import BenchmarkSceneInfo
from spatialarena.eval.depth_prediction_evaluator import (
    DepthEvaluationConfig,
    resolve_depth_config,
)
from spatialarena.eval.huber_irls import fit_huber_irls
import pytest
from spatialarena.eval.depth_space_gt import decode_gt_bounds
from spatialarena.eval.depth_prediction_evaluator import prepare_depth_pair
from spatialarena.eval.inverse_l1_gt import fit_adam
import json
from spatialarena.eval.depth_prediction_evaluator import DepthPairData
from spatialarena.eval.inverse_l1_gt import score_sequence
from spatialarena.eval.inverse_l1_gt import NONPOSITIVE_DEPTH_FLOOR
from pathlib import Path
import subprocess
import sys
from spatialarena.methods import resolve_method
from spatialarena.eval.depth_space_gt import aligned_depth_space_map
from spatialarena.eval.inverse_l1_gt import aligned_inverse_depth_map
from spatialarena.eval import depth_prediction_evaluator as ev

# ---- huber_irls --------------------------------------------------------------------------------------------


def test_affine_and_zero_negative_raw_codes():
    x = np.linspace(-1, 1, 101)
    y = 2 * x + 4
    a, b, d = fit_huber_irls(x, y, device="cpu")
    assert np.allclose([a, b], [2, 4], atol=1e-10)
    assert d["stop"] == "parameter_tolerance"


def test_outliers_matches_independent_optimizer():
    x = np.linspace(0.1, 2, 101)
    y = 2 * x + 1
    y[::10] += 10
    a, b, d = fit_huber_irls(x, y, device="cpu")
    unit = np.median(y)
    delta = 0.05

    def fun(ab):
        e = np.abs((ab[0] * x + ab[1] - y) / unit)
        return np.where(e <= delta, 0.5 * e**2,
                        delta * (e - 0.5 * delta)).mean()

    reference = minimize(fun, [2.0, 1.0], method="BFGS", options={"gtol": 1e-9})
    assert np.allclose([a, b], reference.x, atol=1e-5)
    assert fun([a, b]) <= fun(
        np.linalg.lstsq(np.c_[x, np.ones(len(x))], y, rcond=None)[0])
    values = [v["normalized_huber"] for v in d["trace"]]
    assert np.all(np.diff(values) <= 1e-12)


def test_common_scale_invariance():
    x = np.linspace(0.1, 2, 101)
    y = 2 * x + 1
    y[-1] += 5
    a, b, _ = fit_huber_irls(x, y, device="cpu")
    aa, bb, _ = fit_huber_irls(x * 10, y * 10, device="cpu")
    assert np.allclose([aa, bb], [a, b * 10], atol=1e-6)


def test_constant_prediction_fits_the_offset_only():
    """A flat prediction is scored as the Huber-best single value instead of raising (the frame keeps its penalty)."""
    y = np.r_[np.full(90, 2.0), np.full(10, 50.0)]  # mostly 2 m, with outliers
    a, b, d = fit_huber_irls(np.full(100, 1.1), y, device="cpu")
    assert a == 0.0 and d["constant_prediction"] and d[
        "stop"] == "parameter_tolerance"
    assert abs(
        b - 2.0
    ) < 0.5  # robust: near the bulk of the GT, not pulled to the mean (6.8)
    unit, delta = np.median(y), 0.05

    def fun(offset):
        e = np.abs((offset - y) / unit)
        return np.where(e <= delta, 0.5 * e**2,
                        delta * (e - 0.5 * delta)).mean()

    reference = minimize(lambda v: fun(v[0]), [y.mean()],
                         method="Nelder-Mead",
                         options={
                             "xatol": 1e-10,
                             "fatol": 1e-14
                         })
    assert fun(b) <= fun(reference.x[0]) + 1e-9
    # Still keys the diagnostics the depth-space caller renames.
    assert {
        "gt_inverse_median", "effective_inverse_delta",
        "final_mean_absolute_inverse_residual"
    } <= set(d)


# ---- depth_space_gt ----------------------------------------------------------------------------------------


def test_gt_bounds_round_trip_and_clamp():
    gt = np.linspace(1.0, 10.0, 1001)
    lo, hi = np.quantile(1 / gt, [0.02, 0.98])
    code = (1 / gt - lo) / (hi - lo + 1e-5)
    depth, bounds = decode_gt_bounds(code, gt)
    valid = (code >= 0) & (code <= 1)
    assert np.allclose(depth[valid], gt[valid])
    assert np.allclose(bounds, [lo, hi])
    endpoints, _ = decode_gt_bounds(np.array([-2.0, 0.0, 1.0, 3.0]), gt)
    assert endpoints[0] == endpoints[1] and endpoints[2] == endpoints[3]


@pytest.mark.parametrize("protocol", [
    "inverse_huber_irls_gt", "depth_huber_irls_gt", "gt_bounds_direct",
    "gt_bounds_depth_huber_irls"
])
def test_protocol(tmp_path, protocol):
    scene = BenchmarkSceneInfo(dataset="tum_dynamics",
                               task="VideoDepth",
                               scene="x",
                               root_path=tmp_path)
    c = resolve_depth_config(scene, DepthEvaluationConfig(protocol=protocol))
    assert c.protocol == protocol and c.resize_interpolation == "linear"
    assert c.gt_depth_scale == 1 / 5000 and c.max_depth == 10


# ---- inverse_l1_gt -----------------------------------------------------------------------------------------


def _l1_scene(tmp_path, dataset):
    return BenchmarkSceneInfo(dataset=dataset,
                              task="VideoDepth",
                              scene="sample",
                              root_path=tmp_path)


def test_units(tmp_path):
    for dataset, scale in [("tum_dynamics", 1 / 5000), ("dycheck", 1.0)]:
        c = resolve_depth_config(
            _l1_scene(tmp_path, dataset),
            DepthEvaluationConfig(protocol="inverse_l1_gt"))
        assert c.gt_depth_scale == scale and c.max_depth == 10 and c.protocol == "inverse_l1_gt"
        assert c.resize_interpolation == "linear"


def test_gt_only_dvd(tmp_path):
    (tmp_path / "depths").mkdir()
    (tmp_path / "raw_disparities").mkdir()
    np.save(tmp_path / "gt.npy", np.array([[1.0, 2.0, 0.0]]))
    np.save(tmp_path / "depths/a.npy", np.ones((1, 3)))
    np.save(tmp_path / "raw_disparities/a.npy", np.array([[0.0, -1.0, 5.0]]))
    c = DepthEvaluationConfig(protocol="inverse_l1_gt", max_depth=10)
    pair = prepare_depth_pair(
        scene_info=_l1_scene(tmp_path, "dycheck"),
        gt_path=tmp_path / "gt.npy",
        pred_path=tmp_path / "depths/a.npy",
        config=c,
    )
    assert pair.valid_mask.tolist() == [[True, True, False]]
    assert pair.prediction.tolist() == [[0.0, -1.0, 5.0]]
    np.save(tmp_path / "raw_disparities/a.npy", np.array([[np.nan, -1.0, 5.0]]))
    with pytest.raises(ValueError, match="Nonfinite"):
        prepare_depth_pair(
            scene_info=_l1_scene(tmp_path, "dycheck"),
            gt_path=tmp_path / "gt.npy",
            pred_path=tmp_path / "depths/a.npy",
            config=c,
        )


def test_adam_exact_scale():
    a, b, diag = fit_adam(np.array([1.0, 2.0, 3.0]),
                          np.array([2.0, 4.0, 6.0]),
                          device="cpu")
    assert np.allclose([a, b], [2, 0], atol=1e-5)
    assert diag["final_mean_absolute_inverse_residual"] < 1e-6


def test_linear_preserves_positive_depth():
    from spatialarena.eval.depth_prediction_evaluator import _resize_to_match

    depth = np.array([[0.1, 0.1, 20.0, 20.0], [0.1, 0.1, 20.0, 20.0]],
                     dtype=np.float32)
    result = _resize_to_match(depth, (8, 16), "linear")
    assert result.min() >= depth.min() and result.max() <= depth.max()


def test_l2_initialization_nonzero_shift():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = 3 * x + 10
    a, b, d = fit_adam(x, y, device="cpu", initialization="l2")
    assert np.allclose([d["initial_scale"], d["initial_shift"]], [3, 10],
                       atol=1e-6)
    assert np.allclose([a, b], [3, 10], atol=1e-5)
    assert d["initialization"] == "l2" and d["lr"] == 1e-4 and d[
        "max_iters"] == 1000


def test_l2init_protocol(tmp_path):
    c = resolve_depth_config(
        _l1_scene(tmp_path, "dycheck"),
        DepthEvaluationConfig(protocol="inverse_l1_l2init_gt"))
    assert c.protocol == "inverse_l1_l2init_gt" and c.resize_interpolation == "linear"
    assert c.gt_depth_scale == 1 and c.max_depth == 10


@pytest.mark.parametrize("initialization", ["median", "l2"])
@pytest.mark.parametrize("x,y", [([], []), ([1], [2]), ([1, 2], [1]),
                                 ([1, np.nan], [1, 2])])
def test_adam_rejects_invalid_pairs(initialization, x, y):
    with pytest.raises(ValueError, match="finite paired values"):
        fit_adam(np.asarray(x),
                 np.asarray(y),
                 device="cpu",
                 initialization=initialization)


# ---- inverse_gtmax -----------------------------------------------------------------------------------------


def test_gtmax_is_scale_equivariant_and_keeps_failed_pixels(
        tmp_path, monkeypatch):
    from spatialarena.eval import inverse_l1_gt

    (tmp_path / "raw_disparities").mkdir()
    results = []
    for scale in [1.0, 100.0]:
        monkeypatch.setattr(inverse_l1_gt,
                            "fit_adam",
                            lambda *a, scale=scale, **kw: (1 / scale, 0.0, {}))
        pair = DepthPairData(
            gt_path=tmp_path / "gt.npy",
            pred_path=tmp_path / "raw_disparities/a.npy",
            ground_truth=np.array([[1.0, 2.0]]) * scale,
            prediction=np.array([[-1.0, 0.5]]),
            valid_mask=np.ones((1, 2), dtype=bool),
        )
        config = DepthEvaluationConfig(protocol="inverse_l1_gt",
                                       inverse_floor_policy="gt_max")
        result = score_sequence([pair], config, diagnostic_suffix="a")
        assert result["abs_relative_difference"] == pytest.approx(0.5)
        report = json.loads(
            (tmp_path / "inverse_l1_gt_fits/a.json").read_text())
        assert report["score_pixels"] == 2 and report[
            "nonpositive_aligned_pixels"] == 1
        assert report["inverse_floor"] == pytest.approx(1 / (2 * scale))
        results.append(result)
    assert results[1]["rmse_linear"] == pytest.approx(
        results[0]["rmse_linear"] * 100)
    with pytest.raises(ValueError, match="MonoDepth"):
        score_sequence([pair], config)


# ---- gt_only_nonpositive -----------------------------------------------------------------------------------

K = np.array([[30.0, 0.0, 16.0], [0.0, 30.0, 16.0], [0.0, 0.0, 1.0]])
XX = np.indices((32, 32), dtype=np.float64)[1]
GT = 2.0 + 0.02 * XX


def _pair(tmp_path, prediction):
    (tmp_path / "depths").mkdir(exist_ok=True)
    return DepthPairData(gt_path=tmp_path / "gt.npy",
                         pred_path=tmp_path / "depths" / "a.npy",
                         ground_truth=GT,
                         prediction=prediction,
                         valid_mask=np.ones(GT.shape, dtype=bool),
                         intrinsics=K)


@pytest.mark.parametrize("protocol",
                         ["inverse_huber_irls_gt", "depth_huber_irls_gt"])
def test_nonpositive_pixels_score_as_the_floor(tmp_path, protocol):
    prediction = 2.0 * GT
    prediction[3, 4], prediction[10, 20] = 0.0, -1.0
    floored = prediction.copy()
    floored[3, 4] = floored[10, 20] = NONPOSITIVE_DEPTH_FLOOR
    config = DepthEvaluationConfig(protocol=protocol, max_depth=100.0)

    metrics = score_sequence([_pair(tmp_path, prediction)], config)
    diagnostic = json.loads((tmp_path / f"{protocol}_fit.json").read_text())
    assert diagnostic["floored_nonpositive_pixels"] == 2
    assert diagnostic["nonpositive_depth_floor"] == NONPOSITIVE_DEPTH_FLOOR
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics == pytest.approx(
        score_sequence([_pair(tmp_path, floored)], config))


@pytest.mark.parametrize("protocol",
                         ["inverse_huber_irls_gt", "depth_huber_irls_gt"])
def test_constant_prediction_scores_as_the_best_single_value(
        tmp_path, protocol):
    """A flat map (e.g. a VLM answering 1.1 m everywhere) is scored, not a scene failure, and scores worse than a
    prediction that has the right structure."""
    flat = score_sequence([_pair(tmp_path, np.full(GT.shape, 1.1))],
                          DepthEvaluationConfig(protocol=protocol,
                                                max_depth=100.0))
    fit = json.loads((tmp_path / f"{protocol}_fit.json").read_text())
    fit = fit.get(
        "optimizer",
        fit)  # depth_huber_irls_gt nests the solver record under "optimizer"
    assert fit["constant_prediction"]
    shaped = score_sequence([_pair(tmp_path, 3.0 * GT)],
                            DepthEvaluationConfig(protocol=protocol,
                                                  max_depth=100.0))
    assert all(np.isfinite(value) for value in flat.values())
    absrel = next(key for key in flat
                  if key in {"abs_relative_difference", "abs_rel", "absrel"})
    assert flat[absrel] > shaped[absrel]


class _RecordingWriter:
    """Stands in for the error-map and viewer writers: keeps what the evaluator hands each frame."""

    def __init__(self):
        self.frames = []

    def write(self, pair, aligned, scale, shift, metrics):
        self.frames.append(
            (pair, np.array(aligned), scale, shift, dict(metrics)))


@pytest.mark.parametrize("protocol",
                         ["inverse_huber_irls_gt", "depth_huber_irls_gt"])
def test_gt_only_protocols_feed_per_frame_writers(tmp_path, protocol):
    """Error maps and dashboard viewer images work under the GT-only protocols: each frame's writer gets the aligned
    map the score was computed from, and the scores do not change."""
    config = DepthEvaluationConfig(protocol=protocol, max_depth=100.0)
    writer = _RecordingWriter()
    metrics = score_sequence([_pair(tmp_path, 3.0 * GT)],
                             config,
                             artifacts=writer)
    assert metrics == pytest.approx(
        score_sequence([_pair(tmp_path, 3.0 * GT)], config))
    assert len(writer.frames) == 1
    _pair_data, aligned, _scale, _shift, frame_metrics = writer.frames[0]
    assert aligned.shape == GT.shape
    np.testing.assert_allclose(
        aligned, GT, rtol=1e-4)  # an exact affine prediction aligns onto the GT
    assert frame_metrics == metrics

    sequence = [_pair(tmp_path, 3.0 * GT), _pair(tmp_path, 3.0 * GT + 0.3)]
    writer = _RecordingWriter()
    score_sequence(sequence, config, artifacts=writer)
    assert len(writer.frames) == 2 and all(
        np.isfinite(frame[4]["abs_relative_difference"])
        for frame in writer.frames)


def test_evaluator_passes_writers_to_gt_only_protocols(tmp_path):
    """evaluate_single_depth_pair used to refuse any writer under a GT-only protocol."""
    from spatialarena.eval.depth_prediction_evaluator import evaluate_single_depth_pair

    writer = _RecordingWriter()
    evaluate_single_depth_pair(
        _pair(tmp_path, 3.0 * GT),
        DepthEvaluationConfig(protocol="depth_huber_irls_gt", max_depth=100.0),
        writer)
    assert len(writer.frames) == 1


# ---- gt_only_normals ---------------------------------------------------------------------------------------

NORMALS_K = np.array([[30.0, 0.0, 16.0], [0.0, 30.0, 16.0], [0.0, 0.0, 1.0]])
YY, XX = np.indices((32, 32), dtype=np.float64)
NORMALS_GT = 2.0 + 0.02 * XX  # a plane tilted along x


def _normals_pair(tmp_path, prediction, intrinsics=NORMALS_K):
    (tmp_path / "depths").mkdir(exist_ok=True)
    return DepthPairData(gt_path=tmp_path / "gt.npy",
                         pred_path=tmp_path / "depths" / "a.npy",
                         ground_truth=NORMALS_GT,
                         prediction=prediction,
                         valid_mask=np.ones(NORMALS_GT.shape, dtype=bool),
                         intrinsics=intrinsics)


@pytest.mark.parametrize(
    "protocol, prediction",
    [
        ("inverse_huber_irls_gt", 1.0 / (3.0 / NORMALS_GT + 0.2)
        ),  # affine in inverse depth: the inverse fit is exact
        ("depth_huber_irls_gt", 2.0 * NORMALS_GT +
         0.5),  # affine in depth: the depth-space fit is exact
    ])
def test_an_exactly_alignable_prediction_has_zero_normal_error(
        tmp_path, protocol, prediction):
    metrics = score_sequence([_normals_pair(tmp_path, prediction)],
                             DepthEvaluationConfig(protocol=protocol,
                                                   max_depth=100.0))
    assert metrics["mae_normal_deg"] == pytest.approx(0.0, abs=1e-3)
    assert metrics["abs_relative_difference"] == pytest.approx(0.0, abs=1e-4)


def test_a_wrongly_tilted_surface_has_a_large_normal_error(tmp_path):
    tilted_along_y = 2.0 + 0.02 * YY
    metrics = score_sequence([_normals_pair(tmp_path, tilted_along_y)],
                             DepthEvaluationConfig(
                                 protocol="inverse_huber_irls_gt",
                                 max_depth=100.0))
    assert metrics["mae_normal_deg"] > 10.0


def test_no_normal_error_without_intrinsics_or_when_geometry_metrics_are_off(
        tmp_path):
    prediction = 1.0 / (3.0 / NORMALS_GT + 0.2)
    config = DepthEvaluationConfig(protocol="inverse_huber_irls_gt",
                                   max_depth=100.0)
    assert "mae_normal_deg" not in score_sequence(
        [_normals_pair(tmp_path, prediction, intrinsics=None)], config)
    config = DepthEvaluationConfig(protocol="inverse_huber_irls_gt",
                                   max_depth=100.0,
                                   compute_geometry_metrics=False)
    assert "mae_normal_deg" not in score_sequence(
        [_normals_pair(tmp_path, prediction)], config)


# ---- log_depth_gt ------------------------------------------------------------------------------------------


def _log_pair(tmp_path: Path,
              gt: np.ndarray,
              code: np.ndarray,
              folder: str = "raw_log_depths") -> DepthPairData:
    pred_path = tmp_path / "scene" / folder / "frame.npy"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    return DepthPairData(gt_path=tmp_path / "gt.npy",
                         pred_path=pred_path,
                         ground_truth=gt,
                         prediction=code,
                         valid_mask=np.isfinite(gt))


def test_log_code_is_recovered_exactly(tmp_path):
    rng = np.random.default_rng(0)
    gt = rng.uniform(0.5, 8.0, size=(24, 32))
    a, b = 1.7, 0.9
    code = (np.log(gt) - b) / a  # the normalized log depth a model would emit
    config = DepthEvaluationConfig(protocol="log_huber_irls_gt",
                                   compute_geometry_metrics=False)
    metrics = score_sequence([_log_pair(tmp_path, gt, code)],
                             config,
                             diagnostic_suffix="frame")
    assert metrics["abs_relative_difference"] == pytest.approx(0.0, abs=1e-9)
    fit = json.loads((tmp_path / "scene" / "log_huber_irls_gt_fits" /
                      "frame.json").read_text())
    assert fit["scale"] == pytest.approx(a) and fit["shift"] == pytest.approx(b)
    assert fit["residual_unit"] == 1.0 and fit[
        "normalization"] == "residual / explicit unit"


def test_log_protocol_refuses_decoded_depths(tmp_path):
    gt = np.full((4, 4), 2.0)
    config = DepthEvaluationConfig(protocol="log_huber_irls_gt",
                                   compute_geometry_metrics=False)
    with pytest.raises(ValueError, match="raw_log_depths"):
        score_sequence([_log_pair(tmp_path, gt, gt, folder="depths")], config)


def test_explicit_unit_keeps_delta_absolute():
    # With unit 1 a residual of 0.1 is beyond delta 0.05 (down-weighted) whatever the targets' median.
    x = np.linspace(-1, 1, 200)
    y = 3.0 * x + 10.0
    y[::20] += 0.1
    a, b, fit = fit_huber_irls(x, y, unit=1.0)
    assert fit["normalization"] == "residual / explicit unit" and fit[
        "gt_inverse_median"] == 1.0
    assert a == pytest.approx(3.0, abs=1e-2) and b == pytest.approx(10.0,
                                                                    abs=1e-2)


# ---- native_depth_protocol ---------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("method,space", [("VGGT-OMEGA", "depth"),
                                          ("DA3-L", "depth"),
                                          ("DA3-G", "depth"),
                                          ("LOTUS_D", "inverse_depth"),
                                          ("DVD_V11", "inverse_depth")])
def test_native_protocol_cli(method, space):
    spec = resolve_method(method)
    assert spec.depth_output_space == space
    result = subprocess.run([
        sys.executable,
        str(ROOT / "scripts/test/benchmark_methods.py"), "--method", method,
        "--field", "default_depth_protocol"
    ],
                            capture_output=True,
                            text=True,
                            check=True)
    assert result.stdout.strip() == spec.default_depth_protocol


def test_engine_defaults_preserve_explicit_overrides():
    engine = (
        ROOT /
        "spatialarena/templates/_engine/run_method_case_list.sh").read_text()
    for key, legacy in [("MONO_DEPTH_PROTOCOL", "inverse_huber_irls_gt"),
                        ("VIDEO_DEPTH_PROTOCOL", "inverse_huber_irls_gt")]:
        expression = "${" + key + ":-${METHOD_DEPTH_PROTOCOL:-" + legacy + "}}"
        assert expression in engine
        for override, expected in [("", "depth_huber_irls_gt"),
                                   ("inverse_l1_gt", "inverse_l1_gt")]:
            result = subprocess.run(
                ["bash", "-c", 'printf "%s" "' + expression + '"'],
                env={
                    key: override,
                    "METHOD_DEPTH_PROTOCOL": "depth_huber_irls_gt"
                },
                capture_output=True,
                text=True,
                check=True)
            assert result.stdout == expected


def test_native_depth_does_not_read_raw_disparity_or_reject_dvd_metadata(
        tmp_path):
    (tmp_path / "depths").mkdir()
    (tmp_path / "raw_disparities").mkdir()
    depth = np.array([[1., 2., 4.], [2., 3., 5.]], dtype=np.float32)
    np.save(tmp_path / "depths/a.npy", depth)
    np.save(tmp_path / "raw_disparities/a.npy", 1 / depth)
    np.save(tmp_path / "gt.npy", depth)
    (tmp_path / "dvd_protocol.json").write_text("{}")
    scene = BenchmarkSceneInfo(dataset="dycheck",
                               task="VideoDepth",
                               scene="sample",
                               root_path=tmp_path)
    for protocol, folder, expected in [("depth_huber_irls_gt", "depths", depth),
                                       ("inverse_huber_irls_gt",
                                        "raw_disparities", 1 / depth)]:
        pair = prepare_depth_pair(scene_info=scene,
                                  gt_path=tmp_path / "gt.npy",
                                  pred_path=tmp_path / "depths/a.npy",
                                  config=DepthEvaluationConfig(
                                      protocol=protocol, max_depth=10))
        assert pair.pred_path.parent.name == folder
        np.testing.assert_allclose(pair.prediction, expected)


# ---- stored_sequence_fit -----------------------------------------------------------------------------------


class Recorder:

    def __init__(self):
        self.frames = []

    def write(self, pair, aligned, scale, shift, metrics):  # pylint: disable=unused-argument
        self.frames.append((scale, shift, np.array(aligned)))


def _stored_sequence(tmp_path, frames=3, shape=(6, 8), seed=0):
    rng = np.random.default_rng(seed)
    (tmp_path / "depths").mkdir()
    pairs = []
    for index in range(frames):
        gt = rng.uniform(0.5, 8.0, shape)
        prediction = 0.4 * gt + 0.3 + rng.normal(0.0, 0.05, shape)
        prediction[0,
                   0] = -1.0  # a nonpositive prediction, floored before the fit
        valid = np.ones(shape, dtype=bool)
        valid[-1, :] = False  # GT-invalid pixels still get an aligned value
        pairs.append(
            DepthPairData(gt_path=tmp_path / f"gt_{index}.npy",
                          pred_path=tmp_path / f"depths/{index}.npy",
                          ground_truth=gt,
                          prediction=prediction,
                          valid_mask=valid))
    return pairs


@pytest.mark.parametrize("protocol",
                         ["depth_huber_irls_gt", "inverse_huber_irls_gt"])
def test_saved_fit_rebuilds_aligned_maps(tmp_path, protocol):
    pairs = _stored_sequence(tmp_path)
    config = DepthEvaluationConfig(protocol=protocol,
                                   min_depth=1e-3,
                                   max_depth=10.0)
    recorder = Recorder()
    score_sequence(pairs, config, artifacts=recorder)
    fit = json.loads(
        (tmp_path / f"{protocol}_fit.json").read_text(encoding="utf-8"))
    assert fit["protocol"] == protocol
    for pair, (scale, shift, aligned) in zip(pairs,
                                             recorder.frames,
                                             strict=True):
        assert (scale, shift) == (fit["scale"], fit["shift"])
        if protocol == "inverse_huber_irls_gt":
            rebuilt = aligned_inverse_depth_map(
                pair.prediction,
                fit["scale"],
                fit["shift"],
                raw=fit["representation"] == "raw_disparity",
                inverse_floor=fit["inverse_floor"],
                config=config)
        else:
            rebuilt = aligned_depth_space_map(pair.prediction, fit["scale"],
                                              fit["shift"], config)
        np.testing.assert_array_equal(rebuilt, aligned)


# ---- monodepth_insufficient_gt -----------------------------------------------------------------------------


def test_single_valid_gt_is_reported_and_excluded(tmp_path, monkeypatch):
    scene = BenchmarkSceneInfo(dataset="fixture",
                               task="MonoDepth",
                               scene="test",
                               root_path=tmp_path,
                               depth_gt_path=tmp_path)
    paths = [tmp_path / "bad.npy", tmp_path / "good.npy"]
    monkeypatch.setattr(ev, "resolve_depth_config", lambda s, c: c)
    monkeypatch.setattr(ev,
                        "list_scene_gt_depth_files",
                        lambda s, gt_source="dense": paths)
    monkeypatch.setattr(ev, "find_depth_files", lambda p: paths)
    monkeypatch.setattr(ev, "materialized_mask_paths_by_depth", lambda s: {})
    monkeypatch.setattr(ev, "materialized_intrinsics_by_depth", lambda s: {})

    def prepare(**kw):
        mask = np.array([[True, kw["gt_path"].stem == "good"]])
        return ev.DepthPairData(
            gt_path=kw["gt_path"],
            pred_path=kw["pred_path"],
            ground_truth=np.array([[4.0, 5.0]]),
            prediction=np.array([[4.0, 5.0]]),
            valid_mask=mask,
        )

    monkeypatch.setattr(ev, "prepare_depth_pair", prepare)

    def score(pair, *args):  # pylint: disable=unused-argument
        assert pair.num_valid_pixels == 2
        return {"abs_relative_difference": 0.25}

    monkeypatch.setattr(ev, "evaluate_single_depth_pair", score)
    result = ev.evaluate_depth_scene(
        scene, tmp_path,
        ev.DepthEvaluationConfig(protocol="inverse_huber_irls_gt"))
    assert result["num_evaluated_frames"] == 1
    assert result["num_skipped_frames"] == 1
    assert result["metrics"]["abs_relative_difference"] == 0.25
    assert result["skipped_frame_details"][0]["num_valid_pixels"] == 1
    assert result["skipped_frame_details"][0]["gt_path"].endswith("bad.npy")


# ---- diode_prediction_clip ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "protocol", ["standard", "inverse_huber_irls_gt", "depth_huber_irls_gt"])
def test_diode_clip_configuration(tmp_path, protocol):
    scene = BenchmarkSceneInfo(dataset="marigold",
                               task="MonoDepth",
                               scene="diode",
                               root_path=tmp_path)
    config = ev.resolve_depth_config(
        scene, ev.DepthEvaluationConfig(protocol=protocol))
    assert config.post_clip_policy == "valid_gt_max"
    assert config.post_clip_min == 0.001
    assert config.post_clip_max is None
    assert config.filter_gt_depth_range is False


@pytest.mark.parametrize("upper", [120.0, 250.0])
def test_per_image_valid_gt_clip_actual_score(tmp_path, upper):
    gt = np.array([[0.3, 10.0, upper, 9999.0]], dtype=np.float32)
    mask = np.array([[True, True, True, False]])
    pair = ev.DepthPairData(gt_path=tmp_path / "gt.npy",
                            pred_path=tmp_path / "pred.npy",
                            ground_truth=gt,
                            prediction=gt.copy(),
                            valid_mask=mask)
    scene = BenchmarkSceneInfo(dataset="marigold",
                               task="MonoDepth",
                               scene="diode",
                               root_path=tmp_path)
    config = ev.resolve_depth_config(
        scene,
        ev.DepthEvaluationConfig(protocol="inverse_huber_irls_gt",
                                 inverse_floor_policy="gt_max",
                                 compute_geometry_metrics=False))
    result = ev.evaluate_single_depth_pair(pair, config)
    assert result["abs_relative_difference"] < 1e-5
    assert config.max_depth == 80.0  # The scoring override must not mutate GT mask configuration.


def test_clips_both_limits_and_excludes_invalid_gt():
    pair = ev.DepthPairData(gt_path=Path("gt.npy"),
                            pred_path=Path("pred.npy"),
                            ground_truth=np.array([[1.0, 120.0, 9999.0]],
                                                  dtype=np.float32),
                            prediction=np.array([[-1.0, 500.0, 9999.0]],
                                                dtype=np.float32),
                            valid_mask=np.array([[True, True, False]]))
    config = ev.DepthEvaluationConfig(alignment="none",
                                      post_clip_policy="valid_gt_max",
                                      compute_geometry_metrics=False)
    result = ev.evaluate_single_depth_pair(pair, config)
    assert result["abs_relative_difference"] == pytest.approx(0.4995)


@pytest.mark.parametrize("scene_name,policy", [("diode", "marigold_v2"),
                                               ("diode", "original"),
                                               ("eth3d", "marigold_v2")])
def test_diode_mask_matches_official_formula(tmp_path, scene_name, policy):
    from scipy import ndimage

    gt = np.full((12, 16), 2.0, dtype=np.float32)
    gt[:, 8:] = 5.0
    gt[0, 0] = 0.4
    gt[-1, -1] = 400.0
    mask = np.ones_like(gt, dtype=bool)
    mask[4, 3] = False
    pred = np.full_like(gt, 2.0)
    gp, pp, mp = [tmp_path / f"{name}.npy" for name in ("gt", "pred", "mask")]
    for path, value in [(gp, gt), (pp, pred), (mp, mask)]:
        np.save(path, value)
    scene = BenchmarkSceneInfo(dataset="marigold",
                               task="MonoDepth",
                               scene=scene_name,
                               root_path=tmp_path)
    config = ev.DepthEvaluationConfig(protocol="inverse_huber_irls_gt",
                                      filter_gt_depth_range=False,
                                      diode_mask_policy=policy)
    pair = ev.prepare_depth_pair(scene_info=scene,
                                 gt_path=gp,
                                 pred_path=pp,
                                 config=config,
                                 manifest_mask_paths={gp.resolve(): mp})
    expected = mask.copy()
    if scene_name == "diode" and policy == "marigold_v2":
        grad = np.abs(ndimage.sobel(gt, 0)) + np.abs(ndimage.sobel(gt, 1))
        expected &= (gt > 0.6) & (gt < 350.0) & (grad <= 0.3)
        assert expected.sum() < mask.sum()
    np.testing.assert_array_equal(pair.valid_mask, expected)
