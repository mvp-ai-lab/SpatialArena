# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""GenCeption: the adapter and its raymap decode.

genception_raymap: The GenCeption raymap decode (ported from representations4d's eval_camera_pose.py) on synthetic
raymaps laid out as the official evaluation reads them: known poses and intrinsics must come back.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from scripts.inference.genception import (
    GENCEPTION_HEIGHT,
    GENCEPTION_NUM_FRAMES,
    GENCEPTION_WIDTH,
    _cover_crop_frame,
    _export_clip_depths,
    _float_output,
    _intrinsics_on_originals,
    _load_video,
    _pad_to_81,
    _resize_to_original,
    _select_input_frame_indices,
    _select_output_frame_indices,
)
from spatialarena.geometry.genception_raymap import (
    RAYMAP_HEIGHT,
    RAYMAP_WIDTH,
    _smooth_centres,
    _smooth_rotations,
    decode_poses_from_raymap,
    raymap_masks,
    unscale_origin,
)

# ---- genception_adapter ------------------------------------------------------------------------------------


def test_long_inputs_keep_genceptions_own_temporal_sampling():
    """100 frames -> the 81 GenCeption samples (linspace), each predicted once under its own image name."""
    indices = _select_input_frame_indices(100)
    assert len(indices) == GENCEPTION_NUM_FRAMES and indices[
        0] == 0 and indices[-1] == 99
    assert len(set(indices.tolist())) == GENCEPTION_NUM_FRAMES
    assert _select_input_frame_indices(50).tolist() == list(range(50))


class FakeDepthPipe:
    """Always 81 output frames, like GenCeption; each frame's value is its clip's call number."""

    def __init__(self):
        self.calls = []

    def __call__(self, prompt_embeds, input_video):
        self.calls.append(len(input_video))
        return {
            "video":
                np.full((GENCEPTION_NUM_FRAMES, 1),
                        len(self.calls),
                        dtype=np.float32)
        }

    @staticmethod
    def decode_depth(video):
        """Depth frames of the fake video, at Genception's output size."""
        return np.broadcast_to(
            video[:, :, None],
            (len(video), GENCEPTION_HEIGHT, GENCEPTION_WIDTH)).copy()


def test_the_depth_task_predicts_every_frame_in_81_frame_clips(
        tmp_path, monkeypatch):
    """100 frames -> clips of 81 and 19: all 100 depth maps exist, each under its own image name."""
    import scripts.inference.genception as genception  # pylint: disable=import-outside-toplevel

    paths = []
    for index in range(100):
        path = tmp_path / f"{index:03d}.png"
        Image.new("RGB", (64, 48)).save(path)
        paths.append(str(path))
    exported = {}
    monkeypatch.setattr(
        genception, "export_depth_predictions", lambda output_path, names,
        depths: exported.update({names[0]: depths[0]}))
    pipe = FakeDepthPipe()
    _export_clip_depths(pipe, paths, tmp_path / "depth.npy", tmp_path / "out")
    assert pipe.calls == [81, 19]
    assert sorted(exported) == paths
    assert exported[paths[0]].shape == (48, 64) and exported[paths[80]][
        0, 0] == 1 and exported[paths[81]][0, 0] == 2


def test_load_video_resizes_on_the_cpu_and_keeps_each_original_size(tmp_path):
    """Mixed-size photos become one (T, 480, 832, 3) clip; the original sizes come back for the depth resize."""
    paths = []
    for index, size in enumerate(((1200, 800), (640, 480))):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", size, color=(index * 50, 0, 0)).save(path)
        paths.append(str(path))
    video, sizes = _load_video(paths)
    assert video.shape == (2, GENCEPTION_HEIGHT, GENCEPTION_WIDTH,
                           3) and video.dtype == np.uint8
    assert sizes == [(800, 1200), (480, 640)]


def test_select_output_frame_indices_identity_when_fewer_than_81():
    indices = _select_output_frame_indices(5)
    np.testing.assert_array_equal(indices, np.arange(5))


def test_select_output_frame_indices_identity_when_exactly_81():
    indices = _select_output_frame_indices(GENCEPTION_NUM_FRAMES)
    np.testing.assert_array_equal(indices, np.arange(GENCEPTION_NUM_FRAMES))


def test_select_output_frame_indices_returns_all_81_when_more_than_81():
    # representations4d's own preprocess_video linearly subsamples a >81
    # frame input down to 81 distinct frames (np.linspace(0, T-1, 81)), so
    # every one of the 81 output frames corresponds to a distinct real
    # input frame -- the identity mapping over 81 outputs is correct here,
    # not an approximation.
    indices = _select_output_frame_indices(200)
    np.testing.assert_array_equal(indices, np.arange(GENCEPTION_NUM_FRAMES))
    assert len(indices) == GENCEPTION_NUM_FRAMES


def test_resize_to_original_produces_requested_shape():
    frames = np.random.default_rng(0).integers(0,
                                               255,
                                               size=(3, 480, 832, 3),
                                               dtype=np.uint8)
    resized = _resize_to_original(frames, target_height=100, target_width=150)
    assert resized.shape == (3, 100, 150, 3)
    assert resized.dtype == np.uint8


def test_resize_to_original_handles_float_depth_frames():
    frames = np.random.default_rng(0).uniform(0.0, 5.0,
                                              size=(2, 480,
                                                    832)).astype(np.float32)
    resized = _resize_to_original(frames, target_height=64, target_width=96)
    assert resized.shape == (2, 64, 96)
    assert resized.dtype == np.float32


def test_cover_crop_keeps_the_aspect_ratio_and_centers_the_crop():
    """1280x480 -> scaled by 1 (height already 480) and cropped 224 px each side; 416x240 -> doubled, no crop."""
    wide = Image.fromarray(
        np.tile(
            np.arange(1280, dtype=np.uint16)[None, :, None] // 5,
            (480, 1, 3)).astype(np.uint8))
    frame, transform = _cover_crop_frame(wide)
    assert frame.shape == (GENCEPTION_HEIGHT, GENCEPTION_WIDTH,
                           3) and transform == (1.0, 1.0, 224.0, 0.0)
    assert frame[0, 0, 0] == 224 // 5
    _, transform = _cover_crop_frame(Image.new("RGB", (416, 240)))
    assert transform == (2.0, 2.0, 0.0, 0.0)


def test_intrinsics_on_the_crop_map_back_to_the_original_image():
    """A point projected with the original K lands where the crop K puts it after the crop transform."""
    transform = (0.5, 0.5, 48.0, 0.0
                )  # 1856x960 -> 928x480, cropped 48 px each side
    crop_k = np.array([[300.0, 0.0, 410.0], [0.0, 305.0, 238.0],
                       [0.0, 0.0, 1.0]])
    original_k = _intrinsics_on_originals(crop_k, [transform])[0]
    point = np.array([0.3, -0.2, 2.0])
    u_original, v_original = (original_k @ point)[:2] / point[2]
    u_crop, v_crop = (crop_k @ point)[:2] / point[2]
    assert u_original * transform[0] - transform[2] == pytest.approx(u_crop)
    assert v_original * transform[1] - transform[3] == pytest.approx(v_crop)


def test_pose_clips_are_padded_with_the_last_frame():
    video = np.arange(5, dtype=np.uint8)[:, None, None, None] * np.ones(
        (5, 2, 2, 3), dtype=np.uint8)
    padded = _pad_to_81(video)
    assert len(padded) == GENCEPTION_NUM_FRAMES and (
        padded[5:] == 4).all() and (padded[:5] == video).all()


def test_float_output_prefers_the_unquantized_copy():

    class Pipe:
        _last_video_float = np.full((1, 2, 2, 3), 0.123, dtype=np.float64)

    result = {"video": np.full((1, 2, 2, 3), 255, dtype=np.uint8)}
    assert _float_output(Pipe(), result)[0, 0, 0, 0] == pytest.approx(0.123)
    assert _float_output(object(), result).max() == 1.0


# ---- genception_raymap -------------------------------------------------------------------------------------


def _rotation(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def _raymap_video(
    rotations, centres, focal=(600.0, 610.0),
    principal=(416.0, 240.0)) -> np.ndarray:
    """World rays R @ K^-1 [u, v, 1] outside the origin block; the log-scaled centre inside it."""
    rows, cols = np.mgrid[:RAYMAP_HEIGHT, :RAYMAP_WIDTH]
    camera_rays = np.stack([(cols - principal[0]) / focal[0],
                            (rows - principal[1]) / focal[1],
                            np.ones(rows.shape)],
                           axis=-1)
    camera_rays /= np.linalg.norm(camera_rays, axis=-1, keepdims=True)
    frames = []
    for rotation, centre in zip(rotations, centres):
        frame = (np.einsum("ij,hwj->hwi", rotation, camera_rays) + 1.0) / 2.0
        frame[160:320, 288:544] = (np.sign(centre) * np.log1p(np.abs(centre)) +
                                   1.0) / 2.0
        frames.append(frame)
    return np.stack(frames).astype(np.float32)


def test_masks_follow_the_official_layout():
    origin, direction = raymap_masks()
    rows, cols = np.nonzero(origin)
    assert (rows.min(), rows.max() + 1, cols.min(),
            cols.max() + 1) == (192, 288, 320, 512)
    excluded_rows, excluded_cols = np.nonzero(~direction)
    assert (excluded_rows.min(), excluded_rows.max() + 1) == (128, 352)
    assert (excluded_cols.min(), excluded_cols.max() + 1) == (256, 576)
    assert not (origin & direction).any()


def test_unscale_origin_inverts_the_log_scaling():
    centre = np.array([-3.0, 0.0, 0.25, 12.0])
    np.testing.assert_allclose(unscale_origin(
        np.sign(centre) * np.log1p(np.abs(centre))),
                               centre,
                               atol=1e-12)


@pytest.mark.parametrize("solver", ["robust2", "svd"])
def test_decode_recovers_known_poses_and_intrinsics(solver):
    rotations = [
        _rotation([0.1, 1.0, 0.2], 0.05 * t) @ _rotation([1, 0, 0], 0.1)
        for t in range(6)
    ]
    centres = [np.array([0.4 * t, -0.2 * t, 2.0 + 0.1 * t]) for t in range(6)]
    decoded = decode_poses_from_raymap(_raymap_video(rotations, centres),
                                       solver=solver)
    raw = decoded.poses(smoothed=False)
    np.testing.assert_allclose(raw[:, :3, 3], np.stack(centres),
                               atol=1e-4)  # log-scaled origin decoded
    # robust2 takes the median ray as the forward axis: an approximation (~0.5 deg here), as upstream.
    np.testing.assert_allclose(raw[:, :3, :3],
                               np.stack(rotations),
                               atol=1e-2 if solver == "robust2" else 5e-3)
    np.testing.assert_allclose(raw[:, 3], np.tile([0.0, 0.0, 0.0, 1.0], (6, 1)))
    np.testing.assert_allclose(decoded.intrinsics[[0, 1], [0, 1]],
                               [600.0, 610.0],
                               rtol=1e-2)
    # The robust2 rotation bias shifts the fitted principal point by a few pixels.
    principal_tolerance = 5.0 if solver == "robust2" else 1.0
    np.testing.assert_allclose(decoded.intrinsics[[0, 1], [2, 2]],
                               [416.0, 240.0],
                               atol=principal_tolerance)
    smoothed = decoded.poses()
    for rotation in smoothed[:, :3, :3]:
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
    assert decoded.solver == solver


def test_smoothing_keeps_a_static_camera_and_damps_a_one_frame_spike():
    centres = np.zeros((9, 3))
    centres[4] = [5.0, 0.0, 0.0]
    smoothed = _smooth_centres(centres)
    assert np.abs(
        smoothed).max() == 0.0  # the median filter removes an isolated spike
    rotations = np.repeat(_rotation([0, 1, 0], 0.3)[None], 9, axis=0)
    np.testing.assert_allclose(_smooth_rotations(rotations),
                               rotations,
                               atol=1e-12)
    np.testing.assert_array_equal(_smooth_centres(centres[:2]),
                                  centres[:2])  # fewer than 3 frames: unchanged


def test_decode_rejects_bad_input():
    with pytest.raises(ValueError, match=r"\[T, H, W, 3\]"):
        decode_poses_from_raymap(np.zeros((RAYMAP_HEIGHT, RAYMAP_WIDTH, 3)))
    with pytest.raises(ValueError, match="solver"):
        decode_poses_from_raymap(np.zeros((1, RAYMAP_HEIGHT, RAYMAP_WIDTH, 3)),
                                 solver="procrustes")
    with pytest.raises(ValueError, match="smaller"):
        raymap_masks(240, 416)
