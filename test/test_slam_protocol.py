# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""SLAM protocol rules (spatialarena/slam_protocol.py) and their use across the pipeline.

slam_inputs: The SLAM track's fixed input resolution: protocol sizes, the shared resize/pad, and the adapters' fixed
mode.

slam_depth_export: Per-frame depth / confidence export of the SLAM wrappers (scripts/inference/slam_depth_export.py).
"""

from __future__ import annotations

import math
from pathlib import Path
import pytest
from spatialarena.slam_protocol import (
    EXCLUDED_METHODS,
    H200_FRAME_BUDGET,
    SLAM_RANGES,
    alignment_for_track,
    dataset_fps,
    in_range_track,
    method_slam_ranges,
    range_datasets,
    range_stride,
    range_track_methods,
)
import json
import numpy as np
from PIL import Image
from spatialarena.geometry.read_write_model import Camera
from spatialarena.slam_protocol import DATASET_INPUT_SIZE, SLAM_INPUT_WIDTH, input_root, input_size
from scripts.inference.adapter_utils import FIXED_INPUT_ENV, fixed_input_size, load_patch_grid_tensor
from scripts.inference.fixed_input_launcher import load_fixed_images
from scripts.preprocess.materialize_slam_inputs import fit_camera, fit_image, pad_box
from scripts.inference.slam_depth_export import (
    DepthSink,
    chunk_ranges,
    chunk_scale,
    export_long_chunks,
    export_numbered_maps,
)
from spatialarena.eval.prediction_artifacts import load_confidence_map
from spatialarena.eval.depth_prediction_evaluator import load_depth_map


# ---- slam_protocol -----------------------------------------------------------------------------------------

LONGEST_RECORDING = 3631  # TUM fr2/desk_with_person
OXFORD_FRAMES = 3840  # Oxford Spires sequences after LingBot-Map's preprocessing (first 3,840 cam0 frames at most)
VBR_LONGEST = 18846  # VBR ciampino_train1, every frame (LoGeR's processed release)
KITTI_LONGEST = 4661  # KITTI Odometry sequence 02, every frame


def test_track_max_frames_is_longest_recording_at_track_stride():
    for slam_range in ("short", "long"):
        spec = SLAM_RANGES[slam_range]
        assert spec["max_frames"] == math.ceil(LONGEST_RECORDING / spec["strides"]["tum_dynamics"]), slam_range
    # Oxford's short-range stride keeps it inside the TUM-defined bound, so the short track's membership is unchanged.
    assert math.ceil(OXFORD_FRAMES / range_stride("short", "oxford_spires")) <= SLAM_RANGES["short"]["max_frames"]
    assert SLAM_RANGES["large"]["max_frames"] == max(OXFORD_FRAMES, VBR_LONGEST, KITTI_LONGEST)


def test_range_datasets_and_strides():
    """Each range track's datasets and per-dataset strides."""
    assert range_datasets("short") == ["7scenes", "tum_dynamics", "oxford_spires", "spatialarena"]
    assert range_datasets("long") == ["7scenes", "tum_dynamics", "spatialarena"]
    assert range_datasets("large") == ["oxford_spires", "vbr", "kitti_odometry", "spatialarena"]
    assert range_stride("short", "tum_dynamics") == 10 and range_stride("long", "7scenes") == 3
    assert range_stride("short", "oxford_spires") == 12 and range_stride("large", "vbr") == 1
    with pytest.raises(ValueError):
        range_stride("medium", "tum_dynamics")
    with pytest.raises(ValueError):
        range_stride("long", "vbr")  # VBR is only in the large-scale track
    with pytest.raises(ValueError):
        range_stride("short", "kitti_odometry")  # so is KITTI Odometry


def test_longer_range_methods_are_also_shorter_range_methods():
    """Track membership is nested: large within long within short."""
    assert set(range_track_methods("large")) <= set(range_track_methods("long")) <= set(range_track_methods("short"))


def test_long_sequence_systems_are_in_every_track_and_excluded_methods_in_none():
    """Chunking systems join every track; excluded methods join none."""
    for method in ("VGGT_LONG", "PI_LONG", "VGGT_SLAM"):
        assert all(in_range_track(method, slam_range) for slam_range in SLAM_RANGES)
    for method in EXCLUDED_METHODS:
        assert method not in H200_FRAME_BUDGET
        assert not any(in_range_track(method, slam_range) for slam_range in SLAM_RANGES)


def test_membership_follows_budget():
    assert in_range_track("VGGT", "short") and not in_range_track("VGGT", "long")  # 400 frames: 364 ok, 1211 not
    assert in_range_track("DA3-L", "long") and not in_range_track("DA3-G", "long")  # 1500 vs 1000
    assert not in_range_track("PI3_X", "large")  # 3000-frame budget < the large track's longest input
    assert not in_range_track("UNKNOWN_METHOD", "short")


def test_beyond_budget_attempts_join_the_range_and_pass_the_case_check(capsys, monkeypatch):
    """The long-context methods are attempted on every large sequence, even past their measured budget."""
    import sys

    from spatialarena import slam_protocol

    for method in ("LOGER", "R3", "ZIPMAP", "SCAL3R"):
        assert in_range_track(method, "large") and in_range_track(method, "long")
    assert not in_range_track("ZIPMAP", "large", "rgb_cameras")
    monkeypatch.setattr(sys, "argv", ["slam_protocol", "check", "ZIPMAP", "large", "rgb", "18846"])
    slam_protocol.main()
    assert capsys.readouterr().out.strip() == "ok"


def test_low_budget_methods_are_attempted_on_short_and_long_ranges(capsys, monkeypatch):
    """Methods whose budget is below the short range's longest input (or unmeasured) run every sequence of the short
    and long ranges, their failures recorded, never skipped up front; they are not in the large range."""
    import sys

    from spatialarena import slam_protocol

    for method in ("MAPANYTHING", "MAST3R", "DEJAVIEW", "DUST3R", "POW3R", "IGGT4D"):
        assert method_slam_ranges(method, "rgb") == ["short", "long"], method
    assert method_slam_ranges("MAPANYTHING", "rgb_depth") == ["short", "long"]
    assert method_slam_ranges("DUST3R", "rgb_depth") == []  # RGB-only method
    assert method_slam_ranges("GENCEPTION_POSE", "rgb") == []  # still excluded
    for args, expected in ((["DUST3R", "short", "rgb", "364"], "ok"), (["MAST3R", "long", "rgb", "1211"], "ok"),
                           (["MAST3R", "large", "rgb", "3840"], "not_in_track")):
        monkeypatch.setattr(sys, "argv", ["slam_protocol", "check", *args])
        slam_protocol.main()
        assert capsys.readouterr().out.strip() == expected, args


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_track_support_alignment_and_frame_rates():
    assert method_slam_ranges("VGGT_LONG", "rgb") == ["short", "long", "large"]
    assert method_slam_ranges("VGGT_LONG", "rgb_depth") == []  # RGB-only system
    assert method_slam_ranges("VGGT", "rgb") == ["short"]
    assert method_slam_ranges("DEPTHVLM", "rgb") == []  # predicts no pose
    assert alignment_for_track("rgb") == "sim3" and alignment_for_track("rgb_depth") == "sim3"
    with pytest.raises(ValueError):
        alignment_for_track("rgb_cameras")
    assert dataset_fps("TUM_dynamics") == 30.0 and dataset_fps("oxford_spires") == 20.0
    with pytest.raises(ValueError):
        dataset_fps("kitti")


def test_planning_only_offers_slam_to_range_track_methods():
    from spatialarena.methods import METHOD_SPECS
    from spatialarena.task_track_matrix import runnable_tracks

    planned = {spec.name for spec in METHOD_SPECS if runnable_tracks("SLAM", spec)}
    assert planned == set(range_track_methods("short"))
    assert not planned & EXCLUDED_METHODS
    by_name = {spec.name: spec for spec in METHOD_SPECS}
    assert runnable_tracks("SLAM", by_name["VGGT_LONG"]) == ("rgb",)
    assert runnable_tracks("SfM", by_name["VGGT_LONG"]) == ()  # SLAM-only method


@pytest.mark.parametrize("slam_range", sorted(SLAM_RANGES))
def test_checked_in_case_lists_match_the_range_strides(slam_range):
    """The checked-in case lists use each dataset's stride and cover every sequence."""
    path = REPO_ROOT / "scripts" / "test" / f"full_standard_slam_cases_{slam_range}.txt"
    lines = [line for line in path.read_text().splitlines() if line and not line.startswith("#")]
    counts: dict[str, int] = {}
    for line in lines:
        task, dataset, _scene, interval = line.split("|")
        assert task == "SLAM" and int(interval) == range_stride(slam_range, dataset)
        counts[dataset] = counts.get(dataset, 0) + 1
    # SpatialArena's sequences each run only in the range they were chosen for: 3 short, 2 long, 3 large.
    spatialarena = {"short": 3, "long": 2, "large": 3}[slam_range]
    expected = {"7scenes": 18, "tum_dynamics": 9, "oxford_spires": 10, "vbr": 7, "kitti_odometry": 11,
                "spatialarena": spatialarena}
    assert counts == {dataset: expected[dataset] for dataset in range_datasets(slam_range)}


def test_sfm_evaluator_refuses_slam():
    from spatialarena.eval.sparse_reconstruction_evaluator import run_sparse_evaluation

    with pytest.raises(ValueError, match="slam_trajectory_evaluator"):
        run_sparse_evaluation(benchmark_root=REPO_ROOT, experiment_root=REPO_ROOT, experiment_name="x",
                              methods=["VGGT"], tasks=["SLAM"], datasets=[], scenes=[], include_scannet=False,
                              config=None)


def test_release_extractor_allows_links_inside_the_release_only(tmp_path):
    import io
    import tarfile

    sys_path = REPO_ROOT / "scripts" / "release"
    import sys

    sys.path.insert(0, str(sys_path))
    from extract_benchmark_archives import _safe_members

    root = tmp_path / "release"
    dest = root / "SLAM" / "7scenes"
    dest.mkdir(parents=True)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, target in (
            ("chess_seq-03/images/a.png", "../../../../MVRecon/7scenes/chess/images/a.png"),  # inside the release
            ("chess_seq-03/images/b.png", "../../../../../outside.png"),  # escapes the release
        ):
            info = tarfile.TarInfo(name)
            info.type, info.linkname = tarfile.SYMTYPE, target
            archive.addfile(info)
    buffer.seek(0)
    with tarfile.open(fileobj=buffer) as archive:
        members = _safe_members(archive, dest, root)
        assert next(members).name.endswith("a.png")
        with pytest.raises(ValueError, match="outside"):
            next(members)


# ---- slam_inputs -------------------------------------------------------------------------------------------

ENGINE = Path(__file__).resolve().parents[1] / "spatialarena" / "templates" / "_engine" / "run_method_case_list.sh"
NATIVE_SIZE = {"7scenes": (640, 480), "tum_dynamics": (640, 480), "oxford_spires": (1440, 1080), "vbr": (1388, 700),
               "kitti_odometry": (1241, 376), "spatialarena": (1600, 1296)}


def _frames(directory: Path, size: tuple[int, int], count: int = 2) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        path = directory / f"{index:06d}.png"
        Image.fromarray(np.full((size[1], size[0], 3), index * 40, dtype=np.uint8)).save(path)
        paths.append(str(path))
    return paths


def test_every_slam_dataset_has_one_width_on_the_14_px_grid_and_its_own_aspect_ratio():
    """One width for all datasets; heights on the 14 px grid; the target keeps (almost) each dataset's aspect."""
    assert set(DATASET_INPUT_SIZE) == set(NATIVE_SIZE)
    for dataset, (width, height) in DATASET_INPUT_SIZE.items():
        assert width == SLAM_INPUT_WIDTH and width % 14 == 0 and height % 14 == 0, dataset
        native_w, native_h = NATIVE_SIZE[dataset]
        assert abs(width / height - native_w / native_h) / (native_w / native_h) < 0.01, dataset
    assert input_size("TUM_dynamics") == (504, 378) and input_size("vbr") == (504, 252)
    assert input_root(Path("/b"), "vbr") == Path("/b/SLAM/vbr")
    with pytest.raises(ValueError):
        input_size("kitti")


def test_pad_box_keeps_4_3_frames_whole_and_pads_vbr_and_kitti():
    """640x480 and 1440x1080 are exactly 4:3 (no pad needed); VBR's 1388x700 and KITTI Odometry's wide frames pad
    columns instead of losing them."""
    assert pad_box(640, 480, (504, 378)) == (0.7875, 0, 0, 504, 378)
    assert pad_box(1440, 1080, (504, 378)) == (0.35, 0, 0, 504, 378)
    assert pad_box(1388, 700, (504, 252)) == (0.36, 2, 0, 500, 252)
    assert pad_box(800, 300, (504, 252)) == (0.63, 0, 31, 504, 189)
    # KITTI Odometry (about 3.3:1) pads a few rows top/bottom instead of cropping columns.
    assert pad_box(1241, 376, (504, 154)) == pytest.approx((504 / 1241, 0, 0, 504, 153), abs=1e-9)


def test_fit_image_and_camera_agree():
    """Images and depths land at the target size; the intrinsics follow the same resize-within and pad; the padded
    border is zero, never fabricated real content."""
    image = np.zeros((700, 1388, 3), dtype=np.uint8)
    depth = np.full((700, 1388), 1234, dtype=np.uint16)
    assert fit_image(image, (504, 252)).shape == (252, 504, 3)
    fitted_depth = fit_image(depth, (504, 252), nearest=True)
    assert fitted_depth.shape == (252, 504) and fitted_depth.dtype == np.uint16
    # pad_box(1388, 700, (504, 252)) == (0.36, pad_l=2, pad_t=0, resized_w=500, resized_h=252): a 2px zero column
    # on each side, every real (resized) pixel still 1234.
    np.testing.assert_array_equal(fitted_depth[:, :2], 0)
    np.testing.assert_array_equal(fitted_depth[:, 502:], 0)
    assert (fitted_depth[:, 2:502] == 1234).all()
    exact = np.arange(378 * 504 * 3, dtype=np.uint32).astype(np.uint8).reshape(378, 504, 3)
    np.testing.assert_array_equal(fit_image(exact, (504, 378)), exact)  # already at the target: untouched

    camera = fit_camera(Camera(id=1, model="PINHOLE", width=640, height=480,
                               params=np.array([585.0, 585.0, 320.0, 240.0])), (504, 378))
    scale = 504 / 640
    assert (camera.width, camera.height) == (504, 378)
    np.testing.assert_allclose(camera.params, [585 * scale, 585 * scale, 320.5 * scale - 0.5, 240.5 * scale - 0.5])
    vbr = fit_camera(Camera(id=1, model="PINHOLE", width=1388, height=700,
                            params=np.array([700.0, 700.0, 694.0, 350.0])), (504, 252))
    scale, pad_l, pad_t, _resized_w, _resized_h = pad_box(1388, 700, (504, 252))
    np.testing.assert_allclose(vbr.params[2], (694.0 + 0.5) * scale - 0.5 + pad_l)  # the pad shifts cx
    np.testing.assert_allclose(vbr.params[3], (350.0 + 0.5) * scale - 0.5 + pad_t)


def test_fixed_input_size_only_in_fixed_mode(tmp_path, monkeypatch):
    """None unless the engine sets FIXED_INPUT_RESOLUTION=1; then the frames' common size, mixed sizes refused."""
    paths = _frames(tmp_path / "a", (504, 378))
    monkeypatch.delenv(FIXED_INPUT_ENV, raising=False)
    assert fixed_input_size(paths) is None
    monkeypatch.setenv(FIXED_INPUT_ENV, "1")
    assert fixed_input_size(paths) == (504, 378)
    with pytest.raises(ValueError, match="share one size"):
        fixed_input_size(paths + _frames(tmp_path / "b", (504, 252)))


def test_patch_grid_crop_and_launcher_loader_never_rescale(tmp_path):
    """16 px models get a center crop (504x378 -> 496x368); the VGGT launcher loader keeps the frames as they are."""
    paths = _frames(tmp_path / "images", (504, 378))
    assert tuple(load_patch_grid_tensor(paths, 16).shape) == (2, 3, 368, 496)
    assert tuple(load_patch_grid_tensor(paths, 14).shape) == (2, 3, 378, 504)
    record = tmp_path / "input.json"
    images = load_fixed_images(paths, record=record)
    assert tuple(images.shape) == (2, 3, 378, 504)
    assert json.loads(record.read_text(encoding="utf-8")) == {"model_input_size": [504, 378]}
    with pytest.raises(ValueError, match="patch grid"):
        load_fixed_images(_frames(tmp_path / "odd", (500, 378)))


def test_engine_feeds_full_slam_the_fixed_inputs():
    """full-slam (any of the three case lists) uses the pre-sized inputs at the protocol width."""
    engine = ENGINE.read_text(encoding="utf-8")
    assert "full_standard_slam_cases_large.txt)" in engine
    assert "FIXED_INPUT_RESOLUTION=1" in engine and "slam_protocol input_root" in engine


# ---- slam_depth_export -------------------------------------------------------------------------------------

def poses_along_x(count: int, step: float = 1.0) -> np.ndarray:
    """Camera-to-world poses moving along +x."""
    poses = np.tile(np.eye(4), (count, 1, 1))
    poses[:, 0, 3] = np.arange(count) * step
    return poses


def test_chunk_ranges_match_the_systems_chunking():
    """60-frame chunks sharing 30 frames, as VGGT-Long / Pi-Long cut a sequence; short sequences are one chunk."""
    assert chunk_ranges(50, 60, 30) == [(0, 50)]
    assert chunk_ranges(130, 60, 30) == [(0, 60), (30, 90), (60, 120), (90, 130)]


def test_chunk_scale_is_the_similarity_scale():
    """Any Sim(3) of a chunk's trajectory is recovered as the ratio of camera distances."""
    chunk = poses_along_x(10)
    chunk[:, 1, 3] = np.sin(np.arange(10))
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    final = chunk.copy()
    final[:, :3, 3] = 2.5 * chunk[:, :3, 3] @ rotation.T + np.array([4.0, -1.0, 2.0])
    assert abs(chunk_scale(chunk, final) - 2.5) < 1e-9


def test_long_chunks_are_rescaled_and_later_chunks_win(tmp_path):
    """Each chunk's depth is multiplied by its own scale; a frame in two chunks takes the later one."""
    image_paths = [str(tmp_path / f"img_{index:03d}.png") for index in range(90)]
    final = poses_along_x(90)
    chunk_dir = tmp_path / "_tmp_results_unaligned"
    chunk_dir.mkdir()
    for chunk_index, ((start, stop), scale) in enumerate(zip(chunk_ranges(90, 60, 30), (1.0, 4.0))):
        count = stop - start
        own = poses_along_x(count)
        own[:, 0, 3] = (np.arange(start, stop)) / scale  # the chunk's trajectory is the final one / scale
        np.save(chunk_dir / f"chunk_{chunk_index}.npy", {
            "depth": np.full((count, 4, 6), 2.0 / scale, dtype=np.float32),
            "depth_conf": np.full((count, 4, 6), chunk_index + 1.0, dtype=np.float32),
            "extrinsic": own[:, :3, :4],
        }, allow_pickle=True)
    sink = DepthSink(tmp_path / "out", image_paths)
    export_long_chunks(sink, chunk_dir, chunk_ranges(90, 60, 30), dict(enumerate(final)), "VGGT_LONG")
    metrics = sink.metrics()
    assert metrics.pop("depth_export_seconds") >= 0.0
    assert metrics == {"depth_frames": 90, "depth_confidence": True}
    for index in (0, 45, 89):  # every frame ends at depth 2 in the final trajectory's scale
        assert np.allclose(load_depth_map(tmp_path / "out" / "depths" / f"img_{index:03d}.npz"), 2.0)
    shared = load_confidence_map(tmp_path / "out" / "confidences" / "img_045.npz")
    assert np.allclose(shared, 2.0)  # frame 45 is in both chunks: the later wins
    assert np.allclose(load_confidence_map(tmp_path / "out" / "confidences" / "img_010.npz"), 1.0)


def test_pi_long_chunks_use_local_point_depth(tmp_path):
    """Pi-Long chunks give Pi3 local points; depth is their z."""
    image_paths = [str(tmp_path / f"{index}.jpg") for index in range(5)]
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    points = np.zeros((5, 2, 3, 3), dtype=np.float32)
    points[..., 2] = 3.0
    np.save(chunk_dir / "chunk_0.npy", {"local_points": points, "conf": np.full((5, 2, 3, 1), 0.5),
                                        "camera_poses": poses_along_x(5)}, allow_pickle=True)
    sink = DepthSink(tmp_path / "out", image_paths)
    export_long_chunks(sink, chunk_dir, [(0, 5)], dict(enumerate(poses_along_x(5))), "PI_LONG")
    depth = load_depth_map(tmp_path / "out" / "depths" / "3.npz")
    assert depth.shape == (2, 3) and np.allclose(depth, 3.0)


def test_numbered_maps_follow_the_returned_frame_ids(tmp_path):
    """R3 writes the k-th returned frame as <k>.npy; it maps to that frame's input index and image name."""
    (tmp_path / "depth").mkdir()
    (tmp_path / "conf").mkdir()
    for position, value in enumerate((1.0, 7.0)):
        np.save(tmp_path / "depth" / f"{position:06d}.npy", np.full((2, 2), value))
        np.save(tmp_path / "conf" / f"{position:06d}.npy", np.full((2, 2), value / 10))
    sink = DepthSink(tmp_path / "out", [str(Path("frames") / f"f{index}.png") for index in range(4)])
    export_numbered_maps(sink, tmp_path / "depth", [0, 3], tmp_path / "conf")
    assert np.allclose(load_depth_map(tmp_path / "out" / "depths" / "f3.npz"), 7.0)
    assert not (tmp_path / "out" / "depths" / "f1.npz").exists()
