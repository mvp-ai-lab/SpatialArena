# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""The depth-prior tracks: masked depth (spatialarena/depth_prior_masking.py) and sparse depth
(spatialarena/sparse_depth.py).

sparse_depth: The sparse-depth tracks' priors (spatialarena/sparse_depth.py).
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest
from spatialarena import depth_prior_masking as masking
from spatialarena.methods import TRACK_RGB_MASKED_DEPTH, TRACK_RGB_MASKED_DEPTH_CAMERAS
from PIL import Image
from spatialarena import sparse_depth
from spatialarena.methods import TRACK_RGB_SPARSE_DEPTH, TRACK_RGB_SPARSE_DEPTH_CAMERAS
from spatialarena.task_track_matrix import on_matrix, track_runs_on


# ---- depth_prior_masking -----------------------------------------------------------------------------------

sys.path.insert(0, "scripts/inference")


def test_track_names_match_the_registry():
    assert masking.TRACK_RGB_MASKED_DEPTH == TRACK_RGB_MASKED_DEPTH
    assert masking.TRACK_RGB_MASKED_DEPTH_CAMERAS == TRACK_RGB_MASKED_DEPTH_CAMERAS
    assert masking.protocol({}) == "cells32-drop0.5-v1"


def test_cell_mask_drops_the_ratio_of_cells_and_is_deterministic():
    keep = masking.cell_keep_mask(480, 640, seed=7)
    assert keep.shape == (480, 640)
    cell = 20  # ceil(640 / 32)
    cells = keep[::cell, ::cell]
    assert cells.size == 24 * 32 and cells.sum() == cells.size // 2
    # Every cell is kept or dropped as a whole.
    assert np.array_equal(np.repeat(np.repeat(cells, cell, 0), cell, 1), keep)
    assert np.array_equal(keep, masking.cell_keep_mask(480, 640, seed=7))
    assert not np.array_equal(keep, masking.cell_keep_mask(480, 640, seed=8))


def test_cell_mask_handles_sizes_that_are_not_a_multiple_of_the_cell():
    keep = masking.cell_keep_mask(375, 1242, seed=1)  # KITTI
    assert keep.shape == (375, 1242)
    assert 0.4 < keep.mean() < 0.6


def test_mask_depends_on_the_scene_and_image_path_only(tmp_path):
    depth = np.full((96, 128), 2.0, dtype=np.float32)
    first = tmp_path / "a" / "dtu" / "scan1"
    second = tmp_path / "b" / "dtu" / "scan1"  # the same scene under another root (e.g. SfM/dtu -> MVRecon/dtu)
    image = "images/0001.png"
    masked = masking.mask_depth_prior(depth, first, first / image)
    assert np.array_equal(masked, masking.mask_depth_prior(depth, second, second / image))
    assert not np.array_equal(masked, masking.mask_depth_prior(depth, first, first / "images/0002.png"))
    assert set(np.unique(masked)) == {0.0, 2.0}


def test_already_invalid_pixels_stay_invalid():
    depth = np.full((64, 64), 5.0, dtype=np.float32)
    depth[:, :32] = 0.0
    depth[0, 40] = np.nan
    masked = masking.mask_depth_prior(depth, Path("/s/scene"), "/s/scene/images/x.png")
    assert (masked[:, :32] == 0).all() and masked[0, 40] == 0
    assert np.isfinite(masked).all()
    assert 0.0 < masking.masked_valid_fraction(masked) < 0.5


def test_adapter_loader_masks_every_prior(tmp_path):
    from adapter_utils import load_masked_depth_priors, load_optional_depth_priors

    scene = tmp_path / "scene"
    (scene / "images").mkdir(parents=True)
    (scene / "depths").mkdir()
    for name in ("a", "b"):
        Image.new("RGB", (64, 48)).save(scene / "images" / f"{name}.png")
        np.save(scene / "depths" / f"{name}.npy", np.full((48, 64), 1.5, dtype=np.float32))
    images = [str(scene / "images" / "a.png"), str(scene / "images" / "b.png")]
    full, has_full, missing_full = load_optional_depth_priors(scene, images)
    masked, has_masked, missing_masked = load_masked_depth_priors(scene, images)
    assert has_full.tolist() == has_masked.tolist() == [True, True] and missing_full == missing_masked == []
    for full_map, masked_map, image in zip(full, masked, images):
        assert np.array_equal(masked_map, masking.mask_depth_prior(full_map, scene, image))
        assert 0.4 < masking.masked_valid_fraction(masked_map) < 0.6


def test_settings_come_from_the_environment_and_are_validated():
    assert masking.mask_settings({}) == (0.5, 32)
    assert masking.mask_settings({masking.RATIO_ENV: "", masking.GRID_ENV: ""}) == (0.5, 32)
    env = {masking.RATIO_ENV: "0.7", masking.GRID_ENV: "16"}
    assert masking.mask_settings(env) == (0.7, 16)
    assert masking.protocol(env) == "cells16-drop0.7-v1"
    for bad in ({masking.RATIO_ENV: "1"}, {masking.RATIO_ENV: "-0.1"}, {masking.RATIO_ENV: "half"},
                {masking.GRID_ENV: "0"}):
        with pytest.raises(ValueError):
            masking.mask_settings(bad)


def test_masks_of_one_image_are_nested_across_ratios(tmp_path):
    depth = np.full((120, 160), 4.0, dtype=np.float32)
    scene, image = tmp_path / "scene", tmp_path / "scene" / "images" / "a.png"
    kept = {
        ratio: masking.mask_depth_prior(depth, scene, image, env={masking.RATIO_ENV: str(ratio)}) > 0
        for ratio in (0.0, 0.3, 0.5, 0.7, 0.9)
    }
    assert kept[0.0].all()
    ratios = sorted(kept)
    for lower, higher in zip(ratios, ratios[1:]):
        assert (kept[higher] <= kept[lower]).all()  # every cell dropped at the lower ratio is dropped at the higher
        assert kept[higher].sum() < kept[lower].sum()
    assert abs(kept[0.7].mean() - 0.3) < 0.02


def test_masking_reads_the_ratio_from_the_process_environment(tmp_path, monkeypatch):
    depth = np.full((64, 64), 1.0, dtype=np.float32)
    scene, image = tmp_path / "s", tmp_path / "s" / "images" / "x.png"
    monkeypatch.setenv(masking.RATIO_ENV, "0.75")
    assert abs(masking.masked_valid_fraction(masking.mask_depth_prior(depth, scene, image)) - 0.25) < 0.02


def test_non_default_settings_name_their_own_method_id():
    from spatialarena.script_generator import _method_id  # pylint: disable=import-outside-toplevel

    assert masking.method_variant("rgb_masked_depth", {}) == ""
    assert masking.method_variant("rgb_masked_depth", {masking.RATIO_ENV: "0.5"}) == ""
    assert masking.method_variant("rgb_masked_depth", {masking.RATIO_ENV: "0.7"}) == "-mask0.7"
    assert masking.method_variant("rgb_masked_depth_cameras", {masking.GRID_ENV: "16"}) == "-mask0.5-grid16"
    assert masking.method_variant("rgb_depth", {masking.RATIO_ENV: "0.7"}) == ""
    assert _method_id("PI3_X", "rgb_masked_depth", {"BENCHMARK_DEPTH_MASK_RATIO": "0.7"}) \
        == "PI3_X-mask0.7_rgb_masked_depth"
    assert _method_id("PI3_X", "rgb_masked_depth", {"PI3_CKPT_PATH": "/x"}) == "PI3_X_rgb_masked_depth"
    assert _method_id("PI3_X", "rgb", {"BENCHMARK_DEPTH_MASK_RATIO": "0.7"}) == "PI3_X"


def test_engine_names_the_method_id_like_the_script_generator():
    """run_method_case_list.sh's METHOD_ID block gives the same name as script_generator._method_id."""
    import subprocess  # pylint: disable=import-outside-toplevel

    repo = Path(__file__).resolve().parents[1]
    script = (repo / "spatialarena" / "templates" / "_engine" / "run_method_case_list.sh").read_text()
    start = script.index('if [ "$BENCHMARK_TRACK" = "rgb" ]; then\n    METHOD_ID=')
    block = script[start:script.index("\nfi\n", start) + 4]
    for track, ratio, expected in (("rgb_masked_depth", "0.7", "PI3_X-mask0.7_rgb_masked_depth"),
                                   ("rgb_masked_depth_cameras", "", "PI3_X_rgb_masked_depth_cameras"),
                                   ("rgb_depth", "0.7", "PI3_X_rgb_depth")):
        result = subprocess.run(
            ["bash", "-c", f'{block}\necho "$METHOD_ID"'], capture_output=True, text=True, check=True,
            env={"PATH": "/usr/bin:/bin", "PYTHON": sys.executable, "PROJECT_ROOT": str(repo), "METHOD": "PI3_X",
                 "BENCHMARK_TRACK": track, "BENCHMARK_DEPTH_MASK_RATIO": ratio},
        )
        assert result.stdout.strip() == expected, (track, ratio, result.stderr)


# ---- sparse_depth ------------------------------------------------------------------------------------------

sys.path.insert(0, "scripts/inference")


def test_track_names_and_datasets():
    assert sparse_depth.SPARSE_DEPTH_TRACKS == {TRACK_RGB_SPARSE_DEPTH, TRACK_RGB_SPARSE_DEPTH_CAMERAS}
    assert track_runs_on("VideoDepth", "rgb_sparse_depth", "kitti")
    assert not track_runs_on("VideoDepth", "rgb_sparse_depth", "bonn")  # no LiDAR / SfM points
    assert track_runs_on("SfM", "rgb_sparse_depth_cameras", "imc")
    assert not on_matrix("MVRecon", "rgb_sparse_depth", "7scenes")  # its COLMAP model has no 3D points
    assert not on_matrix("MonoDepth", "rgb_sparse_depth", "ibims1")


def test_points_round_trip_and_scene_key(tmp_path):
    root = tmp_path / "benchmarks"
    data_path = root / "MVRecon" / "dtu" / "scan1"
    path = sparse_depth.prior_path(data_path, data_path / "images" / "0001.png")
    assert path == root / "_sparse_depth" / "dtu" / "scan1" / "0001.npz"
    sparse_depth.write_points(path, [[1.5, 2.5], [10.0, 3.0]], [2.0, 4.0], (20, 10), "sfm")
    xy, z, size, source = sparse_depth.read_points(path)
    assert xy.shape == (2, 2) and z.tolist() == [2.0, 4.0] and size == (20, 10) and source == "sfm"
    # The same scene under SfM resolves to the same priors.
    assert sparse_depth.prior_path(root / "SfM" / "dtu" / "scan1", "0001.png") == path


def test_splat_draws_blocks_that_survive_a_nearest_downsample():
    xy = np.array([[100.0, 50.0], [700.0, 300.0]])
    z = np.array([3.0, 7.0], np.float32)
    size = (1242, 375)
    block = sparse_depth.block_size(size, 512)
    assert block == 3
    dense = sparse_depth.splat_sparse_depth(xy, z, size, block=block)
    assert set(np.unique(dense)) == {0.0, 3.0, 7.0}  # no interpolated values
    assert (dense == 3.0).sum() == block * block
    small = np.asarray(Image.fromarray(dense).resize((512, 155), Image.Resampling.NEAREST))
    assert (small == 3.0).any() and (small == 7.0).any()  # each point survives the resize


def test_overlapping_points_keep_the_nearest():
    dense = sparse_depth.splat_sparse_depth([[5.0, 5.0], [5.4, 5.2]], [9.0, 2.0], (10, 10), block=1)
    assert dense[5, 5] == 2.0


def test_adapter_loader_marks_images_without_points_missing(tmp_path):
    from adapter_utils import load_sparse_depth_priors  # pylint: disable=import-outside-toplevel

    root = tmp_path / "benchmarks"
    data_path = root / "VideoDepth" / "kitti" / "drive"
    (data_path / "images").mkdir(parents=True)
    paths = []
    for name in ("a", "b"):
        Image.new("RGB", (64, 32)).save(data_path / "images" / f"{name}.png")
        paths.append(str(data_path / "images" / f"{name}.png"))
    sparse_depth.write_points(sparse_depth.prior_path(data_path, paths[0]), [[10, 10], [40, 20]], [5.0, 6.0],
                              (64, 32), "lidar")
    depths, has, missing = load_sparse_depth_priors(data_path, paths, model_long_side=64)
    assert has.tolist() == [True, False] and missing == [paths[1]] and depths[1] is None
    assert depths[0].shape == (32, 64) and sorted(np.unique(depths[0]).tolist()) == [0.0, 5.0, 6.0]
