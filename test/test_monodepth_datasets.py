# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""MonoDepth extension datasets: metadata-driven depth limits, the canonical layout writer, and oracle evaluation."""

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from spatialarena.dataset_licenses import is_safe_for_default_public_release, sources_for
from spatialarena.eval.canonical_dataset import BenchmarkSceneInfo, CanonicalDataset
from spatialarena.eval.dataset_registry import DEFAULT_DATASET_SPECS, find_dataset_spec
from spatialarena.eval.depth_prediction_evaluator import (
    MARIGOLD_MONODEPTH_LIMITS,
    DepthEvaluationConfig,
    evaluate_depth_scene,
    materialized_intrinsics_by_depth,
    monodepth_scene_limits,
    resolve_depth_config,
)
import scripts.preprocess.materialize_aerialmetric as aerial
import scripts.preprocess.monodepth_layout as layout
from scripts.preprocess.verify_benchmark_datasets import READY_REQUIREMENTS

NEW_DATASETS = ("aerialmetric", "hammer", "ibims1", "spring", "synth4k")


def scene_info(dataset: str,
               scene: str,
               metadata: dict | None = None,
               root: Path = Path("/nonexistent")):
    """Scene info with the given metadata entry, no files needed."""
    return BenchmarkSceneInfo(dataset=dataset,
                              task="MonoDepth",
                              scene=scene,
                              root_path=root,
                              metadata=metadata)


def resolved_limits(info: BenchmarkSceneInfo) -> tuple:
    """(min, max, scale, alignment resolution, range filter) after resolve_depth_config."""
    config = resolve_depth_config(info, DepthEvaluationConfig())
    return (
        config.min_depth,
        config.max_depth,
        config.gt_depth_scale,
        config.alignment_max_resolution,
        config.filter_gt_depth_range,
    )


@pytest.mark.parametrize("subset", sorted(MARIGOLD_MONODEPTH_LIMITS))
def test_marigold_limits_unchanged_with_and_without_metadata(subset):
    """Marigold limits stay those of the old hardcoded table."""
    expected = MARIGOLD_MONODEPTH_LIMITS[subset]
    assert resolved_limits(scene_info("marigold", subset)) == expected
    # The marigold metadata.json entries carry depth_scale/min_depth/max_depth equal to the table.
    min_depth, max_depth, depth_scale, _, _ = expected
    metadata = {
        "depth_scale": depth_scale,
        "min_depth": min_depth,
        "max_depth": max_depth
    }
    assert resolved_limits(scene_info("marigold", subset, metadata)) == expected


def test_metadata_limits_override_marigold_table_and_drive_new_datasets():
    """Declared metadata keys win over the table and set the new datasets' limits."""
    info = scene_info("marigold", "nyuv2", {"max_depth": 5.0})
    assert resolved_limits(info) == (1e-3, 5.0, 1e-3, None, True)

    metadata = {
        "min_depth": 3.0,
        "max_depth": 150.0,
        "depth_scale": 1.0,
        "filter_gt_depth_range": True
    }
    config = resolve_depth_config(scene_info("synth4k", "cyberpunk", metadata),
                                  DepthEvaluationConfig())
    assert config.alignment == "least_squares"
    assert (config.min_depth, config.max_depth, config.post_clip_min,
            config.post_clip_max) == (3.0, 150.0, 3.0, 150.0)


def test_scene_without_declared_limits_keeps_evaluator_defaults():
    """No declared limits and no table entry keep the evaluator defaults."""
    info = scene_info("hammer", "hammer", {"scene_name": "hammer"})
    assert monodepth_scene_limits(info) is None
    config = resolve_depth_config(info, DepthEvaluationConfig())
    assert (config.min_depth, config.max_depth,
            config.gt_depth_scale) == (1e-6, None, 1.0)


def test_new_datasets_are_registered_licensed_and_verified():
    """Every new dataset has a registry spec, a license entry and a verifier requirement."""
    registered = {(spec.task, spec.dataset) for spec in DEFAULT_DATASET_SPECS}
    for dataset in NEW_DATASETS:
        assert ("MonoDepth", dataset) in registered
        assert find_dataset_spec(
            "MonoDepth", dataset).relative_path == Path("MonoDepth") / dataset
        assert sources_for("MonoDepth", dataset), dataset
        assert ("MonoDepth", dataset) in READY_REQUIREMENTS
    # Game captures and YouTube-derived frames stay out of the public release.
    assert not is_safe_for_default_public_release("MonoDepth", "synth4k")
    assert not is_safe_for_default_public_release("MonoDepth", "aerialmetric")
    for dataset in ("hammer", "ibims1", "spring"):
        assert is_safe_for_default_public_release("MonoDepth", dataset)


def encode_moge_depth(depth: np.ndarray, near: float, far: float) -> bytes:
    """MoGe's write_depth: log-encoded uint16, 0 = unknown, 65535 = inf."""
    finite = np.isfinite(depth) & (depth > 0)
    encoded = np.zeros(depth.shape, dtype=np.uint16)
    encoded[finite] = np.clip(
        np.round(np.log(depth[finite] / near) / np.log(far / near) * 65533 + 1),
        1, 65534).astype(np.uint16)
    encoded[np.isinf(depth)] = 65535
    info = PngImagePlugin.PngInfo()
    info.add_text("near", str(near))
    info.add_text("far", str(far))
    buffer = io.BytesIO()
    Image.fromarray(encoded).save(buffer, format="PNG", pnginfo=info)
    return buffer.getvalue()


def test_decode_moge_depth_round_trip():
    """Decoding MoGe's depth PNG recovers depth, NaN and inf."""
    depth = np.array([[1.0, 2.0, 4.0], [np.nan, np.inf, 8.0]], dtype=np.float32)
    decoded = layout.decode_moge_depth(
        encode_moge_depth(depth, near=1.0, far=8.0))
    finite = np.isfinite(depth)
    np.testing.assert_allclose(decoded[finite], depth[finite], rtol=1e-4)
    assert np.isnan(decoded[1, 0]) and np.isinf(decoded[1, 1])


def test_drop_far_depth_follows_moge_rule():
    """Depth beyond 1000x the 1% quantile, zero and inf are invalid."""
    depth = np.full((10, 10), 2.0, dtype=np.float32)
    depth[0, 0] = 1999.0  # <= 1000 x the 1% quantile (2.0)
    depth[0, 1] = 2001.0
    depth[0, 2] = 0.0
    depth[0, 3] = np.inf
    valid = layout.drop_far_depth(depth)
    assert valid[0,
                 0] and not valid[0, 1] and not valid[0, 2] and not valid[0, 3]
    assert valid.sum() == 97


def test_layout_helpers():
    """Intrinsics scaling, frame keys and uniform subsampling."""
    pixel = layout.normalized_to_pixel_intrinsics(
        np.array([[0.5, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1]]), 640, 480)
    assert pixel == [[320.0, 0.0, 320.0], [0.0, 480.0, 240.0], [0.0, 0.0, 1.0]]
    assert layout.frame_key(
        "scene10_traj1_1/000000") == "scene10_traj1_1__000000"
    assert layout.frame_key(
        "Wild/park/rgbs/0001.jpg") == "Wild__park__rgbs__0001"
    items = [f"{index:03d}" for index in range(10)]
    assert layout.uniform_subsample(items, 4) == ["000", "003", "006", "009"]
    assert layout.uniform_subsample(items[:3], 4) == items[:3]


def write_fixture_dataset(root: Path) -> tuple[Path, list[np.ndarray]]:
    """Two-frame MonoDepth dataset in the canonical layout, one frame with an evaluation mask."""
    dataset_root = root / "MonoDepth" / "toyset"
    scene_root = dataset_root / "scene_a"
    rng = np.random.default_rng(0)
    depths, records = [], []
    for index in range(2):
        key = f"seq__{index:04d}"
        depth = rng.uniform(1.0, 20.0, size=(24, 32)).astype(np.float32)
        depth[0, :4] = 0.0  # invalid pixels stored as 0
        depths.append(depth)
        (scene_root / "images").mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros(
            (24, 32, 3),
            dtype=np.uint8)).save(scene_root / "images" / f"{key}.png")
        layout.save_depth(scene_root / "depths" / f"{key}.npy", depth)
        mask_rel = None
        if index == 1:
            mask = np.ones_like(depth, dtype=bool)
            mask[:, :8] = False
            mask_rel = f"masks/{key}.png"
            layout.save_mask(scene_root / mask_rel, mask)
        records.append(
            layout.FrameRecord(
                key=key,
                image=f"images/{key}.png",
                depth=f"depths/{key}.npy",
                source=f"seq/{index:04d}",
                intrinsics=[[30.0, 0.0, 16.0], [0.0, 30.0, 12.0],
                            [0.0, 0.0, 1.0]],
                mask=mask_rel,
            ))
    entry = layout.write_scene(
        dataset_root=dataset_root,
        dataset="toyset",
        scene="scene_a",
        records=records,
        limits=layout.SceneLimits(min_depth=1e-3, max_depth=50.0),
        provenance={"source": "test"},
    )
    layout.write_dataset_metadata(dataset_root, "toyset", [entry])
    return dataset_root, depths


def fixture_scene_info(dataset_root: Path) -> BenchmarkSceneInfo:
    """The single scene of the fixture dataset."""
    return CanonicalDataset(dataset_root,
                            task_name="MonoDepth").list_scenes()[0]


def test_write_scene_produces_canonical_layout(tmp_path):
    """write_scene / write_dataset_metadata produce what the adapter and evaluator read."""
    dataset_root, _ = write_fixture_dataset(tmp_path)
    metadata = json.loads((dataset_root / "metadata.json").read_text())
    assert metadata["total_images"] == 2
    [scene] = metadata["scenes"]
    assert (scene["min_depth"], scene["max_depth"],
            scene["depth_scale"]) == (1e-3, 50.0, 1.0)

    info = fixture_scene_info(dataset_root)
    assert info.dataset == "toyset" and info.scene == "scene_a"
    assert info.depth_gt_path == dataset_root / "scene_a" / "depths"
    intrinsics = materialized_intrinsics_by_depth(info)
    assert len(intrinsics) == 2
    assert all(
        np.allclose(matrix[0, 0], 30.0) for matrix in intrinsics.values())


@pytest.mark.parametrize("mode", ["identity", "affine"])
def test_oracle_prediction_scores_perfectly(tmp_path, mode):
    """GT (or an affine copy of it) as prediction scores AbsRel 0 and delta1 1."""
    dataset_root, depths = write_fixture_dataset(tmp_path / "bench")
    pred_dir = tmp_path / "pred" / "depths"
    pred_dir.mkdir(parents=True)
    for index, depth in enumerate(depths):
        prediction = depth if mode == "identity" else (depth - 0.7) / 3.5
        np.save(pred_dir / f"seq__{index:04d}.npy",
                prediction.astype(np.float32))

    result = evaluate_depth_scene(fixture_scene_info(dataset_root),
                                  pred_dir.parent, DepthEvaluationConfig())

    assert result["alignment"] == "least_squares"
    assert (result["min_depth"], result["max_depth"]) == (1e-3, 50.0)
    assert result["num_evaluated_frames"] == 2
    # 2 x 24 x 32 pixels, minus 4 zero-depth pixels per frame and the masked 8 columns of frame 1.
    assert result["num_valid_pixels"] == 2 * 24 * 32 - 2 * 4 - 24 * 8 + 4
    assert result["metrics"]["abs_relative_difference"] == pytest.approx(
        0.0, abs=1e-5)
    assert result["metrics"]["delta125_acc"] == pytest.approx(1.0)


def write_aerial_sample(sample_dir: Path, depth_value: float) -> None:
    """One AerialMetric norm-layout sample with constant depth."""
    sample_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((6, 8, 3),
                             dtype=np.uint8)).save(sample_dir / "image.jpg")
    np.save(sample_dir / "depth.npy",
            np.full((6, 8), depth_value, dtype=np.float32))
    (sample_dir / "meta.json").write_text(
        json.dumps({"intrinsics": [[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]}))


def test_aerialmetric_selection_takes_equal_site_shares_and_skips_unscorable_frames(
        tmp_path, monkeypatch):
    """Decoupled selection: equal share per site, unscorable frames skipped, stale files removed."""
    source = tmp_path / "extracted"
    for site in aerial.DECOUPLED_SITES:
        for index in range(10):
            # Sample 00 of every site lies beyond the 400 m range and must be skipped.
            write_aerial_sample(
                source / "decoupled-norm" / site / f"{index:02d}",
                500.0 if index == 0 else 50.0)

    selected = aerial.select_frames(source, "decoupled", 8)

    assert len(selected) == 8
    for site in aerial.DECOUPLED_SITES:
        picks = [item for item in selected if item.startswith(f"{site}/")]
        assert len(picks) == 2 and f"{site}/00" not in picks

    monkeypatch.setattr(layout, "SPLIT_ROOT", tmp_path / "splits")
    dataset_root = tmp_path / "MonoDepth" / "aerialmetric"
    stale = dataset_root / "decoupled_campus"
    stale.mkdir(parents=True)
    (stale / "materialized_full.json").write_text("{}")
    entry = aerial.materialize_scene(
        scene="decoupled",
        source_root=source,
        dataset_root=dataset_root,
        frames_per_scene=8,
        workers=1,
        overwrite=False,
        update_split=True,
    )
    assert entry["total_images"] == 8
    assert len(list((dataset_root / "decoupled" / "depths").glob("*.npy"))) == 8
    assert layout.read_split("aerialmetric", "decoupled") == selected
    assert sorted(aerial.select_frames(source, "decoupled", 0)) == sorted(
        f"{site}/{index:02d}" for site in aerial.DECOUPLED_SITES
        for index in range(1, 10))
    # Half of each site's 9 scorable samples (01..09), uniformly spaced over them.
    half = aerial.select_frames(source, "decoupled", 0, 0.5)
    for site in aerial.DECOUPLED_SITES:
        assert [item for item in half if item.startswith(f"{site}/")
               ] == [f"{site}/{i:02d}" for i in (1, 4, 6, 9)]
    assert [layout.fraction_count(total, 0.5) for total in (9, 1, 0)
           ] == [4, 1, 0]
    with pytest.raises(ValueError):
        layout.fraction_count(10, 0.0)
    aerial.remove_stale_scenes(dataset_root)
    assert not stale.exists() and (dataset_root / "decoupled").is_dir()
