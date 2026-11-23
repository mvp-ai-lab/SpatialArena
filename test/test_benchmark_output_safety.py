# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from pathlib import Path
import struct

import numpy as np
import pytest

from spatialarena.eval.result_layout import (
    path_is_relative_to,
    validate_output_path_outside_benchmark_root,
)
from scripts.inference.adapter_utils import (
    canonical_image_root,
    cleanup_intermediate_paths,
    export_camera_pose_priors,
    export_depth_predictions,
    load_colmap_camera_priors,
    load_depth_prior_file,
    load_depth_priors,
    load_optional_depth_priors,
    resolve_camera_conditioned_inputs,
    resolve_camera_conditioned_image_paths,
)
from spatialarena.eval.prediction_artifacts import (
    requirements_for_task,
    validate_prediction_output,
)
from spatialarena.geometry import read_write_model
from scripts.preprocess.materialize_benchmark_camera_priors import materialize
from scripts.preprocess.materialize_depth_visualizations import materialize_visualizations


def test_path_is_relative_to_accepts_equal_or_nested_paths(
        tmp_path: Path) -> None:
    benchmark_root = tmp_path / "benchmarks"

    assert path_is_relative_to(benchmark_root, benchmark_root)
    assert path_is_relative_to(benchmark_root / "3DGS" / "test", benchmark_root)
    assert not path_is_relative_to(tmp_path / "benchmark_reports",
                                   benchmark_root)


def test_validate_output_path_rejects_benchmark_root_outputs(
        tmp_path: Path) -> None:
    benchmark_root = tmp_path / "benchmarks"
    output_path = benchmark_root / "3DGS" / "test" / "metrics.json"

    with pytest.raises(ValueError, match="outside benchmark_root"):
        validate_output_path_outside_benchmark_root(output_path, benchmark_root)


def test_validate_output_path_allows_external_reports(tmp_path: Path) -> None:
    validate_output_path_outside_benchmark_root(
        tmp_path / "reports" / "metrics.json",
        tmp_path / "benchmarks",
    )


def test_materialize_depth_visualizations_removes_numeric_depth(
        tmp_path: Path) -> None:
    prediction_root = tmp_path / "predictions"
    depth_dir = prediction_root / "depths"
    depth_dir.mkdir(parents=True)
    depth_path = depth_dir / "000000.npy"
    np.save(depth_path, np.arange(9, dtype=np.float32).reshape(3, 3))

    result = materialize_visualizations(
        prediction_root,
        tmp_path / "reports" / "depth_visualizations",
        remove_numeric_depths=True,
    )

    assert result["num_png_written"] == 1
    assert result["num_numeric_removed"] == 1
    assert not depth_path.exists()
    assert (tmp_path / "reports" / "depth_visualizations" /
            "000000.png").is_file()


def test_cleanup_intermediate_paths_removes_scratch_dirs(
        tmp_path: Path) -> None:
    scratch = tmp_path / "_resized_input_images"
    scratch.mkdir()
    (scratch / "000000.png").write_bytes(b"not a real png")

    cleanup_intermediate_paths([scratch], keep=False)

    assert not scratch.exists()


def test_cleanup_intermediate_paths_can_keep_scratch_dirs(
        tmp_path: Path) -> None:
    scratch = tmp_path / "_cut3r_raw"
    scratch.mkdir()

    cleanup_intermediate_paths([scratch], keep=True)

    assert scratch.is_dir()


def test_load_depth_priors_resolves_sintel_dpt_layout(tmp_path: Path) -> None:
    scene = "alley_1"
    image_dir = tmp_path / "training" / "final" / scene
    depth_dir = tmp_path / "training" / "depth" / scene
    image_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)
    image_path = image_dir / "frame_0001.png"
    image_path.write_bytes(b"not decoded by loader")

    depth = np.arange(6, dtype=np.float32).reshape(2, 3)
    with (depth_dir / "frame_0001.dpt").open("wb") as handle:
        handle.write(struct.pack("f", 202021.25))
        handle.write(struct.pack("i", depth.shape[1]))
        handle.write(struct.pack("i", depth.shape[0]))
        depth.tofile(handle)

    loaded = load_depth_priors(image_dir, [str(image_path)])

    assert len(loaded) == 1
    np.testing.assert_allclose(loaded[0], depth)


def test_load_depth_priors_reads_pfm_depth_prior(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    depth_dir = tmp_path / "depths"
    image_dir.mkdir()
    depth_dir.mkdir()
    image_path = image_dir / "000000_key.png"
    image_path.write_bytes(b"not decoded by loader")

    depth = np.arange(6, dtype=np.float32).reshape(2, 3)
    with (depth_dir / "000000_key.pfm").open("wb") as handle:
        handle.write(b"Pf\n")
        handle.write(b"3 2\n")
        handle.write(b"-1.0\n")
        np.flipud(depth).astype(np.float32).tofile(handle)

    loaded = load_depth_priors(tmp_path, [str(image_path)])

    assert len(loaded) == 1
    np.testing.assert_allclose(loaded[0], depth)


def test_load_optional_depth_priors_preserves_missing_positions(
        tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    depth_dir = tmp_path / "depths"
    image_dir.mkdir()
    depth_dir.mkdir()
    first_image = image_dir / "000000_key.png"
    second_image = image_dir / "000001_src001.png"
    first_image.write_bytes(b"not decoded by loader")
    second_image.write_bytes(b"not decoded by loader")

    depth = np.arange(6, dtype=np.float32).reshape(2, 3)
    with (depth_dir / "000000_key.pfm").open("wb") as handle:
        handle.write(b"Pf\n")
        handle.write(b"3 2\n")
        handle.write(b"-1.0\n")
        np.flipud(depth).astype(np.float32).tofile(handle)

    loaded, has_depth, missing = load_optional_depth_priors(
        tmp_path,
        [str(first_image), str(second_image)],
    )

    assert has_depth.tolist() == [True, False]
    assert loaded[1] is None
    assert missing == [str(second_image)]
    np.testing.assert_allclose(loaded[0], depth)


def test_load_depth_prior_file_reads_near_maximum_16_bit_values_as_missing(
        tmp_path: Path) -> None:
    from PIL import Image

    # 7-Scenes millimetre PNGs mark missing depth with 65535 (raw) or 65000-65534 (colour-registered); neither may
    # load as a ~65 m point.
    path = tmp_path / "frame.depth.proj.png"
    Image.fromarray(
        np.array([[1500, 65535, 65061], [0, 2000, 64999]],
                 dtype=np.uint16)).save(path)
    for png_scale in (None, 1000.0):
        np.testing.assert_allclose(load_depth_prior_file(path, path, png_scale),
                                   [[1.5, 0.0, 0.0], [0.0, 2.0, 64.999]])


def test_materialize_sample_json_camera_priors(tmp_path: Path) -> None:
    scene_root = tmp_path / "scene"
    image_dir = scene_root / "images"
    image_dir.mkdir(parents=True)
    from PIL import Image

    for name in ("000000_key.png", "000001_src001.png"):
        Image.new("RGB", (8, 6), (0, 0, 0)).save(image_dir / name)
    sample = {
        "images": [
            {
                "canonical_path": "images/000000_key.png"
            },
            {
                "canonical_path": "images/000001_src001.png"
            },
        ],
        "poses": [
            np.eye(4, dtype=np.float32).tolist(),
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.1],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ).tolist(),
        ],
        "intrinsics": [
            [[4.0, 0.0, 3.5], [0.0, 4.0, 2.5], [0.0, 0.0, 1.0]],
            [[4.0, 0.0, 3.5], [0.0, 4.0, 2.5], [0.0, 0.0, 1.0]],
        ],
    }
    import json

    (scene_root / "sample.json").write_text(json.dumps(sample),
                                            encoding="utf-8")
    output_dir = tmp_path / "priors" / "sparse" / "0"

    assert materialize(scene_root, output_dir) == 2
    poses, intrinsics = load_colmap_camera_priors(
        scene_root,
        [
            str(image_dir / "000000_key.png"),
            str(image_dir / "000001_src001.png")
        ],
        colmap_path=output_dir,
    )

    assert poses.shape == (2, 4, 4)
    assert intrinsics.shape == (2, 3, 3)
    np.testing.assert_allclose(poses[1, :3, 3], [0.1, 0.0, 0.0], atol=1e-6)


def test_materialize_dycheck_camera_priors(tmp_path: Path) -> None:
    import json

    from PIL import Image

    from scripts.preprocess.materialize_dycheck import materialize_dataset

    source = tmp_path / "source" / "apple"
    for folder in ("dense/rgb/1x", "depth/2x", "dense/dycheck/camera"):
        (source / folder).mkdir(parents=True)
    # A camera rotated 90 degrees about z, centred at (1, 2, 3); intrinsics given at twice the linked image size.
    orientation = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    for stem in ("0_00000", "0_00001"):
        Image.new("RGB", (8, 6)).save(source / "dense/rgb/1x" / f"{stem}.png")
        np.save(source / "depth/2x" / f"{stem}.npy",
                np.ones((3, 4), dtype=np.float32))
        camera = {
            "orientation": orientation,
            "position": [1.0, 2.0, 3.0],
            "focal_length": 10.0,
            "principal_point": [8.0, 6.0],
            "image_size": [16, 12],
            "pixel_aspect_ratio": 1.0,
            "skew": 0.0,
            "radial_distortion": [0.0, 0.0, 0.0],
            "tangential_distortion": [0.0, 0.0]
        }
        (source / "dense/dycheck/camera" / f"{stem}.json").write_text(
            json.dumps(camera), encoding="utf-8")
    scene_root = materialize_dataset(tmp_path / "source",
                                     tmp_path / "benchmarks",
                                     scenes=["apple"]) / "apple"
    output_dir = tmp_path / "priors" / "sparse" / "0"

    assert materialize(scene_root, output_dir) == 2
    images = sorted(str(path) for path in (scene_root / "images").iterdir())
    poses, intrinsics = load_colmap_camera_priors(scene_root,
                                                  images,
                                                  colmap_path=output_dir)

    np.testing.assert_allclose(poses[0, :3, :3],
                               np.array(orientation).T,
                               atol=1e-6)
    np.testing.assert_allclose(poses[0, :3, 3], [1.0, 2.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(
        intrinsics[0], [[5.0, 0.0, 4.0], [0.0, 5.0, 3.0], [0.0, 0.0, 1.0]],
        atol=1e-6)


def test_load_colmap_camera_priors_accepts_batch_subset_of_larger_model(
        tmp_path: Path) -> None:
    scene_root = tmp_path / "scene"
    image_dir = scene_root / "images"
    image_dir.mkdir(parents=True)
    from PIL import Image

    names = ("000000_key.png", "000001_src001.png", "000002_src002.png")
    for name in names:
        Image.new("RGB", (8, 6), (0, 0, 0)).save(image_dir / name)
    sample = {
        "images": [{
            "canonical_path": f"images/{name}"
        } for name in names],
        "poses": [np.eye(4, dtype=np.float32).tolist() for _ in names],
        "intrinsics": [
            [[4.0, 0.0, 3.5], [0.0, 4.0, 2.5], [0.0, 0.0, 1.0]] for _ in names
        ],
    }
    import json

    (scene_root / "sample.json").write_text(json.dumps(sample),
                                            encoding="utf-8")
    output_dir = tmp_path / "priors" / "sparse" / "0"

    assert materialize(scene_root, output_dir) == 3

    # A batched inference call only requests one chunk of a much larger scene;
    # COLMAP records outside that chunk must not be treated as a mapping
    # failure (see spatialarena/eval/colmap_view_resolver.py strict mode).
    poses, intrinsics = load_colmap_camera_priors(
        scene_root,
        [str(image_dir / "000001_src001.png")],
        colmap_path=output_dir,
    )

    assert poses.shape == (1, 4, 4)
    assert intrinsics.shape == (1, 3, 3)


def test_load_colmap_camera_priors_still_rejects_unresolved_requested_image(
    tmp_path: Path,) -> None:
    scene_root = tmp_path / "scene"
    image_dir = scene_root / "images"
    image_dir.mkdir(parents=True)
    from PIL import Image

    Image.new("RGB", (8, 6), (0, 0, 0)).save(image_dir / "000000_key.png")
    unregistered = image_dir / "not_in_colmap.png"
    Image.new("RGB", (8, 6), (0, 0, 0)).save(unregistered)
    sample = {
        "images": [{
            "canonical_path": "images/000000_key.png"
        }],
        "poses": [np.eye(4, dtype=np.float32).tolist()],
        "intrinsics": [[[4.0, 0.0, 3.5], [0.0, 4.0, 2.5], [0.0, 0.0, 1.0]]],
    }
    import json

    (scene_root / "sample.json").write_text(json.dumps(sample),
                                            encoding="utf-8")
    output_dir = tmp_path / "priors" / "sparse" / "0"
    assert materialize(scene_root, output_dir) == 1

    with pytest.raises(FileNotFoundError, match="Missing COLMAP camera priors"):
        load_colmap_camera_priors(
            scene_root,
            [str(image_dir / "000000_key.png"),
             str(unregistered)],
            colmap_path=output_dir,
        )


def test_export_camera_pose_priors_satisfies_mvrecon_sparse_requirement(
    tmp_path: Path,) -> None:
    scene_root = tmp_path / "scene"
    image_dir = scene_root / "images"
    image_dir.mkdir(parents=True)
    from PIL import Image

    names = ("000000_key.png", "000001_src001.png")
    image_paths = []
    for name in names:
        path = image_dir / name
        Image.new("RGB", (8, 6), (0, 0, 0)).save(path)
        image_paths.append(str(path))

    output_path = tmp_path / "prediction"
    extrinsics_w2c = np.stack([np.eye(4, dtype=np.float32) for _ in names])
    intrinsics = np.stack([
        np.array([[4.0, 0.0, 3.5], [0.0, 4.0, 2.5], [0.0, 0.0, 1.0]],
                 dtype=np.float32) for _ in names
    ])

    # This is what a depth-only MVRecon batch loop now does: export depths
    # incrementally per batch (unchanged), then export the accumulated,
    # points-free camera poses once at the end.
    export_depth_predictions(output_path, image_paths,
                             np.zeros((2, 6, 8), dtype=np.float32))
    export_camera_pose_priors(
        output_path,
        image_dir,
        image_paths,
        extrinsics_w2c,
        intrinsics,
        width=8,
        height=6,
    )

    sparse_dir = output_path / "sparse" / "0"
    cameras, images, points3D = read_write_model.read_model(str(sparse_dir))
    assert len(cameras) == len(names)
    assert len(images) == len(names)
    assert points3D == {}

    status = validate_prediction_output(
        task="MVRecon",
        dataset="scannetpp",
        method="PI3_X_rgb_cameras",
        scene="scene",
        scene_output_path=output_path,
        requirements=requirements_for_task("mvrecon"),
    )
    assert status.ok
    assert status.missing == []


def test_load_depth_priors_uses_nearest_numeric_timestamp(
        tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    depth_dir = tmp_path / "depths"
    image_dir.mkdir()
    depth_dir.mkdir()
    image_path = image_dir / "1548266470.85761.png"
    image_path.write_bytes(b"not decoded by loader")

    depth = np.full((2, 3), 1234, dtype=np.uint16)
    from PIL import Image

    Image.fromarray(depth).save(depth_dir / "1548266470.88322.png")

    loaded = load_depth_priors(tmp_path, [str(image_path)])

    assert len(loaded) == 1
    np.testing.assert_allclose(loaded[0], depth.astype(np.float32) / 1000.0)


def test_load_depth_priors_accepts_bonn_sampled_timestamp_jitter(
        tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    depth_dir = tmp_path / "depths"
    image_dir.mkdir()
    depth_dir.mkdir()
    image_path = image_dir / "1548339960.10108.png"
    image_path.write_bytes(b"not decoded by loader")

    depth = np.full((2, 3), 2000, dtype=np.uint16)
    from PIL import Image

    Image.fromarray(depth).save(depth_dir / "1548339960.20095.png")

    loaded = load_depth_priors(tmp_path, [str(image_path)])

    assert len(loaded) == 1
    np.testing.assert_allclose(loaded[0], depth.astype(np.float32) / 1000.0)


def test_camera_conditioned_resolver_uses_eth3d_undistorted(
        tmp_path: Path, monkeypatch) -> None:
    scene = tmp_path / "scene"
    distorted = scene / "images" / "dslr_images"
    undistorted = scene / "images" / "dslr_images_undistorted"
    distorted.mkdir(parents=True)
    undistorted.mkdir(parents=True)
    (distorted / "DSC_0001.JPG").touch()
    for name in ("DSC_0001.JPG", "DSC_0002.JPG"):
        (undistorted / name).touch()
    calibration = scene / "dslr_calibration_undistorted"
    calibration.mkdir()
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        (calibration / name).touch()

    camera = read_write_model.Camera(1, "PINHOLE", 10, 10,
                                     np.array([5, 5, 5, 5]))
    records = {
        index:
            read_write_model.Image(
                index,
                np.array([1, 0, 0, 0]),
                np.zeros(3),
                1,
                f"dslr_images_undistorted/DSC_000{index}.JPG",
                np.empty((0, 2)),
                np.empty(0, dtype=int),
            ) for index in (1, 2)
    }
    monkeypatch.setattr(read_write_model, "read_model", lambda _path: ({
        1: camera
    }, records, {}))

    resolved = resolve_camera_conditioned_inputs(scene, colmap_path=calibration)
    assert canonical_image_root(scene) == undistorted
    assert all(
        Path(path).parent == undistorted for path in resolved.image_paths)
    assert resolved.poses.shape == (2, 4, 4)
    assert resolved.intrinsics.shape == (2, 3, 3)


def test_camera_conditioned_resolver_rejects_ambiguous_basename(
        tmp_path: Path, monkeypatch) -> None:
    scene = tmp_path / "scene"
    first = scene / "images" / "a"
    second = scene / "images" / "b"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "frame.jpg").touch()
    (second / "frame.jpg").touch()
    calibration = scene / "sparse" / "0"
    calibration.mkdir(parents=True)
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        (calibration / name).touch()
    camera = read_write_model.Camera(1, "PINHOLE", 10, 10,
                                     np.array([5, 5, 5, 5]))
    record = read_write_model.Image(
        1,
        np.array([1, 0, 0, 0]),
        np.zeros(3),
        1,
        "frame.jpg",
        np.empty((0, 2)),
        np.empty(0, dtype=int),
    )
    monkeypatch.setattr(read_write_model, "read_model", lambda _path: ({
        1: camera
    }, {
        1: record
    }, {}))
    with pytest.raises(FileNotFoundError, match="No COLMAP camera-prior"):
        resolve_camera_conditioned_image_paths(
            scene, [str(first / "frame.jpg"),
                    str(second / "frame.jpg")],
            colmap_path=calibration)
