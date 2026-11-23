# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Per-method inference adapters (mocked models): VGGT, AMB3R, CUT3R, Co-Me, VGGT-Omega, external commands and the
single-image MonoDepth loop.

come_adapter: Mocked end-to-end test for come.py's --spatial_policy shared restoration.

Co-Me's real inference is CUDA-only on this cluster (its own src.infer deserializes a bf16 checkpoint without
map_location="cpu"), so this cannot be verified by actually running the model on CPU (see
scripts/test/run_resolution_consistency_smoke.py's documented limitation). This test instead mocks
run_come/load_come_archive to isolate and verify just the restoration math: that a depth/confidence/intrinsics triple
produced on Co-Me's internal padded-square canvas gets mapped back to each image's true, non-square source
resolution.

vggt_omega_preprocess: Mocked unit test for vggt_omega.py's preprocess_vggt_omega(): VGGT-Omega's real model forward
pass requires CUDA (the upstream implementation itself raises on CPU), so this isolates just the three-branch
preprocessing dispatch extracted from the former inline run_vggt_omega_model, and checks each branch picks the same
path (and files) the pre-refactor code did.

single_image_depth: The shared single-image MonoDepth loop: source-grid export, raw disparities, and the shape guard.
"""

from __future__ import annotations

import sys
import types
import numpy as np
import pytest
import torch
from PIL import Image
from scripts.inference.vggt import crop_to_original_image_region
from spatialarena.eval.depth_prediction_evaluator import load_depth_map
from pathlib import Path
from scripts.inference.amb3r import materialize_symlinked_images
from spatialarena.eval.prediction_artifacts import load_confidence_map
import argparse
from scripts.inference import external_command_adapter
import json

# ---- vggt_adapter ------------------------------------------------------------------------------------------


def test_crop_to_original_image_region_removes_square_padding():
    depth = np.arange(100, dtype=np.float32).reshape(10, 10)
    # Original 4x10 portrait/landscape-free image was horizontally centered
    # in a 10x10 padded square, so valid pixels occupy x=[3,7).
    original_coord = np.array([3.0, 0.0, 7.0, 10.0, 4.0, 10.0],
                              dtype=np.float32)

    cropped = crop_to_original_image_region(
        depth,
        original_coord=original_coord,
        padded_size=10,
    )

    np.testing.assert_array_equal(cropped, depth[:, 3:7])


class _FakeVggtModel:
    """Minimal stand-in for vggt.models.vggt.VGGT: records the input shape it was given and
    returns deterministic, known-shape pose/depth/points tensors sized off it."""

    def __call__(self, batched_images):
        b, n, _c, h, w = batched_images.shape
        pose_enc = torch.zeros(b, n, 9)
        depth = torch.full((b, n, h, w, 1), 2.5)
        depth_conf = torch.ones(b, n, h, w)
        world_points = torch.zeros(b, n, h, w, 3)
        return {
            "pose_enc": pose_enc,
            "depth": depth,
            "depth_conf": depth_conf,
            "world_points": world_points
        }


def _fake_pose_encoding_to_extri_intri(pose_enc, image_hw):
    """Fake replacement for vggt.utils.pose_enc.pose_encoding_to_extri_intri: a fixed intrinsics
    matrix (fx=fy=model width, cx/cy centered) defined w.r.t. the model's own (padded/resized)
    image_hw grid -- exactly what restore_intrinsics/restore_camera_export must map back to each
    image's true source resolution."""
    b, n = pose_enc.shape[:2]
    h, w = image_hw
    extrinsic = torch.eye(4)[:3].expand(b, n, 3, 4).clone()
    intrinsic = torch.zeros(b, n, 3, 3)
    intrinsic[..., 0, 0] = float(w)
    intrinsic[..., 1, 1] = float(h)
    intrinsic[..., 0, 2] = w / 2.0
    intrinsic[..., 1, 2] = h / 2.0
    intrinsic[..., 2, 2] = 1.0
    return extrinsic, intrinsic


def test_run_restores_intrinsics_to_source_resolution_under_shared_policy_with_colmap(
        tmp_path, monkeypatch):
    """End-to-end: --spatial_policy shared + --export_colmap (previously forbidden; now supported
    since intrinsics are restored) must write COLMAP camera width/height/intrinsics for the image's
    true 300x200 source resolution, not the model's padded 56x56 canvas."""
    import scripts.inference.vggt as vggt_module

    fake_vggt = types.ModuleType("vggt")
    fake_vggt.__path__ = [
    ]  # looks like a package, so add_vggt_code_path won't evict it
    fake_vggt_utils = types.ModuleType("vggt.utils")
    fake_vggt_utils_pose_enc = types.ModuleType("vggt.utils.pose_enc")
    fake_vggt_utils_pose_enc.pose_encoding_to_extri_intri = _fake_pose_encoding_to_extri_intri
    monkeypatch.setitem(sys.modules, "vggt", fake_vggt)
    monkeypatch.setitem(sys.modules, "vggt.utils", fake_vggt_utils)
    monkeypatch.setitem(sys.modules, "vggt.utils.pose_enc",
                        fake_vggt_utils_pose_enc)
    monkeypatch.setattr(vggt_module, "load_model",
                        lambda args, device: _FakeVggtModel())

    recorded_colmap_calls = []

    # pylint: disable-next=unused-argument
    def _fake_write_colmap_formats(seq_name, extrinsics, intrinsics,
                                   image_paths, points, colors, **kwargs):
        recorded_colmap_calls.append({"intrinsics": intrinsics, **kwargs})

    monkeypatch.setattr("spatialarena.geometry.utils.write_colmap_formats",
                        _fake_write_colmap_formats)

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    source_w, source_h = 300, 200
    Image.fromarray(np.zeros((source_h, source_w, 3),
                             dtype=np.uint8)).save(images_dir / "a.png")

    output_path = tmp_path / "out"
    argv = [
        "vggt.py",
        "--data_path",
        str(tmp_path),
        "--output_path",
        str(output_path),
        "--resolution",
        "56",
        "--export_depth",
        "--depth_only",
        "--export_colmap",
        "--spatial_policy",
        "shared",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    vggt_module.run()

    assert len(recorded_colmap_calls) == 1
    call = recorded_colmap_calls[0]
    # The whole point: COLMAP geometry must describe the true source image, not the model's canvas.
    assert (call["width"], call["height"]) == (source_w, source_h)
    restored = np.asarray(call["intrinsics"])[0]
    from scripts.inference.adapter_utils import plan_spatial_transform, restore_intrinsics

    transform = plan_spatial_transform((source_w, source_h),
                                       method="vggt",
                                       size=56,
                                       mode="pad")
    model_wh = transform.model_wh
    expected = restore_intrinsics(
        np.array([[model_wh[0], 0, model_wh[0] / 2.0],
                  [0, model_wh[1], model_wh[1] / 2.0], [0, 0, 1.0]]),
        transform,
    )
    np.testing.assert_allclose(restored, expected, atol=1e-4)
    # And it must actually differ from the un-restored, model-grid-resolution intrinsics -- otherwise
    # this test would pass even if restore_camera_export were silently skipped.
    assert not np.allclose(restored[0, 0], model_wh[0])

    depth_files = sorted((output_path / "depths").glob("*.npz"))
    assert len(depth_files) == 1
    depth = load_depth_map(depth_files[0])
    assert depth.shape == (source_h, source_w)


# ---- amb3r_adapter -----------------------------------------------------------------------------------------


def test_materialize_symlinked_images_preserves_order_and_extension(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    # Deliberately non-natural-sort filenames -- materialize_symlinked_images
    # must preserve the *given* image_paths order, not re-derive it from
    # AMB3R's Demo dataset's own natural_sort_key over the dest directory.
    paths = []
    for name in ("frame_b.png", "frame_a.jpg", "frame_c.jpeg"):
        path = source_dir / name
        path.write_bytes(b"fake-image-bytes")
        paths.append(str(path))

    dest_dir = tmp_path / "dest"
    materialize_symlinked_images(paths, dest_dir)

    entries = sorted(dest_dir.iterdir())
    assert [entry.name for entry in entries] == [
        "000000.png",
        "000001.jpg",
        "000002.jpeg",
    ]
    for entry, source_path in zip(entries, paths):
        assert entry.is_symlink()
        assert entry.resolve() == Path(source_path).resolve()


def test_materialize_symlinked_images_overwrites_stale_symlinks(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    first = source_dir / "first.jpg"
    first.write_bytes(b"first")
    second = source_dir / "second.jpg"
    second.write_bytes(b"second")

    dest_dir = tmp_path / "dest"
    materialize_symlinked_images([str(first)], dest_dir)
    materialize_symlinked_images([str(second)], dest_dir)

    target = dest_dir / "000000.jpg"
    assert target.resolve() == second.resolve()


def test_depth_only_run_exports_every_frame_camera_at_source_resolution(
        tmp_path, monkeypatch):
    """--depth_only --export_colmap (what the engine passes for MVRecon / VideoDepth): every frame's camera is exported
    with intrinsics restored to the 300x200 source images -- MVRecon cannot fuse without them."""

    import scripts.inference.amb3r as amb3r_module
    from scripts.inference.adapter_utils import SpatialTransform

    model_w, model_h = amb3r_module.AMB3R_WIDTH, amb3r_module.AMB3R_HEIGHT

    def fake_run_amb3r(_model,
                       image_paths,
                       _args,
                       _device,
                       run_name="",
                       keep_intermediate=False):
        del run_name, keep_intermediate
        n = len(image_paths)
        extrinsic = np.tile(np.eye(4)[:3], (n, 1, 1)).astype(np.float32)
        intrinsic = np.tile(
            np.array([[model_w, 0, model_w / 2], [0, model_h, model_h / 2],
                      [0, 0, 1]],
                     dtype=np.float32), (n, 1, 1))
        depth = np.full((n, model_h, model_w), 2.0, dtype=np.float32)
        conf = np.ones((n, model_h, model_w), dtype=np.float32)
        transforms = [
            SpatialTransform(source_wh=(300, 200),
                             resized_wh=(model_w, model_h)) for _ in image_paths
        ]
        return extrinsic, intrinsic, None, depth, None, conf, None, transforms

    recorded = []
    monkeypatch.setattr(amb3r_module, "load_model",
                        lambda ckpt_path, device: object())
    monkeypatch.setattr(amb3r_module, "add_code_path",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(amb3r_module, "run_amb3r", fake_run_amb3r)
    monkeypatch.setattr(
        "spatialarena.geometry.utils.write_colmap_formats",
        lambda seq_name, extrinsics, intrinsics, image_paths, points, colors, **
        kwargs: recorded.append({
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "image_paths": image_paths,
            **kwargs
        }))

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for name in ("a", "b", "c"):
        Image.fromarray(np.zeros(
            (200, 300, 3), dtype=np.uint8)).save(images_dir / f"{name}.png")
    monkeypatch.setattr(sys, "argv", [
        "amb3r.py",
        "--data_path",
        str(tmp_path),
        "--output_path",
        str(tmp_path / "out"),
        "--export_depth",
        "--depth_only",
        "--export_colmap",
    ])

    amb3r_module.run()

    assert len(recorded) == 1
    call = recorded[0]
    assert len(call["image_paths"]) == 3 and np.asarray(
        call["intrinsics"]).shape == (3, 3, 3)
    assert (call["width"], call["height"]) == (300, 200)
    np.testing.assert_allclose(np.asarray(call["intrinsics"])[0][:2, 2],
                               [150.0, 100.0],
                               atol=1e-3)
    assert len(list((tmp_path / "out" / "depths").glob("*.npz"))) == 3


# ---- cut3r_adapter -----------------------------------------------------------------------------------------

# cut3r.py uses the bare `from adapter_utils import ...` style (its own directory must be on
# sys.path), unlike vggt.py/da3.py's fully-qualified imports.
_INFERENCE_DIR = str(
    Path(__file__).resolve().parents[1] / "scripts" / "inference")
if _INFERENCE_DIR not in sys.path:
    sys.path.insert(0, _INFERENCE_DIR)

from scripts.inference.adapter_utils import restore_prediction  # noqa: E402
from scripts.inference.cut3r import build_shared_views  # noqa: E402


class _Args:
    resolution = 512


def test_build_shared_views_preserves_all_source_content(tmp_path):
    """--spatial_policy shared must never discard real image content: CUT3R's own loader
    (demo.prepare_input -> dust3r.utils.image.load_images) center-crops to its 16px grid and
    additionally forces a lossy 4:3 crop on square inputs -- build_shared_views bypasses that
    loader entirely, so every source pixel must round-trip back through restore_prediction as
    valid (not silently cropped away)."""
    img_path = tmp_path / "a.png"
    source_h, source_w = 200, 300  # non-square 4:3-ish source
    Image.fromarray((np.random.rand(source_h, source_w, 3) * 255).astype(
        np.uint8)).save(img_path)

    views, transforms = build_shared_views([str(img_path)], _Args(), "cpu")

    assert len(views) == 1 and len(transforms) == 1
    view = views[0]
    img = view["img"]
    height, width = img.shape[-2:]
    # CUT3R's patch grid is 16px; a canvas that doesn't divide evenly would be rejected downstream.
    assert height % 16 == 0 and width % 16 == 0
    assert tuple(view["ray_map"].shape) == (1, 6, height, width)
    assert view["true_shape"].numpy().tolist() == [[height, width]]
    assert view["idx"] == 0 and view["instance"] == "0"
    # dust3r's ImgNorm maps [0,1] -> [-1,1]; preprocess_images already returns [0,1].
    assert img.min().item() >= -1.0001 and img.max().item() <= 1.0001

    transform = transforms[0]
    # Mark the known-valid (non-padding) region of a synthetic model-grid prediction, restore it,
    # and confirm the *entire* source image comes back valid -- not just shape-correct.
    fake_depth = np.zeros((height, width), dtype=np.float32)
    left, top, right, bottom = transform.padding
    fake_depth[top:height - bottom, left:width - right] = 7.0
    restored = restore_prediction(fake_depth, transform)
    assert restored.shape == (source_h, source_w)
    valid = ~np.isnan(restored)
    assert valid.mean() == pytest.approx(1.0)
    np.testing.assert_allclose(restored[valid], 7.0)


def test_build_shared_views_on_square_source_avoids_cut3rs_lossy_4x3_crop(
        tmp_path):
    """A square source image is exactly the case CUT3R's own loader mishandles (it forces a 4:3
    crop instead of a clean 16px-grid crop) -- build_shared_views must still preserve it fully,
    since it never calls that loader at all."""
    img_path = tmp_path / "square.png"
    size = 250
    Image.fromarray(
        (np.random.rand(size, size, 3) * 255).astype(np.uint8)).save(img_path)

    views, transforms = build_shared_views([str(img_path)], _Args(), "cpu")
    transform = transforms[0]
    height, width = views[0]["img"].shape[-2:]
    fake_depth = np.zeros((height, width), dtype=np.float32)
    left, top, right, bottom = transform.padding
    fake_depth[top:height - bottom, left:width - right] = 3.0
    restored = restore_prediction(fake_depth, transform)
    assert restored.shape == (size, size)
    assert (~np.isnan(restored)).mean() == pytest.approx(1.0)


# ---- come_adapter ------------------------------------------------------------------------------------------

sys.path.insert(0, "scripts/inference")


def test_depth_only_shared_policy_restores_depth_confidence_and_intrinsics(
        tmp_path, monkeypatch):
    import scripts.inference.come as come_module

    source_w, source_h = 300, 200
    images_dir = tmp_path / "data" / "images"
    images_dir.mkdir(parents=True)
    image_paths = []
    for index in range(2):
        path = images_dir / f"{index:06d}.png"
        Image.new("RGB", (source_w, source_h),
                  color=(index, index, index)).save(path)
        image_paths.append(str(path))

    internal_resolution = 504  # patch_grid_compatible_resolution(504, patch_size=14, grid_multiple=3)
    n = len(image_paths)
    fake_raw = {
        "scene_depths":
            np.full((1, n, internal_resolution, internal_resolution),
                    2.5,
                    dtype=np.float32),
        "scene_depths_conf":
            np.ones((1, n, internal_resolution, internal_resolution),
                    dtype=np.float32),
        "scene_intrinsics":
            np.tile(np.eye(3, dtype=np.float32)[None], (n, 1, 1)),
        "scene_poses":
            np.tile(
                np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                (n, 1)),
    }
    fake_raw["scene_intrinsics"][:, 0, 0] = internal_resolution
    fake_raw["scene_intrinsics"][:, 1, 1] = internal_resolution
    fake_raw["scene_intrinsics"][:, 0, 2] = internal_resolution / 2.0
    fake_raw["scene_intrinsics"][:, 1, 2] = internal_resolution / 2.0

    monkeypatch.setattr(
        come_module,
        "run_come",
        lambda args, paths, run_name="": tmp_path / "fake_raw.npz")
    monkeypatch.setattr(come_module, "load_come_archive", lambda path: fake_raw)
    monkeypatch.setattr(come_module, "come_intermediate_paths",
                        lambda args, raw_path: [])

    captured = {}
    real_export_camera_pose_priors = come_module.export_camera_pose_priors

    def _spy_export_camera_pose_priors(*args, **kwargs):
        captured["width"] = kwargs["width"]
        captured["height"] = kwargs["height"]
        captured["intrinsics"] = args[4]
        return real_export_camera_pose_priors(*args, **kwargs)

    monkeypatch.setattr(come_module, "export_camera_pose_priors",
                        _spy_export_camera_pose_priors)

    output_path = tmp_path / "out"
    argv = [
        "come.py",
        "--data_path",
        str(tmp_path / "data"),
        "--output_path",
        str(output_path),
        "--export_depth",
        "--export_colmap",
        "--depth_only",
        "--spatial_policy",
        "shared",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    come_module.run()

    for index in range(n):
        depth = load_depth_map(output_path / "depths" / f"{index:06d}.npz")
        assert depth.shape == (source_h, source_w)
        conf = load_confidence_map(output_path / "confidences" /
                                   f"{index:06d}.npz")
        assert conf.shape == (source_h, source_w)

    assert (captured["width"], captured["height"]) == (source_w, source_h)
    fx = captured["intrinsics"][0, 0, 0]
    # Restored intrinsics must describe the true source resolution, not the 504x504 model canvas.
    assert not np.isclose(fx, internal_resolution)
    assert fx < source_w * 2  # sane order of magnitude for a 300x200 image, not a stale 504-scale value


# ---- vggt_omega_preprocess ---------------------------------------------------------------------------------

sys.path.insert(0, "scripts/inference")

from scripts.inference.vggt_omega import preprocess_vggt_omega  # noqa: E402


def _make_images(tmp_path: Path, n: int, size=(300, 200)) -> list[str]:
    paths = []
    for i in range(n):
        p = tmp_path / f"{i:06d}.png"
        Image.new("RGB", size, color=(i, i, i)).save(p)
        paths.append(str(p))
    return paths


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        spatial_policy="legacy",
        image_load_resolution=512,
        patch_size=16,
        preprocess_mode="balanced",
        output_path=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_shared_pad_builds_tensor_directly_no_disk_write(tmp_path):
    paths = _make_images(tmp_path, 2)
    args = _args(spatial_policy="shared-pad", output_path=tmp_path / "out")
    model_input, transforms = preprocess_vggt_omega(paths,
                                                    args=args,
                                                    device="cpu")
    assert model_input["resized_dir"] is None
    assert transforms is not None and len(transforms) == 2
    assert model_input["images"].shape[0] == 2


def test_legacy_non_fixed_prewrites_resized_files_then_calls_vendored_loader(
        tmp_path):
    paths = _make_images(tmp_path, 2)
    args = _args(spatial_policy="legacy", output_path=tmp_path / "out")
    captured = {}

    def fake_loader(paths_, *, mode, image_resolution, patch_size):  # pylint: disable=unused-argument
        captured["paths"] = paths_
        captured["image_resolution"] = image_resolution
        return torch.zeros(len(paths_), 3, image_resolution, image_resolution)

    model_input, transforms = preprocess_vggt_omega(
        paths,
        args=args,
        device="cpu",
        load_and_preprocess_images=fake_loader,
        internal_resolution=512)
    assert transforms is None
    assert model_input["resized_dir"] is not None
    assert model_input["resized_dir"].is_dir()
    # The vendored loader must have been called on the pre-resized files, not the originals.
    assert all(
        Path(p).parent == model_input["resized_dir"] for p in captured["paths"])
    assert captured["image_resolution"] == 512


def test_legacy_fixed_input_size_skips_resize_uses_patch_grid_tensor(
        tmp_path, monkeypatch):
    """SLAM protocol pre-sized frames (FIXED_INPUT_RESOLUTION=1): no disk pre-resize, no
    load_and_preprocess_images call -- only load_patch_grid_tensor's 16px-grid crop."""
    import scripts.inference.vggt_omega as vggt_omega_module

    monkeypatch.setenv("FIXED_INPUT_RESOLUTION", "1")
    paths = _make_images(tmp_path, 2, size=(512, 512))
    args = _args(spatial_policy="legacy", output_path=tmp_path / "out")

    captured = {}

    def fake_patch_grid_tensor(paths_, patch_size):  # pylint: disable=unused-argument
        captured["paths"] = paths_
        return torch.zeros(len(paths_), 3, 512, 512)

    def fail_loader(*_a, **_k):
        raise AssertionError(
            "the vendored loader must not be called for fixed-resolution input")

    monkeypatch.setattr(vggt_omega_module, "load_patch_grid_tensor",
                        fake_patch_grid_tensor)
    model_input, transforms = preprocess_vggt_omega(
        paths,
        args=args,
        device="cpu",
        load_and_preprocess_images=fail_loader,
        internal_resolution=512)
    assert transforms is None
    assert model_input["resized_dir"] is None
    assert captured["paths"] == paths


def test_legacy_pad_skips_prewrite_disabled_by_run_but_dispatch_still_prewrites(
        tmp_path):
    """legacy-pad always pre-writes resized files too (same as plain legacy's non-fixed branch);
    run() itself rejects legacy-pad+depth_only before preprocess_vggt_omega is ever called with it."""
    paths = _make_images(tmp_path, 2)
    args = _args(spatial_policy="legacy-pad", output_path=tmp_path / "out")
    captured = {}

    def fake_loader(paths_, *, mode, image_resolution, patch_size):  # pylint: disable=unused-argument
        captured["paths"] = paths_
        return torch.zeros(len(paths_), 3, image_resolution, image_resolution)

    model_input, transforms = preprocess_vggt_omega(
        paths,
        args=args,
        device="cpu",
        load_and_preprocess_images=fake_loader,
        internal_resolution=512)
    assert transforms is None
    assert model_input["resized_dir"] is not None
    assert all(
        Path(p).parent == model_input["resized_dir"] for p in captured["paths"])


# ---- external_command_adapter ------------------------------------------------------------------------------


def _scene(tmp_path, names):
    scene = tmp_path / "scene"
    (scene / "images").mkdir(parents=True)
    (scene / "depths").mkdir()
    for name in names:
        (scene / "images" / name).write_bytes(b"x")
    return scene


def _run(monkeypatch, scene, output, subset_file=None):
    command = (
        "ls {images_dir} > {output_path}/seen.txt && ls {depth_dir} > /dev/null"
        " && echo {interval} > {output_path}/interval.txt")
    monkeypatch.setattr(sys, "argv", [
        "adapter", "--data_path",
        str(scene), "--output_path",
        str(output), "--method_name", "CUSTOM", "--command", command,
        "--interval", "2"
    ])
    if subset_file is None:
        monkeypatch.delenv("BENCHMARK_IMAGE_SUBSET", raising=False)
    else:
        monkeypatch.setenv("BENCHMARK_IMAGE_SUBSET", str(subset_file))
    external_command_adapter.main()
    return (output / "seen.txt").read_text().split(), (
        output / "interval.txt").read_text().strip()


def test_external_command_sees_only_the_engines_image_subset(
        tmp_path, monkeypatch):
    names = [f"{i:03d}.png" for i in range(8)]
    scene = _scene(tmp_path, names)
    # The engine lists a subset of the interval-selected frames (every 2nd: 000, 002, 004, 006).
    subset = tmp_path / "subset.txt"
    subset.write_text("\n".join(
        str(scene / "images" / name) for name in ("000.png", "004.png")) + "\n")
    seen, interval = _run(monkeypatch, scene, tmp_path / "out_subset", subset)
    assert seen == ["000.png", "004.png"
                   ] and interval == "1"  # the subset is already the selection

    seen, interval = _run(monkeypatch, scene, tmp_path / "out_all")
    assert seen == names and interval == "2"  # without a subset, the command reads the whole scene as before


# ---- single_image_depth ------------------------------------------------------------------------------------

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts" / "inference"))
import single_image_depth  # noqa: E402  # pylint: disable=wrong-import-position


def _single_image_scene(tmp_path: Path) -> Path:
    images = tmp_path / "scene" / "images"
    images.mkdir(parents=True)
    Image.new("RGB", (8, 6)).save(images / "a.png")
    Image.new("RGB", (5, 7)).save(images / "b.png")
    return tmp_path / "scene"


def _single_image_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    values = dict(method="TEST",
                  data_path=_single_image_scene(tmp_path),
                  output_path=tmp_path / "out",
                  ckpt_path="ckpt",
                  track="rgb",
                  interval=1,
                  seed=0,
                  export_depth=True,
                  export_colmap=False,
                  dry_run=False)
    values.update(overrides)
    return argparse.Namespace(**values)


def test_exports_each_image_on_its_own_grid_with_raw_disparity(tmp_path):
    args = _single_image_args(tmp_path)
    paths = single_image_depth.select_images(args)
    single_image_depth.run_single_image_depth(
        args, paths, lambda rgb:
        (np.full(rgb.shape[:2], 2.0), np.full(rgb.shape[:2], 0.5)))

    assert load_depth_map(args.output_path / "depths" / "a.npz").shape == (6, 8)
    assert load_depth_map(args.output_path / "depths" / "b.npz").shape == (7, 5)
    assert np.allclose(
        load_depth_map(args.output_path / "raw_disparities" / "b.npz"), 0.5)
    metrics = json.loads(
        (args.output_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["Preprocessed Images"] == 2 and metrics[
        "nonfinite_depth_pixels"] == 0


def test_a_prediction_off_the_source_grid_fails_instead_of_being_resized(
        tmp_path):
    args = _single_image_args(tmp_path)
    paths = single_image_depth.select_images(args)
    with pytest.raises(ValueError, match="not the source image's"):
        single_image_depth.run_single_image_depth(
            args, paths, lambda rgb: (np.zeros(rgb.shape[1::-1]), None))


def test_dense_depth_only(tmp_path):
    with pytest.raises(ValueError, match="dense depth only"):
        single_image_depth.select_images(
            _single_image_args(tmp_path, export_colmap=True))


def test_da3_load_prior_model_reads_the_camera_prior_model(tmp_path):
    """--load_from_colmap's dense-points branch reads the cameras, images and points of the same COLMAP prior model
    the camera priors come from (it used an undefined ``manager`` before)."""
    # pylint: disable=import-outside-toplevel
    from types import SimpleNamespace as Namespace

    from scripts.inference.da3 import load_prior_model
    from spatialarena.geometry.read_write_model import BaseImage, Camera, Point3D, write_model

    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    cameras = {
        1:
            Camera(id=1,
                   model="PINHOLE",
                   width=4,
                   height=3,
                   params=np.array([2.0, 2.0, 2.0, 1.5]))
    }
    images = {
        1:
            BaseImage(id=1,
                      qvec=np.array([1.0, 0, 0, 0]),
                      tvec=np.zeros(3),
                      camera_id=1,
                      name="a.png",
                      xys=np.zeros((0, 2)),
                      point3D_ids=np.zeros(0, dtype=int))
    }
    points = {
        7:
            Point3D(id=7,
                    xyz=np.array([0.0, 0.0, 1.0]),
                    rgb=np.array([1, 2, 3]),
                    error=0.0,
                    image_ids=np.zeros(0, dtype=int),
                    point2D_idxs=np.zeros(0, dtype=int))
    }
    write_model(cameras, images, points, str(sparse), ext=".txt")
    model = load_prior_model(
        Namespace(colmap_path=str(sparse),
                  data_path=tmp_path,
                  colmap_model_subdir="0"))
    assert set(model.cameras) == {1} and set(model.images) == {1}
    np.testing.assert_allclose(model.points3D[7].xyz, [0.0, 0.0, 1.0])
