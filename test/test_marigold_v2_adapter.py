# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Marigold V2 (Log-stage2) adapter, Marigold processing resolution, size-group merging of raw log codes and point-query
coverage, and the ETH3D configuration. Ported from eval_refactor_hsy (45300bb); the log-depth scoring itself is
covered by test_depth_alignment_protocols.py."""

from __future__ import annotations

import importlib
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from spatialarena.eval.depth_prediction_evaluator import load_depth_map
from spatialarena.methods import resolve_method

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(name="adapters")
def fixture_adapters(monkeypatch):
    """The adapters import their siblings by bare name, as the engine runs them."""
    monkeypatch.syspath_prepend(str(ROOT / "scripts" / "inference"))


def test_version_registry_separates_v11_and_v2():
    old = resolve_method("MARIGOLD_V1_1")
    new = resolve_method("MARIGOLD_V2")
    assert old.ckpt_default == "prs-eth/marigold-depth-v1-1"
    assert new.ckpt_default == "huawei-bayerlab/marigold-v2-0/depth/Log-stage2"
    assert new.adapter == "marigold_v2.py"
    assert resolve_method("MARIGOLD-V2") == new
    assert new.default_depth_protocol == "log_huber_irls_gt"
    assert old.default_depth_protocol == "depth_huber_irls_gt"


def test_marigold_v2_command_line_is_the_common_one():
    """The variant is the model name; the spec gives the defaults; --seed
    stays, the old --method selector is gone."""
    # pylint: disable-next=import-outside-toplevel
    from spatialarena.models import registry

    model_cls = registry.get_model("MARIGOLD-V2")
    parser = model_cls.build_parser("MARIGOLD-V2")
    args = parser.parse_args(["--data_path", "d", "--output_path", "o"])
    assert args.method == "MARIGOLD_V2" and args.seed == 42
    assert args.ckpt_path == resolve_method("MARIGOLD_V2").ckpt_default
    assert args.code_path.name == "marigold-v2"
    for removed in ("--method", "--resolution", "--max_points",
                    "--spatial_policy"):
        assert removed not in parser._option_string_actions


def test_export_keeps_raw_signed_log_codes(tmp_path, adapters):
    del adapters
    module = importlib.import_module("marigold_v2")
    image = tmp_path / "rgb.png"
    Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(image)
    raw = np.linspace(-1, 1, 12).reshape(3, 4).astype(np.float32)
    prediction = tmp_path / "prediction.npy"
    np.save(prediction, raw)
    out = tmp_path / "out"
    module.export_log_prediction(out, str(image), prediction)
    np.testing.assert_array_equal(
        load_depth_map(out / "raw_log_depths/rgb.npz"), raw)
    np.testing.assert_allclose(load_depth_map(out / "depths/rgb.npz"),
                               np.exp(raw),
                               rtol=1e-6)
    assert not list(out.rglob("*.npy"))
    np.save(prediction, np.full((3, 4), np.nan))
    with pytest.raises(ValueError):
        module.export_log_prediction(out, str(image), prediction)
    np.save(prediction, np.zeros((4, 3)))
    with pytest.raises(ValueError):
        module.export_log_prediction(out, str(image), prediction)


def test_resize_interpolates_codes_before_exponentiating(tmp_path, adapters):
    del adapters
    import cv2  # pylint: disable=import-outside-toplevel

    module = importlib.import_module("marigold_v2")
    source = tmp_path / "rgb.png"
    Image.new("RGB", (7, 5)).save(source)
    raw = np.linspace(-1, 1, 32 * 16, dtype=np.float32).reshape(16, 32)
    prediction = tmp_path / "raw.npy"
    np.save(prediction, raw)
    module.export_log_prediction(tmp_path / "out",
                                 str(source),
                                 prediction,
                                 processing_size=(32, 16))
    expected = cv2.resize(raw, (7, 5), interpolation=cv2.INTER_LINEAR)
    np.testing.assert_allclose(
        load_depth_map(tmp_path / "out/raw_log_depths/rgb.npz"), expected)
    np.testing.assert_allclose(load_depth_map(tmp_path / "out/depths/rgb.npz"),
                               np.exp(expected),
                               rtol=1e-6)


@pytest.mark.parametrize("fixed_size", [False, True])
def test_run_exports_only_selected_images(tmp_path, monkeypatch, adapters,
                                          fixed_size):
    del adapters
    module = importlib.import_module("marigold_v2")
    source = tmp_path / "source"
    source.mkdir()
    for name in ["chosen", "excluded"]:
        Image.fromarray(np.zeros((3, 4, 3),
                                 dtype=np.uint8)).save(source / f"{name}.png")
    selected = str(source / "chosen.png")
    args = Namespace(code_path=tmp_path,
                     output_path=tmp_path / "out",
                     ckpt_path="checkpoint",
                     method="MARIGOLD_V2",
                     seed=42,
                     track="rgb")
    monkeypatch.setattr(module.single_image, "select_images",
                        lambda a: [selected])
    monkeypatch.setenv("MARIGOLD_V2_PYTHON", "/model-env/python")
    monkeypatch.setenv("MARIGOLD_V2_WIDTH", "32" if fixed_size else "0")
    monkeypatch.setenv("MARIGOLD_V2_HEIGHT", "16" if fixed_size else "0")

    def official_runner(command, *, cwd, check):
        if fixed_size:
            assert command[command.index("--width") + 1] == "32"
            assert command[command.index("--height") + 1] == "16"
        else:
            assert "--width" not in command
        assert command[0] == "/model-env/python"
        assert cwd == tmp_path and check
        image_dir = Path(command[command.index("--image_dir") + 1])
        staged = list(image_dir.iterdir())
        assert len(staged) == 1 and staged[0].resolve() == Path(selected)
        result = Path(command[command.index("--output_dir") +
                              1]) / "images/predictions_npy"
        result.mkdir(parents=True)
        np.save(
            result / "000000.npy",
            np.full((16, 32) if fixed_size else (3, 4), -0.25,
                    dtype=np.float32))

    monkeypatch.setattr(module.subprocess, "run", official_runner)
    module.MarigoldV2(args).run()
    assert [p.name for p in (args.output_path / "raw_log_depths").iterdir()
           ] == ["chosen.npz"]
    restored = load_depth_map(args.output_path / "raw_log_depths/chosen.npz")
    assert restored.shape == (3, 4)
    np.testing.assert_allclose(restored, -0.25)
    metadata = json.loads((args.output_path / "metrics.json").read_text())
    assert metadata["representation"] == "raw_normalized_log_depth"
    assert metadata[
        "runtime_scope"] == "official_subprocess_including_model_load"
    assert metadata["resolution"] == ([32, 16] if fixed_size else
                                      "native_multiple_of_16")


@pytest.mark.parametrize("width,height", [(1008, 0), (0, 672), (1000, 672),
                                          (-16, 672)])
def test_invalid_processing_size_fails_before_inference(tmp_path, monkeypatch,
                                                        adapters, width,
                                                        height):
    del adapters
    module = importlib.import_module("marigold_v2")
    monkeypatch.setattr(
        module.single_image, "select_images",
        lambda a: pytest.fail("selected images before checking the size"))
    monkeypatch.setenv("MARIGOLD_V2_WIDTH", str(width))
    monkeypatch.setenv("MARIGOLD_V2_HEIGHT", str(height))
    with pytest.raises(ValueError, match="positive multiples of 16"):
        module.MarigoldV2(
            Namespace(code_path=tmp_path, output_path=tmp_path / "out")).run()


@pytest.mark.parametrize("value,expected", [
    (None, {}),
    ("0", {
        "processing_resolution": 0
    }),
    ("768", {
        "processing_resolution": 768
    }),
])
def test_marigold_processing_resolution(monkeypatch, adapters, value, expected):
    del adapters
    if value is None:
        monkeypatch.delenv("MARIGOLD_PROCESSING_RES", raising=False)
    else:
        monkeypatch.setenv("MARIGOLD_PROCESSING_RES", value)
    assert importlib.import_module("marigold").processing_options() == expected


@pytest.mark.parametrize("value", ["-1", "bad"])
def test_marigold_invalid_processing_resolution(monkeypatch, adapters, value):
    del adapters
    monkeypatch.setenv("MARIGOLD_PROCESSING_RES", value)
    with pytest.raises(ValueError):
        importlib.import_module("marigold").processing_options()


@pytest.mark.parametrize("missing", [False, True])
def test_group_merge_keeps_raw_log_codes(tmp_path, adapters, missing):
    del adapters
    from single_view_groups import merge  # pylint: disable=import-outside-toplevel

    groups = tmp_path / "groups"
    groups.mkdir()
    for i in range(2):
        (groups / f"group_{i:02d}.txt").write_text(f"image{i}.png\n")
        group = groups / f"group_{i:02d}"
        for folder in ["depths", "raw_log_depths"]:
            (group / folder).mkdir(parents=True)
            if not (missing and i == 1 and folder == "raw_log_depths"):
                value = -0.5 if folder == "raw_log_depths" else 1.0
                np.savez_compressed(group / folder / f"image{i}.npz",
                                    depth=np.full((2, 3),
                                                  value,
                                                  dtype=np.float32))
        (group / "metrics.json").write_text(
            json.dumps({
                "representation": "raw_normalized_log_depth",
                "Preprocessed Images": 1,
            }))
    output = tmp_path / "out"
    if missing:
        with pytest.raises(ValueError, match="Incomplete raw_log_depths"):
            merge(groups, output)
        assert (groups / "group_00/depths/image0.npz").is_file()
        assert not output.exists()
    else:
        merge(groups, output)
        assert len(list((output / "raw_log_depths").glob("*.npz"))) == 2
        np.testing.assert_array_equal(
            load_depth_map(output / "raw_log_depths/image1.npz"), -0.5)


def test_group_merge_concatenates_point_query_coverage(tmp_path, adapters):
    """The point-query adapter's per-image coverage (point_query_coverage.json and metrics' image_query_coverage) is
    one list per scene after the size groups merge, not an unexpected file that fails the merge."""
    del adapters
    from single_view_groups import merge  # pylint: disable=import-outside-toplevel

    groups = tmp_path / "groups"
    groups.mkdir()
    for i in range(2):
        (groups / f"group_{i:02d}.txt").write_text(f"image{i}.png\n")
        group = groups / f"group_{i:02d}"
        (group / "depths").mkdir(parents=True)
        np.savez_compressed(group / "depths" / f"image{i}.npz",
                            depth=np.ones((2, 3), dtype=np.float32))
        coverage = [{
            "image": f"image{i}.png",
            "queries": 256,
            "valid": 250 + i
        }]
        (group / "point_query_coverage.json").write_text(json.dumps(coverage))
        (group / "metrics.json").write_text(
            json.dumps({
                "Preprocessed Images": 1,
                "image_query_coverage": coverage
            }))
    output = tmp_path / "out"
    merge(groups, output)
    merged = json.loads((output / "point_query_coverage.json").read_text())
    assert [entry["valid"] for entry in merged] == [250, 251]
    metrics = json.loads((output / "metrics.json").read_text())
    assert [entry["image"] for entry in metrics["image_query_coverage"]
           ] == ["image0.png", "image1.png"]


@pytest.mark.parametrize("scene,expected", [("eth3d", (1008, 672)),
                                            ("nyuv2", None)])
def test_scene_sizes_apply_to_their_scene_only(monkeypatch, adapters, scene,
                                               expected):
    """MARIGOLD_V2_SCENE_SIZES sets one dataset/scene's processing size; the others keep the native size."""
    del adapters
    module = importlib.import_module("marigold_v2")
    monkeypatch.delenv("MARIGOLD_V2_WIDTH", raising=False)
    monkeypatch.delenv("MARIGOLD_V2_HEIGHT", raising=False)
    monkeypatch.setenv("MARIGOLD_V2_SCENE_SIZES", "marigold/eth3d=1008x672")
    assert module.processing_size(Path("/bench/MonoDepth/marigold") /
                                  scene) == expected
    monkeypatch.setenv("MARIGOLD_V2_SCENE_SIZES", "marigold/eth3d=1000x672")
    with pytest.raises(ValueError, match="MARIGOLD_V2_SCENE_SIZES"):
        module.processing_size(Path("/bench/MonoDepth/marigold") / scene)


def test_full_monodepth_runs_marigold_v2_with_the_eth3d_size(
        tmp_path, monkeypatch):
    """full_monodepth.yaml plans MARIGOLD_V2 on every MonoDepth dataset, and its job scripts carry the ETH3D size,
    one adapter call per size group, and the 96 GB inference memory."""
    # pylint: disable=import-outside-toplevel
    from spatialarena.evaluator import BenchmarkEvaluator

    monkeypatch.setenv("SPATIALARENA_ROOT", str(tmp_path / "root"))
    data = yaml.safe_load(
        (ROOT / "configs/benchmark/full_monodepth.yaml").read_text())
    (entry,) = [
        model for model in data["models"] if model["name"] == "MARIGOLD_V2"
    ]
    assert entry["env"]["MARIGOLD_V2_SCENE_SIZES"] == "marigold/eth3d=1008x672"
    data["models"] = [entry]
    data["datasets"] = [{"task": "MonoDepth", "dataset": "marigold"}]
    for key in ["benchmark_root", "experiment_root"]:
        data[key] = str(tmp_path / key)
    (Path(data["benchmark_root"]) /
     "MonoDepth/marigold/eth3d").mkdir(parents=True)
    (Path(data["benchmark_root"]) /
     "MonoDepth/marigold/metadata.json").write_text(
         json.dumps({
             "dataset_name": "marigold",
             "scenes": [{
                 "scene_name": "eth3d",
                 "num_images": 454
             }]
         }),)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(data))
    bench = BenchmarkEvaluator()
    bench.configure(config)
    generated = bench.generate("test")
    assert len(generated) == 1
    script = generated[0][1].infer_script.read_text()
    for line in ("MARIGOLD_V2_SCENE_SIZES=marigold/eth3d=1008x672",
                 "MAX_IMAGES_PER_RUN=0", "--mem=96G"):
        assert line in script, line
