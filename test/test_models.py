# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Tests of spatialarena.models: the pipeline, registry and plugins."""

import json
import os
import pathlib
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from PIL import Image

from spatialarena import methods
from spatialarena import models
from spatialarena.eval import depth_prediction_evaluator
from spatialarena.geometry import read_write_model
from spatialarena.models import registry
from spatialarena.models import utils

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# A complete external model, as a third-party package would write it. Its
# depth is a fixed function of the pixels on the 28 x 28 padded square, so
# restoring it to the source size can be checked.
TOY_MODEL_SOURCE = textwrap.dedent('''
    import numpy as np

    from spatialarena import methods
    from spatialarena import models
    from spatialarena.models import utils


    @models.register_model(spec=methods.MethodSpec(
        name="TOY_EXTERNAL", adapter="run_model.py",
        code_env="TOY_EXTERNAL_CODE_PATH", code_default="", native=True,
        input_resolution=28))
    class ToyExternal(models.BaseModel):
        name = "TOY_EXTERNAL"
        default_resolution = 28
        spatial_policies = ("shared",)

        def load(self, device):
            return "toy-network"

        def preprocess(self, image_paths, batch_index=None):
            images, transforms = utils.preprocess_images(
                image_paths, method="vggt", size=self.args.resolution,
                mode="pad")
            return models.ModelInput(image_paths, images.numpy(), transforms,
                                     batch_index)

        def forward(self, inputs):
            n, _, h, w = inputs.data.shape
            intrinsics = np.array([[w, 0, w / 2], [0, h, h / 2], [0, 0, 1]],
                                  dtype=np.float32)
            return models.Prediction(
                depth=1.0 + inputs.data.mean(axis=1),
                confidence=np.ones((n, h, w), dtype=np.float32),
                extrinsics_w2c=np.tile(np.eye(4, dtype=np.float32)[:3],
                                       (n, 1, 1)),
                intrinsics=np.tile(intrinsics, (n, 1, 1)),
            )
''')


def _make_scene(tmp_path, sizes=((60, 40), (60, 40), (60, 40))):
    """Writes a scene of random images of the given (width, height)."""
    images_dir = tmp_path / "scene" / "images"
    images_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for index, (width, height) in enumerate(sizes):
        pixels = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(images_dir / f"{index:03d}.png")
    return images_dir.parent


class _StubModel(models.BaseModel):
    """A model that is never run, for registry tests."""

    name = "VGGT"

    def load(self, device):
        return None

    def preprocess(self, image_paths, batch_index=None):
        return models.ModelInput(image_paths)

    def forward(self, inputs):
        return models.Prediction()


@pytest.fixture(name="toy_plugin")
def fixture_toy_plugin(tmp_path, monkeypatch):
    """The toy model as a plugin file, for this process and subprocesses."""
    plugin = tmp_path / "toy_external_model.py"
    plugin.write_text(TOY_MODEL_SOURCE, encoding="utf-8")
    monkeypatch.setenv(registry.PLUGIN_ENV, str(plugin))
    yield plugin
    registry.unregister_model("TOY_EXTERNAL")
    # pylint: disable-next=protected-access
    registry._loaded_plugins.discard(str(plugin))


def test_get_model_resolves_names_and_aliases():
    expected = {"VGGT", "ROBUSTVGGT", "FASTVGGT", "VGGT_OMEGA", "PI3_X", "DA3"}
    assert expected <= set(models.list_models())
    assert models.get_model("vggt").name == "VGGT"
    assert models.get_model("DA3-L") is models.get_model("DA3")
    assert models.get_model("pi3") is models.get_model("PI3_X")
    assert issubclass(models.get_model("RobustVGGT"), models.get_model("VGGT"))
    with pytest.raises(KeyError, match="Unknown model"):
        models.get_model("NOT_A_MODEL")


def test_import_does_not_import_torch():
    code = ("import sys, spatialarena.models.vggt; "
            "print('torch' in sys.modules)")
    result = subprocess.run([sys.executable, "-c", code],
                            cwd=REPO_ROOT,
                            check=True,
                            capture_output=True,
                            text=True)
    assert result.stdout.strip() == "False"


def test_adapter_utils_is_an_alias_of_models_utils(monkeypatch):
    # A script run from scripts/inference imports adapter_utils by name.
    monkeypatch.syspath_prepend(str(REPO_ROOT / "scripts" / "inference"))
    # pylint: disable=import-outside-toplevel,import-error
    import adapter_utils
    from scripts.inference import adapter_utils as package_adapter_utils

    assert adapter_utils is utils
    assert package_adapter_utils is utils


def test_register_model_rejects_a_taken_name_unless_override():
    original = models.get_model("VGGT")
    with pytest.raises(ValueError, match="already registered"):
        models.register_model(_StubModel)
    with pytest.raises(TypeError):
        models.register_model(object)
    try:
        models.register_model(_StubModel, override=True)
        assert models.get_model("VGGT") is _StubModel
    finally:
        models.register_model(original, override=True)


def test_external_model_restores_depth_and_cameras(tmp_path, toy_plugin):
    del toy_plugin  # Only sets SPATIALARENA_MODEL_PLUGINS.
    model_cls = models.get_model("TOY_EXTERNAL")
    assert methods.resolve_method("TOY_EXTERNAL").adapter == "run_model.py"
    widths = (60, 58, 60)
    scene = _make_scene(tmp_path, sizes=[(width, 40) for width in widths])
    output = tmp_path / "out"

    model_cls.main([
        "--data_path",
        str(scene),
        "--output_path",
        str(output),
        "--export_depth",
        "--depth_only",
        "--export_colmap",
    ])

    for index, width in enumerate(widths):
        depth_path = output / "depths" / f"{index:03d}.npz"
        depth = depth_prediction_evaluator.load_depth_map(depth_path)
        assert depth.shape == (40, width)
        assert np.isfinite(depth).all()
    cameras, images, _ = read_write_model.read_model(
        str(output / "sparse" / "0"))
    assert len(images) == 3
    assert {camera.height for camera in cameras.values()} == {40}
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["method"] == "TOY_EXTERNAL"
    assert metrics["depth_only"] is True
    assert metrics["spatial_policy"] == "shared"
    assert metrics["model_input_size"] == [28, 28]


def test_predict_and_restore_depths_without_running(tmp_path, toy_plugin):
    del toy_plugin  # Only sets SPATIALARENA_MODEL_PLUGINS.
    scene = _make_scene(tmp_path)
    model = models.get_model("TOY_EXTERNAL").from_argv(
        ["--data_path",
         str(scene), "--output_path",
         str(tmp_path / "out")])
    image_paths = utils.find_image_paths(scene / "images")

    prediction, inputs = model.predict(image_paths)

    assert np.shape(prediction.depth) == (3, 28, 28)
    restored = model.restore_depths(prediction, inputs)
    assert [depth.shape for depth in restored] == [(40, 60)] * 3


def test_engine_entry_points_find_an_external_plugin(tmp_path, toy_plugin):
    env = dict(os.environ, **{registry.PLUGIN_ENV: str(toy_plugin)})
    adapter = subprocess.run([
        sys.executable, "scripts/test/benchmark_methods.py", "--method",
        "TOY_EXTERNAL", "--field", "adapter"
    ],
                             cwd=REPO_ROOT,
                             env=env,
                             check=True,
                             capture_output=True,
                             text=True)
    assert adapter.stdout.strip() == "run_model.py"

    scene = _make_scene(tmp_path)
    output = tmp_path / "out"
    subprocess.run([
        sys.executable, "scripts/inference/run_model.py", "--model",
        "TOY_EXTERNAL", "--data_path",
        str(scene), "--output_path",
        str(output), "--export_depth", "--depth_only"
    ],
                   cwd=REPO_ROOT,
                   env=env,
                   check=True,
                   capture_output=True,
                   text=True)
    depth_names = sorted(path.name for path in (output / "depths").iterdir())
    assert depth_names == ["000.npz", "001.npz", "002.npz"]


def test_register_method_rejects_an_alias_clash():
    spec = methods.MethodSpec(name="NEW_METHOD",
                              adapter="run_model.py",
                              code_env="",
                              code_default="",
                              aliases=("VGGT",))
    with pytest.raises(ValueError, match="already registered"):
        methods.register_method(spec)


def test_shared_policy_rejects_dense_export():
    model = models.get_model("VGGT").from_argv([
        "--data_path", "scene", "--output_path", "out", "--spatial_policy",
        "shared"
    ])
    with pytest.raises(ValueError, match="depth_only"):
        model.validate_args()
