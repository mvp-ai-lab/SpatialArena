# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Command lines of MapAnything and the DUSt3R / MASt3R / Pow3R wrappers."""

import pathlib

import pytest

from spatialarena.models import mapanything
from spatialarena.models import mapanything_wrappers

REQUIRED = ["--data_path", "scene", "--output_path", "out"]
WRAPPERS = (
    mapanything_wrappers.DUSt3R,
    mapanything_wrappers.MASt3R,
    mapanything_wrappers.Pow3R,
)


def _options(model_cls):
    return set(model_cls.build_parser()._option_string_actions)


def test_mapanything_has_no_resolution_or_old_code_path():
    options = _options(mapanything.MapAnything)
    assert "--image_load_resolution" in options
    assert "--colmap_path" in options
    assert "--resolution" not in options
    assert "--mapanything_code_path" not in options


def test_mapanything_defaults_come_from_the_method_spec(monkeypatch):
    monkeypatch.setenv("METHOD_CODE_ROOT", "/codes")
    monkeypatch.delenv("MAPANYTHING_CODE_PATH", raising=False)
    monkeypatch.delenv("MAPANYTHING_CKPT_PATH", raising=False)
    args = mapanything.MapAnything.build_parser().parse_args(REQUIRED)
    assert args.image_load_resolution == 518
    assert args.ckpt_path == "facebook/map-anything"
    assert args.code_path == pathlib.Path("/codes/map-anything")
    assert args.method == "MAPANYTHING"


@pytest.mark.parametrize("model_cls", WRAPPERS)
def test_wrappers_have_one_code_path_option(model_cls):
    options = _options(model_cls)
    assert "--code_path" in options
    assert "--mapanything_code_path" in options
    for removed in ("--method_code_path", "--resolution",
                    "--image_load_resolution", "--colmap_path"):
        assert removed not in options


@pytest.mark.parametrize("model_cls", WRAPPERS)
def test_wrapper_code_paths_default_to_the_engine_env(model_cls, monkeypatch):
    monkeypatch.setenv(f"{model_cls.name}_CODE_PATH", "/c/method")
    monkeypatch.setenv("MAPANYTHING_CODE_PATH", "/c/map-anything")
    args = model_cls.build_parser().parse_args(REQUIRED)
    assert args.code_path == pathlib.Path("/c/method")
    assert args.mapanything_code_path == pathlib.Path("/c/map-anything")


def test_wrapper_mapanything_code_path_falls_back_to_the_spec(monkeypatch):
    monkeypatch.setenv("METHOD_CODE_ROOT", "/codes")
    monkeypatch.delenv("MAPANYTHING_CODE_PATH", raising=False)
    args = mapanything_wrappers.DUSt3R.build_parser().parse_args(REQUIRED)
    assert args.mapanything_code_path == pathlib.Path("/codes/map-anything")
