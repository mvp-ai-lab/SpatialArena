#!/usr/bin/env python3
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Runs a registered model by name.

    run_model.py --model NAME <model options>
    run_model.py --model NAME --help
    run_model.py --list

NAME is a method name or alias (`spatialarena/methods.py`), a registered
model name, or "package.module:Class". It also selects the variant of a
model that serves several methods (e.g. DA3-L / DA3-G / DA3-NGL), whose
MethodSpec then gives the option defaults.

The evaluation engine runs every method this way, with --engine: it passes
the same option set to every method, and options a model does not have
(e.g. --resolution for a method that resizes on its own) are dropped
instead of rejected.
"""

import argparse
import pathlib
import sys

ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[2])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# pylint: disable-next=wrong-import-position
from spatialarena.models import registry  # noqa: E402

# Options the engine passes to every method: name -> takes a value.
ENGINE_OPTIONS = {
    "--code_path": True,
    "--resolution": True,
    "--image_load_resolution": True,
    "--max_points": True,
    "--conf_thresh": True,
    "--depth_conf_thresh": True,
    "--pixel_limit": True,
    "--point_grid_size": True,
    "--colmap_path": True,
    "--spatial_policy": True,
}


def drop_unknown_engine_options(parser, argv):
    """Removes the engine options `parser` does not have from `argv`.

    Args:
        parser: The model's parser.
        argv: The engine's arguments.

    Returns:
        `argv` without those options (and their values).
    """
    known = set(parser._option_string_actions)
    kept = []
    index = 0
    while index < len(argv):
        option = argv[index].split("=", 1)[0]
        if option in ENGINE_OPTIONS and option not in known:
            takes_value = ENGINE_OPTIONS[option] and "=" not in argv[index]
            index += 2 if takes_value else 1
            continue
        kept.append(argv[index])
        index += 1
    return kept


def main(argv=None, default_model=None):
    """Runs the model named by --model (or `default_model`).

    Args:
        argv: The arguments; None for sys.argv.
        default_model: The model when --model is not given (the per-method
            entry scripts in this directory).
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--engine", action="store_true")
    known, model_argv = parser.parse_known_args(argv)

    registry.load_plugins()
    if known.list or not known.model:
        print(" ".join(registry.list_models()))
        return
    model_cls = registry.get_model(known.model)
    if known.engine:
        model_argv = drop_unknown_engine_options(
            model_cls.build_parser(known.model), model_argv)
    model_cls.main(model_argv, method=known.model)


if __name__ == "__main__":
    main()
