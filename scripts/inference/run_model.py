#!/usr/bin/env python3
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Runs a registered model by name.

    run_model.py --model NAME <model options>
    run_model.py --list

The engine runs a method this way when its MethodSpec has
adapter="run_model.py", so an external model registered with
`spatialarena.models.register_model(spec=...)` needs no script of its own.
NAME may also be "package.module:Class".
"""

import argparse
import pathlib
import sys

ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[2])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# pylint: disable-next=wrong-import-position
from spatialarena.models import registry  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model")
    parser.add_argument("--list", action="store_true")
    known, model_argv = parser.parse_known_args(argv)

    registry.load_plugins()
    if known.list or not known.model:
        print(" ".join(registry.list_models()))
        return
    registry.get_model(known.model).main(model_argv)


if __name__ == "__main__":
    main()
