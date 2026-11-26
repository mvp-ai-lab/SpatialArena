# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Runs Marigold (spatialarena.models.marigold); the engine's adapter for it.

--model picks MARIGOLD (v1-0, default) or MARIGOLD_V1_1.
"""

import pathlib
import sys

ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[2])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# pylint: disable=wrong-import-position
from scripts.inference import run_model  # noqa: E402
from spatialarena.models import marigold as model_module  # noqa: E402

# pylint: enable=wrong-import-position

processing_options = model_module.processing_options


def main(argv=None):
    """Runs MARIGOLD, or the --model version, on argv; see run_model.py."""
    run_model.main(argv, default_model="MARIGOLD")


if __name__ == "__main__":
    main()
