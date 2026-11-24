#!/usr/bin/env python3
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Runs a point-query VLM: --model VLM3|DEPTHLM|QWEN3_VL.

The same as `run_model.py --model NAME` for those models; see
`--model NAME --help` for the options. Imported, this module is an alias of
spatialarena.models.point_query, so its helpers (and anything replaced on
it, as the tests do) are the ones the models use.
"""

import pathlib
import sys

ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[2])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# pylint: disable=wrong-import-position
from scripts.inference import run_model  # noqa: E402
from spatialarena.models import point_query as model_module  # noqa: E402

# pylint: enable=wrong-import-position


def main(argv=None):
    """Runs the point-query VLM --model names; see run_model.py."""
    # Rejects a missing --model or one of another family.
    model_module.split_model_option(argv, model_module.MODELS,
                                    model_module.DESCRIPTION)
    run_model.main(argv)


if __name__ == "__main__":
    main()
else:
    sys.modules[__name__] = model_module
