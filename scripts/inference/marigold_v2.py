# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Runs spatialarena.models.marigold_v2.MarigoldV2; the engine's adapter.

Imported as `marigold_v2` (a script run from this directory), this module is
an alias of spatialarena.models.marigold_v2, so its functions
(`export_log_prediction`, `processing_size`, ...) and the names they use
(`subprocess`, `single_image`) are the model module's own.
"""

import pathlib
import sys

ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[2])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# pylint: disable=wrong-import-position
from scripts.inference import run_model  # noqa: E402
from spatialarena.models import marigold_v2 as model_module  # noqa: E402

# pylint: enable=wrong-import-position


def main(argv=None):
    """Runs MARIGOLD_V2 on argv (default: sys.argv); see run_model.py."""
    run_model.main(argv, default_model="MARIGOLD_V2")


if __name__ == "__main__":
    main()
else:
    sys.modules[__name__] = model_module
