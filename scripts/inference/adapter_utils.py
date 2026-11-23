# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Alias of spatialarena.models.utils for scripts that import adapter_utils.

`import adapter_utils` (a script run from this directory) and
`scripts.inference.adapter_utils` both resolve to the one
spatialarena.models.utils module object, so its per-process state (the
model-load timer, the recorded input size, the preprocessor registry) is
shared however it was imported.
"""

import pathlib
import sys

_ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[2])
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

# The wildcard import only shows the names to static tools (pylint, IDEs);
# the alias below makes the shared module the one used at run time.
# pylint: disable=wrong-import-position,wildcard-import,unused-wildcard-import
from spatialarena.models.utils import *  # noqa: E402,F401,F403
from spatialarena.models import utils as _utils  # noqa: E402

sys.modules[__name__] = _utils
