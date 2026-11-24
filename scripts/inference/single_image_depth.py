# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Alias of spatialarena.models.single_image for scripts that import it.

`import single_image_depth` (a script run from this directory) resolves to
the spatialarena.models.single_image module object: the shared single-image
MonoDepth loop (`select_images`, `run_single_image_depth`,
`resolve_checkpoint_file`, `load_rgb`) and the `SingleImageDepthModel`
family base class.
"""

import pathlib
import sys

_ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[2])
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

# The wildcard import only shows the names to static tools (pylint, IDEs);
# the alias below makes the shared module the one used at run time.
# pylint: disable=wrong-import-position,wildcard-import,unused-wildcard-import
from spatialarena.models.single_image import *  # noqa: E402,F401,F403
from spatialarena.models import single_image as _single_image  # noqa: E402

sys.modules[__name__] = _single_image
