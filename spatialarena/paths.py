# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Where SpatialArena keeps its data by default.

``SPATIALARENA_ROOT`` (default ``~/spatialarena``) is the one root every default path derives from:

    $SPATIALARENA_ROOT/benchmarks/                 benchmark datasets (<Task>/<dataset>/): the benchmark_root
    $SPATIALARENA_ROOT/benchmarks/_evaluation/     evaluation outputs: reports/, artifacts/, viewer_assets/
    $SPATIALARENA_ROOT/benchmarks/_model_codes/    the methods' source checkouts
    $SPATIALARENA_ROOT/benchmarks/_checkpoints/    model checkpoints

The per-folder variables (``BENCHMARK_ROOT``, ``BENCHMARK_EVAL_ROOT``, ``METHOD_CODE_ROOT``, ``CHECKPOINT_ROOT``, ...)
still override their folder. Standalone scripts that may run outside this package's environment (the method
wrappers under ``scripts/inference/``) read ``SPATIALARENA_ROOT`` the same way themselves.
"""

from __future__ import annotations

import os
from pathlib import Path

SPATIALARENA_ROOT = Path(os.environ.get("SPATIALARENA_ROOT", "~/spatialarena")).expanduser()
BENCHMARKS_ROOT = SPATIALARENA_ROOT / "benchmarks"

ROOT_TOKENS = ("${SPATIALARENA_ROOT}", "$SPATIALARENA_ROOT")


def expand_root(value):
    """``value`` (a config mapping, list or string) with ``${SPATIALARENA_ROOT}`` / ``$SPATIALARENA_ROOT`` replaced by
    the resolved root, so run configs can write paths below it. Other ``$VARS`` are left alone: config values such as
    command templates are expanded by the job's shell."""
    if isinstance(value, str):
        for token in ROOT_TOKENS:
            value = value.replace(token, str(SPATIALARENA_ROOT))
        return value
    if isinstance(value, dict):
        return {key: expand_root(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_root(item) for item in value]
    return value
