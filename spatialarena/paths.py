# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Where SpatialArena keeps its data by default.

`SPATIALARENA_ROOT` (default `~/spatialarena`) is the one root every default
path derives from:

    $SPATIALARENA_ROOT/benchmarks/
        Benchmark datasets (<Task>/<dataset>/): the benchmark_root.
    $SPATIALARENA_ROOT/benchmarks/_evaluation/
        Evaluation outputs: reports/, artifacts/, viewer_assets/.
    $SPATIALARENA_ROOT/benchmarks/_model_codes/
        The methods' source checkouts.
    $SPATIALARENA_ROOT/benchmarks/_checkpoints/
        Model checkpoints.

The per-folder variables (`BENCHMARK_ROOT`, `BENCHMARK_EVAL_ROOT`,
`METHOD_CODE_ROOT`, `CHECKPOINT_ROOT`, ...) still override their folder.
Standalone scripts that may run outside this package's environment (the
method wrappers under `scripts/inference/`) read `SPATIALARENA_ROOT` the same
way themselves.
"""

from __future__ import annotations

import os
import pathlib

SPATIALARENA_ROOT = pathlib.Path(
    os.environ.get("SPATIALARENA_ROOT", "~/spatialarena")).expanduser()
BENCHMARKS_ROOT = SPATIALARENA_ROOT / "benchmarks"

ROOT_TOKENS = ("${SPATIALARENA_ROOT}", "$SPATIALARENA_ROOT")


def expand_root(value):
    """Replaces the root tokens in a config value by the resolved root.

    Run configs can so write paths below the root. Other `$VARS` are left
    alone: config values such as command templates are expanded by the job's
    shell.

    Args:
        value: A config mapping, list or string; mappings and lists are
            expanded recursively.

    Returns:
        `value` with `${SPATIALARENA_ROOT}` and `$SPATIALARENA_ROOT` replaced
        by `SPATIALARENA_ROOT`.
    """
    if isinstance(value, str):
        for token in ROOT_TOKENS:
            value = value.replace(token, str(SPATIALARENA_ROOT))
        return value
    if isinstance(value, dict):
        return {key: expand_root(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_root(item) for item in value]
    return value
