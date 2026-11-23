# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""SpatialArena, the benchmark's own captures.

The captures come from a handheld 3-camera LiDAR rig
(docs/benchmark/spatialarena.md). In a benchmark root they sit where every
dataset does, `<benchmark_root>/<Task>/spatialarena/<scene>` (the layout of
the Hugging Face release). `scripts/preprocess/materialize_spatialarena.py`
writes them with the tracks holding the scenes directly
(`<root>/<Task>/<scene>`); set `$SPATIALARENA_ROOT` to such a root to
evaluate from it as is.
"""

from __future__ import annotations

import os
import pathlib

SPATIALARENA = "spatialarena"
SPATIALARENA_ROOT_ENV = "SPATIALARENA_ROOT"
TASKS = ("MonoDepth", "VideoDepth", "MVRecon", "SfM", "SLAM")


def track_root_override(task: str) -> pathlib.Path | None:
    """Returns `$SPATIALARENA_ROOT/<task>` when the variable is set.

    Args:
        task: Benchmark task, one of `TASKS`.

    Returns:
        The task folder below `$SPATIALARENA_ROOT`, or None when the variable
        is unset or empty (the dataset then sits in the benchmark root).
    """
    root = os.environ.get(SPATIALARENA_ROOT_ENV, "").strip()
    if not root:
        return None
    return pathlib.Path(root).expanduser() / task
