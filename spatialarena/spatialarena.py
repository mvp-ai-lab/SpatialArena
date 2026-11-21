# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""SpatialArena, the benchmark's own captures (handheld 3-camera LiDAR rig; docs/benchmark/spatialarena.md).

In a benchmark root it sits where every dataset does, ``<benchmark_root>/<Task>/spatialarena/<scene>`` (the layout
of the Hugging Face release). ``scripts/preprocess/materialize_spatialarena.py`` writes it with the tracks holding the
scenes directly (``<root>/<Task>/<scene>``); set ``$SPATIALARENA_ROOT`` to such a root to evaluate from it as is.
"""

from __future__ import annotations

import os
from pathlib import Path

SPATIALARENA = "spatialarena"
SPATIALARENA_ROOT_ENV = "SPATIALARENA_ROOT"
TASKS = ("MonoDepth", "VideoDepth", "MVRecon", "SfM", "SLAM")


def track_root_override(task: str) -> Path | None:
    """``$SPATIALARENA_ROOT/<task>`` when the variable is set, else None (the dataset sits in the benchmark root)."""
    root = os.environ.get(SPATIALARENA_ROOT_ENV, "").strip()
    return Path(root).expanduser() / task if root else None
