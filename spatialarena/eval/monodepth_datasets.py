# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""MonoDepth datasets beyond the Marigold suite, in the canonical layout.

Each dataset root holds metadata.json (one entry per scene, with the scene's
depth limits, see `depth_prediction_evaluator.monodepth_scene_limits`) and one
folder per scene with images/, depths/ (float32 .npy in metres, 0 = invalid),
optional masks/ and materialized_full.json (per-frame records with pixel
intrinsics).
"""

from spatialarena.eval import canonical_dataset


class DatasetAerialMetric(canonical_dataset.CanonicalDataset):
    """AerialMetric Test-Split (Decoupled sites, Oblique, Wild).

    Built by scripts/preprocess/materialize_aerialmetric.py.
    """
    dataset_name = "aerialmetric"
    task_name = "MonoDepth"


class DatasetHammer(canonical_dataset.CanonicalDataset):
    """HAMMER as processed by MoGe.

    Built by scripts/preprocess/materialize_moge_eval.py.
    """
    dataset_name = "hammer"
    task_name = "MonoDepth"


class DatasetIBims1(canonical_dataset.CanonicalDataset):
    """iBims-1 as processed by MoGe.

    Built by scripts/preprocess/materialize_moge_eval.py.
    """
    dataset_name = "ibims1"
    task_name = "MonoDepth"


class DatasetSpring(canonical_dataset.CanonicalDataset):
    """Spring train frames selected by MoGe.

    Built by scripts/preprocess/materialize_moge_eval.py.
    """
    dataset_name = "spring"
    task_name = "MonoDepth"


class DatasetSynth4K(canonical_dataset.CanonicalDataset):
    """Synth4K (InfiniDepth), one scene per game.

    Built by scripts/preprocess/materialize_synth4k.py.
    """
    dataset_name = "synth4k"
    task_name = "MonoDepth"
