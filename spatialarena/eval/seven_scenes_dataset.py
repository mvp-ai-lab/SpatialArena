# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Dataset adapter for the canonical 7-Scenes layout."""

from spatialarena.eval import canonical_dataset


class DatasetSevenScenes(canonical_dataset.CanonicalDataset):
    """Reads the canonical 7-Scenes scenes (MVRecon task)."""

    dataset_name = "7scenes"
    task_name = "MVRecon"
