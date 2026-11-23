# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Dataset adapter for the canonical Marigold layout."""

from spatialarena.eval import canonical_dataset


class DatasetMarigold(canonical_dataset.CanonicalDataset):
    """Reads the canonical Marigold scenes (MonoDepth task)."""

    dataset_name = "marigold"
    task_name = "MonoDepth"
