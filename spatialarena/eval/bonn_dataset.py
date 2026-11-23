# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Dataset adapter for the canonical Bonn RGB-D layout."""

from spatialarena.eval import canonical_dataset


class DatasetBonn(canonical_dataset.CanonicalDataset):
    """Reads the canonical Bonn RGB-D scenes (VideoDepth task)."""

    dataset_name = "bonn"
    task_name = "VideoDepth"
