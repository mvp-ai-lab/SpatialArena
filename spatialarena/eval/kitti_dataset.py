# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Dataset adapter for the canonical KITTI layout."""

from spatialarena.eval import canonical_dataset


class DatasetKitti(canonical_dataset.CanonicalDataset):
    """Reads the canonical KITTI scenes (VideoDepth task)."""

    dataset_name = "kitti"
    task_name = "VideoDepth"
