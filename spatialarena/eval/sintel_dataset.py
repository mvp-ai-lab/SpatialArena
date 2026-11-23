# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Dataset adapter for the canonical MPI Sintel layout."""

from spatialarena.eval import canonical_dataset


class DatasetSintel(canonical_dataset.CanonicalDataset):
    """Reads the canonical MPI Sintel scenes (VideoDepth task)."""

    dataset_name = "sintel"
    task_name = "VideoDepth"
