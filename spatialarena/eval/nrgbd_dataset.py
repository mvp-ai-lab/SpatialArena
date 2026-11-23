# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Dataset adapter for the canonical Neural RGB-D layout."""

from spatialarena.eval import canonical_dataset


class DatasetNRGBD(canonical_dataset.CanonicalDataset):
    """Reads the canonical Neural RGB-D scenes (MVRecon task)."""

    dataset_name = "nrgbd"
    task_name = "MVRecon"
