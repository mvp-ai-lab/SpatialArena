# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Image Matching Challenge (IMC) dataset in the canonical SfM layout."""

from spatialarena.eval import canonical_dataset


class DatasetIMC(canonical_dataset.CanonicalDataset):
    """IMC scenes for the SfM task."""
    dataset_name = "imc"
    task_name = "SfM"
