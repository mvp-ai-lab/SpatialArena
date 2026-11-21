# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from spatialarena.eval.canonical_dataset import CanonicalDataset


class DatasetSintel(CanonicalDataset):
    dataset_name = "sintel"
    task_name = "VideoDepth"
