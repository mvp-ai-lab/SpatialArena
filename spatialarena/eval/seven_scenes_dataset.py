# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from spatialarena.eval.canonical_dataset import CanonicalDataset


class DatasetSevenScenes(CanonicalDataset):
    dataset_name = "7scenes"
    task_name = "MVRecon"
