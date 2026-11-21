# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from spatialarena.eval.canonical_dataset import CanonicalDataset


class DatasetMarigold(CanonicalDataset):
    dataset_name = "marigold"
    task_name = "MonoDepth"
