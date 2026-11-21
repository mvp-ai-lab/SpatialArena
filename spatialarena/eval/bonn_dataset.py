# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from spatialarena.eval.canonical_dataset import CanonicalDataset


class DatasetBonn(CanonicalDataset):
    dataset_name = "bonn"
    task_name = "VideoDepth"
