# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from spatialarena.eval.canonical_dataset import CanonicalDataset


class DatasetKitti(CanonicalDataset):
    dataset_name = "kitti"
    task_name = "VideoDepth"
