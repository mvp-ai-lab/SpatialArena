# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Cut each Bonn RGB-D scene down to the 110 frames the benchmark evaluates.

Borrowed from
https://github.com/HengyiWang/amb3r/blob/main/benchmark/data/bonn_preprocessing.py

Usage: python scripts/preprocess/bonn_preprocessing.py --data_path <bonn_root>
"""

import argparse
import glob
import os
import shutil

import numpy as np
from tqdm import tqdm


def process_bonn(data_path):
    """Extracts frames 30-140 of every Bonn scene.

    Writes the RGB and depth frames to `rgb_110/` and `depth_110/` and the
    matching ground-truth rows to `groundtruth_110.txt`, next to the originals.

    Args:
        data_path: Folder that contains `rgbd_bonn_dataset/`.
    """

    dataset_dir = os.path.join(data_path, "rgbd_bonn_dataset")
    dirs = sorted(glob.glob(os.path.join(dataset_dir, "*/")))

    for scene_dir in tqdm(dirs, desc="Processing Bonn scenes"):
        # Copy RGB frames 30-140.
        rgb_frames = sorted(glob.glob(os.path.join(scene_dir, "rgb", "*.png")))
        rgb_subset = rgb_frames[30:140]
        rgb_out = os.path.join(scene_dir, "rgb_110")
        os.makedirs(rgb_out, exist_ok=True)
        for frame in rgb_subset:
            shutil.copy(frame, rgb_out)

        # Copy depth frames 30-140.
        depth_frames = sorted(
            glob.glob(os.path.join(scene_dir, "depth", "*.png")))
        depth_subset = depth_frames[30:140]
        depth_out = os.path.join(scene_dir, "depth_110")
        os.makedirs(depth_out, exist_ok=True)
        for frame in depth_subset:
            shutil.copy(frame, depth_out)

        # Slice ground-truth poses 30-140.
        gt_path = os.path.join(scene_dir, "groundtruth.txt")
        if os.path.exists(gt_path):
            gt_poses = np.loadtxt(gt_path)
            gt_subset = gt_poses[30:140]
            np.savetxt(os.path.join(scene_dir, "groundtruth_110.txt"),
                       gt_subset)


def main():
    """Runs the script with the command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",
                        type=str,
                        required=True,
                        help="Path to the Bonn data folder")
    args = parser.parse_args()
    process_bonn(args.data_path)


if __name__ == "__main__":
    main()
