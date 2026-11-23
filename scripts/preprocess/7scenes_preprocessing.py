# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Register the 7-Scenes depth maps to the RGB camera.

For every frame in the train and test splits of each scene, the depth map is
reprojected from the depth sensor into the RGB sensor and written next to the
original as `*.depth.proj.png` (uint16 millimeters, 0 for no depth).

Code from https://github.com/tsattler/visloc_pseudo_gt_limitations/, borrowed
from
https://github.com/HengyiWang/amb3r/blob/main/benchmark/data/7scenes_preprocessing.py

Usage: python scripts/preprocess/7scenes_preprocessing.py --data_path <root>
"""

import argparse
import os
import warnings

import joblib
import numpy as np
from skimage import io
from tqdm import tqdm

focal_length = 525.0

# Focal length of the depth sensor (source:
# https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/).
d_focal_length = 585.0

# RGB image dimensions.
img_w = 640
img_h = 480

# Sub-sampling factor of the eye coordinate tensor.
nn_subsampling = 8

# Transformation from the depth sensor to the RGB sensor, calibrated according
# to
# https://projet.liris.cnrs.fr/voir/activities-dataset/kinect-calibration.html
d_to_rgb = np.array([
    [
        9.9996518012567637e-01,
        2.6765126468950343e-03,
        -7.9041012313000904e-03,
        -2.5558943178152542e-02,
    ],
    [
        -2.7409311281316700e-03,
        9.9996302803027592e-01,
        -8.1504520778013286e-03,
        1.0109636268061706e-04,
    ],
    [
        7.8819942130445332e-03,
        8.1718328771890631e-03,
        9.9993554558014031e-01,
        2.0318321729487039e-03,
    ],
    [0, 0, 0, 1],
])

_SCENES = [
    'chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs'
]


def mkdir(directory):
    """Checks whether the directory exists and creates it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def _register_depth(depth):
    """Reprojects a depth map from the depth sensor into the RGB sensor.

    Args:
        depth: Depth map from the depth sensor, float32 meters.

    Returns:
        A uint16 depth map in millimeters on the RGB image grid, 0 where no
        point lands.
    """
    d_h = depth.shape[0]
    d_w = depth.shape[1]

    # Reproject the depth map to 3D eye coordinates.
    eye_coords = np.zeros((4, d_h, d_w))
    # Set the x and y coordinates.
    eye_coords[0] = 0.5 + np.dstack([np.arange(0, d_w)] * d_h)[0].T
    eye_coords[1] = 0.5 + np.dstack([np.arange(0, d_h)] * d_w)[0]

    eye_coords = eye_coords.reshape(4, -1)
    depth = depth.reshape(-1)

    # Filter pixels with invalid depth.
    depth_mask = (depth > 0) & (depth < 100)
    eye_coords = eye_coords[:, depth_mask]
    depth = depth[depth_mask]

    # Subtract the depth principal point (assumed at the image center).
    eye_coords[0] -= d_w / 2
    eye_coords[1] -= d_h / 2
    # Reproject.
    eye_coords[0:2] /= d_focal_length
    eye_coords[0] *= depth
    eye_coords[1] *= depth
    eye_coords[2] = depth
    eye_coords[3] = 1

    # Transform from the depth sensor to the RGB sensor.
    eye_coords = np.matmul(d_to_rgb, eye_coords)

    # Project.
    depth = eye_coords[2]

    eye_coords[0] /= depth
    eye_coords[1] /= depth
    eye_coords[0:2] *= focal_length

    # Add the RGB principal point (assumed at the image center).
    eye_coords[0] += img_w / 2
    eye_coords[1] += img_h / 2

    registered_depth = np.ones((img_h, img_w), dtype=np.float32) * 2e3

    for pt in range(eye_coords.shape[1]):
        x = round(eye_coords[0, pt])
        y = round(eye_coords[1, pt])
        point_depth = eye_coords[2, pt]

        if x < 0 or y < 0 or y >= d_h or x >= d_w:
            continue

        registered_depth[y, x] = min(registered_depth[y, x], point_depth)

    registered_depth[registered_depth > 1e3] = 0
    return (1000 * registered_depth).astype(np.uint16)


def _process_split(ds, split_file):
    """Registers the depth maps of every sequence listed in one split file.

    Args:
        ds: Scene folder, for example `<root>/chess`.
        split_file: Split file name inside the scene folder.
    """
    with open(ds + '/' + split_file, 'r', encoding='utf-8') as f:
        split = f.readlines()
    # Map sequences to folder names.
    split = ['seq-' + line.strip()[8:].zfill(2) for line in split]

    for seq in split:
        files = os.listdir(ds + '/' + seq)

        # Adjust depth files by mapping them to the RGB sensor.
        depth_files = [f for f in files if f.endswith('depth.png')]

        for depth_file in tqdm(depth_files,
                               desc=f"{os.path.basename(ds)} {seq}"):
            depth = io.imread(ds + '/' + seq + '/' + depth_file)
            depth = depth.astype(np.float32)
            depth /= 1000  # From millimeters to meters.

            registered_depth = _register_depth(depth)

            # Store the calibrated depth.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                io.imsave(
                    ds + '/' + seq + '/' +
                    depth_file.replace("depth.png", "depth.proj.png"),
                    registered_depth)


def process_scene(ds):
    """Registers the depth maps of one scene's train and test splits.

    Args:
        ds: Scene folder, for example `<root>/chess`.
    """
    _process_split(ds, 'TrainSplit.txt')
    _process_split(ds, 'TestSplit.txt')


def main():
    """Runs the script with the command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        type=str,
                        required=True,
                        help='Path to the 7scenes data folder')
    args = parser.parse_args()

    joblib.Parallel(n_jobs=7, verbose=0)(
        map(joblib.delayed(process_scene),
            [os.path.join(args.data_path, scan_name) for scan_name in _SCENES]))


if __name__ == '__main__':
    main()
