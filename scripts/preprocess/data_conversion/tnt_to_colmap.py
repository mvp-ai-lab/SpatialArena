# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Convert Tanks and Temples scenes with their GT poses into COLMAP models.

For each training scene this runs COLMAP feature extraction and sequential
matching, writes the GT poses from `<scene>_COLMAP_SfM.log` as a posed COLMAP
model in `<scene>/sparse`, triangulates points with the poses fixed and
undistorts the images into the scene folder. The COLMAP binary comes from
`$COLMAP_BIN`.

-----------------------------------------------------------------------------
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
"""

import json
import os

import numpy as np

from scripts.preprocess.data_conversion import colmap_database
from spatialarena.geometry import read_write_model

COLMAP_BIN = os.environ.get(
    "COLMAP_BIN", os.path.expanduser("~/Projects/colmap/install/bin/colmap"))


def create_init_files(pinhole_dict_file, db_file, out_dir):
    """Writes a posed COLMAP text model whose image ids match the database.

    Partially adapted from
    https://github.com/Kai-46/nerfplusplus/blob/master/colmap_runner/run_colmap_posed.py

    Args:
        pinhole_dict_file: JSON written by `convert_cam_dict_to_pinhole_dict`.
        db_file: COLMAP database with the extracted images.
        out_dir: Output folder for `cameras.txt`, `images.txt` and an empty
            `points3D.txt`.
    """

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Create the per-image line templates.
    with open(pinhole_dict_file, encoding='utf-8') as fp:
        pinhole_dict = json.load(fp)

    template = {}
    cameras_line_template = ('{camera_id} RADIAL {width} {height} '
                             '{f} {cx} {cy} {k1} {k2}\n')
    images_line_template = ('{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} '
                            '{camera_id} {image_name}\n\n')

    for img_name in pinhole_dict:
        # The parameters are w, h, fx, fy, cx, cy, qvec, t.
        params = pinhole_dict[img_name]
        width = params[0]
        height = params[1]
        fx = params[2]
        cx = params[4]
        cy = params[5]
        qvec = params[6:10]
        tvec = params[10:13]

        cam_line = cameras_line_template.format(camera_id="{camera_id}",
                                                width=width,
                                                height=height,
                                                f=fx,
                                                cx=cx,
                                                cy=cy,
                                                k1=0,
                                                k2=0)
        img_line = images_line_template.format(image_id="{image_id}",
                                               qw=qvec[0],
                                               qx=qvec[1],
                                               qy=qvec[2],
                                               qz=qvec[3],
                                               tx=tvec[0],
                                               ty=tvec[1],
                                               tz=tvec[2],
                                               camera_id="{camera_id}",
                                               image_name=img_name)
        template[img_name] = (cam_line, img_line)

    # Read the image ids from the database.
    database = colmap_database.COLMAPDatabase.connect(db_file)
    table_images = database.execute("SELECT * FROM images")
    img_name2id_dict = {}
    for row in table_images:
        img_name2id_dict[row[1]] = row[0]

    # Every image shares the one camera: its line is the same in each template
    # entry (here, the last one).
    cameras_txt_lines = [list(template.values())[-1][0].format(camera_id=1)]
    images_txt_lines = []
    for img_name, img_id in img_name2id_dict.items():
        image_line = template[img_name][1].format(image_id=img_id, camera_id=1)
        images_txt_lines.append(image_line)

    with open(os.path.join(out_dir, 'cameras.txt'), 'w',
              encoding='utf-8') as fp:
        fp.writelines(cameras_txt_lines)

    with open(os.path.join(out_dir, 'images.txt'), 'w', encoding='utf-8') as fp:
        fp.writelines(images_txt_lines)
        fp.write('\n')

    # Create an empty points3D.txt.
    with open(os.path.join(out_dir, 'points3D.txt'), 'w', encoding='utf-8'):
        pass


def convert_cam_dict_to_pinhole_dict(cam_dict, pinhole_dict_file):
    """Writes world-to-camera poses as pinhole parameters to a JSON file.

    The intrinsics are a fixed initial guess for the 1920x1080 frames
    (f = 0.6 * width, principal point at the center); COLMAP refines them.

    Partially adapted from
    https://github.com/Kai-46/nerfplusplus/blob/master/colmap_runner/run_colmap_posed.py

    Args:
        cam_dict: Image name to 4x4 world-to-camera matrix.
        pinhole_dict_file: Output JSON path; each entry is
            `[w, h, fx, fy, cx, cy, qw, qx, qy, qz, tx, ty, tz]`.
    """
    print('Writing pinhole_dict to: ', pinhole_dict_file)
    height = 1080
    width = 1920

    pinhole_dict = {}
    for img_name in cam_dict:
        world_to_cam = cam_dict[img_name]

        fx = 0.6 * width
        fy = 0.6 * width
        cx = width / 2.0
        cy = height / 2.0

        qvec = read_write_model.rotmat2qvec(world_to_cam[:3, :3])
        tvec = world_to_cam[:3, 3]

        params = [
            width, height, fx, fy, cx, cy, qvec[0], qvec[1], qvec[2], qvec[3],
            tvec[0], tvec[1], tvec[2]
        ]
        pinhole_dict[img_name] = params

    with open(pinhole_dict_file, 'w', encoding='utf-8') as fp:
        json.dump(pinhole_dict, fp, indent=2, sort_keys=True)


def load_COLMAP_poses(cam_file, img_dir, tf='w2c'):  # pylint: disable=invalid-name
    """Reads the camera-to-world poses of a Tanks and Temples `.log` file.

    Each pose takes five lines: a header `<image index> <valid> <n>` and the
    four rows of the matrix. Poses whose valid flag is -1 are skipped.

    Args:
        cam_file: `<scene>_COLMAP_SfM.log` path.
        img_dir: Image folder; its sorted listing maps indices to file names.
        tf: 'c2w' to return camera-to-world matrices keyed by image index;
            anything else returns world-to-camera matrices keyed by file name.

    Returns:
        The poses, as described for `tf`.
    """
    names = sorted(os.listdir(img_dir))

    with open(cam_file, encoding='utf-8') as f:
        lines = f.readlines()

    # The file stores camera-to-world matrices.
    poses = {}
    for idx, line in enumerate(lines):
        if idx % 5 == 0:  # Header line.
            img_idx, valid, _ = line.split(' ')
            if valid != '-1':
                poses[int(img_idx)] = np.eye(4)
        else:
            if int(img_idx) in poses:
                num = np.array([float(n) for n in line.split(' ')])
                poses[int(img_idx)][idx % 5 - 1, :] = num

    if tf == 'c2w':
        return poses
    else:
        # Convert to world-to-camera (the NeRF convention).
        poses_w2c = {}
        for image_index, cam_to_world in poses.items():
            print(f"k {image_index}, names {len(names)}, {names[-1]}")
            poses_w2c[names[image_index]] = np.linalg.inv(cam_to_world)
        return poses_w2c


def load_transformation(trans_file):
    """Reads a 4x4 matrix stored as four space-separated rows."""
    with open(trans_file, encoding='utf-8') as f:
        lines = f.readlines()

    trans = np.eye(4)
    for idx, line in enumerate(lines):
        num = np.array([float(n) for n in line.split(' ')])
        trans[idx, :] = num

    return trans


def align_gt_with_cam(pts, trans):
    """Maps points through the inverse of the 4x4 transform `trans`."""
    trans_inv = np.linalg.inv(trans)
    pts_aligned = pts @ trans_inv[:3, :3].transpose(-1, -2) + trans_inv[:3, -1]
    return pts_aligned


def compute_bound(pts):
    """Returns the center, a 1% padded radius and the bounds of `pts`."""
    bounding_box = np.array([pts.min(axis=0), pts.max(axis=0)])
    center = bounding_box.mean(axis=0)
    radius = np.max(np.linalg.norm(pts - center, axis=-1)) * 1.01
    return center, radius, bounding_box.T.tolist()


def init_colmap(tnt_path: str):
    """Builds the posed, triangulated and undistorted model of each scene.

    Args:
        tnt_path: Tanks and Temples root with one folder per scene.

    Raises:
        FileNotFoundError: A scene has no `images` folder.
    """
    scene_list = [
        "Barn",
        "Caterpillar",
        "Courthouse",
        "Ignatius",
        "Meetingroom",
        "Truck",
    ]

    for scene in scene_list:
        scene_path = os.path.join(tnt_path, scene)
        if not os.path.exists(f"{scene_path}/images"):
            raise FileNotFoundError(
                f"'images` folder cannot be found in {scene_path}."
                "Please check the expected folder structure in "
                "DATA_PREPROCESSING.md")

        # Extract features.
        os.system(f"{COLMAP_BIN} feature_extractor \
                --database_path {scene_path}/database.db \
                --image_path {scene_path}/images \
                --ImageReader.camera_model=RADIAL \
                --FeatureExtraction.use_gpu=true \
                --FeatureExtraction.num_threads=32 \
                --ImageReader.single_camera=true")

        # Match features.
        os.system(f"{COLMAP_BIN} sequential_matcher \
                --database_path {scene_path}/database.db \
                --FeatureMatching.use_gpu=true")

        # Read the GT poses.
        poses = load_COLMAP_poses(
            os.path.join(scene_path, f'{scene}_COLMAP_SfM.log'),
            os.path.join(scene_path, 'images'))

        # Convert them to COLMAP files.
        pinhole_dict_file = os.path.join(scene_path, 'pinhole_dict.json')
        convert_cam_dict_to_pinhole_dict(poses, pinhole_dict_file)

        db_file = os.path.join(scene_path, 'database.db')
        sfm_dir = os.path.join(scene_path, 'sparse')
        create_init_files(pinhole_dict_file, db_file, sfm_dir)

        # Triangulate and bundle-adjust with the poses fixed.
        os.system(f"{COLMAP_BIN} point_triangulator \
                --database_path {scene_path}/database.db \
                --image_path {scene_path}/images \
                --input_path {scene_path}/sparse \
                --output_path {scene_path}/sparse \
                --Mapper.tri_ignore_two_view_tracks=true")

        os.system(f"{COLMAP_BIN} bundle_adjuster \
                --input_path {scene_path}/sparse \
                --output_path {scene_path}/sparse \
                --BundleAdjustment.refine_extrinsics=false")

        # Undistort.
        os.system(f"{COLMAP_BIN} image_undistorter \
            --image_path {scene_path}/images \
            --input_path {scene_path}/sparse \
            --output_path {scene_path} \
            --output_type COLMAP ")


def main():
    """Runs the script with the command-line arguments."""
    tnt_path = os.path.expanduser("~/datasets/tanks_and_temples")
    init_colmap(tnt_path)


if __name__ == '__main__':
    main()
