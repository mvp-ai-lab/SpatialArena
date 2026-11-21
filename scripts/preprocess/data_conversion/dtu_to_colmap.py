# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

# pylint: disable=E1101

# This script is used for converting dtu dataset into COLMAP format.
# the DTU dataset. Please download it at:
#  https://drive.google.com/file/d/135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_/view
# before running this script.
# DTU dataset downloading follows: https://github.com/YoYo000/MVSNet
#  Training set: https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view
#  Test set: https://drive.google.com/open?id=135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_
#  GT Depths map:
#   https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/
#   Depths_raw.zip
#  GT clean points: http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip

import argparse
import os

import torch
import numpy as np
import open3d as o3d
import imageio.v2 as iio
from PIL import Image
from tqdm import trange

from spatialarena.geometry.utils import write_colmap_formats
from spatialarena.utils.io import read_pfm


parser = argparse.ArgumentParser(
    description='Depth fusion with consistency check.')
parser.add_argument('--root_path', type=str,
                    default='[/path/to/]dtu-test-1200')
parser.add_argument('--depth_path', type=str, default='')
parser.add_argument('--dist_thresh', type=float, default=0.2)
parser.add_argument('--num_consist', type=int, default=4)


def parse_cameras(path):
    with open(path, encoding='utf-8') as file:
        cam_txt = file.readlines()
    def f(xs): return list(map(lambda x: list(map(float, x.strip().split())), xs))

    extr_mat = f(cam_txt[1:5])
    intr_mat = f(cam_txt[7:10])

    extr_mat = np.array(extr_mat, np.float32)
    intr_mat = np.array(intr_mat, np.float32)

    return extr_mat, intr_mat


def load_data(data_path: str, scene_name: str):
    projs, rgbs = [], []
    image_paths, world_to_cams, intrinsics = [], [], []
    mask_path = os.path.join(data_path, scene_name, "masks")
    os.makedirs(mask_path, exist_ok=True)

    for view in range(49):
        img_filename = f"{data_path}/{scene_name}/images/{view:08d}.jpg"
        cam_filename = f"{data_path}/{scene_name}/cams/{view:08d}_cam.txt"
        depth_filename = f"{data_path}/{scene_name}/depths/depth_map_{view:04d}.pfm"

        w2c_mat, intr_mat = parse_cameras(cam_filename)
        proj_mat = np.eye(4)
        proj_mat[:3, :4] = np.dot(intr_mat[:3, :3], w2c_mat[:3, :4])
        projs.append(torch.from_numpy(proj_mat))

        depth, _ = read_pfm(depth_filename)
        binary_mask = (depth > 0.01).astype(np.uint8) * 255
        mask_filename = os.path.join(mask_path, f"{view:08d}.png")
        iio.imwrite(mask_filename, binary_mask)

        rgb = np.array(Image.open(img_filename))
        rgbs.append(rgb)

        image_paths.append(img_filename)
        world_to_cams.append(w2c_mat)
        intrinsics.append(intr_mat)

    projs = torch.stack(projs).float()

    if torch.cuda.is_available():
        # depths = depths.cuda()
        projs = projs.cuda()

    world_to_cams = np.stack(world_to_cams, axis=0)
    intrinsics = np.stack(intrinsics, axis=0)

    return projs, rgbs, world_to_cams, intrinsics, image_paths


def main() -> None:
    args = parser.parse_args()
    data_list = os.path.join(args.root_path, "scan_list_test.txt")
    with open(data_list, 'r', encoding='utf-8') as file:
        all_scenes = file.readlines()
    all_scenes = list(map(str.strip, all_scenes))

    pbar = trange(len(all_scenes), desc="Fusing depth maps...")
    for scene in all_scenes:
        _projs, rgbs, world_to_cams, intrinsics, image_paths = load_data(
            args.root_path, scene
        )

        scene_id = int(scene[4:])
        gt_ply_path = os.path.join(args.root_path, f"Points/stl/stl{scene_id:03d}_total.ply")
        assert os.path.exists(gt_ply_path), f"GT point cloud {gt_ply_path} not exists!"
        gt_pcd = o3d.io.read_point_cloud(gt_ply_path)
        final_points = np.asarray(gt_pcd.points)
        final_rgbs = np.asarray(gt_pcd.colors)

        height, width = rgbs[0].shape[:2]

        colmap_output_path = os.path.join(args.root_path, scene, "sparse")
        os.makedirs(colmap_output_path, exist_ok=True)
        new_ply_path = os.path.join(colmap_output_path, "points3D.ply")
        o3d.io.write_point_cloud(new_ply_path, gt_pcd)

        write_colmap_formats(
            f"{scene}/images", world_to_cams, intrinsics, image_paths,
            final_points, (final_rgbs * 255).astype(np.uint8),
            width, height, colmap_output_path,
        )

        pbar.update(1)


if __name__ == '__main__':
    main()
