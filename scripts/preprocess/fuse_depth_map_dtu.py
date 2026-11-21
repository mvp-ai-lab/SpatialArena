# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

# pylint: disable=E1101

# This script is used for converting dtu dataset into COLMAP format.
# the DTU dataset. Please download it at:
#  https://drive.google.com/file/d/135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_/view
# before running this script.
# DTU dataset downloading follows: https://github.com/YoYo000/MVSNet
#

import argparse
import os

import torch
import numpy as np
import open3d as o3d
import cv2
from PIL import Image
from tqdm import trange

import pycolmap

from spatialarena.geometry.geometry import unproject_depth_map_to_point_map
from spatialarena.geometry.read_write_model import write_points3D_binary
from spatialarena.geometry.utils import to_colmap_points, randomly_limit_trues
from spatialarena.geometry.mvs_depth_fusion import depth_fusion


parser = argparse.ArgumentParser(
    description='Depth fusion with consistency check.')
parser.add_argument('--root_path', type=str,
                    default='[/path/to/]dtu-test-1200')
parser.add_argument('--depth_path', type=str, default='')
parser.add_argument('--dist_thresh', type=float, default=0.005)
parser.add_argument('--num_consist', type=int, default=8)
parser.add_argument(
    "--max_points",
    type=int,
    default=5_000_000,
    help="Max number of 3D points to keep when exporting",
)


def resize_to_depth_shape(image, mask, depth_shape, interpolation_method='area'):
    """
    Resize image and mask to match depth map shape.

    Args:
        image: RGB image array (H, W, 3)
        mask: Binary mask array (H, W) or (H, W, 1)
        depth_shape: Target shape (height, width) of depth map
        interpolation_method: 'area', 'linear', 'nearest', 'cubic'

    Returns:
        resized_image, resized_mask
    """
    # Map interpolation methods to OpenCV constants
    interpolation_map = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'area': cv2.INTER_AREA,
        'cubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }

    interp = interpolation_map.get(interpolation_method, cv2.INTER_AREA)

    # Get target dimensions (reverse order for OpenCV: width, height)
    target_dims = (depth_shape[1], depth_shape[0])

    # Resize image with appropriate interpolation
    if len(image.shape) == 3:
        resized_image = cv2.resize(image, target_dims, interpolation=interp)
    else:
        resized_image = cv2.resize(image, target_dims, interpolation=interp)

    # For mask, always use nearest neighbor to preserve binary values
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]  # Take first channel if mask is 3D

    resized_mask = cv2.resize(
        mask, target_dims, interpolation=cv2.INTER_NEAREST)

    # Ensure mask remains binary
    resized_mask = (resized_mask > 0).astype(np.uint8) * 255

    return resized_image, resized_mask


def load_data(
    data_path: str,
    depth_path: str,
    scene_name: str,
    extrinsic_dict: dict,
    intrinsic_dict: dict,
):
    image_paths = []
    depths, projs, rgbs = [], [], []
    valid_masks = []
    mask_path = os.path.join(data_path, scene_name, "masks")
    os.makedirs(mask_path, exist_ok=True)

    proj_list, extr_list, intr_list = [], [], []
    for view in range(49):
        image_filename = f"{data_path}/{scene_name}/images/{view:08d}.jpg"
        mask_filename = f"{mask_path}/{view:08d}.png"
        depth_filename = f"{depth_path}/{scene_name}/depth/{view:06d}.npy"

        rgb = np.array(Image.open(image_filename))
        mask = np.array(Image.open(mask_filename))
        depth = np.load(depth_filename)  # .squeeze(-1)

        # Resize rgb and mask ot align with the resolution of depth map.
        rgb, mask = resize_to_depth_shape(rgb, mask, depth.shape)

        rgbs.append(rgb)
        depths.append(depth[None, ...])

        image_paths.append(image_filename)

        pose = extrinsic_dict[f"{view:08d}.jpg"]
        intrinsic = intrinsic_dict[f"{view:08d}.jpg"]
        extr_list.append(pose)
        intr_list.append(intrinsic)

        proj = np.eye(4)
        proj[:3, :4] = np.dot(intrinsic[:3, :3], pose[:3, :4])

        points = unproject_depth_map_to_point_map(
            depth[None], pose[None], intrinsic[None]
        )
        points = points.reshape(-1, 3)
        valid_mask = (mask != 0)
        valid_masks.append(valid_mask)

        # valid_mask = valid_mask.reshape(-1)
        # points = points[valid_mask]
        # colors = rgb.reshape(-1, 3)[valid_mask]

        proj_list.append(proj)

    depths = np.stack(depths, axis=0)
    projs = np.stack(proj_list, axis=0)
    extrinsics = np.stack(extr_list, axis=0)
    intrinsics = np.stack(intr_list, axis=0)
    valid_masks = np.stack(valid_masks, axis=0)

    return image_paths, rgbs, depths, projs, extrinsics, intrinsics, valid_masks


def retrieve_projs_from_reconstruction(
    sparse: pycolmap.Reconstruction
) -> dict[str, np.ndarray]:
    extrinsic_dict, intrinsic_dict = {}, {}
    cameras = sparse.cameras

    for image in sparse.images.values():
        cam_id = image.camera_id
        camera = cameras[cam_id]
        intrinsics = np.array([
            [camera.focal_length_x, 0, camera.principal_point_x],
            [0, camera.focal_length_y, camera.principal_point_y],
            [0, 0, 1]
        ])
        pose = np.eye(4)
        pose[:3, :4] = image.cam_from_world().matrix()

        extrinsic_dict[image.name] = pose
        intrinsic_dict[image.name] = intrinsics

    return extrinsic_dict, intrinsic_dict


def write_ply(filename, points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(points[:, 3:] / 255.)
    o3d.io.write_point_cloud(filename, pcd, write_ascii=False)


def main() -> None:
    args = parser.parse_args()
    data_list = os.path.join(args.root_path, "scan_list_test.txt")
    with open(data_list, 'r', encoding='utf-8') as file:
        all_scenes = file.readlines()
    all_scenes = list(map(str.strip, all_scenes))

    device = 'cuda'
    MAX_TRIALS = 3

    pbar = trange(len(all_scenes), desc="Fusing depth maps...")
    for scene in all_scenes:
        sparse_path = os.path.join(args.depth_path, scene, "sparse")
        sparse = pycolmap.Reconstruction(sparse_path)

        extrinsic_dict, intrinsic_dict = retrieve_projs_from_reconstruction(
            sparse)

        _image_paths, rgbs, depths, projs, _extrinsics, _intrinsics, valid_masks = load_data(
            args.root_path, args.depth_path, scene, extrinsic_dict, intrinsic_dict
        )
        depths = torch.from_numpy(depths).float().to(device)
        projs = torch.from_numpy(projs).float().to(device)
        valid_masks = torch.from_numpy(valid_masks).to(device)

        num_fused_points = 0
        num_trials = 0
        dist_thresh = args.dist_thresh
        num_consist = args.num_consist
        while num_fused_points == 0 and num_trials <= MAX_TRIALS:
            fused_points = depth_fusion(
                depths, projs, rgbs, valid_masks,
                dist_thresh=dist_thresh,
                num_consist=num_consist,
            )

            mask = np.ones(fused_points.shape[0], dtype=bool)
            mask = randomly_limit_trues(mask, args.max_points)
            fused_points = fused_points[mask]
            num_fused_points = len(fused_points)

            if num_fused_points == 0:
                print('No points obtained from depth fusion, Retrying...')
                dist_thresh *= 2
                num_consist = max(2, num_consist // 2)
            num_trials += 1

        # Overwrite the original points with the filtered points.
        colmap_points3D = to_colmap_points(
            fused_points[:, :3],
            (fused_points[:, 3:] * 255).astype(np.uint8),
        )
        colmap_points_path = os.path.join(sparse_path, "points3D.bin")
        write_points3D_binary(colmap_points3D, colmap_points_path)

        ply_path = os.path.join(sparse_path, "points3D.ply")
        write_ply(ply_path, fused_points)

        pbar.update(1)


if __name__ == "__main__":
    main()
