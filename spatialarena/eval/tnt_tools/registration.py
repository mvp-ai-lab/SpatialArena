# ----------------------------------------------------------------------------
# -                   TanksAndTemples Website Toolbox                        -
# -                    http://www.tanksandtemples.org                        -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2017
# Arno Knapitsch <arno.knapitsch@gmail.com >
# Jaesik Park <syncle@gmail.com>
# Qian-Yi Zhou <Qianyi.Zhou@gmail.com>
# Vladlen Koltun <vkoltun@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ----------------------------------------------------------------------------
#
# This python script is for downloading dataset from www.tanksandtemples.org
# The dataset has a different license, please refer to
# https://tanksandtemples.org/license/
"""Trajectory and point-cloud registration of the Tanks and Temples toolbox.

`eval_tnt_mesh` first aligns the estimated camera trajectory to the GT one
(`trajectory_alignment`), then refines the alignment with cropped,
downsampled ICP (`registration_vol_ds`, `registration_unif`).
"""

import copy

import numpy as np
import open3d as o3d

from spatialarena.eval.tnt_tools import trajectory_io

MAX_POINT_NUMBER = 4e6


def read_mapping(filename):
    """Reads a frame mapping file (sampled to movie frame numbers).

    Returns:
        [n_sampled_frames, n_total_frames, mapping], mapping an
        (n_sampled_frames, 2) array.
    """
    mapping = []
    with open(filename, "r") as file:
        n_sampled_frames = int(file.readline())
        n_total_frames = int(file.readline())
        mapping = np.zeros(shape=(n_sampled_frames, 2))
        metastr = file.readline()
        for frame_index in range(n_sampled_frames):
            metadata = list(map(int, metastr.split()))
            mapping[frame_index, :] = metadata
            metastr = file.readline()
    return [n_sampled_frames, n_total_frames, mapping]


def gen_sparse_trajectory(mapping, f_trajectory):
    """Returns the poses of f_trajectory at the mapping's (1-based) frames."""
    sparse_traj = []
    for entry in mapping:
        sparse_traj.append(f_trajectory[int(entry[1] - 1)])
    return sparse_traj


def trajectory_alignment(map_file, traj_to_register, gt_traj_col, gt_trans):
    """Estimates the transform from an estimated to the GT trajectory.

    RANSAC on the camera positions with known correspondences.

    Args:
        map_file: Optional frame mapping file; used for trajectories of more
            than 1600 poses (every movie frame).
        traj_to_register: The estimated trajectory.
        gt_traj_col: The GT (COLMAP) trajectory.
        gt_trans: Optional transform applied to the GT trajectory first.

    Returns:
        The 4x4 transformation.
    """
    traj_pcd_col = trajectory_io.convert_trajectory_to_pointcloud(gt_traj_col)
    if gt_trans is not None:
        traj_pcd_col.transform(gt_trans)

    corres = o3d.utility.Vector2iVector(
        np.asarray(list(map(lambda x: [x, x], range(len(gt_traj_col))))))
    criteria = o3d.pipelines.registration.RANSACConvergenceCriteria()
    criteria.max_iteration = 100000

    # In this case a log file was used which contains every movie frame (see
    # the tutorial for details).
    if len(traj_to_register) > 1600 and map_file is not None:
        _, _, mapping = read_mapping(map_file)
        traj_col2 = gen_sparse_trajectory(mapping, traj_to_register)
        traj_to_register_pcd = trajectory_io.convert_trajectory_to_pointcloud(
            traj_col2)
    else:
        print("Estimated trajectory will leave as it is, no sparsity op is "
              "performed!")
        traj_to_register_pcd = trajectory_io.convert_trajectory_to_pointcloud(
            traj_to_register)
    randomvar = 0.0
    if randomvar < 1e-5:
        traj_to_register_pcd_rand = traj_to_register_pcd
    else:
        nr_of_cam_pos = len(traj_to_register_pcd.points)
        rand_number_added = np.asanyarray(traj_to_register_pcd.points) * (
            np.random.rand(nr_of_cam_pos, 3) * randomvar - randomvar / 2.0 + 1)
        list_rand = list(rand_number_added)
        traj_to_register_pcd_rand = o3d.geometry.PointCloud()
        for elem in list_rand:
            traj_to_register_pcd_rand.points.append(elem)

    # Rough registration based on aligned COLMAP SfM data.
    o3d_registration = o3d.pipelines.registration
    reg = o3d_registration.registration_ransac_based_on_correspondence(
        traj_to_register_pcd_rand,
        traj_pcd_col,
        corres,
        0.2,
        o3d_registration.TransformationEstimationPointToPoint(True),
        6,
        criteria=criteria,
    )
    return reg.transformation


def crop_and_downsample(
        pcd,
        crop_volume,
        down_sample_method="voxel",
        voxel_size=0.01,
        trans=np.identity(4),
):
    """Transforms, crops and downsamples a copy of a point cloud.

    The array default of trans is safe: the function only reads it.

    Args:
        pcd: Point cloud; not modified.
        crop_volume: Open3D crop volume.
        down_sample_method: "voxel", or "uniform" (only above
            MAX_POINT_NUMBER points).
        voxel_size: Voxel size of the "voxel" method.
        trans: 4x4 transform applied before cropping.

    Returns:
        The cropped and downsampled point cloud.
    """
    pcd_copy = copy.deepcopy(pcd)
    pcd_copy.transform(trans)
    pcd_crop = crop_volume.crop_point_cloud(pcd_copy)
    if down_sample_method == "voxel":
        return pcd_crop.voxel_down_sample(voxel_size)
    elif down_sample_method == "uniform":
        n_points = len(pcd_crop.points)
        if n_points > MAX_POINT_NUMBER:
            ds_rate = int(round(n_points / float(MAX_POINT_NUMBER)))
            return pcd_crop.uniform_down_sample(ds_rate)
    return pcd_crop


def registration_unif(
    source,
    gt_target,
    init_trans,
    crop_volume,
    threshold,
    max_itr,
    max_size=4 * MAX_POINT_NUMBER,
    verbose=True,
):
    """Refines a transform with ICP on uniformly downsampled, cropped clouds.

    Args:
        source: Source point cloud.
        gt_target: GT point cloud.
        init_trans: Initial 4x4 transform of the source.
        crop_volume: Open3D crop volume.
        threshold: ICP correspondence distance.
        max_itr: Maximum ICP iterations.
        max_size: Unused.
        verbose: Whether to print the threshold and enable Open3D debug
            output.

    Returns:
        The ICP result, with the transformation composed with init_trans.
    """
    del max_size  # Unused.
    if verbose:
        print("[Registration] threshold: %f" % threshold)  # pylint: disable=consider-using-f-string
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    source_down = crop_and_downsample(source,
                                      crop_volume,
                                      down_sample_method="uniform",
                                      trans=init_trans)
    target_down = crop_and_downsample(gt_target,
                                      crop_volume,
                                      down_sample_method="uniform")
    reg = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        threshold,
        np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            1e-6, max_iteration=max_itr),
    )
    reg.transformation = np.matmul(reg.transformation, init_trans)
    return reg


def registration_vol_ds(
    source,
    gt_target,
    init_trans,
    crop_volume,
    voxel_size,
    threshold,
    max_itr,
    verbose=True,
):
    """Refines a transform with ICP on voxel-downsampled, cropped clouds.

    Args:
        source: Source point cloud.
        gt_target: GT point cloud.
        init_trans: Initial 4x4 transform of the source.
        crop_volume: Open3D crop volume.
        voxel_size: Voxel size of the downsampling.
        threshold: ICP correspondence distance.
        max_itr: Maximum ICP iterations.
        verbose: Whether to print the settings and enable Open3D debug
            output.

    Returns:
        The ICP result, with the transformation composed with init_trans.
    """
    if verbose:
        # pylint: disable-next=consider-using-f-string
        print("[Registration] voxel_size: %f, threshold: %f" %
              (voxel_size, threshold))
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    source_down = crop_and_downsample(
        source,
        crop_volume,
        down_sample_method="voxel",
        voxel_size=voxel_size,
        trans=init_trans,
    )
    target_down = crop_and_downsample(
        gt_target,
        crop_volume,
        down_sample_method="voxel",
        voxel_size=voxel_size,
    )
    reg = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        threshold,
        np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            1e-6, max_iteration=max_itr),
    )
    reg.transformation = np.matmul(reg.transformation, init_trans)
    return reg
