# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

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

# this script requires Open3D python binding
# please follow the instructions in setup.py before running this script.

import os
import argparse
import json
from pathlib import Path

import torch
import trimesh
import numpy as np
import open3d as o3d

from spatialarena.eval.tnt_dataset import DatasetTanksAndTemples  # type: ignore
from spatialarena.eval.tnt_tools.cull_mesh import get_traj, Mesher
from spatialarena.eval.tnt_tools.config import scenes_tau_dict
from spatialarena.eval.tnt_tools.registration import (
    trajectory_alignment,
    registration_vol_ds,
    registration_unif,
)
from spatialarena.eval.tnt_tools.help_func import auto_orient_and_center_poses
from spatialarena.eval.tnt_tools.trajectory_io import CameraPose, read_trajectory
from spatialarena.eval.tnt_tools.evaluation import EvaluateHisto
from spatialarena.eval.tnt_tools.plot import plot_graph


def run_evaluation(dataset_dir, traj_path, ply_path, out_dir, view_crop):
    scene = os.path.basename(os.path.normpath(dataset_dir))

    if scene not in scenes_tau_dict:
        print(dataset_dir, scene)
        raise ValueError("invalid dataset-dir, not in scenes_tau_dict")

    print("")
    print("===========================")
    print(f"Evaluating {scene}")
    print("===========================")

    dTau = scenes_tau_dict[scene]

    # Put the crop-file, the GT file, the COLMAP SfM log file and
    # the alignment of the according scene in a folder of
    # the same scene name in the dataset_dir
    colmap_ref_logfile = os.path.join(dataset_dir, scene + "_COLMAP_SfM.log")

    # This is for ground-truth pointcloud, we can use it
    alignment = os.path.join(dataset_dir, scene + "_trans.txt")
    gt_filen = os.path.join(dataset_dir, scene + ".ply")
    print(f'alignment: {alignment}')

    # This crop file is also w.r.t the ground-truth pointcloud, we can use it.
    # Otherwise we have to crop the estimated pointcloud by ourself
    cropfile = os.path.join(dataset_dir, scene + ".json")

    # This is not so necessary
    map_file = os.path.join(dataset_dir, scene + "_mapping_reference.txt")
    if not os.path.isfile(map_file):
        map_file = None
    map_file = None

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Load reconstruction and according GT
    print(ply_path)
    pcd = o3d.io.read_point_cloud(ply_path)
    # add center points
    
    mesh = trimesh.load_mesh(ply_path)
    # add center points
    sampled_vertices = mesh.vertices[mesh.faces].mean(axis=1)

    vertices = np.concatenate([mesh.vertices, sampled_vertices], axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    # end add center points

    print(gt_filen)
    gt_pcd = o3d.io.read_point_cloud(gt_filen)

    gt_trans = np.loadtxt(alignment)
    print(traj_path)
    traj_to_register = []
    if traj_path.endswith('.npy'):
        ld = np.load(traj_path)
        for i in range(len(ld)):
            traj_to_register.append(CameraPose(meta=None, mat=ld[i]))
    elif traj_path.endswith('.json'):  # instant-npg or sdfstudio format
        with open(traj_path, encoding='UTF-8') as f:
            meta = json.load(f)
        poses_dict = {}
        for i, frame in enumerate(meta['frames']):
            filepath = frame['file_path']
            new_i = int(filepath[13:18]) - 1
            poses_dict[new_i] = np.array(frame['transform_matrix'])
        poses = []
        for i in range(len(poses_dict)):
            poses.append(poses_dict[i])
        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, _ = auto_orient_and_center_poses(
            poses, method='up', center_poses=True)
        scale_factor = 1.0 / float(torch.max(torch.abs(poses[:, :3, 3])))
        poses[:, :3, 3] *= scale_factor
        poses = poses.numpy()
        for i in range(len(poses)):
            traj_to_register.append(CameraPose(meta=None, mat=poses[i]))

    else:
        traj_to_register = read_trajectory(traj_path)

    gt_traj_col = read_trajectory(colmap_ref_logfile)

    trajectory_transform = trajectory_alignment(
        map_file, traj_to_register, gt_traj_col, gt_trans)

    # big point clouds will be downsampled to this number to speed up alignment
    dist_threshold = dTau
    # Refine alignment by using the actual GT and MVS point clouds
    vol = o3d.visualization.read_selection_polygon_volume(cropfile)

    # Registration refinement in 3 iterations
    r2 = registration_vol_ds(pcd, gt_pcd, trajectory_transform, vol, dTau,
                             dTau * 80, 20)
    r3 = registration_vol_ds(pcd, gt_pcd, r2.transformation, vol, dTau / 2.0,
                             dTau * 20, 20)
    r = registration_unif(pcd, gt_pcd, r3.transformation, vol, 2 * dTau, 20)

    trajectory_transform = r.transformation

    # Histograms and P/R/F1
    plot_stretch = 5
    [
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
    ] = EvaluateHisto(
        pcd,
        gt_pcd,
        trajectory_transform,  # r.transformation,
        vol,
        dTau / 2.0,
        dTau,
        out_dir,
        plot_stretch,
        scene,
        view_crop
    )
    eva = [precision, recall, fscore]
    print("==============================")
    print("evaluation result : %s" % scene)
    print("==============================")
    print("distance tau : %.3f" % dTau)
    print("precision : %.4f" % eva[0])
    print("recall : %.4f" % eva[1])
    print("f-score : %.4f" % eva[2])
    print("==============================")

    data = {
        "dTau": dTau,
        "precision": precision,
        "recall": recall,
        "fscore": fscore
    }

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Plotting
    plot_graph(
        scene,
        fscore,
        dist_threshold,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
        plot_stretch,
        out_dir,
    )


def cull_mesh(ply_path: str, traj_path: str):
    estimate_c2w_list = get_traj(traj_path)

    # for Tanks-and-Temples dataset
    H, W = 1080, 1920
    fx = 1163.8678928442187
    fy = 1172.793101201448
    cx = 962.3120628412543
    cy = 542.0667209577691
    far = 20.0

    mesher = Mesher(H, W, fx, fy, cx, cy, far, points_batch_size=5e5)

    mesher(ply_path, estimate_c2w_list)


def main() -> None:
    r"""Evaluate every Tanks and Temples mesh of an experiment folder (the standalone CLI):

    python -m spatialarena.eval.eval_tnt_mesh \
        --dataset_path ~/datasets \
        --input_dir ~/datasets/tanks_and_temples_exp \
        --suffix 2dgs
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_path',
        type=str,
        help='Parent directory of the DTU dataset path'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        help='Parent directory of the reconstructed mesh path'
    )
    parser.add_argument(
        '--suffix',
        type=str,
        help='Suffix for the experiment output folder'
    )
    args = parser.parse_args()

    tnt_dataset = DatasetTanksAndTemples(
        data_path=Path(args.dataset_path),
        categories=[],
        scenes=[],
        method="ours",
        run_path=Path("./"),
        run_name="eval_mesh",
    )
    dataset_name = "tanks_and_temples_gof"
    scene_infos = tnt_dataset.list_scenes(dataset_folder=dataset_name)

    for scene_info in scene_infos:
        scene_name = scene_info.scene
        scene_path = tnt_dataset.data_path / dataset_name / scene_name
        traj_path = scene_path / f"{scene_name}_COLMAP_SfM.log"
        traj_path = str(traj_path)

        scene_input_dir = os.path.join(
            args.input_dir,  f"{scene_name}_{args.suffix}")
        mesh_dir = os.path.join(scene_input_dir, "evaluation/meshes")
        input_mesh_path = os.path.join(mesh_dir, "mesh_final.ply")

        # input_mesh_path = os.path.join(mesh_dir, "mesh_final_cull.ply")
        # if not os.path.exists(input_mesh_path):
        #     cull_mesh(mesh_path, traj_path)

        run_evaluation(
            dataset_dir=str(scene_path),
            traj_path=traj_path,
            ply_path=input_mesh_path,
            out_dir=mesh_dir,
            view_crop=False,
        )


if __name__ == "__main__":
    main()
