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
"""Tanks and Temples mesh evaluation (precision, recall and F-score).

Adapted from the TanksAndTemples toolbox: aligns a reconstructed mesh to the GT
point cloud through the COLMAP trajectory, refines the alignment by ICP and
writes the metrics and precision/recall plots next to the mesh. Requires the
Open3D Python binding. Run `main` for the standalone CLI.
"""

import argparse
import json
import os
import pathlib

import numpy as np
import open3d as o3d
import torch
import trimesh

from spatialarena.eval import tnt_dataset  # type: ignore
from spatialarena.eval.tnt_tools import config
# Aliased: this module defines its own cull_mesh function.
from spatialarena.eval.tnt_tools import cull_mesh as tnt_cull_mesh
from spatialarena.eval.tnt_tools import evaluation
from spatialarena.eval.tnt_tools import help_func
from spatialarena.eval.tnt_tools import plot
from spatialarena.eval.tnt_tools import registration
from spatialarena.eval.tnt_tools import trajectory_io


def run_evaluation(dataset_dir, traj_path, ply_path, out_dir, view_crop):
    """Evaluates one Tanks and Temples mesh against the scene's GT.

    Args:
        dataset_dir: Scene folder with the GT cloud, crop file, alignment and
            COLMAP SfM log, named after the scene.
        traj_path: Camera trajectory of the reconstruction (.log, .npy or a
            transforms .json).
        ply_path: Reconstructed mesh.
        out_dir: Output folder for metrics.json, error clouds and plots.
        view_crop: Show the cropped reconstruction in an Open3D window.

    Raises:
        ValueError: If the scene has no threshold in `scenes_tau_dict`.
    """
    scene = os.path.basename(os.path.normpath(dataset_dir))

    if scene not in config.scenes_tau_dict:
        print(dataset_dir, scene)
        raise ValueError("invalid dataset-dir, not in scenes_tau_dict")

    print("")
    print("===========================")
    print(f"Evaluating {scene}")
    print("===========================")

    tau = config.scenes_tau_dict[scene]

    # Put the crop file, the GT file, the COLMAP SfM log file and the
    # alignment of the scene in a folder of the same scene name in the
    # dataset_dir.
    colmap_ref_logfile = os.path.join(dataset_dir, scene + "_COLMAP_SfM.log")

    # The alignment of the ground-truth point cloud.
    alignment = os.path.join(dataset_dir, scene + "_trans.txt")
    gt_filen = os.path.join(dataset_dir, scene + ".ply")
    print(f'alignment: {alignment}')

    # This crop file is also w.r.t. the ground-truth point cloud, so we can use
    # it; otherwise we would have to crop the estimated point cloud ourselves.
    cropfile = os.path.join(dataset_dir, scene + ".json")

    # The mapping reference is not needed; it is always disabled below.
    map_file = os.path.join(dataset_dir, scene + "_mapping_reference.txt")
    if not os.path.isfile(map_file):
        map_file = None
    map_file = None

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Load the reconstruction and its GT.
    print(ply_path)
    pcd = o3d.io.read_point_cloud(ply_path)

    mesh = trimesh.load_mesh(ply_path)
    # Add the face center points.
    sampled_vertices = mesh.vertices[mesh.faces].mean(axis=1)

    vertices = np.concatenate([mesh.vertices, sampled_vertices], axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)

    print(gt_filen)
    gt_pcd = o3d.io.read_point_cloud(gt_filen)

    gt_trans = np.loadtxt(alignment)
    print(traj_path)
    traj_to_register = []
    if traj_path.endswith('.npy'):
        loaded_poses = np.load(traj_path)
        for i in range(len(loaded_poses)):
            traj_to_register.append(
                trajectory_io.CameraPose(meta=None, mat=loaded_poses[i]))
    elif traj_path.endswith('.json'):  # Instant-NGP or sdfstudio format.
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
        poses, _ = help_func.auto_orient_and_center_poses(poses,
                                                          method='up',
                                                          center_poses=True)
        scale_factor = 1.0 / float(torch.max(torch.abs(poses[:, :3, 3])))
        poses[:, :3, 3] *= scale_factor
        poses = poses.numpy()
        for i in range(len(poses)):
            traj_to_register.append(
                trajectory_io.CameraPose(meta=None, mat=poses[i]))

    else:
        traj_to_register = trajectory_io.read_trajectory(traj_path)

    gt_traj_col = trajectory_io.read_trajectory(colmap_ref_logfile)

    trajectory_transform = registration.trajectory_alignment(
        map_file, traj_to_register, gt_traj_col, gt_trans)

    # Big point clouds will be downsampled to this number to speed up
    # alignment.
    dist_threshold = tau
    # Refine the alignment by using the actual GT and MVS point clouds.
    vol = o3d.visualization.read_selection_polygon_volume(cropfile)

    # Registration refinement in 3 iterations.
    coarse_registration = registration.registration_vol_ds(
        pcd, gt_pcd, trajectory_transform, vol, tau, tau * 80, 20)
    fine_registration = registration.registration_vol_ds(
        pcd, gt_pcd, coarse_registration.transformation, vol, tau / 2.0,
        tau * 20, 20)
    final_registration = registration.registration_unif(
        pcd, gt_pcd, fine_registration.transformation, vol, 2 * tau, 20)

    trajectory_transform = final_registration.transformation

    # Histograms and P/R/F1.
    plot_stretch = 5
    [
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
    ] = evaluation.EvaluateHisto(
        pcd,
        gt_pcd,
        trajectory_transform,  # r.transformation,
        vol,
        tau / 2.0,
        tau,
        out_dir,
        plot_stretch,
        scene,
        view_crop)
    eva = [precision, recall, fscore]
    print("==============================")
    print(f"evaluation result : {scene}")
    print("==============================")
    print(f"distance tau : {tau:.3f}")
    print(f"precision : {eva[0]:.4f}")
    print(f"recall : {eva[1]:.4f}")
    print(f"f-score : {eva[2]:.4f}")
    print("==============================")

    data = {
        "dTau": tau,
        "precision": precision,
        "recall": recall,
        "fscore": fscore
    }

    with open(os.path.join(out_dir, "metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Plotting.
    plot.plot_graph(
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
    """Culls the mesh at `ply_path` to the views of the trajectory."""
    estimate_c2w_list = tnt_cull_mesh.get_traj(traj_path)

    # Tanks and Temples camera intrinsics.
    height, width = 1080, 1920
    fx = 1163.8678928442187
    fy = 1172.793101201448
    cx = 962.3120628412543
    cy = 542.0667209577691
    far = 20.0

    mesher = tnt_cull_mesh.Mesher(height,
                                  width,
                                  fx,
                                  fy,
                                  cx,
                                  cy,
                                  far,
                                  points_batch_size=5e5)

    mesher(ply_path, estimate_c2w_list)


def main() -> None:
    r"""Evaluates every Tanks and Temples mesh of an experiment folder.

    This is the standalone CLI:

    python -m spatialarena.eval.eval_tnt_mesh \
        --dataset_path ~/datasets \
        --input_dir ~/datasets/tanks_and_temples_exp \
        --suffix 2dgs
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path',
                        type=str,
                        help='Parent directory of the DTU dataset path')
    parser.add_argument('--input_dir',
                        type=str,
                        help='Parent directory of the reconstructed mesh path')
    parser.add_argument('--suffix',
                        type=str,
                        help='Suffix for the experiment output folder')
    args = parser.parse_args()

    dataset = tnt_dataset.DatasetTanksAndTemples(
        data_path=pathlib.Path(args.dataset_path),
        categories=[],
        scenes=[],
        method="ours",
        run_path=pathlib.Path("./"),
        run_name="eval_mesh",
    )
    dataset_name = "tanks_and_temples_gof"
    scene_infos = dataset.list_scenes(dataset_folder=dataset_name)

    for scene_info in scene_infos:
        scene_name = scene_info.scene
        scene_path = dataset.data_path / dataset_name / scene_name
        traj_path = scene_path / f"{scene_name}_COLMAP_SfM.log"
        traj_path = str(traj_path)

        scene_input_dir = os.path.join(args.input_dir,
                                       f"{scene_name}_{args.suffix}")
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
