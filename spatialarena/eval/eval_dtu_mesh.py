# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

# The evaluation code is adapted from GOF:
#  https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/evaluate_dtu_mesh.py
"""DTU mesh / point-cloud evaluation (Chamfer accuracy and completeness).

`eval` scores one reconstruction against a DTU scan's STL points with the
official observation and ground-plane masks and writes results.json plus two
error visualizations; `mvrecon_evaluator` calls it for DTU. `main` is a
standalone CLI that evaluates every mesh of an experiment folder:

    python -m spatialarena.eval.eval_dtu_mesh \
        --dataset_path ~/datasets \
        --input_dir ~/datasets/dtu/dtu_exp \
        --suffix 2dgs
"""

import argparse
import json
import multiprocessing
import os
import pathlib
import random
import re

import cv2
import numpy as np
import open3d as o3d
# Imported by name: tests patch eval_dtu_mesh.loadmat.
from scipy.io import loadmat
from sklearn import neighbors
import torch
import tqdm
import trimesh

from spatialarena.eval import result_layout


def load_dtu_camera(dtu_path: str) -> np.ndarray:
    """Returns the (64, 3, 4) camera-to-world poses of the DTU calibration.

    Args:
        dtu_path: DTU root with Calibration/cal18/pos_<i>.txt.

    Returns:
        One camera-to-world pose per DTU camera position.
    """
    camtoworlds = []
    for i in range(1, 64 + 1):
        filename = os.path.join(dtu_path, f'Calibration/cal18/pos_{i:03d}.txt')

        projection = np.loadtxt(filename, dtype=np.float32)

        # Decompose projection matrix into pose and camera matrix.
        camera_mat, rot_mat, center = cv2.decomposeProjectionMatrix(
            projection)[:3]
        camera_mat = camera_mat / camera_mat[2, 2]
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rot_mat.transpose()
        pose[:3, 3] = (center[:3] / center[3])[:, 0]
        pose = pose[:3]
        camtoworlds.append(pose)

    camtoworlds = np.stack(camtoworlds, axis=0)

    return camtoworlds


def sample_single_tri(args):
    """Samples a grid of points inside one triangle.

    Args:
        args: (n1, n2, v1, v2, tri_vert): the grid steps along the two edges,
            the two edge vectors and the triangle's first vertex.

    Returns:
        The (M, 3) sampled points.
    """
    n1, n2, v1, v2, tri_vert = args

    coords = np.mgrid[:n1 + 1, :n2 + 1]
    coords += 0.5
    coords[0] /= max(n1, 1e-7)
    coords[1] /= max(n2, 1e-7)
    coords = np.transpose(coords, (1, 2, 0))
    inside = coords[coords.sum(axis=-1) < 1]  # m2
    points = v1 * inside[:, :1] + v2 * inside[:, 1:] + tri_vert

    return points


def write_vis_pcd(file, points, colors):
    """Writes a colored point cloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(file, pcd)


def best_fit_transform(A, B):  # pylint: disable=invalid-name
    """Calculates the least-squares rigid transform that maps A onto B.

    Args:
        A: Nxm array of corresponding points.
        B: Nxm array of corresponding points.

    Returns:
        The (m+1)x(m+1) homogeneous transformation matrix that maps A onto B,
        the mxm rotation matrix and the m translation vector.
    """
    assert A.shape == B.shape

    num_dims = A.shape[1]

    # Translate the points to their centroids.
    centroid_a = np.mean(A, axis=0)
    centroid_b = np.mean(B, axis=0)
    centered_a = A - centroid_a
    centered_b = B - centroid_b

    covariance = np.dot(centered_a.T, centered_b)
    u, _, vt = np.linalg.svd(covariance)
    rotation = np.dot(vt.T, u.T)

    # Special reflection case.
    if np.linalg.det(rotation) < 0:
        vt[num_dims - 1, :] *= -1
        rotation = np.dot(vt.T, u.T)

    translation = centroid_b.T - np.dot(rotation, centroid_a.T)

    transform = np.identity(num_dims + 1)
    transform[:num_dims, :num_dims] = rotation
    transform[:num_dims, num_dims] = translation

    return transform, rotation, translation


def require_dtu_points(points: np.ndarray, stage: str) -> None:
    """Checks that a stage left a non-empty, finite (N, 3) point set.

    Raises:
        ValueError: The points are empty, misshapen or not finite.
    """
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"DTU evaluation has no valid points at {stage}")
    if not np.isfinite(points).all():
        raise ValueError(
            f"DTU evaluation has non-finite coordinates at {stage}")


def truncated_distance_mean(distances: np.ndarray, max_distance: float,
                            direction: str) -> float:
    """Returns the mean of the distances below max_distance.

    Raises:
        ValueError: A distance is not finite, or none is below max_distance.
    """
    if not np.isfinite(distances).all():
        raise ValueError(f"DTU evaluation has non-finite {direction} distances")
    valid = distances[distances < max_distance]
    if valid.size == 0:
        raise ValueError(
            f"DTU evaluation has no {direction} distances below "
            f"max_distance={max_distance}; "
            "check reconstruction alignment and overlap with the evaluation "
            "region")
    return float(valid.mean())


def _sample_mesh_points(tri_vert: np.ndarray,
                        downsample_density: float) -> np.ndarray:
    """Samples points inside every non-degenerate triangle of a mesh.

    Args:
        tri_vert: (T, 3, 3) triangle vertices.
        downsample_density: Grid spacing of the samples.

    Returns:
        The (M, 3) sampled points.
    """
    v1 = tri_vert[:, 1] - tri_vert[:, 0]
    v2 = tri_vert[:, 2] - tri_vert[:, 0]
    l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
    non_zero_area = (area2 > 0)[:, 0]
    l1, l2, area2, v1, v2, tri_vert = [
        arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert]
    ]
    threshold = downsample_density * np.sqrt(l1 * l2 / area2)
    n1 = np.floor(l1 / threshold)
    n2 = np.floor(l2 / threshold)

    args_list = []
    for i in range(len(n1)):
        args_list.append((
            n1[i, 0],
            n2[i, 0],
            v1[i],  # Pass as 1D array
            v2[i],  # Pass as 1D array
            tri_vert[i, 0]  # Pass as 1D array
        ))

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as mp_pool:
        new_pts = mp_pool.map(sample_single_tri, args_list, chunksize=1024)

    return np.concatenate(new_pts, axis=0)


def _write_error_visualizations(output_dir: str, scan_number: int,
                                data_down: np.ndarray, data_indices: np.ndarray,
                                dist_d2s: np.ndarray, stl: np.ndarray,
                                above: np.ndarray, dist_s2d: np.ndarray,
                                max_distance: float,
                                visualization_threshold: float) -> None:
    """Writes vis_<scan>_d2s.ply and vis_<scan>_s2d.ply.

    Points are colored from white (no error) to red (error >=
    visualization_threshold), green beyond max_distance and blue outside
    the masks.
    """
    red = np.array([[1, 0, 0]], dtype=np.float64)
    green = np.array([[0, 1, 0]], dtype=np.float64)
    blue = np.array([[0, 0, 1]], dtype=np.float64)
    white = np.array([[1, 1, 1]], dtype=np.float64)
    data_color = np.tile(blue, (data_down.shape[0], 1))
    data_alpha = dist_d2s.clip(
        max=visualization_threshold) / visualization_threshold
    data_color[data_indices] = red * data_alpha + white * (1 - data_alpha)
    data_color[data_indices[dist_d2s[:, 0] >= max_distance]] = green
    write_vis_pcd(f'{output_dir}/vis_{scan_number:03}_d2s.ply', data_down,
                  data_color)
    stl_color = np.tile(blue, (stl.shape[0], 1))
    stl_alpha = dist_s2d.clip(
        max=visualization_threshold) / visualization_threshold
    stl_color[np.where(above)[0]] = red * stl_alpha + white * (1 - stl_alpha)
    stl_color[np.where(above)[0][dist_s2d[:, 0] >= max_distance]] = green
    write_vis_pcd(f'{output_dir}/vis_{scan_number:03}_s2d.ply', stl, stl_color)


# The name of DTU's official script; callers import it as eval_dtu.
def eval(  # pylint: disable=redefined-builtin
    scan_number: int,
    input_mesh_path: str,
    gt_stl_path: str,
    obj_mask_path: str,
    plane_mask_path: str,
    downsample_density: float = 0.2,
    patch_size: int = 60,
    max_distance: float = 20.0,
    visualization_threshold: float = 10.0,
    eval_modal: str = "mesh",
    output_dir: str = "",
    fscore_threshold: float = 1.0,
):
    """Evaluates one DTU reconstruction and writes results.json.

    Args:
        scan_number: DTU scan number.
        input_mesh_path: The reconstructed mesh (.ply) or point cloud.
        gt_stl_path: The scan's STL point cloud.
        obj_mask_path: The scan's ObsMask .mat file.
        plane_mask_path: The scan's ground-plane .mat file.
        downsample_density: Minimum spacing of the evaluated points (mm).
        patch_size: Margin around the observation bounding box (mm).
        max_distance: Distances at or beyond this are outliers (mm).
        visualization_threshold: Error shown fully red (mm).
        eval_modal: "mesh" (sample the triangles) or "pcd".
        output_dir: Directory of results.json and the visualizations.
        fscore_threshold: Distance threshold of precision/recall (mm).

    Raises:
        ValueError: eval_modal is unknown, a stage leaves no valid points,
            or no distance is below max_distance.
    """
    data_pcd = None
    if eval_modal == 'mesh':
        pbar = tqdm.tqdm(total=9)
        pbar.set_description('Reading data in mesh mode...')

        data_mesh = o3d.io.read_triangle_mesh(input_mesh_path)

        vertices = np.asarray(data_mesh.vertices)
        triangles = np.asarray(data_mesh.triangles)
        tri_vert = vertices[triangles]

        pbar.update(1)
        pbar.set_description('sample pcd from mesh')

        new_pts = _sample_mesh_points(tri_vert, downsample_density)
        data_pcd = np.concatenate([vertices, new_pts], axis=0)

    elif eval_modal == 'pcd':
        pbar = tqdm.tqdm(total=8)
        pbar.set_description('Reading data in point cloud mode...')
        data_pcd_o3d = o3d.io.read_point_cloud(input_mesh_path)
        data_pcd = np.asarray(data_pcd_o3d.points)
    else:
        raise ValueError(f"Unsupported eval_modal: {eval_modal!r}")

    pbar.update(1)
    pbar.set_description('Random Shuffle point cloud index')
    require_dtu_points(data_pcd, 'input prediction')
    shuffle_rng = np.random.default_rng()
    shuffle_rng.shuffle(data_pcd, axis=0)

    pbar.update(1)
    pbar.set_description('Downsample point cloud')
    # sklearn's radius_neighbors (the original per-point greedy-dedup
    # approach) returns, for every query point, an array of *all* neighbor
    # indices within downsample_density -- for a dense point cloud (we
    # sample a fixed 1,000,000 points regardless of the object's physical
    # surface area, see MVReconEvaluationConfig.sampling_number) that result
    # set can reach many tens of GB and OOM-kills the job regardless of how
    # much memory it's given (observed repeatedly across many DTU scans,
    # jobs 131893/132008/132501, even at 192G). Open3D's voxel_down_sample
    # achieves the same "no two points closer than downsample_density"
    # spatial thinning with O(N) memory (one bucket per occupied voxel).
    data_pcd_for_downsample = o3d.geometry.PointCloud()
    data_pcd_for_downsample.points = o3d.utility.Vector3dVector(data_pcd)
    data_down = np.asarray(
        data_pcd_for_downsample.voxel_down_sample(downsample_density).points)
    nn_engine = neighbors.NearestNeighbors(n_neighbors=1,
                                           algorithm='kd_tree',
                                           n_jobs=-1)

    pbar.update(1)
    pbar.set_description('Masking point cloud...')
    obs_mask_file = loadmat(obj_mask_path)
    obj_mask, bounding_box, resolution = [
        obs_mask_file[attr] for attr in ['ObsMask', 'BB', 'Res']
    ]
    bounding_box = bounding_box.astype(np.float32)

    inbound = ((data_down >= bounding_box[:1] - patch_size) &
               (data_down < bounding_box[1:] + patch_size * 2)).sum(
                   axis=-1) == 3
    data_in = data_down[inbound]

    require_dtu_points(data_in, 'bounding-box mask')
    data_grid = np.around(
        (data_in - bounding_box[:1]) / resolution).astype(np.int32)
    grid_inbound = ((data_grid >= 0) &
                    (data_grid < np.expand_dims(obj_mask.shape, 0))).sum(
                        axis=-1) == 3
    data_grid_in = data_grid[grid_inbound]
    in_obs = obj_mask[data_grid_in[:, 0], data_grid_in[:, 1],
                      data_grid_in[:, 2]].astype(np.bool_)
    data_in_obs = data_in[grid_inbound][in_obs]

    pbar.update(1)
    require_dtu_points(data_in_obs, 'observation mask')
    pbar.set_description('Reading STL pcd...')
    stl_pcd = o3d.io.read_point_cloud(gt_stl_path)
    stl = np.asarray(stl_pcd.points)

    pbar.update(1)
    pbar.set_description('Computing data2stl')
    require_dtu_points(stl, 'reference STL')
    nn_engine.fit(stl)
    dist_d2s, _ = nn_engine.kneighbors(data_in_obs,
                                       n_neighbors=1,
                                       return_distance=True)
    mean_d2s = truncated_distance_mean(dist_d2s, max_distance,
                                       'prediction-to-reference')

    pbar.update(1)
    pbar.set_description('compute stl2data')
    ground_plane = loadmat(plane_mask_path)['P']

    stl_hom = np.concatenate([stl, np.ones_like(stl[:, :1])], -1)
    above = (ground_plane.reshape((1, 4)) * stl_hom).sum(-1) > 0
    stl_above = stl[above]

    require_dtu_points(stl_above, 'reference plane mask')
    nn_engine.fit(data_in)
    dist_s2d, _ = nn_engine.kneighbors(stl_above,
                                       n_neighbors=1,
                                       return_distance=True)
    mean_s2d = truncated_distance_mean(dist_s2d, max_distance,
                                       'reference-to-prediction')

    pbar.update(1)
    pbar.set_description('visualize error')
    _write_error_visualizations(
        output_dir,
        scan_number,
        data_down,
        np.where(inbound)[0][grid_inbound][in_obs],
        dist_d2s,
        stl,
        above,
        dist_s2d,
        max_distance,
        visualization_threshold,
    )

    pbar.update(1)
    pbar.set_description('done')
    pbar.close()
    over_all = (mean_d2s + mean_s2d) / 2
    print(mean_d2s, mean_s2d, over_all)

    # DTU's own MVSNet protocol only defines Chamfer accuracy/completeness
    # (mean_d2s/mean_s2d above) -- no precision/recall/F-score. These are an
    # addition for parity with the other MVRecon datasets' metric set
    # (compute_reconstruction_metrics), reusing the same d2s/s2d distances
    # already computed above rather than a second nearest-neighbor pass.
    # fscore_threshold=1.0mm is a convention from MVS/surface-reconstruction
    # papers reporting F-score@1mm on DTU-scale (hundreds-of-mm) scenes; it
    # is not part of DTU's official protocol, so treat it as tunable rather
    # than a fixed ground truth.
    precision = float((dist_d2s < fscore_threshold).mean())
    recall = float((dist_s2d < fscore_threshold).mean())
    if precision + recall <= 0:
        fscore = 0.0
    else:
        fscore = float(2.0 * precision * recall / (precision + recall))

    with open(f'{output_dir}/results.json', 'w', encoding='utf-8') as file:
        json.dump(
            {
                'mean_d2s': mean_d2s,
                'mean_s2d': mean_s2d,
                'overall': over_all,
                'precision': precision,
                'recall': recall,
                'fscore': fscore,
            },
            file,
            indent=True)


def evaluate_mesh(
    scan_id: int,
    input_mesh_path: str,
    gt_stl_path: str,
    obj_mask_path: str,
    plane_mask_path: str,
    output_dir: str,
    gt_camera_centers: np.ndarray,
    train_camera_centers: np.ndarray,
    align_coordinate_frame: bool = False,
):
    """Evaluates a mesh, optionally aligned to DTU's frame by the cameras.

    With align_coordinate_frame the mesh is scaled by the camera spread
    ratio and moved by the rigid camera-center fit, then written to
    recon_aligned.ply in output_dir before `eval` runs on it.

    Args:
        scan_id: DTU scan number.
        input_mesh_path: The reconstructed mesh.
        gt_stl_path: The scan's STL point cloud.
        obj_mask_path: The scan's ObsMask .mat file.
        plane_mask_path: The scan's ground-plane .mat file.
        output_dir: Output directory.
        gt_camera_centers: DTU camera centers.
        train_camera_centers: The reconstruction's camera centers.
        align_coordinate_frame: Whether to align the mesh first.
    """
    mesh = trimesh.load(input_mesh_path)

    if align_coordinate_frame:
        scale_cameras = np.linalg.norm(train_camera_centers -
                                       train_camera_centers.mean(axis=0),
                                       axis=1).mean()
        scale_gt_cameras = np.linalg.norm(gt_camera_centers -
                                          gt_camera_centers.mean(axis=0),
                                          axis=1).mean()

        train_camera_centers = (train_camera_centers * scale_gt_cameras /
                                scale_cameras)
        _, rotation, translation = best_fit_transform(train_camera_centers,
                                                      gt_camera_centers)

        mesh.vertices = mesh.vertices * scale_gt_cameras / scale_cameras
        mesh.vertices = mesh.vertices @ rotation.T + translation

        input_mesh_path = os.path.join(output_dir, "recon_aligned.ply")
        mesh.export(input_mesh_path)

    eval(
        scan_id,
        input_mesh_path,
        gt_stl_path,
        obj_mask_path,
        plane_mask_path,
        eval_modal="mesh",
        output_dir=output_dir,
    )


def main() -> None:
    """Evaluates every DTU mesh of an experiment folder (the standalone CLI).

    See the module docstring for the command line.
    """
    # Legacy dataset/manager dependencies are needed only by the standalone
    # CLI.
    # pylint: disable=import-outside-toplevel
    from spatialarena.eval import dtu_dataset as dtu_dataset_module
    from spatialarena.utils import colmap_manager as colmap_manager_module

    # pylint: enable=import-outside-toplevel

    parser = argparse.ArgumentParser(description="Testing script parameters")
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

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

    dtu_dataset = dtu_dataset_module.DatasetDTU(
        data_path=pathlib.Path(args.dataset_path),
        categories=[],
        scenes=[],
        method="ours",
        run_path=pathlib.Path("./"),
        run_name="eval_mesh",
    )
    scene_infos = dtu_dataset.list_scenes(dataset_folder="dtu",
                                          list_filename="scan_list_eval_3dgs")

    gt_cam_to_worlds = load_dtu_camera(os.path.join(args.dataset_path, "dtu"))
    gt_camera_centers = gt_cam_to_worlds[:, :3, 3]

    pbar = tqdm.trange(len(scene_infos), desc="Evaluating DTU meshes...")
    for scene_info in scene_infos:
        colmap_path = result_layout.resolve_sparse_model_path_from_any(
            scene_info.sparse_gt_path)
        if colmap_path is None:
            raise FileNotFoundError(
                f"No COLMAP model found under {scene_info.sparse_gt_path}")
        image_path = scene_info.image_path
        gt_stl_path = scene_info.dense_gt_path
        obj_mask_path = scene_info.mask_path[0]
        plane_mask_path = scene_info.mask_path[1]
        scene_name = scene_info.scene
        scan_id = int(re.findall(r'\d+', scene_name)[0])

        colmap_manager = colmap_manager_module.ColmapManager(colmap_path,
                                                             image_path,
                                                             load_points=False)
        extrinsics = colmap_manager.get_extrinsics_batch()
        cam_to_worlds = np.linalg.inv(extrinsics)
        camera_centers = cam_to_worlds[:, :3, 3]

        num_used_cameras = camera_centers.shape[0]

        scene_input_dir = os.path.join(args.input_dir,
                                       f"{scene_name}_{args.suffix}")
        mesh_dir = os.path.join(scene_input_dir, "evaluation/meshes")
        input_mesh_path = os.path.join(mesh_dir, "mesh_final.ply")

        evaluate_mesh(
            scan_id,
            input_mesh_path,
            gt_stl_path,
            obj_mask_path,
            plane_mask_path,
            output_dir=mesh_dir,
            gt_camera_centers=gt_camera_centers[:num_used_cameras],
            train_camera_centers=camera_centers,
            align_coordinate_frame=True,
        )

        pbar.update(1)


if __name__ == "__main__":
    main()
