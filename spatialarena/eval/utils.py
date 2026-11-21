# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

# pylint: disable=[E1101,E1135]

import os
import argparse
import dataclasses
import datetime
import functools
import multiprocessing
import json
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, List

import torch
import numpy as np
import pycolmap
import open3d as o3d
from scipy.spatial import cKDTree as KDTree

from spatialarena.geometry.align_poses import align_ate_c2b_use_a2b
from spatialarena.eval.result_layout import resolve_sparse_model_path


def trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(y, x=x))


@dataclasses.dataclass(kw_only=True)
class SceneInfo:
    # Dataset name.
    dataset: str
    # Category name.
    category: str
    # Scene name.
    scene: str
    # Path to the workspace directory in the run directory.
    workspace_path: Path
    # Path to the input images.
    image_path: Path
    # Path to the ground-truth sparse reconstruction.
    sparse_gt_path: Path
    # Path to the ground-truth dense reconstruction.
    dense_gt_path: Path
    # Path to the ground-truth depth maps.
    depth_gt_path: Path
    # Path to the masks.
    mask_path: List[Path]
    # Whether to update camera priors from the ground-truth reconstruction.
    camera_priors_from_sparse_gt: bool
    # Additional arguments for the COLMAP reconstruction command.
    colmap_extra_args: list[str]


@dataclasses.dataclass(kw_only=True)
class SceneResult:
    # Scene information for which the result was computed.
    scene_info: SceneInfo
    # Flat list of errors.
    errors: Tuple[np.ndarray, np.ndarray]
    # # Accuracy metrics.
    # accuracy: float
    # completeness: float
    # chamfer: float
    # Number of images in the scene.
    num_images: int
    # Number of registered images in the scene (over all components).
    num_reg_images: int
    # Number of components in the scene.
    num_components: int
    # Number of images in the largest component.
    largest_component: int


@dataclasses.dataclass(kw_only=True)
class SceneMetrics:
    # Area under the curve (AUC) scores at specified error thresholds.
    aucs: np.ndarray
    error_thresholds: np.ndarray
    error_type: str
    # Number of images in the scene.
    num_images: int
    # Number of registered images in the scene (over all components).
    num_reg_images: int
    # Number of components in the scene.
    num_components: int
    # Number of images in the largest component.
    largest_component: int


@dataclasses.dataclass(kw_only=True)
class Sim3:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray


class Dataset(ABC):
    @property
    @abstractmethod
    def position_accuracy_gt(self) -> float:
        """Ground-truth position accuracy in meters."""

    @property
    @abstractmethod
    def rotation_accuracy_gt(self) -> float:
        """Ground-truth rotation accuracy in degrees."""

    @abstractmethod
    def list_scenes(self) -> list[SceneInfo]:
        """List all scenes to evaluate."""

    @abstractmethod
    def prepare_scene(self, scene_info: SceneInfo) -> None:
        """Prepare the scene for reconstruction."""


class PointCloudMerger:
    def __init__(self, mlp_file_path):
        self.mlp_file_path = mlp_file_path
        self.merged_pcd = o3d.geometry.PointCloud()

    def merge_all_point_clouds(self, output_path: str = None):
        """Merge all point clouds from the project"""
        tree = ET.parse(self.mlp_file_path)
        root = tree.getroot()
        base_dir = os.path.dirname(self.mlp_file_path)

        all_points = []
        all_colors = []
        all_normals = []

        for i, mlmesh in enumerate(root.findall('.//MLMesh')):
            filename = mlmesh.get('filename')
            file_path = os.path.join(base_dir, filename)

            print(
                f"Processing {i+1}/{len(root.findall('.//MLMesh'))}: {filename}")

            try:
                pcd = o3d.io.read_point_cloud(file_path)
                transform = self._get_transform(mlmesh)

                if len(pcd.points) > 0:
                    points, colors, normals = self.transform_point_cloud(
                        pcd, transform)
                    all_points.extend(points)

                    if colors is not None:
                        all_colors.extend(colors)

                    if normals is not None:
                        all_normals.extend(normals)

                    print(f"  Added {len(points)} points")
                else:
                    print("  No points found")

            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"Error processing {filename}: {e}")

        # Create final merged point cloud
        self._create_merged_pcd(all_points, all_colors, all_normals)

        if output_path:
            o3d.io.write_point_cloud(output_path, self.merged_pcd)
            print(f"Saved to: {output_path}")

        return self.merged_pcd

    def _get_transform(self, mlmesh_element):
        """Extract transformation matrix"""
        matrix_elem = mlmesh_element.find('MLMatrix44')
        if matrix_elem is not None and matrix_elem.text:
            matrix_text = matrix_elem.text.strip()
            values = [float(x) for x in matrix_text.split()]
            return np.array(values).reshape(4, 4)
        return np.eye(4)

    @classmethod
    def transform_point_cloud(cls, pcd, transform):
        """Apply transformation to point cloud and return points, colors, normals"""
        points = np.asarray(pcd.points)

        # Transform points
        points_homogeneous = np.column_stack([points, np.ones(len(points))])
        transformed_points = (transform @ points_homogeneous.T).T[:, :3]

        # Get colors if available
        colors = None
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)

        # Transform normals if available
        normals = None
        if pcd.has_normals():
            rotation_matrix = transform[:3, :3]
            original_normals = np.asarray(pcd.normals)
            transformed_normals = (rotation_matrix @ original_normals.T).T
            # Normalize normals
            norm_magnitudes = np.linalg.norm(
                transformed_normals, axis=1, keepdims=True)
            norm_magnitudes[norm_magnitudes == 0] = 1  # Avoid division by zero
            normals = transformed_normals / norm_magnitudes

        return transformed_points, colors, normals

    def _create_merged_pcd(self, points, colors, normals):
        """Create the final merged point cloud"""
        self.merged_pcd.points = o3d.utility.Vector3dVector(np.array(points))

        if colors and len(colors) == len(points):
            self.merged_pcd.colors = o3d.utility.Vector3dVector(
                np.array(colors))

        if normals and len(normals) == len(points):
            self.merged_pcd.normals = o3d.utility.Vector3dVector(
                np.array(normals))

        print(
            f"Final merged point cloud: {len(self.merged_pcd.points)} points")


def parse_args() -> argparse.Namespace:
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", default=Path(__file__).parent / "data", type=Path
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["eth3d", "dtu"],
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=[],
        help="Categories to evaluate, if empty all categories are evaluated.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=[],
        help="Scenes to evaluate, if empty all scenes are evaluated.",
    )
    parser.add_argument(
        "--eval_methods",
        nargs="+",
        default=["cut3r", "depthanything3", "fast-vggt", "mapanything", "ttt3r", "vggt"],
    )
    parser.add_argument(
        "--run_path", default=Path(__file__).parent / "runs", type=Path
    )
    parser.add_argument("--run_name", default=datetime_str)
    parser.add_argument("--report_name", default=f"report-{datetime_str}")
    parser.add_argument(
        "--overwrite_database", default=False, action="store_true"
    )
    parser.add_argument(
        "--overwrite_matches", default=False, action="store_true"
    )
    parser.add_argument(
        "--overwrite_reconstruction", default=False, action="store_true"
    )
    parser.add_argument(
        "--overwrite_alignment", default=False, action="store_true"
    )
    parser.add_argument("--colmap_path", required=True)
    parser.add_argument("--use_gpu", default=True, action="store_true")
    parser.add_argument("--use_cpu", dest="use_gpu", action="store_false")
    parser.add_argument(
        "--parallelism",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of processes for parallel reconstruction.",
    )
    parser.add_argument("--quality", default="high")
    parser.add_argument(
        "--error_type",
        default="relative",
        choices=["relative", "absolute"],
        help="Whether to evaluate relative pairwise pose errors in angular "
        "distance or absolute pose errors through GT alignment.",
    )
    parser.add_argument(
        "--rel_error_thresholds",
        type=float,
        nargs="+",
        default=[0.5, 1, 5, 10],
        help="Evaluation thresholds in degrees.",
    )
    parser.add_argument(
        "--abs_pos_error_thresholds",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.2, 0.5],
        help="Evaluation thresholds in meters.",
    )
    parser.add_argument(
        "--abs_rot_error_thresholds",
        type=float,
        nargs="+",
        default=[3, 5, 10, 30],
        help="Evaluation thresholds in meters.",
    )
    args = parser.parse_args()
    args.colmap_path = Path(args.colmap_path).resolve()
    if args.overwrite_database:
        pycolmap.logging.info(
            "Overwriting database also overwrites reconstruction"
        )
        args.overwrite_reconstruction = True
    if args.overwrite_matches:
        pycolmap.logging.info(
            "Overwriting matches also overwrites reconstruction"
        )
        args.overwrite_reconstruction = True
    if args.overwrite_reconstruction:
        pycolmap.logging.info(
            "Overwriting reconstruction also overwrites alignment"
        )
        args.overwrite_alignment = True
    return args


def retrieve_poses_from_reconstruction(
    sparse: pycolmap.Reconstruction
) -> dict[str, np.ndarray]:
    # sparse = pycolmap.Reconstruction(sparse_path)
    poses = {}
    for image in sparse.images.values():
        pose = np.eye(4)
        pose[:3, :4] = image.cam_from_world().matrix()
        poses[image.name] = pose
    return poses


def rotation_distance(R1, R2, eps=1e-7):
    """
    Args:
        R1: rotation matrix from camera 1 to world
        R2: rotation matrix from camera 2 to world
    Return:
        angle: the angular distance between camera 1 and camera 2.
    """

    if isinstance(R1, np.ndarray):
        R1 = torch.from_numpy(R1).float()

    if isinstance(R2, np.ndarray):
        R2 = torch.from_numpy(R2).float()

    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    # R_diff = R1 @ R2.transpose(-2, -1)
    R_diff = R1.transpose(-2, -1) @ R2
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]

    # numerical stability near -1/+1
    angle = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps).acos_()
    angle = np.rad2deg(angle.numpy())

    return angle


def process_scene(
    args: argparse.Namespace,
    scene_info: SceneInfo,
    prepare_scene: callable,
    position_accuracy_gt: float,
    num_threads: int,  # pylint: disable=unused-argument
) -> SceneResult:
    pycolmap.logging.info(
        f"Processing dataset={scene_info.dataset}, "
        f"category={scene_info.category}, "
        f"scene={scene_info.scene}"
    )

    prepare_scene(scene_info)

    sparse_gt = pycolmap.Reconstruction(scene_info.sparse_gt_path)

    # Merge all sub-models into a single reconstruction. Each sub-model will be
    # "randomly" aligned to the other sub-models. We then compute the overall
    # error over the merged reconstruction. With this simple approach, there is
    # a small chance that the randomly aligned images in one sub-model are
    # correctly aligned with other sub-models and the error is therefore
    # underestimated. However, this is very unlikely to happen.
    sparse_merged = pycolmap.Reconstruction()
    num_components = 0
    largest_component = 0
    sparse_path = resolve_sparse_model_path(scene_info.workspace_path)
    if sparse_path is None:
        print(f'[ERROR] No COLMAP model found in {scene_info.workspace_path}/sparse or sparse/0')
        return None

    num_components += 1
    sparse = pycolmap.Reconstruction(sparse_path)

    if sparse is not None:
        largest_component = max(largest_component, sparse.num_images())
        for image in sparse.images.values():
            if image.image_id in sparse_merged.images:
                continue
            if image.camera_id not in sparse_merged.cameras:
                sparse_merged.add_camera(image.camera)
            if image.frame_id not in sparse_merged.frames:
                if image.frame.rig_id not in sparse_merged.rigs:
                    sparse_merged.add_rig(image.frame.rig)
                image.frame.reset_rig_ptr()
                sparse_merged.add_frame(image.frame)
            image.reset_camera_ptr()
            image.reset_frame_ptr()
            sparse_merged.add_image(image)

    sim3 = Sim3(scale=1.0, rotation=np.eye(3), translation=np.zeros(3))
    if args.error_type == "relative":
        dts, dRs = compute_rel_errors(
            sparse_gt=sparse_gt,
            sparse=sparse_merged,
            min_proj_center_dist=position_accuracy_gt,
        )
        errors = np.maximum(dts, dRs)
    elif args.error_type == "absolute":
        dts, dRs, sim3 = compute_abs_errors(
            sparse_gt=sparse_gt,
            sparse=sparse_merged,
        )
        errors = (dts, dRs)
    else:
        raise ValueError(f"Invalid error type: {args.error_type}")

    reconstruction_ply = scene_info.workspace_path / "sparse" / "points3D.ply"
    if not reconstruction_ply.exists():
        print(f'Exporting reconstruction to {str(reconstruction_ply)}')
        sparse.export_PLY(reconstruction_ply)
    transform = np.eye(4)
    transform[:3, :3] = sim3.scale * sim3.rotation
    transform[:3, 3:] = sim3.translation
    pcd = o3d.io.read_point_cloud(str(reconstruction_ply))

    if scene_info.dataset == "eth3d" and scene_info.dense_gt_path is not None:
        scan_path = scene_info.dense_gt_path / "scan_alignment.mlp"
        merged_scan_path = scene_info.dense_gt_path / "scan_alignment_merged.ply"
        if not merged_scan_path.exists():
            merger = PointCloudMerger(scan_path)
            merger.merge_all_point_clouds(output_path=str(merged_scan_path))
    else:
        gt_ply_path = scene_info.sparse_gt_path / "points3D.ply"
        if not reconstruction_ply.exists():
            print(f'Exporting reconstruction to {str(gt_ply_path)}')
            sparse_gt.export_PLY(reconstruction_ply)

    pcd = pcd.transform(transform)
    transformed_ply_path = scene_info.workspace_path / \
        "sparse" / "points3D_aligned.ply"
    o3d.io.write_point_cloud(str(transformed_ply_path), pcd)

    return SceneResult(
        scene_info=scene_info,
        errors=errors,
        num_images=sparse_gt.num_images(),
        num_reg_images=sparse_merged.num_images(),
        num_components=num_components,
        largest_component=largest_component,
    )


def process_scenes(
    args: argparse.Namespace,
    scene_infos: list[SceneInfo],
    prepare_scene: callable,
    position_accuracy_gt: float,
    rotation_accuracy_gt: float,
) -> dict[str, dict[str, SceneMetrics]]:
    pos_error_thresholds, rot_error_thresholds = get_error_thresholds(args)

    num_threads = min(
        args.parallelism, 2 * max(1, int(args.parallelism / len(scene_infos)))
    )
    with multiprocessing.Pool(processes=args.parallelism) as p:
        results = p.map(
            functools.partial(
                process_scene,
                args,
                prepare_scene=prepare_scene,
                position_accuracy_gt=position_accuracy_gt,
                num_threads=num_threads,
            ),
            scene_infos,
        )

    for result in results:
        if result is None:
            continue
        print(f'Processing workspace: {result.scene_info.workspace_path}')
        dts, dRs = result.errors[0], result.errors[1]
        pos_aucs = compute_auc(
            dts, pos_error_thresholds,  min_error=position_accuracy_gt,
        )
        rot_aucs = compute_auc(
            dRs, rot_error_thresholds, min_error=rotation_accuracy_gt
        )
        avg_rotation_error = np.mean(
            result.errors[1][np.isfinite(result.errors[1])]
        ).item()
        avg_translation_error = np.mean(
            result.errors[0][np.isfinite(result.errors[0])]
        ).item()

        ply_path = result.scene_info.workspace_path / "sparse" / "points3D_aligned.ply"
        gt_ply_path = result.scene_info.sparse_gt_path / "points3D.ply"

        if ply_path.exists() and gt_ply_path.exists():
            pcd_gt = o3d.io.read_point_cloud(str(gt_ply_path))
            pcd = o3d.io.read_point_cloud(str(ply_path))

            # pcd_down = pcd.voxel_down_sample(0.05)
            # pcd_gt_down = pcd_gt.voxel_down_sample(0.05)
            pcd.estimate_normals()
            pcd_gt.estimate_normals()

            print('Running ICP alignment between reconstruction and GT scan...')
            threshold = 100 if result.scene_info.dataset.find("dtu") >= 0 else 0.1
            trans_init = np.eye(4)
            reg_p2p = o3d.pipelines.registration.registration_icp(
                pcd,
                pcd_gt,
                threshold,
                trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=1000)
            )
            pcd = pcd.transform(reg_p2p.transformation)
            pcd.estimate_normals()
            # pcd_gt.estimate_normals()

            gt_points = np.asarray(pcd_gt.points)
            pred_points = np.asarray(pcd.points)
            finite_mask = np.all(np.isfinite(pred_points), axis=1)
            pred_points = pred_points[finite_mask]

            gt_normal = np.asarray(pcd_gt.normals)
            pred_normal = np.asarray(pcd.normals)[finite_mask]

            acc = accuracy(gt_points, pred_points, gt_normal, pred_normal)[0]
            comp = completion(gt_points, pred_points, gt_normal, pred_normal)[0]
            chamfer = (acc + comp) / 2
        else:
            acc, comp, chamfer = 0, 0, 0

        metrics = {
            "iteration": 0,
            "rotation error": avg_rotation_error,
            "translation error": avg_translation_error,
            "accuracy": acc,
            "completeness": comp,
            "chamfer distance": chamfer,
            "depth accuracy": 0,
            "depth completeness": 0,
            "depth chamfer distance": 0,
            "psnr": 0,
            "ssim": 0,
            "lpips": 0,
        }

        for auc, error_thresh in zip(pos_aucs, pos_error_thresholds):
            metrics[f"auc_pos_{error_thresh}"] = auc

        for auc, error_thresh in zip(rot_aucs, rot_error_thresholds):
            metrics[f"auc_rot_{error_thresh}"] = auc

        # print(metrics)
        all_metrics = {"0": metrics}

        metrics_path = result.scene_info.workspace_path / "acc_metrics.json"
        print(f'Saving metrics to {metrics_path}')
        json_obj = json.dumps(all_metrics, indent=4)
        with open(metrics_path, 'w', encoding='utf-8') as json_file:
            json_file.write(json_obj)


def normalize_vec(vec: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    return vec / max(eps, np.linalg.norm(vec))


def vec_angular_dist_deg(vec1: np.ndarray, vec2: np.ndarray) -> float:
    cos_dist = np.clip(np.dot(normalize_vec(vec1), normalize_vec(vec2)), -1, 1)
    return np.rad2deg(np.acos(cos_dist))


def get_error_thresholds(args: argparse.Namespace) -> list[float]:
    if args.error_type == "relative":
        return np.array(args.rel_error_thresholds)
    elif args.error_type == "absolute":
        return (
            np.array(args.abs_pos_error_thresholds),
            np.array(args.abs_rot_error_thresholds)
        )
    else:
        raise ValueError(f"Invalid error type: {args.error_type}")


def compute_rel_errors(
    sparse_gt: pycolmap.Reconstruction,
    sparse: pycolmap.Reconstruction,
    min_proj_center_dist: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes angular relative pose errors across all image pairs.

    Notice that this approach leads to a super-linear decrease in the AUC scores
    when multiple images fail to register. Consider that we have N images in
    total in a dataset and M images are registered in the evaluated
    reconstruction. In this case, we can compute "finite" errors for (N-M)^2
    pairs while the dataset has a total of N^2 pairs. In case of many
    unregistered images, the AUC score will drop much more than the
    (intuitively) expected (N-M) / N ratio. One could appropriately normalize by
    computing a single score per image through a suitable normalization of all
    pairwise errors per image. However, this becomes difficult when multiple
    sub-components are incorrectly stitched together in the same reconstruction
    (e.g., in the case of symmetry issues).
    """

    if sparse is None:
        pycolmap.logging.error("Reconstruction failed")
        return len(sparse_gt.images) * [np.inf], len(sparse_gt.images) * [180]

    images = {}
    for image in sparse.images.values():
        images[image.name] = image

    dts = []
    dRs = []
    for this_image_gt in sparse_gt.images.values():
        if this_image_gt.name not in images:
            for _ in range(sparse_gt.num_images() - 1):
                dts.append(np.inf)
                dRs.append(180)
            continue

        this_image = images[this_image_gt.name]

        for other_image_gt in sparse_gt.images.values():
            if this_image_gt.image_id == other_image_gt.image_id:
                continue

            if other_image_gt.name not in images:
                dts.append(np.inf)
                dRs.append(180)
                continue

            other_image = images[other_image_gt.name]

            other_from_this = (
                other_image.cam_from_world()
                * this_image.cam_from_world().inverse()
            )
            other_from_this_gt = (
                other_image_gt.cam_from_world()
                * this_image_gt.cam_from_world().inverse()
            )

            estimated_from_gt = other_from_this.inverse() * other_from_this_gt

            if (
                np.linalg.norm(other_from_this_gt.translation)
                < min_proj_center_dist
            ):
                # If the cameras almost coincide, then the angular direction
                # distance is unstable, because a small position change can
                # cause a large rotational error. In this case, we only measure
                # rotational relative pose error.
                dt = 0
            else:
                dt = vec_angular_dist_deg(
                    other_from_this.translation, other_from_this_gt.translation
                )

            dR = np.rad2deg(estimated_from_gt.rotation.angle())

            dts.append(dt)
            dRs.append(dR)

    return np.array(dts), np.array(dRs)


def compute_abs_errors(
    sparse_gt: pycolmap.Reconstruction, sparse: pycolmap.Reconstruction
) -> tuple[np.ndarray, np.ndarray]:
    """Computes rotational and translational absolute pose errors.

    Assumes that the input reconstructions are aligned in the same coordinate
    system. Computes one error per ground-truth image.
    """

    sparse_name_to_pose = retrieve_poses_from_reconstruction(sparse)
    sparse_gt_name_to_pose = retrieve_poses_from_reconstruction(sparse_gt)
    assert len(sparse_name_to_pose) == len(sparse_gt_name_to_pose), \
        "Number of images in sparse and sparse_gt must be the same!"

    sparse_pose, sparse_gt_pose = [], []
    for name, pose in sparse_name_to_pose.items():
        if name not in sparse_gt_name_to_pose:
            pycolmap.logging.error(
                f"Cannot align models, image {name} not found in GT"
            )
            return

        cam_to_world = np.linalg.inv(pose)
        cam_to_world_gt = np.linalg.inv(sparse_gt_name_to_pose[name])
        sparse_pose.append(cam_to_world)
        sparse_gt_pose.append(cam_to_world_gt)
    sparse_pose = np.stack(sparse_pose, axis=0)
    sparse_gt_pose = np.stack(sparse_gt_pose, axis=0)

    aligned_pose, scale, rotation, t = align_ate_c2b_use_a2b(
        sparse_pose,
        sparse_gt_pose,
    )

    R_aligned, t_aligned = aligned_pose[...,
                                        :3, :3], aligned_pose[..., :3, 3:4]
    R_gt, t_gt = sparse_gt_pose[..., :3, :3], sparse_gt_pose[..., :3, 3:4]

    dRs = rotation_distance(R_aligned[..., :3, :3], R_gt[..., :3, :3])
    dts = np.linalg.norm((t_aligned - t_gt)[..., 0], axis=-1)

    sim3 = Sim3(scale=scale, rotation=rotation, translation=t)

    return dts, dRs, sim3


def compute_recall(errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    num_elements = len(errors)
    errors = np.sort(errors)
    recall = (np.arange(num_elements) + 1) / num_elements
    return errors, recall


def compute_auc(
    errors: np.ndarray, thresholds: list[float], min_error: float = 0
) -> list[float]:
    if len(errors) == 0:
        raise ValueError("No errors to evaluate")

    errors, recall = compute_recall(errors)

    if min_error > 0:
        min_index = np.searchsorted(errors, min_error, side="right")
        min_score = min_index / len(errors)
        recall = np.r_[min_score, min_score, recall[min_index:]]
        errors = np.r_[0, min_error, errors[min_index:]]
    else:
        recall = np.r_[0, recall]
        errors = np.r_[0, errors]

    aucs = np.zeros(len(thresholds), dtype=np.float64)
    for i, t in enumerate(thresholds):
        last_index = np.searchsorted(errors, t, side="right")
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        auc = trapezoid_integral(r, e) / t
        aucs[i] = auc * 100
    return aucs / 1.1


def accuracy(
    gt_points: np.ndarray,
    rec_points: np.ndarray,
    gt_normals: np.ndarray = None,
    rec_normals: np.ndarray = None,
):
    # # Filter out non-finite points before KDTree
    # finite_mask = np.all(np.isfinite(rec_points), axis=1)
    # rec_points = rec_points[finite_mask]

    gt_points_kd_tree = KDTree(gt_points)
    distances, idx = gt_points_kd_tree.query(rec_points, workers=-1)
    acc = np.mean(distances)

    acc_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals[idx] * rec_normals, axis=-1)
        normal_dot = np.abs(normal_dot)

        return acc, acc_median, np.mean(normal_dot), np.median(normal_dot)

    return acc, acc_median


def completion(
    gt_points: np.ndarray,
    rec_points: np.ndarray,
    gt_normals: np.ndarray = None,
    rec_normals: np.ndarray = None,
):
    # # Filter out non-finite points before KDTree
    # finite_mask = np.all(np.isfinite(rec_points), axis=1)
    # rec_points = rec_points[finite_mask]

    gt_points_kd_tree = KDTree(rec_points)
    distances, idx = gt_points_kd_tree.query(gt_points, workers=-1)
    comp = np.mean(distances)
    comp_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals * rec_normals[idx], axis=-1)
        normal_dot = np.abs(normal_dot)

        return comp, comp_median, np.mean(normal_dot), np.median(normal_dot)

    return comp, comp_median


def randomly_limit_trues(mask: np.ndarray, max_trues: int) -> np.ndarray:
    """
    If mask has more than max_trues True values,
    randomly keep only max_trues of them and set the rest to False.
    """
    # 1D positions of all True entries
    true_indices = np.flatnonzero(mask)  # shape = (N_true,)

    # if already within budget, return as-is
    if true_indices.size <= max_trues:
        return mask

    # randomly pick which True positions to keep
    sampled_indices = np.random.choice(
        true_indices, size=max_trues, replace=False
    )  # shape = (max_trues,)

    # build new flat mask: True only at sampled positions
    limited_flat_mask = np.zeros(mask.size, dtype=bool)
    limited_flat_mask[sampled_indices] = True

    # restore original shape
    return limited_flat_mask.reshape(mask.shape)
