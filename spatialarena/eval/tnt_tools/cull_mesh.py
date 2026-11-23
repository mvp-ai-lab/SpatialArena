#!/usr/bin/env python
"""Mesh culling by camera visibility for the Tanks and Temples evaluation.

`eval_tnt_mesh` loads the camera trajectory (`get_traj`) and keeps only the
mesh faces that enough cameras see in front of the rendered depth
(`Mesher`), writing <mesh>_cull.ply.
"""

import copy
import json
import os

import numpy as np
import open3d as o3d
import pyrender
import torch
import torch.nn.functional as F
import trimesh

from spatialarena.eval.tnt_tools import help_func


def extract_depth_from_mesh(  # pylint: disable=invalid-name
    mesh,
    c2w_list,
    H,
    W,
    fx,
    fy,
    cx,
    cy,
    far=20.0,
):
    """Renders the depth of a mesh from each camera.

    Adapted from Go-Surf: https://github.com/JingwenWang95/go-surf

    Args:
        mesh: Trimesh mesh.
        c2w_list: Camera-to-world poses (OpenGL convention), torch tensors.
        H: Image height.
        W: Image width.
        fx: Focal length in x.
        fy: Focal length in y.
        cx: Principal point x.
        cy: Principal point y.
        far: Far clipping plane.

    Returns:
        One depth map (torch tensor) per camera.
    """
    # Allows for GPU-accelerated rendering.
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    scene = pyrender.Scene()
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh)

    camera = pyrender.IntrinsicsCamera(fx=fx,
                                       fy=fy,
                                       cx=cx,
                                       cy=cy,
                                       znear=0.01,
                                       zfar=far)
    camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
    scene.add_node(camera_node)
    renderer = pyrender.OffscreenRenderer(W, H)
    flags = (pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.DEPTH_ONLY |
             pyrender.RenderFlags.SKIP_CULL_FACES)

    depths = []
    for c2w in c2w_list:
        c2w = c2w.detach().numpy()
        # Pyrender uses the OpenGL camera convention, see
        # https://pyrender.readthedocs.io/en/latest/examples/cameras.html;
        # nerfstudio's .json poses are already in OpenGL coordinates, so no
        # conversion from OpenCV is applied.
        c2w_gl = copy.deepcopy(c2w)
        scene.set_pose(camera_node, c2w_gl)
        depth = renderer.render(scene, flags)
        depth = torch.from_numpy(depth)
        depths.append(depth)

    renderer.delete()

    return depths


class Mesher:  # pylint: disable=invalid-name
    """Culls a mesh to the faces the cameras see (adapted from NICE-SLAM).

    Attributes:
        points_batch_size: Maximum points per visibility query batch, to
            bound GPU memory.
        scale: Unused scale, 1.0.
        device: Torch device of the visibility test.
        forecast_radius: Pixel margin of the forecast mask; 0 disables it.
        H: Image height.
        W: Image width.
        fx: Focal length in x.
        fy: Focal length in y.
        cx: Principal point x.
        cy: Principal point y.
        resolution: Unused, 256.
        level_set: Unused, 0.0.
        remove_small_geometry_threshold: Area fraction below which
            `get_connected_mesh` drops a component.
        get_largest_components: Whether `get_connected_mesh` keeps only the
            largest component.
        verbose: Whether to print where the mesh is saved.
    """

    def __init__(self, H, W, fx, fy, cx, cy, far, points_batch_size=5e5):
        """Initializes the mesher with the camera intrinsics.

        Args:
            H: Image height.
            W: Image width.
            fx: Focal length in x.
            fy: Focal length in y.
            cx: Principal point x.
            cy: Principal point y.
            far: Unused far plane (the depth rendering uses 20.0).
            points_batch_size: Maximum points per query batch, to alleviate
                GPU memory usage.
        """
        del far  # Unused.
        self.points_batch_size = int(points_batch_size)
        self.scale = 1.0
        self.device = 'cuda:0'
        self.forecast_radius = 0
        self.H = H
        self.W = W
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.resolution = 256
        self.level_set = 0.0
        self.remove_small_geometry_threshold = 0.2
        self.get_largest_components = True
        self.verbose = True

    @torch.no_grad()
    def point_masks(self, input_points, depth_list, estimate_c2w_list):
        """Splits the input points into seen and forecast points.

        A point is seen when at least 20 cameras see it in their frustum and
        not behind the rendered depth.

        Args:
            input_points: (N, 3) points, a tensor or an array.
            depth_list: Rendered depth per camera.
            estimate_c2w_list: Estimated camera poses.

        Returns:
            The seen mask and the forecast mask, (N,) boolean arrays.
        """
        height = self.H
        width = self.W
        fx = self.fx
        fy = self.fy
        cx = self.cx
        cy = self.cy
        device = self.device
        if not isinstance(input_points, torch.Tensor):
            input_points = torch.from_numpy(input_points)
        input_points = input_points.clone().detach().float()
        mask = []
        forecast_mask = []
        # This eps should be tuned for the scene.
        eps = 0.005
        for _, batch_points in enumerate(
                torch.split(input_points, self.points_batch_size, dim=0)):
            n_pts, _ = batch_points.shape
            valid = torch.zeros(n_pts).to(device).bool()
            valid_num = torch.zeros(n_pts).to(device).int()
            valid_forecast = torch.zeros(n_pts).to(device).bool()
            radius = self.forecast_radius
            for i in range(len(estimate_c2w_list)):
                points = batch_points.to(device).float()
                c2w = estimate_c2w_list[i].to(device).float()
                # The poses are used as is (no OpenGL to OpenCV flip of the
                # nerfstudio .json convention).

                depth = depth_list[i].to(device)
                w2c = torch.inverse(c2w).to(device).float()
                ones = torch.ones_like(points[:, 0]).reshape(-1, 1).to(device)
                homo_points = torch.cat([points, ones],
                                        dim=1).reshape(-1, 4,
                                                       1).to(device).float()
                cam_cord_homo = w2c @ homo_points
                cam_cord = cam_cord_homo[:, :3, :]  # [N, 3, 1]
                intrinsics = np.eye(3)
                intrinsics[0, 0] = fx
                intrinsics[0, 2] = cx
                intrinsics[1, 1] = fy
                intrinsics[1, 2] = cy
                intrinsics = torch.from_numpy(intrinsics).to(device)

                uv = intrinsics.float() @ cam_cord.float()
                z = uv[:, -1:] + 1e-8
                uv = uv[:, :2] / z  # [N, 2, 1]
                u, v = uv[:, 0, 0].float(), uv[:, 1, 0].float()
                z = z[:, 0, 0].float()

                in_frustum = ((u >= 0) & (u <= width - 1) & (v >= 0) &
                              (v <= height - 1) & (z > 0))
                forecast_frustum = ((u >= -radius) & (u <= width - 1 + radius) &
                                    (v >= -radius) &
                                    (v <= height - 1 + radius) & (z > 0))

                depth = depth.reshape(1, 1, height, width)
                vgrid = uv.reshape(1, 1, -1, 2)
                # Normalized to [-1, 1].
                vgrid[..., 0] = (vgrid[..., 0] / (width - 1) * 2.0 - 1.0)
                vgrid[..., 1] = (vgrid[..., 1] / (height - 1) * 2.0 - 1.0)

                depth_sample = F.grid_sample(depth,
                                             vgrid,
                                             padding_mode='border',
                                             align_corners=True)
                depth_sample = depth_sample.reshape(-1)
                is_front_face = torch.where((depth_sample > 0.0),
                                            (z < (depth_sample + eps)),
                                            torch.ones_like(z).bool())
                is_forecast_face = torch.where((depth_sample > 0.0),
                                               (z < (depth_sample + eps)),
                                               torch.ones_like(z).bool())
                in_frustum = in_frustum & is_front_face

                valid = valid | in_frustum.bool()
                valid_num = valid_num + in_frustum.int()

                forecast_frustum = forecast_frustum & is_forecast_face
                forecast_frustum = in_frustum | forecast_frustum
                valid_forecast = valid_forecast | forecast_frustum.bool()
            valid = valid_num >= 20
            mask.append(valid.cpu().numpy())
            forecast_mask.append(valid_forecast.cpu().numpy())

        mask = np.concatenate(mask, axis=0)
        forecast_mask = np.concatenate(forecast_mask, axis=0)

        return mask, forecast_mask

    @torch.no_grad()
    def get_connected_mesh(self, mesh, get_largest_components=False):
        """Keeps the largest component, or the components above the area cut.

        Args:
            mesh: Trimesh mesh.
            get_largest_components: Whether to keep only the largest one.

        Returns:
            The kept components as one mesh.
        """
        print("split")
        components = mesh.split(only_watertight=False)
        print("split completed")
        if get_largest_components:
            areas = np.array([component.area for component in components],
                             dtype=np.float)
            mesh = components[areas.argmax()]
        else:
            new_components = []
            global_area = mesh.area
            for comp in components:
                min_area = self.remove_small_geometry_threshold * global_area
                if comp.area > min_area:
                    new_components.append(comp)
            mesh = trimesh.util.concatenate(new_components)

        return mesh

    @torch.no_grad()
    def cull_mesh(self, mesh, estimate_c2w_list):
        """Culls the faces of a mesh that the cameras do not see.

        Args:
            mesh: Trimesh mesh.
            estimate_c2w_list: Estimated [N, 4, 4] camera poses.

        Returns:
            The culled mesh and the forecast mesh (a copy of the culled mesh
            when forecast_radius is 0).
        """
        step = 1
        print('Start Mesh Culling', end='')

        # Cull with 3D projection.
        print(f' --->> {step}(Projection)', end='')
        forward_depths = extract_depth_from_mesh(mesh,
                                                 estimate_c2w_list,
                                                 H=self.H,
                                                 W=self.W,
                                                 fx=self.fx,
                                                 fy=self.fy,
                                                 cx=self.cx,
                                                 cy=self.cy,
                                                 far=20.0)
        print("after forward depth")
        # Only the forward depths are used; combining them with the depths of
        # the flipped-face (backward) mesh is disabled.
        depth_list = forward_depths

        print("in point masks")
        vertices = mesh.vertices[:, :3]
        mask, forecast_mask = self.point_masks(vertices, depth_list,
                                               estimate_c2w_list)
        print(mask.shape, forecast_mask.shape, mask.mean())

        face_mask = mask[mesh.faces].all(axis=1)
        mesh_with_hole = copy.deepcopy(mesh)
        mesh_with_hole.update_faces(face_mask)
        mesh_with_hole.remove_unreferenced_vertices()
        step += 1

        print("compute componet")
        # Culling by connected components (get_connected_mesh) is disabled.
        print(f' --->> {step}(Component)', end='')
        cull_mesh = mesh_with_hole
        print("after compute componet")
        step += 1

        if abs(self.forecast_radius) > 0:
            # For forecasting.
            print(f' --->> {step}(Forecast:{self.forecast_radius})', end='')
            forecast_face_mask = forecast_mask[mesh.faces].all(axis=1)
            forecast_mesh = copy.deepcopy(mesh)
            forecast_mesh.update_faces(forecast_face_mask)
            forecast_mesh.remove_unreferenced_vertices()

            cull_pc = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(np.array(cull_mesh.vertices)))
            aabb = cull_pc.get_oriented_bounding_box()
            indices = aabb.get_point_indices_within_bounding_box(
                o3d.utility.Vector3dVector(np.array(forecast_mesh.vertices)))
            bound_mask = np.zeros(len(forecast_mesh.vertices))
            bound_mask[indices] = 1.0
            bound_mask = bound_mask.astype(np.bool_)
            forecast_face_mask = bound_mask[forecast_mesh.faces].all(axis=1)
            forecast_mesh.update_faces(forecast_face_mask)
            forecast_mesh.remove_unreferenced_vertices()
            forecast_mesh = self.get_connected_mesh(forecast_mesh,
                                                    self.get_largest_components)
            step += 1
        else:
            forecast_mesh = copy.deepcopy(cull_mesh)

        print(' --->> Done!')

        return cull_mesh, forecast_mesh

    def __call__(self, mesh_path, estimate_c2w_list):
        """Culls the mesh at mesh_path and writes <mesh>_cull.ply."""
        print(f'Loading mesh from {mesh_path}...')
        mesh = trimesh.load(mesh_path, process=True)
        mesh.merge_vertices()

        mesh_out_file = mesh_path.replace('.ply', '_cull.ply')
        cull_mesh, forecast_mesh = self.cull_mesh(
            mesh=mesh,
            estimate_c2w_list=estimate_c2w_list,
        )

        cull_mesh.export(mesh_out_file)
        if self.verbose:
            print(f"\nINFO: Save mesh at {mesh_out_file}!\n")

        torch.cuda.empty_cache()


def read_trajectory(filename):
    """Reads the 4x4 poses of a .log trajectory file."""
    traj = []
    with open(filename, "r") as file:
        metastr = file.readline()
        while metastr:
            metadata = map(int, metastr.split())
            mat = np.zeros(shape=(4, 4))
            for i in range(4):
                matstr = file.readline()
                mat[i, :] = np.fromstring(matstr, dtype=float, sep=" \t")
            traj.append(mat)
            metastr = file.readline()
    return traj


def get_traj(traj_path):
    """Loads a camera trajectory as [4, 4] float tensors.

    Args:
        traj_path: A .npy array of poses, an instant-ngp/sdfstudio .json
            (oriented, centered and scaled to the unit cube), or a .log file.

    Returns:
        The camera-to-world poses.
    """
    print(f'Load trajectory from {traj_path}.')
    traj_to_register = []
    if traj_path.endswith('.npy'):
        loaded = np.load(traj_path)
        for i in range(len(loaded)):
            traj_to_register.append(loaded[i])
    elif traj_path.endswith('.json'):  # instant-npg or sdfstudio format
        with open(traj_path, encoding='UTF-8') as file:
            meta = json.load(file)
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
            traj_to_register.append(poses[i])

    else:
        traj_to_register = read_trajectory(traj_path)

    for i in range(len(traj_to_register)):
        c2w = torch.from_numpy(traj_to_register[i]).float()
        if c2w.shape == (3, 4):
            c2w = torch.cat([c2w, torch.tensor([[0, 0, 0, 1]]).float()], dim=0)
        traj_to_register[i] = c2w  # [4, 4]

    print(f'Trajectory loaded from {traj_path}, including '
          f'{len(traj_to_register)} camera views.')
    return traj_to_register


# if __name__ == "__main__":
#     print('Start culling...')
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--traj-path",
#         type=str,
#         required=True,
#         help="path to trajectory file. See `convert_to_logfile.py` to "
#         "create this file.",
#     )
#     parser.add_argument(
#         "--ply-path",
#         type=str,
#         required=True,
#         help="path to reconstruction ply file",
#     )
#     args = parser.parse_args()

#     estimate_c2w_list = get_traj(args.traj_path)

#     # for TanksandTemples dataset
#     H, W = 1080, 1920
#     fx = 1163.8678928442187
#     fy = 1172.793101201448
#     cx = 962.3120628412543
#     cy = 542.0667209577691
#     far = 20.0

#     mesher = Mesher(H, W, fx, fy, cx, cy, far, points_batch_size=5e5)

#     mesher(args.ply_path, estimate_c2w_list)

#     print('Done!')
