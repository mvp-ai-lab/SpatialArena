# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import os
import struct
import numpy as np
from collections import OrderedDict, namedtuple
from typing import Dict, Tuple, Optional, Union, List
import warnings

from spatialarena.geometry.rotation import qvec2rotmat


# Camera model definitions
CameraModel = namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)

Camera = namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]
)

Image = namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)

Point3D = namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict([(cm.model_id, cm) for cm in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(cm.model_name, cm) for cm in CAMERA_MODELS])


class ColmapManager:
    """
    A class to manage COLMAP reconstruction data, providing easy access to cameras,
    images, and 3D points.
    """
    
    INVALID_POINT3D = -1
    
    def __init__(self, model_path: str, image_path: Optional[str] = None, 
                 load_points: bool = True, sort_images: bool = True):
        """
        Initialize the COLMAP manager.
        
        Args:
            model_path: Path to the folder containing COLMAP output files
            image_path: Optional path to the folder containing original images.
                       If provided, can be used to construct absolute image paths.
            load_points: Whether to load 3D points (can be memory-intensive)
            sort_images: Whether to maintain images in sorted order by name
        """
        self.model_path = model_path
        self.image_path = image_path
        self.load_points = load_points
        self.sort_images = sort_images
        
        # Data structures
        self.cameras: Dict[int, Camera] = {}
        self.images: Dict[int, Image] = {}
        self.points3D: Dict[int, Point3D] = {}
        
        # Lookup maps
        self.name_to_image_id: Dict[str, int] = {}
        self.image_id_to_name: Dict[int, str] = {}
        
        # Sorted image names for consistent ordering
        self.sorted_image_names: List[str] = []
        
        # Numpy arrays for efficient access
        self.points3D_xyz: Optional[np.ndarray] = None
        self.points3D_rgb: Optional[np.ndarray] = None
        self.points3D_ids: Optional[np.ndarray] = None
        
        # Camera parameters for each image (extrinsics)
        self.image_rotation_matrices: Dict[int, np.ndarray] = {}
        self.image_translation_vectors: Dict[int, np.ndarray] = {}
        
        # Convenience arrays for batched operations (aligned with sorted_image_names)
        self.extrinsics_matrices: Optional[np.ndarray] = None  # N x 4 x 4
        self.intrinsics_matrices: Optional[np.ndarray] = None  # N x 3 x 3
        self.intrinsics_params: Optional[np.ndarray] = None    # N x num_params
        
        # Load the data
        self._load()
        
        # Prepare convenience arrays
        self._prepare_convenience_arrays()
    
    def _load(self):
        """Load all COLMAP data files."""
        # Determine format
        if self._detect_format(".bin"):
            ext = ".bin"
        elif self._detect_format(".txt"):
            ext = ".txt"
        else:
            raise FileNotFoundError(f"No COLMAP files found in {self.model_path}")
        
        # Load cameras
        self._load_cameras(ext)
        
        # Load images
        self._load_images(ext)
        
        # Create sorted image names list
        if self.sort_images:
            self.sorted_image_names = sorted(self.name_to_image_id.keys())
        else:
            # Preserve order as loaded from files
            self.sorted_image_names = list(self.name_to_image_id.keys())
        
        # Load points if requested
        if self.load_points:
            self._load_points3D(ext)
    
    def _detect_format(self, ext: str) -> bool:
        """Check if files with given extension exist."""
        cameras_file = os.path.join(self.model_path, f"cameras{ext}")
        images_file = os.path.join(self.model_path, f"images{ext}")
        points_file = os.path.join(self.model_path, f"points3D{ext}")
        
        return (os.path.isfile(cameras_file) and 
                os.path.isfile(images_file) and 
                os.path.isfile(points_file))
    
    def _read_next_bytes(self, fid, num_bytes: int, format_char_sequence: str, 
                         endian_character: str = "<"):
        """Read and unpack bytes from binary file."""
        data = fid.read(num_bytes)
        return struct.unpack(endian_character + format_char_sequence, data)
    
    def _load_cameras(self, ext: str):
        """Load camera data."""
        cameras_file = os.path.join(self.model_path, f"cameras{ext}")
        
        if ext == ".txt":
            self._load_cameras_text(cameras_file)
        else:
            self._load_cameras_binary(cameras_file)
    
    def _load_cameras_text(self, path: str):
        """Load cameras from text format."""
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                
                self.cameras[camera_id] = Camera(
                    id=camera_id,
                    model=model,
                    width=width,
                    height=height,
                    params=params
                )
    
    def _load_cameras_binary(self, path: str):
        """Load cameras from binary format."""
        with open(path, "rb") as f:
            num_cameras = self._read_next_bytes(f, 8, "Q")[0]
            
            for _ in range(num_cameras):
                camera_properties = self._read_next_bytes(
                    f, num_bytes=24, format_char_sequence="iiQQ"
                )
                camera_id = camera_properties[0]
                model_id = camera_properties[1]
                model_name = CAMERA_MODEL_IDS[model_id].model_name
                width = camera_properties[2]
                height = camera_properties[3]
                num_params = CAMERA_MODEL_IDS[model_id].num_params
                params = self._read_next_bytes(
                    f, num_bytes=8 * num_params, format_char_sequence="d" * num_params
                )
                
                self.cameras[camera_id] = Camera(
                    id=camera_id,
                    model=model_name,
                    width=width,
                    height=height,
                    params=np.array(params)
                )

    def _load_images(self, ext: str):
        """Load image data."""
        images_file = os.path.join(self.model_path, f"images{ext}")
        
        if ext == ".txt":
            self._load_images_text(images_file)
        else:
            self._load_images_binary(images_file)
        
        # Precompute rotation matrices and translation vectors
        for img_id, img in self.images.items():
            self.image_rotation_matrices[img_id] = qvec2rotmat(img.qvec)
            self.image_translation_vectors[img_id] = img.tvec.copy()

    def _load_images_text(self, path: str):
        """Load images from text format."""
        with open(path, "r") as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith("#"):
                i += 1
                continue
            
            elems = line.split()
            image_id = int(elems[0])
            qvec = np.array(tuple(map(float, elems[1:5])))
            tvec = np.array(tuple(map(float, elems[5:8])))
            camera_id = int(elems[8])
            image_name = elems[9]
            
            # Next line contains 2D points
            i += 1
            points_line = lines[i].strip()
            points_elems = points_line.split() if points_line else []
            
            if points_elems:
                xys = np.column_stack([
                    tuple(map(float, points_elems[0::3])),
                    tuple(map(float, points_elems[1::3]))
                ])
                point3D_ids = np.array(tuple(map(int, points_elems[2::3])))
            else:
                xys = np.zeros((0, 2))
                point3D_ids = np.array([])
            
            self.images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids
            )
            self.name_to_image_id[image_name] = image_id
            self.image_id_to_name[image_id] = image_name
            
            i += 1
    
    def _load_images_binary(self, path: str):
        """Load images from binary format."""
        with open(path, "rb") as f:
            num_images = self._read_next_bytes(f, 8, "Q")[0]
            
            for _ in range(num_images):
                binary_properties = self._read_next_bytes(
                    f, num_bytes=64, format_char_sequence="idddddddi"
                )
                image_id = binary_properties[0]
                qvec = np.array(binary_properties[1:5])
                tvec = np.array(binary_properties[5:8])
                camera_id = binary_properties[8]
                
                # Read image name
                binary_name = b""
                current_char = self._read_next_bytes(f, 1, "c")[0]
                while current_char != b"\x00":
                    binary_name += current_char
                    current_char = self._read_next_bytes(f, 1, "c")[0]
                image_name = binary_name.decode("utf-8")
                
                # Read 2D points
                num_points2D = self._read_next_bytes(f, 8, "Q")[0]
                if num_points2D > 0:
                    x_y_id_s = self._read_next_bytes(
                        f, num_bytes=24 * num_points2D, 
                        format_char_sequence="ddq" * num_points2D
                    )
                    xys = np.column_stack([
                        tuple(map(float, x_y_id_s[0::3])),
                        tuple(map(float, x_y_id_s[1::3]))
                    ])
                    point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
                else:
                    xys = np.zeros((0, 2))
                    point3D_ids = np.array([])
                
                self.images[image_id] = Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids
                )
                self.name_to_image_id[image_name] = image_id
                self.image_id_to_name[image_id] = image_name
    
    def _load_points3D(self, ext: str):
        """Load 3D points."""
        points_file = os.path.join(self.model_path, f"points3D{ext}")
        
        if ext == ".txt":
            self._load_points3D_text(points_file)
        else:
            self._load_points3D_binary(points_file)
        
        # Convert to numpy arrays for efficient access
        if self.points3D:
            self.points3D_ids = np.array([pid for pid in self.points3D.keys()])
            self.points3D_xyz = np.array([p.xyz for p in self.points3D.values()])
            self.points3D_rgb = np.array([p.rgb for p in self.points3D.values()])
    
    def _load_points3D_text(self, path: str):
        """Load 3D points from text format."""
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                
                # Parse track information
                if len(elems) > 8:
                    track_data = list(map(int, elems[8:]))
                    image_ids = np.array(track_data[0::2])
                    point2D_idxs = np.array(track_data[1::2])
                else:
                    image_ids = np.array([])
                    point2D_idxs = np.array([])
                
                self.points3D[point3D_id] = Point3D(
                    id=point3D_id,
                    xyz=xyz,
                    rgb=rgb,
                    error=error,
                    image_ids=image_ids,
                    point2D_idxs=point2D_idxs
                )
    
    def _load_points3D_binary(self, path: str):
        """Load 3D points from binary format."""
        with open(path, "rb") as f:
            num_points = self._read_next_bytes(f, 8, "Q")[0]
            
            for _ in range(num_points):
                binary_properties = self._read_next_bytes(
                    f, num_bytes=43, format_char_sequence="QdddBBBd"
                )
                point3D_id = binary_properties[0]
                xyz = np.array(binary_properties[1:4])
                rgb = np.array(binary_properties[4:7])
                error = binary_properties[7]
                
                track_length = self._read_next_bytes(f, 8, "Q")[0]
                if track_length > 0:
                    track_elems = self._read_next_bytes(
                        f, num_bytes=8 * track_length,
                        format_char_sequence="ii" * track_length
                    )
                    image_ids = np.array(tuple(map(int, track_elems[0::2])))
                    point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
                else:
                    image_ids = np.array([])
                    point2D_idxs = np.array([])
                
                self.points3D[point3D_id] = Point3D(
                    id=point3D_id,
                    xyz=xyz,
                    rgb=rgb,
                    error=error,
                    image_ids=image_ids,
                    point2D_idxs=point2D_idxs
                )
    
    def _prepare_convenience_arrays(self):
        """Prepare numpy arrays for batched operations."""
        if not self.sorted_image_names:
            return
        
        N = len(self.sorted_image_names)
        
        # Prepare extrinsics matrices (N x 4 x 4)
        self.extrinsics_matrices = np.zeros((N, 4, 4), dtype=np.float32)
        self.intrinsics_matrices = np.zeros((N, 3, 3), dtype=np.float32)
        
        # Prepare intrinsics parameters (N x num_params)
        # First, get the number of params from the first camera
        first_image_name = self.sorted_image_names[0]
        first_image_id = self.name_to_image_id[first_image_name]
        first_camera_id = self.images[first_image_id].camera_id
        first_camera = self.cameras[first_camera_id]
        num_params = len(first_camera.params)
        self.intrinsics_params = np.zeros((N, num_params), dtype=np.float32)
        
        # Fill the arrays
        for idx, image_name in enumerate(self.sorted_image_names):
            image_id = self.name_to_image_id[image_name]
            camera_id = self.images[image_id].camera_id
            camera = self.cameras[camera_id]
            
            # Fill extrinsics matrix
            R = self.image_rotation_matrices[image_id]
            t = self.image_translation_vectors[image_id]
            self.extrinsics_matrices[idx, :3, :3] = R
            self.extrinsics_matrices[idx, :3, 3] = t
            self.extrinsics_matrices[idx, 3, :] = [0, 0, 0, 1]
            
            # Fill intrinsics matrix
            K = self._get_camera_matrix(camera)
            self.intrinsics_matrices[idx] = K
            
            # Fill intrinsics parameters
            self.intrinsics_params[idx, :len(camera.params)] = camera.params
            
            # Warn if camera models differ
            if idx > 0 and camera.model != first_camera.model:
                warnings.warn(f"Camera model differs: {camera.model} vs {first_camera.model}. "
                              f"Intrinsics matrices may not be consistent.")

    def get_image_path(self, image_name: str, absolute: bool = True) -> str:
        """
        Get the path to an image.
        
        Args:
            image_name: Image filename
            absolute: If True, return absolute path (requires image_path to be set).
                     If False, return just the filename.
        
        Returns:
            Path to the image
        """
        if absolute:
            if self.image_path is None:
                raise ValueError("image_path not provided during initialization")
            return os.path.join(self.image_path, image_name)
        else:
            return image_name
    
    def get_all_image_paths(self, absolute: bool = True) -> List[str]:
        """
        Get paths to all images, sorted by name.
        
        Args:
            absolute: If True, return absolute paths (requires image_path to be set).
                     If False, return just filenames.
        
        Returns:
            List of image paths in sorted order
        """
        if absolute:
            if self.image_path is None:
                raise ValueError("image_path not provided during initialization")
            return [os.path.join(self.image_path, name) for name in self.sorted_image_names]
        else:
            return self.sorted_image_names.copy()
    
    def get_image_names(self, sorted_names: bool = True) -> List[str]:
        """
        Get all image names.
        
        Args:
            sorted_names: If True, return sorted names; otherwise return in load order.
        
        Returns:
            List of image names
        """
        if sorted_names:
            return self.sorted_image_names.copy()
        else:
            return list(self.name_to_image_id.keys())
    
    def get_extrinsics_batch(self) -> np.ndarray:
        """
        Get extrinsics matrices for all images as a batch.
        
        Returns:
            N x 4 x 4 array of extrinsics matrices (world to camera),
            aligned with sorted image names
        """
        if self.extrinsics_matrices is None:
            raise RuntimeError("No images loaded")
        return self.extrinsics_matrices.copy()

    def get_intrinsics_batch(self) -> np.ndarray:
        """
        Get intrinsics matrices for all images as a batch.
        
        Returns:
            N x 3 x 3 array of intrinsics matrices,
            aligned with sorted image names
        """
        if self.intrinsics_matrices is None:
            raise RuntimeError("No images loaded")
        return self.intrinsics_matrices.copy()
    
    def get_intrinsics_params_batch(self) -> np.ndarray:
        """
        Get raw intrinsics parameters for all images as a batch.
        
        Returns:
            N x num_params array of intrinsics parameters,
            aligned with sorted image names
        """
        if self.intrinsics_params is None:
            raise RuntimeError("No images loaded")
        return self.intrinsics_params.copy()
    
    def get_camera_intrinsics(self, camera_id: int) -> Dict:
        """
        Get camera intrinsics.
        
        Args:
            camera_id: Camera ID
        
        Returns:
            Dictionary with camera parameters
        """
        if camera_id not in self.cameras:
            raise KeyError(f"Camera {camera_id} not found")
        
        cam = self.cameras[camera_id]
        return {
            'model': cam.model,
            'width': cam.width,
            'height': cam.height,
            'params': cam.params,
            'camera_matrix': self._get_camera_matrix(cam),
            'distortion_params': self._get_distortion_params(cam)
        }
    
    def get_camera_intrinsics_by_image(self, image_name: str) -> Dict:
        """
        Get camera intrinsics for a specific image.
        
        Args:
            image_name: Image filename
        
        Returns:
            Dictionary with camera parameters
        """
        if image_name not in self.name_to_image_id:
            raise KeyError(f"Image {image_name} not found")
        
        image_id = self.name_to_image_id[image_name]
        camera_id = self.images[image_id].camera_id
        return self.get_camera_intrinsics(camera_id)
    
    def get_extrinsics(self, image_id: int) -> Dict[str, np.ndarray]:
        """
        Get camera extrinsics (world to camera).
        
        Args:
            image_id: Image ID
        
        Returns:
            Dictionary with rotation matrix (R) and translation vector (t)
        """
        if image_id not in self.images:
            raise KeyError(f"Image {image_id} not found")
        
        return {
            'R': self.image_rotation_matrices[image_id],
            't': self.image_translation_vectors[image_id]
        }
    
    def get_extrinsics_by_name(self, image_name: str) -> Dict[str, np.ndarray]:
        """
        Get camera extrinsics for a specific image.
        
        Args:
            image_name: Image filename
        
        Returns:
            Dictionary with rotation matrix (R) and translation vector (t)
        """
        if image_name not in self.name_to_image_id:
            raise KeyError(f"Image {image_name} not found")
        
        image_id = self.name_to_image_id[image_name]
        return self.get_extrinsics(image_id)
    
    def get_camera_matrix(self, image_id: int) -> np.ndarray:
        """
        Get the full camera projection matrix (K[R|t]) for an image.
        
        Args:
            image_id: Image ID
        
        Returns:
            3x4 projection matrix
        """
        if image_id not in self.images:
            raise KeyError(f"Image {image_id} not found")
        
        img = self.images[image_id]
        cam = self.cameras[img.camera_id]
        
        # Get camera matrix K
        K = self._get_camera_matrix(cam)
        
        # Get rotation and translation
        R = self.image_rotation_matrices[image_id]
        t = self.image_translation_vectors[image_id]
        
        # Form projection matrix P = K [R | t]
        Rt = np.hstack([R, t.reshape(3, 1)])
        return K @ Rt
    
    def _get_camera_matrix(self, camera: Camera) -> np.ndarray:
        """Construct camera intrinsic matrix K."""
        if camera.model == "SIMPLE_PINHOLE" or camera.model == "SIMPLE_RADIAL":
            fx = camera.params[0]
            fy = fx
            cx = camera.params[1]
            cy = camera.params[2]
        elif camera.model == "PINHOLE":
            fx, fy, cx, cy = camera.params[:4]
        elif camera.model == "OPENCV":
            fx, fy, cx, cy = camera.params[:4]
        else:
            # Default for other models
            fx, fy, cx, cy = camera.params[:4]
        
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
    
    def _get_distortion_params(self, camera: Camera) -> np.ndarray:
        """Get distortion parameters for the camera."""
        if camera.model == "SIMPLE_RADIAL":
            return camera.params[3:4]
        elif camera.model == "RADIAL":
            return camera.params[4:6]
        elif camera.model == "OPENCV":
            return camera.params[4:8]
        else:
            return np.array([])
    
    def get_points3D(
        self, return_numpy: bool = True
    ) -> Union[Dict[int, Point3D], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Get 3D points.
        
        Args:
            return_numpy: If True, return as numpy arrays (xyz, rgb, ids).
                         If False, return as dictionary of Point3D objects.
        
        Returns:
            If return_numpy: tuple of (xyz, rgb, point_ids)
            If not return_numpy: dictionary of Point3D objects
        """
        if not self.load_points:
            raise RuntimeError("Points were not loaded. Set load_points=True when initializing.")
        
        if return_numpy:
            if self.points3D_xyz is None:
                return np.array([]), np.array([]), np.array([])
            return self.points3D_xyz, self.points3D_rgb, self.points3D_ids
        else:
            return self.points3D
    
    def get_image_points3D(self, image_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get 3D points visible in a specific image.
        
        Args:
            image_name: Image filename
        
        Returns:
            Tuple of (points3D_xyz, points2D_xy) for the image
        """
        if image_name not in self.name_to_image_id:
            raise KeyError(f"Image {image_name} not found")
        
        image_id = self.name_to_image_id[image_name]
        img = self.images[image_id]
        
        # Filter valid points
        valid_mask = img.point3D_ids != self.INVALID_POINT3D
        
        if not self.load_points:
            raise RuntimeError("Points were not loaded. Set load_points=True when initializing.")
        
        # Get 3D points
        valid_point_ids = img.point3D_ids[valid_mask]
        points3D_xyz = np.array([self.points3D[pid].xyz for pid in valid_point_ids 
                                 if pid in self.points3D])
        points2D_xy = img.xys[valid_mask]
        
        return points3D_xyz, points2D_xy
    
    def get_image_ids(self) -> List[int]:
        """Get list of all image IDs."""
        return list(self.images.keys())
    
    def get_camera_ids(self) -> List[int]:
        """Get list of all camera IDs."""
        return list(self.cameras.keys())
    
    def get_statistics(self) -> Dict:
        """Get statistics about the loaded data."""
        stats = {
            'num_cameras': len(self.cameras),
            'num_images': len(self.images),
            'num_points3D': len(self.points3D) if self.load_points else None,
            'camera_models': {},
            'image_resolutions': set(),
            'image_path_available': self.image_path is not None
        }
        
        # Count camera models
        for cam in self.cameras.values():
            stats['camera_models'][cam.model] = stats['camera_models'].get(cam.model, 0) + 1
            stats['image_resolutions'].add((cam.width, cam.height))
        
        if self.load_points:
            stats['points_with_tracks'] = sum(1 for p in self.points3D.values() 
                                              if len(p.image_ids) > 0)
            if self.points3D_xyz is not None and len(self.points3D_xyz) > 0:
                stats['points_bbox'] = {
                    'min': self.points3D_xyz.min(axis=0),
                    'max': self.points3D_xyz.max(axis=0)
                }
        
        stats['image_resolutions'] = list(stats['image_resolutions'])
        
        return stats


if __name__ == "__main__":
    # Initialize with image path for absolute paths
    manager = ColmapManager(
        model_path=os.path.expanduser("~/datasets/mipnerf360/bicycle/sparse/0"),
        image_path=os.path.expanduser("~/datasets/mipnerf360/bicycle/images"),
        load_points=True
    )

    stats = manager.get_statistics()
    print(stats)

    # Get all image names (sorted)
    image_names = manager.get_image_names()
    print(f"First image: {image_names[0]}")

    # Get absolute paths to all images
    image_paths = manager.get_all_image_paths(absolute=True)

    # Get batch arrays (N x 4 x 4 and N x 3 x 3)
    extrinsics = manager.get_extrinsics_batch()  # Shape: (N, 4, 4)
    intrinsics = manager.get_intrinsics_batch()   # Shape: (N, 3, 3)
    print(f'extrinsics shape: {extrinsics.shape}, intrinsics shape: {intrinsics.shape}')

    # Access individual image data
    image_name = image_names[0]
    intrinsics_dict = manager.get_camera_intrinsics_by_image(image_name)
    extrinsics_dict = manager.get_extrinsics_by_name(image_name)

    # # The batch arrays are aligned with sorted image names
    # for i, name in enumerate(manager.get_image_names()):
    #     print(f"{name}: extrinsics shape {extrinsics[i].shape}")
