# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Samples 3DGS initialization points from multi-view depth maps.

The sampling prioritizes edges and corners, then samples the other regions
uniformly, and removes the redundancy of overlapping views by a fast voxel
grid filter. `sample_points_from_depth_maps_batch` unprojects depth maps and
samples them; `PointSamplerFor3DGS` samples already unprojected points. Run
as a script, the module samples random data as a smoke test.
"""

import argparse
from typing import Optional, Tuple

import cv2
import numpy as np


class PointSamplerFor3DGS:
    """Samples points from depth maps for 3DGS initialization.

    Prioritizes edge and corner points, where gradients matter most for
    densification, and removes multi-view redundancy by a voxel grid filter.

    Attributes:
        max_points: Maximum number of points to sample.
        edge_weight: Sampling weight multiplier for edge pixels.
        corner_weight: Sampling weight multiplier for corner pixels.
        edge_detection_method: 'canny', 'gradient', or 'sobel'.
        canny_threshold1: The first Canny threshold.
        canny_threshold2: The second Canny threshold.
        gradient_threshold: Threshold of the 'gradient' edge detection.
        use_confidence: Whether to filter and weight by depth confidence.
        confidence_threshold: Points below this confidence are discarded.
        use_redundancy_filter: Whether to apply the voxel grid filter.
        voxel_size: Voxel size of the filter, in meters.
        random_seed: The seed of the numpy global random state.
    """

    def __init__(
        self,
        max_points: int = 1_000_000,
        edge_weight: float = 3.0,
        corner_weight: float = 5.0,
        edge_detection_method: str = 'canny',
        canny_threshold1: int = 50,
        canny_threshold2: int = 150,
        gradient_threshold: float = 0.1,
        use_confidence: bool = True,
        confidence_threshold: float = 2.0,
        # Redundancy handling parameters.
        use_redundancy_filter: bool = True,
        voxel_size: float = 0.01,
        random_seed: int = 42,
    ):
        """Initializes the point sampler and seeds numpy's global RNG.

        The arguments are described in the class attributes.
        """
        self.max_points = max_points
        self.edge_weight = edge_weight
        self.corner_weight = corner_weight
        self.edge_detection_method = edge_detection_method
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2
        self.gradient_threshold = gradient_threshold
        self.use_confidence = use_confidence
        self.confidence_threshold = confidence_threshold

        # Redundancy handling parameters.
        self.use_redundancy_filter = use_redundancy_filter
        self.voxel_size = voxel_size

        self.random_seed = random_seed

        np.random.seed(random_seed)

    def fast_voxel_grid_filter(
            self,
            points: np.ndarray,  # [N, 3]
            colors: np.ndarray,  # [N, 3]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Keeps the first point of each voxel, in O(N) with a dictionary.

        Args:
            points: Input points [N, 3].
            colors: Input colors [N, 3].

        Returns:
            The downsampled points [M, 3] and colors [M, 3].
        """
        if len(points) == 0:
            return points, colors

        # Voxel indices.
        voxel_indices = np.floor(points / self.voxel_size).astype(np.int32)

        # Unique voxel IDs by integer encoding.
        voxel_ids = (voxel_indices[:, 0] * 1000000 +
                     voxel_indices[:, 1] * 1000 + voxel_indices[:, 2])

        # A dictionary stores one point per voxel.
        unique_voxels = {}
        filtered_points_list = []
        filtered_colors_list = []

        for i, voxel_id in enumerate(voxel_ids):
            if voxel_id not in unique_voxels:
                unique_voxels[voxel_id] = i
                filtered_points_list.append(points[i])
                filtered_colors_list.append(colors[i])

        filtered_points = np.array(filtered_points_list)
        filtered_colors = np.array(filtered_colors_list)

        return filtered_points, filtered_colors

    def detect_edges_batch(
            self,
            images: np.ndarray,  # [N, H, W, 3]
    ) -> np.ndarray:
        """Detects edges in a batch of images.

        Args:
            images: Batch of RGB images [N, H, W, 3] in range [0, 255].

        Returns:
            Binary edge maps [N, H, W].
        """
        N, H, W, C = images.shape
        edge_maps = np.zeros((N, H, W), dtype=bool)

        for i in range(N):
            # Convert to grayscale
            gray = cv2.cvtColor(images[i].astype(np.uint8), cv2.COLOR_RGB2GRAY)

            if self.edge_detection_method == 'canny':
                edges = cv2.Canny(gray, self.canny_threshold1,
                                  self.canny_threshold2)
                edge_maps[i] = edges > 0

            elif self.edge_detection_method == 'sobel':
                sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
                edge_maps[i] = gradient_magnitude > np.percentile(
                    gradient_magnitude, 80)

            elif self.edge_detection_method == 'gradient':
                grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                gradient = np.sqrt(grad_x**2 + grad_y**2)
                gradient = gradient / (gradient.max() + 1e-8)
                edge_maps[i] = gradient > self.gradient_threshold

        return edge_maps

    def detect_corners_batch(
            self,
            images: np.ndarray,  # [N, H, W, 3]
    ) -> np.ndarray:
        """Detects corners in a batch of images by Harris corner detection.

        Args:
            images: Batch of RGB images [N, H, W, 3] in range [0, 255].

        Returns:
            Binary corner maps [N, H, W].
        """
        N, H, W, C = images.shape
        corner_maps = np.zeros((N, H, W), dtype=bool)

        for i in range(N):
            # Convert to grayscale
            gray = cv2.cvtColor(images[i].astype(np.uint8), cv2.COLOR_RGB2GRAY)

            # Harris corner detection.
            gray_float = np.float32(gray)
            harris = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)

            # Dilate the corner map to get more points around corners.
            harris = cv2.dilate(harris, None)

            # Threshold.
            corner_threshold = 0.01 * harris.max()
            corner_maps[i] = harris > corner_threshold

        return corner_maps

    def compute_sampling_weights_batch(
            self,
            images: np.ndarray,  # [N, H, W, 3]
            confidence_maps: Optional[np.ndarray] = None,  # [N, H, W]
    ) -> np.ndarray:
        """Computes sampling weights, higher on edges and corners.

        Confidence filtering is applied before the weight computation: points
        below the confidence threshold get weight 0 and are discarded.

        Args:
            images: Batch of RGB images [N, H, W, 3].
            confidence_maps: Optional batch of confidence maps [N, H, W].

        Returns:
            Sampling weights [N, H, W].
        """
        N, H, W, _ = images.shape

        # Step 1: confidence filtering (highest priority).
        if self.use_confidence and confidence_maps is not None:
            # Points at or above the threshold.
            confidence_mask = confidence_maps >= self.confidence_threshold
            weights = confidence_mask.astype(np.float32)
        else:
            weights = np.ones((N, H, W), dtype=np.float32)

        # Step 2: detect edges and corners; only the points that passed the
        # confidence filter are weighted.
        edges = self.detect_edges_batch(images)
        corners = self.detect_corners_batch(images)

        # Higher weights on edges and corners (only where weights > 0).
        edge_mask = edges & (weights > 0)
        corner_mask = corners & (weights > 0)

        weights[edge_mask] *= self.edge_weight
        weights[corner_mask] *= self.corner_weight

        # Step 3: confidence-based weighting (optional).
        if self.use_confidence and confidence_maps is not None:
            for i in range(N):
                conf = confidence_maps[i]
                conf_mask_i = confidence_mask[i]

                if conf_mask_i.any():
                    conf_valid = conf[conf_mask_i]
                    conf_min = conf_valid.min()
                    conf_max = conf_valid.max()

                    if conf_max > conf_min:
                        conf_norm = np.zeros_like(conf)
                        conf_norm[conf_mask_i] = (
                            (conf[conf_mask_i] - conf_min) /
                            (conf_max - conf_min + 1e-8))
                        weights[i] = weights[i] * (0.5 + 0.5 * conf_norm)

        # Views without any weight keep zero weights. The message predates
        # this; such views contribute no points.
        for i in range(N):
            if weights[i].sum() == 0:
                print("  Warning: No points above confidence threshold in "
                      f"view {i}, using all points")
                weights[i] = np.zeros((H, W), dtype=np.float32)

        return weights

    def sample_points_from_views(
            self,
            points_3d: np.ndarray,  # [N, H, W, 3]
            colors: np.ndarray,  # [N, H, W, 3]
            images: np.ndarray,  # [N, H, W, 3]
            confidence_maps: Optional[np.ndarray] = None,  # [N, H, W]
            masks: Optional[np.ndarray] = None,  # [N, H, W]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Samples points from multiple views with redundancy filtering.

        Args:
            points_3d: 3D points in world coordinates [N, H, W, 3].
            colors: RGB colors for each point [N, H, W, 3].
            images: Original images for edge/corner detection [N, H, W, 3].
            confidence_maps: Optional confidence maps [N, H, W].
            masks: Optional masks of the regions to keep [N, H, W].

        Returns:
            The sampled 3D points [P, 3] and colors [P, 3].
        """
        N, H, W, _ = points_3d.shape

        print(f"Sampling from {N} views, resolution {H}x{W}")
        if self.use_confidence:
            print(f"Confidence threshold: {self.confidence_threshold}")

        # Sampling weights for all views.
        weights = self.compute_sampling_weights_batch(
            images, confidence_maps)  # [N, H, W]

        # Apply the additional masks if provided.
        if masks is not None:
            weights[~masks] = 0

        # Flatten everything.
        points_flat = points_3d.reshape(-1, 3)
        colors_flat = colors.reshape(-1, 3)
        weights_flat = weights.reshape(-1)

        # Remove invalid points (NaN or inf) and points with zero weight.
        valid_mask = ~(np.isnan(points_flat).any(axis=1) |
                       np.isinf(points_flat).any(axis=1))
        valid_mask &= (weights_flat > 0)

        points_flat = points_flat[valid_mask]
        colors_flat = colors_flat[valid_mask]
        weights_flat = weights_flat[valid_mask]

        print(f"Points after confidence filtering: {len(points_flat)}")

        if len(points_flat) == 0:
            print("Warning: No points passed confidence threshold!")
            return np.array([]), np.array([])

        # Fast voxel grid filtering removes spatial redundancy.
        if self.use_redundancy_filter:
            print("Applying fast voxel grid filtering...")
            points_filtered, colors_filtered = self.fast_voxel_grid_filter(
                points_flat, colors_flat)
        else:
            points_filtered = points_flat
            colors_filtered = colors_flat

        print(f"Points after redundancy filtering: {len(points_filtered)}")

        if len(points_filtered) == 0:
            print("Warning: No points after redundancy filtering")
            return np.array([]), np.array([])

        # The final number of points: at most max_points.
        n_points = min(self.max_points, len(points_filtered))

        if len(points_filtered) > self.max_points:
            print(f"Sampling {self.max_points} points from "
                  f"{len(points_filtered)} (downsampling)...")
            sampled_indices = np.random.choice(len(points_filtered),
                                               size=n_points,
                                               replace=False)
            sampled_points = points_filtered[sampled_indices]
            sampled_colors = colors_filtered[sampled_indices]
        else:
            print(f"Using all {len(points_filtered)} points (within limit)")
            sampled_points = points_filtered
            sampled_colors = colors_filtered

        print(f"Final point cloud: {len(sampled_points)} points")

        return sampled_points, sampled_colors


def create_edge_aware_sampling_mask_batch(
    images: np.ndarray,  # [N, H, W, 3]
    kernel_size: int = 3,
    dilation_iterations: int = 1,
) -> np.ndarray:
    """Creates sampling masks of edges and their surroundings.

    Args:
        images: Batch of input images [N, H, W, 3].
        kernel_size: Size of the dilation kernel.
        dilation_iterations: Number of dilation iterations.

    Returns:
        Binary masks [N, H, W], True in edge regions.
    """
    N, H, W, _ = images.shape
    masks = np.zeros((N, H, W), dtype=bool)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    for i in range(N):
        # Detect edges.
        gray = cv2.cvtColor(images[i].astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Dilate the edges to include the surrounding pixels.
        dilated_edges = cv2.dilate(edges,
                                   kernel,
                                   iterations=dilation_iterations)
        masks[i] = dilated_edges > 0

    return masks


def sample_points_from_depth_maps_batch(
    depth_maps: np.ndarray,  # [N, H, W]
    extrinsics: np.ndarray,  # [N, 4, 4]
    intrinsics: np.ndarray,  # [N, 3, 3]
    images: np.ndarray,  # [N, H, W, 3]
    confidence_maps: Optional[np.ndarray] = None,  # [N, H, W]
    max_total_points: int = 1_000_000,
    edge_weight: float = 3.0,
    corner_weight: float = 5.0,
    confidence_threshold: float = 2.0,
    use_redundancy_filter: bool = True,
    voxel_size: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unprojects multiple views and samples 3DGS initialization points.

    Args:
        depth_maps: Batch of depth maps [N, H, W].
        extrinsics: Batch of world-to-camera extrinsics [N, 4, 4].
        intrinsics: Batch of camera intrinsics [N, 3, 3].
        images: Batch of RGB images [N, H, W, 3].
        confidence_maps: Batch of confidence maps [N, H, W].
        max_total_points: Maximum total points in the final point cloud.
        edge_weight: Sampling weight multiplier for edge pixels.
        corner_weight: Sampling weight multiplier for corner pixels.
        confidence_threshold: Points below this confidence are discarded.
        use_redundancy_filter: Whether to apply redundancy filtering.
        voxel_size: Voxel size for spatial downsampling (meters).

    Returns:
        The combined point cloud [P, 3] and colors [P, 3].
    """
    N, H, W = depth_maps.shape

    print(f"Processing {N} views at resolution {H}x{W}")
    print(f"Target: {max_total_points} points total")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Redundancy filter: {'ON' if use_redundancy_filter else 'OFF'}")
    if use_redundancy_filter:
        print(f"  Voxel size: {voxel_size}m")

    # Pixel coordinate grid.
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Prepare for broadcasting.
    u = u.reshape(1, H, W)
    v = v.reshape(1, H, W)

    # Intrinsics components.
    fx = intrinsics[:, 0, 0].reshape(N, 1, 1)
    fy = intrinsics[:, 1, 1].reshape(N, 1, 1)
    cx = intrinsics[:, 0, 2].reshape(N, 1, 1)
    cy = intrinsics[:, 1, 2].reshape(N, 1, 1)

    # Unproject to camera coordinates (vectorized).
    x_cam = (u - cx) * depth_maps / fx
    y_cam = (v - cy) * depth_maps / fy
    z_cam = depth_maps

    points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # [N, H, W, 3]

    # Transform to world coordinates.
    points_world = np.zeros_like(points_cam)

    for i in range(N):
        R = extrinsics[i, :3, :3]
        t = extrinsics[i, :3, 3]
        # Transform each point: P_world = R.T @ (P_cam - t).
        points_cam_i = points_cam[i].reshape(-1, 3)
        points_world_i = np.dot(points_cam_i - t, R)
        points_world[i] = points_world_i.reshape((H, W, 3))

    # Edge-aware sampling with fast redundancy handling.
    sampler = PointSamplerFor3DGS(
        max_points=max_total_points,
        edge_weight=edge_weight,
        corner_weight=corner_weight,
        use_confidence=True,
        confidence_threshold=confidence_threshold,
        use_redundancy_filter=use_redundancy_filter,
        voxel_size=voxel_size,
    )

    all_points, all_colors = sampler.sample_points_from_views(
        points_3d=points_world,
        colors=images,
        images=images,
        confidence_maps=confidence_maps,
        masks=None,
    )

    return all_points, all_colors


def main():
    """Samples random data with the command-line settings (a smoke test)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_points", type=int, default=1_000_000)
    parser.add_argument("--edge_weight", type=float, default=3.0)
    parser.add_argument("--corner_weight", type=float, default=5.0)
    parser.add_argument("--confidence_threshold", type=float, default=2.0)
    parser.add_argument("--voxel_size",
                        type=float,
                        default=0.01,
                        help="Voxel size in meters")
    parser.add_argument("--no_redundancy_filter",
                        action="store_true",
                        help="Disable redundancy filtering")
    args = parser.parse_args()

    print("3DGS Point Sampler with Confidence Filtering")
    print(f"  Max points: {args.max_points}")
    print(f"  Edge weight: {args.edge_weight}")
    print(f"  Corner weight: {args.corner_weight}")
    print(f"  Confidence threshold: {args.confidence_threshold}")
    print(f"  Voxel size: {args.voxel_size}m")
    print(
        f"  Redundancy filter: {'OFF' if args.no_redundancy_filter else 'ON'}")

    # Example with random data (for testing).
    N, H, W = 10, 480, 640
    print(f"\nTesting with random data: {N} views, {H}x{W} resolution")

    # Dummy data.
    depth_maps = np.random.rand(N, H, W).astype(np.float32) * 10.0
    extrinsics = np.tile(np.eye(4), (N, 1, 1)).astype(np.float32)
    intrinsics = np.tile(
        np.array([[1000, 0, W / 2], [0, 1000, H / 2], [0, 0, 1]]), (N, 1, 1))
    images = np.random.randint(0, 255, (N, H, W, 3), dtype=np.uint8)
    confidence_maps = np.random.rand(N, H, W).astype(np.float32)

    # Sample the points.
    points_3d, colors = sample_points_from_depth_maps_batch(
        depth_maps=depth_maps,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        images=images,
        confidence_maps=confidence_maps,
        max_total_points=args.max_points,
        edge_weight=args.edge_weight,
        corner_weight=args.corner_weight,
        confidence_threshold=args.confidence_threshold,
        use_redundancy_filter=not args.no_redundancy_filter,
        voxel_size=args.voxel_size,
    )

    print(f"\nSampled {len(points_3d)} points")
    print(f"Points shape: {points_3d.shape}")
    print(f"Colors shape: {colors.shape}")


if __name__ == "__main__":
    main()
