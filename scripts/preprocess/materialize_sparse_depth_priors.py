#!/usr/bin/env python
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Materialize the sparse-depth tracks' priors (spatialarena/sparse_depth.py).

The priors are written under <benchmark_root>/_sparse_depth/.

- `sfm`: for every full-standard scene of the SfM-point datasets, each
  image's observed COLMAP keypoints with the depth of their 3D point in that
  image's camera, from the scene's camera-prior model (the model the
  rgb_cameras track reads: `sparse/0`, `sparse`,
  `dslr_calibration_undistorted`); coordinates are scaled to the image file
  when the model's camera size differs.
- `kitti`: every KITTI VideoDepth scene's frames, the raw Velodyne scan
  (`<date>/<drive>_sync/velodyne_points`) projected into `image_02` with the
  date's calibration (`--kitti_calib_root/<date>/calib_{cam_to_cam,
  velo_to_cam}.txt`, KITTI raw `<date>_calib.zip`); points at least 0.1 m in
  front of the camera and inside the image.

Each prior is also compared with the scene's GT depth where both exist (median
ratio and AbsRel), as a sanity check of the projection; the summary goes to
<benchmark_root>/_sparse_depth/summary.json.

    python scripts/preprocess/materialize_sparse_depth_priors.py sfm kitti \\
        --workers 16
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent import futures
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "inference"))

# pylint: disable=wrong-import-position
from spatialarena import sparse_depth  # noqa: E402
from spatialarena.eval import colmap_view_resolver  # noqa: E402
from spatialarena.eval import dataset_registry  # noqa: E402
from spatialarena.geometry import read_write_model  # noqa: E402
# pylint: enable=wrong-import-position

SFM_DATASETS = {
    # Not ScanNet++: 19 of its 20 COLMAP models have 3D points without tracks
    # (which image sees which point).
    "MVRecon": ("dtu", "eth3d", "tanks_and_temples"),
    "SfM": ("dtu", "eth3d", "imc", "tanks_and_temples", "texturepoorsfm"),
}
CASE_LISTS = (
    REPO_ROOT / "scripts/test/full_standard_depth_cases.txt",
    REPO_ROOT / "scripts/test/full_standard_sfm_cases.txt",
)


def case_scenes(task: str, dataset: str) -> list[str]:
    """Returns the full-standard scenes of a dataset, in case-list order."""
    scenes = []
    for case_list in CASE_LISTS:
        for line in case_list.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                parts = line.split("|")
                if parts[0] == task and parts[1] == dataset and parts[
                        2] not in scenes:
                    scenes.append(parts[2])
    return scenes


def gt_check(data_path: Path, image_path: Path, xy: np.ndarray, z: np.ndarray,
             size: tuple[int, int]) -> dict | None:
    """Compares a prior with the GT depth on the pixels where both exist.

    The GT is read as the depth-prior loader reads it.

    Args:
        data_path: The scene folder.
        image_path: The image the prior belongs to.
        xy: (N, 2) prior pixel coordinates.
        z: (N,) prior depths.
        size: The image (width, height).

    Returns:
        The median GT/prior ratio and AbsRel, or None when there is no GT or
        fewer than 20 shared pixels.
    """
    # adapter_utils lives in scripts/inference, put on sys.path above.
    import adapter_utils  # pylint: disable=import-outside-toplevel

    try:
        gts, has, _missing = adapter_utils.load_optional_depth_priors(
            data_path, [str(image_path)])
    # No GT depth for this layout: no check.
    except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        return None
    if not has[0]:
        return None
    gt = np.asarray(gts[0], np.float32)
    if gt.ndim == 3:
        gt = gt[..., 0]
    prior = sparse_depth.splat_sparse_depth(xy,
                                            z,
                                            size,
                                            target_size=(gt.shape[1],
                                                         gt.shape[0]))
    both = (prior > 0) & np.isfinite(gt) & (gt > 0)
    if both.sum() < 20:
        return None
    ratio = gt[both] / prior[both]
    return {
        "pixels":
            int(both.sum()),
        "median_gt_over_prior":
            float(np.median(ratio)),
        "absrel_vs_gt":
            float(np.median(np.abs(prior[both] - gt[both]) / gt[both]))
    }


def sfm_scene(task: str, dataset: str, scene: str,
              benchmark_root: Path) -> dict:
    """Writes the SfM-point priors of one scene.

    Args:
        task: Task name, e.g. "SfM".
        dataset: Dataset name.
        scene: Scene name.
        benchmark_root: The benchmark root.

    Returns:
        The scene's summary entry.
    """
    import adapter_utils  # pylint: disable=import-outside-toplevel

    data_path = dataset_registry.resolve_dataset_root(benchmark_root, task,
                                                      dataset) / scene
    colmap = adapter_utils.resolve_colmap_prior_path(data_path)
    if colmap is None:
        return {
            "scene": f"{task}/{dataset}/{scene}",
            "status": "no_colmap_model"
        }
    cameras, images, points3d = read_write_model.read_model(str(colmap))
    if not points3d:
        return {"scene": f"{task}/{dataset}/{scene}", "status": "no_points"}
    mappings, _diag = colmap_view_resolver.resolve_colmap_image_mappings(
        data_path, None, colmap_path=colmap, strict=False)
    path_by_id = {
        mapping.record.image_id: mapping.image_path for mapping in mappings
    }
    written, points, checks = 0, [], []
    for image_id, image in images.items():
        image_path = path_by_id.get(image_id)
        if image_path is None:
            continue
        ids = np.asarray(image.point3D_ids)
        valid = ids >= 0
        if not valid.any():
            continue
        xyz = np.stack([
            points3d[int(point_id)].xyz
            for point_id in ids[valid]
            if int(point_id) in points3d
        ])
        xys = np.asarray(image.xys)[valid][[
            int(point_id) in points3d for point_id in ids[valid]
        ]]
        rotation = read_write_model.qvec2rotmat(image.qvec)
        depth = (xyz @ rotation.T + np.asarray(image.tvec))[:, 2]
        camera = cameras[image.camera_id]
        with Image.open(image_path) as handle:
            width, height = handle.size
        xys = xys * [width / float(camera.width), height / float(camera.height)]
        keep = depth > 0
        out = sparse_depth.prior_path(data_path, image_path, benchmark_root)
        sparse_depth.write_points(out, xys[keep], depth[keep], (width, height),
                                  "sfm")
        written += 1
        points.append(int(keep.sum()))
        if len(checks) < 5:
            check = gt_check(data_path, image_path, xys[keep], depth[keep],
                             (width, height))
            if check:
                checks.append(check)
    return {
        "scene": f"{task}/{dataset}/{scene}",
        "status": "ok",
        "source": "sfm",
        "colmap": str(colmap),
        "images": written,
        "points_median": float(np.median(points)) if points else 0.0,
        "gt_checks": checks
    }


def read_calib(path: Path) -> dict[str, np.ndarray]:
    """Reads the numeric `key: values` rows of a KITTI raw calibration file."""
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, rest = line.split(":", 1)
            try:
                values[key.strip()] = np.array(
                    [float(value) for value in rest.split()])
            except ValueError:
                continue
    return values


def _velodyne_to_image(cam_calib: dict[str, np.ndarray],
                       velo_calib: dict[str,
                                        np.ndarray], camera: str) -> np.ndarray:
    """Returns the 3x4 projection of Velodyne points into a rectified camera."""
    projection = cam_calib[f"P_rect_{camera}"].reshape(3, 4)
    rect = np.eye(4)
    rect[:3, :3] = cam_calib["R_rect_00"].reshape(3, 3)
    velo_to_cam = np.eye(4)
    velo_to_cam[:3, :3] = velo_calib["R"].reshape(3, 3)
    velo_to_cam[:3, 3] = velo_calib["T"]
    return projection @ rect @ velo_to_cam


def kitti_scene(scene: str, benchmark_root: Path, calib_root: Path) -> dict:
    """Writes the LiDAR priors of one KITTI VideoDepth scene.

    Args:
        scene: Scene name, `<drive>_sync_<camera>`.
        benchmark_root: The benchmark root.
        calib_root: Folder with `<date>/calib_*.txt`.

    Returns:
        The scene's summary entry.
    """
    data_path = dataset_registry.resolve_dataset_root(
        benchmark_root, "VideoDepth", "kitti") / scene
    date, drive_cam = scene[:10], scene
    drive = drive_cam.rsplit("_", 1)[0]  # 2011_09_26_drive_0002_sync
    camera = drive_cam.rsplit("_", 1)[1]  # 02
    velodyne = data_path.parent / date / drive / "velodyne_points" / "data"
    cam_calib = read_calib(calib_root / date / "calib_cam_to_cam.txt")
    velo_calib = read_calib(calib_root / date / "calib_velo_to_cam.txt")
    transform = _velodyne_to_image(cam_calib, velo_calib, camera)
    images = sorted((data_path / "images").glob("*.png"))
    written, points, checks = 0, [], []
    for image_path in images:
        scan = velodyne / f"{image_path.stem}.bin"
        if not scan.is_file():
            continue
        cloud = np.fromfile(scan, dtype=np.float32).reshape(-1, 4)[:, :3]
        cloud = cloud[cloud[:, 0] > 0.1]
        homogeneous = np.c_[cloud, np.ones(len(cloud))]
        projected = homogeneous @ transform.T
        depth = projected[:, 2]
        front = depth > 0.1
        uv = projected[front, :2] / depth[front, None]
        depth = depth[front]
        with Image.open(image_path) as handle:
            width, height = handle.size
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (
            uv[:, 1] < height)
        sparse_depth.write_points(
            sparse_depth.prior_path(data_path, image_path, benchmark_root),
            uv[inside], depth[inside], (width, height), "lidar")
        written += 1
        points.append(int(inside.sum()))
        if len(checks) < 5:
            check = gt_check(data_path, image_path, uv[inside], depth[inside],
                             (width, height))
            if check:
                checks.append(check)
    return {
        "scene": f"VideoDepth/kitti/{scene}",
        "status": "ok" if written else "no_scans",
        "source": "lidar",
        "images": written,
        "points_median": float(np.median(points)) if points else 0.0,
        "gt_checks": checks
    }


def _parse_args() -> argparse.Namespace:
    """Parses the command-line options."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+", choices=("sfm", "kitti"))
    parser.add_argument("--benchmark_root",
                        type=Path,
                        default=dataset_registry.DEFAULT_BENCHMARK_ROOT)
    parser.add_argument(
        "--kitti_calib_root",
        type=Path,
        default=None,
        help=("Folder with <date>/calib_*.txt "
              "(default: <benchmark_root>/VideoDepth/kitti/_raw_calib)."))
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def _submit_sfm_scenes(pool: futures.Executor,
                       benchmark_root: Path) -> list[futures.Future]:
    """Submits one `sfm_scene` job per distinct SfM-point scene folder."""
    jobs = []
    seen = set()
    for task, datasets in SFM_DATASETS.items():
        for dataset in datasets:
            for scene in case_scenes(task, dataset):
                key = (dataset_registry.resolve_dataset_root(
                    benchmark_root, task, dataset) / scene).resolve()
                if key in seen:
                    continue  # SfM/dtu and MVRecon/dtu are one folder.
                seen.add(key)
                jobs.append(
                    pool.submit(sfm_scene, task, dataset, scene,
                                benchmark_root))
    return jobs


def _print_result(result: dict) -> None:
    """Prints one scene's summary line."""
    checks = result.get("gt_checks") or []
    if checks:
        ratio = np.median([check["median_gt_over_prior"] for check in checks])
    else:
        ratio = float("nan")
    error = ""
    if result.get("error"):
        error = f"  {result.get('error')}"
    print(f"{result.get('scene', '?'):55s} {result['status']:16s} "
          f"images {result.get('images', 0):5d} "
          f"points/img {result.get('points_median', 0):8.0f}  "
          f"GT/prior {ratio:.3f}" + error,
          flush=True)


def _write_summary(benchmark_root: Path, results: list[dict]) -> None:
    """Merges `results` into summary.json, replacing same-scene entries."""
    summary_path = sparse_depth.sparse_depth_root(
        benchmark_root) / "summary.json"
    previous = []
    if summary_path.is_file():
        previous = json.loads(summary_path.read_text())
    by_scene = {item.get("scene"): item for item in previous}
    by_scene.update({item.get("scene"): item for item in results})
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    entries = sorted(by_scene.values(),
                     key=lambda entry: str(entry.get("scene")))
    summary_path.write_text(json.dumps(entries, indent=1) + "\n")
    print(f"Wrote {summary_path}")


def main() -> None:
    """Writes the requested priors in parallel and updates the summary."""
    args = _parse_args()
    calib_root = args.kitti_calib_root
    if not calib_root:
        calib_root = (args.benchmark_root / "VideoDepth" / "kitti" /
                      "_raw_calib")
    jobs = []
    with futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        if "sfm" in args.sources:
            jobs.extend(_submit_sfm_scenes(pool, args.benchmark_root))
        if "kitti" in args.sources:
            for scene in case_scenes("VideoDepth", "kitti"):
                jobs.append(
                    pool.submit(kitti_scene, scene, args.benchmark_root,
                                calib_root))
        results = []
        for future in futures.as_completed(jobs):
            try:
                result = future.result()
            # Report the scene and keep going.
            # pylint: disable-next=broad-exception-caught
            except Exception as exc:  # noqa: BLE001
                result = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}"
                }
            results.append(result)
            _print_result(result)
    _write_summary(args.benchmark_root, results)


if __name__ == "__main__":
    main()
