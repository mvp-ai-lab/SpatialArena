#!/usr/bin/env python
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Sanity-check a materialized MonoDepth dataset.

Checks GT statistics, runs an oracle evaluation and writes visual overlays.
For every scene of the selected datasets:

- GT statistics over all frames, read the way the evaluator reads them
  (`resolve_depth_config` limits, manifest masks): image/depth size agreement,
  non-finite values, valid-pixel fraction, depth percentiles (metres), and
  whether every frame has intrinsics;
- oracle test on `--oracle_frames` frames through `evaluate_depth_scene`: GT
  as the prediction, and GT under a random positive affine map (scale in
  [0.2, 5], shift in [0, 1] x median depth); both must give AbsRel ~ 0 and
  delta1 = 1 after least-squares scale+shift alignment;
- `--overlay_frames` overlay panels per scene (RGB | colored GT depth | GT
  depth edges on the RGB) under `--output_dir/<dataset>/<scene>/`.

Writes `--output_dir/<dataset>.json` and exits non-zero when a check fails.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.preprocess import materialize_depth_visualizations  # noqa: E402
from spatialarena.eval import dataset_registry  # noqa: E402
from spatialarena.eval import depth_prediction_evaluator  # noqa: E402

DepthEvaluationConfig = depth_prediction_evaluator.DepthEvaluationConfig

ORACLE_ABSREL_TOLERANCE = 1e-4


def parse_args() -> argparse.Namespace:
    """Parses the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark_root",
                        type=Path,
                        default=dataset_registry.DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--oracle_frames", type=int, default=8)
    parser.add_argument("--overlay_frames", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def image_for_depth(scene_info, depth_path: Path) -> Path | None:
    """Finds the image of a GT depth frame, from the manifest or by stem.

    Args:
        scene_info: The scene, as listed by its dataset.
        depth_path: The GT depth file.

    Returns:
        The image path, or None when no image matches.
    """
    manifest_path = scene_info.root_path / "materialized_full.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in manifest.get("records", []):
            if (scene_info.root_path /
                    record["depth"]).resolve() == depth_path.resolve():
                return scene_info.root_path / record["image"]
    if not scene_info.image_path:
        return None
    matches = sorted(scene_info.image_path.glob(f"{depth_path.stem}.*"))
    if matches:
        return matches[0]
    return None


def scene_statistics(scene_info, config: DepthEvaluationConfig) -> dict:
    """Computes GT statistics of one scene under its resolved limits and masks.

    Args:
        scene_info: The scene, as listed by its dataset.
        config: The scene's resolved depth evaluation config.

    Returns:
        The statistics, as written to the report.
    """
    gt_files = depth_prediction_evaluator.list_scene_gt_depth_files(scene_info)
    masks = depth_prediction_evaluator.materialized_mask_paths_by_depth(
        scene_info)
    intrinsics = depth_prediction_evaluator.materialized_intrinsics_by_depth(
        scene_info)
    size_mismatch = []
    resized_frames = []
    missing_images = []
    nonfinite_frames = []
    empty_frames = []
    valid_fractions = []
    samples = []
    rng = np.random.default_rng(0)
    for depth_path in gt_files:
        depth = depth_prediction_evaluator.load_depth_map(
            depth_path, scale=config.gt_depth_scale)
        image_path = image_for_depth(scene_info, depth_path)
        if image_path is None or not image_path.is_file():
            missing_images.append(depth_path.name)
        else:
            with Image.open(image_path) as image:
                width, height = image.size
            if (width, height) != (depth.shape[1], depth.shape[0]):
                # A lower-resolution GT is fine (predictions are resized to
                # it) if the aspect ratio agrees.
                aspect_error = abs((width / height) /
                                   (depth.shape[1] / depth.shape[0]) - 1.0)
                if aspect_error > 0.02:
                    size_mismatch.append(depth_path.name)
                else:
                    resized_frames.append(depth_path.name)
        if not np.isfinite(depth).all():
            nonfinite_frames.append(depth_path.name)
        valid = np.isfinite(depth) & (depth > config.min_depth)
        if config.max_depth is not None:
            valid &= depth < config.max_depth
        mask_path = masks.get(
            depth_prediction_evaluator._canonical_existing_path(depth_path))
        if mask_path is not None:
            valid &= depth_prediction_evaluator.load_depth_map(mask_path) > 0
        valid_fractions.append(float(valid.mean()))
        if not valid.any():
            empty_frames.append(depth_path.name)
            continue
        values = depth[valid]
        samples.append(
            rng.choice(values, size=min(values.size, 2000), replace=False))
    if samples:
        values = np.concatenate(samples)
    else:
        values = np.zeros(0)
    percentiles = []
    if values.size:
        percentiles = np.percentile(values, [0, 1, 50, 99, 100]).tolist()
    return {
        "num_frames":
            len(gt_files),
        "num_frames_with_intrinsics":
            len(intrinsics),
        "num_frames_with_mask":
            len(masks),
        "size_mismatch":
            size_mismatch,
        "num_frames_with_lower_resolution_gt":
            len(resized_frames),
        "missing_images":
            missing_images,
        "nonfinite_frames":
            nonfinite_frames,
        "empty_frames":
            empty_frames,
        "valid_fraction_mean":
            float(np.mean(valid_fractions)) if valid_fractions else 0.0,
        "valid_fraction_min":
            float(np.min(valid_fractions)) if valid_fractions else 0.0,
        "depth_percentiles_0_1_50_99_100":
            percentiles,
    }


def oracle_check(scene_info, config: DepthEvaluationConfig, num_frames: int,
                 seed: int) -> dict:
    """Evaluates GT and an affine copy of GT as predictions.

    Both must score perfectly after scale+shift alignment.

    Args:
        scene_info: The scene, as listed by its dataset.
        config: The scene's resolved depth evaluation config.
        num_frames: How many random frames to evaluate.
        seed: The seed for the frame choice and the affine maps.

    Returns:
        Per-mode frame counts and metrics, and "ok" when both passed.
    """
    gt_files = depth_prediction_evaluator.list_scene_gt_depth_files(scene_info)
    rng = np.random.default_rng(seed)
    chosen = sorted(
        rng.choice(len(gt_files),
                   size=min(num_frames, len(gt_files)),
                   replace=False).tolist())
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for mode in ("identity", "affine"):
            pred_dir = Path(tmp) / mode / "depths"
            pred_dir.mkdir(parents=True)
            for index in chosen:
                depth_path = gt_files[index]
                depth = depth_prediction_evaluator.load_depth_map(
                    depth_path, scale=config.gt_depth_scale)
                depth = np.where(np.isfinite(depth), depth, 0.0)
                if mode == "affine":
                    median = 1.0
                    if (depth > 0).any():
                        median = float(np.median(depth[depth > 0]))
                    scale = float(rng.uniform(0.2, 5.0))
                    shift = float(rng.uniform(0.0, 1.0)) * median
                    # The prediction is GT = scale * pred + shift solved for
                    # pred.
                    depth = (depth - shift) / scale
                np.save(pred_dir / f"{depth_path.stem}.npy",
                        depth.astype(np.float32))
            # evaluate_depth_scene resolves the scene's limits itself;
            # geometry metrics are not part of the check.
            scene_result = depth_prediction_evaluator.evaluate_depth_scene(
                scene_info,
                pred_dir.parent,
                DepthEvaluationConfig(compute_geometry_metrics=False),
            )
            metrics = scene_result["metrics"]
            results[mode] = {
                "frames": scene_result["num_evaluated_frames"],
                "abs_relative_difference": metrics["abs_relative_difference"],
                "delta125_acc": metrics["delta125_acc"],
            }
    results["ok"] = all(
        result["frames"] == len(chosen) and result["abs_relative_difference"] <
        ORACLE_ABSREL_TOLERANCE and result["delta125_acc"] > 1.0 - 1e-6
        for result in (results["identity"], results["affine"]))
    return results


def write_overlays(scene_info, config: DepthEvaluationConfig, output_dir: Path,
                   num_frames: int) -> list[str]:
    """Writes RGB | GT depth | depth-edge overlay panels for a few frames.

    Args:
        scene_info: The scene, as listed by its dataset.
        config: The scene's resolved depth evaluation config.
        output_dir: The directory for the panels.
        num_frames: How many evenly spaced frames to draw.

    Returns:
        The written panel paths.
    """
    gt_files = depth_prediction_evaluator.list_scene_gt_depth_files(scene_info)
    if not gt_files or num_frames <= 0:
        return []
    positions = np.linspace(0,
                            len(gt_files) - 1,
                            min(num_frames, len(gt_files))).round().astype(int)
    written = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for position in sorted(set(positions.tolist())):
        depth_path = gt_files[position]
        image_path = image_for_depth(scene_info, depth_path)
        if image_path is None:
            continue
        depth = depth_prediction_evaluator.load_depth_map(
            depth_path, scale=config.gt_depth_scale)
        with Image.open(image_path) as image:
            rgb = np.asarray(
                image.convert("RGB").resize((depth.shape[1], depth.shape[0])))
        colored = materialize_depth_visualizations.depth_to_rgb(depth)
        log_depth = np.log(
            np.where(np.isfinite(depth) & (depth > 0), depth, np.nan))
        grad_y = np.abs(np.diff(log_depth, axis=0, prepend=log_depth[:1]))
        grad_x = np.abs(np.diff(log_depth, axis=1, prepend=log_depth[:, :1]))
        grad = np.nan_to_num(grad_y) + np.nan_to_num(grad_x)
        edges = grad > 0.05
        overlay = rgb.copy()
        overlay[edges] = (255, 0, 255)
        panel = np.concatenate([rgb, colored, overlay], axis=1)
        scale = min(1.0, 2400.0 / panel.shape[1])
        panel_image = Image.fromarray(panel)
        if scale < 1.0:
            panel_image = panel_image.resize(
                (int(panel.shape[1] * scale), int(panel.shape[0] * scale)))
        target = output_dir / f"{depth_path.stem}.jpg"
        panel_image.save(target, quality=90)
        written.append(str(target))
    return written


def check_dataset(args: argparse.Namespace, dataset: str) -> dict:
    """Runs all checks on every scene of one dataset.

    Args:
        args: The parsed command line.
        dataset: The MonoDepth dataset name.

    Returns:
        The dataset report, with "ok" when every scene passed.
    """
    spec = dataset_registry.find_dataset_spec("MonoDepth", dataset)
    scenes = spec.create_dataset(args.benchmark_root).list_scenes()
    report = {"dataset": dataset, "scenes": {}, "ok": bool(scenes)}
    for scene_info in scenes:
        config = depth_prediction_evaluator.resolve_depth_config(
            scene_info, DepthEvaluationConfig())
        stats = scene_statistics(scene_info, config)
        oracle = oracle_check(scene_info, config, args.oracle_frames, args.seed)
        overlays = write_overlays(scene_info, config,
                                  args.output_dir / dataset / scene_info.scene,
                                  args.overlay_frames)
        intrinsics_complete = stats["num_frames_with_intrinsics"] in (
            0, stats["num_frames"])
        scene_ok = (oracle["ok"] and stats["num_frames"] > 0 and
                    intrinsics_complete and not stats["size_mismatch"] and
                    not stats["missing_images"] and not stats["empty_frames"])
        report["ok"] &= scene_ok
        report["scenes"][scene_info.scene] = {
            "ok": scene_ok,
            "limits": {
                "min_depth": config.min_depth,
                "max_depth": config.max_depth,
                "gt_depth_scale": config.gt_depth_scale,
                "filter_gt_depth_range": config.filter_gt_depth_range,
            },
            "statistics": stats,
            "oracle": oracle,
            "overlays": overlays,
        }
        percentiles = [
            round(v, 3) for v in stats["depth_percentiles_0_1_50_99_100"][1:4]
        ]
        identity_absrel = oracle["identity"]["abs_relative_difference"]
        affine_absrel = oracle["affine"]["abs_relative_difference"]
        print(
            f"{dataset}/{scene_info.scene}: ok={scene_ok} "
            f"frames={stats['num_frames']} "
            f"valid={stats['valid_fraction_mean']:.3f} depth p1/p50/p99="
            f"{percentiles} "
            f"oracle absrel identity/affine={identity_absrel:.2e}/"
            f"{affine_absrel:.2e}",
            flush=True,
        )
    return report


def main() -> None:
    """Checks the selected datasets and writes one JSON report per dataset."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_ok = True
    for dataset in args.datasets:
        report = check_dataset(args, dataset)
        (args.output_dir / f"{dataset}.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        all_ok &= report["ok"]
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
