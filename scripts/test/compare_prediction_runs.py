#!/usr/bin/env python3
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Compares a rerun's predictions and scores with a reference run.

Both sides are evaluation roots (<root>/artifacts/full_standard/... and
<root>/reports/...). Each candidate case is matched by its relative path in
one of the --reference roots (the release splits tasks over several roots),
and compared on:

    depths/, confidences/   the same frames, shapes and valid pixels, and
                            the relative difference of the values;
    sparse/<subdir>         the same images and camera sizes, and the
                            difference of intrinsics and poses;
    metrics.json            every field except timing and location;
    reports/.../eval_*.json the difference of each score.

A case is "identical" (bit-for-bit), "close" (every difference within
--rtol; half-precision GPU runs need not reproduce bit-for-bit) or
"DIFFERENT".

Usage:

    compare_prediction_runs.py --reference ROOT [ROOT ...] \\
        --candidate ROOT [--rtol 1e-3] [--json OUT]
"""

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

# pylint: disable=wrong-import-position
from spatialarena.eval import depth_prediction_evaluator  # noqa: E402
from spatialarena.geometry import read_write_model  # noqa: E402
# pylint: enable=wrong-import-position

TIMING_KEYS = {
    "inference time",
    "inference_seconds",
    "model_load_seconds",
    "peak_gpu_memory_mib",
    "Peak GPU Memory",
}
# Fields that say where a run happened, not what it computed.
LOCATION_KEYS = {"Scene", "checkpoint"}
MAP_FOLDERS = ("depths", "confidences")
DIFF_KEYS = (
    "max_rel_diff",
    "max_intrinsics_rel_diff",
    "max_pose_rel_diff",
    "valid_mismatch_fraction",
)


def parse_args():
    """Parses the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference",
                        type=pathlib.Path,
                        nargs="+",
                        required=True,
                        help="Reference evaluation root(s).")
    parser.add_argument("--candidate",
                        type=pathlib.Path,
                        required=True,
                        help="Evaluation root of the rerun.")
    parser.add_argument("--rtol",
                        type=float,
                        default=1e-3,
                        help="Relative tolerance of a 'close' case.")
    parser.add_argument("--json",
                        type=pathlib.Path,
                        default=None,
                        help="Also write the per-case results here.")
    return parser.parse_args()


def find_reference(roots, relative):
    """Returns `relative` under the first root that has it, or None."""
    for root in roots:
        if (root / relative).exists():
            return root / relative
    return None


def max_relative_difference(a, b):
    """Returns max |a - b| / max(|a|, |b|), with 0/0 counted as 0."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    scale = np.maximum(np.abs(a), np.abs(b))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(scale > 0, np.abs(a - b) / scale, 0.0)
    if rel.size == 0:
        return 0.0
    return float(np.nanmax(rel))


def load_map(path):
    """Loads a depth or confidence map as float64."""
    return np.asarray(depth_prediction_evaluator.load_depth_map(path),
                      dtype=np.float64)


def compare_map_pair(a, b, result):
    """Adds the comparison of one frame's maps to `result`.

    Returns:
        The median relative difference of the frame's valid pixels.
    """
    if a.shape != b.shape:
        result["shape_mismatch"] += 1
        result["identical"] = False
        return None
    if np.array_equal(a, b, equal_nan=True):
        return 0.0

    result["identical"] = False
    valid_a = np.isfinite(a) & (a > 0)
    valid_b = np.isfinite(b) & (b > 0)
    mismatch = float(np.mean(valid_a != valid_b))
    result["valid_mismatch_fraction"] = max(result["valid_mismatch_fraction"],
                                            mismatch)
    both = valid_a & valid_b
    if not both.any():
        return None
    scale = np.maximum(np.abs(a[both]), np.abs(b[both]))
    rel = np.abs(a[both] - b[both]) / scale
    result["max_rel_diff"] = max(result["max_rel_diff"], float(rel.max()))
    return float(np.median(rel))


def compare_maps(reference, candidate, folder):
    """Compares the per-frame maps of one folder of a case.

    Args:
        reference: Reference case directory.
        candidate: Candidate case directory.
        folder: "depths" or "confidences".

    Returns:
        A dict of the comparison.
    """
    ref_names = {path.name for path in (reference / folder).glob("*.npz")}
    cand_names = {path.name for path in (candidate / folder).glob("*.npz")}
    result = {
        "frames": len(ref_names),
        "missing": sorted(ref_names - cand_names)[:5],
        "extra": sorted(cand_names - ref_names)[:5],
        "identical": ref_names == cand_names,
        "shape_mismatch": 0,
        "valid_mismatch_fraction": 0.0,
        "max_rel_diff": 0.0,
        "median_rel_diff": 0.0,
    }
    medians = []
    for name in sorted(ref_names & cand_names):
        median = compare_map_pair(load_map(reference / folder / name),
                                  load_map(candidate / folder / name), result)
        if median is not None:
            medians.append(median)
    if medians:
        result["median_rel_diff"] = float(np.median(medians))
    return result


def compare_colmap_model(ref_model, cand_model, result):
    """Adds the comparison of one COLMAP model to `result`."""
    ref_cameras, ref_images, ref_points = read_write_model.read_model(
        str(ref_model))
    cand_cameras, cand_images, cand_points = read_write_model.read_model(
        str(cand_model))
    ref_by_name = {image.name: image for image in ref_images.values()}
    cand_by_name = {image.name: image for image in cand_images.values()}
    if set(ref_by_name) != set(cand_by_name):
        result["problems"].append(f"image names differ ({len(ref_by_name)} "
                                  f"vs {len(cand_by_name)})")
        result["identical"] = False
    if len(ref_points) != len(cand_points):
        result["problems"].append(
            f"points3D {len(ref_points)} vs {len(cand_points)}")

    for name in sorted(set(ref_by_name) & set(cand_by_name)):
        ref_image, cand_image = ref_by_name[name], cand_by_name[name]
        ref_camera = ref_cameras[ref_image.camera_id]
        cand_camera = cand_cameras[cand_image.camera_id]
        ref_size = (ref_camera.model, ref_camera.width, ref_camera.height)
        cand_size = (cand_camera.model, cand_camera.width, cand_camera.height)
        if ref_size != cand_size:
            result["problems"].append(f"{name}: camera {ref_size} vs "
                                      f"{cand_size}")
            result["identical"] = False
            continue
        intrinsics_diff = max_relative_difference(ref_camera.params,
                                                  cand_camera.params)
        pose_diff = max_relative_difference(
            np.concatenate([ref_image.qvec, ref_image.tvec]),
            np.concatenate([cand_image.qvec, cand_image.tvec]))
        result["max_intrinsics_rel_diff"] = max(
            result["max_intrinsics_rel_diff"], intrinsics_diff)
        result["max_pose_rel_diff"] = max(result["max_pose_rel_diff"],
                                          pose_diff)
        if intrinsics_diff or pose_diff:
            result["identical"] = False


def compare_cameras(reference, candidate):
    """Compares the COLMAP models of a case, or returns None without any."""
    sparse_dir = reference / "sparse"
    if not sparse_dir.is_dir():
        return None
    ref_models = sorted(path for path in sparse_dir.glob("*") if path.is_dir())
    if not ref_models:
        return None

    result = {
        "identical": True,
        "max_intrinsics_rel_diff": 0.0,
        "max_pose_rel_diff": 0.0,
        "problems": [],
    }
    for ref_model in ref_models:
        cand_model = candidate / "sparse" / ref_model.name
        if cand_model.is_dir():
            compare_colmap_model(ref_model, cand_model, result)
        else:
            result["problems"].append(f"missing sparse/{ref_model.name}")
            result["identical"] = False
    result["problems"] = result["problems"][:5]
    return result


def compare_metrics_json(reference, candidate):
    """Compares metrics.json except the timing and location fields."""
    ref = json.loads((reference / "metrics.json").read_text(encoding="utf-8"))
    cand = json.loads((candidate / "metrics.json").read_text(encoding="utf-8"))
    skip = TIMING_KEYS | LOCATION_KEYS
    changed = {
        key: [ref[key], cand[key]]
        for key in ref
        if key in cand and key not in skip and ref[key] != cand[key]
    }
    return {
        "changed": changed,
        "dropped": sorted(set(ref) - set(cand) - skip),
        "added": sorted(set(cand) - set(ref) - skip),
    }


def compare_scores(reference, candidate):
    """Compares the per-scene evaluation files of a case."""
    result = {
        "files": 0,
        "identical": True,
        "max_rel_diff": 0.0,
        "worst": None,
        "missing": [],
    }
    for cand_file in sorted(candidate.glob("eval_*.json")):
        ref_file = reference / cand_file.name
        if not ref_file.is_file():
            result["missing"].append(cand_file.name)
            continue
        result["files"] += 1
        ref_scores = json.loads(ref_file.read_text(encoding="utf-8"))
        cand_scores = json.loads(cand_file.read_text(encoding="utf-8"))
        cand_metrics = cand_scores.get("metrics", {})
        for key, ref_value in ref_scores.get("metrics", {}).items():
            cand_value = cand_metrics.get(key)
            if ref_value == cand_value:
                continue
            result["identical"] = False
            numbers = (int, float)
            if not (isinstance(ref_value, numbers) and
                    isinstance(cand_value, numbers)):
                continue
            rel = max_relative_difference(ref_value, cand_value)
            if rel > result["max_rel_diff"]:
                result["max_rel_diff"] = rel
                result["worst"] = [
                    f"{cand_file.name}:{key}", ref_value, cand_value
                ]
    return result


def classify(case, rtol):
    """Returns "identical", "close" or "DIFFERENT" for a compared case."""
    parts = [
        case[key]
        for key in (*MAP_FOLDERS, "cameras", "scores")
        if case.get(key)
    ]
    structural = any(
        part.get("missing") or part.get("extra") or part.get("problems") or
        part.get("shape_mismatch") for part in parts)
    if case["metrics_json"]["changed"] or structural:
        return "DIFFERENT"
    if all(part["identical"] for part in parts):
        return "identical"

    worst = max([part.get(key, 0.0) for part in parts for key in DIFF_KEYS],
                default=0.0)
    if worst <= rtol:
        return "close"
    return "DIFFERENT"


def compare_case(case_dir, relative, args):
    """Compares one candidate case with its reference."""
    references = [
        root / "artifacts" / "full_standard" for root in args.reference
    ]
    case = {"case": str(relative)}
    reference = find_reference(references, relative)
    if reference is None:
        case["status"] = "no reference"
        return case

    for folder in MAP_FOLDERS:
        if (reference / folder).is_dir() or (case_dir / folder).is_dir():
            case[folder] = compare_maps(reference, case_dir, folder)
    case["cameras"] = compare_cameras(reference, case_dir)
    case["metrics_json"] = compare_metrics_json(reference, case_dir)

    experiment, task, dataset, _, scene = relative.parts
    _, suite, method_id = experiment.split("_", 2)
    report_path = pathlib.Path(suite, method_id, task, dataset, scene)
    for report_group in sorted((args.candidate / "reports").glob("*")):
        cand_report = report_group / report_path
        if not cand_report.is_dir():
            continue
        ref_report = find_reference(
            [root / "reports" / report_group.name for root in args.reference],
            report_path)
        if ref_report is not None:
            case["scores"] = compare_scores(ref_report, cand_report)

    case["status"] = classify(case, args.rtol)
    return case


def describe(case):
    """Returns the printed lines of one case."""
    lines = [f"{case['status']:<12} {case['case']}"]
    for folder in MAP_FOLDERS:
        part = case.get(folder)
        if not part or part["identical"]:
            continue
        line = (f"{folder}: max rel {part['max_rel_diff']:.2e} "
                f"median {part['median_rel_diff']:.2e}")
        if part["valid_mismatch_fraction"]:
            line += f" valid-mask {part['valid_mismatch_fraction']:.1e}"
        if part["missing"]:
            line += f" missing {part['missing']}"
        lines.append(line)
    cameras = case.get("cameras")
    if cameras and not cameras["identical"]:
        line = (f"cameras: K {cameras['max_intrinsics_rel_diff']:.2e} "
                f"pose {cameras['max_pose_rel_diff']:.2e}")
        if cameras["problems"]:
            line += f" {cameras['problems']}"
        lines.append(line)
    scores = case.get("scores")
    if scores and not scores["identical"]:
        lines.append(f"scores: max rel {scores['max_rel_diff']:.2e} "
                     f"worst {scores['worst']}")
    meta = case.get("metrics_json", {})
    if meta.get("changed") or meta.get("dropped"):
        lines.append(f"metrics.json changed {meta['changed']} "
                     f"dropped {meta['dropped']}")
    return lines


def main():
    """Compares every candidate case and prints one block per case.

    Returns:
        The exit status: 1 if a case is "DIFFERENT", else 0.
    """
    args = parse_args()
    artifact_root = args.candidate / "artifacts" / "full_standard"
    cases = []
    for metrics_file in sorted(artifact_root.glob("*/*/*/*/*/metrics.json")):
        case_dir = metrics_file.parent
        relative = case_dir.relative_to(artifact_root)
        cases.append(compare_case(case_dir, relative, args))

    counts = {}
    for case in cases:
        print("\n    ".join(describe(case)))
        counts[case["status"]] = counts.get(case["status"], 0) + 1
    print()
    print(", ".join(
        f"{count} {status}" for status, count in sorted(counts.items())))
    if args.json:
        args.json.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    if counts.get("DIFFERENT"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
