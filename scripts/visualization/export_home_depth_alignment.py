# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Exports the real depth-alignment frames of the dashboard home page.

The frames feed `js/home_align.js` ("Depth alignment"). Each frame is an
iBims-1 image, its GT depth and MoGe's raw (affine-invariant) prediction from
the stored MonoDepth artifacts, with the IRLS trace of the benchmark's own
full-resolution fit (`depth_huber_irls_gt`): least squares on every GT pixel,
then Huber IRLS (delta 0.05 of the median GT depth) until the scale and shift
move by less than 1e-7. The fitted scale is checked against the fit stored
with the artifacts.

Outputs, in `--output` (default `app/webpage/assets/align`):

- `<frame>.jpg`: the RGB image;
- `<frame>.bin`: GT depth (uint16 over [0, gtMax], 0 = no GT), then the
  prediction (uint16 over [predMin, predMax]), both at half resolution, little
  endian;
- `manifest.json`: per frame the sizes, ranges, GT pixel count and the trace
  (scale, shift, Huber loss, AbsRel, delta < 1.25 at every step), plus the
  dataset credit.

iBims-1 is CC BY 4.0; the page credits it. Usage:
`python scripts/visualization/export_home_depth_alignment.py`.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from PIL import Image

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS = ("_ffb_eval/monodepth-native-depth-huber-irls-20261028/"
                     "artifacts/full_standard/benchmark_full-depth_MOGE/"
                     "MonoDepth/ibims1/MOGE/ibims1")
DEFAULT_FRAMES = ("corridor_02", "office_08", "corridor_01")
HUBER_DELTA = 0.05
STRIDE = 2
# The min_depth of MonoDepth/ibims1/metadata.json.
MIN_DEPTH = 0.001


def affine_fit(pred: np.ndarray,
               gt: np.ndarray,
               weights: np.ndarray | None = None) -> tuple[float, float]:
    """Fits `gt ~ a * pred + b` by weighted least squares.

    Args:
        pred: The predicted values.
        gt: The ground-truth values.
        weights: Per-value weights; None weighs every value 1.

    Returns:
        The scale `a` and shift `b`.
    """
    if weights is None:
        w = np.ones_like(pred)
    else:
        w = weights
    sw, sp, sg = w.sum(), (w * pred).sum(), (w * gt).sum()
    spp, spg = (w * pred * pred).sum(), (w * pred * gt).sum()
    det = sw * spp - sp * sp
    return float((sw * spg - sp * sg) / det), float((spp * sg - sp * spg) / det)


def irls_trace(pred: np.ndarray, gt: np.ndarray) -> tuple[list[dict], float]:
    """Runs the Huber IRLS fit and records every step.

    Args:
        pred: The predicted depths of the GT pixels.
        gt: The GT depths.

    Returns:
        The steps (step 0 is the least-squares start) and the residual unit
        (the median GT depth).
    """
    unit = float(np.median(gt))

    def stats(a: float, b: float, step: int) -> dict:
        aligned = a * pred + b
        residual = np.abs(aligned - gt) / unit
        huber = np.where(residual <= HUBER_DELTA, 0.5 * residual * residual,
                         HUBER_DELTA * (residual - 0.5 * HUBER_DELTA)).mean()
        ratio = np.maximum(aligned / gt, gt / np.maximum(aligned, 1e-9))
        return {
            "step": step,
            "scale": a,
            "shift": b,
            "loss": float(huber),
            "absRel": float(np.mean(np.abs(aligned - gt) / gt)),
            "delta1": float(np.mean((aligned > 0) & (ratio < 1.25)))
        }

    a, b = affine_fit(pred, gt)
    steps = [stats(a, b, 0)]
    for step in range(1, 101):
        residual = np.abs(a * pred + b - gt) / unit
        weights = np.where(residual <= HUBER_DELTA, 1.0,
                           HUBER_DELTA / np.maximum(residual, 1e-12))
        a2, b2 = affine_fit(pred, gt, weights)
        done = abs(a2 - a) + abs(b2 - b) < 1e-7
        a, b = a2, b2
        steps.append(stats(a, b, step))
        if done:
            break
    return steps, unit


def export_frame(name: str, gt_dir: pathlib.Path, image_dir: pathlib.Path,
                 artifacts: pathlib.Path, out: pathlib.Path) -> dict:
    """Writes one frame's image and depth files and returns its manifest entry.

    Args:
        name: The iBims-1 frame name.
        gt_dir: The folder of the GT depth maps.
        image_dir: The folder of the RGB images.
        artifacts: The MoGe artifact folder.
        out: The output folder.

    Returns:
        The frame's sizes, ranges, GT pixel count and IRLS steps.

    Raises:
        RuntimeError: If the fitted scale differs from the stored fit.
    """
    gt = np.load(gt_dir / f"{name}.npy").astype(np.float64)
    pred = np.load(artifacts / "depths" / f"{name}.npy").astype(np.float64)
    valid = np.isfinite(gt) & (gt > MIN_DEPTH)
    steps, unit = irls_trace(pred[valid], gt[valid])
    stored = json.loads(
        (artifacts / "depth_huber_irls_gt_fits" / f"{name}.json").read_text())
    if abs(stored["scale"] - steps[-1]["scale"]) > 1e-4:
        raise RuntimeError(f"{name}: scale {steps[-1]['scale']} differs from "
                           f"the stored fit's {stored['scale']}")
    gt_s = gt[::STRIDE, ::STRIDE]
    pred_s = pred[::STRIDE, ::STRIDE]
    valid_s = valid[::STRIDE, ::STRIDE]
    height, width = gt_s.shape
    gt_max = float(np.ceil(gt_s[valid_s].max() * 10) / 10)
    pred_min, pred_max = float(pred_s.min()), float(pred_s.max())
    gt_q = np.where(valid_s,
                    np.clip(np.round(gt_s / gt_max * 65534) + 1, 1, 65535),
                    0).astype("<u2")
    pred_q = np.round(
        (pred_s - pred_min) / (pred_max - pred_min) * 65535).astype("<u2")
    (out / f"{name}.bin").write_bytes(gt_q.tobytes() + pred_q.tobytes())
    image = Image.open(image_dir / f"{name}.jpg").convert("RGB")
    image.save(out / f"{name}.jpg", quality=84, optimize=True, progressive=True)
    return {
        "name": name,
        "width": width,
        "height": height,
        "gtMax": gt_max,
        "predMin": pred_min,
        "predMax": pred_max,
        "unit": unit,
        "fitPixels": int(valid.sum()),
        "holes": float(1 - valid.mean()),
        "steps": steps
    }


def main() -> None:
    """Exports the frames, writes the manifest and prints each fit."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("frames",
                        nargs="*",
                        default=list(DEFAULT_FRAMES),
                        help="iBims-1 frame names.")
    parser.add_argument("--benchmark_root",
                        type=pathlib.Path,
                        default=pathlib.Path.home() / "datasets/benchmark")
    parser.add_argument(
        "--artifacts",
        default=DEFAULT_ARTIFACTS,
        help="MoGe iBims-1 artifact folder, relative to --benchmark_root.")
    parser.add_argument("--output",
                        type=pathlib.Path,
                        default=ROOT_DIR / "app/webpage/assets/align")
    args = parser.parse_args()
    scene = args.benchmark_root / "MonoDepth/ibims1/ibims1"
    args.output.mkdir(parents=True, exist_ok=True)
    frames = [
        export_frame(name, scene / "depths", scene / "images",
                     args.benchmark_root / args.artifacts, args.output)
        for name in args.frames
    ]
    manifest = {
        "dataset": "iBims-1",
        "license": "CC BY 4.0",
        "credit": "Koch et al., ECCV Workshops 2018, CC BY 4.0",
        "model": "MoGe (v1, ViT-L)",
        "protocol": ("Huber (δ = 0.05, relative to median GT), IRLS from a "
                     "full-pixel least-squares start"),
        "frames": frames
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False))
    for frame in frames:
        first, last = frame["steps"][0], frame["steps"][-1]
        print(f"{frame['name']}: {len(frame['steps']) - 1} IRLS steps, "
              f"AbsRel {100 * first['absRel']:.2f} % -> "
              f"{100 * last['absRel']:.2f} %, "
              f"{100 * frame['holes']:.1f} % without GT")


if __name__ == "__main__":
    main()
