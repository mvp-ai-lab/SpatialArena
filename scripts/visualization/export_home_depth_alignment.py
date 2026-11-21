# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Export the real depth-alignment frames of the dashboard home page (``js/home_align.js``, "Depth alignment").

Each frame is an iBims-1 image, its GT depth and MoGe's raw (affine-invariant) prediction from the stored MonoDepth
artifacts, with the IRLS trace of the benchmark's own full-resolution fit (``depth_huber_irls_gt``): least squares on
every GT pixel, then Huber IRLS (delta 0.05 of the median GT depth) until the scale and shift move by less than 1e-7.
The fitted scale is checked against the fit stored with the artifacts.

Outputs, in ``--output`` (default ``app/webpage/assets/align``):

- ``<frame>.jpg``: the RGB image;
- ``<frame>.bin``: GT depth (uint16 over [0, gtMax], 0 = no GT), then the prediction (uint16 over
  [predMin, predMax]), both at half resolution, little endian;
- ``manifest.json``: per frame the sizes, ranges, GT pixel count and the trace (scale, shift, Huber loss, AbsRel,
  delta < 1.25 at every step), plus the dataset credit.

iBims-1 is CC BY 4.0; the page credits it. Usage: ``python scripts/visualization/export_home_depth_alignment.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS = ("_ffb_eval/monodepth-native-depth-huber-irls-20261028/artifacts/full_standard/"
                     "benchmark_full-depth_MOGE/MonoDepth/ibims1/MOGE/ibims1")
DEFAULT_FRAMES = ("corridor_02", "office_08", "corridor_01")
HUBER_DELTA = 0.05
STRIDE = 2
MIN_DEPTH = 0.001  # MonoDepth/ibims1 metadata.json min_depth


def affine_fit(pred: np.ndarray, gt: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float]:
    """Weighted least-squares ``gt ~ a * pred + b``."""
    w = np.ones_like(pred) if weights is None else weights
    sw, sp, sg = w.sum(), (w * pred).sum(), (w * gt).sum()
    spp, spg = (w * pred * pred).sum(), (w * pred * gt).sum()
    det = sw * spp - sp * sp
    return float((sw * spg - sp * sg) / det), float((spp * sg - sp * spg) / det)


def irls_trace(pred: np.ndarray, gt: np.ndarray) -> tuple[list[dict], float]:
    """Every step of the Huber IRLS fit (step 0 = the least-squares start) and the residual unit (median GT)."""
    unit = float(np.median(gt))

    def stats(a: float, b: float, step: int) -> dict:
        aligned = a * pred + b
        r = np.abs(aligned - gt) / unit
        huber = np.where(r <= HUBER_DELTA, 0.5 * r * r, HUBER_DELTA * (r - 0.5 * HUBER_DELTA)).mean()
        ratio = np.maximum(aligned / gt, gt / np.maximum(aligned, 1e-9))
        return {"step": step, "scale": a, "shift": b, "loss": float(huber),
                "absRel": float(np.mean(np.abs(aligned - gt) / gt)),
                "delta1": float(np.mean((aligned > 0) & (ratio < 1.25)))}

    a, b = affine_fit(pred, gt)
    steps = [stats(a, b, 0)]
    for step in range(1, 101):
        r = np.abs(a * pred + b - gt) / unit
        weights = np.where(r <= HUBER_DELTA, 1.0, HUBER_DELTA / np.maximum(r, 1e-12))
        a2, b2 = affine_fit(pred, gt, weights)
        done = abs(a2 - a) + abs(b2 - b) < 1e-7
        a, b = a2, b2
        steps.append(stats(a, b, step))
        if done:
            break
    return steps, unit


def export_frame(name: str, gt_dir: Path, image_dir: Path, artifacts: Path, out: Path) -> dict:
    gt = np.load(gt_dir / f"{name}.npy").astype(np.float64)
    pred = np.load(artifacts / "depths" / f"{name}.npy").astype(np.float64)
    valid = np.isfinite(gt) & (gt > MIN_DEPTH)
    steps, unit = irls_trace(pred[valid], gt[valid])
    stored = json.loads((artifacts / "depth_huber_irls_gt_fits" / f"{name}.json").read_text())
    if abs(stored["scale"] - steps[-1]["scale"]) > 1e-4:
        raise RuntimeError(f"{name}: scale {steps[-1]['scale']} differs from the stored fit's {stored['scale']}")
    gt_s, pred_s, valid_s = gt[::STRIDE, ::STRIDE], pred[::STRIDE, ::STRIDE], valid[::STRIDE, ::STRIDE]
    height, width = gt_s.shape
    gt_max = float(np.ceil(gt_s[valid_s].max() * 10) / 10)
    pred_min, pred_max = float(pred_s.min()), float(pred_s.max())
    gt_q = np.where(valid_s, np.clip(np.round(gt_s / gt_max * 65534) + 1, 1, 65535), 0).astype("<u2")
    pred_q = np.round((pred_s - pred_min) / (pred_max - pred_min) * 65535).astype("<u2")
    (out / f"{name}.bin").write_bytes(gt_q.tobytes() + pred_q.tobytes())
    Image.open(image_dir / f"{name}.jpg").convert("RGB").save(out / f"{name}.jpg", quality=84, optimize=True,
                                                              progressive=True)
    return {"name": name, "width": width, "height": height, "gtMax": gt_max, "predMin": pred_min,
            "predMax": pred_max, "unit": unit, "fitPixels": int(valid.sum()), "holes": float(1 - valid.mean()),
            "steps": steps}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("frames", nargs="*", default=list(DEFAULT_FRAMES), help="iBims-1 frame names.")
    parser.add_argument("--benchmark_root", type=Path, default=Path.home() / "datasets/benchmark")
    parser.add_argument("--artifacts", default=DEFAULT_ARTIFACTS,
                        help="MoGe iBims-1 artifact folder, relative to --benchmark_root.")
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "app/webpage/assets/align")
    args = parser.parse_args()
    scene = args.benchmark_root / "MonoDepth/ibims1/ibims1"
    args.output.mkdir(parents=True, exist_ok=True)
    frames = [export_frame(name, scene / "depths", scene / "images", args.benchmark_root / args.artifacts,
                           args.output) for name in args.frames]
    manifest = {"dataset": "iBims-1", "license": "CC BY 4.0", "credit": "Koch et al., ECCV Workshops 2018, CC BY 4.0",
                "model": "MoGe (v1, ViT-L)",
                "protocol": "Huber (δ = 0.05, relative to median GT), IRLS from a full-pixel least-squares start",
                "frames": frames}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    for frame in frames:
        first, last = frame["steps"][0], frame["steps"][-1]
        print(f"{frame['name']}: {len(frame['steps']) - 1} IRLS steps, AbsRel {100 * first['absRel']:.2f} % -> "
              f"{100 * last['absRel']:.2f} %, {100 * frame['holes']:.1f} % without GT")


if __name__ == "__main__":
    main()
