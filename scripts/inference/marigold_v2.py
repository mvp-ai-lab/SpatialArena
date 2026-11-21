# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Official Marigold V2 depth (Log-stage2, huawei-bayerlab/marigold-v2-0): raw normalized log depth.

Ported from eval_refactor_hsy (45300bb). The official ``scripts/infer.py`` runs in its own interpreter
(``MARIGOLD_V2_PYTHON``, default this one) on only the engine-selected images, staged as symlinks. Its signed
normalized log code ``q`` is kept unmodified in ``raw_log_depths/`` -- the only input ``log_huber_irls_gt`` scores --
and ``depths/`` holds ``exp(q)``, a positive proxy for pairing artifacts only.

The processing size is the native one rounded to 16 unless set: per scene by ``MARIGOLD_V2_SCENE_SIZES``
(``<dataset>/<scene>=<W>x<H>``, comma-separated, e.g. ``marigold/eth3d=1008x672``), else for every scene by
``MARIGOLD_V2_WIDTH`` / ``MARIGOLD_V2_HEIGHT``. Sizes are positive multiples of 16; a fixed-size prediction is restored
to the source grid bilinearly in log-code space.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from adapter_utils import export_depth_predictions, save_prediction_map, write_adapter_metrics
from single_image_depth import parse_args, select_images

METHODS = {"MARIGOLD_V2": "huawei-bayerlab/marigold-v2-0/depth/Log-stage2"}


def export_log_prediction(
    output_path: Path, image_path: str, prediction_path: Path,
    *, processing_size: tuple[int, int] | None = None,  # pylint: disable=redefined-outer-name
) -> None:
    """Keep the log prediction unmodified (restored to the source grid if needed); exp(q) is only a pairing proxy."""
    import cv2

    raw = np.load(prediction_path).astype(np.float32).squeeze()
    with Image.open(image_path) as image:
        expected = (image.height, image.width)
    if processing_size is not None and raw.shape == processing_size[::-1] and np.isfinite(raw).all():
        raw = cv2.resize(raw, expected[::-1], interpolation=cv2.INTER_LINEAR)
    if raw.shape != expected or not np.isfinite(raw).all():
        raise ValueError(f"Invalid Marigold V2 log prediction: {raw.shape}, expected {expected}")
    save_prediction_map(output_path / "raw_log_depths", Path(image_path).stem, raw)
    export_depth_predictions(output_path, [image_path], np.exp(np.clip(raw, -80, 80))[None])


def _checked_size(width: int, height: int) -> tuple[int, int] | None:
    if (width or height) and (width <= 0 or height <= 0 or width % 16 or height % 16):
        raise ValueError("Marigold V2 width and height must both be positive multiples of 16")
    return (width, height) if width else None


def processing_size(data_path: Path | None = None) -> tuple[int, int] | None:
    """(width, height) for the scene at ``data_path``: its MARIGOLD_V2_SCENE_SIZES entry (``<dataset>/<scene>``), else
    MARIGOLD_V2_WIDTH / MARIGOLD_V2_HEIGHT; None for the native size."""
    scene_sizes: dict[str, tuple[int, int] | None] = {}
    for entry in filter(None, (part.strip() for part in os.environ.get("MARIGOLD_V2_SCENE_SIZES", "").split(","))):
        scene, _, size = entry.partition("=")
        width, _, height = size.lower().partition("x")
        try:
            scene_sizes[scene.strip().strip("/")] = _checked_size(int(width), int(height))
        except ValueError as exc:
            raise ValueError(f"Bad MARIGOLD_V2_SCENE_SIZES entry {entry!r} (<dataset>/<scene>=<W>x<H>, positive "
                             "multiples of 16)") from exc
    if data_path is not None:
        key = f"{Path(data_path).parent.name}/{Path(data_path).name}"
        if key in scene_sizes:
            return scene_sizes[key]
    return _checked_size(int(os.environ.get("MARIGOLD_V2_WIDTH") or 0), int(os.environ.get("MARIGOLD_V2_HEIGHT") or 0))


def run() -> None:
    args = parse_args(__doc__, METHODS, "marigold-v2")
    # Fail on a bad size before staging or loading anything.
    size: tuple[int, int] | None = processing_size(getattr(args, "data_path", None))
    image_paths = select_images(args)
    if image_paths is None:
        return
    code = args.code_path.resolve()
    executable = os.environ.get("MARIGOLD_V2_PYTHON", sys.executable)
    args.output_path.mkdir(parents=True, exist_ok=True)
    resolution_args: list[str] = []
    if size is not None:
        width, height = size
        resolution_args = ["--width", str(width), "--height", str(height)]
    started = time.time()
    # Stage only the engine-selected RGB images: the official program must not scan the whole dataset.
    with tempfile.TemporaryDirectory(prefix="marigold-v2-", dir=args.output_path) as tmp:
        work = Path(tmp)
        images = work / "images"
        images.mkdir()
        for index, path in enumerate(image_paths):
            (images / f"{index:06d}{Path(path).suffix.lower()}").symlink_to(Path(path).resolve())
        result = work / "result"
        subprocess.run([
            executable, str(code / "scripts/infer.py"), "--modality", "depth",
            "--checkpoint", str(args.ckpt_path), "--image_dir", str(images),
            "--output_dir", str(result), "--seed", str(args.seed),
        ] + resolution_args, cwd=code, check=True)
        for index, path in enumerate(image_paths):
            export_log_prediction(args.output_path, path, result / "images/predictions_npy" / f"{index:06d}.npy",
                                  processing_size=size)
    write_adapter_metrics(
        output_path=args.output_path, method=args.method, image_paths=image_paths,
        started_at=started, checkpoint=args.ckpt_path, depth_only=True,
        metrics_extra={"track": args.track, "representation": "raw_normalized_log_depth",
                       "required_protocol": "log_huber_irls_gt", "checkpoint_variant": "Log-stage2",
                       "resolution": list(size) if size else "native_multiple_of_16", "seed": args.seed},
        # The official program loads the model itself: its runtime includes that, it is not pure inference.
        runtime={"inference_seconds": time.time() - started, "model_load_seconds": None,
                 "peak_gpu_memory_mib": None, "num_input_images": len(image_paths),
                 "runtime_scope": "official_subprocess_including_model_load"},
    )


if __name__ == "__main__":
    run()
