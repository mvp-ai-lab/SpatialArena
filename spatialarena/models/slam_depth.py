# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Per-frame depth and confidence export of the SLAM models.

With --export_depth (the engine's EXPORT_POSE_DEPTH=1 for SfM / SLAM), the
SLAM models (`streaming_slam`, `long_sequence_slam`) write every returned
frame's predicted depth to depths/<image stem>.npz (float32, compressed) and
the method's own confidence to confidences/<image stem>.npz (float16, its
native scale), exactly like the feed-forward models
(`utils.export_depth_predictions`). The maps are streamed to disk a batch at
a time: a large-range SLAM sequence (up to 18,846 frames) never holds every
frame's map in host memory at once.

Depth is in the scale of the exported trajectory wherever the method's
output allows it:

- In-process streaming methods (LoGeR, ZipMap, LingBot-Map, ABot-Recon)
  predict one consistent sequence: their depth is already in the frame of
  their poses. LoGeR / ABot-Recon give local point maps; depth is their z.
- R3 and Scal3R write their own per-frame maps (R3 depth/ + conf/; Scal3R
  depths/*.exr, already rescaled to its trajectory, no confidence).
- VGGT-Long / Pi-Long predict chunks in their own scales and align them with
  Sim(3)s they do not save. Each chunk's scale is recovered from its camera
  centers against the final poses (`chunk_scale`) and applied to its depth;
  a frame shared by two chunks takes the later chunk, as its pose does.
- VGGT-SLAM aligns submaps with projective SL(4) transforms, so no per-frame
  depth scale exists: its keyframes' depths are exported in their submap's
  frame (depth_frame: submap in metrics.json).
"""

from __future__ import annotations

import os
import pathlib
import time
from typing import Iterable

import numpy as np

from spatialarena.models import utils

WRITE_BATCH = 128


class DepthSink:
    """Writes depth / confidence maps of input frames under their names.

    Frames are addressed by their input index.

    Attributes:
        output_path: The case's output directory.
        image_paths: The input frames, in input order.
        written: Input indices written so far.
        with_confidence: Whether any confidence map was written.
        export_seconds: Time spent exporting (device -> host copies and
            writes). In-process methods stream maps out while they run, so
            this is taken off their inference time.
    """

    def __init__(self, output_path: pathlib.Path,
                 image_paths: list[str]) -> None:
        self.output_path = pathlib.Path(output_path)
        self.image_paths = list(image_paths)
        self.written: set[int] = set()
        self.with_confidence = False
        self.export_seconds = 0.0

    def write(self, indices: Iterable[int], depths, confidences=None) -> None:
        """Saves the maps of some input frames.

        Args:
            indices: Input index of each map.
            depths: depths[k] (H x W) is the depth of frame indices[k].
            confidences: Matching confidence maps, or None.
        """
        started = time.perf_counter()
        try:
            self._write(indices, depths, confidences)
        finally:
            self.export_seconds += time.perf_counter() - started

    def _write(self, indices: Iterable[int], depths, confidences=None) -> None:
        depth_dir = self.output_path / "depths"
        depth_dir.mkdir(parents=True, exist_ok=True)
        for position, index in enumerate(indices):
            index = int(index)
            if not 0 <= index < len(self.image_paths):
                continue
            stem = pathlib.Path(self.image_paths[index]).stem
            depth = np.squeeze(np.asarray(depths[position], dtype=np.float32))
            utils.save_prediction_map(depth_dir, stem, depth)
            if confidences is not None:
                utils.save_confidence_map(
                    self.output_path, self.image_paths[index],
                    np.squeeze(
                        np.asarray(confidences[position], dtype=np.float32)))
                self.with_confidence = True
            self.written.add(index)

    def write_tensors(self, depth, confidence=None, offset: int = 0) -> None:
        """Writes torch tensors in batches.

        Args:
            depth: [N, H, W(, 1)] tensor (any device / dtype) of frames
                `offset` .. `offset + N`.
            confidence: Matching tensor, or None.
            offset: Input index of the first frame.
        """
        count = int(depth.shape[0])
        for start in range(0, count, WRITE_BATCH):
            stop = min(start + WRITE_BATCH, count)
            started = time.perf_counter()
            depths = depth[start:stop].float().cpu().numpy()
            confs = confidence[start:stop].float().cpu().numpy(
            ) if confidence is not None else None
            self.export_seconds += time.perf_counter() - started
            self.write(range(offset + start, offset + stop), depths, confs)

    def metrics(self) -> dict:
        """Returns what was exported, for the model's metrics.json."""
        return {
            "depth_frames": len(self.written),
            "depth_confidence": self.with_confidence,
            "depth_export_seconds": self.export_seconds
        }


def chunk_scale(chunk_c2w: np.ndarray, final_c2w: np.ndarray) -> float:
    """Returns the scale mapping a chunk's own trajectory onto the final one.

    Any Sim(3) x -> s R x + t scales every distance by s, so the ratio of the
    final to the chunk distances between the same cameras is s exactly; the
    median over consecutive pairs ignores near-coincident cameras.

    Args:
        chunk_c2w: The chunk's camera-to-world poses, (N, 4, 4).
        final_c2w: The final poses of the same frames, (N, 4, 4).

    Returns:
        The median ratio of camera-center distances (1 when undefined).
    """
    chunk_centers = np.asarray(chunk_c2w)[:, :3, 3]
    final_centers = np.asarray(final_c2w)[:, :3, 3]
    if len(chunk_centers) < 2:
        return 1.0
    chunk_steps = np.linalg.norm(np.diff(chunk_centers, axis=0), axis=1)
    final_steps = np.linalg.norm(np.diff(final_centers, axis=0), axis=1)
    usable = chunk_steps > max(1e-9, 1e-6 * float(chunk_steps.max(initial=0.0)))
    if not usable.any():
        return 1.0
    return float(np.median(final_steps[usable] / chunk_steps[usable]))


def chunk_ranges(frame_count: int, chunk_size: int,
                 overlap: int) -> list[tuple[int, int]]:
    """Returns the chunks of VGGT-Long / Pi-Long `process_long_sequence`.

    Args:
        frame_count: Number of frames.
        chunk_size: Frames per chunk.
        overlap: Frames two consecutive chunks share.

    Returns:
        (start, stop) frame ranges.
    """
    if frame_count <= chunk_size:
        return [(0, frame_count)]
    step = chunk_size - overlap
    count = (frame_count - overlap + step - 1) // step
    return [(index * step, min(index * step + chunk_size, frame_count))
            for index in range(count)]


def export_long_chunks(sink: DepthSink, chunk_dir: pathlib.Path,
                       ranges: list[tuple[int, int]],
                       final_poses: dict[int, np.ndarray], method: str) -> None:
    """Writes the depth of VGGT-Long / Pi-Long chunks in the final scale.

    VGGT-Long chunks hold `depth` [N, H, W], `depth_conf` and `extrinsic`
    (camera-to-world); Pi-Long chunks hold Pi3's `local_points` [N, H, W, 3]
    (depth = z), `conf` (sigmoid) and `camera_poses` (camera-to-world).

    Args:
        sink: Where the maps go.
        chunk_dir: The system's _tmp_results_unaligned (chunk_<k>.npy).
        ranges: Frame range of each chunk (`chunk_ranges`).
        final_poses: Final camera-to-world pose of each input index.
        method: "VGGT_LONG" or "PI_LONG".

    Raises:
        FileNotFoundError: A chunk prediction is missing.
    """
    for chunk_index, (start, stop) in enumerate(ranges):
        path = chunk_dir / f"chunk_{chunk_index}.npy"
        if not path.is_file():
            raise FileNotFoundError(
                f"{method}: missing chunk prediction {path}")
        data = np.load(path, allow_pickle=True).item()
        if method == "PI_LONG":
            depth = np.asarray(data["local_points"])[..., 2]
            confidence = np.asarray(data["conf"])
            chunk_c2w = np.asarray(data["camera_poses"])
        else:
            depth = np.asarray(data["depth"])
            confidence = np.asarray(data.get("depth_conf")) if data.get(
                "depth_conf") is not None else None
            chunk_c2w = np.asarray(data["extrinsic"])
        if chunk_c2w.shape[-2:] == (3, 4):
            bottom = np.broadcast_to(np.array([0.0, 0.0, 0.0, 1.0]),
                                     (*chunk_c2w.shape[:-2], 1, 4))
            chunk_c2w = np.concatenate([chunk_c2w, bottom], axis=-2)
        indices = list(range(start, stop))
        final = np.stack([final_poses[index] for index in indices])
        scale = chunk_scale(chunk_c2w[:len(indices)], final)
        kept_confidence = confidence[:len(indices
                                         )] if confidence is not None else None
        sink.write(indices, depth[:len(indices)] * scale, kept_confidence)
        del data, depth, confidence


def export_numbered_maps(sink: DepthSink,
                         depth_dir: pathlib.Path,
                         frame_ids: list[int],
                         conf_dir: pathlib.Path | None = None) -> None:
    """Writes R3-style maps: <k:06d>.npy for the k-th returned frame.

    Args:
        sink: Where the maps go.
        depth_dir: Directory of the depth maps.
        frame_ids: Input index of each returned frame.
        conf_dir: Directory of the confidence maps, or None.
    """
    for position, frame_id in enumerate(frame_ids):
        depth = np.load(depth_dir / f"{position:06d}.npy")
        conf_path = (conf_dir /
                     f"{position:06d}.npy" if conf_dir is not None else None)
        confidence = np.load(
            conf_path
        ) if conf_path is not None and conf_path.is_file() else None
        sink.write([frame_id], [depth],
                   [confidence] if confidence is not None else None)


def read_exr_depth(path: pathlib.Path) -> np.ndarray:
    """Reads one single-channel EXR depth map (Scal3R depths/<k>.exr).

    Args:
        path: The .exr file.

    Returns:
        The (H, W) depth.

    Raises:
        RuntimeError: OpenCV cannot read the file.
    """
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2  # pylint: disable=import-outside-toplevel

    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Could not read {path} (OpenEXR support)")
    return depth[..., 0] if depth.ndim == 3 else depth


def export_vggt_slam_dumps(sink: DepthSink, dump_dir: pathlib.Path) -> None:
    """Writes the VGGT-SLAM keyframe maps fixed_input_launcher.py dumped.

    fixed_input_launcher.py --dump_vggt_slam_depth writes <staged stem>.npz
    with `depth` and `depth_conf`; the staged stem is the input index.

    Args:
        sink: Where the maps go.
        dump_dir: The dump directory.
    """
    for path in sorted(dump_dir.glob("*.npz")):
        with np.load(path) as data:
            sink.write([int(path.stem)], [data["depth"]], [data["depth_conf"]])
