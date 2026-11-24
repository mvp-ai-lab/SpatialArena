# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Base class of the SLAM models: external systems run on a whole sequence.

The streaming (`streaming_slam`) and long-sequence (`long_sequence_slam`)
SLAM methods run their own released code on the whole strided sequence,
in-process or as a subprocess, and return a camera pose per frame (or per
keyframe). They do not fit `base.BaseModel`'s load / preprocess / forward
split, so `SlamModel.run` replaces the whole pipeline:

1. Select the frames (--interval = SLAM stride, --max_images_per_run cap)
   and link them into a staging folder as 000000.png, ... so every method
   sees the same frames in the same order and its per-index output maps
   back to the original names.
2. Run the method (`run_case`, per family).
3. Write the poses as sparse/<subdir> with nominal intrinsics (focal =
   max(width, height), principal point at the center): the SLAM protocol
   scores camera poses only. With --export_depth, the method's per-frame
   depth and confidence too (`slam_depth`).

Their command line is `base.BaseModel`'s, without the resolution and point
export options (each method resizes on its own and exports poses only).
--ckpt_path is the method's own checkpoint (its MethodSpec's default); the
weights the methods share (SALAD, DINOv2, the long-sequence backbones) are
read under $SHARED_CHECKPOINT_ROOT, the engine's checkpoint root.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import threading
from typing import Any, ClassVar

import numpy as np
from PIL import Image

from spatialarena.models import base
from spatialarena.models import utils

# see spatialarena/paths.py
SPATIALARENA_ROOT = os.path.expanduser(
    os.environ.get("SPATIALARENA_ROOT", "~/spatialarena"))
DEFAULT_CODE_ROOT = pathlib.Path(f"{SPATIALARENA_ROOT}/benchmarks/_model_codes")
DEFAULT_CKPT_ROOT = pathlib.Path(f"{SPATIALARENA_ROOT}/benchmarks/_checkpoints")
# The repository's scripts/inference: the launchers the long-sequence
# systems run through, in the method's own interpreter.
SCRIPTS_DIR = pathlib.Path(__file__).parents[2] / "scripts" / "inference"


def checkpoint_root() -> pathlib.Path:
    """Returns the shared checkpoint root of the methods' other weights.

    $SHARED_CHECKPOINT_ROOT (exported by the evaluation engine), else
    $SPATIALARENA_ROOT/benchmarks/_checkpoints.
    """
    root = os.environ.get("SHARED_CHECKPOINT_ROOT")
    if root:
        return pathlib.Path(root)
    return DEFAULT_CKPT_ROOT


def stage_images(image_paths: list[str], staging: pathlib.Path) -> list[str]:
    """Links the frames into `staging` as 000000.<ext>, 000001.<ext>, ...

    Args:
        image_paths: The selected frames, in order.
        staging: The staging folder (created).

    Returns:
        The staged file names.
    """
    staging.mkdir(parents=True, exist_ok=True)
    names = []
    for index, path in enumerate(image_paths):
        staged = staging / f"{index:06d}{pathlib.Path(path).suffix.lower()}"
        staged.symlink_to(pathlib.Path(path).resolve())
        names.append(staged.name)
    return names


def _descendants(pid: int) -> set[int]:
    pids, frontier = {pid}, [pid]
    while frontier:
        parent = frontier.pop()
        try:
            children = pathlib.Path(
                f"/proc/{parent}/task/{parent}/children").read_text().split()
        except OSError:
            continue
        for child in map(int, children):
            if child not in pids:
                pids.add(child)
                frontier.append(child)
    return pids


class GpuMemorySampler(threading.Thread):
    """Peak GPU memory (MiB) used by a child process tree.

    Sampled from nvidia-smi once per second.

    Attributes:
        pid: Root of the process tree.
        peak: The peak so far, or None before any sample with memory.
    """

    def __init__(self, pid: int):
        super().__init__(daemon=True)
        # Not `self._stop`: that name is threading.Thread's own method,
        # which join() calls.
        self.pid, self.peak, self._stop_event = pid, None, threading.Event()

    def run(self) -> None:
        while not self._stop_event.wait(1.0):
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi", "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits"
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                ).stdout
            except (OSError, subprocess.SubprocessError):
                continue
            tree = _descendants(self.pid)
            used = sum(
                int(mem) for pid, mem in (line.split(",")
                                          for line in out.splitlines()
                                          if "," in line) if int(pid) in tree)
            if used:
                self.peak = max(self.peak or 0, used)

    def stop(self) -> float | None:
        """Stops sampling.

        Returns:
            The peak in MiB, or None when nothing was measured.
        """
        self._stop_event.set()
        self.join(timeout=15)
        return float(self.peak) if self.peak is not None else None


class SlamModel(base.BaseModel):
    """A SLAM method run on the whole strided sequence.

    A family (streaming, long-sequence) subclasses it and implements
    `weight_paths` and `run_case`; each method is a subclass of the family
    with its `name`.

    Attributes:
        code_dir: The method's checkout under the default code root, without
            a MethodSpec.
        ckpt_key: The `weight_paths` entry --ckpt_path replaces; None when
            the method takes no checkpoint of its own.
    """

    uses_resolution = False
    uses_image_load_resolution = False
    exports_points = False

    code_dir: ClassVar[str] = ""
    ckpt_key: ClassVar[str | None] = None

    @classmethod
    def build_parser(cls, method=None):
        # The families' descriptions are formatted lists.
        parser = super().build_parser(method)
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        return parser

    # ------------------------------------------------------------------
    # BaseModel hooks: a SLAM method runs whole sequences in `run`.
    # ------------------------------------------------------------------

    def load(self, device: str) -> Any:
        raise NotImplementedError(
            f"{self.name} runs whole sequences; use run()")

    def preprocess(self,
                   image_paths: list[str],
                   batch_index: int | None = None) -> base.ModelInput:
        raise NotImplementedError(
            f"{self.name} runs whole sequences; use run()")

    def forward(self, inputs: base.ModelInput) -> base.Prediction:
        raise NotImplementedError(
            f"{self.name} runs whole sequences; use run()")

    @property
    def checkpoint(self) -> str | None:
        return str(self.code_path)

    @property
    def code_path(self) -> pathlib.Path:
        """The method's checkout."""
        return self.args.code_path or DEFAULT_CODE_ROOT / self.code_dir

    # ------------------------------------------------------------------
    # The pipeline.
    # ------------------------------------------------------------------

    @classmethod
    def weight_paths(cls, ckpt_root: pathlib.Path) -> dict[str, pathlib.Path]:
        """Returns the weight files of the family under a checkpoint root.

        Args:
            ckpt_root: The shared checkpoint root.
        """
        raise NotImplementedError

    def weights(self) -> dict[str, pathlib.Path]:
        """Returns the weight files of this run.

        `weight_paths` of `checkpoint_root()`, with --ckpt_path (when given)
        as the method's own checkpoint.

        Raises:
            ValueError: --ckpt_path for a method without a checkpoint of its
                own.
        """
        weights = self.weight_paths(checkpoint_root())
        if self.args.ckpt_path:
            if self.ckpt_key is None:
                raise ValueError(f"{self.method_name} loads no checkpoint of "
                                 "its own; --ckpt_path is not used")
            weights[self.ckpt_key] = pathlib.Path(self.args.ckpt_path)
        return weights

    def run(self) -> None:
        """Runs the method on the strided sequence and exports its poses.

        Raises:
            ValueError: Fewer than 3 frames, or --ckpt_path for a method
                without a checkpoint.
        """
        args = self.args
        self.images_dir = utils.canonical_image_root(args.data_path)
        self.image_paths = utils.select_interval(
            utils.find_image_paths(self.images_dir), args.interval,
            args.max_images_per_run)
        if len(self.image_paths) < 3:
            raise ValueError(f"Need at least 3 frames, found "
                             f"{len(self.image_paths)} in {self.images_dir}")
        weights = self.weights()
        if args.dry_run:
            print({
                "method": self.method_name,
                "frames": len(self.image_paths),
                "code_path": str(self.code_path),
                "weights": {
                    k: str(v) for k, v in weights.items()
                },
                "output": str(args.output_path)
            })
            return
        self.run_case(weights)

    def run_case(self, weights: dict[str, pathlib.Path]) -> None:
        """Runs the method and writes the case's artifacts.

        Args:
            weights: `weights()`.
        """
        raise NotImplementedError

    def pose_indices(self, poses: dict[int, np.ndarray]) -> list[int]:
        """Returns the input indices with a pose, sorted.

        Args:
            poses: Camera-to-world pose by returned index.

        Raises:
            RuntimeError: No pose for any input frame.
        """
        indices = sorted(
            index for index in poses if 0 <= index < len(self.image_paths))
        if not indices:
            raise RuntimeError(f"{self.method_name} returned no poses for the "
                               f"{len(self.image_paths)} input frames")
        return indices

    def export_poses(self, poses: dict[int, np.ndarray],
                     indices: list[int]) -> list[str]:
        """Writes the returned poses as a COLMAP model, nominal intrinsics.

        Args:
            poses: Camera-to-world pose by input index.
            indices: `pose_indices` of `poses`.

        Returns:
            The frames with a pose.
        """
        args = self.args
        returned = [self.image_paths[index] for index in indices]
        w2c = np.stack([np.linalg.inv(poses[index]) for index in indices
                       ])[:, :3, :4]
        width, height = Image.open(returned[0]).size
        focal = float(max(width, height))
        intrinsics = np.tile(
            np.array([[focal, 0, width / 2], [0, focal, height / 2],
                      [0, 0, 1]]), (len(returned), 1, 1))
        utils.export_camera_pose_priors(
            args.output_path,
            self.images_dir,
            returned,
            w2c,
            intrinsics,
            width=width,
            height=height,
            colmap_model_subdir=args.colmap_model_subdir)
        return returned
