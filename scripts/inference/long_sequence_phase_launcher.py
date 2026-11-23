# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Run VGGT-Long or Pi-Long in two phases: GPU chunk inference, then CPU chunk alignment, in separate processes.

Both systems first run every model forward pass (loop detection, one pass per overlapping chunk and per loop chunk,
each saved to disk), then align the chunks pairwise with a robust Sim(3) fit, optimize the loop closures, and write
the camera poses. The second part is CPU-bound and on a long sequence takes hours (Pi-Long on a 12,042-frame VBR
sequence: ~20 min of inference, ~2.5 h of alignment), long enough for a low-GPU-usage monitor to kill a GPU job.
This launcher runs the repository's own code unchanged, split at that boundary:

``--phase gpu``  runs the script's ``__main__`` steps until the first chunk alignment, then pickles the little state
                 the alignment needs (frame list, loop pairs, per-chunk camera poses) to ``--state`` and exits.
                 Chunk files keep only the arrays the alignment reads (point maps, confidences, mask): the rest
                 (images, depth, local points) feed only the point-cloud export, which this benchmark does not use;
                 with ``--export_depth`` they also keep the arrays slam_depth_export.export_long_chunks reads.
``--phase align`` rebuilds the system without loading any network, replays the saved chunks, and runs the
                 repository's ``process_long_sequence`` alignment, loop optimization and Sim(3) accumulation, then
                 ``save_camera_poses``. The point-cloud export that follows the accumulation is skipped: poses
                 depend only on the per-chunk poses and the accumulated transforms. Pi-Long's default
                 ``align_lib: torch`` moves its float32 tensors to CUDA; here they stay on the CPU (same code).

If the sequence needs no alignment (one chunk, no loops), the GPU phase finishes the whole run and writes no state.

    python long_sequence_phase_launcher.py --phase gpu --state work/phase_state.pkl \
        [--patch base_models.vggt.utils.load_fn --record work/model_input_size.json] \
        -- /code/pi_long.py --image_dir work/images --config work/config.yaml [--exp_folder_name work/exps]
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import fixed_input_launcher  # pylint: disable=import-error  # sibling file; this directory is on sys.path here

# Per-chunk arrays the alignment reads (process_long_sequence up to accumulate_sim3_transforms), per system class.
ALIGN_KEYS = {
    "Pi_Long": ("points", "conf"),
    "VGGT_Long": ("world_points", "world_points_conf", "mask")
}
# And the ones the per-frame depth export reads (slam_depth_export.export_long_chunks).
DEPTH_KEYS = {
    "Pi_Long": ("local_points", "camera_poses"),
    "VGGT_Long": ("depth", "depth_conf", "extrinsic")
}


class GpuPhaseDone(Exception):
    """Raised at the first chunk alignment: every model forward pass has run and been saved."""


class AlignmentDone(Exception):
    """Raised by the Sim(3) accumulation, the last step poses depend on, carrying its result."""

    def __init__(self, transforms):
        super().__init__("alignment done")
        self.transforms = transforms


class _NoModel:
    """Stands in for every network the systems construct in ``__init__`` during the CPU phase."""

    def __init__(self, *args, **kwargs):
        del args, kwargs

    def to(self, *args, **kwargs):
        """No device to move to."""
        del args, kwargs
        return self

    def eval(self):
        """Already in evaluation mode."""
        return self

    def load_state_dict(self, *args, **kwargs):
        """No weights to load."""
        del args, kwargs


class _CpuTorch:
    """``torch`` for Pi-Long's ``loop_utils.alignment_torch``.

    Its ``torch.tensor(..., device='cuda')`` lands on CPU."""

    def __init__(self, torch_module):
        self._torch = torch_module

    def __getattr__(self, name):
        return getattr(self._torch, name)

    def tensor(self, *args, **kwargs):
        """``torch.tensor`` on the CPU whatever device is asked for."""
        kwargs.pop("device", None)
        return self._torch.tensor(*args, **kwargs)


def parse_args() -> argparse.Namespace:
    """Launcher options, then the system script's own arguments (parsed the way its ``__main__`` does)."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("gpu", "align"), required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument(
        "--patch",
        action="append",
        default=[],
        help=
        "Module whose load_and_preprocess_images keeps fixed-resolution frames (repeatable)."
    )
    parser.add_argument("--record", type=Path, default=None)
    parser.add_argument(
        "--export_depth",
        action="store_true",
        help="GPU phase: also keep the chunk arrays the depth export reads.")
    parser.add_argument("script", type=Path)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    # The repositories' own argument names; VGGT-Long has no --exp_folder_name and always uses ./exps.
    script_parser = argparse.ArgumentParser()
    script_parser.add_argument("--image_dir", required=True)
    script_parser.add_argument("--config", default="./configs/base_config.yaml")
    script_parser.add_argument("--exp_folder_name", default="./exps")
    args.script_args = script_parser.parse_args(
        [arg for arg in args.script_args if arg != "--"])
    return args


def import_system(script: Path, patches: list[str], record: Path | None):
    """Import the script as a module (its ``__main__`` block does not run) with the fixed-input loaders in place."""
    here = Path(__file__).resolve().parent
    sys.path[:] = [
        entry for entry in sys.path if Path(entry or ".").resolve() != here
    ]
    sys.path.insert(0, str(script.resolve().parent))

    def loader(image_path_list, mode="crop"):
        return fixed_input_launcher.load_fixed_images(image_path_list,
                                                      mode,
                                                      record=record)

    for module_name in patches:
        importlib.import_module(module_name).load_and_preprocess_images = loader
    module = importlib.import_module(script.stem)
    name = next(name for name in ALIGN_KEYS if hasattr(module, name))
    return module, getattr(module, name)


@contextlib.contextmanager
def slim_saves(keys: tuple[str, ...]):
    """Make ``np.save`` of a chunk's prediction dict keep only ``keys`` (the call is the system's own)."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    original = np.save

    def save(path, array, *args, **kwargs):
        if isinstance(array, dict):
            missing = [key for key in keys if key not in array]
            if missing:
                raise KeyError(
                    f"chunk predictions lack {missing} (have {sorted(array)})")
            array = {key: array[key] for key in keys}
        return original(path, array, *args, **kwargs)

    np.save = save
    try:
        yield
    finally:
        np.save = original


def run_gpu_phase(module, system_class, args: argparse.Namespace) -> None:
    """The script's ``__main__`` up to the first alignment; then the state for the CPU phase."""
    script_args = args.script_args
    config = module.load_config(script_args.config)
    save_dir = os.path.join(script_args.exp_folder_name,
                            script_args.image_dir.replace("/", "_"),
                            datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        module.copy_file(script_args.config, save_dir)
    if config["Model"]["align_method"] == "numba":
        module.warmup_numba()
    system = system_class(script_args.image_dir, save_dir, config)

    process_single_chunk = system.process_single_chunk
    keys = ALIGN_KEYS[system_class.__name__] + (
        DEPTH_KEYS[system_class.__name__] if args.export_depth else ())

    def slim_chunk(*chunk_args, **chunk_kwargs):
        with slim_saves(keys):
            return process_single_chunk(*chunk_args, **chunk_kwargs)

    def stop_before_alignment(*align_args, **align_kwargs):
        del align_args, align_kwargs
        raise GpuPhaseDone

    system.process_single_chunk = slim_chunk
    module.weighted_align_point_maps = stop_before_alignment
    try:
        system.run()
    except GpuPhaseDone:
        state = {
            "save_dir":
                save_dir,
            "image_dir":
                script_args.image_dir,
            "img_list":
                system.img_list,
            "loop_list":
                system.loop_list,
            "all_camera_poses":
                system.all_camera_poses,
            "all_camera_intrinsics":
                getattr(system, "all_camera_intrinsics", None)
        }
        args.state.write_bytes(pickle.dumps(state))
        print(
            f"GPU phase done: {len(system.all_camera_poses)} chunks saved; alignment state in {args.state}"
        )
        return
    # No alignment was needed: the run is complete, finish it as the script's __main__ does.
    system.close()
    print("GPU phase finished the whole run (no chunk alignment needed)")


def run_align_phase(module, system_class, args: argparse.Namespace) -> None:
    """Rebuild the system without networks, replay the saved chunks, align, and save the camera poses."""
    import safetensors.torch  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel

    state = pickle.loads(args.state.read_bytes())
    config = module.load_config(args.script_args.config)
    for name in ("Pi3", "Pi3X", "VGGTAdapter", "Pi3Adapter",
                 "MapAnythingAdapter", "LoopDetector", "RetrievalDBOW"):
        if hasattr(module, name):
            setattr(module, name, _NoModel)
    safetensors.torch.load_file = lambda *args, **kwargs: {}
    torch.cuda.get_device_capability = lambda *args, **kwargs: (
        8, 0)  # only picks the (unused) inference dtype
    # Pi-Long's align_lib 'torch' (loop_utils/alignment_torch.py) moves every array to CUDA with .cuda() and
    # torch.tensor(..., device='cuda'); keep both on the CPU (same float32 code).
    torch.Tensor.cuda = lambda tensor, *args, **kwargs: tensor
    alignment_torch = sys.modules.get("loop_utils.alignment_torch")
    if alignment_torch is not None:
        alignment_torch.torch = _CpuTorch(torch)

    system = system_class(state["image_dir"], state["save_dir"], config)
    system.img_list = state["img_list"]
    system.loop_list = state["loop_list"]
    system.all_camera_poses = state["all_camera_poses"]
    if state["all_camera_intrinsics"] is not None:
        system.all_camera_intrinsics = state["all_camera_intrinsics"]

    def replay_chunk(range_1, chunk_idx=None, range_2=None, is_loop=False):
        del chunk_idx
        if not (is_loop or range_2 is not None):
            return None  # sequential chunks: already on disk in result_unaligned_dir
        name = f"loop_{range_1[0]}_{range_1[1]}_{range_2[0]}_{range_2[1]}.npy"
        return module.np.load(os.path.join(system.result_loop_dir, name),
                              allow_pickle=True).item()

    accumulate = module.accumulate_sim3_transforms

    def accumulate_then_stop(transforms):
        raise AlignmentDone(accumulate(transforms))

    system.process_single_chunk = replay_chunk
    module.accumulate_sim3_transforms = accumulate_then_stop
    try:
        system.process_long_sequence()
    except AlignmentDone as done:
        system.sim3_list = done.transforms
    else:
        raise RuntimeError(
            "process_long_sequence finished without accumulating the chunk transforms"
        )
    system.save_camera_poses()
    print("Alignment phase done")


def main() -> None:
    """Import the system and run the requested phase."""
    args = parse_args()
    module, system_class = import_system(
        args.script, args.patch, args.record if args.phase == "gpu" else None)
    if args.phase == "gpu":
        run_gpu_phase(module, system_class, args)
    else:
        run_align_phase(module, system_class, args)


if __name__ == "__main__":
    main()
