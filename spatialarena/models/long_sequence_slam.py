# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Long-sequence SLAM systems: VGGT-Long, Pi-Long, VGGT-SLAM.

These systems chunk an arbitrarily long recording internally, so they have
no single-pass frame budget. Each is run unmodified through its own entry
point in a subprocess, with its published default settings (see
`DESCRIPTION`). VGGT-Long and Pi-Long can run as two processes
(scripts/inference/long_sequence_phase_launcher.py): --phase gpu does the
chunk inference and leaves its work directory plus ALIGN_PENDING in
--output_path; --phase align, run later on a CPU-only node, does the chunk
alignment and exports the poses. Poses are exported as in
`slam.SlamModel`.

scripts/inference/long_sequence_slam.py runs them (--model picks the
system). Their weights are read under the shared checkpoint root
(`slam.checkpoint_root`); --ckpt_path replaces the backbone's (VGGT-Long:
VGGT-1B, Pi-Long: Pi3). VGGT-SLAM loads its own.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import yaml

from spatialarena.models import registry
from spatialarena.models import slam
from spatialarena.models import slam_depth
from spatialarena.models import utils

# pylint: disable=line-too-long
DESCRIPTION = """Run a long-sequence SLAM system (VGGT-Long, Pi-Long, VGGT-SLAM) and export benchmark camera poses.

These systems process an arbitrarily long recording by chunking it internally (overlapping chunks or submaps,
aligned and loop-closed), so unlike the feed-forward methods they have no single-pass frame budget. Each is run
unmodified through its own entry point, with its published default settings, in the default ``si_bench`` env:

- ``VGGT_LONG``: ``vggt_long.py`` (github.com/DengKaiCQ/VGGT-Long), VGGT-1B backbone, SALAD loop closure.
- ``PI_LONG``: ``pi_long.py`` (github.com/DengKaiCQ/Pi-Long), Pi3 backbone, SALAD loop closure.
- ``VGGT_SLAM``: ``main.py`` (github.com/MIT-SPARK/VGGT-SLAM) with the arguments of its own TUM evaluation script
  (``evals/eval_tum.sh``: submap 16, one loop, min disparity 50, conf 25 %, loop threshold 0.95). VGGT-SLAM keeps
  only keyframes, so it returns poses for a subset of the input frames; the SLAM evaluator reports that as
  completeness and scores the returned frames, as VGGT-SLAM's own evaluation does.

The selected frames (``--interval`` = SLAM stride) are linked into a staging folder as ``000000.png``... so every
system sees the same frames in the same order and its per-index output maps back to the original names. Poses are
written as ``sparse/0`` with nominal intrinsics (focal = max(width, height), principal point at the center): these
systems' intrinsics are not part of the SLAM protocol, which scores camera poses only.

The weights are read under the shared checkpoint root ``$SHARED_CHECKPOINT_ROOT`` (default
``$SPATIALARENA_ROOT/benchmarks/_checkpoints``); ``--ckpt_path`` replaces the backbone's (VGGT-Long: ``vggt.pt``,
Pi-Long: ``pi3.safetensors``). VGGT-SLAM loads its own.
"""
# pylint: enable=line-too-long

METHODS = ("VGGT_LONG", "PI_LONG", "VGGT_SLAM")
CODE_DIRS = {
    "VGGT_LONG": "VGGT-Long",
    "PI_LONG": "Pi-Long",
    "VGGT_SLAM": "VGGT-SLAM"
}
# VGGT-SLAM's evals/eval_tum.sh settings.
VGGT_SLAM_ARGS = [
    "--max_loops", "1", "--min_disparity", "50", "--conf_threshold", "25",
    "--lc_thres", "0.95", "--submap_size", "16"
]
# The module each VGGT-based system imports load_and_preprocess_images from
# (it hardcodes a 518 px width); for fixed-resolution inputs (full-slam)
# fixed_input_launcher.py swaps it for a loader that keeps the frames as
# they are.
VGGT_LOADER_MODULES = {
    "VGGT_LONG": "base_models.vggt.utils.load_fn",
    "VGGT_SLAM": "vggt.utils.load_fn"
}
FIXED_INPUT_LAUNCHER = slam.SCRIPTS_DIR / "fixed_input_launcher.py"
# VGGT-Long and Pi-Long can run as two processes
# (long_sequence_phase_launcher.py): --phase gpu does the chunk inference
# and leaves its work directory plus ALIGN_PENDING in --output_path; --phase
# align, run later on a CPU-only node, does the chunk alignment and exports
# the poses. A benchmark GPU job thus never idles through the alignment.
PHASE_LAUNCHER = slam.SCRIPTS_DIR / "long_sequence_phase_launcher.py"
SPLIT_METHODS = ("VGGT_LONG", "PI_LONG")
ALIGN_PENDING = "slam_align_pending.json"


def weight_paths(ckpt_root: pathlib.Path) -> dict[str, pathlib.Path]:
    """Returns the weight files under `ckpt_root`.

    Args:
        ckpt_root: The shared checkpoint root.

    Returns:
        Weight file by model (VGGT, Pi3, SALAD, DINO).
    """
    long_dir = ckpt_root / "local" / "slam_long"
    return {
        # facebook/VGGT-1B model.pt
        "VGGT": ckpt_root / "local" / "vggt" / "vggt.pt",
        # yyfz233/Pi3 model.safetensors
        "Pi3": long_dir / "pi3.safetensors",
        "SALAD": long_dir / "dino_salad.ckpt",
        "DINO": long_dir / "dinov2_vitb14_pretrain.pth",
    }


def write_long_config(method: str,
                      code_path: pathlib.Path,
                      weights: dict[str, pathlib.Path],
                      destination: pathlib.Path,
                      fixed: tuple[int, int] | None = None,
                      keep_chunks: bool = False) -> pathlib.Path:
    """Writes the repo's own base_config.yaml with absolute weight paths.

    Everything else is left at the published values.

    Args:
        method: "VGGT_LONG" or "PI_LONG".
        code_path: The system's checkout.
        weights: `weight_paths`.
        destination: The config file to write.
        fixed: For fixed inputs, Pi-Long's image_width is their width,
            which its loader then keeps as is.
        keep_chunks: Keep the per-chunk predictions the system otherwise
            deletes at the end, for the depth export.

    Returns:
        `destination`.
    """
    config = yaml.safe_load((code_path / "configs" /
                             "base_config.yaml").read_text(encoding="utf-8"))
    config["Weights"]["SALAD"] = str(weights["SALAD"])
    config["Weights"]["DNIO"] = str(weights["DINO"])  # sic: the repos' key name
    if method == "VGGT_LONG":
        config["Weights"]["model"] = "VGGT"
        config["Weights"]["VGGT"] = str(weights["VGGT"])
    else:
        config["Model"]["model_type"] = "Pi3"
        config["Weights"]["Pi3"] = str(weights["Pi3"])
        if fixed:
            config["Model"]["image_width"] = fixed[0]
    if keep_chunks:
        config["Model"]["delete_temp_files"] = False
    destination.write_text(yaml.safe_dump(config, sort_keys=False),
                           encoding="utf-8")
    return destination


def read_long_poses(exp_root: pathlib.Path,
                    count: int) -> dict[int, np.ndarray]:
    """Reads camera_poses.txt of VGGT-Long/Pi-Long.

    One flattened 4x4 camera-to-world per staged image, in order.

    Args:
        exp_root: The system's experiment folder.
        count: Number of staged images.

    Returns:
        {input index: c2w}.

    Raises:
        RuntimeError: Not exactly one camera_poses.txt, or a wrong shape.
    """
    files = sorted(exp_root.rglob("camera_poses.txt"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one camera_poses.txt under {exp_root}, "
                           f"found {len(files)}")
    rows = np.loadtxt(files[0], ndmin=2)
    if rows.shape != (count, 16):
        raise RuntimeError(
            f"{files[0]} has shape {rows.shape}, expected ({count}, 16)")
    return {index: row.reshape(4, 4) for index, row in enumerate(rows)}


def read_tum_poses(path: pathlib.Path) -> dict[int, np.ndarray]:
    """Reads the VGGT-SLAM log.

    `frame_id x y z qx qy qz qw` (camera-to-world) per keyframe; frame_id =
    staged index.

    Args:
        path: The log file.

    Returns:
        {input index: c2w}.
    """
    # pylint: disable-next=import-outside-toplevel
    from scipy.spatial import transform

    poses = {}
    for row in np.loadtxt(path, ndmin=2):
        c2w = np.eye(4)
        c2w[:3, :3] = transform.Rotation.from_quat(row[4:8]).as_matrix()
        c2w[:3, 3] = row[1:4]
        poses[int(round(row[0]))] = c2w
    return poses


def system_command(
        method: str,
        code_path: pathlib.Path,
        work: pathlib.Path,
        fixed: bool,
        depth_dump: pathlib.Path | None = None
) -> tuple[list[str], dict[str, str]]:
    """Returns the system's own command line and its environment.

    Args:
        method: One of `METHODS`.
        code_path: The system's checkout.
        work: The work directory, with the staged frames (images/) and
            config.yaml.
        fixed: Whether the inputs are fixed-resolution.
        depth_dump: VGGT-SLAM: run it through fixed_input_launcher.py to
            save its keyframe depths there.

    Returns:
        (command, environment).
    """
    staging = work / "images"
    env = dict(os.environ)
    if method == "VGGT_SLAM":
        command = [
            sys.executable,
            str(code_path / "main.py"), "--image_folder",
            str(staging), *VGGT_SLAM_ARGS, "--log_results", "--skip_dense_log",
            "--log_path",
            str(work / "poses.txt")
        ]
        env["PYTHONPATH"] = os.pathsep.join([
            str(code_path),
            str(code_path / "third_party" / "vggt"),
            str(code_path / "third_party" / "salad"),
            env.get("PYTHONPATH", "")
        ])
    else:
        script = "vggt_long.py" if method == "VGGT_LONG" else "pi_long.py"
        command = [
            sys.executable,
            str(code_path / script), "--image_dir",
            str(staging), "--config",
            str(work / "config.yaml")
        ]
        if method == "PI_LONG":
            command += ["--exp_folder_name", str(work / "exps")]
        env["PYTHONPATH"] = os.pathsep.join(
            [str(code_path), env.get("PYTHONPATH", "")])
    launcher_args = []
    if fixed and method in VGGT_LOADER_MODULES:
        launcher_args += [
            "--patch", VGGT_LOADER_MODULES[method], "--record",
            str(work / "model_input_size.json")
        ]
    if depth_dump is not None and method == "VGGT_SLAM":
        launcher_args += ["--dump_vggt_slam_depth", str(depth_dump)]
    if launcher_args:
        command = [
            command[0],
            str(FIXED_INPUT_LAUNCHER), *launcher_args, "--", *command[1:]
        ]
    return command, env


def phase_command(command: list[str],
                  phase: str,
                  work: pathlib.Path,
                  export_depth: bool = False) -> list[str]:
    """Wraps a `system_command` in long_sequence_phase_launcher.py.

    Args:
        command: The system command (optionally through
            fixed_input_launcher.py, whose --patch/--record the phase
            launcher then applies itself).
        phase: "gpu" or "align".
        work: The work directory.
        export_depth: Keep the chunk arrays the depth export reads (the GPU
            phase otherwise drops them).

    Returns:
        The phase launcher's command.
    """
    launcher_args = ["--phase", phase, "--state", str(work / "phase_state.pkl")]
    if export_depth:
        launcher_args.append("--export_depth")
    if command[1] == str(FIXED_INPUT_LAUNCHER):
        separator = command.index("--")
        launcher_args += command[2:separator]
        command = [command[0], *command[separator + 1:]]
    return [command[0], str(PHASE_LAUNCHER), *launcher_args, "--", *command[1:]]


def run_system(command: list[str], env: dict[str, str], work: pathlib.Path,
               sample_gpu: bool) -> tuple[float, float | None]:
    """Runs `command` in `work`.

    Args:
        command: The command.
        env: Its environment.
        work: Its working directory.
        sample_gpu: Sample the process tree's GPU memory.

    Returns:
        (wall seconds, peak GPU MiB or None).

    Raises:
        SystemExit: The command failed.
    """
    print("Running:", " ".join(command), flush=True)
    started_at = time.time()
    process = subprocess.Popen(command, cwd=work, env=env)
    sampler = slam.GpuMemorySampler(process.pid) if sample_gpu else None
    if sampler:
        sampler.start()
    status = process.wait()
    peak = sampler.stop() if sampler else None
    if status != 0:
        raise SystemExit(f"{command[1]} exited with status {status}")
    return time.time() - started_at, peak


def export_system_depths(method: str, work: pathlib.Path,
                         sink: slam_depth.DepthSink, poses: dict[int,
                                                                 np.ndarray],
                         frame_count: int) -> None:
    """Writes the run's per-frame depth and confidence (`slam_depth`).

    Args:
        method: One of `METHODS`.
        work: The work directory.
        sink: Where the maps go.
        poses: The final {input index: c2w}.
        frame_count: Number of input frames.

    Raises:
        RuntimeError: Not exactly one chunk directory.
    """
    if method == "VGGT_SLAM":
        slam_depth.export_vggt_slam_dumps(sink, work / "depth_dump")
        return
    chunk_dirs = sorted(work.rglob("_tmp_results_unaligned"))
    if len(chunk_dirs) != 1:
        raise RuntimeError(f"Expected one _tmp_results_unaligned under "
                           f"{work}, found {len(chunk_dirs)}")
    model = yaml.safe_load(
        (work / "config.yaml").read_text(encoding="utf-8"))["Model"]
    ranges = slam_depth.chunk_ranges(frame_count, int(model["chunk_size"]),
                                     int(model["overlap"]))
    slam_depth.export_long_chunks(sink, chunk_dirs[0], ranges, poses, method)


class LongSequenceSlam(slam.SlamModel):
    """A long-sequence SLAM system (long_sequence_slam.py)."""

    description = DESCRIPTION

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            "--phase",
            choices=("all", "gpu", "align"),
            default="all",
            help=("VGGT-Long/Pi-Long: 'gpu' = chunk inference only, 'align' "
                  "= the CPU alignment a 'gpu' run left pending, 'all' = "
                  "both in one process. VGGT-SLAM runs 'all' for 'gpu'."))

    @classmethod
    def weight_paths(cls, ckpt_root):
        return weight_paths(ckpt_root)

    def run_case(self, weights):
        # pylint: disable=too-many-locals
        args = self.args
        method = self.method_name
        image_paths = self.image_paths
        code_path = self.code_path
        if args.phase == "align" and method not in SPLIT_METHODS:
            raise SystemExit(f"{method} has no separate alignment phase")
        phase = args.phase if method in SPLIT_METHODS else "all"
        pending_path = args.output_path / ALIGN_PENDING
        fixed = utils.fixed_input_size(image_paths)
        sink = slam_depth.DepthSink(args.output_path,
                                    image_paths) if args.export_depth else None

        work: pathlib.Path | None = None
        try:
            if phase == "align":
                pending = json.loads(pending_path.read_text(encoding="utf-8"))
                work = pathlib.Path(pending["work"])
                if sink is not None and not pending.get("export_depth"):
                    # A GPU phase run without --export_depth dropped the
                    # chunk arrays the depth export reads.
                    pending_path.unlink()
                    raise RuntimeError(
                        f"{method}: the pending GPU phase kept no chunk "
                        "depth; rerun the inference phase")
                depth_dump = work / "depth_dump" if sink is not None else None
                command, env = system_command(method, code_path, work,
                                              bool(fixed), depth_dump)
                align_seconds, _ = run_system(phase_command(
                    command, "align", work),
                                              env,
                                              work,
                                              sample_gpu=False)
                peak = pending["peak_gpu_memory_mib"]
                timing = {
                    "inference_seconds": pending["gpu_seconds"] + align_seconds,
                    "gpu_phase_seconds": pending["gpu_seconds"],
                    "align_phase_seconds": align_seconds
                }
                # write_adapter_metrics' "inference time" counts from
                # started_at: the two phases, not the wait between jobs.
                started_at = time.time() - timing["inference_seconds"]
            else:
                args.output_path.mkdir(parents=True, exist_ok=True)
                # A split run's work directory must outlive this job and be
                # readable from the align job's node.
                work = (pathlib.Path(
                    tempfile.mkdtemp(prefix=f"{method.lower()}_",
                                     dir=args.output_path)) if phase == "gpu"
                        else utils.slam_work_dir(method, args.output_path))
                slam.stage_images(image_paths, work / "images")
                depth_dump = work / "depth_dump" if sink is not None else None
                if method != "VGGT_SLAM":
                    write_long_config(method,
                                      code_path,
                                      weights,
                                      work / "config.yaml",
                                      fixed,
                                      keep_chunks=sink is not None)
                if method in ("VGGT_LONG", "PI_LONG"):
                    # Their loop detector loads DINOv2 from the cwd-relative
                    # ./LoopModels/dinov2; the run's cwd is the work
                    # directory (outputs go to ./exps), so link the repo's
                    # LoopModels there.
                    (work / "LoopModels").symlink_to(code_path / "LoopModels",
                                                     target_is_directory=True)
                command, env = system_command(method, code_path, work,
                                              bool(fixed), depth_dump)
                if phase == "gpu":
                    command = phase_command(command,
                                            "gpu",
                                            work,
                                            export_depth=sink is not None)
                started_at = time.time()
                elapsed, peak = run_system(command, env, work, sample_gpu=True)
                timing = {"inference_seconds": elapsed}
                if phase == "gpu" and (work / "phase_state.pkl").is_file():
                    pending_path.write_text(json.dumps({
                        "method": method,
                        "work": str(work),
                        "started_at": started_at,
                        "gpu_seconds": elapsed,
                        "peak_gpu_memory_mib": peak,
                        "export_depth": sink is not None
                    }) + "\n",
                                            encoding="utf-8")
                    print(f"{method} GPU phase done in {elapsed:.0f} s; "
                          f"alignment pending ({pending_path})")
                    return

            if method == "VGGT_SLAM":
                poses = read_tum_poses(work / "poses.txt")
            else:
                poses = read_long_poses(work / "exps", len(image_paths))
            indices = self.pose_indices(poses)
            if sink is not None:
                export_system_depths(method, work, sink, poses,
                                     len(image_paths))
            returned = self.export_poses(poses, indices)
            input_record = work / "model_input_size.json"
            depth_frame = ("submap" if method == "VGGT_SLAM" else "trajectory")
            utils.write_adapter_metrics(
                output_path=args.output_path,
                method=method,
                image_paths=image_paths,
                started_at=started_at,
                checkpoint=str(code_path),
                depth_only=False,
                metrics_extra={
                    "track":
                        args.track,
                    "frames_with_pose":
                        len(returned),
                    "intrinsics":
                        "nominal",
                    **(sink.metrics() | {
                           "depth_frame": depth_frame
                       } if sink is not None else {})
                },
                # The system runs in a child process: wall time includes its
                # model loading, and peak memory is the child's process-tree
                # peak from nvidia-smi (sampled at 1 s), not
                # torch.cuda.max_memory_allocated. A split run's
                # inference_seconds is its GPU phase plus its alignment phase
                # (each also reported).
                runtime={
                    **timing, "model_load_seconds":
                        None,
                    "peak_gpu_memory_mib":
                        peak,
                    "runtime_scope":
                        "subprocess_wall",
                    "model_input_size":
                        json.loads(input_record.read_text(
                            encoding="utf-8"))["model_input_size"]
                        if input_record.is_file() else None
                },
            )
            pending_path.unlink(missing_ok=True)
            if not args.keep_intermediate:
                shutil.rmtree(work, ignore_errors=True)
        except BaseException:
            # A failed case leaves no work directory (staged frames, chunk
            # arrays: often many GB) in the prediction cache. A failed
            # alignment phase keeps it: it holds the finished GPU phase that
            # a retry needs.
            keep = args.keep_intermediate or phase == "align"
            if work is not None and not keep:
                shutil.rmtree(work, ignore_errors=True)
            raise


@registry.register_model
class VGGTLong(LongSequenceSlam):
    """VGGT-Long (vggt_long.py, VGGT-1B backbone, SALAD loop closure)."""

    name = "VGGT_LONG"
    code_dir = CODE_DIRS["VGGT_LONG"]
    ckpt_key = "VGGT"


@registry.register_model
class PiLong(LongSequenceSlam):
    """Pi-Long (pi_long.py, Pi3 backbone, SALAD loop closure)."""

    name = "PI_LONG"
    aliases = ("PI3_LONG",)
    code_dir = CODE_DIRS["PI_LONG"]
    ckpt_key = "Pi3"


@registry.register_model
class VGGTSlam(LongSequenceSlam):
    """VGGT-SLAM (main.py with its evals/eval_tum.sh settings)."""

    name = "VGGT_SLAM"
    code_dir = CODE_DIRS["VGGT_SLAM"]
