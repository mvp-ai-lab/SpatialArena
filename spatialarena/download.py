# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Dataset and checkpoint download automation for `BenchmarkEvaluator`.

Wraps the existing `scripts/download/*` and `scripts/preprocess/materialize_*`
tooling into a single orchestrated API (`download_all_supported_datasets()`)
instead of a user having to hand-chain the commands documented in
docs/benchmark/evaluator_quickstart.md's Section 2.

Not every registered dataset has a fully-automated download+materialize
chain verified here -- some (`SfM`/`MVRecon` `eth3d`, `dtu`,
`tanks_and_temples`) have a working *download* script but no verified
canonical-layout materializer, so they're marked `status="download_only"`
rather than silently claimed as complete. See each `DatasetDownloadEntry`'s
`notes` for specifics. This module never guesses at a missing step.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from spatialarena.eval.dataset_registry import describe_benchmark_scenes

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class StepResult:
    name: str
    command: list[str]
    returncode: int
    stdout_tail: str
    stderr_tail: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass(frozen=True)
class DatasetDownloadEntry:
    task: str
    dataset: str
    status: str  # "automated" | "download_only" | "manual"
    notes: str = ""
    # Each step is (name, argv). Steps run in order; a failure stops the chain.
    steps: tuple[tuple[str, tuple[str, ...]], ...] = ()
    env: dict[str, str] = field(default_factory=dict)


def _bash(script: str) -> str:
    return str(REPO_ROOT / script)


def _python() -> str:
    return sys.executable


def _slam_inputs_step(dataset: str) -> tuple[str, tuple[str, ...]]:
    """The step writing ``SLAM/<dataset>``, the fixed-resolution frames every method reads, from the
    ``SLAM_source/<dataset>`` sequences the dataset's materialize step wrote (docs/benchmark/slam_protocol.md)."""
    return (
        "slam_inputs",
        (_python(), _bash("scripts/preprocess/materialize_slam_inputs.py"), "--benchmark_root", "{benchmark_root}",
         "--datasets", dataset),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# env values are format strings substituted with {benchmark_root}.

DATASET_DOWNLOADS: tuple[DatasetDownloadEntry, ...] = (
    DatasetDownloadEntry(
        task="MonoDepth",
        dataset="marigold",
        status="automated",
        steps=(
            ("download", (_bash("scripts/download/marigold.sh"),)),
            (
                "materialize",
                (
                    _python(),
                    _bash("scripts/preprocess/materialize_marigold_sample.py"),
                    "--marigold_root",
                    "{benchmark_root}/MonoDepth/marigold",
                    "--scenes",
                    "nyuv2",
                    "kitti",
                    "eth3d",
                    "diode",
                ),
            ),
        ),
        env={"DATA_ROOT_PATH": "{benchmark_root}/MonoDepth"},
    ),
    *(
        DatasetDownloadEntry(
            task="MonoDepth",
            dataset=dataset,
            status="automated",
            notes=f"MoGe's processed {moge_name} (HF Ruicheng/monocular-geometry-evaluation, pinned revision).",
            steps=(
                ("download", (_bash("scripts/download/moge_eval.sh"), moge_name)),
                (
                    "materialize",
                    (
                        _python(),
                        _bash("scripts/preprocess/materialize_moge_eval.py"),
                        "--source_root",
                        "{benchmark_root}/MonoDepth_source/moge_eval",
                        "--benchmark_root",
                        "{benchmark_root}",
                        "--datasets",
                        dataset,
                    ),
                ),
            ),
            env={"DATA_ROOT_PATH": "{benchmark_root}/MonoDepth_source"},
        )
        for dataset, moge_name in (("hammer", "HAMMER"), ("ibims1", "iBims-1"), ("spring", "Spring"))
    ),
    DatasetDownloadEntry(
        task="MonoDepth",
        dataset="aerialmetric",
        status="automated",
        notes="AerialMetric Test-Split (~15 GB download); 100 frames each of Decoupled, Oblique and Wild.",
        steps=(
            ("download", (_bash("scripts/download/aerialmetric.sh"),)),
            (
                "materialize",
                (
                    _python(),
                    _bash("scripts/preprocess/materialize_aerialmetric.py"),
                    "--source_root",
                    "{benchmark_root}/MonoDepth_source/aerialmetric/extracted",
                    "--benchmark_root",
                    "{benchmark_root}",
                ),
            ),
        ),
        env={"DATA_ROOT_PATH": "{benchmark_root}/MonoDepth_source"},
    ),
    DatasetDownloadEntry(
        task="MonoDepth",
        dataset="synth4k",
        status="automated",
        notes="Five game tars (~87 GB); the materializer reads the committed 100-frame lists straight from the tars.",
        steps=(
            ("download", (_bash("scripts/download/synth4k.sh"),)),
            (
                "materialize",
                (
                    _python(),
                    _bash("scripts/preprocess/materialize_synth4k.py"),
                    "--source_root",
                    "{benchmark_root}/MonoDepth_source/synth4k",
                    "--benchmark_root",
                    "{benchmark_root}",
                ),
            ),
        ),
        env={"DATA_ROOT_PATH": "{benchmark_root}/MonoDepth_source"},
    ),
    DatasetDownloadEntry(
        task="VideoDepth",
        dataset="sintel",
        status="automated",
        steps=(("download", (_bash("scripts/download/sintel.sh"),)),),
        env={"DATA_ROOT_PATH": "{benchmark_root}/VideoDepth"},
    ),
    DatasetDownloadEntry(
        task="VideoDepth",
        dataset="bonn",
        status="automated",
        notes="Download script also runs bonn_preprocessing.py itself.",
        steps=(("download_and_preprocess", (_bash("scripts/download/bonn.sh"),)),),
        env={
            "DATA_ROOT_PATH": "{benchmark_root}/VideoDepth",
            "PROJECT_ROOT_PATH": str(REPO_ROOT),
        },
    ),
    DatasetDownloadEntry(
        task="VideoDepth",
        dataset="kitti",
        status="automated",
        notes="Download script also runs kitti_preprocessing.py itself.",
        steps=(("download_and_preprocess", (_bash("scripts/download/kitti.sh"),)),),
        env={
            "DATA_ROOT_PATH": "{benchmark_root}/VideoDepth",
            "PROJECT_ROOT_PATH": str(REPO_ROOT),
        },
    ),
    DatasetDownloadEntry(
        task="VideoDepth",
        dataset="dycheck",
        status="automated",
        notes=("MegaSaM's DyCheck evaluation zips (Google Drive); materialized as camera-0 symlinks into the raw "
               "folder, which must stay."),
        steps=(
            ("download", (_bash("scripts/download/dycheck.sh"),)),
            (
                "materialize",
                (
                    _python(),
                    _bash("scripts/preprocess/materialize_dycheck.py"),
                    "--source_root",
                    "{dycheck_staging_root}/dycheck_raw",
                    "--benchmark_root",
                    "{benchmark_root}",
                ),
            ),
        ),
        env={"DATA_ROOT_PATH": "{dycheck_staging_root}"},
    ),
    *(
        DatasetDownloadEntry(
            task=task,
            dataset="spatialarena",
            status="automated",
            notes="Our own captures: fetched from the benchmark's Hugging Face release and unpacked.",
            steps=(
                (
                    "download",
                    (_python(), _bash("scripts/download/spatialarena.py"), "--benchmark_root", "{benchmark_root}",
                     "--tasks", task),
                ),
            ),
        )
        for task in ("MonoDepth", "VideoDepth", "MVRecon", "SfM", "SLAM")
    ),
    DatasetDownloadEntry(
        task="SfM",
        dataset="imc",
        status="automated",
        notes="Download script also runs imc_to_colmap.py itself.",
        steps=(("download_and_preprocess", (_bash("scripts/download/imc.sh"),)),),
        env={
            "DATA_ROOT_PATH": "{benchmark_root}/SfM",
            "PROJECT_ROOT_PATH": str(REPO_ROOT),
        },
    ),
    DatasetDownloadEntry(
        task="MVRecon",
        dataset="7scenes",
        status="automated",
        notes=(
            "Download script also runs 7scenes_preprocessing.py itself. "
            "SLAM/7scenes is materialized from this data by the SLAM/7scenes "
            "entry (no separate download)."
        ),
        steps=(("download_and_preprocess", (_bash("scripts/download/7scenes.sh"),)),),
        env={
            "DATA_ROOT_PATH": "{benchmark_root}/MVRecon",
            "PROJECT_ROOT_PATH": str(REPO_ROOT),
        },
    ),
    DatasetDownloadEntry(
        task="SfM",
        dataset="texturepoorsfm",
        status="automated",
        steps=(
            ("download", (_bash("scripts/download/texturepoorsfm.sh"),)),
            (
                "materialize",
                (
                    _python(),
                    _bash("scripts/preprocess/materialize_texturepoorsfm.py"),
                    "--source_root",
                    "{texturepoorsfm_staging_root}",
                    "--benchmark_root",
                    "{benchmark_root}",
                ),
            ),
        ),
        env={"OUTPUT_DIR": "{texturepoorsfm_staging_root}"},
    ),
    DatasetDownloadEntry(
        task="MVRecon",
        dataset="scannetpp",
        status="automated",
        notes="Same da3_bench.sh download also materializes MVRecon/hiroom + SfM sides via legacy_relative_paths.",
        steps=(
            ("download", (_bash("scripts/download/da3_bench.sh"),)),
            (
                "materialize",
                (
                    _python(),
                    _bash("scripts/preprocess/materialize_da3_bench.py"),
                    "--source_root",
                    "{da3_bench_staging_root}",
                    "--benchmark_root",
                    "{benchmark_root}",
                    "--datasets",
                    "scannetpp",
                    "hiroom",
                ),
            ),
        ),
        env={"OUTPUT_DIR": "{da3_bench_staging_root}"},
    ),
    DatasetDownloadEntry(
        task="MVRecon",
        dataset="hiroom",
        status="automated",
        notes=("Materialized by the scannetpp entry's da3_bench.sh + materialize_da3_bench.py run "
               "(same source download)."),
        steps=(),
    ),
    DatasetDownloadEntry(
        task="SLAM",
        dataset="7scenes",
        status="automated",
        notes=(
            "Links the 18 official TestSplit.txt sequences of MVRecon/7scenes (run the MVRecon/7scenes entry "
            "first) into one SLAM case per sequence via relative symlinks (docs/benchmark/slam_protocol.md)."
        ),
        steps=(
            (
                "materialize",
                (
                    _python(),
                    _bash("scripts/preprocess/materialize_7scenes_slam.py"),
                    "--source_root",
                    "{benchmark_root}/MVRecon/7scenes",
                    "--benchmark_root",
                    "{benchmark_root}",
                ),
            ),
            _slam_inputs_step("7scenes"),
        ),
    ),
    DatasetDownloadEntry(
        task="SfM",
        dataset="scannetpp",
        status="automated",
        notes="Resolves to MVRecon/scannetpp via DatasetSpec.legacy_relative_paths -- see the MVRecon/scannetpp entry.",
        steps=(),
    ),
    DatasetDownloadEntry(
        task="SfM",
        dataset="hiroom",
        status="automated",
        notes="Resolves to MVRecon/hiroom via DatasetSpec.legacy_relative_paths -- see the MVRecon/hiroom entry.",
        steps=(),
    ),
    DatasetDownloadEntry(
        task="SfM",
        dataset="eth3d",
        status="download_only",
        notes=(
            "SfM/eth3d is a symlink to MVRecon/eth3d -- see the MVRecon/eth3d entry, which downloads and lays out "
            "both."
        ),
        steps=(),
    ),
    DatasetDownloadEntry(
        task="MVRecon",
        dataset="eth3d",
        status="download_only",
        notes=(
            "scripts/download/eth3d.sh downloads each scene's dslr_jpg / dslr_undistorted / dslr_depth archives "
            "straight into the canonical MVRecon/eth3d/<scene>/ layout (the one on disk), and "
            "scripts/preprocess/finalize_mvrecon_layout.sh copies the laser-scan depth to depths/ and writes the "
            "metadata. The GT meshes come from DA3-BENCH (scripts/preprocess/materialize_da3_bench.py, 11 of 13 "
            "scenes). Not yet verified end to end from an empty root."
        ),
        steps=(
            ("download", (_bash("scripts/download/eth3d.sh"), "{benchmark_root}/MVRecon/eth3d")),
            ("finalize", (_bash("scripts/preprocess/finalize_mvrecon_layout.sh"),)),
        ),
        env={"MVRECON_PATH": "{benchmark_root}/MVRecon", "PROJECT_ROOT_PATH": str(REPO_ROOT)},
    ),
    DatasetDownloadEntry(
        task="SfM",
        dataset="dtu",
        status="download_only",
        notes=(
            "scripts/download/dtu.sh fetches DTU's official zips but does not "
            "materialize the canonical layout -- see "
            "scripts/preprocess/fuse_depth_map_dtu.py, which needs manual "
            "invocation and inspection before this can be called automated. "
            "MVRecon/dtu is a separate DatasetSpec with the same caveat -- "
            "see that entry."
        ),
        steps=(("download_raw_only", ("bash", _bash("scripts/download/dtu.sh"), "{dtu_staging_root}")),),
    ),
    DatasetDownloadEntry(
        task="MVRecon",
        dataset="dtu",
        status="download_only",
        notes=(
            "Same scripts/download/dtu.sh raw download as SfM/dtu, and the "
            "same caveat: no verified script here materializes the canonical "
            "MVRecon/dtu layout from DTU's official zips."
        ),
        steps=(("download_raw_only", ("bash", _bash("scripts/download/dtu.sh"), "{dtu_staging_root}")),),
    ),
    DatasetDownloadEntry(
        task="SfM",
        dataset="tanks_and_temples",
        status="manual",
        notes=("No download script found under scripts/download/ for this repo's Tanks-and-Temples copy. "
               "Obtain from the dataset's own official release."),
    ),
    DatasetDownloadEntry(
        task="MVRecon",
        dataset="tanks_and_temples",
        status="manual",
        notes=("No download script found under scripts/download/ for this repo's Tanks-and-Temples copy. "
               "Obtain from the dataset's own official release."),
    ),
    DatasetDownloadEntry(
        task="MVRecon",
        dataset="nrgbd",
        status="automated",
        steps=(
            ("download", (_bash("scripts/download/nrgbd.sh"),)),
            (
                "materialize",
                (
                    _python(),
                    _bash("scripts/preprocess/materialize_nrgbd.py"),
                    "--source_root",
                    "{nrgbd_staging_root}/nrgbd_raw",
                    "--benchmark_root",
                    "{benchmark_root}",
                ),
            ),
        ),
        env={"DATA_ROOT_PATH": "{nrgbd_staging_root}"},
    ),
    DatasetDownloadEntry(
        task="SLAM",
        dataset="tum_dynamics",
        status="automated",
        steps=(
            ("download", (_bash("scripts/download/tum_dynamics.sh"),)),
            (
                "materialize",
                (
                    _python(),
                    _bash("scripts/preprocess/materialize_tum_dynamics.py"),
                    "--source_root",
                    "{tum_dynamics_staging_root}/tum_dynamics_raw",
                    "--benchmark_root",
                    "{benchmark_root}",
                ),
            ),
            _slam_inputs_step("tum_dynamics"),
        ),
        env={"DATA_ROOT_PATH": "{tum_dynamics_staging_root}"},
    ),
    DatasetDownloadEntry(
        task="SLAM",
        dataset="oxford_spires",
        status="automated",
        notes="~76 GB of raw cam0 images (10 sequences); materialized with lingbot-map's preprocessing (its code clone "
        "must be present, see METHOD_CODE_REPOS).",
        steps=(
            ("download", (_bash("scripts/download/oxford_spires.sh"),)),
            (
                "materialize",
                (
                    _python(),
                    _bash("scripts/preprocess/materialize_oxford_spires.py"),
                    "--raw_root",
                    "{oxford_spires_staging_root}/oxford_spires_raw",
                    "--benchmark_root",
                    "{benchmark_root}",
                ),
            ),
            _slam_inputs_step("oxford_spires"),
        ),
        env={"DATA_ROOT_PATH": "{oxford_spires_staging_root}"},
    ),
    DatasetDownloadEntry(
        task="SLAM",
        dataset="vbr",
        status="automated",
        notes="LoGeR's processed VBR release (~121 GB of archives, 7 sequences).",
        steps=(
            ("download", (_bash("scripts/download/vbr.sh"),)),
            (
                "materialize",
                (
                    _python(),
                    _bash("scripts/preprocess/materialize_vbr.py"),
                    "--raw_root",
                    "{vbr_staging_root}/vbr_raw",
                    "--benchmark_root",
                    "{benchmark_root}",
                ),
            ),
            _slam_inputs_step("vbr"),
        ),
        env={"DATA_ROOT_PATH": "{vbr_staging_root}"},
    ),
    DatasetDownloadEntry(
        task="SLAM",
        dataset="kitti_odometry",
        status="automated",
        notes="Official KITTI odometry release (~65 GB color archive); image_2 of sequences 00-10 is extracted.",
        steps=(
            ("download", (_bash("scripts/download/kitti_odometry.sh"),)),
            (
                "materialize",
                (
                    _python(),
                    _bash("scripts/preprocess/materialize_kitti_odometry.py"),
                    "--raw_root",
                    "{kitti_odometry_staging_root}/kitti_odometry_raw",
                    "--benchmark_root",
                    "{benchmark_root}",
                ),
            ),
            _slam_inputs_step("kitti_odometry"),
        ),
        env={"DATA_ROOT_PATH": "{kitti_odometry_staging_root}"},
    ),
)


def list_supported_datasets() -> list[dict]:
    """Return the full dataset download registry as plain dicts -- what
    `download_all_supported_datasets()` can and can't automate, and why."""
    return [
        {
            "task": entry.task,
            "dataset": entry.dataset,
            "status": entry.status,
            "notes": entry.notes,
            "automated_steps": [name for name, _ in entry.steps],
        }
        for entry in DATASET_DOWNLOADS
    ]


# Optional progress hook for the download_all_* functions: called with one event dict per state change,
# {"kind": "dataset" | "checkpoint" | "code", "name": str, "status": "pending" | "running" | "ok" | "failed" |
# "skipped" | "already_present", "step": str | None, "detail": str}. Every item is first reported "pending", so a
# consumer knows the totals up front (spatialarena/progress_ui.py renders these as bars or in the board).
DownloadEventFn = Callable[[dict], None]


def _emit(on_event: DownloadEventFn | None, kind: str, name: str, status: str, step: str | None = None,
          detail: str = "") -> None:
    if on_event is not None:
        on_event({"kind": kind, "name": name, "status": status, "step": step, "detail": detail})


def _result_detail(result: dict) -> str:
    if result.get("reason"):
        return str(result["reason"])
    tail = result.get("stderr_tail") or next(
        (step.get("stderr_tail") for step in result.get("steps", []) if step.get("returncode")), ""
    )
    return (tail or "").strip().splitlines()[-1] if tail and tail.strip() else ""


def _run_step(name: str, argv: tuple[str, ...], *, env: dict[str, str], log_dir: Path) -> StepResult:
    log_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        list(argv),
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    (log_dir / f"{name}.stdout.log").write_text(result.stdout, encoding="utf-8")
    (log_dir / f"{name}.stderr.log").write_text(result.stderr, encoding="utf-8")
    return StepResult(
        name=name,
        command=list(argv),
        returncode=result.returncode,
        stdout_tail="\n".join(result.stdout.splitlines()[-20:]),
        stderr_tail="\n".join(result.stderr.splitlines()[-20:]),
    )


def download_dataset(
    entry: DatasetDownloadEntry,
    *,
    benchmark_root: Path,
    staging_root: Path,
    process_env: dict[str, str],
    log_dir: Path,
    on_step: Callable[[str], None] | None = None,
) -> dict:
    if entry.status != "automated":
        return {
            "task": entry.task,
            "dataset": entry.dataset,
            "status": "skipped",
            "reason": f"registry status is {entry.status!r}, not automated: {entry.notes}",
            "steps": [],
        }

    fmt = {
        "benchmark_root": str(benchmark_root),
        "texturepoorsfm_staging_root": str(staging_root / "texturepoorsfm"),
        "da3_bench_staging_root": str(staging_root / "da3_bench"),
        "nrgbd_staging_root": str(staging_root / "nrgbd"),
        "tum_dynamics_staging_root": str(staging_root / "tum_dynamics"),
        "dycheck_staging_root": str(staging_root / "dycheck"),
        "oxford_spires_staging_root": str(staging_root / "oxford_spires"),
        "vbr_staging_root": str(staging_root / "vbr"),
        "kitti_odometry_staging_root": str(staging_root / "kitti_odometry"),
    }
    env = dict(process_env)
    for key, value in entry.env.items():
        env[key] = value.format(**fmt)

    results: list[StepResult] = []
    for name, argv in entry.steps:
        if on_step is not None:
            on_step(name)
        formatted_argv = tuple(arg.format(**fmt) for arg in argv)
        result = _run_step(name, formatted_argv, env=env, log_dir=log_dir / f"{entry.task}_{entry.dataset}")
        results.append(result)
        if not result.ok:
            break

    ok = all(r.ok for r in results)  # vacuously True when entry.steps is empty (materialized by another entry)
    return {
        "task": entry.task,
        "dataset": entry.dataset,
        "status": "ok" if ok else "failed",
        "steps": [
            {"name": r.name, "returncode": r.returncode, "stderr_tail": r.stderr_tail}
            for r in results
        ],
    }


def download_all_supported_datasets(
    *,
    benchmark_root: Path,
    datasets: list[tuple[str, str]] | None = None,
    staging_root: Path | None = None,
    log_dir: Path | None = None,
    process_env: dict[str, str] | None = None,
    skip_existing: bool = True,
    on_event: DownloadEventFn | None = None,
) -> list[dict]:
    """Run every automated download+materialize chain in DATASET_DOWNLOADS.

    `datasets`, if given, restricts to those (task, dataset) pairs (still
    subject to each entry's own `status`). Entries with `status !=
    "automated"` are reported as `status="skipped"` with the reason, never
    silently attempted. Returns one result dict per attempted entry, in the
    same shape `download_dataset()` returns.
    """
    benchmark_root = Path(benchmark_root).expanduser()
    staging_root = Path(staging_root).expanduser() if staging_root else benchmark_root.parent
    log_dir = Path(log_dir).expanduser() if log_dir else benchmark_root / "_download_logs"
    process_env = process_env if process_env is not None else dict(os.environ)

    entries = DATASET_DOWNLOADS
    if datasets is not None:
        wanted = {(task, ds) for task, ds in datasets}
        entries = tuple(e for e in entries if (e.task, e.dataset) in wanted)

    results = []
    for entry in entries:
        _emit(on_event, "dataset", f"{entry.task}/{entry.dataset}", "pending")
    for entry in entries:
        name = f"{entry.task}/{entry.dataset}"
        _emit(on_event, "dataset", name, "running")
        if skip_existing and entry.status == "automated":
            existing = describe_benchmark_scenes(
                benchmark_root=benchmark_root, tasks=[entry.task], datasets=[entry.dataset]
            )
            if existing and existing[0]["exists"] and (
                not existing[0]["metadata_required"] or existing[0]["metadata_exists"]
            ):
                results.append(
                    {
                        "task": entry.task,
                        "dataset": entry.dataset,
                        "status": "already_present",
                        "steps": [],
                    }
                )
                _emit(on_event, "dataset", name, "already_present")
                continue
        result = download_dataset(
            entry,
            benchmark_root=benchmark_root,
            staging_root=staging_root,
            process_env=process_env,
            log_dir=log_dir,
            on_step=lambda step, name=name: _emit(on_event, "dataset", name, "running", step=step),
        )
        results.append(result)
        _emit(on_event, "dataset", name, result["status"], detail=_result_detail(result))
    return results


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
# Reuses the same 4-way categorization BenchmarkEvaluator._checkpoint_issue()
# already applies: HF hub id, direct URL, local-file path, or empty/unset.
# Only HF hub ids and direct URLs have a genuinely automatable, verified
# source here -- local-file entries (VGGT, DA3-G, FastVGGT, VGGT-Omega on
# this cluster) are this cluster's own private pre-migrated copies with no
# discovered public download source, so those are reported "manual" rather
# than guessed at.


# Some methods' real checkpoint distribution channel is a Google Drive link
# rather than an HF hub id or a plain-HTTP URL -- gdown is needed to resolve
# Drive's confirmation-page redirect. Covers two ckpt_default shapes: a
# $METHOD_CODE_ROOT-relative path (kind="variable_dependent", the checkpoint
# is expected inside the method's own vendored code checkout per its own
# README -- CUT3R's) and a plain canonical local_file path under
# checkpoint_root/local/<method>/ like every other local-file checkpoint
# (AMB3R's, since AMB3R's own load_weights() takes an arbitrary path, not a
# path relative to its code checkout). Each URL here was read directly off
# the method's own official README/download instructions, not guessed. A
# "variable_dependent" checkpoint with no entry here (e.g. COME's, which
# ships committed inside its own code clone -- see spatialarena/
# methods.py) needs no separate download at all.
GDRIVE_CHECKPOINT_URLS: dict[str, str] = {
    "CUT3R": "https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link",
    "AMB3R": "https://drive.google.com/file/d/14x0WW2rUE_he2hUEouP6ywSRnlJDeLel/view?usp=sharing",
}

# Some methods' ckpt_default is a $METHOD_CODE_ROOT-relative path whose real
# distribution channel is a single named file on the HF hub, not the whole
# repo snapshotted into the generic HF cache (that method's own README
# expects the file at this exact relative path). Each (repo_id, filename)
# must be read directly off the method's own README + the HF repo's real
# file listing, not guessed. A gated repo needs the account behind HF_TOKEN
# to have been granted access at https://huggingface.co/<repo_id> before this
# can succeed -- download_checkpoint() reports that plainly rather than
# crashing the whole download_all_supported_models() batch. None at present:
# IGGT4D's checkpoint moved to HF_LOCAL_FILE_CHECKPOINTS (its adapter points
# upstream's relative path at --ckpt_path).
HF_FILE_CHECKPOINTS: dict[str, tuple[str, str]] = {}

# Local-file checkpoints fetched as named files from the HF hub into checkpoint_root/local/<subdir>/ (each file
# keeps its repo-relative path), for methods whose code loads a file path rather than a hub id -- IGGT4D (gated:
# HF_TOKEN's account needs access) and the streaming SLAM methods of scripts/inference/streaming_slam.py, whose
# weight_paths() reads these locations. Value:
# (repo_id, subdir, files); the first file is the method's ckpt_default. Scal3R also needs DINO-SALAD, and ABot-Recon's
# loop closure DINO-SALAD and DINOv2 ViT-B/14, both shared with VGGT-Long / Pi-Long under local/slam_long/.
HF_LOCAL_FILE_CHECKPOINTS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "IGGT4D": ("HorizonRobotics/IGGT4D", "iggt4d", ("model.safetensors",)),
    "LOGER": ("Junyi42/LoGeR", "loger", ("LoGeR/latest.pt", "LoGeR/original_config.yaml")),
    "ZIPMAP": ("coast01/ZipMap", "zipmap", ("checkpoint_aff_inv.pt",)),
    "LINGBOT_MAP": ("robbyant/lingbot-map", "lingbot_map", ("lingbot-map.pt",)),
    "R3": ("KevinXu02/R3", "r3", ("r3.safetensors",)),
    "SCAL3R": ("xbillowy/Scal3R", "scal3r", ("scal3r.pt",)),
    "ABOT_RECON": ("acvlab/ABot-Recon", "abot_recon", ("abot_recon.safetensors", "config.json")),
}

# Some methods' checkpoint ships committed directly inside their own vendored
# code clone (no separate network download at all -- e.g. COME's
# confidence-distill checkpoint is tracked in the CoMe git repo itself), but
# is still canonicalized under checkpoint_root/local/<method>/ like every
# other local-file checkpoint rather than left inside the code checkout.
# Value is the checkpoint's path relative to the method's own code leaf dir
# (see _code_leaf_dir).
SHIPPED_WITH_CODE_CHECKPOINTS: dict[str, str] = {
    "COME": "output/confidence_distill/vggt/20260324_172742/step_02000/checkpoint.pth",
}


# Further HF hub snapshots a method needs besides its own ckpt_default, fetched into the shared HF cache by
# download_checkpoint() after it: (repo_id, allow_patterns or None for the whole repo). DVD runs on the
# Wan2.1-T2V-1.3B DiT + VAE (dvd.py's --wan_path / WAN_MODEL_PATH), of which it needs only these files.
# DepthCrafter and ChronoDepth load Stable Video Diffusion's fp16 image encoder, VAE, scheduler and feature
# extractor (video_diffusion_depth.py's --base_path / SVD_BASE_PATH) around their own UNet; the base repo is gated
# on the HF hub, so its license must be accepted with the account whose token downloads it.
_SVD_BASE = ("stabilityai/stable-video-diffusion-img2vid-xt",
             ("model_index.json", "feature_extractor/*", "scheduler/*", "image_encoder/config.json",
              "image_encoder/*.fp16.safetensors", "vae/config.json", "vae/*.fp16.safetensors"))
HF_EXTRA_SNAPSHOTS: dict[str, tuple[tuple[str, tuple[str, ...] | None], ...]] = {
    "DVD": (("Wan-AI/Wan2.1-T2V-1.3B", ("diffusion_pytorch_model*.safetensors", "Wan2.1_VAE.pth", "config.json")),),
    "DEPTHCRAFTER": (_SVD_BASE,),
    "CHRONODEPTH": (_SVD_BASE,),
}


def _checkpoint_kind(ckpt: str) -> str:
    if not ckpt:
        return "none"
    if ckpt.startswith("http://") or ckpt.startswith("https://"):
        return "url"
    if "$" in ckpt:
        return "variable_dependent"
    if ckpt.startswith("/"):
        return "local_file"
    return "hf_hub"


def list_supported_models() -> list[dict]:
    """Per-method checkpoint download status, derived live from the method
    registry (not a separate hand-maintained list)."""
    from spatialarena.methods import METHOD_SPECS

    out = []
    for spec in METHOD_SPECS:
        kind = _checkpoint_kind(spec.ckpt_default or "")
        gdrive = kind in ("variable_dependent", "local_file") and spec.name in GDRIVE_CHECKPOINT_URLS
        hf_file = kind == "variable_dependent" and spec.name in HF_FILE_CHECKPOINTS
        shipped = kind == "local_file" and spec.name in SHIPPED_WITH_CODE_CHECKPOINTS
        hf_local = kind == "local_file" and spec.name in HF_LOCAL_FILE_CHECKPOINTS
        if gdrive:
            reported_kind = "gdrive"
        elif hf_file:
            reported_kind = "hf_file"
        elif shipped:
            reported_kind = "shipped_with_code"
        elif hf_local:
            reported_kind = "hf_local_files"
        else:
            reported_kind = kind
        automatable = kind in ("hf_hub", "url") or gdrive or hf_file or shipped or hf_local
        out.append(
            {
                "name": spec.name,
                "ckpt_default": spec.ckpt_default,
                "ckpt_kind": reported_kind,
                "automatable": automatable,
                "notes": ""
                if automatable
                else {
                    "none": "No checkpoint registered for this method.",
                    "variable_dependent": (f"Checkpoint path depends on a variable resolved at run time "
                                           f"({spec.ckpt_default})."),
                    "local_file": (f"Local-file checkpoint with no known public source: {spec.ckpt_default}. "
                                   f"Obtain it from the method's own release and place it there, or override "
                                   f"{spec.ckpt_env}."),
                }.get(kind, ""),
            }
        )
    return out


def download_checkpoint(
    spec,
    *,
    shared_checkpoint_root: Path,
    shared_hf_home: Path,
    method_code_root: Path | None = None,
) -> dict:
    """Fetch one method's checkpoint if its ckpt_default is an HF hub id, a
    direct URL, a $METHOD_CODE_ROOT-relative path with a known Google Drive
    source (GDRIVE_CHECKPOINT_URLS) or single HF file (HF_FILE_CHECKPOINTS),
    or a local-file path that ships committed inside the method's own code
    clone (SHIPPED_WITH_CODE_CHECKPOINTS). Returns
    {"name", "status", "path"|"reason"}."""
    ckpt = spec.ckpt_default or ""
    kind = _checkpoint_kind(ckpt)

    if kind == "hf_hub":
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import HfHubHTTPError

        try:
            path = snapshot_download(repo_id=ckpt, cache_dir=str(shared_hf_home))
            for repo_id, patterns in HF_EXTRA_SNAPSHOTS.get(spec.name, ()):
                snapshot_download(repo_id=repo_id, allow_patterns=list(patterns) if patterns else None,
                                  cache_dir=str(shared_hf_home))
        except HfHubHTTPError as exc:
            # Covers both a gated repo (needs HF_TOKEN's account granted
            # access) and a repo that plain doesn't exist yet -- e.g. VLM3's
            # ckpt_default names the repo id its own inference.ipynb uses,
            # but that repo's README still says "Coming Soon!" as of this
            # writing, so it 404s rather than being gated. Report either
            # case with a real reason instead of crashing the whole
            # download_all_supported_models() batch on one method.
            return {
                "name": spec.name,
                "status": "skipped",
                "kind": kind,
                "reason": f"{ckpt} is gated, not yet released, or otherwise inaccessible on the HF hub "
                f"({exc}).",
            }
        return {"name": spec.name, "status": "ok", "kind": kind, "path": path}

    if kind in ("variable_dependent", "local_file") and spec.name in GDRIVE_CHECKPOINT_URLS:
        if kind == "variable_dependent":
            if method_code_root is None:
                return {
                    "name": spec.name,
                    "status": "skipped",
                    "kind": "gdrive",
                    "reason": f"{spec.name}'s checkpoint resolves under $METHOD_CODE_ROOT ({ckpt}) but no "
                    "method_code_root was passed to download_checkpoint()/download_all_supported_models().",
                }
            dest = Path(ckpt.replace("$METHOD_CODE_ROOT", str(method_code_root)))
        else:
            # A plain local_file ckpt_default (e.g. AMB3R's, canonicalized
            # under checkpoint_root/local/ like every other local-file
            # checkpoint) -- resolve the destination from the actual
            # shared_checkpoint_root argument, not the hardcoded absolute
            # string, so an override here is honored.
            dest = shared_checkpoint_root / "local" / spec.name.lower() / Path(ckpt).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            subprocess.run(
                # gdown 6.0.0 (installed here) dropped the --fuzzy flag but
                # auto-detects a full share URL without it -- verified via a
                # real download; the older gdown vggt_omega.sh calls with
                # --fuzzy is a separate, pre-existing script this touches.
                ["gdown", GDRIVE_CHECKPOINT_URLS[spec.name], "-O", str(dest)],
                check=True,
            )
        return {"name": spec.name, "status": "ok", "kind": "gdrive", "path": str(dest)}

    if kind == "variable_dependent" and spec.name in HF_FILE_CHECKPOINTS:
        if method_code_root is None:
            return {
                "name": spec.name,
                "status": "skipped",
                "kind": "hf_file",
                "reason": f"{spec.name}'s checkpoint resolves under $METHOD_CODE_ROOT ({ckpt}) but no "
                "method_code_root was passed to download_checkpoint()/download_all_supported_models().",
            }
        dest = Path(ckpt.replace("$METHOD_CODE_ROOT", str(method_code_root)))
        repo_id, filename = HF_FILE_CHECKPOINTS[spec.name]
        if not dest.exists():
            from huggingface_hub import hf_hub_download
            from huggingface_hub.utils import HfHubHTTPError

            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                fetched = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(dest.parent))
            except HfHubHTTPError as exc:
                return {
                    "name": spec.name,
                    "status": "skipped",
                    "kind": "hf_file",
                    "reason": f"{repo_id} is gated or inaccessible -- request/accept access at "
                    f"https://huggingface.co/{repo_id} with the account behind HF_TOKEN, then retry ({exc}).",
                }
            if Path(fetched) != dest:
                Path(fetched).rename(dest)
        return {"name": spec.name, "status": "ok", "kind": "hf_file", "path": str(dest)}

    if kind == "local_file" and spec.name in HF_LOCAL_FILE_CHECKPOINTS:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import HfHubHTTPError

        repo_id, subdir, files = HF_LOCAL_FILE_CHECKPOINTS[spec.name]
        dest_dir = shared_checkpoint_root / "local" / subdir
        for filename in files:
            if (dest_dir / filename).exists():
                continue
            try:
                hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(dest_dir))
            except HfHubHTTPError as exc:
                return {
                    "name": spec.name,
                    "status": "skipped",
                    "kind": "hf_local_files",
                    "reason": f"{repo_id}/{filename} is gated or inaccessible on the HF hub ({exc}).",
                }
        return {"name": spec.name, "status": "ok", "kind": "hf_local_files", "path": str(dest_dir / files[0])}

    if kind == "local_file" and spec.name in SHIPPED_WITH_CODE_CHECKPOINTS:
        # Canonicalized under shared_checkpoint_root/local/<method>/ like
        # every other local-file checkpoint -- computed from the actual
        # shared_checkpoint_root argument, not spec.ckpt_default's hardcoded
        # absolute string, so an override here is honored the same way the
        # "url" branch above honors it.
        relative = SHIPPED_WITH_CODE_CHECKPOINTS[spec.name]
        dest = shared_checkpoint_root / "local" / spec.name.lower() / Path(relative).name
        if not dest.exists():
            if method_code_root is None:
                return {
                    "name": spec.name,
                    "status": "skipped",
                    "kind": "shipped_with_code",
                    "reason": f"{spec.name}'s checkpoint ships inside its code clone but no method_code_root "
                    "was passed to download_checkpoint()/download_all_supported_models().",
                }
            leaf = _code_leaf_dir(spec.code_default)
            source = method_code_root / leaf / relative
            if not source.exists():
                return {
                    "name": spec.name,
                    "status": "skipped",
                    "kind": "shipped_with_code",
                    "reason": f"{spec.name}'s checkpoint should ship inside its code clone at {source}, "
                    "but that file isn't there -- run download_all_supported_method_code() first.",
                }
            import shutil

            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
        return {"name": spec.name, "status": "ok", "kind": "shipped_with_code", "path": str(dest)}

    if kind == "url":
        import requests

        dest_dir = shared_checkpoint_root / "local" / spec.name.lower()
        dest_dir.mkdir(parents=True, exist_ok=True)
        filename = ckpt.rstrip("/").split("/")[-1]
        dest = dest_dir / filename
        if not dest.exists():
            with requests.get(ckpt, stream=True, timeout=60) as response:
                response.raise_for_status()
                with open(dest, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        handle.write(chunk)
        return {
            "name": spec.name,
            "status": "ok",
            "kind": kind,
            "path": str(dest),
            "note": f"Downloaded but not wired into the registry -- set {spec.ckpt_env}={dest} to use it.",
        }

    return {
        "name": spec.name,
        "status": "skipped",
        "kind": kind,
        "reason": next(e["notes"] for e in list_supported_models() if e["name"] == spec.name),
    }


def download_all_supported_models(
    *,
    methods: list[str] | None = None,
    shared_checkpoint_root: Path | None = None,
    shared_hf_home: Path | None = None,
    method_code_root: Path | None = None,
    on_event: DownloadEventFn | None = None,
) -> list[dict]:
    from spatialarena.evaluator import DEFAULT_METHOD_CODE_ROOT, SHARED_CHECKPOINT_ROOT, SHARED_HF_HOME
    from spatialarena.methods import METHOD_SPECS, resolve_method

    shared_checkpoint_root = Path(shared_checkpoint_root) if shared_checkpoint_root else SHARED_CHECKPOINT_ROOT
    shared_hf_home = Path(shared_hf_home) if shared_hf_home else SHARED_HF_HOME
    method_code_root = Path(method_code_root) if method_code_root else DEFAULT_METHOD_CODE_ROOT

    specs = [resolve_method(name) for name in methods] if methods is not None else list(METHOD_SPECS)
    for spec in specs:
        _emit(on_event, "checkpoint", spec.name, "pending")
    results = []
    for spec in specs:
        _emit(on_event, "checkpoint", spec.name, "running")
        result = download_checkpoint(
            spec,
            shared_checkpoint_root=shared_checkpoint_root,
            shared_hf_home=shared_hf_home,
            method_code_root=method_code_root,
        )
        results.append(result)
        _emit(on_event, "checkpoint", spec.name, result["status"], detail=_result_detail(result))
    return results


# ---------------------------------------------------------------------------
# Method source code
# ---------------------------------------------------------------------------
# Every non-"native" method's code_default is "$METHOD_CODE_ROOT/<leaf>[/...]"
# -- METHOD_CODE_REPOS maps that leaf directory name to the git remote it was
# actually cloned from, read directly off this cluster's own vendored
# `3rd_party/methods/<leaf>/.git/config` (via `git remote get-url origin`)
# and confirmed reachable (`git ls-remote`) at the time this table was built.
# This is the same ground truth a human would `git clone` from, not a
# guessed URL -- a leaf with no entry here has no verified upstream and is
# reported "manual".

METHOD_CODE_REPOS: dict[str, str] = {
    "depth-anything-3": "https://github.com/bytedance-seed/depth-anything-3.git",
    # DA3 clones the official depth-anything-3 repo above rather than amb3r's
    # own vendored thirdparty/depth_anything_3 copy; amb3r is still needed
    # here for the AMB3R method itself.
    "amb3r": "https://github.com/HengyiWang/amb3r.git",
    "RobustVGGT": "https://github.com/cvlab-kaist/RobustVGGT.git",
    "Pi3": "https://github.com/yyfz/Pi3.git",
    "CoMe": "https://github.com/co-me-tokens/CoMe.git",
    "CUT3R": "https://github.com/CUT3R/CUT3R.git",
    "fast3r": "https://github.com/facebookresearch/fast3r.git",
    "vggt-omega": "https://github.com/facebookresearch/vggt-omega.git",
    "map-anything": "https://github.com/facebookresearch/map-anything.git",
    "OmniVGGT-official": "https://github.com/Livioni/OmniVGGT-official.git",
    "dvlt": "https://github.com/nv-tlabs/dvlt.git",
    "IGGT_official": "https://github.com/lifuguan/IGGT_official.git",
    "IGGT4D": "https://github.com/HorizonRobotics/IGGT4D.git",
    "pow3r": "https://github.com/naver/pow3r.git",
    "dust3r": "https://github.com/naver/dust3r.git",
    "mast3r": "https://github.com/naver/mast3r.git",
    "DepthVLM": "https://github.com/hanxunyu/DepthVLM.git",
    "VLM3": "https://github.com/facebookresearch/VLM3.git",
    "DepthLM_Official": "https://github.com/facebookresearch/DepthLM_Official.git",
    "SenseNova-Vision": "https://github.com/OpenSenseNova/SenseNova-Vision.git",
    "qwen3-vl": "https://github.com/qwenlm/qwen3-vl.git",
    "representations4d": "https://github.com/google-deepmind/representations4d.git",
    "VGGT-Long": "https://github.com/DengKaiCQ/VGGT-Long.git",
    "Pi-Long": "https://github.com/DengKaiCQ/Pi-Long.git",
    "VGGT-SLAM": "https://github.com/MIT-SPARK/VGGT-SLAM.git",
    # Streaming / long-context SLAM methods (scripts/inference/streaming_slam.py), pinned in METHOD_CODE_COMMITS.
    "LoGeR": "https://github.com/Junyi42/LoGeR.git",
    "ZipMap": "https://github.com/Haian-Jin/ZipMap.git",
    "lingbot-map": "https://github.com/robbyant/lingbot-map.git",
    "R3": "https://github.com/KevinXu02/R3.git",
    "Scal3R": "https://github.com/zju3dv/Scal3R.git",
    "ABot-Recon": "https://github.com/amap-cvlab/ABot-Recon.git",
    # Official single-image MonoDepth baselines (scripts/inference/single_image_depth.py) and DVD (VideoDepth),
    # pinned in METHOD_CODE_COMMITS.
    "MoGe": "https://github.com/microsoft/MoGe.git",
    "Marigold": "https://github.com/prs-eth/Marigold.git",
    "Lotus": "https://github.com/EnVision-Research/Lotus.git",
    "ml-depth-pro": "https://github.com/apple/ml-depth-pro.git",
    "UniDepth": "https://github.com/lpiccinelli-eth/UniDepth.git",
    "InfiniDepth": "https://github.com/zju3dv/InfiniDepth.git",
    "DVD": "https://github.com/EnVision-Research/DVD.git",
    # VideoDepth diffusion baselines (scripts/inference/video_diffusion_depth.py), pinned in METHOD_CODE_COMMITS.
    "DepthCrafter": "https://github.com/Tencent/DepthCrafter.git",
    "ChronoDepth": "https://github.com/jiahao-shao1/ChronoDepth.git",
}

# Commit each clone is checked out at, for repos whose evaluated version is pinned (the commit the benchmark
# results were produced with; commit dates in the comments).
METHOD_CODE_COMMITS: dict[str, str] = {
    "LoGeR": "7685b7a41949e62ed8a69309642465aee4cb89ba",  # 2026-04-27
    "ZipMap": "e0f1f400852e8d4770ff1dbb6fd017a6048afbfd",  # 2026-06-10
    "lingbot-map": "8fdf984a7f9caf391622ea5843a8410900d27ef2",  # 2026-10-06
    "R3": "e345f1112fbc9c451d44c9c5aa22d9bfec2a954d",  # 2026-06-19
    "Scal3R": "26a2fa73898440fefe056a80dfb08763e5b81314",  # 2026-09-15
    # The commits the MonoDepth baselines / DVD were integrated and first run with (cloned 2026-10-23).
    "MoGe": "74fbce054ebed49800de42d0ad0e83495065719a",
    "Marigold": "2bfbdeae5d10a50b71f1ba20c865d46e480f6010",
    "Lotus": "b737e12238d2a41fc582ee8c6c6b1104d318d799",
    "ml-depth-pro": "9e65e4dbe9568d23c546fcec53302b10445e109e",
    "UniDepth": "8d8cfe4c7ee15297099983607febf0d4f32eb3d6",
    "InfiniDepth": "669a9b8354fb0f3f2f73814207ebdd519dc5c6d3",
    "DVD": "62f52d16faa2cac10e31eb7815ece6e61d5449ec",
    # The upstream heads when DepthCrafter and ChronoDepth were integrated (2026-10-30).
    "DepthCrafter": "e5186120a71ecb56b69f9e89654198b4eadfaf60",
    "ChronoDepth": "2580e891eb33a5c3a95eab27b745f64ac28b3514",
    "ABot-Recon": "81a773722782bcc41655ea06792fa8a74e90b507",  # 2026-09-25
}


def _code_leaf_dir(code_default: str) -> str:
    """The first path component under $METHOD_CODE_ROOT -- e.g.
    "$METHOD_CODE_ROOT/depth-anything-3/src" -> "depth-anything-3" (DA3-L/G/
    NGL share the official ByteDance-Seed clone, three registry entries),
    and "$METHOD_CODE_ROOT/RobustVGGT" -> "RobustVGGT" (shared by VGGT,
    FASTVGGT, ROBUSTVGGT -- one clone, three registry entries)."""
    rest = code_default.split("$METHOD_CODE_ROOT/", 1)[-1]
    return rest.split("/", 1)[0]


def list_supported_method_code() -> list[dict]:
    """Per-method vendored-code download status, derived live from the
    method registry. A method is "automated" only if its code_default's
    leaf directory has a verified entry in METHOD_CODE_REPOS."""
    from spatialarena.methods import METHOD_SPECS

    out = []
    for spec in METHOD_SPECS:
        leaf = _code_leaf_dir(spec.code_default)
        repo_url = METHOD_CODE_REPOS.get(leaf)
        out.append(
            {
                "name": spec.name,
                "code_default": spec.code_default,
                "leaf_dir": leaf,
                "repo_url": repo_url,
                "status": "automated" if repo_url else "manual",
                "notes": ""
                if repo_url
                else (f"No verified upstream repo for {leaf!r} -- clone it yourself per the method's own "
                      f"paper/GitHub listing and place it at {spec.code_default}."),
            }
        )
    return out


def download_method_code(
    spec,
    *,
    method_code_root: Path,
    skip_existing: bool = True,
    log_dir: Path | None = None,
) -> dict:
    """Clone one method's vendored code if its leaf dir has a verified repo
    in METHOD_CODE_REPOS. Methods sharing a leaf (e.g. VGGT/FASTVGGT/
    ROBUSTVGGT -> RobustVGGT) naturally short-circuit to "already_present"
    on the second and third call once the first has cloned it."""
    leaf = _code_leaf_dir(spec.code_default)
    repo_url = METHOD_CODE_REPOS.get(leaf)
    if not repo_url:
        return {
            "name": spec.name,
            "status": "skipped",
            "reason": next(e["notes"] for e in list_supported_method_code() if e["name"] == spec.name),
        }

    dest = Path(method_code_root) / leaf
    if skip_existing and dest.exists():
        return {"name": spec.name, "status": "already_present", "path": str(dest)}

    log_dir = Path(log_dir).expanduser() if log_dir else Path(method_code_root) / "_download_logs"
    result = _run_step(
        f"clone_{leaf}",
        ("git", "clone", repo_url, str(dest)),
        env=dict(os.environ),
        log_dir=log_dir,
    )
    if not result.ok:
        return {"name": spec.name, "status": "failed", "stderr_tail": result.stderr_tail}
    commit = METHOD_CODE_COMMITS.get(leaf)
    if commit:
        result = _run_step(
            f"checkout_{leaf}",
            ("git", "-C", str(dest), "checkout", "--quiet", commit),
            env=dict(os.environ),
            log_dir=log_dir,
        )
        if not result.ok:
            return {"name": spec.name, "status": "failed", "stderr_tail": result.stderr_tail}
    return {"name": spec.name, "status": "ok", "path": str(dest)}


def download_all_supported_method_code(
    *,
    methods: list[str] | None = None,
    method_code_root: Path | None = None,
    skip_existing: bool = True,
    on_event: DownloadEventFn | None = None,
) -> list[dict]:
    from spatialarena.evaluator import DEFAULT_METHOD_CODE_ROOT
    from spatialarena.methods import METHOD_SPECS, resolve_method

    method_code_root = Path(method_code_root) if method_code_root else DEFAULT_METHOD_CODE_ROOT
    specs = [resolve_method(name) for name in methods] if methods is not None else list(METHOD_SPECS)
    for spec in specs:
        _emit(on_event, "code", spec.name, "pending")
    results = []
    for spec in specs:
        _emit(on_event, "code", spec.name, "running")
        result = download_method_code(spec, method_code_root=method_code_root, skip_existing=skip_existing)
        results.append(result)
        _emit(on_event, "code", spec.name, result["status"], detail=_result_detail(result))
    return results
