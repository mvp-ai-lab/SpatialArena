# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Pack this repo's canonical benchmark datasets and upload them to a
Hugging Face dataset repo, so a fresh consumer of this repo can fetch
already-preprocessed data instead of re-running every scripts/download/
+ scripts/preprocess/ step themselves.

Every packaged dataset folder gets its own LICENSE_NOTICE.md, sourced from
spatialarena/dataset_licenses.py's SOURCE_LICENSES registry -- that
registry records each dataset's *original* source license/citation as
verified against the source's own site, not guessed at. Datasets whose
provenance/license is "unconfirmed" (see that module's docstring) are
excluded from the default packaged set; pass --include-unconfirmed to
override that after you've personally verified the source. Scenes whose
source forbids redistribution (WITHHELD_SCENES, e.g. Marigold's ScanNet
subset) are always left out, and the notice says how to rebuild them.

This script never uploads anything by itself: --dry-run (the default)
only stages the folder layout + license notices on local disk and reports
what *would* be uploaded. Real network transfer requires both --stage
(actually copies the (large) dataset files, not just the notices) and
--upload (talks to the Hugging Face Hub) to be passed explicitly.

Example:

    python scripts/release/pack_and_upload_benchmark_datasets.py \\
        --benchmark_root /path/to/benchmark_root \\
        --staging_root /path/to/staging \\
        --tasks MVRecon MonoDepth SfM VideoDepth SLAM \\
        --repo_id my-org/my-benchmark \\
        --stage --upload
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatialarena.dataset_licenses import (  # noqa: E402
    SOURCE_LICENSES,
    SourceLicense,
    is_redistributable,
    is_safe_for_default_public_release,
    sources_for,
    withheld_scenes,
)
from spatialarena.eval.dataset_registry import (  # noqa: E402
    DEFAULT_BENCHMARK_ROOT,
    describe_benchmark_scenes,
    select_dataset_specs,
)

DEFAULT_TASKS = ("MVRecon", "MonoDepth", "SfM", "VideoDepth", "SLAM")


def _license_notice_text(task: str, dataset: str, sources: tuple[SourceLicense, ...]) -> str:
    if not sources:
        return (
            f"# {task}/{dataset}\n\n"
            "No license registry entry found for this dataset -- do not redistribute "
            "until one is added to spatialarena/dataset_licenses.py.\n"
        )
    lines = [f"# {task}/{dataset} -- original source license(s)\n"]
    for source in sources:
        lines.append(f"## {source.name}\n")
        lines.append(f"- Homepage: {source.homepage or '(unknown)'}")
        lines.append(f"- Status: `{source.status}`")
        lines.append(f"- License: {source.license_name}")
        if source.license_url:
            lines.append(f"- License URL: {source.license_url}")
        if source.citation:
            lines.append(f"- Citation: {source.citation}")
        if source.notes:
            lines.append(f"- Notes: {source.notes}")
        lines.append("")
    withheld = withheld_scenes(task, dataset)
    if withheld:
        lines.append("## Scenes not included\n")
        for scene, (key, restore) in sorted(withheld.items()):
            source = SOURCE_LICENSES[key]
            lines.append(
                f"- `{scene}` ({source.name}, `{source.status}`: {source.license_name}) is not redistributed here. "
                f"{restore}"
            )
        lines.append("")
    lines.append(
        "This folder is a preprocessed/re-laid-out copy of the above source(s), "
        "produced by this repo's scripts/download/ and scripts/preprocess/ tooling. "
        "All rights to the original data remain with its original creator(s); this "
        "packaging does not grant any additional rights beyond what the sources above "
        "state.\n"
    )
    return "\n".join(lines)


def plan_dataset_package(
    *,
    benchmark_root: Path,
    tasks: tuple[str, ...],
    include_unconfirmed: bool,
    datasets: list[tuple[str, str]] | None = None,
) -> list[dict]:
    """Return one plan entry per (task, dataset) this run would package,
    each carrying its presence check (from describe_benchmark_scenes) and
    license status -- never guesses at whether data actually exists on
    disk, always checks.

    Several registered (task, dataset) pairs resolve to the *same* physical
    directory on disk -- either via DatasetSpec.legacy_relative_paths (e.g.
    SfM/hiroom, which falls back to MVRecon/hiroom when not materialized)
    or via a plain filesystem symlink this repo's own layout already uses
    for space-saving (e.g. SfM/eth3d -> ../MVRecon/eth3d). Copying both
    sides would silently double the real bytes transferred/uploaded, so the
    first (task, dataset) to claim a given resolved path is staged for
    real; later ones referencing the same path are recorded as
    `duplicate_of` and get a pointer instead of a second copy.
    """
    specs = select_dataset_specs(tasks=tasks)
    if datasets is not None:
        wanted = {(t, d) for t, d in datasets}
        specs = [s for s in specs if (s.task, s.dataset) in wanted]

    plan = []
    seen_resolved_paths: dict[Path, str] = {}
    for spec in specs:
        sources = sources_for(spec.task, spec.dataset)
        safe = is_safe_for_default_public_release(spec.task, spec.dataset)
        withheld = withheld_scenes(spec.task, spec.dataset)
        included = (safe or include_unconfirmed) and is_redistributable(spec.task, spec.dataset)
        existing = describe_benchmark_scenes(
            benchmark_root=benchmark_root, tasks=[spec.task], datasets=[spec.dataset]
        )
        present = bool(existing) and existing[0]["exists"] and (
            not existing[0]["metadata_required"] or existing[0]["metadata_exists"]
        )

        resolved_path = spec.data_path(benchmark_root).resolve() if present else None
        duplicate_of = seen_resolved_paths.get(resolved_path) if resolved_path else None
        if resolved_path is not None and duplicate_of is None:
            seen_resolved_paths[resolved_path] = f"{spec.task}/{spec.dataset}"

        plan.append(
            {
                "task": spec.task,
                "dataset": spec.dataset,
                "relative_path": str(spec.relative_path),
                "resolved_path": str(resolved_path) if resolved_path else None,
                "duplicate_of": duplicate_of,
                "sources": [s.name for s in sources],
                "status": "safe" if safe else "unconfirmed" if sources else "no_license_entry",
                "included": included and present,
                "present_on_disk": present,
                "scene_count": max(0, existing[0]["scene_count"] - len(withheld)) if existing else 0,
                "withheld_scenes": sorted(withheld),
            }
        )
    return plan


def _link_or_copy(src, dst) -> None:
    try:
        os.link(src, dst)
    except OSError:  # e.g. a different filesystem
        shutil.copy2(src, dst)


def _withheld_metadata(src: Path, withheld: set[str]) -> str:
    """``metadata.json`` text without the withheld scenes, with the dataset totals recomputed."""
    metadata = json.loads(src.read_text(encoding="utf-8"))
    scenes = [scene for scene in metadata.get("scenes", []) if scene.get("scene_name") not in withheld]
    metadata["scenes"] = scenes
    if "total_images" in metadata:
        metadata["total_images"] = sum(int(scene.get("total_images", 0)) for scene in scenes)
    if "average_images_per_scene" in metadata:
        metadata["average_images_per_scene"] = metadata.get("total_images", 0) / len(scenes) if scenes else 0.0
    return json.dumps(metadata, indent=2) + "\n"


def _stage_top_level_file(item: Path, target: Path, withheld: set[str], copy_fn) -> None:
    if withheld and item.name == "metadata.json":
        target.write_text(_withheld_metadata(item, withheld), encoding="utf-8")
    else:
        copy_fn(item, target)


def _archive_dataset(
    resolved_path: Path,
    dst: Path,
    archive_jobs: list | None = None,
    workers: int = 1,
    withheld: set[str] | None = None,
) -> None:
    """Stage a dataset as one uncompressed ``<entry>.tar`` per top-level directory (usually one scene) plus its
    top-level files as-is. ``tar`` keeps symlinks inside the dataset, so nothing is duplicated and extracting every
    archive in place restores the exact tree; a scene with a link leaving the dataset is archived dereferenced.
    Each archive is written to a temporary name and renamed when complete, so an interrupted run resumes by skipping
    finished archives.

    Reading many small files from network storage is latency-bound, so archives should be built concurrently.
    With ``archive_jobs`` the pending builds are appended for the caller to run in one pool across datasets
    (so one huge folder does not serialize the rest); otherwise they run here with ``workers`` threads.
    ``withheld`` scene folders are left out and dropped from ``metadata.json``."""
    withheld = withheld or set()
    items = sorted(item for item in resolved_path.iterdir() if item.name not in withheld)
    dirs = [item for item in items if item.is_dir() and not item.is_symlink()]
    for item in items:
        if item in dirs or (dst / item.name).exists():
            continue
        if item.is_symlink():
            (dst / item.name).symlink_to(os.readlink(item))
        else:
            _stage_top_level_file(item, dst / item.name, withheld, _link_or_copy)

    jobs = []
    for item in dirs:
        archive = dst / f"{item.name}.tar"
        if archive.exists():
            continue
        jobs.append(functools.partial(_build_archive, resolved_path, item.name, dst))
    if archive_jobs is not None:
        archive_jobs.extend(jobs)
    else:
        run_archive_jobs(jobs, workers)


def _has_external_symlink(resolved_path: Path, name: str) -> bool:
    """Whether any symlink under ``resolved_path/name`` points outside ``resolved_path`` (e.g. DyCheck's frames link
    into ``VideoDepth_source/dycheck_raw``); such a link would dangle once the archive is extracted elsewhere."""
    for root, dirnames, filenames in os.walk(resolved_path / name):
        for entry in dirnames + filenames:
            path = Path(root) / entry
            if path.is_symlink() and not path.resolve().is_relative_to(resolved_path.resolve()):
                return True
    return False


def _build_archive(resolved_path: Path, name: str, dst: Path) -> None:
    partial = dst / f".{name}.tar.partial"
    # Store the target files instead of the links when any link leaves the dataset; in-tree links stay links.
    dereference = ["--dereference"] if _has_external_symlink(resolved_path, name) else []
    subprocess.run(["tar", "-cf", str(partial), *dereference, "-C", str(resolved_path), name], check=True)
    partial.rename(dst / f"{name}.tar")


def run_archive_jobs(jobs: list, workers: int) -> None:
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for future in [pool.submit(job) for job in jobs]:
            future.result()  # re-raise the first failure


def stage_dataset(
    *,
    benchmark_root: Path,
    staging_root: Path,
    task: str,
    dataset: str,
    resolved_path: Path,
    duplicate_of: str | None,
    copy_data: bool,
    hardlink: bool = False,
    archive_scenes: bool = False,
    archive_workers: int = 1,
    archive_jobs: list | None = None,
) -> Path:
    dst = staging_root / task / dataset
    dst.mkdir(parents=True, exist_ok=True)

    sources = sources_for(task, dataset)
    (dst / "LICENSE_NOTICE.md").write_text(_license_notice_text(task, dataset, sources), encoding="utf-8")

    if duplicate_of is not None:
        (dst / "SEE_INSTEAD.md").write_text(
            f"This dataset resolves to the exact same on-disk data as `{duplicate_of}` "
            f"(same source: {resolved_path}). To avoid uploading duplicate bytes, the actual "
            f"files are only packaged under `{duplicate_of}/` -- copy/symlink from there if "
            "you need this data under this path too.\n",
            encoding="utf-8",
        )
        return dst

    withheld = set(withheld_scenes(task, dataset))
    if copy_data and archive_scenes:
        _archive_dataset(resolved_path, dst, archive_jobs, archive_workers, withheld)
    elif copy_data:
        # Symlinks inside a dataset are materialized (the Hub stores no symlinks). With hardlink=True files are
        # hard-linked instead of copied, so staging costs no extra space when staging_root shares the filesystem.
        copy_fn = _link_or_copy if hardlink else shutil.copy2
        for item in resolved_path.iterdir():
            target = dst / item.name
            if target.exists() or item.name in withheld:
                continue
            if item.is_dir():
                shutil.copytree(item, target, copy_function=copy_fn)
            else:
                _stage_top_level_file(item, target, withheld, copy_fn)

    return dst


def _write_dataset_card(staging_root: Path, plan: list[dict]) -> Path:
    included = [p for p in plan if p["included"]]
    excluded = [p for p in plan if not p["included"]]

    lines = [
        "# Benchmark datasets\n",
        "Preprocessed, canonically-laid-out copies of the datasets used by this repo's",
        "`BenchmarkEvaluator` (see `docs/benchmark/evaluator_quickstart.md`), packaged for",
        "direct download instead of re-running every `scripts/download/` +",
        "`scripts/preprocess/` step yourself.\n",
        "**Every dataset folder below is redistributed under its own original source's",
        "license -- see that folder's own `LICENSE_NOTICE.md`. This repository's own",
        "license does not override those terms.**\n",
        "## Included datasets\n",
        "| Task | Dataset | Scenes | License status | Source(s) | Data |",
        "|---|---|---|---|---|---|",
    ]
    for p in included:
        data_note = f"see `{p['duplicate_of']}/`" if p["duplicate_of"] else "packaged here"
        if p.get("withheld_scenes"):
            data_note += f" (without {', '.join(p['withheld_scenes'])}; see `LICENSE_NOTICE.md`)"
        lines.append(
            f"| {p['task']} | {p['dataset']} | {p['scene_count']} | `{p['status']}` | "
            f"{', '.join(p['sources']) or '(none)'} | {data_note} |"
        )
    if excluded:
        lines.append("\n## Excluded this run\n")
        lines.append("| Task | Dataset | Reason |")
        lines.append("|---|---|---|")
        for p in excluded:
            reason = "not present on disk" if not p["present_on_disk"] else f"status={p['status']}"
            if p["present_on_disk"] and not is_redistributable(p["task"], p["dataset"]):
                reason = "source forbids redistribution"
            lines.append(f"| {p['task']} | {p['dataset']} | {reason} |")

    if any(staging_root.glob("*/*/*.tar")):
        lines[5:5] = [
            "**Layout:** each dataset scene is stored as one uncompressed `<Task>/<dataset>/<scene>.tar` (the raw",
            "tree is ~900k files). After downloading, extract everything in place with the bundled script:\n",
            "```bash",
            "hf download Bluefish2333/benchmark --repo-type dataset --local-dir /path/to/benchmark_root",
            "python /path/to/benchmark_root/extract_benchmark_archives.py /path/to/benchmark_root --delete",
            "```\n",
        ]
    if (staging_root / "methods" / "METHODS.md").is_file():
        lines.append("\n## Methods: official code and checkpoints\n")
        lines.append("Third-party code and weights are not redistributed here; see `methods/METHODS.md` (and")
        lines.append(
            "`methods/methods_manifest.json`) for each method's upstream repository, pinned commit, license and"
        )
        lines.append("checkpoint source, produced by `scripts/release/build_method_manifests.py`.")

    text = "\n".join(lines) + "\n"
    path = staging_root / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def datasets_on_hub(repo_id: str) -> set[tuple[str, str]]:
    """``(task, dataset)`` folders already in the Hub dataset repo (empty if the repo does not exist yet)."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        files = HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        return set()
    return {(parts[0], parts[1]) for parts in (f.split("/") for f in files) if len(parts) >= 3}


def card_plan(full_plan: list[dict], staged: set[tuple[str, str]], on_hub: set[tuple[str, str]]) -> list[dict]:
    """The README card describes the whole repo, not just this run: a dataset is listed as included when this run
    staged it or an earlier run already uploaded it (a partial ``--datasets`` run must not drop the others)."""
    return [{**p, "included": (p["task"], p["dataset"]) in staged | on_hub} for p in full_plan]


def upload_to_hub(staging_root: Path, repo_id: str, *, private: bool = False, num_workers: int | None = None) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    # huggingface_hub 2.x's upload_folder (which replaced upload_large_folder) streams the files through hf_xet,
    # commits a large folder in batches, and resumes when rerun. Its concurrency is hf_xet's, adaptive by default:
    # cap it at num_workers. upload_large_folder sized its pool from os.cpu_count() (the whole node, not the Slurm
    # allocation); on a 208-core node that meant ~100 concurrent uploads, which tripped Hub rate limits and Xet
    # shard I/O errors. An explicit HF_XET_* setting in the environment wins.
    if num_workers:
        for name in ("HF_XET_MAX_CONCURRENT_FILE_INGESTION", "HF_XET_CLIENT_AC_MAX_UPLOAD_CONCURRENCY"):
            os.environ.setdefault(name, str(num_workers))
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(staging_root),
        commit_message="Upload the benchmark release",
        ignore_patterns=["*.log", "*.err", "*.partial"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark_root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--staging_root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Restrict to specific 'Task/dataset' pairs (e.g. SfM/imc MVRecon/7scenes). "
        "Default: everything in --tasks.",
    )
    parser.add_argument(
        "--include_unconfirmed",
        action="store_true",
        help="Also package datasets with unconfirmed provenance/license (see spatialarena/dataset_licenses.py). "
        "Only pass this after you've personally verified the source yourself.",
    )
    parser.add_argument(
        "--stage",
        action="store_true",
        help="Actually copy dataset files into --staging_root (can be very large). Without this, only "
        "LICENSE_NOTICE.md/README.md are written so you can review the plan first.",
    )
    parser.add_argument(
        "--hardlink",
        action="store_true",
        help="With --stage, hard-link data files instead of copying (no extra space on the same filesystem).",
    )
    parser.add_argument(
        "--archive_scenes",
        action="store_true",
        help="With --stage, store each top-level dataset folder as one .tar (keeps symlinks; far fewer Hub files).",
    )
    parser.add_argument(
        "--archive_workers", type=int, default=8, help="Archives built concurrently (one pool across all datasets)."
    )
    parser.add_argument("--upload", action="store_true", help="Upload --staging_root to --repo_id via huggingface_hub.")
    parser.add_argument("--repo_id", default=None, help="e.g. my-org/my-benchmark. Required with --upload.")
    parser.add_argument("--upload_workers", type=int, default=8, help="Concurrent upload workers (default 8).")
    parser.add_argument("--private", action="store_true", help="Create the HF repo as private (default: public).")
    args = parser.parse_args()

    if args.upload and not args.repo_id:
        parser.error("--upload requires --repo_id")

    dataset_filter = None
    if args.datasets:
        dataset_filter = []
        for item in args.datasets:
            task, _, dataset = item.partition("/")
            if not dataset:
                parser.error(f"--datasets entries must look like 'Task/dataset', got {item!r}")
            dataset_filter.append((task, dataset))

    plan = plan_dataset_package(
        benchmark_root=args.benchmark_root,
        tasks=tuple(args.tasks),
        include_unconfirmed=args.include_unconfirmed,
        datasets=dataset_filter,
    )

    print(json.dumps(plan, indent=2))

    if not args.stage and not args.upload:
        print("\n(dry run -- pass --stage to write license notices + copy data, --upload to publish)")
        return

    args.staging_root.mkdir(parents=True, exist_ok=True)
    archive_jobs: list = []
    for entry in plan:
        if not entry["included"]:
            continue
        stage_dataset(
            benchmark_root=args.benchmark_root,
            staging_root=args.staging_root,
            task=entry["task"],
            dataset=entry["dataset"],
            resolved_path=Path(entry["resolved_path"]),
            duplicate_of=entry["duplicate_of"],
            copy_data=args.stage,
            hardlink=args.hardlink,
            archive_scenes=args.archive_scenes,
            archive_workers=args.archive_workers,
            archive_jobs=archive_jobs,
        )
    if archive_jobs:  # one pool across all datasets
        print(f"Building {len(archive_jobs)} archives with {args.archive_workers} workers", flush=True)
        run_archive_jobs(archive_jobs, args.archive_workers)
    if args.archive_scenes:  # once, after the loop
        shutil.copy2(Path(__file__).with_name("extract_benchmark_archives.py"), args.staging_root)
    card_entries = plan
    if dataset_filter is not None or args.repo_id:
        full_plan = plan_dataset_package(
            benchmark_root=args.benchmark_root,
            tasks=tuple(dict.fromkeys((*DEFAULT_TASKS, *args.tasks))),
            include_unconfirmed=True,
        )
        staged = {(p["task"], p["dataset"]) for p in plan if p["included"]}
        card_entries = card_plan(full_plan, staged, datasets_on_hub(args.repo_id) if args.repo_id else set())
    card_path = _write_dataset_card(args.staging_root, card_entries)
    print(f"Wrote dataset card: {card_path}")

    if args.upload:
        upload_to_hub(args.staging_root, args.repo_id, private=args.private, num_workers=args.upload_workers)
        print(f"Uploaded to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
