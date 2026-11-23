# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Packs the canonical benchmark datasets and uploads them to Hugging Face.

The upload goes to a Hugging Face dataset repo, so a fresh consumer of this
repo can fetch already-preprocessed data instead of re-running every
scripts/download/ + scripts/preprocess/ step themselves.

Every packaged dataset folder gets its own LICENSE_NOTICE.md, sourced from
spatialarena/dataset_licenses.py's SOURCE_LICENSES registry; that registry
records each dataset's *original* source license/citation as verified against
the source's own site, not guessed at. Datasets whose provenance/license is
"unconfirmed" (see that module's docstring) are excluded from the default
packaged set; pass --include-unconfirmed to override that after you've
personally verified the source. Scenes whose source forbids redistribution
(WITHHELD_SCENES, e.g. Marigold's ScanNet subset) are always left out, and the
notice says how to rebuild them.

This script never uploads anything by itself: --dry-run (the default) only
stages the folder layout + license notices on local disk and reports what
*would* be uploaded. Real network transfer requires both --stage (actually
copies the (large) dataset files, not just the notices) and --upload (talks to
the Hugging Face Hub) to be passed explicitly.

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
from concurrent import futures
import functools
import json
import os
import pathlib
import shutil
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from spatialarena import dataset_licenses  # noqa: E402
from spatialarena.eval import dataset_registry  # noqa: E402
# Tests patch these two names on this module, so they are imported by name and
# called through this module's own bindings.
from spatialarena.eval.dataset_registry import (  # noqa: E402
    describe_benchmark_scenes)
from spatialarena.eval.dataset_registry import (  # noqa: E402
    select_dataset_specs)

# pylint: enable=wrong-import-position

DEFAULT_TASKS = ("MVRecon", "MonoDepth", "SfM", "VideoDepth", "SLAM")


def _license_notice_text(
        task: str, dataset: str, sources: tuple[dataset_licenses.SourceLicense,
                                                ...]) -> str:
    """Returns the LICENSE_NOTICE.md text of one packaged dataset.

    Args:
        task: The task name.
        dataset: The dataset name.
        sources: The dataset's original sources.

    Returns:
        The notice: each source's license and citation, the withheld scenes
        and how to rebuild them, and the redistribution disclaimer.
    """
    if not sources:
        return (f"# {task}/{dataset}\n\n"
                "No license registry entry found for this dataset -- do not "
                "redistribute until one is added to "
                "spatialarena/dataset_licenses.py.\n")
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
    withheld = dataset_licenses.withheld_scenes(task, dataset)
    if withheld:
        lines.append("## Scenes not included\n")
        for scene, (key, restore) in sorted(withheld.items()):
            source = dataset_licenses.SOURCE_LICENSES[key]
            lines.append(f"- `{scene}` ({source.name}, `{source.status}`: "
                         f"{source.license_name}) is not redistributed here. "
                         f"{restore}")
        lines.append("")
    lines.append(
        "This folder is a preprocessed/re-laid-out copy of the above "
        "source(s), produced by this repo's scripts/download/ and "
        "scripts/preprocess/ tooling. All rights to the original data remain "
        "with its original creator(s); this packaging does not grant any "
        "additional rights beyond what the sources above state.\n")
    return "\n".join(lines)


def plan_dataset_package(
    *,
    benchmark_root: pathlib.Path,
    tasks: tuple[str, ...],
    include_unconfirmed: bool,
    datasets: list[tuple[str, str]] | None = None,
) -> list[dict]:
    """Plans which (task, dataset) folders this run would package.

    Each entry carries its presence check (from describe_benchmark_scenes)
    and license status; it never guesses at whether data actually exists on
    disk, it always checks.

    Several registered (task, dataset) pairs resolve to the *same* physical
    directory on disk, either via DatasetSpec.legacy_relative_paths (e.g.
    SfM/hiroom, which falls back to MVRecon/hiroom when not materialized) or
    via a plain filesystem symlink this repo's own layout already uses for
    space-saving (e.g. SfM/eth3d -> ../MVRecon/eth3d). Copying both sides
    would silently double the real bytes transferred/uploaded, so the first
    (task, dataset) to claim a given resolved path is staged for real; later
    ones referencing the same path are recorded as `duplicate_of` and get a
    pointer instead of a second copy.

    Args:
        benchmark_root: The benchmark root.
        tasks: The tasks to package.
        include_unconfirmed: Whether to include datasets whose license is
            unconfirmed.
        datasets: If given, only these (task, dataset) pairs.

    Returns:
        One plan entry per (task, dataset).
    """
    specs = select_dataset_specs(tasks=tasks)
    if datasets is not None:
        wanted = {(task, dataset) for task, dataset in datasets}
        specs = [spec for spec in specs if (spec.task, spec.dataset) in wanted]

    plan = []
    seen_resolved_paths: dict[pathlib.Path, str] = {}
    for spec in specs:
        sources = dataset_licenses.sources_for(spec.task, spec.dataset)
        safe = dataset_licenses.is_safe_for_default_public_release(
            spec.task, spec.dataset)
        withheld = dataset_licenses.withheld_scenes(spec.task, spec.dataset)
        included = ((safe or include_unconfirmed) and
                    dataset_licenses.is_redistributable(spec.task,
                                                        spec.dataset))
        existing = describe_benchmark_scenes(benchmark_root=benchmark_root,
                                             tasks=[spec.task],
                                             datasets=[spec.dataset])
        present = bool(existing) and existing[0]["exists"] and (
            not existing[0]["metadata_required"] or
            existing[0]["metadata_exists"])

        if present:
            resolved_path = spec.data_path(benchmark_root).resolve()
        else:
            resolved_path = None
        if resolved_path:
            duplicate_of = seen_resolved_paths.get(resolved_path)
        else:
            duplicate_of = None
        if resolved_path is not None and duplicate_of is None:
            seen_resolved_paths[resolved_path] = f"{spec.task}/{spec.dataset}"

        if safe:
            status = "safe"
        elif sources:
            status = "unconfirmed"
        else:
            status = "no_license_entry"
        if existing:
            scene_count = max(0, existing[0]["scene_count"] - len(withheld))
        else:
            scene_count = 0
        if resolved_path:
            resolved_text = str(resolved_path)
        else:
            resolved_text = None
        plan.append({
            "task": spec.task,
            "dataset": spec.dataset,
            "relative_path": str(spec.relative_path),
            "resolved_path": resolved_text,
            "duplicate_of": duplicate_of,
            "sources": [source.name for source in sources],
            "status": status,
            "included": included and present,
            "present_on_disk": present,
            "scene_count": scene_count,
            "withheld_scenes": sorted(withheld),
        })
    return plan


def _link_or_copy(src, dst) -> None:
    """Hard-links `src` to `dst`, or copies it if linking fails."""
    try:
        os.link(src, dst)
    except OSError:
        # E.g. a different filesystem.
        shutil.copy2(src, dst)


def _withheld_metadata(src: pathlib.Path, withheld: set[str]) -> str:
    """Returns `metadata.json` text without the withheld scenes.

    The dataset totals are recomputed.

    Args:
        src: The dataset's metadata.json.
        withheld: The withheld scene names.

    Returns:
        The new metadata.json text.
    """
    metadata = json.loads(src.read_text(encoding="utf-8"))
    scenes = [
        scene for scene in metadata.get("scenes", [])
        if scene.get("scene_name") not in withheld
    ]
    metadata["scenes"] = scenes
    if "total_images" in metadata:
        metadata["total_images"] = sum(
            int(scene.get("total_images", 0)) for scene in scenes)
    if "average_images_per_scene" in metadata:
        if scenes:
            metadata["average_images_per_scene"] = metadata.get(
                "total_images", 0) / len(scenes)
        else:
            metadata["average_images_per_scene"] = 0.0
    return json.dumps(metadata, indent=2) + "\n"


def _stage_top_level_file(item: pathlib.Path, target: pathlib.Path,
                          withheld: set[str], copy_fn) -> None:
    """Stages a dataset's top-level file (metadata.json without withheld)."""
    if withheld and item.name == "metadata.json":
        target.write_text(_withheld_metadata(item, withheld), encoding="utf-8")
    else:
        copy_fn(item, target)


def _archive_dataset(
    resolved_path: pathlib.Path,
    dst: pathlib.Path,
    archive_jobs: list | None = None,
    workers: int = 1,
    withheld: set[str] | None = None,
) -> None:
    """Stages a dataset as one uncompressed tar per top-level directory.

    Each top-level directory (usually one scene) becomes `<entry>.tar`; the
    top-level files are staged as-is. `tar` keeps symlinks inside the
    dataset, so nothing is duplicated and extracting every archive in place
    restores the exact tree; a scene with a link leaving the dataset is
    archived dereferenced. Each archive is written to a temporary name and
    renamed when complete, so an interrupted run resumes by skipping finished
    archives.

    Reading many small files from network storage is latency-bound, so
    archives should be built concurrently.

    Args:
        resolved_path: The dataset folder.
        dst: The staging folder.
        archive_jobs: If given, the pending builds are appended for the
            caller to run in one pool across datasets (so one huge folder does
            not serialize the rest); otherwise they run here.
        workers: The number of threads when the builds run here.
        withheld: Scene folders to leave out (and drop from metadata.json).
    """
    withheld = withheld or set()
    items = sorted(
        item for item in resolved_path.iterdir() if item.name not in withheld)
    dirs = [item for item in items if item.is_dir() and not item.is_symlink()]
    for item in items:
        if item in dirs or (dst / item.name).exists():
            continue
        if item.is_symlink():
            (dst / item.name).symlink_to(os.readlink(item))
        else:
            _stage_top_level_file(item, dst / item.name, withheld,
                                  _link_or_copy)

    jobs = []
    for item in dirs:
        archive = dst / f"{item.name}.tar"
        if archive.exists():
            continue
        jobs.append(
            functools.partial(_build_archive, resolved_path, item.name, dst))
    if archive_jobs is not None:
        archive_jobs.extend(jobs)
    else:
        run_archive_jobs(jobs, workers)


def _has_external_symlink(resolved_path: pathlib.Path, name: str) -> bool:
    """Returns whether a symlink under `resolved_path/name` leaves the dataset.

    E.g. DyCheck's frames link into `VideoDepth_source/dycheck_raw`; such a
    link would dangle once the archive is extracted elsewhere.

    Args:
        resolved_path: The dataset folder.
        name: The top-level entry to check.

    Returns:
        True if any symlink points outside `resolved_path`.
    """
    for root, dirnames, filenames in os.walk(resolved_path / name):
        for entry in dirnames + filenames:
            path = pathlib.Path(root) / entry
            if path.is_symlink() and not path.resolve().is_relative_to(
                    resolved_path.resolve()):
                return True
    return False


def _build_archive(resolved_path: pathlib.Path, name: str,
                   dst: pathlib.Path) -> None:
    """Writes `dst/<name>.tar` of one top-level entry of the dataset."""
    partial = dst / f".{name}.tar.partial"
    # Store the target files instead of the links when any link leaves the
    # dataset; in-tree links stay links.
    if _has_external_symlink(resolved_path, name):
        dereference = ["--dereference"]
    else:
        dereference = []
    command = [
        "tar", "-cf",
        str(partial), *dereference, "-C",
        str(resolved_path), name
    ]
    subprocess.run(command, check=True)
    partial.rename(dst / f"{name}.tar")


def run_archive_jobs(jobs: list, workers: int) -> None:
    """Runs the archive builds in a thread pool; re-raises the first failure."""
    with futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for future in [pool.submit(job) for job in jobs]:
            future.result()


def stage_dataset(
    *,
    benchmark_root: pathlib.Path,
    staging_root: pathlib.Path,
    task: str,
    dataset: str,
    resolved_path: pathlib.Path,
    duplicate_of: str | None,
    copy_data: bool,
    hardlink: bool = False,
    archive_scenes: bool = False,
    archive_workers: int = 1,
    archive_jobs: list | None = None,
) -> pathlib.Path:
    """Stages one dataset folder: its license notice and, optionally, data.

    Args:
        benchmark_root: The benchmark root (unused; kept for callers).
        staging_root: The staging root.
        task: The task name.
        dataset: The dataset name.
        resolved_path: The dataset's folder on disk.
        duplicate_of: The (task/dataset) that already packages the same
            data; the folder then gets a SEE_INSTEAD.md pointer only.
        copy_data: Whether to stage the data files.
        hardlink: Whether to hard-link data files instead of copying.
        archive_scenes: Whether to store each top-level folder as one tar.
        archive_workers: The archive threads when `archive_jobs` is None.
        archive_jobs: If given, collects the pending archive builds.

    Returns:
        The staged dataset folder.
    """
    del benchmark_root
    dst = staging_root / task / dataset
    dst.mkdir(parents=True, exist_ok=True)

    sources = dataset_licenses.sources_for(task, dataset)
    (dst / "LICENSE_NOTICE.md").write_text(_license_notice_text(
        task, dataset, sources),
                                           encoding="utf-8")

    if duplicate_of is not None:
        (dst / "SEE_INSTEAD.md").write_text(
            f"This dataset resolves to the exact same on-disk data as "
            f"`{duplicate_of}` (same source: {resolved_path}). To avoid "
            f"uploading duplicate bytes, the actual files are only packaged "
            f"under `{duplicate_of}/` -- copy/symlink from there if you need "
            "this data under this path too.\n",
            encoding="utf-8",
        )
        return dst

    withheld = set(dataset_licenses.withheld_scenes(task, dataset))
    if copy_data and archive_scenes:
        _archive_dataset(resolved_path, dst, archive_jobs, archive_workers,
                         withheld)
    elif copy_data:
        # Symlinks inside a dataset are materialized (the Hub stores no
        # symlinks). With hardlink=True files are hard-linked instead of
        # copied, so staging costs no extra space when staging_root shares the
        # filesystem.
        if hardlink:
            copy_fn = _link_or_copy
        else:
            copy_fn = shutil.copy2
        for item in resolved_path.iterdir():
            target = dst / item.name
            if target.exists() or item.name in withheld:
                continue
            if item.is_dir():
                shutil.copytree(item, target, copy_function=copy_fn)
            else:
                _stage_top_level_file(item, target, withheld, copy_fn)

    return dst


def _write_dataset_card(staging_root: pathlib.Path,
                        plan: list[dict]) -> pathlib.Path:
    """Writes the release's README.md (the Hugging Face dataset card).

    Args:
        staging_root: The staging root.
        plan: The plan entries the card lists (`card_plan`).

    Returns:
        The README path.
    """
    included = [entry for entry in plan if entry["included"]]
    excluded = [entry for entry in plan if not entry["included"]]

    lines = [
        "# Benchmark datasets\n",
        ("Preprocessed, canonically-laid-out copies of the datasets used by "
         "this repo's"),
        ("`BenchmarkEvaluator` (see `docs/benchmark/evaluator_quickstart.md`), "
         "packaged for"),
        "direct download instead of re-running every `scripts/download/` +",
        "`scripts/preprocess/` step yourself.\n",
        ("**Every dataset folder below is redistributed under its own original "
         "source's"),
        ("license -- see that folder's own `LICENSE_NOTICE.md`. This "
         "repository's own"),
        "license does not override those terms.**\n",
        "## Included datasets\n",
        "| Task | Dataset | Scenes | License status | Source(s) | Data |",
        "|---|---|---|---|---|---|",
    ]
    for entry in included:
        if entry["duplicate_of"]:
            data_note = f"see `{entry['duplicate_of']}/`"
        else:
            data_note = "packaged here"
        if entry.get("withheld_scenes"):
            data_note += (f" (without {', '.join(entry['withheld_scenes'])}; "
                          "see `LICENSE_NOTICE.md`)")
        lines.append(f"| {entry['task']} | {entry['dataset']} | "
                     f"{entry['scene_count']} | `{entry['status']}` | "
                     f"{', '.join(entry['sources']) or '(none)'} | "
                     f"{data_note} |")
    if excluded:
        lines.append("\n## Excluded this run\n")
        lines.append("| Task | Dataset | Reason |")
        lines.append("|---|---|---|")
        for entry in excluded:
            if not entry["present_on_disk"]:
                reason = "not present on disk"
            else:
                reason = f"status={entry['status']}"
            if entry["present_on_disk"] and not (
                    dataset_licenses.is_redistributable(entry["task"],
                                                        entry["dataset"])):
                reason = "source forbids redistribution"
            lines.append(f"| {entry['task']} | {entry['dataset']} | {reason} |")

    if any(staging_root.glob("*/*/*.tar")):
        lines[5:5] = [
            ("**Layout:** each dataset scene is stored as one uncompressed "
             "`<Task>/<dataset>/<scene>.tar` (the raw"),
            ("tree is ~900k files). After downloading, extract everything in "
             "place with the bundled script:\n"),
            "```bash",
            ("hf download Bluefish2333/SpatialArena-Dataset --repo-type dataset "
             "--local-dir /path/to/benchmark_root"),
            ("python /path/to/benchmark_root/extract_benchmark_archives.py "
             "/path/to/benchmark_root --delete"),
            "```\n",
        ]
    if (staging_root / "methods" / "METHODS.md").is_file():
        lines.append("\n## Methods: official code and checkpoints\n")
        lines.append("Third-party code and weights are not redistributed "
                     "here; see `methods/METHODS.md` (and")
        lines.append("`methods/methods_manifest.json`) for each method's "
                     "upstream repository, pinned commit, license and")
        lines.append("checkpoint source, produced by "
                     "`scripts/release/build_method_manifests.py`.")

    text = "\n".join(lines) + "\n"
    path = staging_root / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def datasets_on_hub(repo_id: str) -> set[tuple[str, str]]:
    """Lists the (task, dataset) folders already in the Hub dataset repo.

    Args:
        repo_id: The Hugging Face dataset repo.

    Returns:
        The folders; empty if the repo does not exist yet.
    """
    # pylint: disable=import-outside-toplevel
    import huggingface_hub
    from huggingface_hub import utils as hub_utils

    # pylint: enable=import-outside-toplevel

    try:
        files = huggingface_hub.HfApi().list_repo_files(repo_id=repo_id,
                                                        repo_type="dataset")
    except hub_utils.RepositoryNotFoundError:
        return set()
    return {(parts[0], parts[1])
            for parts in (file.split("/") for file in files)
            if len(parts) >= 3}


def card_plan(full_plan: list[dict], staged: set[tuple[str, str]],
              on_hub: set[tuple[str, str]]) -> list[dict]:
    """Marks which datasets the README card lists as included.

    The card describes the whole repo, not just this run: a dataset is listed
    as included when this run staged it or an earlier run already uploaded it
    (a partial `--datasets` run must not drop the others).

    Args:
        full_plan: The plan of every dataset.
        staged: The (task, dataset) pairs this run staged.
        on_hub: The (task, dataset) pairs already on the Hub.

    Returns:
        A copy of each plan entry with `included` updated.
    """
    entries = []
    for entry in full_plan:
        card_entry = dict(entry)
        card_entry["included"] = (entry["task"],
                                  entry["dataset"]) in staged | on_hub
        entries.append(card_entry)
    return entries


def upload_to_hub(staging_root: pathlib.Path,
                  repo_id: str,
                  *,
                  private: bool = False,
                  num_workers: int | None = None) -> None:
    """Uploads the staging root to a Hugging Face dataset repo.

    Args:
        staging_root: The staging root.
        repo_id: The Hugging Face dataset repo (created if missing).
        private: Whether a new repo is private.
        num_workers: The upload concurrency cap, if any.
    """
    import huggingface_hub  # pylint: disable=import-outside-toplevel

    api = huggingface_hub.HfApi()
    api.create_repo(repo_id=repo_id,
                    repo_type="dataset",
                    private=private,
                    exist_ok=True)
    # huggingface_hub 2.x's upload_folder (which replaced upload_large_folder)
    # streams the files through hf_xet, commits a large folder in batches, and
    # resumes when rerun. Its concurrency is hf_xet's, adaptive by default: cap
    # it at num_workers. upload_large_folder sized its pool from os.cpu_count()
    # (the whole node, not the Slurm allocation); on a 208-core node that meant
    # ~100 concurrent uploads, which tripped Hub rate limits and Xet shard I/O
    # errors. An explicit HF_XET_* setting in the environment wins.
    if num_workers:
        for name in ("HF_XET_MAX_CONCURRENT_FILE_INGESTION",
                     "HF_XET_CLIENT_AC_MAX_UPLOAD_CONCURRENCY"):
            os.environ.setdefault(name, str(num_workers))
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(staging_root),
        commit_message="Upload the benchmark release",
        ignore_patterns=["*.log", "*.err", "*.partial"],
    )


def _parse_args() -> tuple[argparse.Namespace, list[tuple[str, str]] | None]:
    """Parses the command line.

    Returns:
        The arguments and the `--datasets` (task, dataset) filter, or None.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark_root",
                        type=pathlib.Path,
                        default=dataset_registry.DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--staging_root", type=pathlib.Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=("Restrict to specific 'Task/dataset' pairs (e.g. SfM/imc "
              "MVRecon/7scenes). Default: everything in --tasks."),
    )
    parser.add_argument(
        "--include_unconfirmed",
        action="store_true",
        help=("Also package datasets with unconfirmed provenance/license (see "
              "spatialarena/dataset_licenses.py). Only pass this after you've "
              "personally verified the source yourself."),
    )
    parser.add_argument(
        "--stage",
        action="store_true",
        help=("Actually copy dataset files into --staging_root (can be very "
              "large). Without this, only LICENSE_NOTICE.md/README.md are "
              "written so you can review the plan first."),
    )
    parser.add_argument(
        "--hardlink",
        action="store_true",
        help=("With --stage, hard-link data files instead of copying (no extra "
              "space on the same filesystem)."),
    )
    parser.add_argument(
        "--archive_scenes",
        action="store_true",
        help=("With --stage, store each top-level dataset folder as one .tar "
              "(keeps symlinks; far fewer Hub files)."),
    )
    parser.add_argument(
        "--archive_workers",
        type=int,
        default=8,
        help="Archives built concurrently (one pool across all datasets).")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload --staging_root to --repo_id via huggingface_hub.")
    parser.add_argument(
        "--repo_id",
        default=None,
        help="e.g. my-org/my-benchmark. Required with --upload.")
    parser.add_argument("--upload_workers",
                        type=int,
                        default=8,
                        help="Concurrent upload workers (default 8).")
    parser.add_argument("--private",
                        action="store_true",
                        help="Create the HF repo as private (default: public).")
    args = parser.parse_args()

    if args.upload and not args.repo_id:
        parser.error("--upload requires --repo_id")

    dataset_filter = None
    if args.datasets:
        dataset_filter = []
        for item in args.datasets:
            task, _, dataset = item.partition("/")
            if not dataset:
                parser.error("--datasets entries must look like "
                             f"'Task/dataset', got {item!r}")
            dataset_filter.append((task, dataset))
    return args, dataset_filter


def _write_card(args: argparse.Namespace,
                dataset_filter: list[tuple[str, str]] | None,
                plan: list[dict]) -> pathlib.Path:
    """Writes the dataset card, covering the whole repo for partial runs.

    Args:
        args: The command line arguments.
        dataset_filter: The `--datasets` filter, or None.
        plan: This run's plan.

    Returns:
        The README path.
    """
    card_entries = plan
    if dataset_filter is not None or args.repo_id:
        full_plan = plan_dataset_package(
            benchmark_root=args.benchmark_root,
            tasks=tuple(dict.fromkeys((*DEFAULT_TASKS, *args.tasks))),
            include_unconfirmed=True,
        )
        staged = {(entry["task"], entry["dataset"])
                  for entry in plan
                  if entry["included"]}
        if args.repo_id:
            on_hub = datasets_on_hub(args.repo_id)
        else:
            on_hub = set()
        card_entries = card_plan(full_plan, staged, on_hub)
    return _write_dataset_card(args.staging_root, card_entries)


def main() -> None:
    """Plans, stages and uploads the release as the flags say."""
    args, dataset_filter = _parse_args()

    plan = plan_dataset_package(
        benchmark_root=args.benchmark_root,
        tasks=tuple(args.tasks),
        include_unconfirmed=args.include_unconfirmed,
        datasets=dataset_filter,
    )

    print(json.dumps(plan, indent=2))

    if not args.stage and not args.upload:
        print("\n(dry run -- pass --stage to write license notices + copy "
              "data, --upload to publish)")
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
            resolved_path=pathlib.Path(entry["resolved_path"]),
            duplicate_of=entry["duplicate_of"],
            copy_data=args.stage,
            hardlink=args.hardlink,
            archive_scenes=args.archive_scenes,
            archive_workers=args.archive_workers,
            archive_jobs=archive_jobs,
        )
    if archive_jobs:
        # One pool across all datasets.
        print(
            f"Building {len(archive_jobs)} archives with "
            f"{args.archive_workers} workers",
            flush=True)
        run_archive_jobs(archive_jobs, args.archive_workers)
    if args.archive_scenes:
        # Once, after the loop.
        shutil.copy2(
            pathlib.Path(__file__).with_name("extract_benchmark_archives.py"),
            args.staging_root)
    card_path = _write_card(args, dataset_filter, plan)
    print(f"Wrote dataset card: {card_path}")

    if args.upload:
        upload_to_hub(args.staging_root,
                      args.repo_id,
                      private=args.private,
                      num_workers=args.upload_workers)
        print(f"Uploaded to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
