# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Verify benchmark dataset metadata and canonical layout."""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from spatialarena.eval import dataset_registry  # noqa: E402
# pylint: enable=wrong-import-position

DatasetKey = tuple[str, str]


@dataclasses.dataclass(frozen=True, kw_only=True)
class LayoutRequirement:
    """What a ready dataset's canonical layout must contain.

    Attributes:
        require_images: Every scene needs `images/`.
        require_depths: Every scene needs `depths/`.
        require_sparse: Every scene needs `sparse/0`.
        tar_backed: Every scene needs at least one `.tar` archive.
        canonical_pending_ok: Missing canonical folders are warnings, not
            errors.
    """
    require_images: bool = False
    require_depths: bool = False
    require_sparse: bool = False
    tar_backed: bool = False
    canonical_pending_ok: bool = False


@dataclasses.dataclass(frozen=True, kw_only=True)
class VerificationIssue:
    """One problem found in a dataset.

    Attributes:
        level: "error" or "warning".
        task: Benchmark task.
        dataset: Dataset name.
        message: What is wrong.
    """
    level: str
    task: str
    dataset: str
    message: str


READY_REQUIREMENTS: dict[DatasetKey, LayoutRequirement] = {
    ("MonoDepth", "marigold"):
        LayoutRequirement(require_images=True, require_depths=True),
    ("MonoDepth", "aerialmetric"):
        LayoutRequirement(require_images=True, require_depths=True),
    ("MonoDepth", "hammer"):
        LayoutRequirement(require_images=True, require_depths=True),
    ("MonoDepth", "ibims1"):
        LayoutRequirement(require_images=True, require_depths=True),
    ("MonoDepth", "spring"):
        LayoutRequirement(require_images=True, require_depths=True),
    ("MonoDepth", "synth4k"):
        LayoutRequirement(require_images=True, require_depths=True),
    ("VideoDepth", "sintel"):
        LayoutRequirement(require_images=True, require_depths=True),
    ("VideoDepth", "bonn"):
        LayoutRequirement(require_images=True, require_depths=True),
    ("VideoDepth", "kitti"):
        LayoutRequirement(require_images=True, require_depths=True),
    ("SfM", "imc"):
        LayoutRequirement(
            require_images=True,
            require_depths=True,
            require_sparse=True,
        ),
    ("MVRecon", "eth3d"):
        LayoutRequirement(require_images=True, require_depths=True),
    ("MVRecon", "dtu"):
        LayoutRequirement(require_images=True, require_depths=True),
    ("MVRecon", "tanks_and_temples"):
        LayoutRequirement(require_images=True, require_depths=True),
    ("MVRecon", "7scenes"):
        LayoutRequirement(
            require_images=True,
            require_depths=True,
            canonical_pending_ok=True,
        ),
    ("SfM", "eth3d"):
        LayoutRequirement(require_images=True, require_sparse=True),
    ("SfM", "tanks_and_temples"):
        LayoutRequirement(require_images=True, require_sparse=True),
    ("SfM", "dtu"):
        LayoutRequirement(require_images=True, require_sparse=True),
    ("MVRecon", "scannetpp"):
        LayoutRequirement(require_images=True, require_depths=True),
    ("MVRecon", "hiroom"):
        LayoutRequirement(require_images=True, require_depths=True),
    ("SfM", "scannetpp"):
        LayoutRequirement(require_images=True, require_sparse=True),
    ("SfM", "hiroom"):
        LayoutRequirement(require_images=True, require_sparse=True),
    ("SfM", "texturepoorsfm"):
        LayoutRequirement(require_images=True, require_sparse=True),
    ("MVRecon", "nrgbd"):
        LayoutRequirement(require_images=True,
                          require_depths=True,
                          require_sparse=True),
    ("SLAM", "7scenes"):
        LayoutRequirement(require_images=True,
                          require_depths=True,
                          require_sparse=True),
    ("SLAM", "tum_dynamics"):
        LayoutRequirement(require_images=True,
                          require_depths=True,
                          require_sparse=True),
}

PENDING_DATASETS: set[DatasetKey] = {
    ("SfM", "eth3d"),
    ("SfM", "tanks_and_temples"),
}


def _issue(level: str, record: dict, message: str) -> VerificationIssue:
    """Builds an issue for the record's task and dataset."""
    return VerificationIssue(
        level=level,
        task=record["task"],
        dataset=record["dataset"],
        message=message,
    )


def _load_metadata(record: dict) -> dict | None:
    """Loads the record's metadata.json, or returns None if it is missing."""
    metadata_path = pathlib.Path(record["metadata_path"])
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _scene_has_tar(scene: dict) -> bool:
    """Whether the scene folder exists and contains a `.tar` archive."""
    scene_root = pathlib.Path(scene["root_directory"])
    return scene_root.exists() and any(scene_root.glob("*.tar"))


def _scene_issues(record: dict, scene: dict,
                  requirement: LayoutRequirement) -> list[VerificationIssue]:
    """Checks one listed scene against the dataset's layout requirement."""
    issues: list[VerificationIssue] = []
    scene_name = scene["scene"]
    scene_root = pathlib.Path(scene["root_directory"])
    if not scene_root.exists():
        issues.append(
            _issue("error", record,
                   f"scene root missing for {scene_name}: {scene_root}"))
        return issues

    if requirement.tar_backed and not _scene_has_tar(scene):
        issues.append(
            _issue("error", record,
                   f"tar-backed scene has no tar archive: {scene_name}"))

    required_by_field = {
        "has_images": requirement.require_images,
        "has_depths": requirement.require_depths,
        "has_sparse": requirement.require_sparse,
    }
    for field, label in (
        ("has_images", "images"),
        ("has_depths", "depths"),
        ("has_sparse", "sparse/0"),
    ):
        if not required_by_field[field] or scene[field]:
            continue

        level = "warning" if requirement.canonical_pending_ok else "error"
        issues.append(
            _issue(level, record,
                   f"scene {scene_name} missing canonical {label}"))
    return issues


def verify_record(record: dict) -> list[VerificationIssue]:
    """Checks one dataset record from `describe_benchmark_scenes`.

    A missing root or metadata file is an error, or a warning for datasets in
    `PENDING_DATASETS`. The metadata totals must agree with the scene list, and
    datasets in `READY_REQUIREMENTS` must have scenes with the required
    canonical folders.

    Args:
        record: Dataset record.

    Returns:
        The issues found, in check order.
    """
    key = (record["task"], record["dataset"])
    requirement = READY_REQUIREMENTS.get(key)
    is_pending = key in PENDING_DATASETS
    issues: list[VerificationIssue] = []

    if not record["exists"]:
        level = "warning" if is_pending else "error"
        issues.append(
            _issue(level, record,
                   f"dataset root missing: {record['root_directory']}"))
        return issues

    if not record["metadata_exists"]:
        level = "warning" if is_pending else "error"
        issues.append(
            _issue(level, record,
                   f"metadata missing: {record['metadata_path']}"))
        return issues

    metadata = _load_metadata(record)
    if metadata is None:
        issues.append(
            _issue("error", record,
                   f"metadata could not be loaded: {record['metadata_path']}"))
        return issues

    scene_metadata = metadata.get("scenes", [])
    metadata_total = int(metadata.get("total_images", 0))
    scene_total = sum(
        int(scene.get("total_images", 0)) for scene in scene_metadata)

    if metadata_total != scene_total:
        issues.append(
            _issue(
                "error",
                record,
                f"metadata total_images={metadata_total} "
                f"but scene sum={scene_total}",
            ))

    if len(scene_metadata) != record["scene_count"]:
        issues.append(
            _issue(
                "error",
                record,
                f"metadata scene count={len(scene_metadata)} "
                f"but adapter listed {record['scene_count']}",
            ))

    if requirement is not None and record["scene_count"] == 0:
        issues.append(
            _issue("error", record, "ready dataset has zero listed scenes"))

    if requirement is None:
        return issues

    for scene in record["scenes"]:
        issues.extend(_scene_issues(record, scene, requirement))

    return issues


def _ready_records(records: list[dict]) -> list[dict]:
    """Returns the records of datasets in `READY_REQUIREMENTS`."""
    return [
        record for record in records
        if (record["task"], record["dataset"]) in READY_REQUIREMENTS
    ]


def verify_records(records: list[dict],
                   only_ready: bool = False) -> list[VerificationIssue]:
    """Checks every record (or only the ready datasets) with `verify_record`."""
    if only_ready:
        records = _ready_records(records)
    issues: list[VerificationIssue] = []
    for record in records:
        issues.extend(verify_record(record))
    return issues


def print_text_report(records: list[dict], issues: list[VerificationIssue],
                      only_ready: bool) -> None:
    """Prints a Markdown table of the datasets followed by the issues."""
    selected_records = records
    if only_ready:
        selected_records = _ready_records(records)

    print("Benchmark dataset verification")
    print(f"Datasets checked: {len(selected_records)}")
    print("")
    print("| Task | Dataset | Metadata | Scenes | Images |")
    print("| --- | --- | --- | ---: | ---: |")
    for record in selected_records:
        metadata = "yes" if record["metadata_exists"] else "no"
        print(f"| {record['task']} | {record['dataset']} | {metadata} | "
              f"{record['scene_count']} | {record['total_images']} |")

    errors = [issue for issue in issues if issue.level == "error"]
    warnings = [issue for issue in issues if issue.level == "warning"]
    print("")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    for issue in issues:
        print(f"{issue.level.upper()}: {issue.task}/{issue.dataset}: "
              f"{issue.message}")


def parse_args() -> argparse.Namespace:
    """Parses the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark_root",
        type=pathlib.Path,
        default=dataset_registry.DEFAULT_BENCHMARK_ROOT,
        help="Root containing benchmark task folders.",
    )
    parser.add_argument("--tasks",
                        nargs="*",
                        default=[],
                        help="Optional task filter.")
    parser.add_argument("--datasets",
                        nargs="*",
                        default=[],
                        help="Optional dataset filter.")
    parser.add_argument(
        "--include_scannet",
        action="store_true",
        help="Include standalone ScanNet records in verification.",
    )
    parser.add_argument(
        "--only_ready",
        action="store_true",
        help=("Verify only datasets expected to be ready in the current "
              "non-ScanNet pass."),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures.",
    )
    parser.add_argument("--json",
                        action="store_true",
                        help="Emit JSON instead of a text report.")
    return parser.parse_args()


def main() -> None:
    """Runs the script with the command-line arguments."""
    args = parse_args()
    records = dataset_registry.describe_benchmark_scenes(
        benchmark_root=args.benchmark_root.expanduser(),
        tasks=args.tasks,
        datasets=args.datasets,
        include_scannet=args.include_scannet,
    )
    issues = verify_records(records, only_ready=args.only_ready)
    errors = [issue for issue in issues if issue.level == "error"]
    warnings = [issue for issue in issues if issue.level == "warning"]

    if args.json:
        payload = {
            "records": records,
            "issues": [issue.__dict__ for issue in issues],
            "error_count": len(errors),
            "warning_count": len(warnings),
        }
        print(json.dumps(payload, indent=2))
    else:
        print_text_report(records, issues, only_ready=args.only_ready)

    if errors or (args.strict and warnings):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
