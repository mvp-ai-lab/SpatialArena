# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = REPO_ROOT / "scripts" / "release" / "pack_and_upload_benchmark_datasets.py"
_spec = importlib.util.spec_from_file_location("pack_and_upload_benchmark_datasets", _SCRIPT_PATH)
pack_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = pack_mod
_spec.loader.exec_module(pack_mod)


def _fake_spec(task, dataset, relative_path, resolved_path=None):
    resolved = Path(resolved_path) if resolved_path is not None else Path("/fake") / relative_path
    return SimpleNamespace(
        task=task,
        dataset=dataset,
        relative_path=Path(relative_path),
        data_path=lambda benchmark_root, _resolved=resolved: _resolved,
    )


def test_plan_marks_unconfirmed_datasets_excluded_by_default():
    fake_specs = [_fake_spec("SfM", "texturepoorsfm", "SfM/texturepoorsfm")]
    with patch.object(pack_mod, "select_dataset_specs", return_value=fake_specs), \
         patch.object(
             pack_mod,
             "describe_benchmark_scenes",
             return_value=[{"exists": True, "metadata_required": True, "metadata_exists": True, "scene_count": 30}],
         ):
        plan = pack_mod.plan_dataset_package(
            benchmark_root=Path("/fake"), tasks=("SfM",), include_unconfirmed=False
        )
    assert len(plan) == 1
    assert plan[0]["status"] == "unconfirmed"
    assert plan[0]["included"] is False


def test_plan_includes_unconfirmed_when_explicitly_requested():
    fake_specs = [_fake_spec("SfM", "texturepoorsfm", "SfM/texturepoorsfm")]
    with patch.object(pack_mod, "select_dataset_specs", return_value=fake_specs), \
         patch.object(
             pack_mod,
             "describe_benchmark_scenes",
             return_value=[{"exists": True, "metadata_required": True, "metadata_exists": True, "scene_count": 30}],
         ):
        plan = pack_mod.plan_dataset_package(
            benchmark_root=Path("/fake"), tasks=("SfM",), include_unconfirmed=True
        )
    assert plan[0]["included"] is True


def test_plan_excludes_safe_dataset_missing_from_disk():
    fake_specs = [_fake_spec("MVRecon", "7scenes", "MVRecon/7scenes")]
    with patch.object(pack_mod, "select_dataset_specs", return_value=fake_specs), \
         patch.object(
             pack_mod,
             "describe_benchmark_scenes",
             return_value=[{"exists": False, "metadata_required": True, "metadata_exists": False, "scene_count": 0}],
         ):
        plan = pack_mod.plan_dataset_package(
            benchmark_root=Path("/fake"), tasks=("MVRecon",), include_unconfirmed=False
        )
    assert plan[0]["status"] == "safe"
    assert plan[0]["present_on_disk"] is False
    assert plan[0]["included"] is False


def test_stage_dataset_writes_license_notice_and_copies_when_requested(tmp_path):
    benchmark_root = tmp_path / "benchmark_root"
    resolved = benchmark_root / "MVRecon" / "7scenes"
    (resolved / "scene1").mkdir(parents=True)
    (resolved / "scene1" / "images.txt").write_text("fake")
    (resolved / "metadata.json").write_text("{}")
    staging_root = tmp_path / "staging"

    dst = pack_mod.stage_dataset(
        benchmark_root=benchmark_root,
        staging_root=staging_root,
        task="MVRecon",
        dataset="7scenes",
        resolved_path=resolved,
        duplicate_of=None,
        copy_data=True,
    )

    assert (dst / "LICENSE_NOTICE.md").exists()
    assert "Microsoft 7-Scenes" in (dst / "LICENSE_NOTICE.md").read_text()
    assert (dst / "scene1" / "images.txt").read_text() == "fake"
    assert (dst / "metadata.json").exists()


def test_stage_dataset_skips_data_copy_when_copy_data_false(tmp_path):
    benchmark_root = tmp_path / "benchmark_root"
    resolved = benchmark_root / "SfM" / "imc"
    (resolved / "scene1").mkdir(parents=True)
    staging_root = tmp_path / "staging"

    dst = pack_mod.stage_dataset(
        benchmark_root=benchmark_root,
        staging_root=staging_root,
        task="SfM",
        dataset="imc",
        resolved_path=resolved,
        duplicate_of=None,
        copy_data=False,
    )

    assert (dst / "LICENSE_NOTICE.md").exists()
    assert not (dst / "scene1").exists()


def test_stage_dataset_writes_pointer_instead_of_copying_a_duplicate(tmp_path):
    benchmark_root = tmp_path / "benchmark_root"
    resolved = benchmark_root / "MVRecon" / "eth3d"
    (resolved / "scene1").mkdir(parents=True)
    staging_root = tmp_path / "staging"

    dst = pack_mod.stage_dataset(
        benchmark_root=benchmark_root,
        staging_root=staging_root,
        task="SfM",
        dataset="eth3d",
        resolved_path=resolved,
        duplicate_of="MVRecon/eth3d",
        copy_data=True,
    )

    assert (dst / "LICENSE_NOTICE.md").exists()
    assert "MVRecon/eth3d" in (dst / "SEE_INSTEAD.md").read_text()
    assert not (dst / "scene1").exists()


def test_plan_marks_a_dataset_as_duplicate_when_it_shares_a_resolved_path():
    same_dir = "/fake/MVRecon/eth3d"
    fake_specs = [
        _fake_spec("MVRecon", "eth3d", "MVRecon/eth3d", resolved_path=same_dir),
        _fake_spec("SfM", "eth3d", "SfM/eth3d", resolved_path=same_dir),
    ]
    with patch.object(pack_mod, "select_dataset_specs", return_value=fake_specs), \
         patch.object(
             pack_mod,
             "describe_benchmark_scenes",
             return_value=[{"exists": True, "metadata_required": True, "metadata_exists": True, "scene_count": 13}],
         ):
        plan = pack_mod.plan_dataset_package(
            benchmark_root=Path("/fake"), tasks=("MVRecon", "SfM"), include_unconfirmed=False
        )
    by_key = {(p["task"], p["dataset"]): p for p in plan}
    assert by_key[("MVRecon", "eth3d")]["duplicate_of"] is None
    assert by_key[("SfM", "eth3d")]["duplicate_of"] == "MVRecon/eth3d"


def test_write_dataset_card_lists_included_and_excluded(tmp_path):
    plan = [
        {"task": "MVRecon", "dataset": "7scenes", "scene_count": 7, "status": "safe",
         "sources": ["Microsoft 7-Scenes"], "included": True, "present_on_disk": True, "duplicate_of": None},
        {"task": "SfM", "dataset": "texturepoorsfm", "scene_count": 30, "status": "unconfirmed",
         "sources": ["TexturePoorSfM"], "included": False, "present_on_disk": True, "duplicate_of": None},
    ]
    path = pack_mod._write_dataset_card(tmp_path, plan)  # pylint: disable=protected-access
    text = path.read_text()
    assert "MVRecon" in text and "7scenes" in text
    assert "Excluded this run" in text
    assert "texturepoorsfm" in text


def test_stage_dataset_hardlinks_files_and_materializes_symlinked_dirs(tmp_path):
    import os

    from scripts.release.pack_and_upload_benchmark_datasets import stage_dataset

    src = tmp_path / "bench" / "VideoDepth" / "bonn"
    (src / "raw" / "rgb").mkdir(parents=True)
    (src / "raw" / "rgb" / "0.png").write_bytes(b"pixels")
    (src / "scene").mkdir()
    (src / "scene" / "images").symlink_to("../raw/rgb", target_is_directory=True)

    dst = stage_dataset(
        benchmark_root=tmp_path / "bench",
        staging_root=tmp_path / "staging",
        task="VideoDepth",
        dataset="bonn",
        resolved_path=src,
        duplicate_of=None,
        copy_data=True,
        hardlink=True,
    )

    staged_image = dst / "scene" / "images" / "0.png"
    assert not (dst / "scene" / "images").is_symlink()  # the Hub stores no symlinks
    assert staged_image.read_bytes() == b"pixels"
    assert os.stat(staged_image).st_ino == os.stat(src / "raw" / "rgb" / "0.png").st_ino  # hard link, no copy
    assert (dst / "LICENSE_NOTICE.md").is_file()


def test_archive_scenes_round_trip_preserves_tree_and_symlinks(tmp_path):
    import os

    from scripts.release.extract_benchmark_archives import extract_all
    from scripts.release.pack_and_upload_benchmark_datasets import stage_dataset

    src = tmp_path / "bench" / "VideoDepth" / "bonn"
    (src / "raw" / "rgb").mkdir(parents=True)
    (src / "raw" / "rgb" / "0.png").write_bytes(b"pixels")
    (src / "scene").mkdir()
    (src / "scene" / "images").symlink_to("../raw/rgb", target_is_directory=True)
    (src / "metadata.json").write_text("{}")

    dst = stage_dataset(
        benchmark_root=tmp_path / "bench", staging_root=tmp_path / "staging", task="VideoDepth", dataset="bonn",
        resolved_path=src, duplicate_of=None, copy_data=True, archive_scenes=True,
    )
    assert sorted(p.name for p in dst.iterdir()) == ["LICENSE_NOTICE.md", "metadata.json", "raw.tar", "scene.tar"]

    assert extract_all(tmp_path / "staging", delete=True) == 2
    assert not list(dst.glob("*.tar"))
    assert (dst / "scene" / "images").is_symlink() and os.readlink(dst / "scene" / "images") == "../raw/rgb"
    assert (dst / "scene" / "images" / "0.png").read_bytes() == b"pixels"


def test_archive_scenes_dereferences_links_leaving_the_dataset(tmp_path):
    from scripts.release.extract_benchmark_archives import extract_all
    from scripts.release.pack_and_upload_benchmark_datasets import stage_dataset

    raw = tmp_path / "bench" / "VideoDepth_source" / "dycheck_raw" / "apple"
    raw.mkdir(parents=True)
    (raw / "0_00000.png").write_bytes(b"pixels")
    src = tmp_path / "bench" / "VideoDepth" / "dycheck"
    (src / "apple" / "images").mkdir(parents=True)
    (src / "apple" / "images" / "00000.png").symlink_to("../../../../VideoDepth_source/dycheck_raw/apple/0_00000.png")

    dst = stage_dataset(
        benchmark_root=tmp_path / "bench", staging_root=tmp_path / "staging", task="VideoDepth", dataset="dycheck",
        resolved_path=src, duplicate_of=None, copy_data=True, archive_scenes=True,
    )
    assert extract_all(tmp_path / "staging", delete=True) == 1
    frame = dst / "apple" / "images" / "00000.png"
    assert not frame.is_symlink() and frame.read_bytes() == b"pixels"


def test_extractor_refuses_paths_outside_destination(tmp_path):
    import io
    import tarfile

    import pytest

    from scripts.release.extract_benchmark_archives import extract_all

    bad = tmp_path / "Task" / "ds" / "evil.tar"
    bad.parent.mkdir(parents=True)
    with tarfile.open(bad, "w") as archive:
        info = tarfile.TarInfo("../../escape.txt")
        info.size = 1
        archive.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="outside"):
        extract_all(tmp_path)
    assert not (tmp_path.parent / "escape.txt").exists()


def _write_marigold_tree(root):
    import json

    scenes = {"nyuv2": 2, "scannet": 3}
    for scene, count in scenes.items():
        (root / scene / "images").mkdir(parents=True)
        for index in range(count):
            (root / scene / "images" / f"{index}.png").write_bytes(b"pixels")
    metadata = {
        "dataset_name": "marigold",
        "total_images": 5,
        "average_images_per_scene": 2.5,
        "scenes": [{"scene_name": scene, "total_images": count} for scene, count in scenes.items()],
    }
    (root / "metadata.json").write_text(json.dumps(metadata))


def test_archive_scenes_leaves_out_withheld_scenes(tmp_path):
    import json

    src = tmp_path / "bench" / "MonoDepth" / "marigold"
    _write_marigold_tree(src)

    dst = pack_mod.stage_dataset(
        benchmark_root=tmp_path / "bench", staging_root=tmp_path / "staging", task="MonoDepth", dataset="marigold",
        resolved_path=src, duplicate_of=None, copy_data=True, archive_scenes=True,
    )

    assert sorted(p.name for p in dst.iterdir()) == ["LICENSE_NOTICE.md", "metadata.json", "nyuv2.tar"]
    metadata = json.loads((dst / "metadata.json").read_text())
    assert [s["scene_name"] for s in metadata["scenes"]] == ["nyuv2"]
    assert metadata["total_images"] == 2 and metadata["average_images_per_scene"] == 2.0
    assert json.loads((src / "metadata.json").read_text())["total_images"] == 5  # the source is untouched
    notice = (dst / "LICENSE_NOTICE.md").read_text()
    assert "Scenes not included" in notice and "scannet_val_sampled_800_1.tar" in notice


def test_plain_copy_leaves_out_withheld_scenes(tmp_path):
    src = tmp_path / "bench" / "MonoDepth" / "marigold"
    _write_marigold_tree(src)

    dst = pack_mod.stage_dataset(
        benchmark_root=tmp_path / "bench", staging_root=tmp_path / "staging", task="MonoDepth", dataset="marigold",
        resolved_path=src, duplicate_of=None, copy_data=True,
    )

    assert (dst / "nyuv2" / "images" / "0.png").is_file()
    assert not (dst / "scannet").exists()
    assert "scannet" not in (dst / "metadata.json").read_text()


def test_plan_counts_scenes_without_withheld_ones():
    fake_specs = [_fake_spec("MonoDepth", "marigold", "MonoDepth/marigold")]
    with patch.object(pack_mod, "select_dataset_specs", return_value=fake_specs), \
         patch.object(
             pack_mod,
             "describe_benchmark_scenes",
             return_value=[{"exists": True, "metadata_required": True, "metadata_exists": True, "scene_count": 5}],
         ):
        plan = pack_mod.plan_dataset_package(
            benchmark_root=Path("/fake"), tasks=("MonoDepth",), include_unconfirmed=True
        )
    assert plan[0]["included"] is True
    assert plan[0]["scene_count"] == 4
    assert plan[0]["withheld_scenes"] == ["scannet"]


def test_card_plan_keeps_datasets_uploaded_by_earlier_runs():
    from scripts.release.pack_and_upload_benchmark_datasets import card_plan

    full_plan = [
        {"task": "SLAM", "dataset": "vbr", "included": True},
        {"task": "VideoDepth", "dataset": "bonn", "included": False},
        {"task": "VideoDepth", "dataset": "dycheck", "included": False},
        {"task": "VideoDepth", "dataset": "sintel", "included": False},
    ]
    entries = card_plan(full_plan, staged={("VideoDepth", "dycheck")}, on_hub={("SLAM", "vbr"), ("VideoDepth", "bonn")})
    assert {(p["dataset"], p["included"]) for p in entries} == {
        ("vbr", True), ("bonn", True), ("dycheck", True), ("sintel", False)
    }
