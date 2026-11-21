# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Method and dataset registries, method profiles and dataset licenses.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from scripts.test.benchmark_methods import TRACK_RGB_CAMERAS, TRACK_RGB_DEPTH, resolve_method, validate_track
from scripts.test.generate_benchmark_pdf_report import write_pdf
import importlib.util
from spatialarena.eval.dataset_registry import describe_benchmark_scenes, resolve_dataset_root
from spatialarena.dataset_licenses import (
    DATASET_SOURCES,
    SOURCE_LICENSES,
    is_safe_for_default_public_release,
    sources_for,
)


def only(items):
    """The one element of ``items``."""
    items = list(items)
    assert len(items) == 1, items
    return items[0]


# ---- benchmark_method_registry -----------------------------------------------------------------------------

def test_registry_resolves_requested_methods_and_da3_aliases() -> None:
    assert resolve_method("DA3").name == "DA3-G"
    assert resolve_method("DA3-L").ckpt_default
    assert resolve_method("MapAnything").name == "MAPANYTHING"
    assert resolve_method("Omni-VGGT").name == "OMNIVGGT"
    assert resolve_method("déjà view".replace("é", "e").replace(" ", "-")).name == "DEJAVIEW"


def test_registry_validates_multimodal_tracks() -> None:
    assert TRACK_RGB_DEPTH in validate_track("MAPANYTHING", TRACK_RGB_DEPTH).tracks
    assert TRACK_RGB_CAMERAS in validate_track("PI3-X", TRACK_RGB_CAMERAS).tracks


def test_external_adapter_dry_run_reports_standard_layout(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    images = scene / "images"
    images.mkdir(parents=True)
    (images / "000000.png").write_bytes(b"not-a-real-image-but-path-exists")

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/inference/external_command_adapter.py",
            "--data_path",
            str(scene),
            "--output_path",
            str(tmp_path / "out"),
            "--method_name",
            "MAPANYTHING",
            "--track",
            "rgb_depth",
            "--export_depth",
            "--depth_only",
            "--dry_run",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["method"] == "MAPANYTHING"
    assert payload["depths"].endswith("/depths")


def test_pdf_report_generation_from_synthetic_reports(tmp_path: Path) -> None:
    report_root = tmp_path / "DA3-G"
    depth_report = report_root / "MonoDepth" / "marigold" / "nyuv2" / "eval_depth.json"
    sparse_report = report_root / "SfM" / "imc" / "scene" / "eval_sparse.json"
    depth_report.parent.mkdir(parents=True)
    sparse_report.parent.mkdir(parents=True)
    depth_report.write_text(
        json.dumps(
            {
                "scene_results": [
                    {
                        "method": "DA3-G",
                        "task": "MonoDepth",
                        "dataset": "marigold",
                        "scene": "nyuv2",
                        "num_evaluated_frames": 2,
                        "num_metric_units": 2,
                        "num_valid_pixels": 10,
                        "metrics": {"abs_relative_difference": 0.1, "rmse_linear": 1.0, "delta125_acc": 0.9},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    sparse_report.write_text(
        json.dumps(
            {
                "scene_results": [
                    {
                        "method": "DA3-G",
                        "task": "SfM",
                        "dataset": "imc",
                        "scene": "scene",
                        "num_matched_images": 5,
                        "metrics": {"ate_rmse": 0.2, "relative_rotation_error_deg_mean": 1.0, "mAA_30": 0.8,
                                    "num_pairs": 4},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "report.pdf"
    write_pdf([report_root], output, title="Synthetic", exclude_scannet=False)
    assert output.is_file()
    assert output.stat().st_size > 0


def test_point_query_vlms_query_a_grid_not_just_the_image_centre():
    """DepthLM and Qwen3-VL densify scalar point answers: one query (the adapter's default) makes every map constant."""
    from spatialarena.methods import METHOD_SPECS

    grids = {spec.name: spec.point_grid_size for spec in METHOD_SPECS}
    assert grids["DEPTHLM"] == 16 and grids["QWEN3_VL"] == 16
    assert all(size == 0 for name, size in grids.items() if name not in {"DEPTHLM", "QWEN3_VL"})


# ---- benchmark_method_profiles -----------------------------------------------------------------------------

def load_registry():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "test"
        / "benchmark_methods.py"
    )
    spec = importlib.util.spec_from_file_location("benchmark_methods", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_method_resolution_profiles_match_canonical_defaults():
    registry = load_registry()

    expected = {
        "DA3-G": (504, 504, 0),
        "DA3-NGL": (504, 504, 0),
        "VGGT": (518, 518, 0),
        "PI3_X": (512, 512, 262144),
        "MAPANYTHING": (518, 518, 0),
        "OMNIVGGT": (518, 518, 0),
        "DUST3R": (512, 512, 0),
        "MAST3R": (512, 512, 0),
        "POW3R": (512, 512, 0),
        "DEJAVIEW": (504, 504, 0),
    }

    for method, (resolution, image_load_resolution, pixel_limit) in expected.items():
        spec = registry.validate_track(method, "rgb")
        assert spec.input_resolution == resolution
        assert spec.image_load_resolution == image_load_resolution
        assert spec.pixel_limit == pixel_limit


def test_pi3_alias_uses_pi3_evaluation_profile():
    registry = load_registry()

    spec = registry.validate_track("PI3", "rgb_cameras")

    assert spec.name == "PI3_X"
    assert spec.input_resolution == 512
    assert spec.image_load_resolution == 512
    assert spec.pixel_limit == 512 * 512
    assert spec.depth_conf_thresh == 0.0


def test_pi3_adapter_direct_defaults_match_registry_profile():
    adapter_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "inference"
        / "pi3.py"
    )
    source = adapter_path.read_text(encoding="utf-8")

    assert 'parser.add_argument("--resolution", type=int, default=512)' in source
    assert 'parser.add_argument("--image_load_resolution", type=int, default=512)' in source
    assert 'parser.add_argument("--pixel_limit", type=int, default=262144)' in source
    assert 'parser.add_argument("--conf_thresh", type=float, default=0.0)' in source
    assert "load_pi3_official_eval_rgb_tensor" in source
    assert "official_width_resize14" in source


# ---- dataset_registry --------------------------------------------------------------------------------------

def write_metadata(root: Path, dataset_name: str, scene_roots: list[Path]) -> None:
    metadata = {
        "dataset_name": dataset_name,
        "root_directory": str(root),
        "total_images": len(scene_roots),
        "average_images_per_scene": 1,
        "scenes": [
            {
                "dataset_name": dataset_name,
                "scene_name": scene_root.name,
                "root_directory": str(scene_root),
                "total_images": 1,
            }
            for scene_root in scene_roots
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_registry_describes_selected_dataset(tmp_path):
    dataset_root = tmp_path / "VideoDepth" / "bonn"
    scene = dataset_root / "scene_a"
    (scene / "images").mkdir(parents=True)
    (scene / "depths").mkdir()
    write_metadata(dataset_root, "bonn", [scene])

    record = only(describe_benchmark_scenes(
        benchmark_root=tmp_path,
        tasks=["VideoDepth"],
        datasets=["bonn"],
    ))

    assert record["task"] == "VideoDepth"
    assert record["dataset"] == "bonn"
    assert record["exists"] is True
    assert record["metadata_exists"] is True
    assert record["scene_count"] == 1
    assert record["total_images"] == 1
    assert record["scenes"][0]["scene"] == "scene_a"
    assert record["scenes"][0]["has_images"] is True
    assert record["scenes"][0]["has_depths"] is True


def test_registry_counts_slam_input_frames(tmp_path):
    """SLAM/<dataset>/metadata.json (materialize_slam_inputs.py) records a scene's frames as num_images."""
    dataset_root = tmp_path / "SLAM" / "vbr"
    (dataset_root / "seq_a" / "images").mkdir(parents=True)
    metadata = {"dataset_name": "vbr", "scenes": [{"scene_name": "seq_a", "num_images": 7, "input_size": [504, 252]}]}
    (dataset_root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    record = only(describe_benchmark_scenes(benchmark_root=tmp_path, tasks=["SLAM"], datasets=["vbr"]))

    assert record["total_images"] == 7
    assert record["scenes"][0]["total_images"] == 7


def test_registry_reports_missing_metadata(tmp_path):
    records = describe_benchmark_scenes(
        benchmark_root=tmp_path,
        tasks=["MVRecon"],
        datasets=["dtu"],
    )

    assert records[0]["dataset"] == "dtu"
    assert records[0]["exists"] is False
    assert records[0]["metadata_exists"] is False
    assert records[0]["scene_count"] == 0
    assert records[0]["root_directory"].endswith("MVRecon/dtu")


def test_evaluate_entrypoint_lists_scenes_without_colmap_import(tmp_path):
    dataset_root = tmp_path / "VideoDepth" / "bonn"
    scene = dataset_root / "scene_a"
    (scene / "images").mkdir(parents=True)
    write_metadata(dataset_root, "bonn", [scene])

    result = subprocess.run(
        [
            sys.executable,
            "spatialarena/evaluate.py",
            "--list_scenes",
            "--benchmark_root",
            str(tmp_path),
            "--datasets",
            "bonn",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    records = json.loads(result.stdout)
    assert records[0]["task"] == "VideoDepth"
    assert records[0]["dataset"] == "bonn"
    assert records[0]["scene_count"] == 1


def test_tanks_and_temples_sfm_resolves_in_the_release_layout(tmp_path):
    """The release ships SfM Tanks and Temples as SfM/tanks_and_temples (no 3DGS/ folder); the 3DGS path wins when
    both exist, so existing roots keep their resolved path."""
    release = tmp_path / "release"
    (release / "SfM" / "tanks_and_temples").mkdir(parents=True)
    assert resolve_dataset_root(release, "SfM", "tanks_and_temples") == release / "SfM" / "tanks_and_temples"

    internal = tmp_path / "internal"
    for folder in ("3DGS/tanks_and_temples", "SfM/tanks_and_temples"):
        (internal / folder).mkdir(parents=True)
    assert resolve_dataset_root(internal, "SfM", "tanks_and_temples") == internal / "3DGS" / "tanks_and_temples"


# ---- dataset_licenses --------------------------------------------------------------------------------------

def test_every_dataset_source_key_resolves_to_a_registered_license():
    referenced = {key for keys in DATASET_SOURCES.values() for key in keys}
    missing = referenced - set(SOURCE_LICENSES)
    assert not missing, f"DATASET_SOURCES references unregistered license keys: {missing}"


def test_covers_every_default_dataset_registry_entry_for_the_release_tasks():
    from spatialarena.eval.dataset_registry import DEFAULT_DATASET_SPECS

    release_tasks = {"MVRecon", "MonoDepth", "SfM", "VideoDepth", "SLAM"}
    registered = {(s.task, s.dataset) for s in DEFAULT_DATASET_SPECS if s.task in release_tasks}
    missing = registered - set(DATASET_SOURCES)
    assert not missing, f"registered datasets with no license-source mapping: {missing}"


def test_texturepoorsfm_is_marked_unconfirmed_not_guessed():
    sources = sources_for("SfM", "texturepoorsfm")
    assert len(sources) == 1
    assert sources[0].status == "unconfirmed"
    assert not is_safe_for_default_public_release("SfM", "texturepoorsfm")


def test_scannetpp_is_marked_restricted_noncommercial():
    sources = sources_for("MVRecon", "scannetpp")
    assert sources[0].status == "restricted_noncommercial"
    assert is_safe_for_default_public_release("MVRecon", "scannetpp")  # confirmed license, just restrictive


def test_unregistered_pair_is_not_safe_for_default_release():
    assert not is_safe_for_default_public_release("NotATask", "not_a_dataset")


def test_marigold_withholds_scannet_and_stays_unconfirmed_for_nyuv2():
    from spatialarena.dataset_licenses import is_redistributable, withheld_scenes

    assert SOURCE_LICENSES["scannet"].status == "no_redistribution"
    assert "scannet" in DATASET_SOURCES[("MonoDepth", "marigold")]
    assert set(withheld_scenes("MonoDepth", "marigold")) == {"scannet"}
    assert is_redistributable("MonoDepth", "marigold")  # ScanNet is withheld, the rest may ship
    assert not is_safe_for_default_public_release("MonoDepth", "marigold")  # NYUv2 is unconfirmed


def test_a_shipped_no_redistribution_source_blocks_the_dataset(monkeypatch):
    from spatialarena import dataset_licenses

    monkeypatch.setitem(dataset_licenses.WITHHELD_SCENES, ("MonoDepth", "marigold"), {})
    assert not dataset_licenses.is_redistributable("MonoDepth", "marigold")
    assert not is_safe_for_default_public_release("MonoDepth", "marigold")
