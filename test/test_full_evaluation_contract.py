# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import subprocess
import sys
from pathlib import Path


def test_full_standard_pose_case_generation_includes_texturepoorsfm(
        tmp_path: Path) -> None:
    benchmark_root = tmp_path / "benchmarks"
    scene_name = "1000__10bag000"
    scene_root = benchmark_root / "SfM" / "texturepoorsfm" / scene_name
    (scene_root / "images").mkdir(parents=True)
    (scene_root / "sparse" / "0").mkdir(parents=True)
    (benchmark_root / "SfM" / "texturepoorsfm").mkdir(parents=True,
                                                      exist_ok=True)
    (benchmark_root / "SfM" / "texturepoorsfm" / "metadata.json").write_text(
        """{
  "dataset_name": "texturepoorsfm",
  "root_directory": "%s",
  "total_images": 1,
  "average_images_per_scene": 1,
    "scenes": [
    {
      "dataset_name": "texturepoorsfm",
      "scene_name": "%s",
      "root_directory": "%s/%s",
      "total_images": 1
    }
  ]
}
""" % (
            benchmark_root / "SfM" / "texturepoorsfm",
            scene_name,
            benchmark_root / "SfM" / "texturepoorsfm",
            scene_name,
        ),
        encoding="utf-8",
    )

    output = tmp_path / "full_standard_sfm_cases.txt"
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "scripts/test/generate_full_standard_cases.py",
            "--benchmark_root",
            str(benchmark_root),
            "--output",
            str(output),
            "--tasks",
            "SfM",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    lines = [
        line for line in output.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert lines == [f"SfM|texturepoorsfm|{scene_name}|1"]


def test_full_evaluation_contract_no_longer_mentions_doppelgangers_catalog(
) -> None:
    contract = Path(__file__).resolve().parents[1] / "docs" / "evaluation.md"
    text = contract.read_text(encoding="utf-8")

    assert "texturepoorsfm" in text
    assert "Berlin_Cathedral" not in text
    assert "doppelgangers" not in text


def test_checked_in_full_standard_pose_case_list_includes_texturepoorsfm(
) -> None:
    """texturepoorsfm has 1020 bags over 17 objects, and bags of one object share
    frames. The v2 split holds out whole objects: the benchmark uses 30 bags from
    the 5 test objects only, so no benchmark frame can appear in training."""
    import json

    repo_root = Path(__file__).resolve().parents[1]
    split = json.loads(
        (repo_root / "scripts" / "preprocess" / "assets" /
         "texturepoorsfm_split_v2.json").read_text(encoding="utf-8"))
    case_file = repo_root / "scripts" / "test" / "full_standard_sfm_cases.txt"
    text = case_file.read_text(encoding="utf-8")

    texturepoorsfm_lines = [
        line for line in text.splitlines()
        if line.startswith("SfM|texturepoorsfm|")
    ]
    assert [line.split("|")[2] for line in texturepoorsfm_lines
           ] == split["benchmark_scenes"]
    assert len(texturepoorsfm_lines) == 30

    objects = {
        line.split("|")[2].split("__")[0] for line in texturepoorsfm_lines
    }
    assert objects == set(split["test_objects"])
    assert not objects & set(split["train_objects"])
    assert "doppelgangers" not in text
