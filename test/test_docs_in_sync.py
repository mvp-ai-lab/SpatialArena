# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Generated or registry-derived parts of the docs stay in sync with the code.

evaluation_doc_methods: docs/evaluation.md's protocol matrix lists the evaluated methods per track; keep it in sync
with the planner.
"""

from pathlib import Path
from spatialarena.methods import resolve_method
from spatialarena.task_track_matrix import TASK_TRACK_MATRIX, evaluated_methods, runnable_tracks
from scripts.preprocess.export_benchmark_data_doc import (
    BEGIN_MARKER,
    END_MARKER,
    build_inventory_markdown,
    update_marked_section,
)


# ---- evaluation_doc_methods --------------------------------------------------------------------------------

DOC = Path(__file__).resolve().parents[1] / "docs" / "evaluation.md"
HEADER = ("| Task | Dataset | Images Used | Ground Truth Used | Metrics Used | Protocol "
          "| Evaluated Methods (per track) |")


def expected_cell(task: str) -> str:
    parts = []
    for track in TASK_TRACK_MATRIX[task]:
        for slam_range in (("short", "long") if task == "SLAM" else (None,)):
            names = evaluated_methods(task, track, slam_range)
            label = f"`{track}`" + (f" {slam_range}" if slam_range else "")
            parts.append(f"{label} ({len(names)}): {', '.join(names) if names else '--'}")
    return "<br>".join(parts)


def test_protocol_matrix_lists_the_planned_methods_per_track():
    lines = DOC.read_text(encoding="utf-8").splitlines()
    start = lines.index(HEADER) + 2
    first_cell: dict[str, str] = {}
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        first_cell.setdefault(cells[0], cells[-1])
    assert set(first_cell) == set(TASK_TRACK_MATRIX)
    for task, cell in first_cell.items():
        assert cell == expected_cell(task), f"{task}: regenerate the protocol matrix column"


def test_task_restricted_methods_are_planned_only_for_their_tasks():
    assert evaluated_methods("SfM", "rgb").count("GENCEPTION_POSE") == 1
    for name in ("SENSENOVA_VISION", "DEPTHVLM", "GENCEPTION_DEPTH"):
        assert runnable_tracks("SfM", resolve_method(name)) == ()
        assert runnable_tracks("MVRecon", resolve_method(name)) == ()
        assert runnable_tracks("MonoDepth", resolve_method(name)) == ("rgb",)
    assert "VGGT_LONG" in evaluated_methods("SLAM", "rgb", "long")
    assert "VGGT_LONG" not in evaluated_methods("MVRecon", "rgb")


# ---- benchmark_data_doc_export -----------------------------------------------------------------------------

def test_update_marked_section_replaces_existing_block():
    existing = "\n".join(
        [
            "# Datasets",
            "",
            BEGIN_MARKER,
            "old generated content",
            END_MARKER,
            "",
            "tail",
        ]
    )
    generated = "\n".join([BEGIN_MARKER, "new generated content", END_MARKER, ""])

    updated = update_marked_section(existing, generated)

    assert "old generated content" not in updated
    assert "new generated content" in updated
    assert updated.startswith("# Datasets")
    assert updated.rstrip().endswith("tail")


def test_build_inventory_markdown_contains_dataset_summary(tmp_path):
    markdown = build_inventory_markdown(benchmark_root=tmp_path)

    assert BEGIN_MARKER in markdown
    assert END_MARKER in markdown
    assert "## Scene Inventory" in markdown
    assert "Dataset Summary" in markdown
    assert "Ground-Truth Labels and Metrics" in markdown
    assert "abs_relative_difference" in markdown
    assert "VideoDepth" in markdown
