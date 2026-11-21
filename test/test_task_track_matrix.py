# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import subprocess
from pathlib import Path

from spatialarena.methods import METHOD_SPECS, resolve_method
from spatialarena.task_track_matrix import (
    TASK_TRACK_MATRIX,
    matrix_tracks,
    runnable_tracks,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_matrix_excludes_circular_combinations():
    assert TASK_TRACK_MATRIX["MonoDepth"] == ("rgb", "rgb_masked_depth")
    assert TASK_TRACK_MATRIX["VideoDepth"] == ("rgb", "rgb_cameras", "rgb_masked_depth", "rgb_sparse_depth")
    assert "rgb_depth" not in TASK_TRACK_MATRIX["MonoDepth"]
    assert "rgb_depth" not in TASK_TRACK_MATRIX["VideoDepth"]
    assert TASK_TRACK_MATRIX["SfM"] == (
        "rgb", "rgb_depth", "rgb_masked_depth", "rgb_masked_depth_cameras",
        "rgb_sparse_depth", "rgb_sparse_depth_cameras")
    assert TASK_TRACK_MATRIX["SLAM"] == ("rgb", "rgb_depth", "rgb_masked_depth")
    assert "rgb_cameras" not in TASK_TRACK_MATRIX["SfM"]
    assert "rgb_cameras" not in TASK_TRACK_MATRIX["SLAM"]
    assert TASK_TRACK_MATRIX["MVRecon"] == (
        "rgb", "rgb_cameras", "rgb_depth", "rgb_masked_depth", "rgb_masked_depth_cameras",
        "rgb_sparse_depth", "rgb_sparse_depth_cameras"
    )


def test_masked_depth_tracks_run_on_every_task_and_with_cameras_on_mvrecon_and_sfm_only():
    """RGB + masked depth is a track of all five tasks; RGB + masked depth + cameras of MVRecon and SfM only."""
    from spatialarena.task_track_matrix import on_matrix, track_runs_on  # pylint: disable=import-outside-toplevel

    pi3 = resolve_method("PI3_X")
    for task in ("MonoDepth", "VideoDepth", "MVRecon", "SfM", "SLAM"):
        assert "rgb_masked_depth" in runnable_tracks(task, pi3)
        assert ("rgb_masked_depth_cameras" in runnable_tracks(task, pi3)) == (task in ("MVRecon", "SfM"))
    # Masked GT depth is incomplete even where the GT is dense, so MVRecon's rgb_depth restriction does not apply.
    assert track_runs_on("MVRecon", "rgb_masked_depth", "hiroom")
    assert on_matrix("MVRecon", "rgb_masked_depth_cameras", "scannetpp")
    assert not on_matrix("VideoDepth", "rgb_masked_depth_cameras", "sintel")
    # The methods whose adapters take both GT depth and GT cameras.
    assert {spec.name for spec in METHOD_SPECS if "rgb_masked_depth" in spec.tracks} == {
        "PI3_X", "MAPANYTHING", "OMNIVGGT"}
    assert {spec.name for spec in METHOD_SPECS if "rgb_masked_depth_cameras" in spec.tracks} == {
        "PI3_X", "MAPANYTHING", "OMNIVGGT"}


def test_engine_track_input_helpers():
    """The engine hands depth priors to every depth track and camera priors to every camera track."""
    script = (
        REPO_ROOT / "spatialarena" / "templates" / "_engine" / "run_method_case_list.sh"
    ).read_text(encoding="utf-8")
    functions = "\n".join(
        script[script.index(f"{name}() {{"):].split("\n}\n", 1)[0] + "\n}"
        for name in ("track_uses_depth", "track_uses_cameras")
    )
    expected = {
        "rgb": (1, 1),
        "rgb_depth": (0, 1),
        "rgb_cameras": (1, 0),
        "rgb_masked_depth": (0, 1),
        "rgb_masked_depth_cameras": (0, 0),
    }
    for track, (depth_code, camera_code) in expected.items():
        for helper, code in (("track_uses_depth", depth_code), ("track_uses_cameras", camera_code)):
            result = subprocess.run(["bash", "-c", f"{functions}\nBENCHMARK_TRACK={track}\n{helper}"], check=False)
            assert result.returncode == code, (track, helper)
    assert '[ "$BENCHMARK_TRACK" = "rgb_masked_depth_cameras" ] && [[ " MVRecon SfM " != *" $canonical_task "* ]]' \
        in script


def test_matrix_tracks_unknown_task_is_empty():
    assert matrix_tracks("NotARealTask") == ()


def test_runnable_tracks_intersects_matrix_and_method_tracks():
    pi3x = resolve_method("PI3_X")
    assert pi3x.tracks == (
        "rgb", "rgb_depth", "rgb_cameras", "rgb_masked_depth", "rgb_masked_depth_cameras",
        "rgb_sparse_depth", "rgb_sparse_depth_cameras")

    assert runnable_tracks("MonoDepth", pi3x) == ("rgb", "rgb_masked_depth")
    assert set(runnable_tracks("VideoDepth", pi3x)) == {"rgb", "rgb_cameras", "rgb_masked_depth", "rgb_sparse_depth"}
    assert set(runnable_tracks("MVRecon", pi3x)) == {
        "rgb", "rgb_cameras", "rgb_depth", "rgb_masked_depth", "rgb_masked_depth_cameras",
        "rgb_sparse_depth", "rgb_sparse_depth_cameras"}
    assert set(runnable_tracks("SfM", pi3x)) == {
        "rgb", "rgb_depth", "rgb_masked_depth", "rgb_masked_depth_cameras",
        "rgb_sparse_depth", "rgb_sparse_depth_cameras"}
    assert set(runnable_tracks("SLAM", pi3x)) == {"rgb", "rgb_depth", "rgb_masked_depth"}


def test_runnable_tracks_for_rgb_only_method_is_always_just_rgb():
    da3_l = resolve_method("DA3-L")
    assert da3_l.tracks == ("rgb",)
    for task in TASK_TRACK_MATRIX:
        assert runnable_tracks(task, da3_l) == ("rgb",)


def test_da3_ngl_only_gains_rgb_cameras():
    da3_ngl = resolve_method("DA3-NGL")
    assert set(da3_ngl.tracks) == {"rgb", "rgb_cameras"}
    assert set(runnable_tracks("VideoDepth", da3_ngl)) == {"rgb", "rgb_cameras"}
    # MVRecon allows rgb_cameras in the matrix, but DA3-NGL doesn't support rgb_depth,
    # so only rgb/rgb_cameras are runnable for it there.
    assert set(runnable_tracks("MVRecon", da3_ngl)) == {"rgb", "rgb_cameras"}
    # SfM/SLAM don't allow rgb_cameras in the matrix at all.
    assert runnable_tracks("SfM", da3_ngl) == ("rgb",)


def test_every_method_spec_is_covered_by_the_matrix_without_crashing():
    for spec in METHOD_SPECS:
        for task in TASK_TRACK_MATRIX:
            # Must not raise, and must always be a subset of the method's own tracks.
            tracks = runnable_tracks(task, spec)
            assert set(tracks) <= set(spec.tracks)


def test_bash_guard_matches_matrix_for_rgb_cameras():
    """Drift guard: run_method_case_list.sh (spatialarena/templates/_engine/) must reject rgb_cameras for
    exactly the tasks the matrix leaves it out of (MonoDepth, SfM, SLAM), so a hand-run case list cannot produce
    reports for a (task, track) pair plan() never generates."""
    script = (
        REPO_ROOT / "spatialarena" / "templates" / "_engine" / "run_method_case_list.sh"
    ).read_text(encoding="utf-8")
    without_cameras = sorted(task for task, tracks in TASK_TRACK_MATRIX.items() if "rgb_cameras" not in tracks)
    assert without_cameras == ["MonoDepth", "SLAM", "SfM"]
    guard = 'if [ "$BENCHMARK_TRACK" = "rgb_cameras" ] && [[ " MonoDepth SfM SLAM " == *" $canonical_task "* ]]; then'
    assert guard in script
    assert "has no rgb_cameras track" in script


def _engine_function(script: str, name: str) -> str:
    """The text of one top-level shell function of the engine script."""
    start = script.index(f"\n{name}() {{\n") + 1
    return script[start:script.index("\n}\n", start) + 3]


def test_depth_only_cases_export_cameras_for_videodepth_and_mvrecon():
    """VideoDepth and MVRecon inference export the predicted cameras (MVRecon fuses with them; VideoDepth keeps them
    for the dashboard's demo replay); MonoDepth does not, nor do methods whose adapters predict no camera."""
    script = (
        REPO_ROOT / "spatialarena" / "templates" / "_engine" / "run_method_case_list.sh"
    ).read_text(encoding="utf-8")
    functions = _engine_function(script, "canonical_task_name") + _engine_function(script, "depth_only_exports_poses")
    cases = {("PI3_X", "VideoDepth"): 0, ("VGGT_OMEGA", "MVRecon"): 0,
             ("AMB3R", "MVRecon"): 0, ("ROBUSTVGGT", "MVRecon"): 0, ("ROBUSTVGGT", "VideoDepth"): 0,
             ("PI3_X", "MonoDepth"): 1, ("DEPTHVLM", "VideoDepth"): 1}
    for (method, task), expected in cases.items():
        result = subprocess.run(["bash", "-c", f'{functions}\nMETHOD={method}\ndepth_only_exports_poses {task}'],
                                check=False)
        assert result.returncode == expected, (method, task)
    assert 'if depth_only_exports_poses "$task"; then\n            cmd+=(--export_colmap)' in script


def test_mvrecon_rgb_depth_runs_only_on_sparse_or_incomplete_gt_depth():
    """MVRecon rgb_depth is given GT depth, so it runs only where that depth is sparse or incomplete."""
    from spatialarena.task_track_matrix import track_runs_on  # pylint: disable=import-outside-toplevel

    assert [d for d in ("7scenes", "dtu", "eth3d", "hiroom", "nrgbd", "scannetpp", "tanks_and_temples")
            if track_runs_on("MVRecon", "rgb_depth", d)] == ["7scenes", "dtu", "eth3d"]
    assert track_runs_on("MVRecon", "rgb", "hiroom") and track_runs_on("MVRecon", "rgb_cameras", "scannetpp")
    assert track_runs_on("SfM", "rgb_depth", "hiroom")


def test_bash_guard_matches_matrix_for_mvrecon_rgb_depth():
    """Drift guard: the engine runs MVRecon rgb_depth on exactly TRACK_DATASETS' datasets, so a hand-run case list
    cannot produce reports plan() never generates."""
    from spatialarena.task_track_matrix import TRACK_DATASETS  # pylint: disable=import-outside-toplevel

    script = (
        REPO_ROOT / "spatialarena" / "templates" / "_engine" / "run_method_case_list.sh"
    ).read_text(encoding="utf-8")
    line = next(line for line in script.splitlines() if line.startswith("MVRECON_RGB_DEPTH_DATASETS="))
    assert set(line.split("=", 1)[1].strip('"').split()) == set(TRACK_DATASETS[("MVRecon", "rgb_depth")])
    assert '[[ " $MVRECON_RGB_DEPTH_DATASETS " != *" ${dataset,,} "* ]]' in script


def test_sfm_rgb_depth_runs_only_where_the_released_gt_has_depth():
    """SfM [+D] takes the GT depth maps: none in the released DTU, Tanks and Temples or Weak-Texture SfM GT."""
    from spatialarena.task_track_matrix import track_runs_on  # pylint: disable=import-outside-toplevel

    assert [d for d in ("dtu", "eth3d", "hiroom", "imc", "scannetpp", "spatialarena", "tanks_and_temples",
                        "texturepoorsfm") if track_runs_on("SfM", "rgb_depth", d)] == [
        "eth3d", "hiroom", "imc", "scannetpp", "spatialarena"]
    assert track_runs_on("SfM", "rgb", "dtu")


def test_bash_guard_matches_matrix_for_sfm_rgb_depth():
    """Drift guard: the engine runs SfM rgb_depth on exactly TRACK_DATASETS' datasets."""
    from spatialarena.task_track_matrix import TRACK_DATASETS  # pylint: disable=import-outside-toplevel

    script = (
        REPO_ROOT / "spatialarena" / "templates" / "_engine" / "run_method_case_list.sh"
    ).read_text(encoding="utf-8")
    line = next(line for line in script.splitlines() if line.startswith("SFM_RGB_DEPTH_DATASETS="))
    assert set(line.split("=", 1)[1].strip('"').split()) == set(TRACK_DATASETS[("SfM", "rgb_depth")])
    assert '[[ " $SFM_RGB_DEPTH_DATASETS " != *" ${dataset,,} "* ]]' in script


def test_videodepth_rgb_cameras_runs_only_where_the_released_gt_has_cameras():
    """[+C] runs on SpatialArena only: KITTI has just the OXTS vehicle trajectory, the released GT no Sintel/DyCheck
    cameras."""
    from spatialarena.task_track_matrix import track_runs_on  # pylint: disable=import-outside-toplevel

    assert [d for d in ("bonn", "dycheck", "kitti", "sintel", "spatialarena")
            if track_runs_on("VideoDepth", "rgb_cameras", d)] == ["spatialarena"]
    assert track_runs_on("VideoDepth", "rgb", "kitti") and track_runs_on("MVRecon", "rgb_cameras", "dtu")


def test_bash_guard_matches_matrix_for_videodepth_rgb_cameras():
    """Drift guard: the engine runs VideoDepth rgb_cameras on exactly TRACK_DATASETS' datasets."""
    from spatialarena.task_track_matrix import TRACK_DATASETS  # pylint: disable=import-outside-toplevel

    script = (
        REPO_ROOT / "spatialarena" / "templates" / "_engine" / "run_method_case_list.sh"
    ).read_text(encoding="utf-8")
    line = next(line for line in script.splitlines() if line.startswith("VIDEODEPTH_RGB_CAMERAS_DATASETS="))
    assert set(line.split("=", 1)[1].strip('"').split()) == set(TRACK_DATASETS[("VideoDepth", "rgb_cameras")])
    assert '[[ " $VIDEODEPTH_RGB_CAMERAS_DATASETS " != *" ${dataset,,} "* ]]' in script


def test_on_matrix_rules_out_depth_tasks_given_depth():
    from spatialarena.task_track_matrix import on_matrix

    assert on_matrix("MonoDepth", "rgb", "marigold")
    assert not on_matrix("MonoDepth", "rgb_depth", "marigold")
    assert not on_matrix("VideoDepth", "rgb_depth", "sintel")
    assert not on_matrix("VideoDepth", "rgb_cameras", "sintel")
    assert on_matrix("VideoDepth", "rgb_cameras", "spatialarena")
    assert on_matrix("MVRecon", "rgb_depth", "DTU") and not on_matrix("MVRecon", "rgb_depth", "hiroom")
    assert on_matrix("WeakTextureSfM", "rgb_depth", "texturepoorsfm")  # unlisted task: not filtered


def test_slam_rgb_depth_runs_only_on_sequences_with_depth_maps():
    """SLAM [+D] skips the datasets and the SpatialArena sequences whose inputs have no depth maps."""
    # pylint: disable=import-outside-toplevel
    from spatialarena.task_track_matrix import track_runs_on, track_runs_on_case

    for dataset in ("oxford_spires", "kitti_odometry", "vbr"):
        assert not track_runs_on("SLAM", "rgb_depth", dataset) and track_runs_on("SLAM", "rgb", dataset)
    assert track_runs_on_case("SLAM", "rgb_depth", "7scenes", "chess_seq-03")
    assert track_runs_on_case("SLAM", "rgb_depth", "spatialarena", "ArtHouse")
    assert not track_runs_on_case("SLAM", "rgb_depth", "spatialarena", "EdinburghCastle1")
    assert not track_runs_on_case("SLAM", "rgb_depth", "spatialarena", "StPancrasOutside")
    assert track_runs_on_case("SLAM", "rgb", "spatialarena", "EdinburghCastle1")


def test_bash_guard_matches_matrix_for_slam_rgb_depth():
    """Drift guard: the engine runs SLAM rgb_depth on exactly TRACK_DATASETS' datasets minus TRACK_EXCLUDED_SCENES."""
    # pylint: disable=import-outside-toplevel
    from spatialarena.task_track_matrix import TRACK_DATASETS, TRACK_EXCLUDED_SCENES

    script = (
        REPO_ROOT / "spatialarena" / "templates" / "_engine" / "run_method_case_list.sh"
    ).read_text(encoding="utf-8")

    def values(name):
        line = next(line for line in script.splitlines() if line.startswith(f"{name}="))
        return set(line.split("=", 1)[1].strip('"').split())

    assert values("SLAM_RGB_DEPTH_DATASETS") == set(TRACK_DATASETS[("SLAM", "rgb_depth")])
    excluded = TRACK_EXCLUDED_SCENES[("SLAM", "rgb_depth")]
    assert values("SLAM_RGB_DEPTH_EXCLUDED_SCENES") == set().union(*excluded.values())
    assert '[[ " $SLAM_RGB_DEPTH_EXCLUDED_SCENES " == *" $scene "* ]]' in script


def test_slam_rgb_depth_excluded_scenes_have_no_depth_inputs():
    """Each excluded sequence is one without depth maps, and every other case-list sequence of the dataset has them,
    when the SLAM inputs are present locally."""
    # pylint: disable=import-outside-toplevel
    import pytest

    from spatialarena import slam_protocol
    from spatialarena.paths import SPATIALARENA_ROOT
    from spatialarena.task_track_matrix import TRACK_EXCLUDED_SCENES

    root = slam_protocol.input_root(Path(SPATIALARENA_ROOT), "spatialarena")
    if not root.is_dir():
        pytest.skip("SLAM inputs not available")
    excluded = TRACK_EXCLUDED_SCENES[("SLAM", "rgb_depth")]["spatialarena"]
    for scene in excluded:
        assert not (root / scene / "depths").is_dir(), scene
    for slam_range in ("short", "long"):
        cases = REPO_ROOT / "scripts" / "test" / f"full_standard_slam_cases_{slam_range}.txt"
        for line in cases.read_text(encoding="utf-8").splitlines():
            parts = line.split("|")
            if len(parts) > 2 and parts[1] == "spatialarena" and parts[2] not in excluded:
                assert (root / parts[2] / "depths").is_dir(), parts[2]
