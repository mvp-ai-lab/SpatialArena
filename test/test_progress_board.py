# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Progress display modes (spatialarena/progress_ui.py) and the interactive board (progress_board.py)."""

import asyncio

import pytest

pytest.importorskip("textual")

from spatialarena.progress import PipelineProgress, RunProgress  # noqa: E402
from spatialarena.progress_board import BoardApp  # noqa: E402
from spatialarena.progress_ui import (  # noqa: E402
    BarsReporter, ProgressReporter, group_runs, resolve_progress_mode,
    run_with_progress,
)


def _snapshot() -> PipelineProgress:

    def run(method,
            task,
            dataset,
            stage,
            done_inf,
            done_eval,
            total,
            slam_range=""):
        return RunProgress(label=f"{method} {task}/{dataset}",
                           stage=stage,
                           total_cases=total,
                           inference_done=done_inf,
                           evaluation_done=done_eval,
                           method=method,
                           track="rgb",
                           task=task,
                           dataset=dataset,
                           slam_range=slam_range)

    return PipelineProgress(runs=[
        run("PI3_X", "SLAM", "tum_dynamics", "evaluation", 9, 4, 9, "long"),
        run("PI3_X", "SLAM", "7scenes", "done", 18, 18, 18, "long"),
        run("VGGT", "SLAM", "7scenes", "inference", 7, 0, 18, "short"),
        run("VGGT", "SfM", "eth3d", "failed", 3, 0, 13),
        run("DA3-L", "SfM", "eth3d", "queued", 0, 0, 13),
    ])


def test_mode_resolution(monkeypatch):
    monkeypatch.delenv("BENCHMARK_PROGRESS_MODE", raising=False)
    assert resolve_progress_mode(None) == "bars"
    assert resolve_progress_mode(None, "board") == "board"
    assert resolve_progress_mode("none", "board") == "none"
    monkeypatch.setenv("BENCHMARK_PROGRESS_MODE", "board")
    assert resolve_progress_mode(None) == "board"
    with pytest.raises(ValueError):
        resolve_progress_mode("fancy")


def test_group_runs_aggregates_stages_and_cases():
    groups = {
        g["key"]: g
        for g in group_runs(_snapshot().runs, lambda run: run.track_key)
    }
    assert set(groups) == {"SLAM|rgb|long", "SLAM|rgb|short", "SfM|rgb"}
    long = groups["SLAM|rgb|long"]
    assert (long["runs"], long["done"], long["evaluation"]) == (2, 1, 1)
    assert (long["evaluation_cases"], long["inference_cases"]) == ([22, 27],
                                                                   [27, 27])
    assert groups["SfM|rgb"]["failed"] == 1 and groups["SfM|rgb"]["queued"] == 1


def test_board_views_keys_and_quit_confirmation():

    async def scenario():
        app = BoardApp(title="test board")
        async with app.run_test(size=(180, 50)) as pilot:
            app.apply_snapshot(_snapshot())
            app.apply_download({
                "kind": "dataset",
                "name": "SLAM/7scenes",
                "status": "running",
                "step": "materialize",
                "detail": ""
            })
            app.apply_download({
                "kind": "checkpoint",
                "name": "VGGT",
                "status": "failed",
                "step": None,
                "detail": "HTTP 403"
            })
            await pilot.pause()
            assert app.query_one("#tracks-table").row_count == 3
            assert app.query_one("#methods-table").row_count == 3
            assert app.query_one("#datasets-table").row_count == 3
            assert app.query_one("#runs-table").row_count == 5
            assert app.query_one("#downloads-table").row_count == 2

            await pilot.press("4")
            assert app.query_one("#views").active == "runs"
            await pilot.press(
                "f")  # all -> active (queued / inference / evaluation)
            assert app.run_filter == "active" and app.query_one(
                "#runs-table").row_count == 3
            await pilot.press("f")  # -> failed
            assert app.query_one("#runs-table").row_count == 1
            assert "VGGT" in str(app.query_one("#run-detail").render())

            await pilot.press("q")  # still running: first q only asks
            await pilot.pause()
            assert app.is_running
            app.mark_finished(None)
            await pilot.press("q")
            await pilot.pause()
        assert not app.is_running

    asyncio.run(scenario())


def test_board_falls_back_to_bars_without_a_terminal(monkeypatch):
    monkeypatch.setattr("spatialarena.progress_ui.board_available", lambda:
                        (False, "no terminal"))
    seen = []

    def work(reporter):
        seen.append(type(reporter))
        reporter.download({
            "kind": "code",
            "name": "VGGT",
            "status": "ok",
            "step": None,
            "detail": ""
        })
        return 42

    with pytest.warns(UserWarning, match="no terminal"):
        assert run_with_progress("board", work, title="t") == 42
    assert seen == [BarsReporter]
    assert run_with_progress("none", type, title="t") is ProgressReporter


def test_board_reporter_switches_to_bars_after_the_board_closes():
    app = BoardApp(title="t")
    reporter = app.reporter
    reporter.evaluation(
        _snapshot())  # board not running: nothing raised, state kept
    reporter.switch_to_bars()
    reporter.evaluation(_snapshot())
    reporter.download({
        "kind": "dataset",
        "name": "SfM/eth3d",
        "status": "running",
        "step": None,
        "detail": ""
    })
    reporter.close()


def test_board_runs_the_work_in_a_thread_and_reports_its_result(monkeypatch):
    import time

    monkeypatch.setattr("spatialarena.progress_ui.board_available", lambda:
                        (True, ""))
    original_run = BoardApp.run
    seen = {}

    async def pilot_until_finished(pilot):
        app = pilot.app
        for _ in range(200):
            if app.finished:
                break
            await pilot.pause(0.02)
        seen["runs_rows"] = app.query_one("#runs-table").row_count
        seen["download_rows"] = app.query_one("#downloads-table").row_count
        await pilot.press("q")

    monkeypatch.setattr(
        BoardApp, "run", lambda self: original_run(
            self, headless=True, auto_pilot=pilot_until_finished))

    def work(reporter):
        for event_status in ("pending", "running", "ok"):
            reporter.download({
                "kind": "code",
                "name": "VGGT",
                "status": event_status,
                "step": None,
                "detail": ""
            })
        reporter.evaluation(_snapshot())
        time.sleep(0.05)
        return "result"

    assert run_with_progress("board", work, title="t") == "result"
    assert seen == {"runs_rows": 5, "download_rows": 1}
