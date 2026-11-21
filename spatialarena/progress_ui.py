# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""How downloads and evaluations show progress in the terminal: ``"bars"``, ``"board"`` or ``"none"``.

- ``bars``: tqdm progress bars -- runs finished / inference cases / evaluation cases for an evaluation, one bar per
  kind (datasets, checkpoints, method code) for downloads. Works in logs and non-interactive shells.
- ``board``: an interactive full-screen terminal board (Textual) with views by track, method and dataset, per run,
  and per download item; keys switch views and filter. Needs an interactive terminal and the ``textual`` package
  (``pip install textual``, or the ``board`` extra); otherwise it falls back to ``bars`` with a warning.
- ``none``: no progress output.

The mode comes from the call's ``progress=`` argument, else ``BenchmarkEvaluator(progress_mode=...)`` / the config
file's ``progress_mode`` / the ``BENCHMARK_PROGRESS_MODE`` environment variable, else ``bars``.

Work and display are decoupled through ``ProgressReporter``: the work calls ``reporter.evaluation(snapshot)`` with a
``PipelineProgress`` and ``reporter.download(event)`` with download events (``spatialarena.download``'s
``on_event``). ``run_with_progress`` runs the work with the chosen reporter; with the board, the work runs in a
background thread while the board owns the terminal.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import warnings
from collections import OrderedDict
from typing import Any, Callable, TypeVar

from spatialarena.progress import STAGES, PipelineProgress, ProgressBars, RunProgress

PROGRESS_MODES = ("bars", "board", "none")
DEFAULT_PROGRESS_MODE = "bars"
DOWNLOAD_KINDS = ("dataset", "checkpoint", "code")
DOWNLOAD_DONE = ("ok", "failed", "skipped", "already_present")
T = TypeVar("T")


def resolve_progress_mode(mode: str | None, default: str | None = None) -> str:
    """The call's mode, else ``default`` (the evaluator's setting), else $BENCHMARK_PROGRESS_MODE, else "bars"."""
    resolved = mode or default or os.environ.get("BENCHMARK_PROGRESS_MODE") or DEFAULT_PROGRESS_MODE
    if resolved not in PROGRESS_MODES:
        raise ValueError(f"Unknown progress mode {resolved!r}; expected one of {PROGRESS_MODES}")
    return resolved


class ProgressReporter:
    """Receives progress from running work; the base class shows nothing (mode "none")."""

    def __init__(self) -> None:
        # Set by the board when its user stops watching a SLURM wait; the waiting loop then returns early.
        self.stop_requested = threading.Event()

    def evaluation(self, snapshot: PipelineProgress) -> None:
        pass

    def download(self, event: dict) -> None:
        pass

    def close(self) -> None:
        pass


class BarsReporter(ProgressReporter):
    def __init__(self) -> None:
        super().__init__()
        self._evaluation_bars: ProgressBars | None = None
        self._download_bars: dict[str, Any] = {}
        self._downloads = DownloadState()

    def evaluation(self, snapshot: PipelineProgress) -> None:
        if self._evaluation_bars is None:
            self._evaluation_bars = ProgressBars()
        self._evaluation_bars.update(snapshot)

    def download(self, event: dict) -> None:
        from tqdm.auto import tqdm

        self._downloads.apply(event)
        kind = event["kind"]
        if kind not in self._download_bars:
            self._download_bars[kind] = tqdm(
                total=0, desc=f"{kind:10s}", unit="item", position=len(self._download_bars),
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {unit} [{elapsed}]{postfix}",
            )
        bar = self._download_bars[kind]
        done, total = self._downloads.counts(kind)
        bar.total, bar.n = total, done
        if event["status"] == "running":
            bar.set_postfix_str(event["name"] + (f" ({event['step']})" if event.get("step") else ""), refresh=False)
        elif event["status"] == "failed":
            bar.write(f"download failed: {kind} {event['name']}: {event.get('detail', '')}")
        bar.refresh()

    def close(self) -> None:
        if self._evaluation_bars is not None:
            self._evaluation_bars.close()
        for bar in self._download_bars.values():
            bar.close()


class DownloadState:
    """Latest event per download item, in first-seen order."""

    def __init__(self) -> None:
        self.items: OrderedDict[tuple[str, str], dict] = OrderedDict()

    def apply(self, event: dict) -> None:
        self.items[(event["kind"], event["name"])] = dict(event)

    def counts(self, kind: str | None = None) -> tuple[int, int]:
        items = [item for (k, _), item in self.items.items() if kind in (None, k)]
        return sum(item["status"] in DOWNLOAD_DONE for item in items), len(items)


def make_reporter(mode: str) -> ProgressReporter:
    return BarsReporter() if mode == "bars" else ProgressReporter()


def board_available() -> tuple[bool, str]:
    """Whether the board can run here: the textual package and an interactive terminal."""
    try:
        import textual  # noqa: F401  # pylint: disable=unused-import  # only its availability is checked
    except ImportError:
        return False, "the 'textual' package is not installed (pip install textual)"
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False, "stdin/stdout is not an interactive terminal"
    return True, ""


def run_with_progress(
    mode: str,
    work: Callable[[ProgressReporter], T],
    *,
    title: str,
    detachable: bool = False,
) -> T:
    """Run ``work(reporter)`` showing its progress in ``mode``; returns what ``work`` returns.

    ``detachable``: the work only *watches* jobs that run elsewhere (a SLURM wait). Quitting the board then sets
    ``reporter.stop_requested`` so the work can return early. Otherwise (native evaluation, downloads) quitting the
    board only stops displaying it: the work keeps running and its progress continues as bars.
    """
    if mode == "board":
        available, reason = board_available()
        if not available:
            warnings.warn(f"progress board unavailable ({reason}); showing progress bars instead", stacklevel=3)
            mode = "bars"
    if mode != "board":
        reporter = make_reporter(mode)
        try:
            return work(reporter)
        finally:
            reporter.close()

    from spatialarena.progress_board import BoardApp

    app = BoardApp(title=title, detachable=detachable)
    reporter = app.reporter
    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["result"] = work(reporter)
        # pylint: disable-next=broad-exception-caught
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller's thread below
            outcome["error"] = exc
        finally:
            reporter.work_finished(outcome.get("error"))

    worker = threading.Thread(target=target, name="benchmark-progress-work", daemon=True)
    app.on_ready_callback = worker.start
    app.run()
    if worker.ident is None:  # the board never became ready (e.g. failed to start): run the work without it
        worker.start()
    if worker.is_alive():
        if detachable:
            reporter.stop_requested.set()
        else:
            print(f"{title}: board closed; the work continues, progress shown as bars.", file=sys.stderr)
        reporter.switch_to_bars()
        while worker.is_alive():
            worker.join(timeout=0.5)
    reporter.close()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


def group_runs(runs: list[RunProgress], key: Callable[[RunProgress], str]) -> list[dict]:
    """Aggregate runs by ``key``: runs per stage (``group[stage]``) plus finished/total cases per phase
    (``group["inference_cases"]``, ``group["evaluation_cases"]``)."""
    groups: OrderedDict[str, dict] = OrderedDict()
    for run in sorted(runs, key=key):
        group = groups.setdefault(key(run), {"key": key(run), "runs": 0, **dict.fromkeys(STAGES, 0),
                                             "inference_cases": [0, 0], "evaluation_cases": [0, 0],
                                             "failed_cases": 0})
        group["runs"] += 1
        group[run.stage] += 1
        group["inference_cases"][0] += run.inference_done
        group["evaluation_cases"][0] += run.evaluation_done
        group["inference_cases"][1] += run.total_cases
        group["evaluation_cases"][1] += run.total_cases
        group["failed_cases"] += run.failed_cases
    return list(groups.values())


def elapsed_text(started: float) -> str:
    seconds = int(time.monotonic() - started)
    return f"{seconds // 3600:d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"
