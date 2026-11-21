# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Interactive terminal board for evaluation and download progress (progress mode "board").

Views (keys 1-5): Tracks (task | track | SLAM range), Methods, Datasets, Runs (one row per run, filterable with
``f``; the highlighted run's details below the table) and Downloads (one row per dataset / checkpoint / method-code
item). ``q`` closes the board -- while work is still running it asks for a second ``q`` first. See
``spatialarena/progress_ui.py`` for how the board is driven; this module needs the ``textual`` package.
"""

from __future__ import annotations

import time

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Footer, Header, Static, TabbedContent, TabPane

from spatialarena.progress import STAGES, PipelineProgress, RunProgress
from spatialarena.progress_ui import (
    DOWNLOAD_KINDS,
    BarsReporter,
    DownloadState,
    ProgressReporter,
    elapsed_text,
    group_runs,
)

STAGE_STYLE = {"queued": "dim", "inference": "cyan", "evaluation": "magenta", "done": "green", "failed": "bold red"}
DOWNLOAD_STYLE = {"pending": "dim", "running": "cyan", "ok": "green", "already_present": "green",
                  "skipped": "yellow", "failed": "bold red"}
RUN_FILTERS = ("all", "active", "failed", "done")
GROUP_COLUMNS = ("Runs", "Queued", "Inference", "Evaluation", "Done", "Failed", "Inference cases",
                 "Evaluation cases", "Progress")
GROUP_VIEWS = {
    "tracks": ("Track", lambda run: run.track_key),
    "methods": ("Method", lambda run: f"{run.method}/{run.track}" if run.track != "rgb" else run.method),
    "datasets": ("Dataset", lambda run: f"{run.task}/{run.dataset}"),
}


def progress_bar(done: int, total: int, width: int = 20) -> Text:
    fraction = done / total if total else 0.0
    filled = round(fraction * width)
    style = "green" if total and done >= total else "cyan"
    return Text("█" * filled, style=style) + Text("░" * (width - filled), style="dim") + Text(f" {fraction:4.0%}")


def run_progress_fraction(run: RunProgress) -> tuple[int, int]:
    """Progress through both phases: (inference + evaluation cases done, 2 x cases)."""
    return run.inference_done + run.evaluation_done, 2 * run.total_cases


class BoardReporter(ProgressReporter):
    """Forwards progress from the work thread to the board; after the board closes, to progress bars."""

    def __init__(self, app: "BoardApp") -> None:
        super().__init__()
        self.app = app
        self._bars: BarsReporter | None = None
        self._latest_snapshot: PipelineProgress | None = None
        self._downloads: list[dict] = []

    def _to_board(self, method, *args) -> bool:
        if self._bars is not None:
            return False
        try:
            self.app.call_from_thread(method, *args)
            return True
        except RuntimeError:  # the board is not running (not started yet, or closed)
            return False

    def evaluation(self, snapshot: PipelineProgress) -> None:
        self._latest_snapshot = snapshot
        if not self._to_board(self.app.apply_snapshot, snapshot) and self._bars is not None:
            self._bars.evaluation(snapshot)

    def download(self, event: dict) -> None:
        self._downloads.append(event)
        if not self._to_board(self.app.apply_download, event) and self._bars is not None:
            self._bars.download(event)

    def work_finished(self, error: BaseException | None) -> None:
        self._to_board(self.app.mark_finished, error)

    def switch_to_bars(self) -> None:
        """The board closed while work is still running: continue as progress bars from the latest state."""
        self._bars = BarsReporter()
        for event in self._downloads:
            self._bars.download(event)
        if self._latest_snapshot is not None:
            self._bars.evaluation(self._latest_snapshot)

    def close(self) -> None:
        if self._bars is not None:
            self._bars.close()


class BoardApp(App):
    CSS = """
    #summary { height: auto; padding: 0 1; background: $boost; }
    DataTable { height: 1fr; }
    #run-detail { height: auto; max-height: 9; border: round $primary; padding: 0 1; }
    """
    BINDINGS = [
        Binding("1", "show_view('tracks')", "Tracks"),
        Binding("2", "show_view('methods')", "Methods"),
        Binding("3", "show_view('datasets')", "Datasets"),
        Binding("4", "show_view('runs')", "Runs"),
        Binding("5", "show_view('downloads')", "Downloads"),
        Binding("f", "cycle_filter", "Filter runs"),
        Binding("q", "quit_board", "Quit"),
    ]

    def __init__(self, *, title: str, detachable: bool = False) -> None:
        super().__init__()
        self.title = title
        self.detachable = detachable
        self.reporter = BoardReporter(self)
        self.snapshot = PipelineProgress()
        self.downloads = DownloadState()
        self.started = time.monotonic()
        self.finished = False
        self.error: BaseException | None = None
        self.run_filter = "all"
        self._quit_armed = False
        self._shown_runs: list[RunProgress] = []
        self.on_ready_callback = None

    # ---- layout ------------------------------------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(id="summary")
        with TabbedContent(initial="tracks", id="views"):
            for view, (_label, _key) in GROUP_VIEWS.items():
                with TabPane(view.capitalize(), id=view):
                    yield DataTable(id=f"{view}-table", zebra_stripes=True, cursor_type="row")
            with TabPane("Runs", id="runs"):
                yield DataTable(id="runs-table", zebra_stripes=True, cursor_type="row")
                yield Static(id="run-detail")
            with TabPane("Downloads", id="downloads"):
                yield DataTable(id="downloads-table", zebra_stripes=True, cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        for view, (label, _key) in GROUP_VIEWS.items():
            self.query_one(f"#{view}-table", DataTable).add_columns(label, *GROUP_COLUMNS)
        self.query_one("#runs-table", DataTable).add_columns(
            "Method", "Track", "Task", "Dataset", "Range", "Stage", "Inference", "Evaluation", "Failed", "Progress"
        )
        self.query_one("#downloads-table", DataTable).add_columns("Kind", "Name", "Status", "Step", "Detail")
        self.set_interval(1.0, self.refresh_summary)
        self.refresh_all()

    def on_ready(self) -> None:
        if self.on_ready_callback is not None:
            self.on_ready_callback()

    # ---- updates from the work thread --------------------------------------------------------------------------

    def apply_snapshot(self, snapshot: PipelineProgress) -> None:
        self.snapshot = snapshot
        self.refresh_all()

    def apply_download(self, event: dict) -> None:
        self.downloads.apply(event)
        self._fill_downloads()
        self.refresh_summary()

    def mark_finished(self, error: BaseException | None) -> None:
        self.finished, self.error = True, error
        if error is not None:
            self.notify(f"{type(error).__name__}: {error}", title="Failed", severity="error", timeout=30)
        else:
            self.notify("Finished -- press q to close the board.", title="Done", timeout=30)
        self.refresh_summary()

    # ---- rendering ----------------------------------------------------------------------------------------------

    def refresh_all(self) -> None:
        for view, (_label, key) in GROUP_VIEWS.items():
            self._fill_groups(view, key)
        self._fill_runs()
        self._fill_downloads()
        self.refresh_summary()

    def refresh_summary(self) -> None:
        counts = self.snapshot.stage_counts()
        parts = []
        if self.snapshot.runs:
            stage_text = Text(f"Runs {len(self.snapshot.runs)}: ")
            for stage in STAGES:
                stage_text += Text(f"{stage} {counts[stage]}  ", style=STAGE_STYLE[stage] if counts[stage] else "dim")
            parts.append(stage_text)
            parts.append(Text("cases: inference {}/{}  evaluation {}/{}".format(
                *self.snapshot.cases("inference"), *self.snapshot.cases("evaluation"))))
        if self.downloads.items:
            done, total = self.downloads.counts()
            failed = sum(item["status"] == "failed" for item in self.downloads.items.values())
            parts.append(Text(f"downloads {done}/{total}" + (f" ({failed} failed)" if failed else "")))
        status = "failed" if self.error else "finished -- press q to close" if self.finished else "running"
        parts.append(Text(f"elapsed {elapsed_text(self.started)}  status: {status}",
                          style="bold red" if self.error else "bold green" if self.finished else ""))
        self.query_one("#summary", Static).update(Text("  |  ").join(parts))

    @staticmethod
    def _refill(table: DataTable, rows: list[tuple], keys: list[str] | None = None) -> None:
        cursor = table.cursor_row
        table.clear()
        for index, row in enumerate(rows):
            table.add_row(*row, key=keys[index] if keys else None)
        if rows:
            table.move_cursor(row=min(max(cursor, 0), len(rows) - 1))

    def _fill_groups(self, view: str, key) -> None:
        rows = []
        for group in group_runs(self.snapshot.runs, key):
            rows.append((
                group["key"], str(group["runs"]),
                *(Text(str(group[stage]), style=STAGE_STYLE[stage] if group[stage] else "dim") for stage in STAGES),
                "{}/{}".format(*group["inference_cases"]), "{}/{}".format(*group["evaluation_cases"]),
                progress_bar(group["inference_cases"][0] + group["evaluation_cases"][0],
                             group["inference_cases"][1] + group["evaluation_cases"][1]),
            ))
        self._refill(self.query_one(f"#{view}-table", DataTable), rows)

    def _filtered_runs(self) -> list[RunProgress]:
        if self.run_filter == "active":
            return [run for run in self.snapshot.runs if run.stage in ("queued", "inference", "evaluation")]
        if self.run_filter in ("failed", "done"):
            return [run for run in self.snapshot.runs if run.stage == self.run_filter]
        return list(self.snapshot.runs)

    def _fill_runs(self) -> None:
        self._shown_runs = self._filtered_runs()
        rows = [
            (run.method, run.track, run.task, run.dataset, run.slam_range or "-",
             Text(run.stage, style=STAGE_STYLE[run.stage]),
             f"{run.inference_done}/{run.total_cases}", f"{run.evaluation_done}/{run.total_cases}",
             Text(str(run.failed_cases), style="bold red" if run.failed_cases else "dim"),
             progress_bar(*run_progress_fraction(run), width=12))
            for run in self._shown_runs
        ]
        table = self.query_one("#runs-table", DataTable)
        self._refill(table, rows, keys=[str(index) for index in range(len(rows))])
        self.query_one("#runs", TabPane).border_title = f"filter: {self.run_filter}"
        self._show_run_detail(table.cursor_row)

    def _fill_downloads(self) -> None:
        rows = [
            (item["kind"], item["name"], Text(item["status"], style=DOWNLOAD_STYLE.get(item["status"], "")),
             item.get("step") or "", (item.get("detail") or "")[:120])
            for item in sorted(self.downloads.items.values(),
                               key=lambda item: (DOWNLOAD_KINDS.index(item["kind"]) if item["kind"] in DOWNLOAD_KINDS
                                                 else 99))
        ]
        self._refill(self.query_one("#downloads-table", DataTable), rows)

    def _show_run_detail(self, row: int) -> None:
        detail = self.query_one("#run-detail", Static)
        if not (0 <= row < len(self._shown_runs)):
            detail.update(Text(f"{len(self._shown_runs)} runs shown (filter: {self.run_filter}; press f to change)",
                               style="dim"))
            return
        run = self._shown_runs[row]
        lines = [
            Text(run.label, style="bold"),
            Text(f"stage: {run.stage}   inference {run.inference_done}/{run.total_cases}   "
                 f"evaluation {run.evaluation_done}/{run.total_cases}   failed cases {run.failed_cases}"),
            Text(f"jobs: inference {run.infer_job_id or '-'}  evaluation {run.eval_job_id or '-'}"),
            Text(f"report dir: {run.report_dir or '-'}"),
        ]
        if run.note:
            lines.append(Text(f"note: {run.note}", style="yellow"))
        detail.update(Text("\n").join(lines))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id == "runs-table":
            self._show_run_detail(event.cursor_row)

    # ---- actions ------------------------------------------------------------------------------------------------

    def action_show_view(self, view: str) -> None:
        self.query_one("#views", TabbedContent).active = view

    def action_cycle_filter(self) -> None:
        self.run_filter = RUN_FILTERS[(RUN_FILTERS.index(self.run_filter) + 1) % len(RUN_FILTERS)]
        self.action_show_view("runs")
        self._fill_runs()

    def action_quit_board(self) -> None:
        if self.finished or self._quit_armed:
            self.exit()
            return
        self._quit_armed = True
        message = (
            "Press q again to stop watching -- the SLURM jobs keep running; check later with progress()/collect()."
            if self.detachable
            else "Still running -- press q again to close the board; the work continues with progress bars."
        )
        self.notify(message, title="Quit?", severity="warning", timeout=8)
        self.set_timer(8, self._disarm_quit)

    def _disarm_quit(self) -> None:
        self._quit_armed = False
