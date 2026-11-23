# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Interactive terminal board for evaluation and download progress.

This is progress mode "board". Views (keys 1-5): Tracks (task | track | SLAM
range), Methods, Datasets, Runs (one row per run, filterable with `f`; the
highlighted run's details below the table) and Downloads (one row per dataset
/ checkpoint / method-code item). `q` closes the board -- while work is still
running it asks for a second `q` first. See `spatialarena/progress_ui.py` for
how the board is driven; this module needs the `textual` package.
"""

from __future__ import annotations

import time

# Aliased: `text` and `app` are common local and attribute names here.
from rich import text as rich_text
from textual import app as textual_app
from textual import binding
from textual import widgets

from spatialarena import progress
from spatialarena import progress_ui

STAGE_STYLE = {
    "queued": "dim",
    "inference": "cyan",
    "evaluation": "magenta",
    "done": "green",
    "failed": "bold red"
}
DOWNLOAD_STYLE = {
    "pending": "dim",
    "running": "cyan",
    "ok": "green",
    "already_present": "green",
    "skipped": "yellow",
    "failed": "bold red"
}
RUN_FILTERS = ("all", "active", "failed", "done")
GROUP_COLUMNS = ("Runs", "Queued", "Inference", "Evaluation", "Done", "Failed",
                 "Inference cases", "Evaluation cases", "Progress")


def _method_key(run: progress.RunProgress) -> str:
    """Returns the Methods view row of a run: the method, with its track."""
    if run.track != "rgb":
        return f"{run.method}/{run.track}"
    return run.method


GROUP_VIEWS = {
    "tracks": ("Track", lambda run: run.track_key),
    "methods": ("Method", _method_key),
    "datasets": ("Dataset", lambda run: f"{run.task}/{run.dataset}"),
}


def progress_bar(done: int, total: int, width: int = 20) -> rich_text.Text:
    """Returns a text progress bar with the percentage."""
    fraction = done / total if total else 0.0
    filled = round(fraction * width)
    style = "green" if total and done >= total else "cyan"
    return (rich_text.Text("█" * filled, style=style) +
            rich_text.Text("░" * (width - filled), style="dim") +
            rich_text.Text(f" {fraction:4.0%}"))


def run_progress_fraction(run: progress.RunProgress) -> tuple[int, int]:
    """Returns the progress through both phases of a run.

    Args:
        run: The run.

    Returns:
        (inference + evaluation cases done, 2 x cases).
    """
    return run.inference_done + run.evaluation_done, 2 * run.total_cases


class BoardReporter(progress_ui.ProgressReporter):
    """Forwards progress from the work thread to the board.

    After the board closes, progress goes to progress bars.

    Attributes:
        app: The board.
    """

    def __init__(self, app: BoardApp) -> None:
        super().__init__()
        self.app = app
        self._bars: progress_ui.BarsReporter | None = None
        self._latest_snapshot: progress.PipelineProgress | None = None
        self._downloads: list[dict] = []

    def _to_board(self, method, *args) -> bool:
        """Calls a board method in the board's thread; False if it cannot."""
        if self._bars is not None:
            return False
        try:
            self.app.call_from_thread(method, *args)
            return True
        # The board is not running (not started yet, or closed).
        except RuntimeError:
            return False

    def evaluation(self, snapshot: progress.PipelineProgress) -> None:
        self._latest_snapshot = snapshot
        shown = self._to_board(self.app.apply_snapshot, snapshot)
        if not shown and self._bars is not None:
            self._bars.evaluation(snapshot)

    def download(self, event: dict) -> None:
        self._downloads.append(event)
        shown = self._to_board(self.app.apply_download, event)
        if not shown and self._bars is not None:
            self._bars.download(event)

    def work_finished(self, error: BaseException | None) -> None:
        """Tells the board the work ended, with its error if it failed."""
        self._to_board(self.app.mark_finished, error)

    def switch_to_bars(self) -> None:
        """Continues as progress bars from the latest state.

        Called when the board closed while work is still running.
        """
        self._bars = progress_ui.BarsReporter()
        for event in self._downloads:
            self._bars.download(event)
        if self._latest_snapshot is not None:
            self._bars.evaluation(self._latest_snapshot)

    def close(self) -> None:
        if self._bars is not None:
            self._bars.close()


def _refill_table(table: widgets.DataTable,
                  rows: list[tuple],
                  keys: list[str] | None = None) -> None:
    """Replaces the rows of a table, keeping the cursor row where possible."""
    cursor = table.cursor_row
    table.clear()
    for index, row in enumerate(rows):
        table.add_row(*row, key=keys[index] if keys else None)
    if rows:
        table.move_cursor(row=min(max(cursor, 0), len(rows) - 1))


def _download_sort_key(item: dict) -> int:
    """Orders download items by kind, unknown kinds last."""
    if item["kind"] in progress_ui.DOWNLOAD_KINDS:
        return progress_ui.DOWNLOAD_KINDS.index(item["kind"])
    return 99


class BoardApp(textual_app.App):
    """The board app.

    Attributes:
        title: Title shown in the header.
        detachable: Whether the work only watches jobs that run elsewhere.
        reporter: The reporter the work sends progress to.
        snapshot: The latest evaluation progress.
        downloads: The latest event of each download item.
        started: `time.monotonic()` at start.
        finished: Whether the work ended.
        error: The work's error, if it failed.
        run_filter: Filter of the Runs view, one of `RUN_FILTERS`.
        on_ready_callback: Called once the board is ready (starts the work).
    """

    CSS = """
    #summary { height: auto; padding: 0 1; background: $boost; }
    DataTable { height: 1fr; }
    #run-detail {
        height: auto; max-height: 9; border: round $primary; padding: 0 1;
    }
    """
    BINDINGS = [
        binding.Binding("1", "show_view('tracks')", "Tracks"),
        binding.Binding("2", "show_view('methods')", "Methods"),
        binding.Binding("3", "show_view('datasets')", "Datasets"),
        binding.Binding("4", "show_view('runs')", "Runs"),
        binding.Binding("5", "show_view('downloads')", "Downloads"),
        binding.Binding("f", "cycle_filter", "Filter runs"),
        binding.Binding("q", "quit_board", "Quit"),
    ]

    def __init__(self, *, title: str, detachable: bool = False) -> None:
        super().__init__()
        self.title = title
        self.detachable = detachable
        self.reporter = BoardReporter(self)
        self.snapshot = progress.PipelineProgress()
        self.downloads = progress_ui.DownloadState()
        self.started = time.monotonic()
        self.finished = False
        self.error: BaseException | None = None
        self.run_filter = "all"
        self._quit_armed = False
        self._shown_runs: list[progress.RunProgress] = []
        self.on_ready_callback = None

    # ---- Layout. ------------------------------------------------------------

    def compose(self) -> textual_app.ComposeResult:
        """Builds the summary line, the view tabs and the footer."""
        yield widgets.Header(show_clock=True)
        yield widgets.Static(id="summary")
        with widgets.TabbedContent(initial="tracks", id="views"):
            for view, (_label, _key) in GROUP_VIEWS.items():
                with widgets.TabPane(view.capitalize(), id=view):
                    yield widgets.DataTable(id=f"{view}-table",
                                            zebra_stripes=True,
                                            cursor_type="row")
            with widgets.TabPane("Runs", id="runs"):
                yield widgets.DataTable(id="runs-table",
                                        zebra_stripes=True,
                                        cursor_type="row")
                yield widgets.Static(id="run-detail")
            with widgets.TabPane("Downloads", id="downloads"):
                yield widgets.DataTable(id="downloads-table",
                                        zebra_stripes=True,
                                        cursor_type="row")
        yield widgets.Footer()

    def on_mount(self) -> None:
        """Adds the table columns and starts the summary refresh."""
        for view, (label, _key) in GROUP_VIEWS.items():
            table = self.query_one(f"#{view}-table", widgets.DataTable)
            table.add_columns(label, *GROUP_COLUMNS)
        runs_table = self.query_one("#runs-table", widgets.DataTable)
        runs_table.add_columns("Method", "Track", "Task", "Dataset", "Range",
                               "Stage", "Inference", "Evaluation", "Failed",
                               "Progress")
        downloads_table = self.query_one("#downloads-table", widgets.DataTable)
        downloads_table.add_columns("Kind", "Name", "Status", "Step", "Detail")
        self.set_interval(1.0, self.refresh_summary)
        self.refresh_all()

    def on_ready(self) -> None:
        """Calls `on_ready_callback` once the board is ready."""
        if self.on_ready_callback is not None:
            self.on_ready_callback()

    # ---- Updates from the work thread. --------------------------------------

    def apply_snapshot(self, snapshot: progress.PipelineProgress) -> None:
        """Shows a new evaluation progress snapshot."""
        self.snapshot = snapshot
        self.refresh_all()

    def apply_download(self, event: dict) -> None:
        """Shows a download event."""
        self.downloads.apply(event)
        self._fill_downloads()
        self.refresh_summary()

    def mark_finished(self, error: BaseException | None) -> None:
        """Shows that the work ended, with its error if it failed."""
        self.finished = True
        self.error = error
        if error is not None:
            self.notify(f"{type(error).__name__}: {error}",
                        title="Failed",
                        severity="error",
                        timeout=30)
        else:
            self.notify("Finished -- press q to close the board.",
                        title="Done",
                        timeout=30)
        self.refresh_summary()

    # ---- Rendering. ---------------------------------------------------------

    def refresh_all(self) -> None:
        """Refills every view and the summary line."""
        for view, (_label, key) in GROUP_VIEWS.items():
            self._fill_groups(view, key)
        self._fill_runs()
        self._fill_downloads()
        self.refresh_summary()

    def refresh_summary(self) -> None:
        """Rewrites the summary line: stages, cases, downloads, status."""
        counts = self.snapshot.stage_counts()
        parts = []
        if self.snapshot.runs:
            stage_text = rich_text.Text(f"Runs {len(self.snapshot.runs)}: ")
            for stage in progress.STAGES:
                style = STAGE_STYLE[stage] if counts[stage] else "dim"
                stage_text += rich_text.Text(f"{stage} {counts[stage]}  ",
                                             style=style)
            parts.append(stage_text)
            infer_done, infer_total = self.snapshot.cases("inference")
            eval_done, eval_total = self.snapshot.cases("evaluation")
            parts.append(
                rich_text.Text(f"cases: inference {infer_done}/{infer_total}  "
                               f"evaluation {eval_done}/{eval_total}"))
        if self.downloads.items:
            done, total = self.downloads.counts()
            failed = sum(item["status"] == "failed"
                         for item in self.downloads.items.values())
            downloads = f"downloads {done}/{total}"
            if failed:
                downloads += f" ({failed} failed)"
            parts.append(rich_text.Text(downloads))
        if self.error:
            status = "failed"
            style = "bold red"
        elif self.finished:
            status = "finished -- press q to close"
            style = "bold green"
        else:
            status = "running"
            style = ""
        elapsed = progress_ui.elapsed_text(self.started)
        parts.append(
            rich_text.Text(f"elapsed {elapsed}  status: {status}", style=style))
        summary = self.query_one("#summary", widgets.Static)
        summary.update(rich_text.Text("  |  ").join(parts))

    def _fill_groups(self, view: str, key) -> None:
        """Refills a grouped view (Tracks, Methods or Datasets)."""
        rows = []
        for group in progress_ui.group_runs(self.snapshot.runs, key):
            stage_cells = []
            for stage in progress.STAGES:
                style = STAGE_STYLE[stage] if group[stage] else "dim"
                stage_cells.append(
                    rich_text.Text(str(group[stage]), style=style))
            infer_done, infer_total = group["inference_cases"]
            eval_done, eval_total = group["evaluation_cases"]
            rows.append((
                group["key"],
                str(group["runs"]),
                *stage_cells,
                f"{infer_done}/{infer_total}",
                f"{eval_done}/{eval_total}",
                progress_bar(infer_done + eval_done, infer_total + eval_total),
            ))
        _refill_table(self.query_one(f"#{view}-table", widgets.DataTable), rows)

    def _filtered_runs(self) -> list[progress.RunProgress]:
        """Returns the runs that pass the Runs view filter."""
        if self.run_filter == "active":
            return [
                run for run in self.snapshot.runs
                if run.stage in ("queued", "inference", "evaluation")
            ]
        if self.run_filter in ("failed", "done"):
            return [
                run for run in self.snapshot.runs
                if run.stage == self.run_filter
            ]
        return list(self.snapshot.runs)

    def _fill_runs(self) -> None:
        """Refills the Runs view and the run details."""
        self._shown_runs = self._filtered_runs()
        rows = []
        for run in self._shown_runs:
            failed_style = "bold red" if run.failed_cases else "dim"
            rows.append((
                run.method,
                run.track,
                run.task,
                run.dataset,
                run.slam_range or "-",
                rich_text.Text(run.stage, style=STAGE_STYLE[run.stage]),
                f"{run.inference_done}/{run.total_cases}",
                f"{run.evaluation_done}/{run.total_cases}",
                rich_text.Text(str(run.failed_cases), style=failed_style),
                progress_bar(*run_progress_fraction(run), width=12),
            ))
        table = self.query_one("#runs-table", widgets.DataTable)
        _refill_table(table,
                      rows,
                      keys=[str(index) for index in range(len(rows))])
        runs_pane = self.query_one("#runs", widgets.TabPane)
        runs_pane.border_title = f"filter: {self.run_filter}"
        self._show_run_detail(table.cursor_row)

    def _fill_downloads(self) -> None:
        """Refills the Downloads view."""
        rows = []
        items = sorted(self.downloads.items.values(), key=_download_sort_key)
        for item in items:
            status_style = DOWNLOAD_STYLE.get(item["status"], "")
            rows.append((
                item["kind"],
                item["name"],
                rich_text.Text(item["status"], style=status_style),
                item.get("step") or "",
                (item.get("detail") or "")[:120],
            ))
        _refill_table(self.query_one("#downloads-table", widgets.DataTable),
                      rows)

    def _show_run_detail(self, row: int) -> None:
        """Shows the details of a Runs view row, or a hint if none."""
        detail = self.query_one("#run-detail", widgets.Static)
        if not 0 <= row < len(self._shown_runs):
            detail.update(
                rich_text.Text(
                    f"{len(self._shown_runs)} runs shown (filter: "
                    f"{self.run_filter}; press f to change)",
                    style="dim"))
            return
        run = self._shown_runs[row]
        lines = [
            rich_text.Text(run.label, style="bold"),
            rich_text.Text(
                f"stage: {run.stage}   inference "
                f"{run.inference_done}/{run.total_cases}   evaluation "
                f"{run.evaluation_done}/{run.total_cases}   failed cases "
                f"{run.failed_cases}"),
            rich_text.Text(
                f"jobs: inference {run.infer_job_id or '-'}  evaluation "
                f"{run.eval_job_id or '-'}"),
            rich_text.Text(f"report dir: {run.report_dir or '-'}"),
        ]
        if run.note:
            lines.append(rich_text.Text(f"note: {run.note}", style="yellow"))
        detail.update(rich_text.Text("\n").join(lines))

    def on_data_table_row_highlighted(
            self, event: widgets.DataTable.RowHighlighted) -> None:
        """Shows the details of the highlighted Runs view row."""
        if event.data_table.id == "runs-table":
            self._show_run_detail(event.cursor_row)

    # ---- Actions. -----------------------------------------------------------

    def action_show_view(self, view: str) -> None:
        """Switches to a view."""
        self.query_one("#views", widgets.TabbedContent).active = view

    def action_cycle_filter(self) -> None:
        """Switches to the Runs view with the next run filter."""
        next_index = (RUN_FILTERS.index(self.run_filter) + 1) % len(RUN_FILTERS)
        self.run_filter = RUN_FILTERS[next_index]
        self.action_show_view("runs")
        self._fill_runs()

    def action_quit_board(self) -> None:
        """Closes the board; while work runs, only on a second press."""
        if self.finished or self._quit_armed:
            self.exit()
            return
        self._quit_armed = True
        if self.detachable:
            message = ("Press q again to stop watching -- the SLURM jobs keep "
                       "running; check later with progress()/collect().")
        else:
            message = ("Still running -- press q again to close the board; "
                       "the work continues with progress bars.")
        self.notify(message, title="Quit?", severity="warning", timeout=8)
        self.set_timer(8, self._disarm_quit)

    def _disarm_quit(self) -> None:
        self._quit_armed = False
