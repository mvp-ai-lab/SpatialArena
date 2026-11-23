# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The user-facing evaluation API, `BenchmarkEvaluator`.

The API reference (docs/benchmark/evaluator_quickstart.md) documents the full
contract this module implements: discover, select, plan and validate
(read-only), `configure()` (YAML or JSON config files), `evaluate()`, `wait()`
and `collect()` for `job_system="slurm"` or `"native"`, and `export_report()`.

Typical use:

    bench = BenchmarkEvaluator()
    bench.configure("configs/benchmark/full_monodepth.yaml")
    report = bench.wait(bench.evaluate())
    bench.export_report()
"""

from __future__ import annotations

import copy
import dataclasses
import datetime
import json
import os
import pathlib
import subprocess
import time
from typing import Any

from spatialarena import methods as method_registry
from spatialarena import paths
from spatialarena import script_generator
from spatialarena import slurm_utils
from spatialarena import task_track_matrix
from spatialarena.eval import dataset_registry
# Tests patch `describe_benchmark_scenes` and `sacct_states` on this module, so
# the evaluator calls them through this module's own bindings.
from spatialarena.eval.dataset_registry import describe_benchmark_scenes
from spatialarena.slurm_utils import sacct_states

# Canonical shared locations for downloaded checkpoints and vendored method
# source code. Both live directly under the shared benchmark aux root
# (DEFAULT_BENCHMARK_ROOT/_checkpoints, DEFAULT_BENCHMARK_ROOT/_model_codes), so
# automatic downloads always land somewhere reusable across runs and evaluator
# instances, not under the repo checkout or a run-local temp dir. Both are also
# per-instance attributes that can be overridden (constructor arguments, or the
# CHECKPOINT_ROOT and METHOD_CODE_ROOT environment variables via configure());
# these module constants are only the defaults.
DEFAULT_CHECKPOINT_ROOT = (dataset_registry.DEFAULT_BENCHMARK_ROOT /
                           "_checkpoints")
DEFAULT_HF_HOME = DEFAULT_CHECKPOINT_ROOT / "huggingface"
DEFAULT_METHOD_CODE_ROOT = (dataset_registry.DEFAULT_BENCHMARK_ROOT /
                            "_model_codes")

# Back-compat names: spatialarena/download.py falls back to these when
# download_all_supported_models() is called directly (outside a
# BenchmarkEvaluator instance) without explicit shared_checkpoint_root or
# shared_hf_home arguments.
SHARED_CHECKPOINT_ROOT = DEFAULT_CHECKPOINT_ROOT
SHARED_HF_HOME = DEFAULT_HF_HOME

# Generated job scripts (and their SLURM logs) of every evaluation:
# <benchmark_root>/_eval_job_scripts/<name>/.
DEFAULT_JOB_SCRIPT_DIRNAME = "_eval_job_scripts"
DEFAULT_BENCHMARK_EVAL_ROOT = pathlib.Path(
    f"{paths.SPATIALARENA_ROOT}/benchmarks/_evaluation")

# Single source of truth for which SLURM "suite" a task's real runs use, and
# therefore the canonical artifact and report location and the
# KEEP_PREDICTIONS=1 guard (see run_method_case_list.sh's `full-*` handling).
TASK_SUITE = {
    "MonoDepth": "full-depth",
    "VideoDepth": "full-depth",
    "MVRecon": "full-depth",
    "SfM": "full-pose",
    "SLAM": "full-slam",
}
# full-slam runs one SLAM range track per experiment ("short": stride 10,
# "long": stride 3; see spatialarena/slam_protocol.py); a SLAM work item's
# "slam_range" selects it.
DEFAULT_SLAM_RANGE = "short"

# The CASE_PARALLELISM defaults of run_method_benchmark_infer.sh and _eval.sh
# (4 for GPU inference, 6 for CPU evaluation; the comments in their #SBATCH
# headers explain why). Used when the caller does not override
# case_parallelism.
DEFAULT_INFER_PARALLELISM = 4
DEFAULT_EVAL_PARALLELISM = 6

_VALID_JOB_SYSTEMS = ("slurm", "native")
_CUSTOM_ADAPTER_REQUIRED_FIELDS = ("name", "track", "command")


@dataclasses.dataclass
class ValidationIssue:
    """One problem found by `BenchmarkEvaluator.validate()`.

    Attributes:
        level: "error", "warning" or "info".
        scope: "dataset", "model" or "run".
        message: A human-readable description of the problem.
        path: The file or folder the issue is about, if any.
    """
    level: str
    scope: str
    message: str
    path: str | None = None

    def to_dict(self) -> dict:
        """Returns the issue as a JSON-serializable dict."""
        return {
            "level": self.level,
            "scope": self.scope,
            "message": self.message,
            "path": self.path
        }


@dataclasses.dataclass
class ValidationReport:
    """The result of `BenchmarkEvaluator.validate()`.

    Attributes:
        ok: Whether no issue has level "error".
        issues: Every issue found.
        summary: Error, warning and run counts.
        selected_runs: The runs the plan would evaluate.
    """
    ok: bool
    issues: list[ValidationIssue] = dataclasses.field(default_factory=list)
    summary: dict = dataclasses.field(default_factory=dict)
    selected_runs: list[dict] | None = None

    def to_dict(self) -> dict:
        """Returns the report as a JSON-serializable dict."""
        return {
            "ok": self.ok,
            "issues": [issue.to_dict() for issue in self.issues],
            "summary": self.summary,
            "selected_runs": self.selected_runs,
        }


@dataclasses.dataclass
class EvaluationPlan:
    """The resolved selection of one evaluation, from `plan()`.

    Attributes:
        config: A snapshot of the evaluator's configuration.
        selected_datasets: The dataset selectors.
        selected_models: The model selectors.
        selected_runs: The (task, dataset, model, track) work items to run.
        output_root: Where the run's scripts, plan and SLURM logs go.
    """
    config: dict
    selected_datasets: list[dict]
    selected_models: list[dict]
    selected_runs: list[dict]
    output_root: pathlib.Path

    def to_dict(self) -> dict:
        """Returns the plan as a JSON-serializable dict."""
        return {
            "config": self.config,
            "selected_datasets": self.selected_datasets,
            "selected_models": self.selected_models,
            "selected_runs": self.selected_runs,
            "output_root": str(self.output_root),
        }


@dataclasses.dataclass
class EvaluationReport:
    """The state or result of one evaluation.

    `evaluate()` returns it with the submitted jobs in `summary`; `collect()`
    returns it with the scene and dataset results once every job finished.

    Attributes:
        ok: Whether no case failed.
        config: A snapshot of the evaluator's configuration.
        plan: The plan that was evaluated.
        scene_results: One row per evaluated scene.
        dataset_results: Passed, failed and skipped counts per dataset.
        summary: The status, run id, jobs and totals.
        report_dir: The run's output root.
    """
    ok: bool
    config: dict
    plan: EvaluationPlan
    scene_results: list[dict] = dataclasses.field(default_factory=list)
    dataset_results: list[dict] = dataclasses.field(default_factory=list)
    summary: dict = dataclasses.field(default_factory=dict)
    report_dir: pathlib.Path | None = None

    def to_dict(self) -> dict:
        """Returns the report as a JSON-serializable dict."""
        if self.report_dir:
            report_dir = str(self.report_dir)
        else:
            report_dir = None
        return {
            "ok": self.ok,
            "config": self.config,
            "plan": self.plan.to_dict(),
            "scene_results": self.scene_results,
            "dataset_results": self.dataset_results,
            "summary": self.summary,
            "report_dir": report_dir,
        }


def _ensure_item_list(items: Any, *, what: str) -> list[dict]:
    """Returns a copy of a list of selector dicts, checking its type."""
    if isinstance(items, dict):
        raise TypeError(
            f"select_{what}() requires a list of items, got a single dict")
    items = list(items)
    for item in items:
        if not isinstance(item, dict):
            raise TypeError(f"Each {what[:-1]} selector must be a dict, "
                            f"got {type(item).__name__}")
    return [dict(item) for item in items]  # A defensive copy per item.


def _slurm_resources(value: Any, *, where: str) -> dict[str, dict[str, str]]:
    """Checks a `slurm_resources` mapping of phase to #SBATCH options.

    Args:
        value: The mapping; its phases are 'infer' and 'eval'.
        where: Where the mapping came from, for error messages.

    Returns:
        A copy of the mapping.

    Raises:
        TypeError: `value` is not a dict.
        ValueError: `value` names a phase other than 'infer' and 'eval'.
    """
    if not isinstance(value, dict):
        raise TypeError(
            f"{where}: slurm_resources must map 'infer' / 'eval' to #SBATCH "
            f"options, got {value!r}")
    unknown = set(value) - {"infer", "eval"}
    if unknown:
        raise ValueError(
            f"{where}: slurm_resources takes the phases 'infer' and 'eval', "
            f"not {sorted(unknown)}")
    return {phase: dict(options) for phase, options in value.items()}


def merged_slurm_resources(*levels: dict | None) -> dict[str, dict[str, str]]:
    """Merges per-phase #SBATCH overrides, later levels on top.

    The levels are the config's, then a dataset item's (e.g. light SLAM
    evaluation jobs in an all-task config), then a model item's (e.g. more
    memory for one method). Options are merged one by one within a phase.

    Args:
        *levels: `slurm_resources` mappings, lowest precedence first; None
            entries are skipped.

    Returns:
        The merged mapping of phase to #SBATCH options.
    """
    merged: dict[str, dict[str, str]] = {}
    for level in levels:
        for phase, options in (level or {}).items():
            phase_options = dict(merged.get(phase, {}))
            phase_options.update(options)
            merged[phase] = phase_options
    return merged


def _is_custom_model_item(item: dict) -> bool:
    """Returns whether a model selector is a custom adapter (has a command)."""
    return "command" in item


def _string_env(env: dict) -> dict[str, str]:
    """Returns a work item's `env:` mapping with string keys and values."""
    return {str(key): str(value) for key, value in env.items()}


def _custom_adapter_env(work_item: dict) -> dict[str, str] | None:
    """Builds the `export VAR=value` lines a work item's wrapper script sets.

    A custom-adapter work item (one with a "command" key; see select_models()'s
    custom-adapter item in the API reference,
    docs/benchmark/evaluator_quickstart.md) needs these set before the engine
    runs.

    Args:
        work_item: One work item of the plan.

    Returns:
        The variables to export, or None for a registered method without an
        explicit per-model `env:` override (the registry already supplies
        everything).
    """
    if not _is_custom_model_item(work_item):
        # Temporary port of PR #21 (eval_refactor_hsy, commit d88025f4):
        # registered methods did not support a per-model `env:` override at
        # all (only custom-adapter items did, below); this lets a config set
        # e.g. `env: {SPATIAL_POLICY: shared-pad}` on a registered method's
        # models[] entry too.
        env = {}
        if isinstance(work_item.get("env"), dict):
            env = _string_env(work_item["env"])
        return env or None
    env: dict[str, str] = {"METHOD_COMMAND": str(work_item["command"])}
    for doc_key, env_var in (
        ("resolution", "RESOLUTION"),
        ("image_load_resolution", "IMAGE_LOAD_RESOLUTION"),
        ("pixel_limit", "PIXEL_LIMIT"),
        ("conf_thresh", "CONF_THRESH"),
        ("depth_conf_thresh", "DEPTH_CONF_THRESH"),
    ):
        if doc_key in work_item:
            env[env_var] = str(work_item[doc_key])
    if "code_path" in work_item:
        env["_CUSTOM_METHOD_CODE_PATH"] = str(work_item["code_path"])
        env["METHOD_CODE_ENV"] = "_CUSTOM_METHOD_CODE_PATH"
    if "ckpt_path" in work_item:
        env["_CUSTOM_METHOD_CKPT_PATH"] = str(work_item["ckpt_path"])
        env["METHOD_CKPT_ENV"] = "_CUSTOM_METHOD_CKPT_PATH"
    if isinstance(work_item.get("env"), dict):
        env.update(_string_env(work_item["env"]))
    return env


# run_method_case_list.sh names SUMMARY_TSV and CASE_FAILURES_TSV after
# CASE_FILE's basename (SUMMARY_SUFFIX="_$(basename "$CASE_FILE" .txt)") UNLESS
# that basename matches the suite's own canonical default case-file name; see
# run_method_case_list.sh's "Name summary/case_failures TSVs after CASE_FILE"
# comment. Every case file this pipeline generates is named after its work item
# (script_generator.run_label), which never matches either default, so the
# suffix always applies in practice.
_DEFAULT_CASE_FILE_BASENAME = {
    "full-depth": "full_standard_depth_cases.txt",
    "full-pose": "full_standard_sfm_cases.txt",
}


def _report_tsv_paths(job: dict) -> tuple[pathlib.Path, pathlib.Path]:
    """Returns the summary and case-failures TSV paths of one run."""
    report_dir = pathlib.Path(job["report_dir"])
    case_file_name = pathlib.Path(job["case_file"]).name
    suite = job["work_item"].get("suite", "")
    default_basename = _DEFAULT_CASE_FILE_BASENAME.get(suite)
    if default_basename and case_file_name == default_basename:
        suffix = ""
    else:
        suffix = f"_{pathlib.Path(job['case_file']).stem}"
    return (report_dir / f"summary{suffix}.tsv",
            report_dir / f"case_failures{suffix}.tsv")


def _read_tsv_data_rows(path: pathlib.Path) -> list[list[str]]:
    """Splits a summary*.tsv or case_failures*.tsv into field rows.

    Skips the header line run_method_case_list.sh always writes first
    (`printf "method\\ttask\\tdataset\\t..." > "$SUMMARY_TSV"` before any case
    runs); without this, the header itself would be read as a scene result.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.split("\t") for line in lines[1:] if line]


def _run_phase_with_progress(
    script: pathlib.Path,
    parallelism: int | None,
    run: "RunProgress",
    stage: str,
    tsv_job: dict,
    progress: "PipelineProgress",
    reporter: "ProgressReporter",
) -> int:
    """Runs one phase's wrapper script in native mode, reporting progress.

    While the script runs, `run`'s `<stage>_done` is refreshed from the run's
    summary TSV every 2 s.

    Args:
        script: The phase's wrapper script.
        parallelism: A CASE_PARALLELISM override, or None for the default.
        run: The run's progress entry.
        stage: "inference" or "evaluation".
        tsv_job: The job dict `_report_tsv_paths()` reads.
        progress: The pipeline progress to report.
        reporter: Where progress is shown.

    Returns:
        The script's exit code.
    """
    from spatialarena import progress as pipeline_progress

    env = os.environ.copy()
    if parallelism is not None:
        env["CASE_PARALLELISM"] = str(parallelism)
    run.stage = stage
    process = subprocess.Popen(["bash", str(script)], env=env)
    while True:
        try:
            returncode = process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            returncode = None
        rows, failures = pipeline_progress.finished_case_rows(tsv_job)
        setattr(run, f"{stage}_done", min(rows + failures, run.total_cases))
        run.failed_cases = failures
        reporter.evaluation(progress)
        if returncode is not None:
            return returncode


# The status (last column, prediction_retention) of a summary*.tsv row written
# by run_method_case_list.sh.
_PASSED_ROW_STATUSES = frozenset(
    {"kept", "removed_after_eval", "reused_existing_eval"})


def _summary_row_status(fields: list[str]) -> tuple[str, str | None]:
    """Classifies one summary*.tsv row.

    Only scored cases pass: a missing prediction in the evaluation phase, or a
    case still marked inference_* after it, failed; skipped_* rows (not
    applicable, missing inputs, degenerate alignment) are skipped. Rows without
    the status column (older TSVs) count as passed.

    Args:
        fields: The row's fields.

    Returns:
        ("passed" | "failed" | "skipped", note).
    """
    status = fields[8].strip() if len(fields) > 8 else ""
    if not status or status in _PASSED_ROW_STATUSES:
        return "passed", None
    if status.startswith("skipped_"):
        return "skipped", status
    if status.startswith("inference_"):
        return "failed", f"{status}: not evaluated"
    return "failed", status


def _read_job_results(
    job: dict,
    job_ok: bool,
    bucket: dict[str, int],
    *,
    infer_note: str = "",
    eval_note: str = "",
) -> list[dict]:
    """Reads one run's summary and case-failures TSVs into scene-result rows.

    Shared by collect() (job_system="slurm", polled via sacct) and
    _evaluate_native() (job_system="native", run synchronously): the report
    folder layout and the row format are the same, only how "did this run
    finish ok" is determined differs between the two callers.

    Args:
        job: The run's job record.
        job_ok: Whether the run's jobs finished successfully.
        bucket: Passed, failed and skipped counts of the run's dataset,
            incremented as rows are read.
        infer_note: The inference job's state, for runs without output.
        eval_note: The evaluation job's state, for runs without output.

    Returns:
        The scene-result rows.
    """
    task = job["work_item"]["task"]
    dataset = job["work_item"]["dataset"]
    if not job.get("report_dir") or not job.get("case_file"):
        # The dispatcher could not generate or submit this run, so it has no
        # report folder.
        bucket["failed"] += 1
        note = job.get("dispatch_error") or "run was not generated or submitted"
        return [{
            "method": job["work_item"]["model"],
            "task": task,
            "dataset": dataset,
            "scene": None,
            "status": "failed",
            "report_dir": None,
            "note": note,
        }]
    report_dir = pathlib.Path(job["report_dir"])
    summary_tsv, failures_tsv = _report_tsv_paths(job)
    scene_results: list[dict] = []

    if job_ok and summary_tsv.exists():
        for fields in _read_tsv_data_rows(summary_tsv):
            if len(fields) < 4:
                continue
            status, note = _summary_row_status(fields)
            row = {
                "method": fields[0],
                "task": task,
                "dataset": fields[2] if len(fields) > 2 else dataset,
                "scene": fields[3] if len(fields) > 3 else None,
                "status": status,
                "report_dir": str(report_dir),
            }
            if note:
                row["note"] = note
            scene_results.append(row)
            bucket[status] = bucket.get(status, 0) + 1
    if failures_tsv.exists():
        for fields in _read_tsv_data_rows(failures_tsv):
            if len(fields) < 4:
                continue
            scene_results.append({
                "method": fields[0],
                "task": task,
                "dataset": fields[2] if len(fields) > 2 else dataset,
                "scene": fields[3] if len(fields) > 3 else None,
                "status": "failed",
                "report_dir": str(report_dir),
            })
            bucket["failed"] += 1
    if not job_ok and not (summary_tsv.exists() or failures_tsv.exists()):
        # Terminal (or, for native, exited) without ever writing a TSV, e.g.
        # killed before run_method_case_list.sh started looping.
        scene_results.append({
            "method": job["work_item"]["model"],
            "task": task,
            "dataset": dataset,
            "scene": None,
            "status": "failed",
            "report_dir": str(report_dir),
            "note": f"{infer_note} {eval_note}, no output written".strip(),
        })
        bucket["failed"] += 1
    return scene_results


def _dataset_results(
        dataset_pass_fail: dict[tuple[str, str], dict[str, int]]) -> list[dict]:
    """Returns the per-dataset result rows, sorted by (task, dataset)."""
    return [{
        "task": task,
        "dataset": dataset,
        "passed": counts["passed"],
        "failed": counts["failed"],
        "skipped": counts.get("skipped", 0)
    } for (task, dataset), counts in sorted(dataset_pass_fail.items())]


def _expanded_path_or(value: str | pathlib.Path | None,
                      default: pathlib.Path | None) -> pathlib.Path | None:
    """Returns `value` as a user-expanded path, or `default` if it is None."""
    if value is None:
        return default
    return pathlib.Path(value).expanduser()


def _dispatcher_status(report: EvaluationReport) -> str:
    """Returns the status the detached dispatcher of a report records."""
    from spatialarena import progress as pipeline_progress

    return pipeline_progress.dispatcher_jobs(report.summary["state_path"])[0]


def _native_progress_snapshot(jobs: list[dict]) -> "PipelineProgress":
    """Returns the progress of a finished native evaluation's jobs."""
    from spatialarena import progress as pipeline_progress

    runs = []
    for job in jobs:
        total = pipeline_progress.count_cases(job.get("case_file"))
        infer_ok = job.get("infer_returncode") == 0
        eval_ok = job.get("eval_returncode") == 0
        runs.append(
            pipeline_progress.new_run(
                job["work_item"],
                stage="done" if infer_ok and eval_ok else "failed",
                total_cases=total,
                report_dir=job.get("report_dir"),
                inference_done=total if infer_ok else 0,
                evaluation_done=total if eval_ok else 0,
            ))
    return pipeline_progress.PipelineProgress(runs=runs)


class BenchmarkEvaluator:
    """Discovers, selects, plans and runs benchmark evaluations.

    The API reference (docs/benchmark/evaluator_quickstart.md) documents the
    full API contract.

    Attributes:
        benchmark_root: The benchmark datasets root.
        benchmark_eval_root: Where evaluation outputs go.
        method_code_root: The vendored method source code.
        checkpoint_root: The shared checkpoint store.
        hf_home: The Hugging Face cache inside the checkpoint store.
        experiment_root: The parent of the run's output folder; None means
            `<benchmark_root>/_eval_job_scripts`.
        experiment_name: The run's output folder name.
        job_system: "slurm" or "native".
        case_parallelism: A CASE_PARALLELISM override for every job, or None.
        slurm_exclude: SLURM nodes the generated jobs avoid (#SBATCH
            --exclude).
        max_in_flight: evaluate()'s default bound on runs in flight.
        evaluate_only: Submit only the evaluation jobs: every run's
            predictions already exist (no inference job, no GPU).
        reevaluate: Score every case again even when its evaluation file has
            results (the engine's REEVALUATE=1), e.g. after an evaluator fix.
        force_reinfer: Re-run inference even when the content-addressed
            prediction cache already has a matching entry (the engine's
            FORCE_REINFER=1).
        slurm_resources: Per-phase overrides of the engine's #SBATCH resource
            lines, e.g. {"eval": {"mem": "64G"}}.
        progress_mode: How downloads and evaluations show progress: "bars",
            "board" or "none"; None defers to $BENCHMARK_PROGRESS_MODE, then
            "bars".
        report_config: The config file's `report:` section.
    """

    def __init__(
        self,
        *,
        benchmark_root: str | pathlib.Path | None = None,
        benchmark_eval_root: str | pathlib.Path | None = None,
        method_code_root: str | pathlib.Path | None = None,
        checkpoint_root: str | pathlib.Path | None = None,
        experiment_root: str | pathlib.Path | None = None,
        experiment_name: str | None = None,
        job_system: str = "slurm",
        case_parallelism: int | None = None,
        progress_mode: str | None = None,
    ) -> None:
        self._init_progress_mode = progress_mode
        self._init_benchmark_root = benchmark_root
        self._init_benchmark_eval_root = benchmark_eval_root
        self._init_method_code_root = method_code_root
        self._init_checkpoint_root = checkpoint_root
        self._init_experiment_root = experiment_root
        self._init_experiment_name = experiment_name
        self._init_job_system = job_system
        self._init_case_parallelism = case_parallelism
        self.reset()

    def reset(self) -> None:
        """Restores the constructor's settings and clears every selection.

        Raises:
            ValueError: The constructor's job_system is unknown.
        """
        self.benchmark_root: pathlib.Path = _expanded_path_or(
            self._init_benchmark_root, dataset_registry.DEFAULT_BENCHMARK_ROOT)
        self.benchmark_eval_root: pathlib.Path = _expanded_path_or(
            self._init_benchmark_eval_root, DEFAULT_BENCHMARK_EVAL_ROOT)
        self.method_code_root: pathlib.Path = _expanded_path_or(
            self._init_method_code_root, DEFAULT_METHOD_CODE_ROOT)
        self.checkpoint_root: pathlib.Path = _expanded_path_or(
            self._init_checkpoint_root, DEFAULT_CHECKPOINT_ROOT)
        self.hf_home: pathlib.Path = self.checkpoint_root / "huggingface"
        self.experiment_root: pathlib.Path | None = _expanded_path_or(
            self._init_experiment_root, None)
        self.experiment_name: str | None = self._init_experiment_name
        if self._init_job_system not in _VALID_JOB_SYSTEMS:
            raise ValueError(
                f"Unknown job_system {self._init_job_system!r}; expected one "
                f"of {_VALID_JOB_SYSTEMS}")
        self.job_system: str = self._init_job_system
        self.case_parallelism: int | None = self._init_case_parallelism
        self.slurm_exclude: str | None = None
        self.max_in_flight: int | None = None
        self.evaluate_only: bool = False
        self.reevaluate: bool = False
        # force_reinfer is for testing pipeline or adapter changes the cache
        # key does not capture (e.g. --spatial_policy, a preprocessing fix);
        # see run_method_case_list.sh's cache-key hash payload.
        self.force_reinfer: bool = False
        self.slurm_resources: dict[str, dict[str, str]] = {}
        # See spatialarena/progress_ui.py for the progress modes.
        from spatialarena import progress_ui

        self.progress_mode: str | None = None
        if self._init_progress_mode:
            self.progress_mode = progress_ui.resolve_progress_mode(
                self._init_progress_mode)
        self._selected_datasets: list[dict] = []
        self._selected_models: list[dict] = []
        self._selected_runs: list[dict] | None = None
        self._last_plan: EvaluationPlan | None = None
        self._last_validation: ValidationReport | None = None
        self._last_report: EvaluationReport | None = None
        self.report_config: dict = {}

    # ---- Job system. ----------------------------------------------------

    def select_job_system(self, type: str) -> None:  # noqa: A002
        """Selects where jobs run: "slurm" or "native" (local subprocesses).

        Args:
            type: The job system.

        Raises:
            ValueError: `type` is unknown.
        """
        if type not in _VALID_JOB_SYSTEMS:
            raise ValueError(f"Unknown job_system {type!r}; expected one of "
                             f"{_VALID_JOB_SYSTEMS}")
        self.job_system = type

    def set_job_sys(self, type: str) -> None:  # noqa: A002
        """Legacy alias for select_job_system()."""
        self.select_job_system(type)

    # ---- Datasets. ------------------------------------------------------

    def discover_datasets(
        self,
        path: str | pathlib.Path | None = None,
        *,
        tasks: list[str] | None = None,
        datasets: list[str] | None = None,
        scenes: list[str] | None = None,
        include_scannet: bool = False,
    ) -> dict:
        """Lists the registered datasets and whether they are on disk.

        Args:
            path: The benchmark root to look in; defaults to benchmark_root.
            tasks: Restrict to these tasks.
            datasets: Restrict to these datasets.
            scenes: Restrict to these scenes.
            include_scannet: Whether to include ScanNet.

        Returns:
            A dict with the root, the selected (task, dataset) specs, every
            record, the missing records and a summary of counts.
        """
        if path is not None:
            root = pathlib.Path(path).expanduser()
        else:
            root = self.benchmark_root
        records = describe_benchmark_scenes(
            benchmark_root=root,
            tasks=tasks,
            datasets=datasets,
            scenes=scenes,
            include_scannet=include_scannet,
        )
        missing = [
            record for record in records if not record["exists"] or
            (record["metadata_required"] and not record["metadata_exists"])
        ]
        selected_specs = [{
            "task": record["task"],
            "dataset": record["dataset"]
        } for record in records]
        summary = {
            "dataset_count": len(records),
            "missing_count": len(missing),
            "total_scenes": sum(record["scene_count"] for record in records),
        }
        return {
            "root": str(root),
            "selected_specs": selected_specs,
            "records": records,
            "missing": missing,
            "summary": summary,
        }

    def check_available_datasets(self, *args, **kwargs) -> dict:
        """Legacy alias for discover_datasets()."""
        return self.discover_datasets(*args, **kwargs)

    def list_supported_datasets(self) -> list[dict]:
        """Lists every registered (task, dataset) pair and how it downloads.

        For each pair, whether download_all_supported_datasets() can fetch and
        materialize it automatically, do a download-only partial step, or
        nothing (manual). spatialarena/download.py's registry and its docstring
        explain why not everything is automated.
        """
        from spatialarena import download

        return download.list_supported_datasets()

    def download_all_supported_datasets(
        self,
        *,
        datasets: list[tuple[str, str]] | None = None,
        staging_root: str | pathlib.Path | None = None,
        log_dir: str | pathlib.Path | None = None,
        skip_existing: bool = True,
        progress: str | None = None,
    ) -> list[dict]:
        """Runs every automated download and materialize chain.

        The datasets go into self.benchmark_root. Entries
        list_supported_datasets() reports as anything other than "automated"
        are reported skipped, never silently attempted.

        Args:
            datasets: Restrict to these (task, dataset) pairs.
            staging_root: Where raw downloads are staged.
            log_dir: Where the download logs go.
            skip_existing: Skip datasets that are already materialized.
            progress: "bars", "board" or "none"; defaults to the evaluator's
                progress_mode.

        Returns:
            One result dict per dataset.
        """
        return self._run_downloads(
            progress,
            "Downloading datasets",
            datasets=datasets,
            staging_root=staging_root,
            log_dir=log_dir,
            skip_existing=skip_existing,
            kinds=("dataset",),
        )["dataset"]

    def download_all_supported(
        self,
        *,
        datasets: list[tuple[str, str]] | None = None,
        methods: list[str] | None = None,
        staging_root: str | pathlib.Path | None = None,
        log_dir: str | pathlib.Path | None = None,
        skip_existing: bool = True,
        progress: str | None = None,
    ) -> dict[str, list[dict]]:
        """Downloads everything a run needs, in one progress display.

        That is method source code, checkpoints, then datasets.

        Args:
            datasets: Restrict the datasets to these (task, dataset) pairs.
            methods: Restrict the code and checkpoints to these methods.
            staging_root: Where raw dataset downloads are staged.
            log_dir: Where the download logs go.
            skip_existing: Skip what is already on disk.
            progress: "bars", "board" or "none"; defaults to the evaluator's
                progress_mode.

        Returns:
            {"code": [...], "checkpoint": [...], "dataset": [...]}, each in the
            shape of the matching download_all_supported_*() call.
        """
        return self._run_downloads(
            progress,
            "Downloading benchmark code, checkpoints and datasets",
            datasets=datasets,
            methods=methods,
            staging_root=staging_root,
            log_dir=log_dir,
            skip_existing=skip_existing,
            kinds=("code", "checkpoint", "dataset"),
        )

    def _run_downloads(
        self,
        progress: str | None,
        title: str,
        *,
        kinds: tuple[str, ...],
        datasets: list[tuple[str, str]] | None = None,
        methods: list[str] | None = None,
        staging_root: str | pathlib.Path | None = None,
        log_dir: str | pathlib.Path | None = None,
        skip_existing: bool = True,
        method_code_root: str | pathlib.Path | None = None
    ) -> dict[str, list[dict]]:
        """Runs the requested download kinds with one progress display."""
        from spatialarena import download
        from spatialarena import progress_ui

        def work(reporter) -> dict[str, list[dict]]:
            results: dict[str, list[dict]] = {}
            if "code" in kinds:
                if method_code_root:
                    code_root = pathlib.Path(method_code_root).expanduser()
                else:
                    code_root = self.method_code_root
                results["code"] = download.download_all_supported_method_code(
                    methods=methods,
                    method_code_root=code_root,
                    skip_existing=skip_existing,
                    on_event=reporter.download,
                )
            if "checkpoint" in kinds:
                results["checkpoint"] = download.download_all_supported_models(
                    methods=methods,
                    shared_checkpoint_root=self.checkpoint_root,
                    shared_hf_home=self.hf_home,
                    method_code_root=self.method_code_root,
                    on_event=reporter.download,
                )
            if "dataset" in kinds:
                staging_dir = None
                if staging_root:
                    staging_dir = pathlib.Path(staging_root).expanduser()
                log_path = None
                if log_dir:
                    log_path = pathlib.Path(log_dir).expanduser()
                results["dataset"] = download.download_all_supported_datasets(
                    benchmark_root=self.benchmark_root,
                    datasets=datasets,
                    staging_root=staging_dir,
                    log_dir=log_path,
                    skip_existing=skip_existing,
                    on_event=reporter.download,
                )
            return results

        mode = progress_ui.resolve_progress_mode(progress, self.progress_mode)
        return progress_ui.run_with_progress(mode, work, title=title)

    def select_datasets(self,
                        items: list[dict],
                        *,
                        append: bool = False) -> None:
        """Selects the datasets to evaluate.

        Args:
            items: Dataset selectors, each with 'task' and 'dataset' (and
                optionally 'scenes', 'slurm_resources', ...).
            append: Add to the current selection instead of replacing it.

        Raises:
            TypeError: `items` is not a list of dicts.
            ValueError: A selector lacks 'task' or 'dataset'.
        """
        items = _ensure_item_list(items, what="datasets")
        for item in items:
            if "task" not in item or "dataset" not in item:
                raise ValueError("Each dataset selector needs 'task' and "
                                 f"'dataset': {item!r}")
            if "slurm_resources" in item:
                item["slurm_resources"] = _slurm_resources(
                    item["slurm_resources"],
                    where=f"dataset {item['task']}/{item['dataset']}")
        if append:
            self._selected_datasets.extend(items)
        else:
            self._selected_datasets = items
        self._invalidate_derived_state()

    def use_datasets(self, *args, **kwargs) -> None:
        """Legacy alias for select_datasets()."""
        self.select_datasets(*args, **kwargs)

    # ---- Models. --------------------------------------------------------

    def discover_models(self, model_types: list[str] | None = None) -> dict:
        """Describes the registered methods.

        Args:
            model_types: Restrict to these method names or aliases.

        Returns:
            A dict with one description per method and a summary of counts.
        """
        if model_types:
            wanted = {
                method_registry.normalize_key(name) for name in model_types
            }
            specs = [
                spec for spec in method_registry.METHOD_SPECS
                if method_registry.normalize_key(spec.name) in wanted
            ]
        else:
            specs = list(method_registry.METHOD_SPECS)
        methods = [_describe_method(spec) for spec in specs]
        return {
            "methods": methods,
            "summary": {
                "method_count": len(methods),
                "native_count": sum(1 for spec in specs if spec.native),
            },
        }

    def check_available_models(self, *args, **kwargs) -> dict:
        """Legacy alias for discover_models()."""
        return self.discover_models(*args, **kwargs)

    def list_supported_models(self) -> list[dict]:
        """Lists every registered method's checkpoint-download status.

        For each method, whether download_all_supported_models() can fetch it
        automatically (a Hugging Face hub id or a direct URL) or not (a
        local-file path with no known public source, a path depending on a
        run-time variable, or none registered); see spatialarena/download.py's
        docstring.
        """
        from spatialarena import download

        return download.list_supported_models()

    def download_all_supported_models(
            self,
            *,
            methods: list[str] | None = None,
            progress: str | None = None) -> list[dict]:
        """Fetches every automatable method checkpoint.

        Checkpoints come from a Hugging Face hub snapshot download or a
        direct-URL wget, into the shared checkpoint store. Method *source
        code* is not touched; see list_supported_method_code() and
        download_all_supported_method_code() for that. Entries
        list_supported_models() reports as not automatable are skipped, never
        guessed at.

        Args:
            methods: Restrict to these methods.
            progress: "bars", "board" or "none"; defaults to the evaluator's
                progress_mode.

        Returns:
            One result dict per checkpoint.
        """
        return self._run_downloads(progress,
                                   "Downloading checkpoints",
                                   methods=methods,
                                   kinds=("checkpoint",))["checkpoint"]

    def list_supported_method_code(self) -> list[dict]:
        """Lists every registered method's source-code download status.

        For each method, whether download_all_supported_method_code() can
        `git clone` it automatically (a verified upstream repo URL) or not (no
        confirmed source; it needs manual vendoring per Section 4 of
        docs/benchmark/evaluator_quickstart.md). spatialarena/download.py's
        METHOD_CODE_REPOS docstring explains how those URLs were verified.
        """
        from spatialarena import download

        return download.list_supported_method_code()

    def download_all_supported_method_code(
        self,
        *,
        methods: list[str] | None = None,
        method_code_root: str | pathlib.Path | None = None,
        skip_existing: bool = True,
        progress: str | None = None,
    ) -> list[dict]:
        """Clones every automatable method's vendored source code.

        Repos come from METHOD_CODE_REPOS. Methods sharing one upstream repo
        (e.g. VGGT, FASTVGGT and ROBUSTVGGT) clone it once. Entries
        list_supported_method_code() reports as not automatable are skipped,
        never guessed at.

        Args:
            methods: Restrict to these methods.
            method_code_root: Where the code goes; defaults to this
                evaluator's method_code_root.
            skip_existing: Skip checkouts that already exist.
            progress: "bars", "board" or "none"; defaults to the evaluator's
                progress_mode.

        Returns:
            One result dict per repo checkout.
        """
        return self._run_downloads(progress,
                                   "Downloading method code",
                                   methods=methods,
                                   method_code_root=method_code_root,
                                   skip_existing=skip_existing,
                                   kinds=("code",))["code"]

    def select_models(self, items: list[dict], *, append: bool = False) -> None:
        """Selects the models to evaluate.

        Args:
            items: Model selectors: a registered method ('name', optionally
                'track', 'env', 'slurm_resources', ...) or a custom adapter
                (with 'name', 'track' and 'command').
            append: Add to the current selection instead of replacing it.

        Raises:
            TypeError: `items` is not a list of dicts.
            ValueError: A selector lacks a required field.
            KeyError: A method name is not registered.
        """
        items = _ensure_item_list(items, what="models")
        for item in items:
            if "slurm_resources" in item:
                item["slurm_resources"] = _slurm_resources(
                    item["slurm_resources"], where=f"model {item.get('name')}")
            if _is_custom_model_item(item):
                missing = [
                    key for key in _CUSTOM_ADAPTER_REQUIRED_FIELDS
                    if key not in item
                ]
                if missing:
                    raise ValueError(
                        "Custom adapter model selector missing required "
                        f"field(s) {missing}: {item!r}")
            else:
                if "name" not in item:
                    raise ValueError(
                        "Each model selector needs 'name' (or 'command' for a "
                        f"custom adapter): {item!r}")
                # Raises KeyError with a clear message if unknown.
                method_registry.resolve_method(item["name"])
        if append:
            self._selected_models.extend(items)
        else:
            self._selected_models = items
        self._invalidate_derived_state()

    def use_models(self, *args, **kwargs) -> None:
        """Legacy alias for select_models()."""
        self.select_models(*args, **kwargs)

    # ---- Runs. ----------------------------------------------------------

    def select_runs(self, items: list[dict], *, append: bool = False) -> None:
        """Selects explicit runs, overriding the datasets x models product.

        Args:
            items: Run selectors, each with 'task', 'dataset' and 'model'.
            append: Add to the current run selection instead of replacing it.

        Raises:
            TypeError: `items` is not a list of dicts.
            ValueError: A selector lacks a required field.
        """
        items = _ensure_item_list(items, what="runs")
        for item in items:
            missing = [
                key for key in ("task", "dataset", "model") if key not in item
            ]
            if missing:
                raise ValueError(f"Each run selector needs {missing}: {item!r}")
        if append and self._selected_runs is not None:
            self._selected_runs = self._selected_runs + items
        else:
            self._selected_runs = items
        self._invalidate_derived_state()

    # ---- Plan. ----------------------------------------------------------

    def plan(self) -> EvaluationPlan:
        """Resolves the selection into the runs an evaluation would run.

        Returns:
            The plan.

        Raises:
            ValueError: experiment_name is not set.
        """
        output_root = self._require_output_root()
        selected_runs = self._runs_to_evaluate()
        config = self._config_snapshot()
        evaluation_plan = EvaluationPlan(
            config=config,
            selected_datasets=copy.deepcopy(self._selected_datasets),
            selected_models=copy.deepcopy(self._selected_models),
            selected_runs=copy.deepcopy(selected_runs),
            output_root=output_root,
        )
        self._last_plan = evaluation_plan
        return evaluation_plan

    def _runs_to_evaluate(self) -> list[dict]:
        """Returns the explicit run selection, else the default product."""
        if self._selected_runs is not None:
            return self._selected_runs
        return self._default_run_product()

    def _default_run_product(self) -> list[dict]:
        """Returns the selected datasets x models that are compatible.

        Compatibility is the Task/Track Compatibility Matrix x each model's own
        supported tracks, and the datasets a track runs on.
        """
        runs: list[dict] = []
        for dataset_item in self._selected_datasets:
            task = dataset_item["task"]
            for model_item in self._selected_models:
                track = model_item.get("track", "rgb")
                if not task_track_matrix.track_runs_on(task, track,
                                                       dataset_item["dataset"]):
                    continue
                if _is_custom_model_item(model_item):
                    if track not in task_track_matrix.TASK_TRACK_MATRIX.get(
                            task, ()):
                        continue
                    model_name = model_item["name"]
                else:
                    spec = method_registry.resolve_method(model_item["name"])
                    if track not in task_track_matrix.runnable_tracks(
                            task, spec):
                        continue
                    model_name = spec.name
                run = _work_item(dataset_item, model_item, task, model_name,
                                 track)
                if task == "SLAM" and not _is_custom_model_item(model_item):
                    runs.extend(
                        _slam_range_runs(run, dataset_item, model_item,
                                         model_name, track))
                    continue
                runs.append(run)
        return runs

    # ---- Validate. ------------------------------------------------------

    def validate(self) -> ValidationReport:
        """Checks the selection, the datasets, the models and the output root.

        Returns:
            The validation report; `ok` is False if any issue is an error.
        """
        issues: list[ValidationIssue] = []

        if not self._selected_datasets and self._selected_runs is None:
            issues.append(
                ValidationIssue(level="warning",
                                scope="dataset",
                                message="No datasets selected."))
        if not self._selected_models and self._selected_runs is None:
            issues.append(
                ValidationIssue(level="warning",
                                scope="model",
                                message="No models selected."))

        self._validate_datasets(issues)
        self._validate_models(issues)

        selected_runs = self._runs_to_evaluate()
        if (self._selected_datasets or self._selected_models or
                self._selected_runs is not None):
            if not selected_runs:
                issues.append(
                    ValidationIssue(
                        level="error",
                        scope="run",
                        message=(
                            "No (task, dataset, model, track) combination "
                            "survived task/track compatibility filtering -- "
                            "nothing would be evaluated."),
                    ))

        try:
            output_root = self._require_output_root()
        except ValueError as exc:
            issues.append(
                ValidationIssue(level="error", scope="run", message=str(exc)))
        else:
            if output_root.exists() and any(output_root.iterdir()):
                issues.append(
                    ValidationIssue(
                        level="warning",
                        scope="run",
                        message=("Output path already has content; a new run "
                                 "may mix with stale results."),
                        path=str(output_root),
                    ))

        ok = not any(issue.level == "error" for issue in issues)
        error_count = sum(1 for issue in issues if issue.level == "error")
        warning_count = sum(1 for issue in issues if issue.level == "warning")
        report = ValidationReport(
            ok=ok,
            issues=issues,
            summary={
                "error_count": error_count,
                "warning_count": warning_count,
                "run_count": len(selected_runs),
            },
            selected_runs=selected_runs,
        )
        self._last_validation = report
        return report

    def _validate_datasets(self, issues: list[ValidationIssue]) -> None:
        """Appends an issue per selected dataset that is missing on disk."""
        for item in self._selected_datasets:
            task = item.get("task")
            dataset = item.get("dataset")
            if "benchmark_root" in item:
                root = pathlib.Path(item["benchmark_root"]).expanduser()
            else:
                root = self.benchmark_root
            discovery = self.discover_datasets(
                root,
                tasks=[task] if task else None,
                datasets=[dataset] if dataset else None)
            if not discovery["records"]:
                issues.append(
                    ValidationIssue(
                        level="error",
                        scope="dataset",
                        message=(f"No registered dataset spec for "
                                 f"task={task!r} dataset={dataset!r}."),
                    ))
                continue
            for record in discovery["records"]:
                name = f"{record['task']}/{record['dataset']}"
                if not record["exists"]:
                    issues.append(
                        ValidationIssue(
                            level="error",
                            scope="dataset",
                            message=f"Dataset root does not exist for {name}.",
                            path=record["root_directory"],
                        ))
                elif (record["metadata_required"] and
                      not record["metadata_exists"]):
                    issues.append(
                        ValidationIssue(
                            level="error",
                            scope="dataset",
                            message=f"Missing metadata.json for {name}.",
                            path=record["metadata_path"],
                        ))

    def _validate_models(self, issues: list[ValidationIssue]) -> None:
        """Appends an issue per selected model with a bad name or track."""
        for item in self._selected_models:
            if _is_custom_model_item(item):
                # Custom adapters are the caller's responsibility to point
                # correctly.
                continue
            try:
                spec = method_registry.resolve_method(item["name"])
            except KeyError:
                issues.append(
                    ValidationIssue(
                        level="error",
                        scope="model",
                        message=("Unsupported model name or alias: "
                                 f"{item.get('name')!r}."),
                    ))
                continue
            track = item.get("track", "rgb")
            if track not in spec.tracks:
                issues.append(
                    ValidationIssue(
                        level="error",
                        scope="model",
                        message=(
                            f"{spec.name} does not support track {track!r}; "
                            f"valid: {', '.join(spec.tracks)}."),
                    ))
            # A config's own checkpoint (the method's ckpt_env in its env:) is
            # the one the jobs use: check that one.
            override = None
            if spec.ckpt_env:
                override = item.get("env", {}).get(spec.ckpt_env)
            if override:
                checked_spec = dataclasses.replace(spec, ckpt_default=override)
            else:
                checked_spec = spec
            issue = _checkpoint_issue(checked_spec, self.hf_home)
            if issue is not None:
                issues.append(issue)

    # ---- Config and misc. -----------------------------------------------

    def print_configuration(self) -> None:
        """Prints the evaluator's settings and selections."""
        config = self._config_snapshot()
        print(f"benchmark_root:   {config['benchmark_root']}")
        print(f"experiment_root:  {config['experiment_root']}")
        print(f"experiment_name:  {config['experiment_name']}")
        print(f"job_system:       {config['job_system']}")
        print(f"case_parallelism: {config['case_parallelism']}")
        print(f"selected_datasets ({len(self._selected_datasets)}):")
        for item in self._selected_datasets:
            print(f"  - {item}")
        print(f"selected_models ({len(self._selected_models)}):")
        for item in self._selected_models:
            print(f"  - {item}")
        if self._selected_runs is not None:
            print(f"selected_runs ({len(self._selected_runs)}) "
                  "[overrides the default cartesian product]:")
            for item in self._selected_runs:
                print(f"  - {item}")

    def _config_snapshot(self) -> dict:
        """Returns the evaluator's settings as a dict."""
        experiment_root = None
        if self.experiment_root:
            experiment_root = str(self.experiment_root)
        return {
            "benchmark_root": str(self.benchmark_root),
            "experiment_root": experiment_root,
            "experiment_name": self.experiment_name,
            "job_system": self.job_system,
            "case_parallelism": self.case_parallelism,
            "progress_mode": self.progress_mode,
            "slurm_exclude": self.slurm_exclude,
            "max_in_flight": self.max_in_flight,
            "evaluate_only": self.evaluate_only,
            "reevaluate": self.reevaluate,
            "force_reinfer": self.force_reinfer,
            "slurm_resources": self.slurm_resources,
        }

    def _require_output_root(self) -> pathlib.Path:
        """Returns `<experiment_root>/<experiment_name>`, the run's folder.

        It holds the run's generated scripts, plan and SLURM logs.
        experiment_root defaults to `<benchmark_root>/_eval_job_scripts`
        (shared storage, never a tmp folder).

        Raises:
            ValueError: experiment_name is not set.
        """
        if self.experiment_name is None:
            raise ValueError(
                "experiment_name must be set (via the constructor, "
                "configure(), or direct attribute assignment) "
                "before plan()/validate().")
        if self.experiment_root is not None:
            experiment_root = self.experiment_root
        else:
            experiment_root = self.benchmark_root / DEFAULT_JOB_SCRIPT_DIRNAME
        return experiment_root / self.experiment_name

    def _invalidate_derived_state(self) -> None:
        self._last_plan = None
        self._last_validation = None

    # ---- Config file. ---------------------------------------------------

    def configure(self, config_file: str | pathlib.Path, **overrides) -> None:
        """Loads a YAML or JSON config file into the evaluator's state.

        Precedence (highest to lowest), per "Configuration" in the API
        reference (docs/benchmark/evaluator_quickstart.md): explicit
        constructor values and **overrides passed to this call > config file
        values > environment variables > built-in defaults.

        Args:
            config_file: The .yaml, .yml or .json config file.
            **overrides: Config keys that take precedence over the file.

        Raises:
            FileNotFoundError: The config file does not exist.
            ValueError: The file has an unsupported extension or is not a
                mapping, or a value is invalid.
        """
        path = pathlib.Path(config_file).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            import yaml

            data = yaml.safe_load(text) or {}
        elif suffix == ".json":
            data = json.loads(text) or {}
        else:
            raise ValueError(
                f"Unsupported config file extension {path.suffix!r} for {path} "
                "(expected .yaml, .yml, or .json).")
        if not isinstance(data, dict):
            raise ValueError(
                f"Config file {path} must contain a mapping at the top level.")
        # Expands ${SPATIALARENA_ROOT}/... paths (spatialarena/paths.py).
        data = paths.expand_root(data)

        def pick(key: str, explicit_init, env_var: str | None = None):
            """Returns the value of one setting, or None to keep the current.

            Precedence: overrides > explicit constructor value > file > env
            var > None.
            """
            if key in overrides:
                return overrides[key]
            if explicit_init is not None:
                return explicit_init
            if key in data:
                return data[key]
            if env_var:
                env_value = os.environ.get(env_var)
                if env_value:
                    return env_value
            return None

        benchmark_root = pick("benchmark_root", self._init_benchmark_root,
                              "BENCHMARK_AUX_ROOT")
        if benchmark_root is not None:
            self.benchmark_root = pathlib.Path(benchmark_root).expanduser()

        benchmark_eval_root = pick("benchmark_eval_root",
                                   self._init_benchmark_eval_root,
                                   "BENCHMARK_EVAL_ROOT")
        if benchmark_eval_root is not None:
            self.benchmark_eval_root = pathlib.Path(
                benchmark_eval_root).expanduser()

        method_code_root = pick("method_code_root", self._init_method_code_root,
                                "METHOD_CODE_ROOT")
        if method_code_root is not None:
            self.method_code_root = pathlib.Path(method_code_root).expanduser()

        checkpoint_root = pick("checkpoint_root", self._init_checkpoint_root,
                               "CHECKPOINT_ROOT")
        if checkpoint_root is not None:
            self.checkpoint_root = pathlib.Path(checkpoint_root).expanduser()
            self.hf_home = self.checkpoint_root / "huggingface"

        experiment_root = pick("experiment_root", self._init_experiment_root)
        if experiment_root is not None:
            self.experiment_root = pathlib.Path(experiment_root).expanduser()

        experiment_name = pick("experiment_name", self._init_experiment_name)
        if experiment_name is not None:
            self.experiment_name = experiment_name

        progress_mode = pick("progress_mode", self._init_progress_mode,
                             "BENCHMARK_PROGRESS_MODE")
        if progress_mode is not None:
            from spatialarena import progress_ui

            self.progress_mode = progress_ui.resolve_progress_mode(
                progress_mode)

        # job_system's constructor default ("slurm") is indistinguishable from
        # "not explicitly set": only treat it as an explicit constructor value
        # here if it differs from that default.
        job_system_init = None
        if self._init_job_system != "slurm":
            job_system_init = self._init_job_system
        job_system_value = pick("job_system", job_system_init)
        if job_system_value is not None:
            if isinstance(job_system_value, dict):
                job_system_value = job_system_value.get("type", self.job_system)
            self.select_job_system(job_system_value)

        case_parallelism = pick("case_parallelism", self._init_case_parallelism)
        if case_parallelism is not None:
            self.case_parallelism = int(case_parallelism)

        slurm_exclude = pick("slurm_exclude", None)
        if slurm_exclude is not None:
            if isinstance(slurm_exclude, (list, tuple)):
                self.slurm_exclude = ",".join(slurm_exclude)
            else:
                self.slurm_exclude = str(slurm_exclude)

        max_in_flight = pick("max_in_flight", None)
        if max_in_flight is not None:
            self.max_in_flight = int(max_in_flight)

        slurm_resources = pick("slurm_resources", None)
        if slurm_resources is not None:
            self.slurm_resources = _slurm_resources(slurm_resources,
                                                    where="config")

        evaluate_only = pick("evaluate_only", None)
        if evaluate_only is not None:
            self.evaluate_only = bool(evaluate_only)

        reevaluate = pick("reevaluate", None)
        if reevaluate is not None:
            self.reevaluate = bool(reevaluate)

        force_reinfer = pick("force_reinfer", None)
        if force_reinfer is not None:
            self.force_reinfer = bool(force_reinfer)

        datasets = overrides.get("datasets", data.get("datasets"))
        if datasets is not None:
            self.select_datasets(datasets)

        models = overrides.get("models", data.get("models"))
        if models is not None:
            self.select_models(models)

        runs = overrides.get("runs", data.get("runs"))
        if runs is not None:
            self.select_runs(runs)

        report = overrides.get("report", data.get("report"))
        if report is not None:
            self.report_config = dict(report)

    def generate(
        self,
        run_id: str | None = None
    ) -> list[tuple[dict, "script_generator.GeneratedRunScripts"]]:
        """Writes every planned run's case file and scripts, submitting nothing.

        These are the same infer and eval scripts evaluate() submits, so they
        can be reviewed (cases, resources, report folders) first.

        Args:
            run_id: The run id in the generated names; defaults to the current
                time.

        Returns:
            (work item with its suite, generated scripts) per run, in plan
            order.

        Raises:
            ValueError: validate() reported errors.
        """
        validation = self.validate()
        if not validation.ok:
            errors = [
                issue.message
                for issue in validation.issues
                if issue.level == "error"
            ]
            raise ValueError(
                f"generate() refused: validate() reported error(s): {errors}")
        run_id = run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return [
            self._generate_for_work_item(work_item,
                                         run_id,
                                         case_parallelism=self.case_parallelism)
            for work_item in self.plan().selected_runs
        ]

    def evaluate(
        self,
        *,
        case_parallelism: int | None = None,
        max_in_flight: int | None = None,
        progress: str | None = None,
    ) -> EvaluationReport:
        """Submits the current plan's runs to SLURM, or runs them natively.

        With `job_system="native"` (runs synchronously), progress is shown
        while it runs, as bars or the interactive board. On SLURM `evaluate()`
        returns at once; follow it with `wait()` for the same display and then
        the collected report, or poll `progress()`.

        By default (`max_in_flight=None`) every run's infer and eval job pair is
        submitted immediately: fine for a handful of runs (a smoke test, a
        single method), but a large plan (dozens or hundreds of runs) would
        flood the scheduler's queue all at once. `max_in_flight=N` submits
        through a bounded-concurrency dispatcher instead: a detached
        background process (see `spatialarena.dispatcher`) that keeps at most
        `N` runs' worth of jobs outstanding at a time, submitting the next run
        as soon as an earlier one frees a slot. `evaluate()` still returns
        immediately either way; the dispatcher polls independently of this
        process and can outlive it.

        Args:
            case_parallelism: A CASE_PARALLELISM override for every job;
                defaults to the evaluator's case_parallelism.
            max_in_flight: The bound on runs in flight (SLURM only); defaults
                to the evaluator's max_in_flight.
            progress: "bars", "board" or "none" (native only); defaults to the
                evaluator's progress_mode.

        Returns:
            The submitted (SLURM), dispatching or finished (native) report.

        Raises:
            ValueError: validate() reported errors, or max_in_flight was given
                for a native evaluation.
        """
        if max_in_flight is None:
            max_in_flight = self.max_in_flight
        validation = self.validate()
        if not validation.ok:
            errors = [
                issue.message
                for issue in validation.issues
                if issue.level == "error"
            ]
            raise ValueError(
                f"evaluate() refused: validate() reported error(s): {errors}")

        plan = self.plan()
        output_root = plan.output_root
        # None (the common case, no override) leaves the CASE_PARALLELISM
        # defaults of run_method_benchmark_infer.sh and _eval.sh
        # (DEFAULT_INFER_PARALLELISM, DEFAULT_EVAL_PARALLELISM) in place; an
        # override is exported in the generated scripts
        # (_generate_for_work_item), never passed through sbatch --export (see
        # sbatch_submit()).
        if case_parallelism is None:
            case_parallelism = self.case_parallelism
        infer_parallelism = case_parallelism
        eval_parallelism = case_parallelism
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.job_system == "native":
            if max_in_flight is not None:
                raise ValueError(
                    "max_in_flight is a job_system='slurm'-only concept "
                    "(bounded SLURM queue depth).")
            from spatialarena import progress_ui

            def run_native(reporter):
                return self._evaluate_native(plan, run_id, infer_parallelism,
                                             eval_parallelism, reporter)

            return progress_ui.run_with_progress(
                progress_ui.resolve_progress_mode(progress, self.progress_mode),
                run_native,
                title=f"Benchmark evaluation {run_id} (native)",
            )

        if max_in_flight is not None:
            from spatialarena import dispatcher

            return dispatcher.launch_detached_dispatcher(
                self,
                plan=plan,
                run_id=run_id,
                max_in_flight=max_in_flight,
                infer_parallelism=infer_parallelism,
                eval_parallelism=eval_parallelism,
            )

        job_records: list[dict] = []
        for work_item in plan.selected_runs:
            item_with_suite, generated = self._generate_for_work_item(
                work_item, run_id, case_parallelism=infer_parallelism)
            infer_job_id = None
            if not self.evaluate_only:
                infer_job_id = slurm_utils.sbatch_submit(generated.infer_script)
            dependency = None
            if infer_job_id:
                dependency = f"afterok:{infer_job_id}"
            eval_job_id = slurm_utils.sbatch_submit(
                generated.eval_script,
                dependency=dependency,
            )
            job_records.append({
                "work_item": item_with_suite,
                "infer_script": str(generated.infer_script),
                "eval_script": str(generated.eval_script),
                "case_file": str(generated.case_file),
                "report_dir": str(generated.report_dir),
                "exp_name": generated.exp_name,
                "infer_job_id": infer_job_id,
                "eval_job_id": eval_job_id,
            })

        state_path = output_root / "scripts" / "submitted_jobs.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {"run_id": run_id, "jobs": job_records}
        state_path.write_text(
            json.dumps(state, indent=2, default=str),
            encoding="utf-8",
        )

        report = EvaluationReport(
            ok=True,
            config=plan.config,
            plan=plan,
            scene_results=[],
            dataset_results=[],
            summary={
                "status": "submitted",
                "run_id": run_id,
                "job_count": len(job_records),
                "jobs": job_records,
                "state_path": str(state_path),
            },
            report_dir=output_root,
        )
        self._last_report = report
        return report

    def _generate_for_work_item(
        self,
        work_item: dict,
        run_id: str,
        case_parallelism: int | None = None
    ) -> tuple[dict, "script_generator.GeneratedRunScripts"]:
        """Resolves one plan work item into its generated scripts.

        Shared by evaluate()'s submit-everything path and
        spatialarena.dispatcher's bounded-concurrency path, so both submit jobs
        built the exact same way. A `case_parallelism` override is exported in
        the scripts (CASE_PARALLELISM), never passed to sbatch.
        """
        suite = TASK_SUITE.get(work_item["task"])
        if suite is None:
            raise ValueError(
                f"No SLURM suite mapping for task {work_item['task']!r}.")
        extra_env = _custom_adapter_env(work_item)
        # Engine settings exported on top of the adapter's, in this order.
        engine_env: dict[str, str] = {}
        if suite == "full-slam":
            from spatialarena import slam_protocol

            slam_range = work_item.get("slam_range") or DEFAULT_SLAM_RANGE
            track = work_item.get("track", "rgb")
            if not _is_custom_model_item(
                    work_item) and not slam_protocol.in_range_track(
                        work_item["model"], slam_range, track):
                raise ValueError(
                    f"{work_item['model']} ({work_item.get('track', 'rgb')}) "
                    f"is not part of the {slam_range}-range "
                    "SLAM track; see spatialarena/slam_protocol.py")
            work_item = dict(work_item,
                             slam_range=slam_range,
                             interval=slam_protocol.range_stride(
                                 slam_range, work_item["dataset"]))
            engine_env["SLAM_RANGE"] = slam_range
        if self.reevaluate:
            engine_env["REEVALUATE"] = "1"
        if self.force_reinfer:
            engine_env["FORCE_REINFER"] = "1"
        if case_parallelism is not None:
            engine_env["CASE_PARALLELISM"] = str(case_parallelism)
        if engine_env:
            extra_env = dict(extra_env or {})
            extra_env.update(engine_env)
        entries = self._resolve_case_entries(work_item)
        item_with_suite = dict(work_item, suite=suite)
        generated = script_generator.generate_run_scripts(
            item_with_suite,
            entries,
            output_root=self._require_output_root(),
            run_id=run_id,
            benchmark_eval_root=self.benchmark_eval_root,
            checkpoint_root=self.checkpoint_root,
            method_code_root=self.method_code_root,
            benchmark_aux_root=self.benchmark_root,
            keep_predictions=True,
            keep_depth_visualizations=True,
            generate_pdf_report=False,
            extra_env=extra_env,
            slurm_exclude=self.slurm_exclude,
            slurm_resources=merged_slurm_resources(
                self.slurm_resources, work_item.get("slurm_resources")),
        )
        return item_with_suite, generated

    def _evaluate_native(
        self,
        plan: EvaluationPlan,
        run_id: str,
        infer_parallelism: int | None,
        eval_parallelism: int | None,
        reporter: "ProgressReporter | None" = None,
    ) -> EvaluationReport:
        """Runs every run's infer and eval scripts as local subprocesses.

        This is job_system="native": runs execute synchronously on whatever
        machine this process is running on. The wrapper scripts are exactly
        the ones evaluate()'s SLURM path submits; their #SBATCH lines are just
        comments to `bash`, so no separate native template is needed. The
        caller makes sure this process itself runs on a compute node (e.g. via
        `salloc` or `srun`), not the login node; evaluate() cannot enforce
        that.
        """
        from spatialarena import progress as pipeline_progress
        from spatialarena import progress_ui

        if reporter is None:
            reporter = progress_ui.ProgressReporter()

        output_root = plan.output_root
        job_records: list[dict] = []
        scene_results: list[dict] = []
        dataset_pass_fail: dict[tuple[str, str], dict[str, int]] = {}

        # Generate every run's scripts up front so the bars know the total case
        # count from the start.
        generated_runs = [
            self._generate_for_work_item(work_item, run_id)
            for work_item in plan.selected_runs
        ]
        progress = pipeline_progress.PipelineProgress(runs=[
            pipeline_progress.new_run(item,
                                      total_cases=pipeline_progress.count_cases(
                                          generated.case_file),
                                      report_dir=generated.report_dir)
            for item, generated in generated_runs
        ])
        reporter.evaluation(progress)

        for work_item, (item_with_suite,
                        generated), run in zip(plan.selected_runs,
                                               generated_runs, progress.runs):
            tsv_job = {
                "work_item": item_with_suite,
                "report_dir": str(generated.report_dir),
                "case_file": str(generated.case_file)
            }
            infer_returncode: int | None = None
            if self.evaluate_only:
                # Predictions already exist: only the evaluation phase runs.
                infer_ok = True
            else:
                infer_returncode = _run_phase_with_progress(
                    generated.infer_script, infer_parallelism, run, "inference",
                    tsv_job, progress, reporter)
                infer_ok = infer_returncode == 0

            eval_returncode: int | None = None
            eval_ok = False
            if infer_ok:
                run.inference_done = run.total_cases
                eval_returncode = _run_phase_with_progress(
                    generated.eval_script, eval_parallelism, run, "evaluation",
                    tsv_job, progress, reporter)
                eval_ok = eval_returncode == 0
            run.stage = "done" if infer_ok and eval_ok else "failed"
            if eval_ok:
                run.evaluation_done = run.total_cases
            reporter.evaluation(progress)

            job = {
                "work_item": item_with_suite,
                "infer_script": str(generated.infer_script),
                "eval_script": str(generated.eval_script),
                "case_file": str(generated.case_file),
                "report_dir": str(generated.report_dir),
                "exp_name": generated.exp_name,
                "infer_returncode": infer_returncode,
                "eval_returncode": eval_returncode,
            }
            job_records.append(job)

            key = (work_item["task"], work_item["dataset"])
            bucket = dataset_pass_fail.setdefault(key, {
                "passed": 0,
                "failed": 0,
                "skipped": 0
            })
            scene_results.extend(
                _read_job_results(
                    job,
                    infer_ok and eval_ok,
                    bucket,
                    infer_note=f"infer_returncode={infer_returncode}",
                    eval_note=f"eval_returncode={eval_returncode}",
                ))

        dataset_results = _dataset_results(dataset_pass_fail)
        passed = sum(result["passed"] for result in dataset_results)
        failed = sum(result["failed"] for result in dataset_results)

        report = EvaluationReport(
            ok=failed == 0,
            config=plan.config,
            plan=plan,
            scene_results=scene_results,
            dataset_results=dataset_results,
            summary={
                "status": "finished",
                "run_id": run_id,
                "job_count": len(job_records),
                "jobs": job_records,
                "passed": passed,
                "failed": failed,
            },
            report_dir=output_root,
        )
        self._last_report = report
        return report

    def collect(self,
                report: EvaluationReport | None = None) -> EvaluationReport:
        """Collects the results of a SLURM evaluation once its jobs finished.

        Args:
            report: The report evaluate() returned; defaults to the last one in
                this process.

        Returns:
            The finished report with scene and dataset results, or `report`
            unchanged while any job is still queued or running.

        Raises:
            ValueError: There is no report to collect.
        """
        report = report if report is not None else self._last_report
        if report is None:
            raise ValueError(
                "collect() needs a report -- pass the EvaluationReport "
                "evaluate() returned, "
                "or call evaluate() first in this process.")
        jobs = report.summary.get("jobs", [])
        if not jobs and report.summary.get("status") == "dispatching":
            # evaluate(max_in_flight=N): the detached dispatcher records every
            # run's jobs in its state file; collect once it reports "finished"
            # (runs that failed before submission have no job ids and count as
            # failed).
            from spatialarena import progress as pipeline_progress

            dispatch_status, jobs = pipeline_progress.dispatcher_jobs(
                report.summary["state_path"])
            if dispatch_status != "finished":
                return report
        if not jobs:
            return report

        job_ids = [
            job_id for job in jobs
            for job_id in (job.get("infer_job_id"), job.get("eval_job_id"))
            if job_id
        ]
        states = sacct_states(job_ids)

        def is_terminal(job_id: str | None) -> bool:
            return job_id is None or slurm_utils.is_terminal_state(
                states.get(job_id, ""))

        if not all(
                is_terminal(job.get("infer_job_id")) and
                is_terminal(job.get("eval_job_id")) for job in jobs):
            # Still queued or running: unchanged, per the evaluator API
            # reference.
            return report

        scene_results: list[dict] = []
        dataset_pass_fail: dict[tuple[str, str], dict[str, int]] = {}
        for job in jobs:
            infer_state = states.get(job.get("infer_job_id") or "", "")
            eval_state = states.get(job.get("eval_job_id") or "", "")
            # evaluate_only runs have no inference job; only the evaluation job
            # has to complete.
            infer_ok = not job.get("infer_job_id") or "COMPLETED" in infer_state
            job_ok = infer_ok and "COMPLETED" in eval_state
            key = (job["work_item"]["task"], job["work_item"]["dataset"])
            bucket = dataset_pass_fail.setdefault(key, {
                "passed": 0,
                "failed": 0,
                "skipped": 0
            })
            scene_results.extend(
                _read_job_results(
                    job,
                    job_ok,
                    bucket,
                    infer_note=f"infer={infer_state or 'unknown'}",
                    eval_note=f"eval={eval_state or 'unknown'}",
                ))

        dataset_results = _dataset_results(dataset_pass_fail)
        passed = sum(result["passed"] for result in dataset_results)
        failed = sum(result["failed"] for result in dataset_results)

        summary = dict(report.summary)
        summary["status"] = "finished"
        summary["passed"] = passed
        summary["failed"] = failed
        finalized = EvaluationReport(
            ok=failed == 0,
            config=report.config,
            plan=report.plan,
            scene_results=scene_results,
            dataset_results=dataset_results,
            summary=summary,
            report_dir=report.report_dir,
        )
        self._last_report = finalized
        return finalized

    def progress(self,
                 report: EvaluationReport | None = None) -> "PipelineProgress":
        """Returns the per-stage progress of an evaluation.

        Every run moves through queued -> inference -> evaluation -> done |
        failed; the snapshot gives each run's stage and finished/total cases
        per stage, plus pipeline totals (`.render()` prints a table,
        `.to_dict()` serializes it). See `spatialarena/progress.py` for where
        each number comes from.

        Args:
            report: The report evaluate() returned; defaults to the last one in
                this process.

        Raises:
            ValueError: There is no report.
        """
        from spatialarena import progress as pipeline_progress

        report = report if report is not None else self._last_report
        if report is None:
            raise ValueError(
                "progress() needs a report -- pass the one evaluate() "
                "returned, or call evaluate() first.")
        summary = report.summary
        if summary.get("status") == "dispatching":
            return pipeline_progress.dispatcher_progress(summary["state_path"])
        jobs = summary.get("jobs", [])
        if self.job_system == "native" or (jobs and
                                           "infer_returncode" in jobs[0]):
            return _native_progress_snapshot(jobs)
        if jobs:
            return pipeline_progress.slurm_progress(jobs)
        return pipeline_progress.PipelineProgress()

    def wait(
        self,
        report: EvaluationReport | None = None,
        *,
        poll_interval: float = 60.0,
        timeout: float | None = None,
        progress: str | None = None,
    ) -> EvaluationReport:
        """Blocks until every run of a SLURM evaluation finished, then collects.

        This includes `max_in_flight` dispatching. Progress is shown meanwhile
        as bars (runs finished, inference cases, evaluation cases) or the
        interactive board. A native evaluation is already finished when
        `evaluate()` returns.

        Args:
            report: The report evaluate() returned; defaults to the last one in
                this process.
            poll_interval: Seconds between progress polls.
            timeout: Seconds to wait at most; None waits until done.
            progress: "bars", "board" or "none"; defaults to the evaluator's
                progress_mode.

        Returns:
            The collected report; on `timeout`, or when the board's user stops
            watching, `collect()`'s result so far (unchanged if runs are still
            active).

        Raises:
            ValueError: There is no report.
        """
        from spatialarena import progress_ui

        report = report if report is not None else self._last_report
        if report is None:
            raise ValueError("wait() needs a report -- pass the one evaluate() "
                             "returned, or call evaluate() first.")

        def watch(reporter) -> None:
            started = time.monotonic()
            while not reporter.stop_requested.is_set():
                snapshot = self.progress(report)
                reporter.evaluation(snapshot)
                dispatching = report.summary.get("status") == "dispatching"
                if snapshot.runs and snapshot.finished and not (
                        dispatching and
                        _dispatcher_status(report) != "finished"):
                    return
                if (timeout is not None and
                        time.monotonic() - started >= timeout):
                    return
                reporter.stop_requested.wait(poll_interval)

        progress_ui.run_with_progress(
            progress_ui.resolve_progress_mode(progress, self.progress_mode),
            watch,
            title=f"Benchmark evaluation {report.summary.get('run_id', '')}",
            detachable=True,
        )
        return self.collect(report)

    def _resolve_case_entries(
            self,
            work_item: dict) -> list[script_generator.GeneratedCaseListEntry]:
        """Returns the case-list entries (one per scene) of a work item."""
        task = work_item["task"]
        dataset = work_item["dataset"]
        interval = work_item.get("interval", 1)
        order = work_item.get("order")
        if work_item.get("scenes"):
            scene_names = list(work_item["scenes"])
        else:
            if "benchmark_root" in work_item:
                root = pathlib.Path(work_item["benchmark_root"]).expanduser()
            else:
                root = self.benchmark_root
            records = describe_benchmark_scenes(benchmark_root=root,
                                                tasks=[task],
                                                datasets=[dataset])
            if not records or not records[0]["scenes"]:
                raise ValueError(f"No scenes discovered for task={task!r} "
                                 f"dataset={dataset!r} under {root}; "
                                 "pass explicit scenes=[...] in "
                                 "select_datasets()/select_runs().")
            scene_names = _evaluated_scene_names(work_item,
                                                 records[0]["scenes"])
        return [
            script_generator.GeneratedCaseListEntry(task=task,
                                                    dataset=dataset,
                                                    scene=scene,
                                                    interval=interval,
                                                    order=order)
            for scene in scene_names
        ]

    def export_report(
            self,
            output_path: str | pathlib.Path | None = None) -> pathlib.Path:
        """Writes the last finished report's files, per "Output Layout".

        That is summary.json, validation.json and plan.json plus a report/
        folder (aggregated TSV, a JSON visualization payload and a best-effort
        PDF), per "Output Layout" in the API reference
        (docs/benchmark/evaluator_quickstart.md). Also (re)writes
        <experiment_root>/<experiment_name>/scripts/gather_metrics.sh, a
        standalone script that reruns the same aggregation later without
        needing `spatialarena` at all.

        Uses the evaluator's own last state (self._last_report, _last_plan and
        _last_validation): call evaluate() and collect() (and, for a populated
        validation.json, validate()) first.

        Args:
            output_path: The folder to write; defaults to the config file's
                `report.output_path` (see configure()).

        Returns:
            The output folder.

        Raises:
            ValueError: There is no output path or no finished report.
        """
        if output_path is None:
            output_path = self.report_config.get("output_path")
            if output_path is None:
                raise ValueError("export_report() needs an output_path, or a "
                                 "report.output_path in the configure() file.")
        report = self._last_report
        if report is None:
            raise ValueError(
                "export_report() needs a finished report -- call evaluate() "
                "(and collect(), for job_system='slurm') first.")

        output_path = pathlib.Path(output_path).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)

        plan = self._last_plan or report.plan
        (output_path / "plan.json").write_text(json.dumps(plan.to_dict(),
                                                          indent=2,
                                                          default=str),
                                               encoding="utf-8")
        if self._last_validation is not None:
            validation_json = json.dumps(self._last_validation.to_dict(),
                                         indent=2,
                                         default=str)
            (output_path / "validation.json").write_text(validation_json,
                                                         encoding="utf-8")
        summary_payload = dict(report.summary)
        summary_payload["ok"] = report.ok
        summary_payload["dataset_results"] = report.dataset_results
        (output_path / "summary.json").write_text(json.dumps(summary_payload,
                                                             indent=2,
                                                             default=str),
                                                  encoding="utf-8")

        report_roots = sorted(
            {job["report_dir"] for job in report.summary.get("jobs", [])})
        if report_roots:
            self._write_aggregated_report(output_path / "report", report_roots)

        if (self.experiment_root is not None and
                self.experiment_name is not None):
            self._write_gather_metrics_script(self._require_output_root(),
                                              report_roots)

        return output_path

    def _write_aggregated_report(self, report_dir_out: pathlib.Path,
                                 report_roots: list[str]) -> None:
        """Writes summary.tsv, report.json and report.pdf of report roots."""
        report_dir_out.mkdir(parents=True, exist_ok=True)
        roots = [pathlib.Path(root) for root in report_roots]

        from scripts.test import summarize_benchmark_reports

        rows = summarize_benchmark_reports.deduplicate_scene_metrics(
            summarize_benchmark_reports.collect_scene_metrics(
                roots, benchmark_root=self.benchmark_root))
        summaries = summarize_benchmark_reports.aggregate_by_dataset(rows)
        summary_tsv_path = report_dir_out / "summary.tsv"
        summary_tsv_path.write_text(
            summarize_benchmark_reports.render_tsv(summaries), encoding="utf-8")

        try:
            from app.webpage.python import benchmark_analyzer

            visualization = benchmark_analyzer.build_benchmark_visualization(
                summary_tsv_path, roots)
            (report_dir_out / "report.json").write_text(json.dumps(
                visualization, indent=2, default=str),
                                                        encoding="utf-8")
        # Best-effort: summary.tsv above is the load-bearing artifact.
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            (report_dir_out / "report_json_error.txt").write_text(
                str(exc), encoding="utf-8")

        try:
            from scripts.test import generate_benchmark_pdf_report

            generate_benchmark_pdf_report.write_pdf(
                roots,
                report_dir_out / "report.pdf",
                title=self.experiment_name or "Benchmark Report",
                exclude_scannet=False,
            )
        # PDF generation is optional (the evaluator API reference).
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            (report_dir_out / "report_pdf_error.txt").write_text(
                str(exc), encoding="utf-8")

    def _write_gather_metrics_script(self, output_root: pathlib.Path,
                                     report_roots: list[str]) -> pathlib.Path:
        """Writes scripts/gather_metrics.sh, which reruns the aggregation."""
        scripts_dir = output_root / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        script_path = scripts_dir / "gather_metrics.sh"
        project_root = pathlib.Path(__file__).resolve().parents[1]
        report_dir_out = output_root / "report"
        roots_args = " \\\n    ".join(f'"{root}"' for root in report_roots)
        if not roots_args:
            roots_args = ("# (no report roots recorded yet -- rerun "
                          "export_report() after evaluate()/collect())")
        report_root_flags = " ".join(
            f'--report-root "{root}"' for root in report_roots)
        title = self.experiment_name or "Benchmark Report"
        lines = [
            "#!/bin/bash",
            "# Generated by BenchmarkEvaluator.export_report() -- do not "
            "hand-edit;",
            "# rerun export_report() to regenerate. Standalone: does not "
            "import",
            "# spatialarena, re-derives summary.tsv/report.json/report.pdf "
            "for this",
            "# run's own report roots on demand.",
            "set -euo pipefail",
            f'cd "{project_root}"',
            "",
            "PYTHON=${PYTHON:-python}",
            "REPORT_ROOTS=(",
            f"    {roots_args}",
            ")",
            f'OUT_DIR="{report_dir_out}"',
            'mkdir -p "$OUT_DIR"',
            "",
            '"$PYTHON" scripts/test/summarize_benchmark_reports.py \\',
            f'    --benchmark-root "{self.benchmark_root}" \\',
            '    "${REPORT_ROOTS[@]}" \\',
            '    --output-tsv "$OUT_DIR/summary.tsv" '
            '--output-markdown "$OUT_DIR/summary.md"',
            "",
            '"$PYTHON" app/webpage/python/benchmark_analyzer.py \\',
            '    --input "$OUT_DIR/summary.tsv" '
            '--output "$OUT_DIR/results.json" \\',
            f"    {report_root_flags}",
            "",
            '"$PYTHON" scripts/test/generate_benchmark_pdf_report.py \\',
            '    "${REPORT_ROOTS[@]}" \\',
            f'    --output "$OUT_DIR/report.pdf" --title "{title}"',
            "",
            'echo "Wrote $OUT_DIR/summary.tsv, $OUT_DIR/summary.md, '
            '$OUT_DIR/results.json, $OUT_DIR/report.pdf"',
        ]
        script_path.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
        script_path.chmod(script_path.stat().st_mode | 0o111)
        return script_path


def _work_item(dataset_item: dict, model_item: dict, task: str, model_name: str,
               track: str) -> dict:
    """Returns the work item of one compatible (dataset, model) selection."""
    run: dict = {
        "task": task,
        "dataset": dataset_item["dataset"],
        "model": model_name,
        "track": track,
    }
    if "scenes" in dataset_item:
        run["scenes"] = dataset_item["scenes"]
    for key in ("benchmark_root", "experiment_root", "experiment_name"):
        if key in dataset_item:
            run[key] = dataset_item[key]
    for key, value in model_item.items():
        if key not in ("name", "track"):
            run[key] = value
    if "slurm_resources" in dataset_item or "slurm_resources" in model_item:
        run["slurm_resources"] = merged_slurm_resources(
            dataset_item.get("slurm_resources"),
            model_item.get("slurm_resources"))
    return run


def _slam_range_runs(run: dict, dataset_item: dict, model_item: dict,
                     model_name: str, track: str) -> list[dict]:
    """Returns one SLAM work item per range track the method takes part in.

    An explicit "slam_range" on the dataset or model item narrows it to that
    track.
    """
    from spatialarena import slam_protocol

    wanted = dataset_item.get("slam_range") or model_item.get("slam_range")
    dataset_name = dataset_item["dataset"].lower()
    runs = []
    for slam_range in slam_protocol.method_slam_ranges(model_name, track):
        if (wanted in (None, slam_range) and
                dataset_name in slam_protocol.range_datasets(slam_range)):
            runs.append(dict(run, slam_range=slam_range))
    return runs


def _evaluated_scene_names(work_item: dict, scenes: list[dict]) -> list[str]:
    """Returns the discovered scenes of a work item that are evaluated.

    MVRecon scores against a GT point cloud or mesh: a scene without one (ETH3D
    meadow and terrace, which DA3-BENCH does not ship) is left out, as in the
    canonical case lists (generate_full_standard_cases.py). A SLAM sequence
    that names its range track (SpatialArena's) runs only in that one.
    """
    task = work_item["task"]
    slam_range = None
    if dataset_registry.canonical_task_name(task) == "SLAM":
        slam_range = work_item.get("slam_range")
    scene_names = []
    for scene in scenes:
        if task == "MVRecon" and not scene.get("has_gt_geometry", True):
            continue
        if slam_range and scene.get("slam_range") not in (None, slam_range):
            continue
        scene_names.append(scene["scene"])
    return scene_names


def _describe_method(spec: method_registry.MethodSpec) -> dict:
    """Returns the discover_models() description of one method."""
    tracks_by_task = {
        task: list(task_track_matrix.runnable_tracks(task, spec))
        for task in task_track_matrix.TASK_TRACK_MATRIX
    }
    return {
        "name": spec.name,
        "aliases": list(spec.aliases),
        "native": spec.native,
        "adapter": spec.adapter,
        "tracks": list(spec.tracks),
        "task_tracks": tracks_by_task,
        "code_env": spec.code_env,
        "code_default": spec.code_default,
        "ckpt_env": spec.ckpt_env,
        "ckpt_default": spec.ckpt_default,
        "external_command_env": spec.external_command_env,
        "input_resolution": spec.input_resolution,
        "image_load_resolution": spec.image_load_resolution,
        "pixel_limit": spec.pixel_limit,
        "conf_thresh": spec.conf_thresh,
        "depth_conf_thresh": spec.depth_conf_thresh,
        "resolution_source": spec.resolution_source,
        "notes": spec.notes,
    }


def _checkpoint_issue(spec: method_registry.MethodSpec,
                      hf_home: pathlib.Path) -> ValidationIssue | None:
    """Checks, best-effort, that a checkpoint resolves in the shared store.

    Not exhaustive: a ckpt_default containing an unexpanded shell variable
    (e.g. `$METHOD_CODE_ROOT/...`) can only be fully resolved once script
    generation (Phase 3) knows the real METHOD_CODE_ROOT, so that case is
    reported as a warning, not an error.

    Args:
        spec: The method, with the checkpoint the jobs would use.
        hf_home: The shared Hugging Face cache.

    Returns:
        The issue found, or None.
    """
    ckpt = spec.ckpt_default
    if not ckpt:
        return None
    if ckpt.startswith("http://") or ckpt.startswith("https://"):
        return ValidationIssue(
            level="warning",
            scope="model",
            message=(f"{spec.name}'s checkpoint is a direct download URL; "
                     "resolvability isn't checked here."),
            path=ckpt,
        )
    if "$" in ckpt:
        return ValidationIssue(
            level="warning",
            scope="model",
            message=(f"{spec.name}'s checkpoint path depends on a variable "
                     "resolved at run time."),
            path=ckpt,
        )
    if ckpt.startswith("/"):
        if not pathlib.Path(ckpt).exists():
            return ValidationIssue(
                level="error",
                scope="model",
                message=f"{spec.name}'s checkpoint path does not exist.",
                path=ckpt,
            )
        return None
    # Otherwise assume a Hugging Face hub id (e.g. "org/name").
    cache_dir = hf_home / "hub" / f"models--{ckpt.replace('/', '--')}"
    if not cache_dir.is_dir():
        return ValidationIssue(
            level="warning",
            scope="model",
            message=(f"{spec.name}'s checkpoint ({ckpt}) isn't cached in the "
                     "shared checkpoint store yet; a job would need network "
                     "access to fetch it."),
            path=str(cache_dir),
        )
    return None
