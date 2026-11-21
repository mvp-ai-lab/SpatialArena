# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""FeedForward benchmark suite."""

from spatialarena.evaluator import (
    BenchmarkEvaluator,
    EvaluationPlan,
    EvaluationReport,
    ValidationIssue,
    ValidationReport,
)

evaluator = BenchmarkEvaluator()
"""Convenience singleton for short scripts and notebooks -- see
the API reference (docs/benchmark/evaluator_quickstart.md). Construct a fresh ``BenchmarkEvaluator()`` instead
for anything that needs isolated state (e.g. concurrent runs)."""

__all__ = [
    "BenchmarkEvaluator",
    "EvaluationPlan",
    "EvaluationReport",
    "ValidationIssue",
    "ValidationReport",
    "evaluator",
]
