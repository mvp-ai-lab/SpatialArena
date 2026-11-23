# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""FeedForward benchmark suite.

The package exports the `BenchmarkEvaluator` API (see
docs/benchmark/evaluator_quickstart.md) and a ready-made `evaluator`
instance of it.
"""

# The public API classes are re-exported here by name.
from spatialarena.evaluator import BenchmarkEvaluator
from spatialarena.evaluator import EvaluationPlan
from spatialarena.evaluator import EvaluationReport
from spatialarena.evaluator import ValidationIssue
from spatialarena.evaluator import ValidationReport

evaluator = BenchmarkEvaluator()
"""Convenience singleton for short scripts and notebooks.

See the API reference (docs/benchmark/evaluator_quickstart.md). Construct a
fresh `BenchmarkEvaluator()` instead for anything that needs isolated state
(e.g. concurrent runs).
"""

__all__ = [
    "BenchmarkEvaluator",
    "EvaluationPlan",
    "EvaluationReport",
    "ValidationIssue",
    "ValidationReport",
    "evaluator",
]
