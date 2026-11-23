# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""VideoDepth evaluation over standardized prediction artifacts."""

from spatialarena.eval import depth_prediction_evaluator

if __name__ == "__main__":
    depth_prediction_evaluator.main(default_tasks=["VideoDepth"])
