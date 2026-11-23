# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Multi-view depth evaluation over standardized prediction artifacts.

Runs `depth_prediction_evaluator.main` with the MVRecon task as the default:

    python -m spatialarena.eval.eval_mvdepth --help
"""

from spatialarena.eval import depth_prediction_evaluator

if __name__ == "__main__":
    depth_prediction_evaluator.main(default_tasks=["MVRecon"])
