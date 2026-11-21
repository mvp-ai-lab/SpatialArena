# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Multi-view depth evaluation over standardized prediction artifacts."""

from spatialarena.eval.depth_prediction_evaluator import main


if __name__ == "__main__":
    main(default_tasks=["MVRecon"])
