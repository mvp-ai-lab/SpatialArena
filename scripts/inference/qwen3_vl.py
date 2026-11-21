# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Run Qwen3-VL point-query metric-depth baseline for benchmark depth evaluation."""

from point_query_depth_adapter import run


if __name__ == "__main__":
    import sys

    sys.argv.extend(["--method_name", "QWEN3_VL"])
    run()
