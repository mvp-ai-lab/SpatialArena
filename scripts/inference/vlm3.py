# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Run VLM3 point-query metric depth for benchmark depth evaluation."""

from point_query_depth_adapter import run

if __name__ == "__main__":
    import sys

    sys.argv.extend(["--method_name", "VLM3"])
    run()
