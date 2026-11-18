#!/bin/bash
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

# Refresh benchmark metrics from every discovered report root, then serve the dashboard.
#
# Usage: app/webpage/run.sh [PORT]
#   PORT                     HTTP port (default 8020)
#   PYTHON                   interpreter with spatialarena deps (default: python3)
#   BENCHMARK_REPORTS_ROOT   reports tree to collect from
#   SKIP_COLLECT=1           serve the existing data/ without re-collecting
#   COLLECT_ARGS             extra collect_benchmark_metrics.py flags, e.g. "--all-scenes"

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${1:-8020}"
PYTHON="${PYTHON:-python3}"

if [[ "${SKIP_COLLECT:-0}" != "1" ]]; then
    # shellcheck disable=SC2086
    "$PYTHON" "$APP_DIR/python/collect_benchmark_metrics.py" ${COLLECT_ARGS:-}
fi

exec "$PYTHON" "$APP_DIR/server.py" "$PORT"
