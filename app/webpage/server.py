#!/usr/bin/env python3
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Standalone static-file HTTP server for the benchmark dashboard.

Serves this directory regardless of the caller's working directory, so it can be
started from anywhere::

    python app/webpage/server.py 8020
"""

from __future__ import annotations

import argparse
import http.server
import os
import socketserver
import webbrowser
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent

try:
    HTTP_TIMEOUT_SECONDS = int(os.environ.get("BENCHMARK_DASHBOARD_HTTP_TIMEOUT", "1800"))
except ValueError:
    HTTP_TIMEOUT_SECONDS = 1800


class DashboardRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Static-file handler that disables caching."""

    def setup(self) -> None:
        self.request.settimeout(HTTP_TIMEOUT_SECONDS)
        super().setup()

    def end_headers(self) -> None:
        # results.json and the viewer indexes are regenerated in place; never serve stale copies.
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


class ThreadingDashboardServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Threaded so slow lazy viewer assets do not block JSON and page requests."""

    allow_reuse_address = True
    daemon_threads = True
    block_on_close = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("port", nargs="?", type=int, default=8020, help="Port to listen on (default: 8020).")
    parser.add_argument("--host", default="", help="Interface to bind (default: all interfaces).")
    parser.add_argument("--open-browser", action="store_true", help="Open the dashboard in a browser.")
    args = parser.parse_args()

    os.chdir(APP_ROOT)

    with ThreadingDashboardServer((args.host, args.port), DashboardRequestHandler) as httpd:
        url = f"http://localhost:{args.port}/"
        print(f"Benchmark dashboard: {url}")
        print(f"Serving {APP_ROOT} (request timeout {HTTP_TIMEOUT_SECONDS}s); Ctrl+C to stop")
        if not (APP_ROOT / "data" / "results.json").is_file():
            print("warning: data/results.json is missing; run python/collect_benchmark_metrics.py first")
        if args.open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped")


if __name__ == "__main__":
    main()
