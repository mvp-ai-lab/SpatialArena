# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""CLI wrapper around the benchmark method registry.

The registry itself (`MethodSpec`, `METHOD_SPECS`, `resolve_method`,
`valid_tracks`, `validate_track`, `specs_as_dicts`, ...) lives in
`spatialarena.methods` so that `BenchmarkEvaluator` can import it. This file
re-exports it unchanged and keeps only the `--field ...` CLI that
`run_method_case_list.sh` depends on; that script parses the printed output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# This script is invoked directly (`python .../benchmark_methods.py`, not
# `-m`) by run_method_case_list.sh, so sys.path[0] is this file's own
# directory, not the project root. A bare `import spatialarena` would rely on
# the package being installed, which is unreliable here (a stale editable
# install on this env points at a checkout that no longer exists), so insert
# the project root explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# The names are imported directly because this module re-exports them; tests
# import them from here.
# pylint: disable-next=unused-import
from spatialarena.methods import (  # noqa: F401,E402
    TRACK_RGB, TRACK_RGB_CAMERAS, TRACK_RGB_DEPTH, MethodSpec, METHOD_SPECS,
    method_names, normalize_key, resolve_method, specs_as_dicts, valid_tracks,
    validate_track,
)


def parse_args() -> argparse.Namespace:
    """Parses the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list",
                        action="store_true",
                        help="Print supported canonical method names")
    parser.add_argument("--json",
                        action="store_true",
                        help="Print registry as JSON")
    parser.add_argument("--method", help="Resolve one method or alias")
    parser.add_argument("--track",
                        default=TRACK_RGB,
                        help="Validate modality track for --method")
    parser.add_argument(
        "--field",
        choices=(
            "name",
            "adapter",
            "code_env",
            "code_default",
            "ckpt_env",
            "ckpt_default",
            "tracks",
            "tasks",
            "native",
            "external_command_env",
            "input_resolution",
            "point_grid_size",
            "image_load_resolution",
            "pixel_limit",
            "conf_thresh",
            "depth_conf_thresh",
            "resolution_source",
            "notes",
            "depth_output_space",
            "default_depth_protocol",
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Prints the requested registry view to stdout."""
    args = parse_args()
    # External models add their methods when their module is imported.
    # pylint: disable-next=import-outside-toplevel
    from spatialarena.models import registry

    registry.load_plugins()
    if args.json:
        print(json.dumps(specs_as_dicts(), indent=2))
        return
    if args.list:
        print(" ".join(method_names()))
        return
    if args.method:
        try:
            spec = validate_track(args.method, args.track)
        except (KeyError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        data = specs_as_dicts((spec,))[0]
        if args.field:
            value = data[args.field]
            if isinstance(value, (tuple, list)):
                print(" ".join(value))
            elif isinstance(value, bool):
                print("1" if value else "0")
            else:
                print(value)
        else:
            print(json.dumps(data, indent=2))
        return
    print(" ".join(method_names()))


if __name__ == "__main__":
    main()
