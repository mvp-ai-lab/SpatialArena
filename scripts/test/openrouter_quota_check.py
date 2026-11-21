# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Check whether an OpenRouter key can afford the API-VLM MonoDepth evaluation, from a small real pilot.

1. Reads the key's credit limit / remaining credit (``GET /api/v1/key``) and account balance (``/api/v1/credits``).
2. Counts the MonoDepth images each method will query (every image of every canonical MonoDepth case).
3. Runs ``--pilot_per_dataset`` images of every dataset through each model with the real adapter request, and
   extrapolates the billed cost per dataset (image tokens depend on aspect ratio and resolution).

Pilot answers land in the adapter's response cache, so the full run does not pay for them again.

    OPENROUTER_API_KEY=... python scripts/test/openrouter_quota_check.py --benchmark_root ~/datasets/benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from spatialarena.methods import resolve_method  # noqa: E402
from scripts.inference import openrouter_vlm_depth as adapter  # noqa: E402
from scripts.inference.adapter_utils import canonical_image_root, find_image_paths  # noqa: E402

CASE_FILE = ROOT_DIR / "scripts" / "test" / "full_standard_depth_cases.txt"


def get_json(url: str, api_key: str) -> dict:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def monodepth_images(benchmark_root: Path, datasets: set[str] | None) -> dict[str, list[str]]:
    """Every image of every canonical MonoDepth case, grouped by dataset."""
    images: dict[str, list[str]] = {}
    for line in CASE_FILE.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("|")
        if len(parts) < 3 or line.startswith("#") or parts[0] != "MonoDepth":
            continue
        dataset, scene = parts[1], parts[2]
        if datasets and dataset not in datasets:
            continue
        scene_dir = benchmark_root / "MonoDepth" / dataset / scene
        if not scene_dir.is_dir():
            raise FileNotFoundError(f"Missing MonoDepth scene {scene_dir}")
        images.setdefault(dataset, []).extend(find_image_paths(canonical_image_root(scene_dir)))
    return images


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark_root", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=list(adapter.OPENROUTER_MODELS))
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--pilot_per_dataset", type=int, default=2)
    parser.add_argument("--skip_pilot", action="store_true")
    parser.add_argument("--json_out", type=Path, default=None)
    cli = parser.parse_args()

    api_key = adapter.read_api_key()
    key_info = get_json("https://openrouter.ai/api/v1/key", api_key).get("data", {})
    try:
        credits = get_json("https://openrouter.ai/api/v1/credits", api_key).get("data", {})
    except urllib.error.HTTPError:  # needs a management-capable key on some accounts
        credits = {}
    balance = None
    if credits:
        balance = float(credits.get("total_credits", 0)) - float(credits.get("total_usage", 0))
    limit_remaining = key_info.get("limit_remaining")
    print(f"key label={key_info.get('label')!r} limit={key_info.get('limit')} limit_remaining={limit_remaining} "
          f"usage={key_info.get('usage')} free_tier={key_info.get('is_free_tier')}")
    if credits:
        print(f"account credits total={credits.get('total_credits')} used={credits.get('total_usage')} "
              f"balance={balance:.2f}")

    images = monodepth_images(cli.benchmark_root.expanduser(), set(cli.datasets) if cli.datasets else None)
    counts = {dataset: len(paths) for dataset, paths in images.items()}
    print("images per dataset:", counts, "total", sum(counts.values()))
    if cli.skip_pilot:
        return 0

    report: dict = {"key": key_info, "balance": balance, "images": counts, "methods": {}}
    grand_total = 0.0
    for method in cli.methods:
        spec = resolve_method(method)
        args = argparse.Namespace(
            point_grid_size=8, image_load_resolution=spec.image_load_resolution, reasoning_effort="low",
            max_tokens=8192, max_retries=4, timeout=300.0, allow_data_collection=False,
            cache_dir=Path("~/.cache/feedforward_benchmark/openrouter").expanduser(),
        )
        method_total = 0.0
        per_dataset = {}
        for dataset, paths in images.items():
            step = max(1, len(paths) // max(1, cli.pilot_per_dataset))
            sample = paths[::step][: cli.pilot_per_dataset]
            ledger = adapter.UsageLedger(max_cost_usd=0)
            parsed = 0
            for path in sample:
                result = adapter.query_image(path, spec.ckpt_default, api_key, args, ledger)
                parsed += result["record"]["parsed_points"]
            calls = ledger.api_calls
            # Cached pilot images were paid for earlier; extrapolate only from fresh calls.
            per_image = ledger.cost_usd / calls if calls else float("nan")
            estimate = per_image * counts[dataset]
            method_total += estimate if calls else 0.0
            per_dataset[dataset] = {
                "pilot_images": len(sample), "fresh_calls": calls, "pilot_cost": ledger.cost_usd,
                "usd_per_image": per_image, "estimate_usd": estimate,
                "prompt_tokens_per_call": ledger.prompt_tokens / calls if calls else None,
                "completion_tokens_per_call": ledger.completion_tokens / calls if calls else None,
                "reasoning_tokens_per_call": ledger.reasoning_tokens / calls if calls else None,
                "parsed_fraction": parsed / (64 * len(sample)) if sample else None,
            }
            print(f"  {method:16s} {dataset:13s} ${per_image:.4f}/img x {counts[dataset]:5d} = ${estimate:8.2f} "
                  f"(in {per_dataset[dataset]['prompt_tokens_per_call']}, "
                  f"out {per_dataset[dataset]['completion_tokens_per_call']}, "
                  f"parsed {per_dataset[dataset]['parsed_fraction']:.0%})", flush=True)
        report["methods"][method] = {"slug": spec.ckpt_default, "estimate_usd": method_total, "datasets": per_dataset}
        grand_total += method_total
        print(f"{method:16s} ({spec.ckpt_default}) estimated full MonoDepth cost: ${method_total:.2f}", flush=True)

    available = limit_remaining if limit_remaining is not None else balance
    report["estimate_total_usd"] = grand_total
    print(f"TOTAL estimated: ${grand_total:.2f}; available: {available}")
    if available is not None:
        verdict = "enough" if float(available) >= grand_total * 1.2 else "NOT enough (need 20% headroom)"
        print(f"verdict: {verdict}")
    if cli.json_out:
        cli.json_out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
