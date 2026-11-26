# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Checks whether an OpenRouter key can afford the API-VLM MonoDepth run.

The estimate comes from a small real pilot:

1. Reads the key's credit limit and remaining credit (`GET /api/v1/key`) and
   the account balance (`/api/v1/credits`).
2. Counts the MonoDepth images each method will query (every image of every
   canonical MonoDepth case).
3. Runs `--pilot_per_dataset` images of every dataset through each model with
   the real adapter request, and extrapolates the billed cost per dataset
   (image tokens depend on aspect ratio and resolution).

Pilot answers land in the adapter's response cache, so the full run does not
pay for them again.

    OPENROUTER_API_KEY=... python scripts/test/openrouter_quota_check.py \\
        --benchmark_root ~/datasets/benchmark
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.error
import urllib.request

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# pylint: disable=wrong-import-position
from scripts.inference import adapter_utils  # noqa: E402
from spatialarena.models import openrouter_vlm as adapter  # noqa: E402
from spatialarena import methods  # noqa: E402

# pylint: enable=wrong-import-position

CASE_FILE = ROOT_DIR / "scripts" / "test" / "full_standard_depth_cases.txt"


def get_json(url: str, api_key: str) -> dict:
    """Returns the decoded JSON of a GET request authorized with `api_key`.

    Args:
        url: The API endpoint.
        api_key: The OpenRouter API key.

    Returns:
        The decoded response.
    """
    request = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def monodepth_images(benchmark_root: pathlib.Path,
                     datasets: set[str] | None) -> dict[str, list[str]]:
    """Lists every image of every canonical MonoDepth case.

    Args:
        benchmark_root: The benchmark root.
        datasets: The datasets to keep; None or empty keeps all.

    Returns:
        The image paths, grouped by dataset.

    Raises:
        FileNotFoundError: If a case's scene folder is missing.
    """
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
        image_root = adapter_utils.canonical_image_root(scene_dir)
        images.setdefault(dataset,
                          []).extend(adapter_utils.find_image_paths(image_root))
    return images


def _read_credit(api_key: str) -> tuple[dict, float | None]:
    """Reads and prints the key's limits and the account balance.

    Args:
        api_key: The OpenRouter API key.

    Returns:
        The key info and the account balance (None if it cannot be read).
    """
    key_info = get_json("https://openrouter.ai/api/v1/key",
                        api_key).get("data", {})
    try:
        credits = get_json("https://openrouter.ai/api/v1/credits",
                           api_key).get("data", {})
    except urllib.error.HTTPError:
        # Needs a management-capable key on some accounts.
        credits = {}
    balance = None
    if credits:
        balance = float(credits.get("total_credits", 0)) - float(
            credits.get("total_usage", 0))
    limit_remaining = key_info.get("limit_remaining")
    print(f"key label={key_info.get('label')!r} "
          f"limit={key_info.get('limit')} "
          f"limit_remaining={limit_remaining} "
          f"usage={key_info.get('usage')} "
          f"free_tier={key_info.get('is_free_tier')}")
    if credits:
        print(f"account credits total={credits.get('total_credits')} "
              f"used={credits.get('total_usage')} "
              f"balance={balance:.2f}")
    return key_info, balance


def _per_call(total: float, calls: int) -> float | None:
    """Returns `total / calls`, or None when there were no calls."""
    if calls:
        return total / calls
    return None


def _pilot_method(method: str, images: dict[str, list[str]], counts: dict[str,
                                                                          int],
                  api_key: str, pilot_per_dataset: int) -> dict:
    """Runs the pilot of one method and prints its per-dataset estimates.

    Args:
        method: The method name.
        images: The image paths per dataset.
        counts: The image count per dataset.
        api_key: The OpenRouter API key.
        pilot_per_dataset: How many images of each dataset to query.

    Returns:
        The method's report entry (`slug`, `estimate_usd`, `datasets`).
    """
    spec = methods.resolve_method(method)
    args = argparse.Namespace(
        point_grid_size=8,
        image_load_resolution=spec.image_load_resolution,
        reasoning_effort="low",
        max_tokens=8192,
        max_retries=4,
        timeout=300.0,
        allow_data_collection=False,
        cache_dir=pathlib.Path(
            "~/.cache/feedforward_benchmark/openrouter").expanduser(),
    )
    method_total = 0.0
    per_dataset = {}
    for dataset, paths in images.items():
        step = max(1, len(paths) // max(1, pilot_per_dataset))
        sample = paths[::step][:pilot_per_dataset]
        ledger = adapter.UsageLedger(max_cost_usd=0)
        parsed = 0
        for path in sample:
            result = adapter.query_image(path, spec.ckpt_default, api_key, args,
                                         ledger)
            parsed += result["record"]["parsed_points"]
        calls = ledger.api_calls
        # Cached pilot images were paid for earlier; extrapolate only from
        # fresh calls.
        if calls:
            per_image = ledger.cost_usd / calls
        else:
            per_image = float("nan")
        estimate = per_image * counts[dataset]
        if calls:
            method_total += estimate
        if sample:
            parsed_fraction = parsed / (64 * len(sample))
        else:
            parsed_fraction = None
        per_dataset[dataset] = {
            "pilot_images":
                len(sample),
            "fresh_calls":
                calls,
            "pilot_cost":
                ledger.cost_usd,
            "usd_per_image":
                per_image,
            "estimate_usd":
                estimate,
            "prompt_tokens_per_call":
                _per_call(ledger.prompt_tokens, calls),
            "completion_tokens_per_call":
                _per_call(ledger.completion_tokens, calls),
            "reasoning_tokens_per_call":
                _per_call(ledger.reasoning_tokens, calls),
            "parsed_fraction":
                parsed_fraction,
        }
        print(
            f"  {method:16s} {dataset:13s} ${per_image:.4f}/img x "
            f"{counts[dataset]:5d} = ${estimate:8.2f} "
            f"(in {per_dataset[dataset]['prompt_tokens_per_call']}, "
            f"out {per_dataset[dataset]['completion_tokens_per_call']}, "
            f"parsed {per_dataset[dataset]['parsed_fraction']:.0%})",
            flush=True)
    print(
        f"{method:16s} ({spec.ckpt_default}) estimated full MonoDepth cost: "
        f"${method_total:.2f}",
        flush=True)
    return {
        "slug": spec.ckpt_default,
        "estimate_usd": method_total,
        "datasets": per_dataset
    }


def main() -> int:
    """Prints the credit, the image counts and the pilot cost estimate."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark_root", type=pathlib.Path, required=True)
    parser.add_argument("--methods",
                        nargs="+",
                        default=list(adapter.OPENROUTER_MODELS))
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--pilot_per_dataset", type=int, default=2)
    parser.add_argument("--skip_pilot", action="store_true")
    parser.add_argument("--json_out", type=pathlib.Path, default=None)
    cli = parser.parse_args()

    api_key = adapter.read_api_key()
    key_info, balance = _read_credit(api_key)
    limit_remaining = key_info.get("limit_remaining")

    if cli.datasets:
        datasets = set(cli.datasets)
    else:
        datasets = None
    images = monodepth_images(cli.benchmark_root.expanduser(), datasets)
    counts = {dataset: len(paths) for dataset, paths in images.items()}
    print("images per dataset:", counts, "total", sum(counts.values()))
    if cli.skip_pilot:
        return 0

    report: dict = {
        "key": key_info,
        "balance": balance,
        "images": counts,
        "methods": {}
    }
    grand_total = 0.0
    for method in cli.methods:
        report["methods"][method] = _pilot_method(method, images, counts,
                                                  api_key,
                                                  cli.pilot_per_dataset)
        grand_total += report["methods"][method]["estimate_usd"]

    if limit_remaining is not None:
        available = limit_remaining
    else:
        available = balance
    report["estimate_total_usd"] = grand_total
    print(f"TOTAL estimated: ${grand_total:.2f}; available: {available}")
    if available is not None:
        if float(available) >= grand_total * 1.2:
            verdict = "enough"
        else:
            verdict = "NOT enough (need 20% headroom)"
        print(f"verdict: {verdict}")
    if cli.json_out:
        cli.json_out.write_text(json.dumps(report, indent=2, default=str),
                                encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
