# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Write manifests of every benchmarked method's official code and checkpoint sources.

Third-party code and weights are *not* re-hosted in the benchmark release: many of them are non-commercial,
"all rights reserved", unlicensed, or gated. Instead this records where each comes from, derived from the same
registries ``spatialarena/download.py`` fetches with (``METHOD_CODE_REPOS``, ``GDRIVE_CHECKPOINT_URLS``,
``HF_FILE_CHECKPOINTS``, ``SHIPPED_WITH_CODE_CHECKPOINTS``, ``HF_EXTRA_SNAPSHOTS`` and ``MethodSpec.ckpt_default``),
plus the exact commit and license file of each local code checkout, so a benchmark run can be reproduced from the
original sources.

Output (under ``--output_dir``): ``methods_manifest.json`` and ``METHODS.md``.
"""

from __future__ import annotations

import os
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatialarena.download import (  # noqa: E402
    GDRIVE_CHECKPOINT_URLS,
    HF_EXTRA_SNAPSHOTS,
    HF_FILE_CHECKPOINTS,
    HF_LOCAL_FILE_CHECKPOINTS,
    METHOD_CODE_REPOS,
    SHIPPED_WITH_CODE_CHECKPOINTS,
    _checkpoint_kind,
    _code_leaf_dir,
)
from spatialarena.methods import METHOD_SPECS  # noqa: E402

# see spatialarena/paths.py
SPATIALARENA_ROOT = os.path.expanduser(os.environ.get("SPATIALARENA_ROOT", "~/spatialarena"))
DEFAULT_CODE_ROOT = Path(f"{SPATIALARENA_ROOT}/benchmarks/_model_codes")
# Sources for checkpoints the registry only knows as local files, each verified from evidence in this repo or the
# checkpoint itself (not guessed). Local checkpoints without an entry are reported as "source not recorded".
LOCAL_CHECKPOINT_SOURCES: dict[str, dict] = {
    "DA3-G": {"kind": "hf_hub", "repo_id": "depth-anything/DA3-GIANT",
              "url": "https://huggingface.co/depth-anything/DA3-GIANT",
              "evidence": "model card bundled with the local checkpoint names depth-anything/da3-giant"},
    "GENCEPTION_DEPTH": {"kind": "script", "script": "scripts/download/genception.sh",
                         "url": "https://storage.googleapis.com/representations4d/checkpoints/genception"},
    "GENCEPTION_POSE": {"kind": "script", "script": "scripts/download/genception.sh",
                        "url": "https://storage.googleapis.com/representations4d/checkpoints/genception"},
}
LICENSE_MARKERS = (
    ("Project Registration License", "PRL v1.0: research use free, project use needs registration (see LICENSE)"),
    ("VGGT License", "VGGT License (see LICENSE)"),
    ("Apache License", "Apache-2.0"),
    ("MIT License", "MIT"),
    ("Attribution-NonCommercial-ShareAlike", "CC BY-NC-SA 4.0"),
    ("Attribution-NonCommercial", "CC BY-NC 4.0"),
    ("Noncommercial", "non-commercial (see LICENSE)"),
    ("NonCommercial", "non-commercial (see LICENSE)"),
    ("non-commercial", "non-commercial (see LICENSE)"),
    ("All rights reserved", "all rights reserved (see LICENSE)"),
    ("All Rights Reserved", "all rights reserved (see LICENSE)"),
)


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=60
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def license_of(path: Path) -> dict:
    for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING"):
        candidate = path / name
        if candidate.is_file():
            head = candidate.read_text(encoding="utf-8", errors="replace")[:3000]
            for marker, label in LICENSE_MARKERS:
                if marker in head:
                    return {"file": name, "license": label}
            return {"file": name, "license": "see LICENSE"}
    return {"file": None, "license": "no license file (all rights reserved by default)"}


def checkpoint_source(spec) -> dict:
    ckpt = spec.ckpt_default or ""
    kind = _checkpoint_kind(ckpt)
    if spec.name in LOCAL_CHECKPOINT_SOURCES:
        return dict(LOCAL_CHECKPOINT_SOURCES[spec.name])
    if spec.name in HF_FILE_CHECKPOINTS:
        repo_id, filename = HF_FILE_CHECKPOINTS[spec.name]
        return {"kind": "hf_file", "repo_id": repo_id, "filename": filename,
                "url": f"https://huggingface.co/{repo_id}/blob/main/{filename}"}
    if spec.name in HF_LOCAL_FILE_CHECKPOINTS:
        repo_id, _subdir, files = HF_LOCAL_FILE_CHECKPOINTS[spec.name]
        return {"kind": "hf_file", "repo_id": repo_id, "filename": files[0],
                "url": f"https://huggingface.co/{repo_id}/blob/main/{files[0]}"}
    if spec.name in GDRIVE_CHECKPOINT_URLS:
        return {"kind": "google_drive", "url": GDRIVE_CHECKPOINT_URLS[spec.name]}
    if spec.name in SHIPPED_WITH_CODE_CHECKPOINTS:
        return {"kind": "shipped_with_code", "path_in_code": SHIPPED_WITH_CODE_CHECKPOINTS[spec.name]}
    if kind == "hf_hub":
        return {"kind": "hf_hub", "repo_id": ckpt, "url": f"https://huggingface.co/{ckpt}"}
    if kind == "url":
        return {"kind": "url", "url": ckpt}
    if kind == "none":
        return {"kind": "none"}
    return {"kind": "local_copy", "note": "public source not recorded in this repo; see the method's own README"}


def build(code_root: Path) -> list[dict]:
    entries = []
    for spec in METHOD_SPECS:
        entry = {"method": spec.name, "tracks": list(spec.tracks)}
        leaf = _code_leaf_dir(spec.code_default)
        local = code_root / leaf
        entry["code"] = {
            "leaf_dir": leaf,
            "repo_url": METHOD_CODE_REPOS.get(leaf),
            "commit": git_commit(local) if local.is_dir() else None,
            **(license_of(local) if local.is_dir() else {"file": None, "license": "unknown (no local checkout)"}),
        }
        entry["checkpoint"] = checkpoint_source(spec)
        if spec.name in HF_EXTRA_SNAPSHOTS:
            entry["checkpoint"]["also_requires"] = [
                {"kind": "hf_hub", "repo_id": repo_id, "files": list(patterns) if patterns else "all",
                 "url": f"https://huggingface.co/{repo_id}"}
                for repo_id, patterns in HF_EXTRA_SNAPSHOTS[spec.name]
            ]
        entries.append(entry)
    return entries


def render_markdown(entries: list[dict]) -> str:
    lines = [
        "# Methods: official code and checkpoints\n",
        "Third-party code and weights are **not** redistributed here. Each method is fetched from its original",
        "source, pinned to the commit used for these benchmark results. `spatialarena.download` automates the",
        "fetch (`download_all_supported_models()` / method-code cloning); gated Hugging Face repos need access",
        "granted to your own account first. Each project's own license applies.\n",
        "| Method | Code repository | Commit | Code license | Checkpoint source |",
        "|---|---|---|---|---|",
    ]
    for e in entries:
        code = e.get("code") or {}
        repo = code.get("repo_url") or "(no verified upstream)"
        commit = (code.get("commit") or "")[:12] or "-"
        lic = code.get("license", "-")
        ck = e["checkpoint"]
        src = ck.get("url") or ck.get("repo_id") or ck.get("path_in_code") or ck.get("note") or ck["kind"]
        lines.append(f"| {e['method']} | {repo} | `{commit}` | {lic} | {ck['kind']}: {src} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--code_root", type=Path, default=DEFAULT_CODE_ROOT)
    args = parser.parse_args()
    entries = build(args.code_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "methods_manifest.json").write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "METHODS.md").write_text(render_markdown(entries), encoding="utf-8")
    print(f"Wrote {len(entries)} methods to {args.output_dir}")


if __name__ == "__main__":
    main()
