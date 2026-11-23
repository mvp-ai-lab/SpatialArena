# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Writes manifests of each benchmarked method's official code and checkpoints.

Third-party code and weights are *not* re-hosted in the benchmark release:
many of them are non-commercial, "all rights reserved", unlicensed, or gated.
Instead this records where each comes from, derived from the same registries
`spatialarena/download.py` fetches with (`METHOD_CODE_REPOS`,
`GDRIVE_CHECKPOINT_URLS`, `HF_FILE_CHECKPOINTS`,
`SHIPPED_WITH_CODE_CHECKPOINTS`, `HF_EXTRA_SNAPSHOTS` and
`MethodSpec.ckpt_default`), plus the exact commit and license file of each
local code checkout, so a benchmark run can be reproduced from the original
sources.

Output (under `--output_dir`): `methods_manifest.json` and `METHODS.md`.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from spatialarena import download  # noqa: E402
from spatialarena import methods  # noqa: E402

# pylint: enable=wrong-import-position

# See spatialarena/paths.py.
SPATIALARENA_ROOT = os.path.expanduser(
    os.environ.get("SPATIALARENA_ROOT", "~/spatialarena"))
DEFAULT_CODE_ROOT = pathlib.Path(f"{SPATIALARENA_ROOT}/benchmarks/_model_codes")
# Sources for checkpoints the registry only knows as local files, each verified
# from evidence in this repo or the checkpoint itself (not guessed). Local
# checkpoints without an entry are reported as "source not recorded".
LOCAL_CHECKPOINT_SOURCES: dict[str, dict] = {
    "DA3-G": {
        "kind":
            "hf_hub",
        "repo_id":
            "depth-anything/DA3-GIANT",
        "url":
            "https://huggingface.co/depth-anything/DA3-GIANT",
        "evidence": ("model card bundled with the local checkpoint names "
                     "depth-anything/da3-giant")
    },
    "GENCEPTION_DEPTH": {
        "kind":
            "script",
        "script":
            "scripts/download/genception.sh",
        "url":
            "https://storage.googleapis.com/representations4d/checkpoints/genception"
    },
    "GENCEPTION_POSE": {
        "kind":
            "script",
        "script":
            "scripts/download/genception.sh",
        "url":
            "https://storage.googleapis.com/representations4d/checkpoints/genception"
    },
}
LICENSE_MARKERS = (
    ("Project Registration License",
     ("PRL v1.0: research use free, project use needs registration "
      "(see LICENSE)")),
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


def git_commit(path: pathlib.Path) -> str | None:
    """Returns the HEAD commit of a git checkout, or None if it has none."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=60)
    except (subprocess.SubprocessError, OSError):
        return None
    return result.stdout.strip()


def license_of(path: pathlib.Path) -> dict:
    """Identifies the license of a code checkout from its license file.

    Args:
        path: The code checkout.

    Returns:
        The license file name (`file`, None if there is none) and a short
        license label (`license`).
    """
    for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING"):
        candidate = path / name
        if candidate.is_file():
            head = candidate.read_text(encoding="utf-8",
                                       errors="replace")[:3000]
            for marker, label in LICENSE_MARKERS:
                if marker in head:
                    return {"file": name, "license": label}
            return {"file": name, "license": "see LICENSE"}
    return {
        "file": None,
        "license": "no license file (all rights reserved by default)"
    }


def checkpoint_source(spec) -> dict:
    """Describes where a method's checkpoint comes from.

    Args:
        spec: The method's `MethodSpec`.

    Returns:
        The source: its `kind` plus kind-specific keys such as `repo_id`,
        `filename`, `url` or `path_in_code`.
    """
    ckpt = spec.ckpt_default or ""
    kind = download._checkpoint_kind(ckpt)
    if spec.name in LOCAL_CHECKPOINT_SOURCES:
        return dict(LOCAL_CHECKPOINT_SOURCES[spec.name])
    if spec.name in download.HF_FILE_CHECKPOINTS:
        repo_id, filename = download.HF_FILE_CHECKPOINTS[spec.name]
        return {
            "kind": "hf_file",
            "repo_id": repo_id,
            "filename": filename,
            "url": f"https://huggingface.co/{repo_id}/blob/main/{filename}"
        }
    if spec.name in download.HF_LOCAL_FILE_CHECKPOINTS:
        repo_id, _subdir, files = download.HF_LOCAL_FILE_CHECKPOINTS[spec.name]
        return {
            "kind": "hf_file",
            "repo_id": repo_id,
            "filename": files[0],
            "url": f"https://huggingface.co/{repo_id}/blob/main/{files[0]}"
        }
    if spec.name in download.GDRIVE_CHECKPOINT_URLS:
        return {
            "kind": "google_drive",
            "url": download.GDRIVE_CHECKPOINT_URLS[spec.name]
        }
    if spec.name in download.SHIPPED_WITH_CODE_CHECKPOINTS:
        return {
            "kind": "shipped_with_code",
            "path_in_code": download.SHIPPED_WITH_CODE_CHECKPOINTS[spec.name]
        }
    if kind == "hf_hub":
        return {
            "kind": "hf_hub",
            "repo_id": ckpt,
            "url": f"https://huggingface.co/{ckpt}"
        }
    if kind == "url":
        return {"kind": "url", "url": ckpt}
    if kind == "none":
        return {"kind": "none"}
    return {
        "kind":
            "local_copy",
        "note": ("public source not recorded in this repo; see the method's "
                 "own README")
    }


def _code_entry(leaf: str, local: pathlib.Path) -> dict:
    """Describes a method's code: upstream repo, local commit and license."""
    if local.is_dir():
        commit = git_commit(local)
    else:
        commit = None
    code = {
        "leaf_dir": leaf,
        "repo_url": download.METHOD_CODE_REPOS.get(leaf),
        "commit": commit,
    }
    if local.is_dir():
        code.update(license_of(local))
    else:
        code.update({"file": None, "license": "unknown (no local checkout)"})
    return code


def build(code_root: pathlib.Path) -> list[dict]:
    """Builds the manifest entry of every registered method.

    Args:
        code_root: The folder of the local method code checkouts.

    Returns:
        One entry (method, tracks, code, checkpoint) per method.
    """
    entries = []
    for spec in methods.METHOD_SPECS:
        entry = {"method": spec.name, "tracks": list(spec.tracks)}
        leaf = download._code_leaf_dir(spec.code_default)
        entry["code"] = _code_entry(leaf, code_root / leaf)
        entry["checkpoint"] = checkpoint_source(spec)
        if spec.name in download.HF_EXTRA_SNAPSHOTS:
            also_requires = []
            for repo_id, patterns in download.HF_EXTRA_SNAPSHOTS[spec.name]:
                if patterns:
                    files = list(patterns)
                else:
                    files = "all"
                also_requires.append({
                    "kind": "hf_hub",
                    "repo_id": repo_id,
                    "files": files,
                    "url": f"https://huggingface.co/{repo_id}"
                })
            entry["checkpoint"]["also_requires"] = also_requires
        entries.append(entry)
    return entries


def render_markdown(entries: list[dict]) -> str:
    """Renders the manifest entries as the METHODS.md table."""
    lines = [
        "# Methods: official code and checkpoints\n",
        ("Third-party code and weights are **not** redistributed here. Each "
         "method is fetched from its original"),
        ("source, pinned to the commit used for these benchmark results. "
         "`spatialarena.download` automates the"),
        ("fetch (`download_all_supported_models()` / method-code cloning); "
         "gated Hugging Face repos need access"),
        ("granted to your own account first. Each project's own license "
         "applies.\n"),
        ("| Method | Code repository | Commit | Code license | Checkpoint "
         "source |"),
        "|---|---|---|---|---|",
    ]
    for entry in entries:
        code = entry.get("code") or {}
        repo = code.get("repo_url") or "(no verified upstream)"
        commit = (code.get("commit") or "")[:12] or "-"
        code_license = code.get("license", "-")
        checkpoint = entry["checkpoint"]
        source = (checkpoint.get("url") or checkpoint.get("repo_id") or
                  checkpoint.get("path_in_code") or checkpoint.get("note") or
                  checkpoint["kind"])
        lines.append(f"| {entry['method']} | {repo} | `{commit}` | "
                     f"{code_license} | {checkpoint['kind']}: {source} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Writes the manifests into `--output_dir`."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", type=pathlib.Path, required=True)
    parser.add_argument("--code_root",
                        type=pathlib.Path,
                        default=DEFAULT_CODE_ROOT)
    args = parser.parse_args()
    entries = build(args.code_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "methods_manifest.json").write_text(
        json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "METHODS.md").write_text(render_markdown(entries),
                                                encoding="utf-8")
    print(f"Wrote {len(entries)} methods to {args.output_dir}")


if __name__ == "__main__":
    main()
