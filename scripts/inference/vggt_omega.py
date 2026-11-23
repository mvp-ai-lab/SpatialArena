#!/usr/bin/env python3
# Copyright (c) 2026 Yu Chen. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Runs spatialarena.models.vggt_omega.VGGTOmega; the engine's adapter."""

import pathlib
import sys

ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[2])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# pylint: disable-next=wrong-import-position
from spatialarena.models import vggt_omega as model_module  # noqa: E402


def main():
    model_module.VGGTOmega.main()


if __name__ == "__main__":
    main()
