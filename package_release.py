#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Package a release archive of the harness for deposition.

The archive contains the repository contents at the root (no wrapping folder),
which is the layout Zenodo and GitHub releases both render well.

    python package_release.py --version 0.1.3 --out ../10_submission/protennoia-v0.1.3.zip
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

INCLUDE_FILES = [
    ".gitignore",
    "CHANGELOG.md",
    "CITATION.cff",
    "LICENSE",
    "README.md",
    "requirements.txt",
    "merge_live.py",
    "package_release.py",
]
INCLUDE_DIRS = ["protennoia", "tests", "docs", "results", "paper"]
EXCLUDE_DIRS = {"__pycache__", ".git", ".mypy_cache", ".pytest_cache"}
EXCLUDE_SUFFIXES = (".pyc", ".pyo", ".zip", ".log", ".tmp")


def add_tree(zf: zipfile.ZipFile, root: str, prefix: str = ""):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS)
        rel_dir = os.path.relpath(dirpath, root).replace("\\", "/")
        if rel_dir == ".":
            rel_dir = ""
        for name in sorted(filenames):
            if name.endswith(EXCLUDE_SUFFIXES):
                continue
            full = os.path.join(dirpath, name)
            arc = f"{prefix}{rel_dir}/{name}" if rel_dir else f"{prefix}{name}"
            zf.write(full, arc)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, help="e.g. 0.1.3")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    out = a.out or os.path.join(ROOT, "10_submission", f"protennoia-v{a.version}.zip")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in INCLUDE_FILES:
            full = os.path.join(HERE, name)
            if os.path.exists(full):
                zf.write(full, name)
        for d in INCLUDE_DIRS:
            full = os.path.join(HERE, d)
            if os.path.isdir(full):
                add_tree(zf, full, prefix=f"{d}/")

    size = os.path.getsize(out)
    with zipfile.ZipFile(out) as zf:
        entries = zf.namelist()
    print(f"wrote {out}")
    print(f"entries: {len(entries)}, bytes: {size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
