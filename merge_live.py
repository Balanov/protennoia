#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge live-model measurement files into one results document.

A long measurement run can fail on a single model (memory pressure, a transient
server error), and re-running the whole sweep is expensive. This script merges
the per-model files back into the canonical results file, refusing to overwrite
an existing measurement of the same model unless --replace is given.

    python merge_live.py results/ollama_matrix.json results/ollama_matrix_27b.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--replace", action="store_true",
                    help="overwrite an existing measurement of the same model")
    a = ap.parse_args(argv)

    with open(a.target, encoding="utf-8") as fh:
        target = json.load(fh)
    target.setdefault("systems", {})

    added, skipped, replaced = [], [], []
    for src in a.sources:
        if not os.path.exists(src):
            print(f"skip (missing): {src}")
            continue
        with open(src, encoding="utf-8") as fh:
            data = json.load(fh)
        for name, profile in data.get("systems", {}).items():
            if name in target["systems"]:
                if a.replace:
                    target["systems"][name] = profile
                    replaced.append(name)
                else:
                    skipped.append(name)
                continue
            target["systems"][name] = profile
            added.append(name)

    with open(a.target, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(target, fh, ensure_ascii=False, indent=2, sort_keys=False)
        fh.write("\n")

    print(f"target: {a.target}")
    print(f"added:   {added}")
    if replaced:
        print(f"replaced: {replaced}")
    if skipped:
        print(f"skipped (already present, use --replace to overwrite): {skipped}")
    print(f"systems in file: {len(target['systems'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
