#!/usr/bin/env python3
"""Rebuild near-online teacher cache index metadata (plan §9.4)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from dl_opd.teacher_cache import TeacherCache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--ttl_epochs", type=int, default=2)
    args = ap.parse_args()
    cache = TeacherCache(ttl_epochs=args.ttl_epochs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache.save_index(out / "cache_index.jsonl")
    (out / "cache_meta.json").write_text(
        json.dumps({"ttl_epochs": args.ttl_epochs, "stats": cache.stats()}, indent=2),
        encoding="utf-8",
    )
    print("initialized empty cache at", out)


if __name__ == "__main__":
    main()
