#!/usr/bin/env python3
"""Merge group jsonl shards → phase1_mixed.jsonl (train.py-ready)."""
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from config import (
    DATA_ROOT,
    MANIFEST_DIR,
    PHASE1_ROOT,
    QUOTA_GROUP1_COCO,
    QUOTA_GROUP1_COCONUT,
    QUOTA_GROUP1_PACO,
    QUOTA_GROUP2,
    QUOTA_GROUP3,
)
from io_utils import read_jsonl, write_jsonl


def load_cap(path: Path, cap: Optional[int], seed: int) -> List[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = list(read_jsonl(path))
    if cap is not None and len(rows) > cap:
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[:cap]
    return rows


def train_row(row: dict) -> dict:
    return {
        "image": row["image"],
        "mask_merged": row["mask_merged"],
        "prompt": row["prompt"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-jsonl", type=Path, default=PHASE1_ROOT / "phase1_mixed.jsonl")
    ap.add_argument("--coco-cap", type=int, default=QUOTA_GROUP1_COCO)
    ap.add_argument("--coconut-cap", type=int, default=QUOTA_GROUP1_COCONUT)
    ap.add_argument("--paco-cap", type=int, default=QUOTA_GROUP1_PACO)
    ap.add_argument("--ref-cap", type=int, default=QUOTA_GROUP2)
    ap.add_argument("--g3-cap", type=int, default=QUOTA_GROUP3)
    ap.add_argument("--skip-group1", action="store_true")
    ap.add_argument("--skip-group2", action="store_true")
    ap.add_argument("--skip-group3", action="store_true")
    args = ap.parse_args()

    parts: Dict[str, List[dict]] = {}
    if not args.skip_group1:
        parts["coco"] = load_cap(MANIFEST_DIR / "group1_coco.jsonl", args.coco_cap, args.seed)
        parts["coconut"] = load_cap(
            MANIFEST_DIR / "group1_coconut.jsonl", args.coconut_cap, args.seed + 1
        )
        parts["paco"] = load_cap(MANIFEST_DIR / "group1_paco.jsonl", args.paco_cap, args.seed + 2)
    if not args.skip_group2:
        parts["ref"] = load_cap(MANIFEST_DIR / "group2_ref.jsonl", args.ref_cap, args.seed + 3)
    if not args.skip_group3:
        parts["g3"] = load_cap(
            MANIFEST_DIR / "group3_open_vocab.jsonl", args.g3_cap, args.seed + 4
        )

    merged: List[dict] = []
    stats = {}
    for name, rows in parts.items():
        stats[name] = len(rows)
        merged.extend(train_row(r) for r in rows)

    rng = random.Random(args.seed)
    rng.shuffle(merged)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n = write_jsonl(args.out_jsonl, merged)

    meta = {
        "data_root_for_train_py": str(DATA_ROOT.resolve()),
        "total_rows": n,
        "per_shard": stats,
        "out_jsonl": str(args.out_jsonl.resolve()),
        "train_command_hint": (
            "python train.py --data-dir %s --dataset-jsonl %s -- ..."
            % (
                DATA_ROOT.as_posix(),
                args.out_jsonl.relative_to(DATA_ROOT).as_posix(),
            )
        ),
    }
    meta_path = args.out_jsonl.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Mixed %d rows -> %s" % (n, args.out_jsonl))
    print("Counts:", stats)
    print("Meta:", meta_path)


if __name__ == "__main__":
    main()
