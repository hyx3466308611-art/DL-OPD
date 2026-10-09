#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase1 训练用混合清单 — Group1（COCO+COCONut+PACO）+ Group2（Ref）+ Group3（HF Stage1）。

融合过程：
  1) 从各分片 manifest 读入 jsonl（需先运行 build_group1/2/3.py）。
  2) 按 cap 对每一分片 shuffle 后截断（默认论文规模；Group3 默认 **48529**）。
  3) 合并为 train.py 所需三字段：image, mask_merged, prompt。
  4) 全局 shuffle，写出 phase1_mixed.jsonl 与 .meta.json（含各分片行数统计）。

输出：
  data/phase1/phase1_mixed.jsonl
  data/phase1/phase1_mixed.meta.json

训练示例：
  python train.py --data-dir ./data --dataset-jsonl phase1/phase1_mixed.jsonl ...

CLI 参数（均可覆盖）：
  --coco-cap       默认 200000
  --coconut-cap    默认 140000
  --paco-cap       默认 100000
  --ref-cap        默认 321327
  --g3-cap         默认 48529（与 Stage1 open_vocab 论文规模一致）
  --seed           默认 42
  --skip-group1/2/3
  --out-jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from config import (
    DATA_ROOT,
    G1_COCO_JSONL,
    G1_COCONUT_JSONL,
    G1_PACO_JSONL,
    G2_REF_JSONL,
    G3_OPEN_VOCAB_JSONL,
    PHASE1_MIXED,
    QUOTA_G1_COCO,
    QUOTA_G1_COCONUT,
    QUOTA_G1_PACO,
    QUOTA_G1_PACO_USED,
    QUOTA_G2_REF,
    QUOTA_G3_EFFECTIVE_MASKS,
    DEFAULT_SEED,
)
from io_utils import read_jsonl, train_row, write_jsonl


def load_cap(path: Path, cap: Optional[int], seed: int) -> List[dict]:
    if not path.is_file():
        raise FileNotFoundError("Missing manifest: %s (run build_group*.py first)" % path)
    rows = list(read_jsonl(path))
    if cap is not None and len(rows) > cap:
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[:cap]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Mix Phase1 training jsonl")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out-jsonl", type=Path, default=PHASE1_MIXED)
    ap.add_argument("--coco-cap", type=int, default=QUOTA_G1_COCO)
    ap.add_argument("--coconut-cap", type=int, default=QUOTA_G1_COCONUT)
    ap.add_argument("--paco-cap", type=int, default=QUOTA_G1_PACO_USED)
    ap.add_argument("--ref-cap", type=int, default=QUOTA_G2_REF)
    ap.add_argument("--g3-cap", type=int, default=QUOTA_G3_EFFECTIVE_MASKS)
    ap.add_argument("--skip-group1", action="store_true")
    ap.add_argument("--skip-group2", action="store_true")
    ap.add_argument("--skip-group3", action="store_true")
    args = ap.parse_args()

    parts: Dict[str, List[dict]] = {}
    if not args.skip_group1:
        parts["coco"] = load_cap(G1_COCO_JSONL, args.coco_cap, args.seed)
        parts["coconut"] = load_cap(G1_COCONUT_JSONL, args.coconut_cap, args.seed + 1)
        parts["paco"] = load_cap(G1_PACO_JSONL, args.paco_cap, args.seed + 2)
    if not args.skip_group2:
        parts["ref"] = load_cap(G2_REF_JSONL, args.ref_cap, args.seed + 3)
    if not args.skip_group3:
        parts["g3"] = load_cap(G3_OPEN_VOCAB_JSONL, args.g3_cap, args.seed + 4)

    merged: List[dict] = []
    stats = {}
    for name, rows in parts.items():
        stats[name] = len(rows)
        merged.extend(train_row(r) for r in rows)

    random.Random(args.seed).shuffle(merged)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n = write_jsonl(args.out_jsonl, merged)

    rel_jsonl = args.out_jsonl.relative_to(DATA_ROOT).as_posix()
    meta = {
        "data_root_for_train_py": str(DATA_ROOT.resolve()),
        "total_rows": n,
        "per_shard": stats,
        "g3_cap_note": "Default g3-cap=%d (paper Stage1 open_vocab)" % QUOTA_G3_EFFECTIVE_MASKS,
        "out_jsonl": str(args.out_jsonl.resolve()),
        "train_command_hint": "python train.py --data-dir %s --dataset-jsonl %s"
        % (DATA_ROOT.as_posix(), rel_jsonl),
    }
    meta_path = args.out_jsonl.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[phase1 mix] %d rows -> %s" % (n, args.out_jsonl))
    print("[phase1 mix] counts:", stats)
    print("[phase1 mix] meta:", meta_path)


if __name__ == "__main__":
    main()
