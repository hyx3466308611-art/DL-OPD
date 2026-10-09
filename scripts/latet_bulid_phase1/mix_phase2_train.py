#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase2 训练用混合清单 — Group4（对话正负）+ 从 Group1/2 回放 + Group3（cap 10000）。

说明：
  作者 HF 包内常有预混的 converseg_stage2_data/stage2_mixed.jsonl（~220K 行）。
  本脚本在 **从零重建** 且 mask 不全时，用本地 manifest **近似** 论文 Stage2 混合：
    - 正样本：group4_conversational_positive.jsonl（build_group4.py）
    - 负样本：group4_conversational_negative.jsonl
    - 回放：从 Phase1 的 coco/coconut/paco/ref 分片再采样（不含 G3，避免重复）
    - 开放词表：group3_open_vocab.jsonl，默认最多 **10000** 行

若 HF stage2_mixed.jsonl 已完整下载，可用 --use-hf-stage2-mixed 仅做「路径存在性过滤 +
可选 g3 子集 cap」（按 open_vocabulary 路径启发式截断到 10000 行）。

输出：
  data/phase2/phase2_mixed.jsonl
  data/phase2/phase2_mixed.meta.json

训练示例：
  python train.py --data-dir ./data --dataset-jsonl phase2/phase2_mixed.jsonl ...

CLI：
  --use-hf-stage2-mixed   直接过滤 HF 预混文件（不拼 group4+replay）
  --g4-pos-cap / --g4-neg-cap
  --replay-coco/coconut/paco/ref  回放各分片 cap
  --g3-cap  默认 10000
  --seed
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
    G4_NEG_JSONL,
    G4_POS_JSONL,
    HF_STAGE2_MIXED,
    PHASE2_MIXED,
    QUOTA_G3_EFFECTIVE_MASKS,
    QUOTA_PHASE2_REPLAY_COCO,
    QUOTA_PHASE2_REPLAY_COCONUT,
    QUOTA_PHASE2_REPLAY_PACO,
    QUOTA_PHASE2_REPLAY_REF,
    DEFAULT_SEED,
)
from io_utils import filter_existing_rows, read_jsonl, train_row, write_jsonl


def load_cap(path: Path, cap: Optional[int], seed: int) -> List[dict]:
    if not path.is_file():
        raise FileNotFoundError("Missing: %s" % path)
    rows = list(read_jsonl(path))
    if cap is not None and len(rows) > cap:
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[:cap]
    return rows


def is_g3_row(row: dict) -> bool:
    p = row.get("image", "") + row.get("mask_merged", "")
    return "open_vocabulary_regions_data" in p


def mix_from_hf_stage2(g3_cap: int, seed: int) -> List[dict]:
    if not HF_STAGE2_MIXED.is_file():
        raise SystemExit("HF stage2_mixed not found: %s" % HF_STAGE2_MIXED)
    rows = list(read_jsonl(HF_STAGE2_MIXED))
    # 相对 data/ 或 stage2 目录 — 统一解析
    fixed: List[dict] = []
    stage2_base = (DATA_ROOT / "ConverSeg-Training-Data" / "converseg_stage2_data").relative_to(
        DATA_ROOT
    ).as_posix()
    for item in rows:
        img = item["image"].replace("\\", "/")
        msk = item["mask_merged"].replace("\\", "/")
        if not img.startswith("ConverSeg-Training-Data/"):
            img = "%s/%s" % (stage2_base, img.lstrip("/"))
            msk = "%s/%s" % (stage2_base, msk.lstrip("/"))
        fixed.append({"image": img, "mask_merged": msk, "prompt": item["prompt"]})
    valid = filter_existing_rows(fixed)
    g3 = [r for r in valid if is_g3_row(r)]
    other = [r for r in valid if not is_g3_row(r)]
    if g3_cap is not None and len(g3) > g3_cap:
        rng = random.Random(seed)
        rng.shuffle(g3)
        g3 = g3[:g3_cap]
    merged = other + g3
    random.Random(seed).shuffle(merged)
    return [train_row(r) for r in merged]


def mix_compose(args: argparse.Namespace) -> tuple[List[dict], Dict[str, int]]:
    parts: Dict[str, List[dict]] = {}
    parts["g4_pos"] = load_cap(G4_POS_JSONL, args.g4_pos_cap, args.seed)
    parts["g4_neg"] = load_cap(G4_NEG_JSONL, args.g4_neg_cap, args.seed + 1)
    parts["replay_coco"] = load_cap(G1_COCO_JSONL, args.replay_coco, args.seed + 2)
    parts["replay_coconut"] = load_cap(G1_COCONUT_JSONL, args.replay_coconut, args.seed + 3)
    parts["replay_paco"] = load_cap(G1_PACO_JSONL, args.replay_paco, args.seed + 4)
    parts["replay_ref"] = load_cap(G2_REF_JSONL, args.replay_ref, args.seed + 5)
    parts["g3"] = load_cap(G3_OPEN_VOCAB_JSONL, args.g3_cap, args.seed + 6)

    stats = {k: len(v) for k, v in parts.items()}
    merged: List[dict] = []
    for rows in parts.values():
        merged.extend(train_row(r) for r in rows)
    random.Random(args.seed).shuffle(merged)
    return merged, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Mix Phase2 training jsonl")
    ap.add_argument("--use-hf-stage2-mixed", action="store_true")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out-jsonl", type=Path, default=PHASE2_MIXED)
    ap.add_argument("--g4-pos-cap", type=int, default=None)
    ap.add_argument("--g4-neg-cap", type=int, default=None)
    ap.add_argument("--replay-coco", type=int, default=QUOTA_PHASE2_REPLAY_COCO)
    ap.add_argument("--replay-coconut", type=int, default=QUOTA_PHASE2_REPLAY_COCONUT)
    ap.add_argument("--replay-paco", type=int, default=QUOTA_PHASE2_REPLAY_PACO)
    ap.add_argument("--replay-ref", type=int, default=QUOTA_PHASE2_REPLAY_REF)
    ap.add_argument("--g3-cap", type=int, default=QUOTA_G3_EFFECTIVE_MASKS)
    args = ap.parse_args()

    if args.use_hf_stage2_mixed:
        merged = mix_from_hf_stage2(args.g3_cap, args.seed)
        stats = {"mode": "hf_stage2_mixed_filtered", "total": len(merged)}
    else:
        merged, stats = mix_compose(args)
        stats["mode"] = "compose_g4_replay_g3"

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n = write_jsonl(args.out_jsonl, merged)
    rel_jsonl = args.out_jsonl.relative_to(DATA_ROOT).as_posix()
    meta = {
        "data_root_for_train_py": str(DATA_ROOT.resolve()),
        "total_rows": n,
        "per_shard": stats,
        "g3_cap": args.g3_cap,
        "out_jsonl": str(args.out_jsonl.resolve()),
        "train_command_hint": "python train.py --data-dir %s --dataset-jsonl %s"
        % (DATA_ROOT.as_posix(), rel_jsonl),
    }
    meta_path = args.out_jsonl.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[phase2 mix] %d rows -> %s" % (n, args.out_jsonl))
    print("[phase2 mix]", stats)


if __name__ == "__main__":
    main()
