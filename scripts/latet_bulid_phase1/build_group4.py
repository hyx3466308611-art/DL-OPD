#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group 4（Conversational，论文 ~106K 正 + ~106K 负）— Hugging Face Stage2，**不**自建 Data Engine。

构建过程：
  1) 读取 HF Stage2 对话式正负清单：
     conversational_positive_data/conversational_positive_data.jsonl
     conversational_negative_data/conversational_negative_data.jsonl
  2) jsonl 内路径相对 converseg_stage2_data/（含 conversational_*_data/ 前缀），
     改写为 data/ 下 ConverSeg-Training-Data/converseg_stage2_data/...（勿重复拼子目录名）。
  3) 只保留 image、mask_merged 文件均存在的行（负样本为空 mask 文件，但仍需在磁盘上）。
  4) 可选 --max-rows 对正负分别截断。

**Group4 不参与 Phase1 训练**；清单供 Phase2 混合脚本 mix_phase2_train.py 使用。
若 HF 已提供完整 stage2_mixed.jsonl 且 mask 齐全，也可直接训练该文件（见 README）。

输出位置：
  data/phase2/manifests/group4_conversational_positive.jsonl
  data/phase2/manifests/group4_conversational_negative.jsonl
  图像/mask：仍在 HF 的 conversational_*_data/images|masks/

CLI：
  --pos-max-rows / --neg-max-rows   默认 None（全部有效行）；可设为论文 106127 / 106125
  --seed
  --skip-positive / --skip-negative
"""
from __future__ import annotations

import argparse
import json

from config import (
    G4_NEG_JSONL,
    G4_POS_JSONL,
    HF_G4_NEG_JSONL,
    HF_G4_POS_JSONL,
    PHASE2_MANIFESTS,
    DEFAULT_SEED,
)
from hf_paths import rewrite_stage2
from io_utils import cap_shuffle, filter_existing_rows, write_jsonl


def _row_from_hf_item(item: dict) -> dict:
    img, msk = rewrite_stage2(item)
    return {
        "image": img,
        "mask_merged": msk,
        "prompt": item["prompt"],
        "source": item.get("source", "hf_conversational"),
    }


def _export_split(
    hf_jsonl: Path,
    out_jsonl: Path,
    max_rows: int | None,
    seed: int,
    label: str,
) -> None:
    if not hf_jsonl.is_file():
        raise SystemExit("Missing HF Group4 jsonl (%s): %s" % (label, hf_jsonl))
    rows = []
    with hf_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            rows.append(_row_from_hf_item(item))
    valid = filter_existing_rows(rows)
    print(
        "[group4:%s] hf=%d valid=%d skipped=%d"
        % (label, len(rows), len(valid), len(rows) - len(valid))
    )
    if max_rows is not None:
        valid = cap_shuffle(valid, max_rows, seed)
    PHASE2_MANIFESTS.mkdir(parents=True, exist_ok=True)
    n = write_jsonl(out_jsonl, valid)
    print("[group4:%s] wrote %d -> %s" % (label, n, out_jsonl))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Group4 manifests (HF Stage2)")
    ap.add_argument("--pos-max-rows", type=int, default=None)
    ap.add_argument("--neg-max-rows", type=int, default=None)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--skip-positive", action="store_true")
    ap.add_argument("--skip-negative", action="store_true")
    args = ap.parse_args()

    if not args.skip_positive:
        _export_split(
            HF_G4_POS_JSONL,
            G4_POS_JSONL,
            args.pos_max_rows,
            args.seed,
            "positive",
        )
    if not args.skip_negative:
        _export_split(
            HF_G4_NEG_JSONL,
            G4_NEG_JSONL,
            args.neg_max_rows,
            args.seed + 1,
            "negative",
        )
    print("[group4] done. manifests under data/phase2/manifests/")


if __name__ == "__main__":
    main()
