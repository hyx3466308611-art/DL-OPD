#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group 3（Open-vocabulary regions，论文 ~48K）— Hugging Face Stage1，**不**自建 mask。

构建过程：
  1) 读取 HF 清单：
     data/ConverSeg-Training-Data/converseg_stage1_data/open_vocabulary_regions_data/
       open_vocabulary_regions_data.jsonl
  2) 将 image / mask_merged 路径改写为相对仓库 data/ 根的路径（前缀 ConverSeg-Training-Data/...）。
  3) **只保留 image 与 mask_merged 在磁盘上均存在的行**（当前 HF 导出常只有 ~1 万张 mask）。
  4) 默认导出全部有效行（约 48529）；仅调试时用 --max-rows 截断。

输出位置：
  清单：data/phase1/manifests/group3_open_vocab.jsonl
  图像/mask：**不复制**，仍位于 HF 目录下的 images/、masks/

CLI 参数：
  --max-rows        有效行上限（默认 10000；设为 0 或负数表示不截断，用全部有效行）
  --no-cap          与 --max-rows 0 等价，导出全部「文件齐全」的行
  --seed            截断 shuffle 种子
  --out-jsonl       输出路径

依赖：需先下载 HF ConverSeg-Training-Data 的 stage1 open_vocabulary 部分到 data/ 下。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import (
    DATA_ROOT,
    G3_OPEN_VOCAB_JSONL,
    HF_G3_JSONL,
    PHASE1_MANIFESTS,
    QUOTA_G3_EFFECTIVE_MASKS,
    DEFAULT_SEED,
)
from io_utils import cap_shuffle, filter_existing_rows, write_jsonl


def _resolve_hf_paths(item: dict) -> tuple[str, str]:
    stage1_base = (
        DATA_ROOT / "ConverSeg-Training-Data" / "converseg_stage1_data"
    ).relative_to(DATA_ROOT).as_posix()
    img = item["image"].replace("\\", "/")
    msk = item["mask_merged"].replace("\\", "/")
    if not img.startswith("ConverSeg-Training-Data/"):
        img = "%s/%s" % (stage1_base, img.lstrip("/"))
        msk = "%s/%s" % (stage1_base, msk.lstrip("/"))
    return img, msk


def load_hf_rows() -> list[dict]:
    if not HF_G3_JSONL.is_file():
        raise SystemExit("Missing HF Group3 jsonl: %s" % HF_G3_JSONL)
    rows: list[dict] = []
    with HF_G3_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            img, msk = _resolve_hf_paths(item)
            rows.append(
                {
                    "image": img,
                    "mask_merged": msk,
                    "prompt": item["prompt"],
                    "source": "hf_open_vocab_stage1",
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Group3 manifest (HF Stage1, filtered)")
    ap.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Cap valid rows after filter (default: all valid, paper ~%d)" % QUOTA_G3_EFFECTIVE_MASKS,
    )
    ap.add_argument(
        "--no-cap",
        action="store_true",
        help="Same as default: keep all rows with existing image+mask",
    )
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out-jsonl", type=Path, default=G3_OPEN_VOCAB_JSONL)
    args = ap.parse_args()

    raw_rows = load_hf_rows()
    valid = filter_existing_rows(raw_rows)
    skipped = len(raw_rows) - len(valid)
    print(
        "[group3] HF jsonl rows=%d, valid (files exist)=%d, skipped=%d"
        % (len(raw_rows), len(valid), skipped)
    )

    cap = None if args.no_cap or args.max_rows is None or args.max_rows <= 0 else args.max_rows
    if cap is not None:
        valid = cap_shuffle(valid, cap, args.seed)

    PHASE1_MANIFESTS.mkdir(parents=True, exist_ok=True)
    n = write_jsonl(args.out_jsonl, valid)
    print("[group3] wrote %d rows -> %s" % (n, args.out_jsonl))


if __name__ == "__main__":
    main()
