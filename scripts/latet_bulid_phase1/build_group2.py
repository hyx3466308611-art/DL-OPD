#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group 2（Basic referring / 基础指代，论文 ~321K）— RefCOCO、RefCOCO+、RefCOCOg 的 train 指代句。

构建过程：
  1) 读取 data/raw/refer/{refcoco,refcoco+,refcocog}/ 下 refs pickle 与 instances.json。
  2) 仅 split==train；每个 ref 对应 COCO train2014 一张实例 mask（按 ref_id 去重存 png）。
  3) 同一 ref 的多条 sentence 各写一行 jsonl，prompt 为数据集原句（不改写）。
  4) 图像路径使用 instances.json 的 file_name（不用 ref 里带 _16 后缀的错误名）。

实现：调用 scripts/build_phase1/export_group2_ref.py。

输出位置：
  清单：data/phase1/manifests/group2_ref.jsonl
  mask：data/phase1/masks/group2_ref/g2_{dataset}_{ref_id}.png
  原图：data/raw/coco/train2014/

CLI 参数：
  --max-rows   总行数上限（默认 None = 全部 train 句，约 321327）
  --seed       截断时 shuffle 种子
  --out-jsonl  默认 group2_ref.jsonl

依赖 raw：data/raw/refer/、data/raw/coco/train2014/
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from config import G2_REF_JSONL, LEGACY_BUILD_DIR, REPO_ROOT, DEFAULT_SEED


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Group2 (RefCOCO family)")
    ap.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Cap rows (default: export 脚本导出全部 train 句，约 321K)",
    )
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out-jsonl", type=Path, default=G2_REF_JSONL)
    args = ap.parse_args()

    cmd = [
        sys.executable,
        str(LEGACY_BUILD_DIR / "export_group2_ref.py"),
        "--seed",
        str(args.seed),
        "--out-jsonl",
        str(args.out_jsonl),
    ]
    if args.max_rows is not None:
        cmd.extend(["--max-rows", str(args.max_rows)])

    print("[group2] %s" % " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(REPO_ROOT))
    print("[group2] done -> %s" % args.out_jsonl)


if __name__ == "__main__":
    main()
