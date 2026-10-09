#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group 1（Literal / 字面概念，论文 Sec. 6.2.1）— 从零构建 Phase1 中的 COCO + COCONut + PACO 三份清单与 mask。

构建过程（与论文对齐的实现，非作者未公开的 Data Engine 源码）：
  1) COCO train2017 instances：按 (image_id, category_id) 合并该图该类所有实例为一张二值 mask；
     prompt 固定为「Segment all the {category} in the image」；默认导出 200_000 行。
  2) COCONut-S panoptic：同样按 (image, category) 合并 segment；主档面积 ≥768px，小档 512–768 约 8k 配额；
     目标 **140_000** 行（--coconut-target）；prompt 同上。
  3) PACO train：按 (image, part_category) 合并 part 标注；prompt「Segment all the {part} in the image」；
     默认 100_000 行。

实现方式：调用仓库内已验证的 scripts/build_phase1/export_group1_*.py（避免重复维护 mask 逻辑）。

输出位置（相对仓库根）：
  清单：data/phase1/manifests/group1_coco.jsonl
        data/phase1/manifests/group1_coconut.jsonl
        data/phase1/manifests/group1_paco.jsonl
  mask：data/phase1/masks/group1_coco/   命名 g1_coco_{imageId}_{catId}.png
        data/phase1/masks/group1_coconut/  g1_cnut_{imageId}_{catId}.png
        data/phase1/masks/group1_paco/     g1_paco_{imageId}_{catId}.png
  原图：仍在 data/raw/coco/train2017/（不复制）

主要 CLI 参数：
  --skip-coco / --skip-coconut / --skip-paco   跳过某一子源
  --coco-max-rows      默认 config.QUOTA_G1_COCO (200000)
  --coconut-target     默认 config.QUOTA_G1_COCONUT (140000)
  --paco-max-rows      默认 config.QUOTA_G1_PACO (100000)
  --seed               随机打乱取 cap 的种子（默认 42）
  --coconut-resume     COCONut 断点续导（传给 export 脚本）

依赖 raw：
  data/raw/coco/...、data/raw/coconut/...、data/raw/paco/...
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from config import (
    G1_COCO_JSONL,
    G1_COCONUT_JSONL,
    G1_PACO_JSONL,
    LEGACY_BUILD_DIR,
    QUOTA_G1_COCO,
    QUOTA_G1_COCONUT,
    QUOTA_G1_PACO,
    REPO_ROOT,
    DEFAULT_SEED,
)


def _run(script: str, extra: list[str]) -> None:
    cmd = [sys.executable, str(LEGACY_BUILD_DIR / script)] + extra
    print("[group1] %s" % " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Group1 (COCO + COCONut + PACO)")
    ap.add_argument("--skip-coco", action="store_true")
    ap.add_argument("--skip-coconut", action="store_true")
    ap.add_argument("--skip-paco", action="store_true")
    ap.add_argument("--coco-max-rows", type=int, default=QUOTA_G1_COCO)
    ap.add_argument("--coconut-target", type=int, default=QUOTA_G1_COCONUT)
    ap.add_argument("--paco-max-rows", type=int, default=QUOTA_G1_PACO)
    ap.add_argument(
        "--paco-min-area",
        type=int,
        default=512,
        help="PACO merged mask min pixels (lower if <100k candidates at 512)",
    )
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--coconut-resume", action="store_true")
    args = ap.parse_args()

    if not args.skip_coco:
        _run(
            "export_group1_coco.py",
            [
                "--max-rows",
                str(args.coco_max_rows),
                "--seed",
                str(args.seed),
                "--out-jsonl",
                str(G1_COCO_JSONL),
            ],
        )
    if not args.skip_coconut:
        coconut_args = [
            "--target-rows",
            str(args.coconut_target),
            "--seed",
            str(args.seed),
            "--out-jsonl",
            str(G1_COCONUT_JSONL),
        ]
        if args.coconut_resume:
            coconut_args.append("--resume")
        _run("export_group1_coconut.py", coconut_args)
    if not args.skip_paco:
        _run(
            "export_group1_paco.py",
            [
                "--max-rows",
                str(args.paco_max_rows),
                "--min-area",
                str(args.paco_min_area),
                "--seed",
                str(args.seed),
                "--out-jsonl",
                str(G1_PACO_JSONL),
            ],
        )
    print("[group1] done. manifests under data/phase1/manifests/, masks under data/phase1/masks/group1_*")


if __name__ == "__main__":
    main()
