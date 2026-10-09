#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终验收：统计并抽查所有 Phase1/2 相关 jsonl（行数 + 路径是否存在）。

用法（仓库根）：
  python scripts/latet_bulid_phase1/check_all_jsonl.py
  python scripts/latet_bulid_phase1/check_all_jsonl.py --full   # 全量逐行查路径（慢）

退出码 0 = 通过；1 = 存在 FAIL。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

from config import (
    DATA_ROOT,
    G1_COCO_JSONL,
    G1_COCONUT_JSONL,
    G1_PACO_JSONL,
    G2_REF_JSONL,
    G3_OPEN_VOCAB_JSONL,
    G4_NEG_JSONL,
    G4_POS_JSONL,
    PACO_MIN_AREA,
    PHASE1_MIXED,
    PHASE2_MIXED,
    QUOTA_G1_COCO,
    QUOTA_G1_COCONUT,
    QUOTA_G1_PACO_USED,
    QUOTA_G2_REF,
    QUOTA_G3_PAPER,
    REPO_ROOT,
)
from io_utils import read_jsonl, row_files_exist

# Phase1 mix 期望（与 mix_phase1_train 默认 cap 一致）
EXPECTED_PHASE1_TOTAL = (
    QUOTA_G1_COCO + QUOTA_G1_COCONUT + QUOTA_G1_PACO_USED + QUOTA_G2_REF + QUOTA_G3_PAPER
)


class R:
    fails: List[str] = []
    warns: List[str] = []

    @classmethod
    def ok(cls, msg: str) -> None:
        print("  [OK]   %s" % msg)

    @classmethod
    def warn(cls, msg: str) -> None:
        cls.warns.append(msg)
        print("  [WARN] %s" % msg)

    @classmethod
    def fail(cls, msg: str) -> None:
        cls.fails.append(msg)
        print("  [FAIL] %s" % msg)


def count_lines(p: Path) -> int:
    if not p.is_file():
        R.fail("missing file: %s" % p.relative_to(REPO_ROOT))
        return 0
    n = sum(1 for _ in read_jsonl(p))
    return n


def check_paths_sample(
    p: Path, label: str, sample: int, full: bool
) -> Tuple[int, int, int]:
    """Return (rows, missing_image, missing_mask)."""
    if not p.is_file():
        return 0, 0, 0
    rows = list(read_jsonl(p))
    n = len(rows)
    if full or sample <= 0 or n <= sample:
        check = rows
    else:
        rng = random.Random(42)
        check = rng.sample(rows, sample)
    miss_i = miss_m = 0
    for row in check:
        ip = DATA_ROOT / str(row["image"]).replace("\\", "/")
        mp = DATA_ROOT / str(row["mask_merged"]).replace("\\", "/")
        if not ip.is_file():
            miss_i += 1
        if not mp.is_file():
            miss_m += 1
    scope = len(check)
    if miss_i or miss_m:
        R.fail(
            "%s: checked %d rows, missing image=%d mask=%d"
            % (label, scope, miss_i, miss_m)
        )
    else:
        R.ok("%s: path check (%d/%d rows) OK" % (label, scope, n))
    return n, miss_i, miss_m


def check_shard(
    p: Path,
    label: str,
    expect: Optional[int],
    exact: bool,
    sample: int,
    full: bool,
) -> int:
    print("\n--- %s ---" % label)
    n = count_lines(p)
    if n == 0 and not p.is_file():
        return 0
    print("  [..]   path: %s" % p.relative_to(REPO_ROOT))
    print("  [..]   lines: %d" % n)
    if expect is not None:
        if exact and n != expect:
            R.fail("%s: expected exactly %d, got %d" % (label, expect, n))
        elif not exact and n < expect:
            R.fail("%s: expected >= %d, got %d" % (label, expect, n))
        else:
            R.ok("%s line count OK (target %d)" % (label, expect))
    check_paths_sample(p, label, sample, full)
    return n


def check_mixed_phase1(sample: int, full: bool) -> None:
    print("\n=== phase1_mixed.jsonl ===")
    n = check_shard(
        PHASE1_MIXED,
        "phase1_mixed",
        EXPECTED_PHASE1_TOTAL,
        True,
        sample,
        full,
    )
    if n and n != EXPECTED_PHASE1_TOTAL:
        R.fail("phase1 total %d != expected %d" % (n, EXPECTED_PHASE1_TOTAL))
    meta = PHASE1_MIXED.with_suffix(".meta.json")
    if meta.is_file():
        m = json.loads(meta.read_text(encoding="utf-8"))
        print("  [..]   meta per_shard:", m.get("per_shard"))
        ps = m.get("per_shard") or {}
        targets = {
            "coco": QUOTA_G1_COCO,
            "coconut": QUOTA_G1_COCONUT,
            "paco": QUOTA_G1_PACO_USED,
            "ref": QUOTA_G2_REF,
            "g3": QUOTA_G3_PAPER,
        }
        for k, exp in targets.items():
            if k in ps and ps[k] != exp:
                R.fail("meta per_shard[%s]=%s expected %d" % (k, ps[k], exp))
            elif k in ps:
                R.ok("meta %s=%d" % (k, ps[k]))


def check_mixed_phase2(sample: int, full: bool) -> None:
    print("\n=== phase2_mixed.jsonl ===")
    n = count_lines(PHASE2_MIXED)
    if not PHASE2_MIXED.is_file():
        R.fail("missing phase2_mixed.jsonl")
        return
    print("  [..]   lines: %d" % n)
    meta = PHASE2_MIXED.with_suffix(".meta.json")
    if meta.is_file():
        m = json.loads(meta.read_text(encoding="utf-8"))
        print("  [..]   meta per_shard:", m.get("per_shard"))
        ps = m.get("per_shard") or {}
        if ps.get("g3") != QUOTA_G3_PAPER:
            R.fail("phase2 meta g3=%s expected %d" % (ps.get("g3"), QUOTA_G3_PAPER))
        else:
            R.ok("phase2 meta g3=%d" % QUOTA_G3_PAPER)
    check_paths_sample(PHASE2_MIXED, "phase2_mixed", sample, full)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--full",
        action="store_true",
        help="Check every row paths (slow for ref/mixed)",
    )
    ap.add_argument(
        "--sample",
        type=int,
        default=2000,
        help="Random sample size per jsonl for path check (default 2000)",
    )
    args = ap.parse_args()

    print("REPO:", REPO_ROOT)
    print("PACO policy: min-area=%d, used rows=%d" % (PACO_MIN_AREA, QUOTA_G1_PACO_USED))
    print("Expected phase1_mixed total:", EXPECTED_PHASE1_TOTAL)

    check_shard(G1_COCO_JSONL, "group1_coco", QUOTA_G1_COCO, True, args.sample, args.full)
    check_shard(
        G1_COCONUT_JSONL, "group1_coconut", QUOTA_G1_COCONUT, True, args.sample, args.full
    )
    check_shard(
        G1_PACO_JSONL,
        "group1_paco",
        QUOTA_G1_PACO_USED,
        True,
        args.sample,
        args.full,
    )
    check_shard(G2_REF_JSONL, "group2_ref", QUOTA_G2_REF, False, args.sample, args.full)
    check_shard(
        G3_OPEN_VOCAB_JSONL, "group3_open_vocab", QUOTA_G3_PAPER, True, args.sample, args.full
    )
    check_shard(G4_POS_JSONL, "group4_pos", None, False, args.sample, args.full)
    check_shard(G4_NEG_JSONL, "group4_neg", None, False, args.sample, args.full)

    check_mixed_phase1(args.sample, args.full)
    check_mixed_phase2(args.sample, args.full)

    print("\n=== Summary ===")
    if R.warns:
        for w in R.warns:
            print("  WARN:", w)
    if R.fails:
        for f in R.fails:
            print("  FAIL:", f)
        sys.exit(1)
    print("  All jsonl checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
