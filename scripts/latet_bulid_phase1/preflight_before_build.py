#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建前自检：确认 run_build_all.sh 所需 raw / HF / 环境 / 磁盘 / 可写目录均就绪。

用法（在仓库根目录）：
  python scripts/latet_bulid_phase1/preflight_before_build.py

退出码：0 = 可以开跑；1 = 存在 [FAIL]，请先修复再构建。
[WARN] 不阻断退出（例如 G3 mask 未满 48529，但已有 >= g3-min-valid）。
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from config import (
    COCO_INST2017,
    COCO_TRAIN2014,
    COCO_TRAIN2017,
    COCONUT_JSON,
    COCONUT_MASK_DIR,
    DATA_ROOT,
    HF_G4_NEG_JSONL,
    HF_G4_POS_JSONL,
    HF_G3_JSONL,
    HF_G3_ROOT,
    HF_STAGE2,
    LEGACY_BUILD_DIR,
    PHASE1_ROOT,
    PHASE2_ROOT,
    QUOTA_G3_EFFECTIVE_MASKS,
    REPO_ROOT,
)
from hf_paths import rewrite_stage1, rewrite_stage2

REF_SPECS = [
    ("refcoco", "refs(unc).p"),
    ("refcoco+", "refs(unc).p"),
    ("refcocog", "refs(umd).p"),
]

LEGACY_EXPORT_SCRIPTS = [
    "export_group1_coco.py",
    "export_group1_coconut.py",
    "export_group1_paco.py",
    "export_group2_ref.py",
    "mask_coco.py",
    "group1_common.py",
]

PYTHON_MODULES = ("tqdm", "numpy", "PIL")

# Group1+2 导出 mask 体积大，建议剩余空间（GiB）
MIN_DISK_GIB_FAIL = 80
MIN_DISK_GIB_WARN = 120
# 构建计划使用的 G3 有效行数下限（与 --max-rows 10000 对齐）
G3_MIN_VALID_DEFAULT = QUOTA_G3_EFFECTIVE_MASKS
# G4 至少应能筛出的有效行（各一半即允许 Phase2 明显缩水但非空）
G4_MIN_VALID_EACH = 1_000


class Results:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def ok(self, msg: str) -> None:
        print("  [OK]   %s" % msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print("  [WARN] %s" % msg)

    def fail(self, msg: str) -> None:
        self.failures.append(msg)
        print("  [FAIL] %s" % msg)


def count_files(d: Path, pattern: str) -> int:
    if not d.is_dir():
        return -1
    return sum(1 for _ in d.glob(pattern))


def must_file(r: Results, p: Path, label: str) -> bool:
    if p.is_file():
        r.ok("%s (%s)" % (label, p.relative_to(REPO_ROOT)))
        return True
    r.fail("%s 缺失: %s" % (label, p.relative_to(REPO_ROOT)))
    return False


def must_dir(r: Results, p: Path, label: str) -> bool:
    if p.is_dir():
        r.ok("%s (%s/)" % (label, p.relative_to(REPO_ROOT)))
        return True
    r.fail("%s 缺失目录: %s" % (label, p.relative_to(REPO_ROOT)))
    return False


def check_python(r: Results) -> None:
    print("\n=== Python 环境 ===")
    r.ok("executable: %s" % sys.executable)
    r.ok("version: %s" % sys.version.split()[0])
    for mod in PYTHON_MODULES:
        try:
            __import__(mod)
            r.ok("import %s" % mod)
        except ImportError:
            r.fail("无法 import %s（请在 converseg 环境中 pip install）" % mod)


def check_repo_scripts(r: Results) -> None:
    print("\n=== 构建脚本 ===")
    bd = REPO_ROOT / "scripts" / "latet_bulid_phase1"
    for name in (
        "build_group1.py",
        "build_group2.py",
        "build_group3.py",
        "build_group4.py",
        "mix_phase1_train.py",
        "mix_phase2_train.py",
        "run_build_all.sh",
    ):
        p = bd / name
        if p.is_file():
            r.ok(name)
        else:
            r.fail("缺少 %s" % p.relative_to(REPO_ROOT))
    if not LEGACY_BUILD_DIR.is_dir():
        r.fail("缺少 legacy 目录 %s" % LEGACY_BUILD_DIR.relative_to(REPO_ROOT))
        return
    for name in LEGACY_EXPORT_SCRIPTS:
        p = LEGACY_BUILD_DIR / name
        if p.is_file():
            r.ok("build_phase1/%s" % name)
        else:
            r.fail("缺少 %s" % p.relative_to(REPO_ROOT))


def check_disk_and_write(r: Results) -> None:
    print("\n=== 磁盘与写入权限 ===")
    try:
        usage = shutil.disk_usage(DATA_ROOT if DATA_ROOT.is_dir() else REPO_ROOT)
        free_gib = usage.free / (1024**3)
        total_gib = usage.total / (1024**3)
        print("  [..]   data 所在分区: 剩余 %.1f GiB / 共 %.1f GiB" % (free_gib, total_gib))
        if free_gib < MIN_DISK_GIB_FAIL:
            r.fail(
                "剩余空间 %.1f GiB < %d GiB，Group1/2 写 mask 很可能中途写满"
                % (free_gib, MIN_DISK_GIB_FAIL)
            )
        elif free_gib < MIN_DISK_GIB_WARN:
            r.warn(
                "剩余空间 %.1f GiB，建议 >= %d GiB（COCONut 14 万行耗时长、占空间大）"
                % (free_gib, MIN_DISK_GIB_WARN)
            )
        else:
            r.ok("磁盘剩余空间充足 (>= %d GiB)" % MIN_DISK_GIB_WARN)
    except OSError as e:
        r.warn("无法读取磁盘用量: %s" % e)

    for d in (PHASE1_ROOT, PHASE2_ROOT):
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            r.ok("可写 %s/" % d.relative_to(REPO_ROOT))
        except OSError as e:
            r.fail("无法写入 %s: %s" % (d.relative_to(REPO_ROOT), e))


def check_raw_group1_group2(r: Results) -> None:
    print("\n=== Raw：Group1 (COCO / COCONut / PACO) ===")
    must_file(r, COCO_INST2017, "COCO instances_train2017.json")
    must_file(r, COCONUT_JSON, "COCONut coconut_s.json")
    from config import PACO_TRAIN

    must_file(r, PACO_TRAIN, "PACO paco_lvis_v1_train.json")

    n17 = count_files(COCO_TRAIN2017, "*.jpg")
    if n17 < 0:
        r.fail("COCO train2017 目录不存在: %s" % COCO_TRAIN2017.relative_to(REPO_ROOT))
    elif n17 < 100_000:
        r.fail("train2017 仅 %d 张 jpg（完整集约 118287）" % n17)
    else:
        r.ok("train2017: %d jpg" % n17)

    n_pan = count_files(COCONUT_MASK_DIR, "*.png")
    if n_pan < 0:
        r.fail("COCONut panoptic 目录缺失: coconut_s/")
    elif n_pan < 100_000:
        r.fail("coconut_s/*.png 仅 %d（通常 >= 11 万）" % n_pan)
    else:
        r.ok("coconut_s panoptic png: %d" % n_pan)

    print("\n=== Raw：Group2 (Refer + train2014) ===")
    n14 = count_files(COCO_TRAIN2014, "*.jpg")
    if n14 < 0:
        r.fail("COCO train2014 目录缺失")
    elif n14 < 50_000:
        r.fail("train2014 仅 %d jpg" % n14)
    else:
        r.ok("train2014: %d jpg" % n14)

    from config import REFER_ROOT

    for split, pkl in REF_SPECS:
        rd = REFER_ROOT / split
        if not rd.is_dir():
            r.fail("refer/%s/ 目录缺失" % split)
            continue
        must_file(r, rd / "instances.json", "refer/%s instances.json" % split)
        must_file(r, rd / pkl, "refer/%s %s" % (split, pkl))


def count_hf_jsonl_valid(jsonl: Path, rewrite_fn) -> tuple[int, int]:
    total = valid = 0
    first_miss: tuple[str, str] | None = None
    with jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            item = json.loads(line)
            img, msk = rewrite_fn(item)
            ip = DATA_ROOT / img
            mp = DATA_ROOT / msk
            if ip.is_file() and mp.is_file():
                valid += 1
            elif first_miss is None:
                first_miss = (str(ip), str(mp))
    if first_miss and valid == 0 and total > 0:
        print("  [..]   首条缺失示例 image=%s" % first_miss[0])
        print("  [..]   首条缺失示例 mask =%s" % first_miss[1])
    return total, valid


def check_hf_group3(r: Results, g3_min_valid: int) -> None:
    print("\n=== HF Stage1：Group3 (open_vocabulary) ===")
    if not must_file(r, HF_G3_JSONL, "Stage1 jsonl"):
        return
    must_dir(r, HF_G3_ROOT / "images", "G3 images")
    must_dir(r, HF_G3_ROOT / "masks", "G3 masks")
    ni = count_files(HF_G3_ROOT / "images", "*.png")
    nm = count_files(HF_G3_ROOT / "masks", "*.png")
    print("  [..]   目录内 png: images=%d masks=%d" % (max(ni, 0), max(nm, 0)))

    print("  [..]   扫描 jsonl 行与磁盘文件一致性（约 4.8 万行）…")
    total, valid = count_hf_jsonl_valid(HF_G3_JSONL, rewrite_stage1)
    print("  [..]   jsonl=%d,  image+mask 均存在=%d" % (total, valid))
    if valid < g3_min_valid:
        r.fail(
            "Group3 有效行 %d < %d（mix 默认 --g3-cap=%d，请补 HF mask 或降低 cap）"
            % (valid, g3_min_valid, g3_min_valid)
        )
    else:
        r.ok("Group3 有效行 %d >= %d（可构建）" % (valid, g3_min_valid))
    if valid < 48_529:
        r.warn(
            "Stage1 全量 mask 未齐（有效 %d / 论文 48529），当前流程按 %d cap 训练"
            % (valid, min(g3_min_valid, valid))
        )


def check_hf_group4(r: Results) -> None:
    print("\n=== HF Stage2：Group4 (conversational +/-) ===")
    print("  [..]   本构建流程不使用 stage2_mixed.jsonl，仅检查正负分片。")

    must_dir(r, HF_STAGE2, "converseg_stage2_data")
    for label, jsonl in (
        ("positive", HF_G4_POS_JSONL),
        ("negative", HF_G4_NEG_JSONL),
    ):
        if not must_file(r, jsonl, "G4 %s jsonl" % label):
            continue
        split = jsonl.parent.name
        ni = count_files(jsonl.parent / "images", "*.png")
        nm = count_files(jsonl.parent / "masks", "*.png")
        print(
            "  [..]   %s 目录 png: images=%d masks=%d"
            % (split, max(ni, 0), max(nm, 0))
        )
        print("  [..]   扫描 G4 %s …" % label)
        total, valid = count_hf_jsonl_valid(jsonl, rewrite_stage2)
        print("  [..]   %s: jsonl=%d, valid=%d" % (label, total, valid))
        if valid < G4_MIN_VALID_EACH:
            r.fail("G4 %s 有效行 %d < %d" % (label, valid, G4_MIN_VALID_EACH))
        else:
            r.ok("G4 %s 有效行 %d" % (label, valid))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Preflight before latet_bulid_phase1")
    ap.add_argument(
        "--g3-min-valid",
        type=int,
        default=G3_MIN_VALID_DEFAULT,
        help="要求 Stage1 至少有多少行 image+mask 齐全（默认与 g3-cap 10000 一致）",
    )
    ap.add_argument(
        "--skip-hf-scan",
        action="store_true",
        help="仅检查 jsonl/目录存在，不逐行扫 HF（更快但不防 G4 中途大量缺文件）",
    )
    args = ap.parse_args()

    r = Results()
    print("REPO_ROOT:", REPO_ROOT.resolve())
    print("DATA_ROOT:", DATA_ROOT.resolve())

    if not DATA_ROOT.is_dir():
        r.fail("data/ 目录不存在")
        print("\n=== 结果 ===")
        print("  FAIL: %d" % len(r.failures))
        sys.exit(1)

    check_python(r)
    check_repo_scripts(r)
    check_disk_and_write(r)
    check_raw_group1_group2(r)

    if args.skip_hf_scan:
        print("\n=== HF（跳过逐行扫描）===")
        must_file(r, HF_G3_JSONL, "Stage1 jsonl")
        must_file(r, HF_G4_POS_JSONL, "G4 positive jsonl")
        must_file(r, HF_G4_NEG_JSONL, "G4 negative jsonl")
    else:
        check_hf_group3(r, args.g3_min_valid)
        check_hf_group4(r)

    print("\n=== 结果 ===")
    if r.warnings:
        print("  警告 %d 条（可构建，但请注意）:" % len(r.warnings))
        for w in r.warnings:
            print("    - %s" % w)
    if r.failures:
        print("  失败 %d 条 — 请先修复再执行 run_build_all.sh:" % len(r.failures))
        for f in r.failures:
            print("    - %s" % f)
        sys.exit(1)

    print("  [OK] 前置条件满足，可以执行:")
    print("    bash scripts/latet_bulid_phase1/run_build_all.sh")
    sys.exit(0)


if __name__ == "__main__":
    main()
