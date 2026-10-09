#!/usr/bin/env python3
"""Check data/ tree layout and readiness for Phase-1 build or train (no mask export)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple

from config import (
    COCO_INST2017,
    COCO_TRAIN2014,
    COCO_TRAIN2017,
    COCONUT_JSON,
    COCONUT_MASK_DIR,
    DATA_ROOT,
    HF_G3_JSONL,
    HF_G3_ROOT,
    MANIFEST_DIR,
    PACO_TRAIN,
    PHASE1_ROOT,
    QUOTA_GROUP1_COCO,
    QUOTA_GROUP1_COCONUT,
    QUOTA_GROUP1_PACO,
    QUOTA_GROUP2,
    QUOTA_GROUP3,
    REFER_ROOT,
)
from io_utils import read_jsonl

REPO_ROOT = Path(__file__).resolve().parents[2]
HF_STAGE2 = DATA_ROOT / "ConverSeg-Training-Data" / "converseg_stage2_data"
HF_STAGE2_MIXED = HF_STAGE2 / "stage2_mixed.jsonl"


def _ok(msg: str) -> None:
    print("  [OK] %s" % msg)


def _warn(msg: str) -> None:
    print("  [WARN] %s" % msg)


def _fail(msg: str) -> None:
    print("  [FAIL] %s" % msg)


def _count_glob(d: Path, pattern: str, min_expected: int | None = None) -> int:
    if not d.is_dir():
        _fail("missing dir: %s" % d)
        return 0
    n = sum(1 for _ in d.glob(pattern))
    label = "%s (%s): %d files" % (d.relative_to(DATA_ROOT), pattern, n)
    if min_expected is not None and n < min_expected:
        _fail("%s (need >= %d)" % (label, min_expected))
    elif min_expected is not None:
        _ok(label)
    else:
        print("  [..] %s" % label)
    return n


def _must_file(p: Path, label: str) -> bool:
    if p.is_file():
        _ok("%s -> %s" % (label, p.relative_to(DATA_ROOT)))
        return True
    _fail("%s missing: %s" % (label, p.relative_to(DATA_ROOT)))
    return False


def _jsonl_lines(p: Path, expect: int | None, label: str) -> int:
    if not p.is_file():
        _fail("%s missing: %s" % (label, p.relative_to(DATA_ROOT)))
        return 0
    n = sum(1 for _ in read_jsonl(p))
    if expect is not None and n != expect:
        _fail("%s: %d lines (expected %d)" % (label, n, expect))
    elif expect is not None:
        _ok("%s: %d lines" % (label, n))
    else:
        print("  [..] %s: %d lines" % (label, n))
    return n


def check_raw() -> bool:
    print("\n=== Raw (Group1/2 build inputs) ===")
    ok = True
    ok &= _must_file(COCO_INST2017, "COCO instances_train2017")
    ok &= _must_file(COCONUT_JSON, "COCONut coconut_s.json")
    ok &= _must_file(PACO_TRAIN, "PACO paco_lvis_v1_train")
    n17 = _count_glob(COCO_TRAIN2017, "*.jpg", min_expected=100_000)
    n14 = _count_glob(COCO_TRAIN2014, "*.jpg", min_expected=50_000)
    _count_glob(COCONUT_MASK_DIR, "*.png", min_expected=100_000)
    ok = ok and n17 >= 100_000 and n14 >= 50_000
    for split in ("refcoco", "refcoco+", "refcocog"):
        rd = REFER_ROOT / split
        if rd.is_dir():
            _ok("refer/%s/" % split)
        else:
            _fail("refer/%s/" % split)
            ok = False
    return ok


def check_hf_g3() -> bool:
    print("\n=== HF Stage1 Group3 (open_vocab) ===")
    ok = True
    ok &= _must_file(HF_G3_JSONL, "HF Stage1 jsonl")
    img_d = HF_G3_ROOT / "images"
    msk_d = HF_G3_ROOT / "masks"
    ni = _count_glob(img_d, "*.png", min_expected=8_000)
    nm = _count_glob(msk_d, "*.png", min_expected=QUOTA_GROUP3)
    n = _jsonl_lines(HF_G3_JSONL, QUOTA_GROUP3, "HF Stage1 manifest")
    ok = ok and ni >= 8_000 and nm >= QUOTA_GROUP3 and n == QUOTA_GROUP3
    return ok


def check_hf_stage2_optional() -> None:
    print("\n=== HF Stage2 (Phase2 only, optional) ===")
    if HF_STAGE2_MIXED.is_file():
        _jsonl_lines(HF_STAGE2_MIXED, None, "stage2_mixed.jsonl")
    else:
        _warn("stage2_mixed.jsonl not found (Phase2 not ready)")


def check_phase1_artifacts() -> Tuple[bool, bool]:
    """Returns (layout_ok, ready_to_train)."""
    print("\n=== Phase1 build outputs (data/phase1/) ===")
    layout = True
    train_ready = True
    expected_dirs = [
        MANIFEST_DIR,
        PHASE1_ROOT / "masks" / "group1_coco",
        PHASE1_ROOT / "masks" / "group1_coconut",
        PHASE1_ROOT / "masks" / "group1_paco",
        PHASE1_ROOT / "masks" / "group2_ref",
    ]
    for d in expected_dirs:
        if d.is_dir():
            _ok("dir %s/" % d.relative_to(DATA_ROOT))
        else:
            _warn("dir missing (run export): %s/" % d.relative_to(DATA_ROOT))
            layout = False
            train_ready = False

    shards = [
        ("group1_coco.jsonl", QUOTA_GROUP1_COCO, True),
        ("group1_coconut.jsonl", QUOTA_GROUP1_COCONUT, True),
        ("group1_paco.jsonl", QUOTA_GROUP1_PACO, True),
        ("group2_ref.jsonl", QUOTA_GROUP2, False),
        ("group3_open_vocab.jsonl", QUOTA_GROUP3, True),
    ]
    for fn, exp, exact in shards:
        p = MANIFEST_DIR / fn
        if not p.is_file():
            _warn("manifest missing: %s" % fn)
            train_ready = False
            continue
        n = sum(1 for _ in read_jsonl(p))
        if exact and n != exp:
            _warn("%s: %d rows (target %d)" % (fn, n, exp))
            train_ready = False
        elif not exact and n < exp:
            _warn("%s: %d rows (target >= %d)" % (fn, n, exp))
            train_ready = False
        else:
            _ok("%s: %d rows" % (fn, n))

    mixed = PHASE1_ROOT / "phase1_mixed.jsonl"
    if mixed.is_file():
        nm = sum(1 for _ in read_jsonl(mixed))
        target = (
            QUOTA_GROUP1_COCO
            + QUOTA_GROUP1_COCONUT
            + QUOTA_GROUP1_PACO
            + 321_327
            + QUOTA_GROUP3
        )
        if abs(nm - target) > 500:
            _warn("phase1_mixed.jsonl: %d rows (~%d expected)" % (nm, target))
            train_ready = False
        else:
            _ok("phase1_mixed.jsonl: %d rows (~%d)" % (nm, target))
    else:
        _warn("phase1_mixed.jsonl missing (run mix_phase1.py)")
        train_ready = False

    return layout, train_ready


def print_tree_hint() -> None:
    print("\n=== Expected layout (relative to repo/data/) ===")
    print(
        """
  raw/coco/train2017/          # COCO images
  raw/coco/train2014/          # RefCOCO images
  raw/coco/annotations2017/instances_train2017.json
  raw/coconut/coconut_s.json + coconut_s/*.png
  raw/paco/annotations/paco_lvis_v1_train.json
  raw/refer/{refcoco,refcoco+,refcocog}/

  ConverSeg-Training-Data/converseg_stage1_data/open_vocabulary_regions_data/
    open_vocabulary_regions_data.jsonl  images/  masks/

  phase1/manifests/*.jsonl
  phase1/masks/group1_{coco,coconut,paco}/  group2_ref/
  phase1/phase1_mixed.jsonl
"""
    )


def main() -> None:
    print("REPO_ROOT:", REPO_ROOT.resolve())
    print("DATA_ROOT:", DATA_ROOT.resolve())

    raw_ok = check_raw()
    g3_ok = check_hf_g3()
    check_hf_stage2_optional()
    _, train_ready = check_phase1_artifacts()
    print_tree_hint()

    print("\n=== Summary ===")
    if raw_ok and g3_ok:
        _ok("Raw + HF G3: ready to RUN export/mix on this machine")
    else:
        _fail("Fix raw or HF G3 before running export scripts")

    if train_ready:
        _ok("Phase1: manifests + mixed look complete -> run check_phase1.py for path audit")
    else:
        _warn("Phase1: run build (export + mix) or finish upload, then check_phase1.py")

    sys.exit(0 if raw_ok and g3_ok else 1)


if __name__ == "__main__":
    main()
