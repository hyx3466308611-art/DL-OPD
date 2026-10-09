#!/usr/bin/env python3
"""Validate Phase-1 manifests (counts, schema, paths)."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
from io_utils import read_jsonl

REQUIRED_KEYS = ("image", "mask_merged", "prompt")


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        return -1
    return sum(1 for _ in read_jsonl(path))


def check_shard(
    name: str,
    path: Path,
    expect_min: Optional[int],
    expect_exact: Optional[int],
    sample_paths: int,
    data_root: Path,
    allow_dup_mask: bool = False,
) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    if not path.is_file():
        issues.append("%s: missing file %s" % (name, path))
        return False, issues

    n = 0
    missing_img = 0
    missing_mask = 0
    bad_keys = 0
    dup_mask = 0
    seen_mask: set = set()
    sources: Counter = Counter()

    for row in read_jsonl(path):
        n += 1
        if not all(k in row for k in REQUIRED_KEYS):
            bad_keys += 1
            continue
        src = row.get("source")
        if src:
            sources[src] += 1
        mm = row["mask_merged"]
        if mm in seen_mask:
            dup_mask += 1
        seen_mask.add(mm)

        if sample_paths > 0 and n <= sample_paths:
            ip = data_root / row["image"]
            mp = data_root / row["mask_merged"]
            if not ip.is_file():
                missing_img += 1
            if not mp.is_file():
                missing_mask += 1

    if expect_exact is not None and n != expect_exact:
        issues.append("%s: rows=%d, expected exactly %d" % (name, n, expect_exact))
    elif expect_min is not None and n < expect_min:
        issues.append("%s: rows=%d, expected >= %d" % (name, n, expect_min))

    if bad_keys:
        issues.append("%s: %d rows missing required keys" % (name, bad_keys))
    if dup_mask and not allow_dup_mask:
        issues.append("%s: %d duplicate mask_merged paths" % (name, dup_mask))
    elif dup_mask and allow_dup_mask:
        print("    (info: %d duplicate masks — expected for %s)" % (dup_mask, name))
    if sample_paths > 0 and (missing_img or missing_mask):
        issues.append(
            "%s: path check (first %d rows): missing image=%d mask=%d"
            % (name, min(n, sample_paths), missing_img, missing_mask)
        )

    ok = not issues
    extra = ""
    if sources:
        extra = " sources=%s" % dict(sources)
    print("  %s: %d rows%s%s" % (name, n, extra, " OK" if ok else " ISSUES"))
    for msg in issues:
        print("    -", msg)
    return ok, issues


def check_mixed(path: Path, data_root: Path, sample_paths: int) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    if not path.is_file():
        issues.append("mixed: missing %s (run mix_phase1.py)" % path)
        print("  mixed: MISSING -> run mix_phase1.py")
        return False, issues

    n = 0
    missing = 0
    for row in read_jsonl(path):
        n += 1
        if n <= sample_paths:
            for key in REQUIRED_KEYS:
                if key not in row:
                    issues.append("mixed: row %d missing %s" % (n, key))
                    break
            else:
                if not (data_root / row["image"]).is_file():
                    missing += 1
                if not (data_root / row["mask_merged"]).is_file():
                    missing += 1

    meta_path = path.with_suffix(".meta.json")
    meta_total = None
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta_total = meta.get("total_rows")
        per = meta.get("per_shard", {})
        print("  mixed: %d rows (meta total_rows=%s per_shard=%s)" % (n, meta_total, per))
        if meta_total is not None and meta_total != n:
            issues.append("mixed: meta total_rows %s != actual %d" % (meta_total, n))
    else:
        print("  mixed: %d rows (no .meta.json)" % n)

    if sample_paths > 0 and missing:
        issues.append(
            "mixed: path check (first %d rows): %d missing image/mask files"
            % (min(n, sample_paths), missing)
        )

    ok = not issues
    if ok:
        print("  mixed: OK")
    else:
        for msg in issues:
            print("    -", msg)
    return ok, issues


def main() -> None:
    ap = argparse.ArgumentParser(description="Check Phase-1 build artifacts.")
    ap.add_argument(
        "--sample-paths",
        type=int,
        default=200,
        help="Verify image/mask files exist for first N rows per shard (0=skip).",
    )
    ap.add_argument("--skip-mixed", action="store_true")
    ap.add_argument(
        "--mixed-jsonl",
        type=Path,
        default=PHASE1_ROOT / "phase1_mixed.jsonl",
    )
    args = ap.parse_args()

    print("DATA_ROOT:", DATA_ROOT.resolve())
    print("Manifests under:", MANIFEST_DIR.resolve())
    print()

    shards = [
        ("coco", MANIFEST_DIR / "group1_coco.jsonl", QUOTA_GROUP1_COCO, QUOTA_GROUP1_COCO),
        (
            "coconut",
            MANIFEST_DIR / "group1_coconut.jsonl",
            QUOTA_GROUP1_COCONUT,
            QUOTA_GROUP1_COCONUT,
        ),
        ("paco", MANIFEST_DIR / "group1_paco.jsonl", QUOTA_GROUP1_PACO, QUOTA_GROUP1_PACO),
        ("ref", MANIFEST_DIR / "group2_ref.jsonl", QUOTA_GROUP2, None),
        ("g3", MANIFEST_DIR / "group3_open_vocab.jsonl", QUOTA_GROUP3, QUOTA_GROUP3),
    ]

    all_ok = True
    print("Shards:")
    for name, path, exact_or_min, exact_optional in shards:
        expect_exact = exact_or_min if exact_optional else None
        expect_min = exact_or_min if exact_optional is None else None
        ok, _ = check_shard(
            name,
            path,
            expect_min=expect_min,
            expect_exact=expect_exact,
            sample_paths=args.sample_paths,
            data_root=DATA_ROOT,
            allow_dup_mask=(name == "ref"),
        )
        all_ok = all_ok and ok

    target_mixed = (
        QUOTA_GROUP1_COCO
        + QUOTA_GROUP1_COCONUT
        + QUOTA_GROUP1_PACO
        + QUOTA_GROUP2
        + QUOTA_GROUP3
    )
    print()
    print("Expected phase1_mixed rows (paper caps): ~%d" % target_mixed)

    if not args.skip_mixed:
        print()
        ok, _ = check_mixed(args.mixed_jsonl, DATA_ROOT, args.sample_paths)
        all_ok = all_ok and ok

    print()
    if all_ok:
        print("PASS: Phase-1 checks OK.")
        sys.exit(0)
    print("FAIL: see issues above.")
    sys.exit(1)


if __name__ == "__main__":
    main()
