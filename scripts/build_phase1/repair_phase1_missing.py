#!/usr/bin/env python3
"""Scan Phase-1 manifests for missing files; download HF Group3 masks; re-import & mix."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from config import DATA_ROOT, HF_G3_JSONL, MANIFEST_DIR, PHASE1_ROOT
from io_utils import read_jsonl, write_jsonl

REQUIRED = ("image", "mask_merged", "prompt")
HF_REPO = "aadarsh99/ConverSeg-Training-Data"
STAGE1_PREFIX = "converseg_stage1_data/open_vocabulary_regions_data/"


def _resolve_hf_paths(item: dict) -> Tuple[str, str]:
    stage1_base = (
        DATA_ROOT / "ConverSeg-Training-Data" / "converseg_stage1_data"
    ).relative_to(DATA_ROOT).as_posix()
    img = item["image"].replace("\\", "/")
    msk = item["mask_merged"].replace("\\", "/")
    if not img.startswith("ConverSeg-Training-Data/"):
        img = "%s/%s" % (stage1_base, img.lstrip("/"))
        msk = "%s/%s" % (stage1_base, msk.lstrip("/"))
    return img, msk


def scan_shard(path: Path) -> Dict[str, int]:
    miss_img = miss_mask = 0
    n = 0
    for row in read_jsonl(path):
        n += 1
        ip = DATA_ROOT / row["image"]
        mp = DATA_ROOT / row["mask_merged"]
        if not ip.is_file():
            miss_img += 1
        if not mp.is_file():
            miss_mask += 1
    return {"rows": n, "missing_image": miss_img, "missing_mask": miss_mask}


def collect_g3_missing_masks() -> List[str]:
    """Return repo-relative paths under ConverSeg-Training-Data for missing mask PNGs."""
    missing: List[str] = []
    if not HF_G3_JSONL.is_file():
        return missing
    with HF_G3_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            _, msk_rel = _resolve_hf_paths(item)
            if not (DATA_ROOT / msk_rel).is_file():
                # hf_hub local_dir layout: local-dir / path_in_repo
                raw = item["mask_merged"].replace("\\", "/").lstrip("/")
                if raw.startswith("open_vocabulary_regions_data/"):
                    repo_path = "converseg_stage1_data/" + raw
                else:
                    repo_path = STAGE1_PREFIX + raw.split("open_vocabulary_regions_data/", 1)[-1]
                missing.append(repo_path)
    return missing


def download_hf_masks(repo_paths: List[str], local_dir: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit(
            "Install huggingface_hub: pip install -U huggingface_hub"
        ) from e

    local_dir.mkdir(parents=True, exist_ok=True)
    total = len(repo_paths)
    for i, repo_path in enumerate(repo_paths, 1):
        hf_hub_download(
            repo_id=HF_REPO,
            filename=repo_path,
            repo_type="dataset",
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        if i % 500 == 0 or i == total:
            print("  downloaded %d / %d" % (i, total))


def import_g3_filtered(out: Path) -> int:
    rows = []
    skipped = 0
    with HF_G3_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            img, msk = _resolve_hf_paths(item)
            if not (DATA_ROOT / img).is_file() or not (DATA_ROOT / msk).is_file():
                skipped += 1
                continue
            rows.append(
                {
                    "image": img,
                    "mask_merged": msk,
                    "prompt": item["prompt"],
                    "source": "hf_open_vocab",
                }
            )
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    n = write_jsonl(out, rows)
    print("group3: wrote %d rows (skipped %d missing)" % (n, skipped))
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--download-g3-masks",
        action="store_true",
        help="Download missing Stage1 open_vocab masks from HuggingFace.",
    )
    ap.add_argument(
        "--skip-download",
        action="store_true",
        help="Only scan + re-import/mix (no HF download).",
    )
    ap.add_argument("--remix", action="store_true", help="Run mix_phase1 after import.")
    ap.add_argument("--ref-cap", type=int, default=321_327)
    ap.add_argument("--coconut-cap", type=int, default=140_000)
    args = ap.parse_args()

    shards = [
        ("coco", MANIFEST_DIR / "group1_coco.jsonl"),
        ("coconut", MANIFEST_DIR / "group1_coconut.jsonl"),
        ("paco", MANIFEST_DIR / "group1_paco.jsonl"),
        ("ref", MANIFEST_DIR / "group2_ref.jsonl"),
        ("g3", MANIFEST_DIR / "group3_open_vocab.jsonl"),
    ]
    print("Scan manifests (full rows):")
    any_miss = False
    for name, path in shards:
        if not path.is_file():
            print("  %s: MISSING %s" % (name, path))
            any_miss = True
            continue
        st = scan_shard(path)
        print(
            "  %s: rows=%d missing_image=%d missing_mask=%d"
            % (name, st["rows"], st["missing_image"], st["missing_mask"])
        )
        if st["missing_image"] or st["missing_mask"]:
            any_miss = True

    missing_masks = collect_g3_missing_masks()
    print("HF G3 masks missing on disk: %d" % len(missing_masks))

    if args.download_g3_masks and not args.skip_download and missing_masks:
        print("Downloading from %s ..." % HF_REPO)
        download_hf_masks(missing_masks, DATA_ROOT / "ConverSeg-Training-Data")
        missing_masks = collect_g3_missing_masks()
        print("After download, HF G3 masks still missing: %d" % len(missing_masks))

    g3_out = MANIFEST_DIR / "group3_open_vocab.jsonl"
    import_g3_filtered(g3_out)

    if args.remix:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "mix_phase1.py"),
            "--ref-cap",
            str(args.ref_cap),
            "--coconut-cap",
            str(args.coconut_cap),
        ]
        print("Running:", " ".join(cmd))
        subprocess.check_call(cmd)

    st_g3 = scan_shard(g3_out)
    if st_g3["missing_image"] or st_g3["missing_mask"]:
        print("FAIL: group3 still has missing paths.")
        sys.exit(1)
    if any_miss and missing_masks:
        print("WARN: other shards had issues or HF masks incomplete.")
        sys.exit(1 if missing_masks else 0)
    print("Done.")


if __name__ == "__main__":
    main()
