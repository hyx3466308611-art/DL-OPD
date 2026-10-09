#!/usr/bin/env python3
"""
Export RefCOCO / RefCOCO+ / RefCOCOg *eval* splits to jsonl for eval.py.

Produces one row per referring sentence (same unit as standard RIS tables):
  data/eval_manifests/refcoco/
    refcoco_val.jsonl
    refcoco_testA.jsonl
    refcoco_testB.jsonl
    refcocop_val.jsonl ...
    refcocog_umd_val.jsonl
    refcocog_umd_test.jsonl
    refcocog_google_val.jsonl   # Table 2 val(G)

Masks are written under data/eval_manifests/refcoco/masks/.
Images stay in data/raw/coco/train2014/.

Usage:
  python scripts/export_refcoco_eval_manifests.py
  python scripts/export_refcoco_eval_manifests.py --datasets refcoco --splits val
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "build_phase1"))

from config import COCO_TRAIN2014, DATA_ROOT, REFER_ROOT  # noqa: E402
from io_utils import rel_to_data_root, save_mask_png  # noqa: E402
from mask_coco import ann_to_mask  # noqa: E402

OUT_ROOT = DATA_ROOT / "eval_manifests" / "refcoco"
MASK_DIR = OUT_ROOT / "masks"

# (dataset_dir_name, refs_pickle, split_name_in_pickle, out_stem)
EXPORT_SPECS = [
    ("refcoco", "refs(unc).p", "val", "refcoco_val"),
    ("refcoco", "refs(unc).p", "testA", "refcoco_testA"),
    ("refcoco", "refs(unc).p", "testB", "refcoco_testB"),
    ("refcoco+", "refs(unc).p", "val", "refcocop_val"),
    ("refcoco+", "refs(unc).p", "testA", "refcocop_testA"),
    ("refcoco+", "refs(unc).p", "testB", "refcocop_testB"),
    ("refcocog", "refs(umd).p", "val", "refcocog_umd_val"),
    ("refcocog", "refs(umd).p", "test", "refcocog_umd_test"),
    ("refcocog", "refs(google).p", "val", "refcocog_google_val"),
]


def load_instances(refer_dir: Path) -> dict:
    with (refer_dir / "instances.json").open("r", encoding="utf-8") as f:
        coco = json.load(f)
    return {
        "images": {im["id"]: im for im in coco["images"]},
        "annotations": {an["id"]: an for an in coco["annotations"]},
    }


def export_one(dataset_name: str, refs_pickle: str, split: str, out_stem: str) -> int:
    refer_dir = REFER_ROOT / dataset_name
    refs_path = refer_dir / refs_pickle
    if not refs_path.is_file():
        raise FileNotFoundError(refs_path)
    refs = pickle.load(open(refs_path, "rb"), encoding="latin1")
    coco = load_instances(refer_dir)
    rows = []
    skipped = 0

    for ref in tqdm(refs, desc=f"{out_stem}"):
        if ref.get("split") != split:
            continue
        ann = coco["annotations"].get(ref["ann_id"])
        im = coco["images"].get(ref["image_id"])
        if ann is None or im is None:
            skipped += 1
            continue
        img_path = COCO_TRAIN2014 / im["file_name"]
        if not img_path.is_file():
            skipped += 1
            continue
        h, w = int(im["height"]), int(im["width"])
        try:
            mask_bool = ann_to_mask(ann, h, w)
        except Exception:
            skipped += 1
            continue

        mask_name = f"{out_stem}_ref{ref['ref_id']}.png"
        mask_path = MASK_DIR / mask_name
        if not mask_path.is_file():
            save_mask_png(mask_path, mask_bool)

        # Absolute paths: eval.py resolves relatives against the jsonl directory.
        img_abs = str(img_path.resolve())
        mask_abs = str(mask_path.resolve())

        for sent_i, sent in enumerate(ref.get("sentences", [])):
            prompt = str(sent.get("sent", "")).strip()
            if not prompt:
                continue
            sid = sent.get("sent_id", sent_i)
            rows.append(
                {
                    "id": f"{out_stem}_ref{ref['ref_id']}_s{sid}",
                    "image": img_abs,
                    "mask": mask_abs,
                    "prompt": prompt,
                    "dataset": dataset_name,
                    "split": split,
                    "ref_id": int(ref["ref_id"]),
                    "ann_id": int(ref["ann_id"]),
                    "image_rel_data": rel_to_data_root(img_path, DATA_ROOT),
                }
            )

    out_jsonl = OUT_ROOT / f"{out_stem}.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[ok] {out_stem}: {len(rows)} sentences -> {out_jsonl} (skipped refs/imgs={skipped})")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="all", help="all | refcoco,refcoco+,refcocog")
    ap.add_argument("--splits", default="all", help="all | val,testA,testB,test")
    args = ap.parse_args()

    if not COCO_TRAIN2014.is_dir():
        raise FileNotFoundError(f"Missing COCO train2014 images: {COCO_TRAIN2014}")
    if not REFER_ROOT.is_dir():
        raise FileNotFoundError(f"Missing refer root: {REFER_ROOT}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    want_ds = None if args.datasets.strip() == "all" else {x.strip() for x in args.datasets.split(",")}
    want_sp = None if args.splits.strip() == "all" else {x.strip() for x in args.splits.split(",")}

    total = 0
    for ds, pkl, split, stem in EXPORT_SPECS:
        if want_ds is not None and ds not in want_ds:
            continue
        if want_sp is not None and split not in want_sp:
            continue
        total += export_one(ds, pkl, split, stem)
    print(f"Done. total sentences={total}. manifests under {OUT_ROOT}")


if __name__ == "__main__":
    main()
