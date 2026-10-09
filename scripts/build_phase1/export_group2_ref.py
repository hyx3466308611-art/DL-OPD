#!/usr/bin/env python3
"""Group 2: RefCOCO / RefCOCO+ / RefCOCOg train sentences → jsonl + masks."""
import argparse
import json
import pickle
import random
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm

from config import COCO_TRAIN2014, DATA_ROOT, MANIFEST_DIR, MASK_ROOT, REFER_ROOT
from io_utils import rel_to_data_root, save_mask_png, write_jsonl
from mask_coco import ann_to_mask

REF_SPECS: List[Tuple[str, str]] = [
    ("refcoco", "refs(unc).p"),
    ("refcoco+", "refs(unc).p"),
    ("refcocog", "refs(umd).p"),
]


def load_instances(refer_dir: Path) -> dict:
    with (refer_dir / "instances.json").open("r", encoding="utf-8") as f:
        coco = json.load(f)
    imgs = {im["id"]: im for im in coco["images"]}
    anns = {an["id"]: an for an in coco["annotations"]}
    return {"images": imgs, "annotations": anns}


def collect_rows(
    dataset_name: str,
    refs_pickle: str,
    mask_dir: Path,
) -> List[dict]:
    refer_dir = REFER_ROOT / dataset_name
    refs = pickle.load(open(refer_dir / refs_pickle, "rb"), encoding="latin1")
    coco = load_instances(refer_dir)
    rows: List[dict] = []

    for ref in tqdm(refs, desc=dataset_name):
        if ref.get("split") != "train":
            continue
        ann = coco["annotations"].get(ref["ann_id"])
        if ann is None:
            continue
        im = coco["images"].get(ref["image_id"])
        if im is None:
            continue
        # ref["file_name"] is e.g. COCO_train2014_000581857_16.jpg; image on disk is im["file_name"]
        img_path = COCO_TRAIN2014 / im["file_name"]
        if not img_path.is_file():
            continue
        h, w = im["height"], im["width"]
        try:
            mask_bool = ann_to_mask(ann, h, w)
        except Exception:
            continue
        if int(mask_bool.sum()) < 32:
            continue

        mask_name = f"g2_{dataset_name.replace('+', 'p')}_{ref['ref_id']}.png"
        mask_path = mask_dir / mask_name
        if not mask_path.is_file():
            save_mask_png(mask_path, mask_bool)

        img_rel = rel_to_data_root(img_path, DATA_ROOT)
        mask_rel = rel_to_data_root(mask_path, DATA_ROOT)

        for sent in ref.get("sentences", []):
            prompt = str(sent.get("sent", "")).strip()
            if not prompt:
                continue
            rows.append(
                {
                    "image": img_rel,
                    "mask_merged": mask_rel,
                    "prompt": prompt,
                    "source": "ref:%s" % dataset_name,
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=None, help="Cap total rows (default all ~321K)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-jsonl", type=Path, default=MANIFEST_DIR / "group2_ref.jsonl")
    args = ap.parse_args()

    mask_dir = MASK_ROOT / "group2_ref"
    mask_dir.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: List[dict] = []
    for name, pkl in REF_SPECS:
        part = collect_rows(name, pkl, mask_dir)
        print("%s: %d rows" % (name, len(part)))
        all_rows.extend(part)

    if args.max_rows is not None and len(all_rows) > args.max_rows:
        random.Random(args.seed).shuffle(all_rows)
        all_rows = all_rows[: args.max_rows]

    n = write_jsonl(args.out_jsonl, all_rows)
    print("Wrote %d rows -> %s" % (n, args.out_jsonl))


if __name__ == "__main__":
    main()
