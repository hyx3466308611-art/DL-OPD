#!/usr/bin/env python3
"""
Group 1 PACO (paper Sec. 6.2.1): part-level, merge same part category on one image;
prompt: Segment all the [part] in the image.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from config import COCO_TRAIN2017, DATA_ROOT, MANIFEST_DIR, MASK_ROOT, PACO_TRAIN
from group1_common import literal_prompt, merge_instance_masks
from io_utils import rel_to_data_root, save_mask_png, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=100_000)
    ap.add_argument("--min-area", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-jsonl", type=Path, default=MANIFEST_DIR / "group1_paco.jsonl")
    args = ap.parse_args()

    with PACO_TRAIN.open("r", encoding="utf-8") as f:
        paco = json.load(f)
    part_map: Dict[int, str] = {c["id"]: c["name"] for c in paco["categories"]}
    img_map = {im["id"]: im for im in paco["images"]}

    by_pair: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
    for ann in tqdm(paco["annotations"], desc="scan paco"):
        im = img_map.get(ann["image_id"])
        if im is None:
            continue
        fn = im["file_name"]
        if fn.startswith("train2017/"):
            fn = fn.split("/", 1)[1]
        img_path = COCO_TRAIN2017 / fn
        if not img_path.is_file():
            continue
        by_pair[(int(ann["image_id"]), int(ann["category_id"]))].append(ann)

    candidates: List[dict] = []
    for (image_id, category_id), anns in tqdm(by_pair.items(), desc="filter paco pairs"):
        im = img_map[image_id]
        h, w = im["height"], im["width"]
        area = int(merge_instance_masks(anns, h, w).sum())
        if area < args.min_area:
            continue
        fn = im["file_name"]
        if fn.startswith("train2017/"):
            fn = fn.split("/", 1)[1]
        img_path = COCO_TRAIN2017 / fn
        part = part_map.get(category_id, "part")
        candidates.append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "img_path": img_path,
                "im": im,
                "anns": anns,
                "prompt": literal_prompt(part.replace("_", " ")),
            }
        )

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    selected = candidates[: args.max_rows]

    mask_dir = MASK_ROOT / "group1_paco"
    mask_dir.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    for item in tqdm(selected, desc="export paco"):
        im = item["im"]
        h, w = im["height"], im["width"]
        mask_bool = merge_instance_masks(item["anns"], h, w)
        mask_path = mask_dir / (
            "g1_paco_%d_%d.png" % (item["image_id"], item["category_id"])
        )
        if not mask_path.is_file():
            save_mask_png(mask_path, mask_bool)
        rows.append(
            {
                "image": rel_to_data_root(item["img_path"], DATA_ROOT),
                "mask_merged": rel_to_data_root(mask_path, DATA_ROOT),
                "prompt": item["prompt"],
                "source": "paco",
            }
        )

    n = write_jsonl(args.out_jsonl, rows)
    print("Wrote %d rows -> %s" % (n, args.out_jsonl))


if __name__ == "__main__":
    main()
