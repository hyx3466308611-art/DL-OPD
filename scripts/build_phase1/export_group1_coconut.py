#!/usr/bin/env python3
"""
Group 1 COCONut-S (paper Sec. 6.2.1): refined panoptic masks, merged by category per image;
prompt: Segment all the [category] in the image.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from config import (
    COCO_TRAIN2017,
    COCONUT_JSON,
    COCONUT_MASK_DIR,
    DATA_ROOT,
    MANIFEST_DIR,
    MASK_ROOT,
)
from group1_common import literal_prompt
from io_utils import read_jsonl, rel_to_data_root, save_mask_png, write_jsonl

HARD_MIN_AREA = 512


def panoptic_rgb_to_id(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        return rgb.astype(np.int32)
    r = rgb[:, :, 0].astype(np.int32)
    g = rgb[:, :, 1].astype(np.int32)
    b = rgb[:, :, 2].astype(np.int32)
    return r + 256 * g + 256 * 256 * b


def _load_pan(pan_path: Path, pan_cache: Dict[str, np.ndarray]) -> np.ndarray:
    pan_key = str(pan_path)
    if pan_key not in pan_cache:
        with Image.open(pan_path) as im:
            pan_cache[pan_key] = panoptic_rgb_to_id(np.array(im.convert("RGB")))
        if len(pan_cache) > 32:
            pan_cache.pop(next(iter(pan_cache)))
    return pan_cache[pan_key]


def _merged_category_mask(
    pan: np.ndarray, seg_ids: List[int]
) -> np.ndarray:
    out = np.zeros(pan.shape, dtype=bool)
    for sid in seg_ids:
        out |= pan == sid
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-rows", type=int, default=140_000)
    ap.add_argument("--min-area", type=int, default=768)
    ap.add_argument("--min-area-small", type=int, default=HARD_MIN_AREA)
    ap.add_argument("--small-quota", type=int, default=8_000)
    ap.add_argument("--small-idle-stop", type=int, default=200_000)
    ap.add_argument("--save-every", type=int, default=2_000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-jsonl", type=Path, default=MANIFEST_DIR / "group1_coconut.jsonl")
    args = ap.parse_args()

    if args.min_area_small < HARD_MIN_AREA:
        raise SystemExit("min-area-small must be >= %d" % HARD_MIN_AREA)
    if args.min_area <= args.min_area_small:
        raise SystemExit("min-area must be > min-area-small")

    with COCONUT_JSON.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    cat_map: Dict[int, str] = {c["id"]: c["name"] for c in coco["categories"]}
    img_map = {im["id"]: im for im in coco["images"]}

    # (image_id, category_id) -> {img_path, pan_path, seg_ids}
    groups: Dict[Tuple[int, int], dict] = defaultdict(
        lambda: {"seg_ids": [], "img_path": None, "pan_path": None}
    )
    for entry in tqdm(coco["annotations"], desc="scan coconut"):
        image_id = entry.get("image_id")
        pan_name = entry.get("file_name")
        if image_id is None or not pan_name:
            continue
        im = img_map.get(image_id)
        if im is None:
            continue
        img_path = COCO_TRAIN2017 / im["file_name"]
        pan_path = COCONUT_MASK_DIR / pan_name
        if not img_path.is_file() or not pan_path.is_file():
            continue
        for seg in entry.get("segments_info") or []:
            sid = seg.get("id")
            cid = seg.get("category_id")
            if sid is None or cid is None:
                continue
            key = (int(image_id), int(cid))
            g = groups[key]
            g["img_path"] = img_path
            g["pan_path"] = pan_path
            g["seg_ids"].append(int(sid))

    pairs: List[dict] = []
    pan_cache: Dict[str, np.ndarray] = {}
    for (image_id, category_id), g in tqdm(groups.items(), desc="area coconut"):
        if not g["seg_ids"]:
            continue
        pan = _load_pan(g["pan_path"], pan_cache)
        mask_bool = _merged_category_mask(pan, g["seg_ids"])
        area = int(mask_bool.sum())
        if area < args.min_area_small:
            continue
        pairs.append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "img_path": g["img_path"],
                "pan_path": g["pan_path"],
                "seg_ids": g["seg_ids"],
                "area": area,
            }
        )

    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    main_goal = args.target_rows - args.small_quota
    rows: List[dict] = []
    seen: Set[Tuple[int, int]] = set()
    main_count = 0
    small_count = 0

    mask_dir = MASK_ROOT / "group1_coconut"
    mask_dir.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    if args.resume and args.out_jsonl.is_file():
        rows = list(read_jsonl(args.out_jsonl))
        for r in rows:
            mm = r["mask_merged"].replace("\\", "/")
            # g1_cnut_{image}_{cat}.png
            name = Path(mm).name
            parts = name.replace(".png", "").split("_")
            if len(parts) >= 4 and parts[0] == "g1" and parts[1] == "cnut":
                seen.add((int(parts[2]), int(parts[3])))
        main_count = min(len(rows), main_goal)
        small_count = max(0, len(rows) - main_goal)
        print("Resume: %d rows from %s" % (len(rows), args.out_jsonl))

    pan_cache = {}
    idle_small = 0
    last_save = len(rows)
    main_saturated = main_count >= main_goal

    pbar = tqdm(pairs, desc="export coconut")
    for item in pbar:
        if len(rows) >= args.target_rows:
            break
        key = (item["image_id"], item["category_id"])
        if key in seen:
            continue

        area = item["area"]
        if area < args.min_area_small:
            continue
        use_small = False
        if area >= args.min_area:
            if main_count >= main_goal:
                continue
        else:
            if main_count < main_goal:
                continue
            if small_count >= args.small_quota:
                continue
            use_small = True

        pan = _load_pan(item["pan_path"], pan_cache)
        mask_bool = _merged_category_mask(pan, item["seg_ids"])
        mask_path = mask_dir / (
            "g1_cnut_%d_%d.png" % (item["image_id"], item["category_id"])
        )
        if not mask_path.is_file():
            save_mask_png(mask_path, mask_bool)

        cat = cat_map.get(item["category_id"], "region")
        rows.append(
            {
                "image": rel_to_data_root(item["img_path"], DATA_ROOT),
                "mask_merged": rel_to_data_root(mask_path, DATA_ROOT),
                "prompt": literal_prompt(cat),
                "source": "coconut",
            }
        )
        seen.add(key)
        if use_small:
            small_count += 1
            idle_small = 0
        else:
            main_count += 1
            if main_count >= main_goal:
                main_saturated = True

        if len(rows) - last_save >= args.save_every:
            write_jsonl(args.out_jsonl, rows)
            last_save = len(rows)

        pbar.set_postfix(rows=len(rows), main=main_count, small=small_count, refresh=False)

        if main_saturated and small_count < args.small_quota:
            idle_small += 1
            if idle_small >= args.small_idle_stop:
                print("[info] Early stop: small tier scarce.")
                break

    n = write_jsonl(args.out_jsonl, rows)
    print(
        "Wrote %d rows (main~%d small=%d) -> %s"
        % (n, main_count, small_count, args.out_jsonl)
    )


if __name__ == "__main__":
    main()
