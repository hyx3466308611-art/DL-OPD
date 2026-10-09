#!/usr/bin/env python3
"""Create missing mask PNGs from existing phase1 manifests (does not rewrite jsonl)."""
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from config import (
    COCO_INST2017,
    COCO_TRAIN2017,
    COCO_TRAIN2014,
    COCONUT_JSON,
    COCONUT_MASK_DIR,
    DATA_ROOT,
    MANIFEST_DIR,
    MASK_ROOT,
    PACO_TRAIN,
    REFER_ROOT,
)
from export_group1_coconut import panoptic_rgb_to_id, _merged_category_mask
from export_group2_ref import REF_SPECS, load_instances
from group1_common import merge_instance_masks
from io_utils import read_jsonl, save_mask_png
from mask_coco import ann_to_mask

RE_COCO = re.compile(r"g1_coco_(\d+)_(\d+)\.png$")
RE_PACO = re.compile(r"g1_paco_(\d+)_(\d+)\.png$")
RE_CNUT = re.compile(r"g1_cnut_(\d+)_(\d+)\.png$")
RE_REF = re.compile(r"g2_(refcoco|refcocop|refcocog)_(\d+)\.png$")


def _unique_mask_paths(jsonl: Path, max_masks: Optional[int] = None) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for row in read_jsonl(jsonl):
        mm = row["mask_merged"].replace("\\", "/")
        if mm not in seen:
            seen.add(mm)
            out.append(mm)
            if max_masks is not None and len(out) >= max_masks:
                break
    return out


def _load_coco_pair_anns():
    from collections import defaultdict

    with COCO_INST2017.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    imgs = {im["id"]: im for im in coco["images"]}
    by_pair: Dict[Tuple[int, int], list] = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd"):
            continue
        by_pair[(int(ann["image_id"]), int(ann["category_id"]))].append(ann)
    return by_pair, imgs


def _load_paco_pair_anns():
    from collections import defaultdict

    with PACO_TRAIN.open("r", encoding="utf-8") as f:
        paco = json.load(f)
    imgs = {im["id"]: im for im in paco["images"]}
    by_pair: Dict[Tuple[int, int], list] = defaultdict(list)
    for ann in paco["annotations"]:
        by_pair[(int(ann["image_id"]), int(ann["category_id"]))].append(ann)
    return by_pair, imgs


def _load_coconut_category_maps():
    from collections import defaultdict

    with COCONUT_JSON.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    img_map = {im["id"]: im for im in coco["images"]}
    groups: Dict[Tuple[int, int], dict] = defaultdict(
        lambda: {"seg_ids": [], "pan_path": None}
    )
    for entry in coco["annotations"]:
        image_id = entry.get("image_id")
        pan_name = entry.get("file_name")
        if image_id is None or not pan_name:
            continue
        if img_map.get(image_id) is None:
            continue
        pan_path = COCONUT_MASK_DIR / pan_name
        for seg in entry.get("segments_info") or []:
            sid = seg.get("id")
            cid = seg.get("category_id")
            if sid is None or cid is None:
                continue
            key = (int(image_id), int(cid))
            groups[key]["pan_path"] = pan_path
            groups[key]["seg_ids"].append(int(sid))
    return groups


def _load_ref_masks() -> Dict[str, dict]:
    """mask filename -> {ann, im}"""
    out: Dict[str, dict] = {}
    for dataset_name, pkl in REF_SPECS:
        refer_dir = REFER_ROOT / dataset_name
        refs = pickle.load(open(refer_dir / pkl, "rb"), encoding="latin1")
        coco = load_instances(refer_dir)
        tag = dataset_name.replace("+", "p")
        for ref in refs:
            if ref.get("split") != "train":
                continue
            ann = coco["annotations"].get(ref["ann_id"])
            im = coco["images"].get(ref["image_id"])
            if ann is None or im is None:
                continue
            mask_name = "g2_%s_%d.png" % (tag, ref["ref_id"])
            out[mask_name] = {"ann": ann, "im": im}
    return out


def materialize_coco(jsonl: Path, max_masks: Optional[int] = None) -> Tuple[int, int]:
    by_pair, imgs = _load_coco_pair_anns()
    need = skip = 0
    for mm in tqdm(_unique_mask_paths(jsonl, max_masks), desc="coco masks"):
        mp = DATA_ROOT / mm
        if mp.is_file():
            skip += 1
            continue
        m = RE_COCO.search(mm.replace("\\", "/"))
        if not m:
            need += 1
            continue
        key = (int(m.group(1)), int(m.group(2)))
        anns = by_pair.get(key)
        if not anns:
            need += 1
            continue
        im = imgs.get(key[0])
        if im is None:
            need += 1
            continue
        mask_bool = merge_instance_masks(anns, im["height"], im["width"])
        mp.parent.mkdir(parents=True, exist_ok=True)
        save_mask_png(mp, mask_bool)
        skip += 1
    return skip, need


def materialize_paco(jsonl: Path, max_masks: Optional[int] = None) -> Tuple[int, int]:
    by_pair, imgs = _load_paco_pair_anns()
    need = skip = 0
    for mm in tqdm(_unique_mask_paths(jsonl, max_masks), desc="paco masks"):
        mp = DATA_ROOT / mm
        if mp.is_file():
            skip += 1
            continue
        m = RE_PACO.search(mm.replace("\\", "/"))
        if not m:
            need += 1
            continue
        key = (int(m.group(1)), int(m.group(2)))
        anns = by_pair.get(key)
        if not anns:
            need += 1
            continue
        im = imgs.get(key[0])
        if im is None:
            need += 1
            continue
        mask_bool = merge_instance_masks(anns, im["height"], im["width"])
        mp.parent.mkdir(parents=True, exist_ok=True)
        save_mask_png(mp, mask_bool)
        skip += 1
    return skip, need


def materialize_coconut(jsonl: Path, max_masks: Optional[int] = None) -> Tuple[int, int]:
    groups = _load_coconut_category_maps()
    pan_cache: Dict[str, np.ndarray] = {}
    need = skip = 0
    for mm in tqdm(_unique_mask_paths(jsonl, max_masks), desc="coconut masks"):
        mp = DATA_ROOT / mm
        if mp.is_file():
            skip += 1
            continue
        m = RE_CNUT.search(Path(mm).name)
        if not m:
            need += 1
            continue
        key = (int(m.group(1)), int(m.group(2)))
        item = groups.get(key)
        if item is None or not item["seg_ids"]:
            need += 1
            continue
        pan_path = item["pan_path"]
        pan_key = str(pan_path)
        if pan_key not in pan_cache:
            with Image.open(pan_path) as im:
                pan_cache[pan_key] = panoptic_rgb_to_id(np.array(im.convert("RGB")))
            if len(pan_cache) > 32:
                pan_cache.pop(next(iter(pan_cache)))
        pan = pan_cache[pan_key]
        mask_bool = _merged_category_mask(pan, item["seg_ids"])
        mp.parent.mkdir(parents=True, exist_ok=True)
        save_mask_png(mp, mask_bool)
        skip += 1
    return skip, need


def materialize_ref(jsonl: Path, max_masks: Optional[int] = None) -> Tuple[int, int]:
    ref_map = _load_ref_masks()
    need = skip = 0
    for mm in tqdm(_unique_mask_paths(jsonl, max_masks), desc="ref masks"):
        mp = DATA_ROOT / mm
        if mp.is_file():
            skip += 1
            continue
        name = Path(mm.replace("\\", "/")).name
        rec = ref_map.get(name)
        if rec is None:
            need += 1
            continue
        ann, im = rec["ann"], rec["im"]
        try:
            mask_bool = ann_to_mask(ann, im["height"], im["width"])
        except Exception:
            need += 1
            continue
        if int(mask_bool.sum()) < 32:
            need += 1
            continue
        mp.parent.mkdir(parents=True, exist_ok=True)
        save_mask_png(mp, mask_bool)
        skip += 1
    return skip, need


SHARDS = {
    "coco": (MANIFEST_DIR / "group1_coco.jsonl", materialize_coco),
    "coconut": (MANIFEST_DIR / "group1_coconut.jsonl", materialize_coconut),
    "paco": (MANIFEST_DIR / "group1_paco.jsonl", materialize_paco),
    "ref": (MANIFEST_DIR / "group2_ref.jsonl", materialize_ref),
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fill missing mask PNGs listed in existing manifests."
    )
    ap.add_argument(
        "--shard",
        choices=list(SHARDS.keys()) + ["all"],
        default="all",
        help="Which manifest to materialize from.",
    )
    ap.add_argument(
        "--max-masks",
        type=int,
        default=None,
        help="Smoke test: process at most N unique mask paths per shard.",
    )
    args = ap.parse_args()
    names = list(SHARDS.keys()) if args.shard == "all" else [args.shard]
    for name in names:
        jsonl, fn = SHARDS[name]
        if not jsonl.is_file():
            print("%s: skip (no %s)" % (name, jsonl))
            continue
        ok, fail = fn(jsonl, max_masks=args.max_masks)
        print("%s: wrote/ok=%d failed_or_skip=%d" % (name, ok, fail))


if __name__ == "__main__":
    main()
