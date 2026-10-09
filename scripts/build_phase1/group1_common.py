"""Shared helpers for paper-style Group 1 (literal) exports."""
from __future__ import annotations

import numpy as np

from mask_coco import ann_to_mask


def literal_prompt(category_name: str) -> str:
    """Paper Sec. 6.2.1: Segment all the [category] in the image."""
    name = category_name.strip().lower().replace("_", " ")
    return "Segment all the %s in the image" % name


def merge_instance_masks(annotations: list, height: int, width: int) -> np.ndarray:
    merged = np.zeros((height, width), dtype=bool)
    for ann in annotations:
        try:
            merged |= ann_to_mask(ann, height, width)
        except Exception:
            continue
    return merged


def merged_area(annotations: list, height: int, width: int) -> int:
    return int(merge_instance_masks(annotations, height, width).sum())
