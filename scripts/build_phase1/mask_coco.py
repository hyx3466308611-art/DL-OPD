"""Decode COCO polygon / RLE to boolean H×W mask."""
from typing import List, Union

import numpy as np

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None  # type: ignore[misc, assignment]


def ann_to_mask(ann: dict, height: int, width: int) -> np.ndarray:
    seg = ann.get("segmentation")
    if not seg:
        raise ValueError("empty segmentation")
    if isinstance(seg, dict):
        if mask_utils is None:
            raise ImportError("pycocotools required for RLE masks: pip install pycocotools")
        m = mask_utils.decode(seg)
        if m.ndim == 3:
            m = m[:, :, 0]
        return m.astype(bool)
    if isinstance(seg, list):
        if mask_utils is None:
            return _polygons_to_mask(seg, height, width)
        rles = mask_utils.frPyObjects(seg, height, width)
        m = mask_utils.decode(rles)
        if m.ndim == 3:
            m = np.logical_or.reduce(m, axis=2)
        return m.astype(bool)
    raise ValueError(f"unknown segmentation type: {type(seg)}")


def _polygons_to_mask(polygons: List, height: int, width: int) -> np.ndarray:
    from PIL import Image, ImageDraw

    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    for poly in polygons:
        if len(poly) < 6:
            continue
        xy = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
        draw.polygon(xy, outline=1, fill=1)
    return np.array(img, dtype=np.uint8) > 0
