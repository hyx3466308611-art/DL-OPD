"""Mask corruption ops for teacher refinement training (Path-1)."""

from __future__ import annotations

import random
from typing import Optional

import cv2
import numpy as np


def _rng(rng: Optional[random.Random]) -> random.Random:
    return rng if rng is not None else random


def _ensure_u8(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0.5).astype(np.uint8) if mask.dtype != np.uint8 else (mask > 0).astype(np.uint8)
    return m


def corrupt_erode(mask: np.ndarray, severity: float = 0.5, rng: Optional[random.Random] = None) -> np.ndarray:
    r = _rng(rng)
    m = _ensure_u8(mask)
    k = max(1, int(round(1 + 6 * severity)))
    it = max(1, int(round(1 + 2 * severity * r.random())))
    return cv2.erode(m, np.ones((k, k), np.uint8), iterations=it)


def corrupt_dilate(mask: np.ndarray, severity: float = 0.5, rng: Optional[random.Random] = None) -> np.ndarray:
    r = _rng(rng)
    m = _ensure_u8(mask)
    k = max(1, int(round(1 + 6 * severity)))
    it = max(1, int(round(1 + 2 * severity * r.random())))
    return cv2.dilate(m, np.ones((k, k), np.uint8), iterations=it)


def corrupt_drop_components(mask: np.ndarray, severity: float = 0.5, rng: Optional[random.Random] = None) -> np.ndarray:
    r = _rng(rng)
    m = _ensure_u8(mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 2:
        return corrupt_erode(m, severity, r)
    # label 0 is background
    keep = m.copy()
    comps = list(range(1, n))
    r.shuffle(comps)
    n_drop = max(1, int(round(len(comps) * (0.2 + 0.6 * severity))))
    for c in comps[:n_drop]:
        keep[labels == c] = 0
    if keep.any():
        return keep
    return corrupt_erode(m, severity, r)


def corrupt_translate(mask: np.ndarray, severity: float = 0.5, rng: Optional[random.Random] = None) -> np.ndarray:
    r = _rng(rng)
    m = _ensure_u8(mask)
    h, w = m.shape[:2]
    max_shift = max(1, int(round(0.08 * min(h, w) * (0.3 + severity))))
    dx = r.randint(-max_shift, max_shift)
    dy = r.randint(-max_shift, max_shift)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)


def corrupt_boundary_jitter(mask: np.ndarray, severity: float = 0.5, rng: Optional[random.Random] = None) -> np.ndarray:
    r = _rng(rng)
    m = _ensure_u8(mask)
    k = max(1, int(round(1 + 5 * severity)))
    if r.random() < 0.5:
        out = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    else:
        out = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    if r.random() < 0.5:
        out = corrupt_translate(out, severity * 0.5, r)
    return out


def corrupt_mask(
    mask: np.ndarray,
    severity: float = 0.5,
    rng: Optional[random.Random] = None,
    allow_empty: bool = False,
) -> np.ndarray:
    """
    Apply one random corruption to a binary mask.
    severity in [0, 1]: higher => stronger corruption.
    """
    r = _rng(rng)
    severity = float(np.clip(severity, 0.0, 1.0))
    m = _ensure_u8(mask)
    if m.sum() == 0:
        return m

    ops = [
        corrupt_erode,
        corrupt_dilate,
        corrupt_drop_components,
        corrupt_translate,
        corrupt_boundary_jitter,
    ]
    op = r.choice(ops)
    out = op(m, severity, r)

    # occasional second op at higher severity
    if severity > 0.45 and r.random() < 0.35:
        out = r.choice(ops)(out, severity, r)

    if not allow_empty and out.sum() == 0:
        # fall back to mild erode so the prompt is not trivially empty
        out = corrupt_erode(m, min(0.3, severity), r)
    return (out > 0).astype(np.uint8)


def curriculum_severity(step: int, total_steps: int, min_s: float = 0.2, max_s: float = 0.85) -> float:
    if total_steps <= 1:
        return max_s
    t = min(1.0, max(0.0, (step - 1) / float(total_steps - 1)))
    return min_s + (max_s - min_s) * t


def curriculum_mask_scale(
    step: int,
    total_steps: int,
    start_scale: float,
    end_scale: float | None,
    warmup_frac: float = 0.5,
) -> float:
    """Hold start_scale, then linear ramp to end_scale (safer than jumping scale)."""
    start_scale = float(start_scale)
    if end_scale is None:
        return start_scale
    end_scale = float(end_scale)
    if abs(end_scale - start_scale) < 1e-6:
        return start_scale
    warmup_frac = float(np.clip(warmup_frac, 0.0, 0.95))
    warmup_steps = int(total_steps * warmup_frac)
    if step <= warmup_steps:
        return start_scale
    ramp_steps = max(1, total_steps - warmup_steps)
    t = min(1.0, max(0.0, (step - warmup_steps) / float(ramp_steps)))
    return start_scale + (end_scale - start_scale) * t
