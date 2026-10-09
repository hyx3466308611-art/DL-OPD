"""Unified CIS / recovery / negative metrics (plan §12)."""

from __future__ import annotations

from typing import Any

import numpy as np


def _u8(mask: np.ndarray) -> np.ndarray:
    if mask.dtype == np.uint8:
        return (mask > 0).astype(np.uint8)
    return (mask > 0.5).astype(np.uint8)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _u8(a), _u8(b)
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    if union <= 0:
        return 1.0 if inter <= 0 else 0.0
    return inter / union


def recovery_gain(pred_after: np.ndarray, pred_before: np.ndarray, gt: np.ndarray) -> float:
    return mask_iou(pred_after, gt) - mask_iou(pred_before, gt)


def empty_accuracy(pred: np.ndarray, gt_is_empty: bool, area_eps: float = 1.0) -> float | None:
    """1 if pred empty when GT empty; None if not an empty-GT sample."""
    if not gt_is_empty:
        return None
    return 1.0 if float(_u8(pred).sum()) <= area_eps else 0.0


def false_positive_area(pred: np.ndarray, gt_is_empty: bool, image_hw: tuple[int, int] | None = None) -> float | None:
    """Fraction of image predicted positive when GT is empty."""
    if not gt_is_empty:
        return None
    p = _u8(pred)
    h, w = image_hw if image_hw is not None else p.shape[:2]
    return float(p.sum()) / float(max(1, h * w))


def boundary_maps(mask: np.ndarray) -> np.ndarray:
    m = _u8(mask).astype(np.float32)
    dx = np.abs(m[:, 1:] - m[:, :-1])
    dy = np.abs(m[1:, :] - m[:-1, :])
    bx = np.pad(dx, ((0, 0), (0, 1)))
    by = np.pad(dy, ((0, 1), (0, 0)))
    return np.clip(bx + by, 0, 1)


def boundary_f_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pb, tb = boundary_maps(pred), boundary_maps(gt)
    inter = float((pb * tb).sum())
    prec = inter / (float(pb.sum()) + eps)
    rec = inter / (float(tb.sum()) + eps)
    if prec + rec <= eps:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def hausdorff_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    """Symmetric HD on binary masks; returns 0 if both empty, large if one empty."""
    try:
        import cv2
    except ImportError:
        return float("nan")
    p, g = _u8(pred), _u8(gt)
    if p.sum() == 0 and g.sum() == 0:
        return 0.0
    if p.sum() == 0 or g.sum() == 0:
        return float(max(p.shape))
    dt_p = cv2.distanceTransform((1 - p).astype(np.uint8), cv2.DIST_L2, 3)
    dt_g = cv2.distanceTransform((1 - g).astype(np.uint8), cv2.DIST_L2, 3)
    # max dist from pred boundary pixels to gt and vice versa
    bp = boundary_maps(p) > 0
    bg = boundary_maps(g) > 0
    d1 = float(dt_g[bp].max()) if bp.any() else 0.0
    d2 = float(dt_p[bg].max()) if bg.any() else 0.0
    return max(d1, d2)


def over_under_ratio(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> dict[str, float]:
    p, g = _u8(pred), _u8(gt)
    fp = float(np.logical_and(p == 1, g == 0).sum())
    fn = float(np.logical_and(p == 0, g == 1).sum())
    gt_area = float(g.sum()) + eps
    return {"over_ratio": fp / gt_area, "under_ratio": fn / gt_area}


def ece_binary(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Sample-level ECE for flattened probs/labels in [0,1]."""
    probs = probs.astype(np.float64).ravel()
    labels = labels.astype(np.float64).ravel()
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (probs >= bins[i]) & (probs < bins[i + 1] if i < n_bins - 1 else probs <= bins[i + 1])
        if not np.any(m):
            continue
        conf = float(probs[m].mean())
        acc = float(labels[m].mean())
        ece += (m.mean()) * abs(acc - conf)
    return float(ece)


def summarize_recovery_batch(
    before: list[np.ndarray],
    after: list[np.ndarray],
    gts: list[np.ndarray],
) -> dict[str, Any]:
    gains = [recovery_gain(a, b, g) for a, b, g in zip(after, before, gts)]
    ious_b = [mask_iou(b, g) for b, g in zip(before, gts)]
    ious_a = [mask_iou(a, g) for a, g in zip(after, gts)]
    return {
        "n": len(gains),
        "mean_recovery_gain": float(np.mean(gains)) if gains else 0.0,
        "pos_frac": float(np.mean([g > 0 for g in gains])) if gains else 0.0,
        "mean_iou_before": float(np.mean(ious_b)) if ious_b else 0.0,
        "mean_iou_after": float(np.mean(ious_a)) if ious_a else 0.0,
    }
