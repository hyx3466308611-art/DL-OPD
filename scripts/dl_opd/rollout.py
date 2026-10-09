"""Current-state student/teacher rollout helpers (plan §6.1)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def empty_mask_batch(batch_size: int, h: int, w: int, device, dtype) -> torch.Tensor:
    return torch.zeros(batch_size, 1, h, w, device=device, dtype=dtype)


def resize_mask_tensor(masks: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """masks: [B,1,H,W] or [B,H,W] → [B,1,h,w] in [0,1]."""
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)
    return F.interpolate(masks.float(), size=(h, w), mode="bilinear", align_corners=False).clamp(0, 1)


@torch.no_grad()
def decode_logits_to_prob(logits: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """logits [B,h,w] → prob [B,1,out_h,out_w]."""
    if logits.ndim == 4:
        logits = logits[:, 0]
    p = torch.sigmoid(logits).unsqueeze(1)
    return F.interpolate(p, size=(out_h, out_w), mode="bilinear", align_corners=False).clamp(0, 1)


def soft_iou_torch(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """pred/gt [B,1,H,W] or [B,H,W] → [B] soft IoU (threshold 0.5 for binary)."""
    if pred.ndim == 4:
        pred = pred[:, 0]
    if gt.ndim == 4:
        gt = gt[:, 0]
    p = (pred > 0.5).float()
    t = (gt > 0.5).float()
    inter = (p * t).flatten(1).sum(1)
    union = ((p + t) > 0).float().flatten(1).sum(1)
    empty = union <= 0
    iou = inter / union.clamp_min(eps)
    # both empty → 1
    both_empty = (p.flatten(1).sum(1) <= 0) & (t.flatten(1).sum(1) <= 0)
    iou = torch.where(both_empty, torch.ones_like(iou), iou)
    iou = torch.where(empty & (~both_empty), torch.zeros_like(iou), iou)
    return iou


def teacher_confidence_from_prob(prob: torch.Tensor) -> torch.Tensor:
    """Simple confidence: mean |p-0.5|*2 on predicted foreground, else 1-mean(p)."""
    if prob.ndim == 4:
        prob = prob[:, 0]
    fg = prob > 0.5
    conf = []
    for i in range(prob.shape[0]):
        pi = prob[i]
        if bool(fg[i].any()):
            conf.append(float(pi[fg[i]].mean().item()))
        else:
            conf.append(float(1.0 - pi.mean().item()))
    return torch.tensor(conf, device=prob.device, dtype=prob.dtype)


def np_from_prob(prob_b1hw: torch.Tensor) -> list[np.ndarray]:
    x = prob_b1hw.detach().float().cpu().numpy()
    if x.ndim == 4:
        x = x[:, 0]
    return [x[i] for i in range(x.shape[0])]
