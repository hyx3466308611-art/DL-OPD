"""Fix v1 losses: negative suppression, identity L1 preserve, masked dice."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def batch_mask_iou(pred_bin: torch.Tensor, target_bin: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred_bin.float()
    tgt = target_bin.float()
    inter = (pred * tgt).sum(dim=(1, 2))
    union = (pred + tgt - pred * tgt).sum(dim=(1, 2))
    return inter / (union + eps)


def negative_suppression_loss(logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """L_neg = BCE(logits, 0) on empty-GT samples."""
    empty = gt.sum(dim=(1, 2)) < 0.5
    if not empty.any():
        return logits.new_zeros(())
    z = torch.zeros_like(gt[empty])
    return F.binary_cross_entropy_with_logits(logits[empty], z)


def identity_preserve_loss_l1(
    logits: torch.Tensor,
    input_soft: torch.Tensor,
    gt: torch.Tensor,
    iou_thresh: float,
) -> torch.Tensor:
    with torch.no_grad():
        inp_bin = (input_soft > 0.5).float()
        gt_bin = (gt > 0.5).float()
        per_iou = batch_mask_iou(inp_bin, gt_bin)
        w = (per_iou >= iou_thresh).float()
    if w.sum() < 0.5:
        return logits.new_zeros(())
    pred = torch.sigmoid(logits)
    l1 = F.l1_loss(pred, input_soft, reduction="none").mean(dim=(1, 2))
    return (l1 * w).sum() / w.sum().clamp(min=1.0)


def dice_loss_masked(logits: torch.Tensor, gt: torch.Tensor, dice_fn) -> torch.Tensor:
    """Skip dice on empty-GT rows (avoid degenerate gradients)."""
    non_empty = gt.sum(dim=(1, 2)) >= 0.5
    if not non_empty.any():
        return logits.new_zeros(())
    return dice_fn(logits[non_empty], gt[non_empty])
