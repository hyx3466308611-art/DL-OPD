"""Joint distillation losses for Mask-Policy OPD (plan §6.4).

KL direction is fixed as KL(p_teacher || p_student) for teacher→student distillation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    lambda_opd: float = 1.0
    lambda_bd: float = 0.5
    lambda_rep: float = 0.0  # enable in full / Stage later
    lambda_col: float = 0.0  # ramp in Stage-4
    lambda_anc: float = 0.2
    lambda_dice: float = 1.0
    empty_bce_weight: float = 3.0
    temperature: float = 2.0
    w_qs1: float = 1.0
    w_qs2: float = 1.0
    identity_weight: float = 0.5


def _flat(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


def soft_dice_loss(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """prob/target: [B,H,W] or [B,1,H,W] in [0,1]. Skips pure-empty targets safely."""
    p = _flat(prob.float())
    t = _flat(target.float())
    # per-sample
    inter = (p * t).sum(dim=1)
    denom = p.sum(dim=1) + t.sum(dim=1)
    # empty GT: if target sum ~0, dice is undefined; contribute 0 when pred also empty,
    # else penalize via (1 - 0) only when we still want dice — caller should skip for empty.
    empty = t.sum(dim=1) < eps
    dice = (2 * inter + eps) / (denom + eps)
    loss = 1.0 - dice
    loss = torch.where(empty, torch.zeros_like(loss), loss)
    return loss.mean()


def bce_logits_or_prob(pred: torch.Tensor, target: torch.Tensor, pred_is_logit: bool) -> torch.Tensor:
    t = target.float()
    if pred_is_logit:
        return F.binary_cross_entropy_with_logits(pred, t)
    p = pred.clamp(1e-6, 1.0 - 1e-6)
    return F.binary_cross_entropy(p, t)


def supervised_loss(
    qs1: torch.Tensor,
    qs2: torch.Tensor,
    gt: torch.Tensor,
    *,
    empty_mask: torch.Tensor | None,
    weights: LossWeights,
    pred_is_logit: bool = False,
) -> torch.Tensor:
    """L_sup on qs1 and qs2. empty_mask: [B] bool — use weighted BCE-only to zeros."""
    losses = []
    for w, q in ((weights.w_qs1, qs1), (weights.w_qs2, qs2)):
        if empty_mask is None:
            bce = bce_logits_or_prob(q, gt, pred_is_logit)
            dice = soft_dice_loss(
                torch.sigmoid(q) if pred_is_logit else q,
                gt,
            )
            losses.append(w * (bce + weights.lambda_dice * dice))
            continue
        # per-sample mix
        B = q.shape[0]
        sample_losses = []
        for i in range(B):
            qi = q[i : i + 1]
            gi = gt[i : i + 1]
            if bool(empty_mask[i]):
                bce = bce_logits_or_prob(qi, torch.zeros_like(gi), pred_is_logit)
                sample_losses.append(weights.empty_bce_weight * bce)
            else:
                bce = bce_logits_or_prob(qi, gi, pred_is_logit)
                dice = soft_dice_loss(torch.sigmoid(qi) if pred_is_logit else qi, gi)
                sample_losses.append(bce + weights.lambda_dice * dice)
        losses.append(w * torch.stack(sample_losses).mean())
    return sum(losses) / len(losses)


def pixel_kl_teacher_to_student(
    student_prob: torch.Tensor,
    teacher_prob: torch.Tensor,
    *,
    temperature: float = 2.0,
    weight_map: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """KL(Bern(p_t^T) || Bern(p_s^T)) averaged with optional spatial weights.

    student_prob / teacher_prob: [B,H,W] or [B,1,H,W] in [0,1].
    """
    T = max(float(temperature), 1e-3)
    # temperature on Bernoulli probs via logit soft: p^(1/T) then renormalize binary
    ps = student_prob.float().clamp(eps, 1 - eps)
    pt = teacher_prob.float().clamp(eps, 1 - eps)
    # soft Bernoulli with temperature: use logits / T
    logit_s = torch.log(ps) - torch.log(1 - ps)
    logit_t = torch.log(pt) - torch.log(1 - pt)
    ps_t = torch.sigmoid(logit_s / T)
    pt_t = torch.sigmoid(logit_t / T)
    # KL(t||s) = pt log(pt/ps) + (1-pt) log((1-pt)/(1-ps))
    kl = pt_t * (torch.log(pt_t + eps) - torch.log(ps_t + eps)) + (1 - pt_t) * (
        torch.log(1 - pt_t + eps) - torch.log(1 - ps_t + eps)
    )
    if weight_map is None:
        return kl.mean()
    w = weight_map.float()
    if w.ndim == 3 and kl.ndim == 4:
        w = w.unsqueeze(1)
    elif w.ndim == 4 and kl.ndim == 3:
        kl = kl.unsqueeze(1)
    denom = w.sum().clamp_min(eps)
    return (kl * w).sum() / denom


def default_opd_weight_map(
    student_prob: torch.Tensor,
    teacher_prob: torch.Tensor,
    gt: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """w = entropy * disagreement * (boundary hint if gt)."""
    ps = student_prob.float().clamp(eps, 1 - eps)
    pt = teacher_prob.float().clamp(eps, 1 - eps)
    ent = -(ps * torch.log(ps) + (1 - ps) * torch.log(1 - ps))
    disagree = (ps - pt).abs()
    w = (ent + eps) * (disagree + eps)
    if gt is not None:
        # simple boundary: |grad| of gt via finite differences
        g = gt.float()
        if g.ndim == 3:
            g = g.unsqueeze(1)
        dx = (g[:, :, :, 1:] - g[:, :, :, :-1]).abs()
        dy = (g[:, :, 1:, :] - g[:, :, :-1, :]).abs()
        bx = F.pad(dx, (0, 1, 0, 0))
        by = F.pad(dy, (0, 0, 0, 1))
        bd = (bx + by).clamp(0, 1)
        if w.ndim == 3:
            bd = bd.squeeze(1)
        w = w * (1.0 + bd)
    return w


def boundary_dice_loss(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Boundary Dice on soft masks (non-empty GT only recommended)."""
    p = pred.float()
    t = gt.float()
    if p.ndim == 3:
        p = p.unsqueeze(1)
        t = t.unsqueeze(1)

    def _boundary(x: torch.Tensor) -> torch.Tensor:
        dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs()
        dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs()
        bx = F.pad(dx, (0, 1, 0, 0))
        by = F.pad(dy, (0, 0, 0, 1))
        return (bx + by).clamp(0, 1)

    pb, tb = _boundary(p), _boundary(t)
    inter = (pb * tb).flatten(1).sum(1)
    denom = pb.flatten(1).sum(1) + tb.flatten(1).sum(1)
    empty = tb.flatten(1).sum(1) < eps
    dice = (2 * inter + eps) / (denom + eps)
    loss = 1.0 - dice
    loss = torch.where(empty, torch.zeros_like(loss), loss)
    return loss.mean()


def collapse_loss(
    qs1: torch.Tensor,
    qs2: torch.Tensor,
    gt: torch.Tensor,
    *,
    improve_mask: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Distill qs2 → one-pass qs1 only where qs2 improves vs GT. stop-grad on qs2."""
    if improve_mask.dtype != torch.bool:
        improve_mask = improve_mask > 0.5
    if not bool(improve_mask.any()):
        return qs1.new_zeros(())
    # select improved samples
    idx = improve_mask.nonzero(as_tuple=False).view(-1)
    s = qs1.index_select(0, idx)
    t = qs2.detach().index_select(0, idx)
    return pixel_kl_teacher_to_student(s, t, temperature=temperature)


def identity_loss(qs2: torch.Tensor, qs1: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(qs2, qs1.detach())


def representation_loss_placeholder(
    student_feats: dict[str, torch.Tensor] | None,
    teacher_feats: dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    """L_rep: align named features if provided; else zero.

    Full alignment (token/EOS/adapter/mask token + projections) is wired when
    feature hooks are registered in the training loop.
    """
    if not student_feats or not teacher_feats:
        device = torch.device("cpu")
        return torch.zeros((), device=device)
    keys = sorted(set(student_feats) & set(teacher_feats))
    if not keys:
        return next(iter(student_feats.values())).new_zeros(())
    losses = []
    for k in keys:
        s, t = student_feats[k], teacher_feats[k].detach()
        if s.shape != t.shape:
            # flatten mean-pool mismatch as fallback
            s = s.float().flatten(1).mean(1)
            t = t.float().flatten(1).mean(1)
            losses.append(F.mse_loss(s, t))
        else:
            losses.append(F.mse_loss(s.float(), t.float()))
    return sum(losses) / len(losses)


@dataclass
class OPDLossBundle:
    total: torch.Tensor
    l_sup: torch.Tensor
    l_opd: torch.Tensor
    l_bd: torch.Tensor
    l_rep: torch.Tensor
    l_col: torch.Tensor
    l_anc: torch.Tensor
    l_id: torch.Tensor

    def as_log_dict(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach().item()),
            "l_sup": float(self.l_sup.detach().item()),
            "l_opd": float(self.l_opd.detach().item()),
            "l_bd": float(self.l_bd.detach().item()),
            "l_rep": float(self.l_rep.detach().item()),
            "l_col": float(self.l_col.detach().item()),
            "l_anc": float(self.l_anc.detach().item()),
            "l_id": float(self.l_id.detach().item()),
        }


def assemble_opd_losses(
    *,
    qs1: torch.Tensor,
    qs2: torch.Tensor,
    gt: torch.Tensor,
    qt2: torch.Tensor | None,
    g_kd: torch.Tensor,
    empty_mask: torch.Tensor,
    identity_mask: torch.Tensor | None,
    improve_mask: torch.Tensor | None,
    weights: LossWeights,
    anchor_loss: torch.Tensor | None = None,
    student_feats: dict | None = None,
    teacher_feats: dict | None = None,
    pred_is_logit: bool = False,
) -> OPDLossBundle:
    """Combine all terms. g_kd: [B] weights in [0,1]."""

    def _prob(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) if pred_is_logit else x

    l_sup = supervised_loss(
        qs1, qs2, gt, empty_mask=empty_mask, weights=weights, pred_is_logit=pred_is_logit
    )

    # OPD only on non-empty + g_kd>0
    l_opd = qs1.new_zeros(())
    if qt2 is not None and float(g_kd.sum().item()) > 0:
        ps = _prob(qs2)
        pt = _prob(qt2).detach()
        wmap = default_opd_weight_map(ps, pt, gt=None if bool(empty_mask.any()) else gt)
        # scale spatial loss by per-sample g_kd
        B = ps.shape[0]
        per = []
        for i in range(B):
            gk = float(g_kd[i].item())
            if gk <= 0 or bool(empty_mask[i]):
                per.append(ps.new_zeros(()))
                continue
            wi = wmap[i : i + 1]
            li = pixel_kl_teacher_to_student(
                ps[i : i + 1],
                pt[i : i + 1],
                temperature=weights.temperature,
                weight_map=wi,
            )
            per.append(gk * li)
        l_opd = torch.stack(per).mean()

    # boundary on non-empty
    l_bd = qs1.new_zeros(())
    if (~empty_mask).any():
        idx = (~empty_mask).nonzero(as_tuple=False).view(-1)
        l_bd = boundary_dice_loss(_prob(qs2).index_select(0, idx), gt.index_select(0, idx))

    l_rep = representation_loss_placeholder(student_feats, teacher_feats)
    if not torch.is_tensor(l_rep) or l_rep.device != qs1.device:
        l_rep = qs1.new_zeros(())

    l_col = qs1.new_zeros(())
    if improve_mask is not None and weights.lambda_col > 0:
        l_col = collapse_loss(
            _prob(qs1),
            _prob(qs2),
            gt,
            improve_mask=improve_mask & (~empty_mask),
            temperature=weights.temperature,
        )

    l_id = qs1.new_zeros(())
    if identity_mask is not None and bool(identity_mask.any()):
        idx = identity_mask.nonzero(as_tuple=False).view(-1)
        l_id = identity_loss(_prob(qs2).index_select(0, idx), _prob(qs1).index_select(0, idx))

    l_anc = anchor_loss if anchor_loss is not None else qs1.new_zeros(())

    total = (
        l_sup
        + weights.lambda_opd * l_opd
        + weights.lambda_bd * l_bd
        + weights.lambda_rep * l_rep
        + weights.lambda_col * l_col
        + weights.lambda_anc * l_anc
        + weights.identity_weight * l_id
    )
    return OPDLossBundle(
        total=total,
        l_sup=l_sup,
        l_opd=l_opd,
        l_bd=l_bd,
        l_rep=l_rep,
        l_col=l_col,
        l_anc=l_anc,
        l_id=l_id,
    )
