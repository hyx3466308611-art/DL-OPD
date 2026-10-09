"""Loss unit tests: KL direction, empty GT, no NaN (§15.3 #3–4)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dl_opd.losses import (
    LossWeights,
    assemble_opd_losses,
    pixel_kl_teacher_to_student,
    supervised_loss,
)


def test_kl_teacher_to_student_decreases_when_student_moves_to_teacher():
    t = torch.full((1, 8, 8), 0.8)
    s_far = torch.full((1, 8, 8), 0.2)
    s_near = torch.full((1, 8, 8), 0.75)
    kl_far = pixel_kl_teacher_to_student(s_far, t)
    kl_near = pixel_kl_teacher_to_student(s_near, t)
    assert kl_near < kl_far


def test_empty_gt_bce_no_nan():
    qs1 = torch.rand(2, 1, 16, 16)
    qs2 = torch.rand(2, 1, 16, 16)
    gt = torch.zeros(2, 1, 16, 16)
    empty = torch.tensor([True, True])
    loss = supervised_loss(qs1, qs2, gt, empty_mask=empty, weights=LossWeights())
    assert torch.isfinite(loss)


def test_assemble_gate_off_zero_opd():
    B, H, W = 2, 16, 16
    qs1 = torch.rand(B, 1, H, W)
    qs2 = torch.rand(B, 1, H, W)
    qt = torch.rand(B, 1, H, W)
    gt = (torch.rand(B, 1, H, W) > 0.5).float()
    empty = torch.zeros(B, dtype=torch.bool)
    g_kd = torch.zeros(B)
    bundle = assemble_opd_losses(
        qs1=qs1,
        qs2=qs2,
        gt=gt,
        qt2=qt,
        g_kd=g_kd,
        empty_mask=empty,
        identity_mask=None,
        improve_mask=None,
        weights=LossWeights(),
    )
    assert bundle.l_opd.item() == 0.0
    assert torch.isfinite(bundle.total)


if __name__ == "__main__":
    test_kl_teacher_to_student_decreases_when_student_moves_to_teacher()
    test_empty_gt_bce_no_nan()
    test_assemble_gate_off_zero_opd()
    print("ok")
