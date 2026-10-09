"""MaskStateEncoder shape / finite tests (§15.3 #1)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.teacher_correction_model import MaskStateEncoder


def _run(mask):
    enc = MaskStateEncoder(transformer_dim=256)
    enc.eval()
    with torch.no_grad():
        y = enc(mask, 64, 64)
    assert y.shape == (mask.shape[0], 256, 64, 64)
    assert torch.isfinite(y).all()


def test_variants():
    B = 2
    _run(torch.zeros(B, 1, 128, 128))
    _run(torch.ones(B, 1, 128, 128))
    pt = torch.zeros(B, 1, 128, 128)
    pt[:, :, 64, 64] = 1
    _run(pt)
    _run(torch.rand(B, 1, 96, 120))


if __name__ == "__main__":
    test_variants()
    print("ok")
