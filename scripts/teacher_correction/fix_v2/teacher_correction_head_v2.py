"""fix_v2 correction head: global + spatial confidence gated fusion."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.language_adapter import LanguageAdapter
from models.teacher_correction_model import MaskStateEncoder


class TeacherCorrectionHead(nn.Module):
    """Text prompts fused with mask-state features using two-level confidence gates."""

    def __init__(self, plm: LanguageAdapter, transformer_dim: int):
        super().__init__()
        self.plm = plm
        self.mask_encoder = MaskStateEncoder(transformer_dim=transformer_dim)

        # sigmoid(-1) ~= 0.269: initially weaken unreliable mask prompts.
        self.global_gate = nn.Parameter(torch.tensor(-1.0))
        hidden = max(1, transformer_dim // 4)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(transformer_dim, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    @property
    def gate_value(self) -> float:
        return torch.sigmoid(self.global_gate).item()

    def forward(
        self,
        texts: list[str],
        H: int,
        W: int,
        image_paths: list[str],
        input_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask_in = torch.clamp(input_masks, 0.0, 1.0)
        sparse, dense_text = self.plm(texts, H, W, image_paths=image_paths)
        mask_feat = self.mask_encoder(mask_in, H, W)

        g_global = torch.sigmoid(self.global_gate)
        g_spatial = self.spatial_gate(mask_feat)
        effective_gate = g_global * g_spatial
        dense = dense_text + effective_gate * mask_feat
        return sparse, dense

    def load_previous_head_partial(self, state_dict: dict) -> tuple[list[str], list[str]]:
        """
        Transfer only shape-compatible mask_encoder parameters.

        Deliberately do not import the old scalar/spatial gates: fix_v2 must start
        with global_gate=-1 and learn its new 3x3 spatial gate from scratch.
        """
        own = self.state_dict()
        mapped = {
            key: value
            for key, value in state_dict.items()
            if key.startswith("mask_encoder.") and key in own and value.shape == own[key].shape
        }
        incompatible = self.load_state_dict(mapped, strict=False)
        return list(incompatible.missing_keys), list(incompatible.unexpected_keys)
