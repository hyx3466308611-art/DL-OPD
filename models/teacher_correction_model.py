"""
Correction Teacher: fuse text dense prompts with corrupted-mask state encoding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.language_adapter import LanguageAdapter


class MaskStateEncoder(nn.Module):
    """
    Maps a soft mask [B,1,H,W] in [0,1] to SAM2 dense prompt space [B,C,H,W].
    Supports continuous probability maps (e.g. student sigmoid outputs in Stage 2).
    """

    def __init__(self, transformer_dim: int = 256, mid_channels: int = 64):
        super().__init__()
        c = mid_channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, stride=2, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, c * 2),
            nn.SiLU(),
        )
        self.body = nn.Sequential(
            nn.Conv2d(c * 2, c * 2, 3, padding=1),
            nn.GroupNorm(8, c * 2),
            nn.SiLU(),
            nn.Conv2d(c * 2, c * 2, 3, padding=1),
            nn.GroupNorm(8, c * 2),
            nn.SiLU(),
        )
        self.to_prompt = nn.Conv2d(c * 2, transformer_dim, 1)

    def forward(self, mask_b1hw: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
        x = torch.clamp(mask_b1hw, 0.0, 1.0)
        x = self.stem(x)
        x = self.body(x) + x
        x = self.to_prompt(x)
        x = F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)
        return x


class TeacherCorrectionHead(nn.Module):
    """PLM text prompts + mask state -> sparse/dense SAM prompts (gated fusion)."""

    def __init__(self, plm: LanguageAdapter, transformer_dim: int):
        super().__init__()
        self.plm = plm
        self.mask_encoder = MaskStateEncoder(transformer_dim=transformer_dim)
        # sigmoid(0)=0.5: neutral mask injection at start
        self.gate = nn.Parameter(torch.tensor(0.0))

    @property
    def gate_value(self) -> float:
        return torch.sigmoid(self.gate).item()

    def forward(
        self,
        texts: list[str],
        H: int,
        W: int,
        image_paths: list[str],
        input_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        input_masks: [B,1,H_img,W_img] float in [0.0, 1.0] (hard or soft probabilities).
        """
        mask_in = torch.clamp(input_masks, 0.0, 1.0)
        sparse, dense_text = self.plm(texts, H, W, image_paths=image_paths)
        mask_feat = self.mask_encoder(mask_in, H, W)
        g = torch.sigmoid(self.gate)
        dense = dense_text + g * mask_feat
        return sparse, dense


class TeacherCorrectionHeadV2(nn.Module):
    """
    Confidence-aware gated fusion: global gate + spatial gate; suppress mask branch on negatives.
    Used by fix_v1 training / eval (spatial conv init ~1 so V1 checkpoints transfer cleanly).
    """

    def __init__(self, plm: LanguageAdapter, transformer_dim: int):
        super().__init__()
        self.plm = plm
        self.mask_encoder = MaskStateEncoder(transformer_dim=transformer_dim)
        self.gate = nn.Parameter(torch.tensor(-0.2))
        self.spatial_gate_conv = nn.Sequential(
            nn.Conv2d(transformer_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.spatial_gate_conv[0].weight)
        nn.init.constant_(self.spatial_gate_conv[0].bias, 2.0)

    @property
    def gate_value(self) -> float:
        return torch.sigmoid(self.gate).item()

    def forward(
        self,
        texts: list[str],
        H: int,
        W: int,
        image_paths: list[str],
        input_masks: torch.Tensor,
        *,
        suppress_mask_branch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        suppress_mask_branch: optional [B] float in {0,1}; 0 => no mask injection (negative / empty GT).
        """
        mask_in = torch.clamp(input_masks, 0.0, 1.0)
        sparse, dense_text = self.plm(texts, H, W, image_paths=image_paths)
        mask_feat = self.mask_encoder(mask_in, H, W)
        global_g = torch.sigmoid(self.gate)
        spatial_g = self.spatial_gate_conv(mask_feat)
        effective_gate = global_g * spatial_g

        B = mask_in.shape[0]
        if suppress_mask_branch is not None:
            s = suppress_mask_branch.to(mask_in.device, dtype=mask_feat.dtype).view(B, 1, 1, 1)
            effective_gate = effective_gate * s

        mask_sum = mask_in.flatten(1).sum(dim=1)
        empty_in = (mask_sum < 1e-4).to(mask_feat.dtype).view(B, 1, 1, 1)
        effective_gate = effective_gate * (1.0 - empty_in * 0.92)

        dense = dense_text + effective_gate * mask_feat
        return sparse, dense

    def load_v1_head_partial(self, state_dict: dict, strict: bool = False) -> None:
        """Load mask_encoder + scalar gate from TeacherCorrectionHead V1 checkpoint."""
        own = self.state_dict()
        mapped = {}
        for k, v in state_dict.items():
            if k in own and v.shape == own[k].shape:
                mapped[k] = v
            elif k == "gate":
                mapped["gate"] = v
        self.load_state_dict(mapped, strict=False)
