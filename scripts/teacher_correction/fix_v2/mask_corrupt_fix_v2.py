"""fix_v2 corruption policy isolated from the shared/base corruption implementation."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from data.mask_corrupter import (
    FORCEABLE_OPS,
    PER_OP_DIAGNOSIS_OPS,
    CurriculumStage,
    MaskCorrupter,
    STAGE_CONFIGS,
    _OP_MAP,
    _ensure_u8,
    _op_morph,
    mask_iou,
)


FIX_V2_STAGE_CONFIGS = {
    stage: replace(config) for stage, config in STAGE_CONFIGS.items()
}
FIX_V2_STAGE_CONFIGS[CurriculumStage.B] = replace(
    FIX_V2_STAGE_CONFIGS[CurriculumStage.B],
    mask_dropout_prob=0.15,
)
FIX_V2_STAGE_CONFIGS[CurriculumStage.C] = replace(
    FIX_V2_STAGE_CONFIGS[CurriculumStage.C],
    mask_dropout_prob=0.30,
)


class MaskCorrupterFixV2(MaskCorrupter):
    """
    B/C dropout strengthened and non-C destructive corruptions protected by a 5% area floor.

    force_op is retained for per-operator diagnosis and always disables compound corruption.
    """

    def stage_config(self, stage: CurriculumStage):
        return FIX_V2_STAGE_CONFIGS[stage]

    def corrupt(
        self,
        gt_mask: np.ndarray,
        stage: CurriculumStage,
        *,
        bg_mask: Optional[np.ndarray] = None,
        force_dropout: Optional[bool] = None,
        force_op: Optional[str] = None,
        allow_compound: Optional[bool] = None,
    ) -> tuple[np.ndarray, dict]:
        rng = self._rng()
        gt = _ensure_u8(gt_mask)
        cfg = self.stage_config(stage)
        severity = rng.uniform(cfg.severity_lo, cfg.severity_hi)

        if force_op is not None:
            if force_op not in FORCEABLE_OPS:
                raise ValueError(f"Unknown force_op={force_op!r}; choose from {sorted(FORCEABLE_OPS)}")
            if force_op == "mask_dropout":
                corrupted = np.zeros_like(gt)
                return corrupted, self._meta(corrupted, gt, force_op, severity, stage, force=True)
            if force_op == "empty_gt":
                if gt.sum() == 0:
                    return gt.copy(), self._meta(gt, gt, force_op, severity, stage, force=True)
                meta = self._meta(np.zeros_like(gt), gt, force_op, severity, stage, force=True)
                meta.update(skip=True, skip_reason="gt_not_empty")
                return np.zeros_like(gt), meta
            if gt.sum() == 0:
                meta = self._meta(gt, gt, force_op, severity, stage, force=True)
                meta.update(skip=True, skip_reason="empty_gt_incompatible")
                return gt.copy(), meta
            corrupted = self._apply(force_op, gt, severity, rng, bg_mask)
            corrupted = self._area_guard(corrupted, gt, stage, rng)
            return corrupted, self._meta(corrupted, gt, force_op, severity, stage, force=True)

        do_dropout = (
            force_dropout if force_dropout is not None else rng.random() < cfg.mask_dropout_prob
        )
        if do_dropout:
            corrupted = np.zeros_like(gt)
            return corrupted, self._meta(corrupted, gt, "mask_dropout", severity, stage)

        if gt.sum() == 0:
            return gt.copy(), self._meta(gt, gt, "empty_gt", severity, stage)

        op_name = rng.choice(cfg.allowed_ops)
        corrupted = self._apply(op_name, gt, severity, rng, bg_mask)

        compound_enabled = True if allow_compound is None else bool(allow_compound)
        if compound_enabled and rng.random() < cfg.compound_prob:
            pool = [name for name in cfg.allowed_ops if name != op_name]
            op2 = rng.choice(pool)
            corrupted = self._apply(op2, corrupted, severity * 0.5, rng, bg_mask)
            op_name = f"{op_name}+{op2}"

        corrupted = self._area_guard(corrupted, gt, stage, rng)
        return corrupted, self._meta(corrupted, gt, op_name, severity, stage)

    @staticmethod
    def _apply(name, mask, severity, rng, bg_mask):
        fn = _OP_MAP[name]
        if name == "false_pos":
            return fn(mask, severity, rng, bg_mask)
        return fn(mask, severity, rng)

    @staticmethod
    def _area_guard(corrupted, gt, stage, rng):
        corrupted = _ensure_u8(corrupted)
        gt_area = float(gt.sum())
        corrupted_area = float(corrupted.sum())
        if gt_area > 0 and corrupted_area / gt_area < 0.05 and stage != CurriculumStage.C:
            corrupted = _op_morph(gt, 0.1, rng)
        return _ensure_u8(corrupted)

    @staticmethod
    def _meta(corrupted, gt, op, severity, stage, force=False):
        return {
            "op": op,
            "severity": severity,
            "stage": stage.value,
            "input_iou": mask_iou(corrupted, gt),
            "force_op": force,
        }


__all__ = [
    "FIX_V2_STAGE_CONFIGS",
    "FORCEABLE_OPS",
    "PER_OP_DIAGNOSIS_OPS",
    "MaskCorrupterFixV2",
]
