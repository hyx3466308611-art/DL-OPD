"""Optional boundary/elastic boost; force_op diagnosis delegates to base MaskCorrupter."""

from __future__ import annotations

from typing import Optional

import numpy as np

from data.mask_corrupter import (
    CurriculumStage,
    MaskCorrupter,
    STAGE_CONFIGS,
    _OP_MAP,
    _ensure_u8,
    _min_foreground_fraction,
    _op_morph,
    mask_iou,
)


class MaskCorrupterFixV1(MaskCorrupter):
    def __init__(self, seed: int = 1337, boundary_elastic_boost: float = 0.0):
        super().__init__(seed=seed)
        self.boundary_elastic_boost = float(boundary_elastic_boost)

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
        # 诊断强制算子：与主协议同一实现，保证可比
        if force_op is not None:
            return super().corrupt(
                gt_mask,
                stage,
                bg_mask=bg_mask,
                force_dropout=force_dropout,
                force_op=force_op,
                allow_compound=allow_compound,
            )

        rng = self._rng()
        gt = _ensure_u8(gt_mask)
        cfg = STAGE_CONFIGS[stage]
        severity = rng.uniform(cfg.severity_lo, cfg.severity_hi)

        do_dropout = (
            force_dropout if force_dropout is not None else (rng.random() < cfg.mask_dropout_prob)
        )
        if do_dropout:
            corrupted = np.zeros_like(gt)
            meta = {
                "op": "mask_dropout",
                "severity": severity,
                "stage": stage.value,
                "input_iou": mask_iou(corrupted, gt),
            }
            return corrupted, meta

        if gt.sum() == 0:
            corrupted = gt.copy()
            meta = {"op": "empty_gt", "severity": severity, "stage": stage.value, "input_iou": 1.0}
            return corrupted, meta

        boost_ops = ("boundary", "elastic")
        allowed = cfg.allowed_ops
        if (
            self.boundary_elastic_boost > 0
            and stage in (CurriculumStage.B, CurriculumStage.C)
            and rng.random() < self.boundary_elastic_boost
        ):
            pool = tuple(o for o in boost_ops if o in allowed)
            op_name = rng.choice(pool if pool else allowed)
        else:
            op_name = rng.choice(allowed)

        op_fn = _OP_MAP[op_name]
        if op_name == "false_pos":
            corrupted = op_fn(gt, severity, rng, bg_mask)
        else:
            corrupted = op_fn(gt, severity, rng)

        compound_p = 0.0 if allow_compound is False else cfg.compound_prob
        if rng.random() < compound_p:
            pool = [o for o in allowed if o != op_name]
            if not pool:
                pool = list(allowed)
            op2 = rng.choice(pool)
            fn2 = _OP_MAP[op2]
            sev2 = severity * 0.5
            if op2 == "false_pos":
                corrupted = fn2(corrupted, sev2, rng, bg_mask)
            else:
                corrupted = fn2(corrupted, sev2, rng)
            op_name = f"{op_name}+{op2}"

        corrupted = _ensure_u8(corrupted)
        if "+" in op_name and gt.sum() > 0:
            corrupted = _min_foreground_fraction(corrupted, gt, min_frac=0.10)
        if corrupted.sum() == 0 and stage != CurriculumStage.C:
            corrupted = _op_morph(gt, min(0.35, severity), rng)

        meta = {
            "op": op_name,
            "severity": severity,
            "stage": stage.value,
            "input_iou": mask_iou(corrupted, gt),
        }
        return corrupted, meta
