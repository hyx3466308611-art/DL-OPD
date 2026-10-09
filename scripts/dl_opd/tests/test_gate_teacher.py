"""Unit tests for teacher gating (§15.3 #2)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dl_opd.gate_teacher import GateConfig, GateInputs, TargetMode, decide_teacher_gate


def test_empty_gt_forces_zero():
    d = decide_teacher_gate(GateInputs(gt_empty=True, teacher_iou=0.9, student_iou=0.0))
    assert d.force_zero_target and d.g_kd == 0 and d.target_mode == TargetMode.GT_ONLY


def test_high_iou_blocks_kd():
    d = decide_teacher_gate(
        GateInputs(student_iou=0.92, teacher_iou=0.95, teacher_confidence=0.9),
        GateConfig(tau_hi=0.85),
    )
    assert d.g_kd == 0 and d.apply_identity


def test_teacher_not_better():
    d = decide_teacher_gate(
        GateInputs(student_iou=0.5, teacher_iou=0.4, teacher_confidence=0.9),
        GateConfig(delta=0.02),
    )
    assert d.g_kd == 0 and d.target_mode == TargetMode.GT_ONLY


def test_reliable_teacher():
    d = decide_teacher_gate(
        GateInputs(student_iou=0.3, teacher_iou=0.7, teacher_confidence=0.8),
    )
    assert d.g_kd == 1.0 and d.target_mode == TargetMode.TEACHER_HARD


def test_low_confidence_soft_or_skip():
    d = decide_teacher_gate(
        GateInputs(student_iou=0.3, teacher_iou=0.7, teacher_confidence=0.4),
        GateConfig(conf_hard=0.55, conf_soft=0.35),
    )
    assert d.target_mode == TargetMode.TEACHER_SOFT and 0 < d.g_kd < 1


def test_reject_corrupt():
    d = decide_teacher_gate(GateInputs(data_corrupt=True))
    assert d.target_mode == TargetMode.REJECT


if __name__ == "__main__":
    for fn in [
        test_empty_gt_forces_zero,
        test_high_iou_blocks_kd,
        test_teacher_not_better,
        test_reliable_teacher,
        test_low_confidence_soft_or_skip,
        test_reject_corrupt,
    ]:
        fn()
        print("ok", fn.__name__)
