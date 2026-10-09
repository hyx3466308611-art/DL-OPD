"""Sample-level teacher gating for Mask-Policy OPD (implementation plan §6.2).

Priority (high → low):
  1 invalid / corrupt sample → reject
  2 empty GT / negative prompt → GT-only (zeros), no teacher mask
  3 ambiguous multi-solution → candidate-set / soft / reject
  4 high input IoU (over-edit risk) → no pixel KD
  5 low teacher confidence or dual-teacher inconsistency → soft or skip KD
  6 teacher not better than student → GT-only
  7 reliable teacher improvement → teacher-hard (or soft)

All thresholds are configurable; defaults are initial values, not claimed optima.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class TargetMode(str, Enum):
    REJECT = "reject"
    GT_ONLY = "gt_only"
    TEACHER_HARD = "teacher_hard"
    TEACHER_SOFT = "teacher_soft"
    CANDIDATE_SET = "candidate_set"
    IDENTITY = "identity"  # stabilize qs2 toward qs1 when high-IoU


@dataclass
class GateConfig:
    tau_hi: float = 0.85
    delta: float = 0.02
    conf_hard: float = 0.55
    conf_soft: float = 0.35
    consistency_min: float = 0.70
    enable_dual_teacher: bool = False
    enable_identity_on_high_iou: bool = True
    empty_gt_eps: float = 1e-6


@dataclass
class GateInputs:
    """Per-sample signals used by the gate (all optional except flags we need)."""

    gt_valid: bool = True
    gt_empty: bool = False
    multi_solution: bool = False
    data_corrupt: bool = False
    input_iou: float | None = None  # IoU(qs1, M); for empty GT use Empty Acc convention upstream
    teacher_iou: float | None = None  # IoU(qt2, M)
    student_iou: float | None = None  # same as input_iou usually
    teacher_confidence: float | None = None  # [0,1]
    dual_teacher_consistency: float | None = None  # [0,1] if dual teacher enabled
    recovery_gain: float | None = None  # teacher_iou - student_iou if both set


@dataclass
class GateDecision:
    g_kd: float  # 0 / soft weight / 1
    target_mode: TargetMode
    reason: str
    use_teacher_mask: bool
    force_zero_target: bool
    apply_identity: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["target_mode"] = self.target_mode.value
        return d


def _rg(inp: GateInputs) -> float | None:
    if inp.recovery_gain is not None:
        return float(inp.recovery_gain)
    if inp.teacher_iou is None or inp.student_iou is None:
        return None
    return float(inp.teacher_iou) - float(inp.student_iou)


def decide_teacher_gate(inp: GateInputs, cfg: GateConfig | None = None) -> GateDecision:
    """Return KD gate and target mode for one sample."""
    cfg = cfg or GateConfig()

    if inp.data_corrupt or not inp.gt_valid:
        return GateDecision(
            g_kd=0.0,
            target_mode=TargetMode.REJECT,
            reason="invalid_or_corrupt",
            use_teacher_mask=False,
            force_zero_target=False,
        )

    if inp.gt_empty:
        return GateDecision(
            g_kd=0.0,
            target_mode=TargetMode.GT_ONLY,
            reason="empty_gt_force_zero",
            use_teacher_mask=False,
            force_zero_target=True,
        )

    if inp.multi_solution:
        return GateDecision(
            g_kd=0.0,
            target_mode=TargetMode.CANDIDATE_SET,
            reason="multi_solution_no_hard_label",
            use_teacher_mask=False,
            force_zero_target=False,
        )

    i_s = inp.student_iou if inp.student_iou is not None else inp.input_iou
    if i_s is not None and i_s > cfg.tau_hi:
        return GateDecision(
            g_kd=0.0,
            target_mode=TargetMode.IDENTITY if cfg.enable_identity_on_high_iou else TargetMode.GT_ONLY,
            reason=f"high_iou_overedit_risk_iou={i_s:.4f}>tau_hi={cfg.tau_hi}",
            use_teacher_mask=False,
            force_zero_target=False,
            apply_identity=cfg.enable_identity_on_high_iou,
            extras={"input_iou": i_s},
        )

    conf = inp.teacher_confidence
    cons = inp.dual_teacher_consistency
    if cfg.enable_dual_teacher:
        if cons is None or cons < cfg.consistency_min:
            if conf is not None and conf >= cfg.conf_soft:
                return GateDecision(
                    g_kd=0.25,
                    target_mode=TargetMode.TEACHER_SOFT,
                    reason="dual_teacher_inconsistent_soft_only",
                    use_teacher_mask=True,
                    force_zero_target=False,
                    extras={"consistency": cons, "confidence": conf},
                )
            return GateDecision(
                g_kd=0.0,
                target_mode=TargetMode.GT_ONLY,
                reason="dual_teacher_inconsistent_skip_kd",
                use_teacher_mask=False,
                force_zero_target=False,
                extras={"consistency": cons},
            )

    if conf is not None and conf < cfg.conf_soft:
        return GateDecision(
            g_kd=0.0,
            target_mode=TargetMode.GT_ONLY,
            reason=f"teacher_confidence_too_low={conf:.4f}",
            use_teacher_mask=False,
            force_zero_target=False,
        )

    rg = _rg(inp)
    i_t = inp.teacher_iou
    if i_t is not None and i_s is not None:
        if i_t <= i_s + cfg.delta or (rg is not None and rg <= 0):
            return GateDecision(
                g_kd=0.0,
                target_mode=TargetMode.GT_ONLY,
                reason=f"teacher_not_better_it={i_t:.4f}_is={i_s:.4f}_delta={cfg.delta}",
                use_teacher_mask=False,
                force_zero_target=False,
                extras={"recovery_gain": rg},
            )

    if conf is not None and conf < cfg.conf_hard:
        return GateDecision(
            g_kd=0.5,
            target_mode=TargetMode.TEACHER_SOFT,
            reason=f"teacher_mid_confidence={conf:.4f}_soft_kd",
            use_teacher_mask=True,
            force_zero_target=False,
            extras={"recovery_gain": rg, "confidence": conf},
        )

    return GateDecision(
        g_kd=1.0,
        target_mode=TargetMode.TEACHER_HARD,
        reason="teacher_reliable_improvement",
        use_teacher_mask=True,
        force_zero_target=False,
        extras={"recovery_gain": rg, "confidence": conf},
    )


def summarize_gate_batch(decisions: list[GateDecision]) -> dict[str, Any]:
    """Aggregate gate branch counts for logging."""
    from collections import Counter

    c = Counter(d.reason.split("_")[0] if False else d.reason for d in decisions)
    # count by target_mode and whether KD on
    by_mode = Counter(d.target_mode.value for d in decisions)
    kd_on = sum(1 for d in decisions if d.g_kd > 0)
    return {
        "n": len(decisions),
        "kd_on": kd_on,
        "kd_rate": kd_on / max(1, len(decisions)),
        "by_mode": dict(by_mode),
        "by_reason": dict(c),
    }
