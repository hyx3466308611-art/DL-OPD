"""Earlier Stage C and slightly longer hard curriculum for fix v1."""

from __future__ import annotations

from data.mask_corrupter import CurriculumStage


def curriculum_stage_from_step_fix(step: int, total_steps: int, c_start: float = 0.45) -> CurriculumStage:
    """
    A: first 25% steps (was 30%)
    B: until c_start (default 45% — Stage C starts earlier than base 70%)
    C: remainder
    """
    if total_steps <= 1:
        return CurriculumStage.C
    p = min(1.0, max(0.0, (step - 1) / float(total_steps - 1)))
    if p < 0.25:
        return CurriculumStage.A
    if p < c_start:
        return CurriculumStage.B
    return CurriculumStage.C
