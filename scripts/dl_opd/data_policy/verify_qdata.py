"""Multi-stage Q_data verification (plan §8). Separated from generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class QualityScores:
    c_sem: float
    c_geo: float
    c_teacher: float
    c_novel: float
    q_data: float
    accept: bool
    reject_reason: str | None = None
    notes: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerifyConfig:
    q_min: float = 0.35
    c_sem_min: float = 0.5
    c_geo_min: float = 0.4
    c_teacher_min: float = 0.0  # empty GT may ignore teacher
    c_novel_min: float = 0.2


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def verify_candidate(
    *,
    is_negative: bool,
    label_policy: str,
    semantic_ok: bool | None = None,
    geo_ok: bool | None = None,
    teacher_conf: float | None = None,
    teacher_recovery_gain: float | None = None,
    novelty: float | None = None,
    duplicate: bool = False,
    cfg: VerifyConfig | None = None,
) -> QualityScores:
    """Rule-based verifier for MVP. VLM/SAM hooks can feed semantic_ok/geo_ok later."""
    cfg = cfg or VerifyConfig()

    if duplicate:
        return QualityScores(
            c_sem=0.0,
            c_geo=0.0,
            c_teacher=0.0,
            c_novel=0.0,
            q_data=0.0,
            accept=False,
            reject_reason="duplicate",
        )

    # C_sem
    if semantic_ok is None:
        # heuristic: negatives and empty policy get mid score until external check
        c_sem = 0.7 if is_negative or label_policy == "empty" else 0.6
    else:
        c_sem = 1.0 if semantic_ok else 0.0

    # C_geo — not applicable the same way for empty
    if is_negative or label_policy == "empty":
        c_geo = 1.0 if (geo_ok is None or geo_ok) else 0.0
    elif geo_ok is None:
        c_geo = 0.6
    else:
        c_geo = 1.0 if geo_ok else 0.0

    # C_teacher — for empty GT, do NOT require teacher to draw something
    if is_negative or label_policy == "empty":
        c_teacher = 1.0  # N/A → neutral pass (documented rule)
    else:
        conf = _clip01(teacher_conf if teacher_conf is not None else 0.5)
        rg = teacher_recovery_gain
        if rg is not None and rg < 0:
            c_teacher = 0.2
        else:
            c_teacher = conf

    c_novel = _clip01(novelty if novelty is not None else 0.5)

    q = c_sem * c_geo * c_teacher * c_novel
    reason = None
    accept = True
    if c_sem < cfg.c_sem_min:
        accept, reason = False, "c_sem_low"
    elif c_geo < cfg.c_geo_min and not (is_negative or label_policy == "empty"):
        accept, reason = False, "c_geo_low"
    elif (not (is_negative or label_policy == "empty")) and c_teacher < cfg.c_teacher_min:
        accept, reason = False, "c_teacher_low"
    elif c_novel < cfg.c_novel_min:
        accept, reason = False, "c_novel_low"
    elif q < cfg.q_min:
        accept, reason = False, "q_data_low"

    return QualityScores(
        c_sem=_clip01(c_sem),
        c_geo=_clip01(c_geo),
        c_teacher=_clip01(c_teacher),
        c_novel=_clip01(c_novel),
        q_data=_clip01(q),
        accept=accept,
        reject_reason=reason,
        notes={"label_policy": label_policy, "is_negative": is_negative},
    )
