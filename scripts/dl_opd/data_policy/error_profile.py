"""Fixed diagnostic set → error profile u (plan §7.1–7.2)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

CONCEPT_KEYS = (
    "entities",
    "spatial",
    "relations",
    "affordances",
    "physics",
)

U_KEYS = (
    "E_ent",
    "E_spa",
    "E_rel",
    "E_aff",
    "E_phy",
    "E_neg",
    "E_bd",
    "E_cal",
)

CONCEPT_TO_E = {
    "entities": "E_ent",
    "spatial": "E_spa",
    "relations": "E_rel",
    "affordances": "E_aff",
    "physics": "E_phy",
}


@dataclass
class DiagnosticRecord:
    sample_id: str
    image_id: str
    prompt: str
    concept: str
    is_negative: bool
    iou: float
    pred_empty: bool
    gt_empty: bool
    confidence: float | None = None
    boundary_f: float | None = None
    teacher_disagree: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorProfile:
    """Normalized error vector u and bookkeeping."""

    u: dict[str, float]
    n_diag: int
    by_concept_n: dict[str, int]
    by_concept_err: dict[str, float]
    raw: dict[str, float]
    version: str = "v1"
    outer_cycle: int = 0

    def as_vector(self) -> list[float]:
        return [float(self.u[k]) for k in U_KEYS]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "ErrorProfile":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return ErrorProfile(**d)


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def compute_error_profile(
    records: Iterable[DiagnosticRecord],
    *,
    outer_cycle: int = 0,
    version: str = "v1",
) -> ErrorProfile:
    """Rule-based MVP: error rate = 1 - mean IoU (or empty-acc failure for negatives)."""
    recs = list(records)
    raw = {k: 0.0 for k in U_KEYS}
    counts = {k: 0 for k in U_KEYS}
    by_c_n: dict[str, int] = {c: 0 for c in CONCEPT_KEYS}
    by_c_err: dict[str, float] = {c: 0.0 for c in CONCEPT_KEYS}

    cal_abs = []
    bd_errs = []

    for r in recs:
        if r.is_negative or r.gt_empty:
            # empty accuracy failure
            fail = 0.0 if r.pred_empty else 1.0
            raw["E_neg"] += fail
            counts["E_neg"] += 1
        else:
            ekey = CONCEPT_TO_E.get(r.concept, None)
            err = _clip01(1.0 - float(r.iou))
            if ekey:
                raw[ekey] += err
                counts[ekey] += 1
            if r.concept in by_c_n:
                by_c_n[r.concept] += 1
                by_c_err[r.concept] += err
            if r.boundary_f is not None:
                bd_errs.append(_clip01(1.0 - float(r.boundary_f)))
            if r.confidence is not None:
                # |conf - iou| as crude calibration proxy
                cal_abs.append(abs(float(r.confidence) - float(r.iou)))

    for k in U_KEYS:
        if counts[k] > 0:
            raw[k] /= counts[k]

    if bd_errs:
        raw["E_bd"] = float(np.mean(bd_errs))
        counts["E_bd"] = len(bd_errs)
    if cal_abs:
        raw["E_cal"] = float(np.mean(cal_abs))
        counts["E_cal"] = len(cal_abs)

    # normalize to [0,1] already; keep raw copy
    u = {k: _clip01(raw[k]) for k in U_KEYS}
    for c in CONCEPT_KEYS:
        if by_c_n[c] > 0:
            by_c_err[c] /= by_c_n[c]

    return ErrorProfile(
        u=u,
        n_diag=len(recs),
        by_concept_n=by_c_n,
        by_concept_err=by_c_err,
        raw=raw,
        version=version,
        outer_cycle=outer_cycle,
    )


def dominant_error_dims(profile: ErrorProfile, topk: int = 2) -> list[str]:
    items = sorted(profile.u.items(), key=lambda kv: kv[1], reverse=True)
    return [k for k, _ in items[:topk]]
