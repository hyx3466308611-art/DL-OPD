"""Data value V(I,p) and concept sampling P(c) (plan §7.3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .error_profile import CONCEPT_KEYS, CONCEPT_TO_E, ErrorProfile


@dataclass
class ValueWeights:
    alpha: float = 1.0  # uncertainty U
    beta: float = 1.0  # student-teacher disagreement
    gamma: float = 1.0  # boundary error
    delta: float = 1.0  # novelty
    eta: float = 1.0  # concept weakness


@dataclass
class CandidateFeatures:
    uncertainty: float | None = None
    disagreement: float | None = None  # missing → None, not 0
    boundary_err: float | None = None
    novelty: float | None = None
    concept: str = "entities"


def _norm(x: float | None, default: float = 0.0) -> float:
    if x is None:
        return default
    return float(max(0.0, min(1.0, x)))


def score_data_value(
    feat: CandidateFeatures,
    profile: ErrorProfile,
    weights: ValueWeights | None = None,
    *,
    missing_disagree_default: float = 0.5,
) -> dict[str, Any]:
    """Return V and component breakdown. Missing D_ST uses explicit fallback."""
    w = weights or ValueWeights()
    U = _norm(feat.uncertainty)
    if feat.disagreement is None:
        D = missing_disagree_default
        d_missing = True
    else:
        D = _norm(feat.disagreement)
        d_missing = False
    Ebd = _norm(feat.boundary_err)
    Nov = _norm(feat.novelty, default=0.5)
    ekey = CONCEPT_TO_E.get(feat.concept, "E_ent")
    weak = float(profile.u.get(ekey, 0.0))
    V = w.alpha * U + w.beta * D + w.gamma * Ebd + w.delta * Nov + w.eta * weak
    return {
        "V": float(V),
        "U": U,
        "D_ST": D,
        "D_ST_missing": d_missing,
        "E_bd": Ebd,
        "Novelty": Nov,
        "ConceptWeakness": weak,
        "concept": feat.concept,
    }


def concept_sampling_probs(
    profile: ErrorProfile,
    *,
    counts: dict[str, int] | None = None,
    rho: float = 1.0,
    mu: float = 0.5,
    eps: float = 1e-3,
    min_prob: float = 0.05,
) -> dict[str, float]:
    """P(c) ∝ (E_c+eps)^rho * (N_c+eps)^(-mu), then enforce min_prob and renormalize."""
    counts = counts or {c: profile.by_concept_n.get(c, 0) for c in CONCEPT_KEYS}
    raw = {}
    for c in CONCEPT_KEYS:
        Ec = float(profile.by_concept_err.get(c, profile.u.get(CONCEPT_TO_E[c], 0.0)))
        Nc = float(counts.get(c, 0))
        raw[c] = ((Ec + eps) ** rho) * ((Nc + eps) ** (-mu))
    s = sum(raw.values()) or 1.0
    p = {c: raw[c] / s for c in CONCEPT_KEYS}
    # enforce floor
    for c in CONCEPT_KEYS:
        p[c] = max(min_prob, p[c])
    s2 = sum(p.values())
    return {c: p[c] / s2 for c in CONCEPT_KEYS}


def sampling_entropy(probs: dict[str, float]) -> float:
    vals = np.array(list(probs.values()), dtype=np.float64)
    vals = vals[vals > 0]
    return float(-(vals * np.log(vals + 1e-12)).sum())
