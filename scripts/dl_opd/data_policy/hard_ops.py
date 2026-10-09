"""Five hard-prompt generators (plan §7.4). Generator NEVER sees full GT / answer text."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

OPERATOR_NAMES = (
    "semantic_rewrite",
    "minimal_semantic_pair",
    "combinatorial_constraint",
    "affordance_physics_cf",
    "hallucination_targeted",
)

# Map dominant E_* → preferred operators
E_TO_OPS = {
    "E_ent": ("minimal_semantic_pair", "semantic_rewrite"),
    "E_spa": ("minimal_semantic_pair", "combinatorial_constraint"),
    "E_rel": ("minimal_semantic_pair", "combinatorial_constraint"),
    "E_aff": ("affordance_physics_cf", "combinatorial_constraint"),
    "E_phy": ("affordance_physics_cf", "combinatorial_constraint"),
    "E_neg": ("hallucination_targeted",),
    "E_bd": ("semantic_rewrite",),
    "E_cal": ("semantic_rewrite", "minimal_semantic_pair"),
}


@dataclass
class GeneratorInput:
    """Answer-isolated inputs only."""

    image_id: str
    image_path: str
    original_prompt: str
    concept: str
    error_type: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedCandidate:
    operator: str
    prompt: str
    parent_prompt: str
    concept: str
    is_negative: bool
    expected_label_policy: str  # reuse_mask / relabel / empty / candidate_set
    diff_field: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def select_operators(dominant_e: list[str], max_ops: int = 2) -> list[str]:
    out: list[str] = []
    for e in dominant_e:
        for op in E_TO_OPS.get(e, ("semantic_rewrite",)):
            if op not in out:
                out.append(op)
            if len(out) >= max_ops:
                return out
    return out or ["semantic_rewrite"]


def _rewrite_prompt(p: str, rng) -> str:
    templates = [
        "Please segment: {p}",
        "Mask the region described by: {p}",
        "Find and outline: {p}",
        "{p} (pixel-level)",
    ]
    return rng.choice(templates).format(p=p.strip())


_SPATIAL_FLIP = [
    (r"\bleft\b", "right"),
    (r"\bright\b", "left"),
    (r"\btop\b", "bottom"),
    (r"\bbottom\b", "top"),
    (r"\bin front of\b", "behind"),
    (r"\bbehind\b", "in front of"),
]


def _minimal_pair(p: str, rng) -> tuple[str, str, bool]:
    """Return (new_prompt, diff_field, is_negative_likely)."""
    low = p.lower()
    for pat, repl in _SPATIAL_FLIP:
        if re.search(pat, low):
            return re.sub(pat, repl, p, flags=re.I), "spatial_relation", False
    # attribute flip heuristics
    attr_pairs = [
        ("red", "blue"),
        ("large", "small"),
        ("open", "closed"),
        ("empty", "full"),
    ]
    for a, b in attr_pairs:
        if a in low:
            return re.sub(a, b, p, flags=re.I), "attribute", False
        if b in low:
            return re.sub(b, a, p, flags=re.I), "attribute", False
    # fallback: add a false attribute → often negative
    return f"{p} that does not exist in the image", "existence", True


def _combinatorial(p: str, rng) -> str:
    extras = [
        "on the left side",
        "that is safe to touch",
        "supporting another object",
        "closest to the camera",
    ]
    return f"{p.rstrip('. ')}, {rng.choice(extras)}"


def _affordance_cf(p: str, rng) -> str:
    templates = [
        "Where can one safely place a hot pan related to: {p}",
        "What could be removed without collapsing the structure near: {p}",
        "Which region affords holding, given: {p}",
    ]
    return rng.choice(templates).format(p=p)


def _hallucination(p: str, rng) -> str:
    ghosts = [
        "a flying elephant",
        "an invisible unicorn",
        "a car that is not present",
        "the object mentioned but missing from the scene",
    ]
    return f"Segment {rng.choice(ghosts)} (ignore: {p[:80]})"


def generate_one(inp: GeneratorInput, operator: str, rng) -> GeneratedCandidate:
    # CRITICAL: do not read GT / answer from inp.meta
    if "gt" in inp.meta or "mask" in inp.meta or "answer" in inp.meta:
        # strip forbidden keys if accidentally present
        safe_meta = {
            k: v
            for k, v in inp.meta.items()
            if k not in ("gt", "mask", "answer", "gt_mask", "answer_text")
        }
    else:
        safe_meta = dict(inp.meta)

    p0 = inp.original_prompt
    if operator == "semantic_rewrite":
        return GeneratedCandidate(
            operator=operator,
            prompt=_rewrite_prompt(p0, rng),
            parent_prompt=p0,
            concept=inp.concept,
            is_negative=False,
            expected_label_policy="reuse_mask",
            meta={"generator_input_keys": sorted(safe_meta.keys())},
        )
    if operator == "minimal_semantic_pair":
        nprompt, diff, is_neg = _minimal_pair(p0, rng)
        return GeneratedCandidate(
            operator=operator,
            prompt=nprompt,
            parent_prompt=p0,
            concept=inp.concept,
            is_negative=is_neg,
            expected_label_policy="empty" if is_neg else "relabel",
            diff_field=diff,
            meta={"generator_input_keys": sorted(safe_meta.keys())},
        )
    if operator == "combinatorial_constraint":
        return GeneratedCandidate(
            operator=operator,
            prompt=_combinatorial(p0, rng),
            parent_prompt=p0,
            concept=inp.concept,
            is_negative=False,
            expected_label_policy="candidate_set",
            meta={"generator_input_keys": sorted(safe_meta.keys())},
        )
    if operator == "affordance_physics_cf":
        return GeneratedCandidate(
            operator=operator,
            prompt=_affordance_cf(p0, rng),
            parent_prompt=p0,
            concept=inp.concept,
            is_negative=False,
            expected_label_policy="candidate_set",
            meta={"generator_input_keys": sorted(safe_meta.keys())},
        )
    if operator == "hallucination_targeted":
        return GeneratedCandidate(
            operator=operator,
            prompt=_hallucination(p0, rng),
            parent_prompt=p0,
            concept=inp.concept,
            is_negative=True,
            expected_label_policy="empty",
            meta={"generator_input_keys": sorted(safe_meta.keys())},
        )
    raise ValueError(f"Unknown operator: {operator}")


def generate_candidates(
    inputs: list[GeneratorInput],
    operators: list[str],
    rng,
    *,
    max_per_image: int = 2,
) -> list[GeneratedCandidate]:
    out: list[GeneratedCandidate] = []
    for inp in inputs:
        n = 0
        for op in operators:
            if n >= max_per_image:
                break
            out.append(generate_one(inp, op, rng))
            n += 1
    return out
