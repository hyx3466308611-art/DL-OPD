"""Short-lived Hard Buffer with TTL, dedup, and sampling (plan §8.4)."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class HardSample:
    sample_id: str
    image_id: str
    image_path: str
    prompt: str
    parent_sample_id: str | None = None
    concept: str = "unknown"
    error_type: str = "unknown"
    operator: str = "unknown"
    is_negative: bool = False
    label_type: str = "hard"  # hard / soft / candidate / empty
    mask_path: str | None = None
    q_data: float = 0.0
    c_sem: float = 0.0
    c_geo: float = 0.0
    c_teacher: float = 0.0
    c_novel: float = 0.0
    value: float = 0.0
    outer_cycle_created: int = 0
    last_trained_cycle: int = -1
    meta: dict[str, Any] = field(default_factory=dict)

    def age(self, current_cycle: int) -> int:
        return max(0, current_cycle - self.outer_cycle_created)


class HardBuffer:
    def __init__(
        self,
        *,
        ttl_cycles: int = 2,
        capacity: int = 50000,
        dedup_key_fields: tuple[str, ...] = ("image_id", "prompt"),
    ):
        if ttl_cycles < 1:
            raise ValueError("ttl_cycles must be >= 1")
        self.ttl_cycles = ttl_cycles
        self.capacity = capacity
        self.dedup_key_fields = dedup_key_fields
        self._items: dict[str, HardSample] = {}
        self._reject_log: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._items)

    def _dedup_key(self, s: HardSample) -> str:
        parts = []
        for f in self.dedup_key_fields:
            parts.append(str(getattr(s, f, s.meta.get(f, ""))))
        return "||".join(parts)

    def add(self, sample: HardSample, *, replace_if_higher_q: bool = True) -> bool:
        key = self._dedup_key(sample)
        # also index by sample_id
        existing = None
        for it in self._items.values():
            if self._dedup_key(it) == key:
                existing = it
                break
        if existing is not None:
            if replace_if_higher_q and sample.q_data >= existing.q_data:
                del self._items[existing.sample_id]
                self._items[sample.sample_id] = sample
                return True
            return False
        if len(self._items) >= self.capacity:
            # drop oldest lowest value
            victim = min(self._items.values(), key=lambda x: (x.value, -x.outer_cycle_created))
            del self._items[victim.sample_id]
        self._items[sample.sample_id] = sample
        return True

    def reject(self, payload: dict[str, Any], reason: str) -> None:
        self._reject_log.append({"reason": reason, "time": time.time(), **payload})

    def refresh(self, current_cycle: int) -> dict[str, int]:
        """Evict samples older than TTL."""
        dead = [
            sid
            for sid, s in self._items.items()
            if s.age(current_cycle) > self.ttl_cycles
        ]
        for sid in dead:
            del self._items[sid]
        return {"evicted": len(dead), "remaining": len(self._items)}

    def sample(
        self,
        n: int,
        *,
        rng,
        concept_quotas: dict[str, float] | None = None,
    ) -> list[HardSample]:
        items = list(self._items.values())
        if not items or n <= 0:
            return []
        # weight by value * q_data
        weights = []
        for s in items:
            w = max(1e-6, float(s.value) * max(1e-3, float(s.q_data)))
            if concept_quotas and s.concept in concept_quotas:
                w *= float(concept_quotas[s.concept])
            weights.append(w)
        total = sum(weights)
        probs = [w / total for w in weights]
        # multinomial without replacement
        chosen: list[HardSample] = []
        pool = list(range(len(items)))
        p = list(probs)
        for _ in range(min(n, len(items))):
            idx = rng.choices(pool, weights=p, k=1)[0]
            chosen.append(items[idx])
            # remove
            j = pool.index(idx)
            pool.pop(j)
            p.pop(j)
            if not pool:
                break
            ssum = sum(p) or 1.0
            p = [x / ssum for x in p]
        return chosen

    def concept_histogram(self) -> dict[str, int]:
        from collections import Counter

        return dict(Counter(s.concept for s in self._items.values()))

    def age_stats(self, current_cycle: int) -> dict[str, float]:
        ages = [s.age(current_cycle) for s in self._items.values()]
        if not ages:
            return {"n": 0, "mean_age": 0.0, "max_age": 0.0}
        return {
            "n": len(ages),
            "mean_age": float(sum(ages) / len(ages)),
            "max_age": float(max(ages)),
        }

    def save_manifest(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [asdict(s) for s in self._items.values()]
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def load_manifest(self, path: str | Path) -> int:
        path = Path(path)
        if not path.is_file():
            return 0
        n = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                meta = d.pop("meta", {})
                s = HardSample(**d, meta=meta)
                self._items[s.sample_id] = s
                n += 1
        return n

    def save_rejects(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in self._reject_log:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def new_sample_id(prefix: str = "hard") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"
