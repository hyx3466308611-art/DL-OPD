"""Near-online teacher query cache (plan §9.3–9.4).

Cache keys MUST include image/prompt id, student state source, student ckpt version,
teacher version, and state parameters. Do not reuse across mismatched versions.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _hash_mask(mask: np.ndarray, ndigits: int = 4) -> str:
    m = (np.clip(mask.astype(np.float32), 0, 1) * (10**ndigits)).astype(np.int32)
    return hashlib.sha1(m.tobytes()).hexdigest()[:16]


@dataclass
class CacheKey:
    image_id: str
    prompt: str
    state_source: str  # current / random / historical / empty
    student_ckpt: str
    teacher_ckpt: str
    state_hash: str
    extra: str = ""

    def fingerprint(self) -> str:
        payload = "|".join(
            [
                self.image_id,
                self.prompt,
                self.state_source,
                self.student_ckpt,
                self.teacher_ckpt,
                self.state_hash,
                self.extra,
            ]
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()


@dataclass
class CacheEntry:
    key: CacheKey
    teacher_mask: np.ndarray
    teacher_confidence: float
    created_epoch: int
    created_cycle: int
    created_time: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)


class TeacherCache:
    def __init__(self, *, ttl_epochs: int = 2, capacity: int = 100000):
        self.ttl_epochs = ttl_epochs
        self.capacity = capacity
        self._store: dict[str, CacheEntry] = {}
        self.hits = 0
        self.misses = 0
        self.invalidations = 0

    def make_key(
        self,
        *,
        image_id: str,
        prompt: str,
        state_source: str,
        student_ckpt: str,
        teacher_ckpt: str,
        state_mask: np.ndarray,
        extra: str = "",
    ) -> CacheKey:
        return CacheKey(
            image_id=image_id,
            prompt=prompt,
            state_source=state_source,
            student_ckpt=student_ckpt,
            teacher_ckpt=teacher_ckpt,
            state_hash=_hash_mask(state_mask),
            extra=extra,
        )

    def get(self, key: CacheKey, *, current_epoch: int) -> CacheEntry | None:
        fp = key.fingerprint()
        ent = self._store.get(fp)
        if ent is None:
            self.misses += 1
            return None
        if current_epoch - ent.created_epoch > self.ttl_epochs:
            del self._store[fp]
            self.invalidations += 1
            self.misses += 1
            return None
        # version mismatch hard check
        if (
            ent.key.student_ckpt != key.student_ckpt
            or ent.key.teacher_ckpt != key.teacher_ckpt
        ):
            del self._store[fp]
            self.invalidations += 1
            self.misses += 1
            return None
        self.hits += 1
        return ent

    def put(self, entry: CacheEntry) -> None:
        if len(self._store) >= self.capacity:
            # drop oldest
            oldest = min(self._store.values(), key=lambda e: e.created_time)
            del self._store[oldest.key.fingerprint()]
        self._store[entry.key.fingerprint()] = entry

    def invalidate_student(self, student_ckpt: str) -> int:
        dead = [k for k, e in self._store.items() if e.key.student_ckpt != student_ckpt]
        # actually: invalidate entries that don't match NEW student — caller passes new ckpt
        dead = [k for k, e in self._store.items() if e.key.student_ckpt != student_ckpt]
        for k in dead:
            del self._store[k]
        self.invalidations += len(dead)
        return len(dead)

    def clear(self) -> None:
        self._store.clear()

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "size": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / max(1, total),
            "invalidations": self.invalidations,
            "ttl_epochs": self.ttl_epochs,
        }

    def save_index(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for e in self._store.values():
            rows.append(
                {
                    "fp": e.key.fingerprint(),
                    "image_id": e.key.image_id,
                    "prompt": e.key.prompt,
                    "student_ckpt": e.key.student_ckpt,
                    "teacher_ckpt": e.key.teacher_ckpt,
                    "state_source": e.key.state_source,
                    "created_epoch": e.created_epoch,
                    "created_cycle": e.created_cycle,
                    "teacher_confidence": e.teacher_confidence,
                }
            )
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
            encoding="utf-8",
        )
