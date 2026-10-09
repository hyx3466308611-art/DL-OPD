"""Q_data + teacher cache version invalidation (§15.3 #5 #9)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dl_opd.data_policy.verify_qdata import verify_candidate
from dl_opd.teacher_cache import CacheEntry, TeacherCache


def test_qdata_empty_ignores_teacher_draw():
    q = verify_candidate(is_negative=True, label_policy="empty", teacher_conf=0.1, novelty=0.8)
    assert q.accept and q.c_teacher == 1.0


def test_qdata_duplicate_reject():
    q = verify_candidate(is_negative=False, label_policy="hard", duplicate=True)
    assert not q.accept and q.reject_reason == "duplicate"


def test_cache_version_mismatch():
    cache = TeacherCache(ttl_epochs=2)
    key = cache.make_key(
        image_id="i",
        prompt="p",
        state_source="current",
        student_ckpt="s1",
        teacher_ckpt="t1",
        state_mask=np.zeros((8, 8), np.float32),
    )
    cache.put(
        CacheEntry(
            key=key,
            teacher_mask=np.ones((8, 8), np.float32),
            teacher_confidence=0.9,
            created_epoch=0,
            created_cycle=0,
        )
    )
    key2 = cache.make_key(
        image_id="i",
        prompt="p",
        state_source="current",
        student_ckpt="s2",  # changed
        teacher_ckpt="t1",
        state_mask=np.zeros((8, 8), np.float32),
    )
    assert cache.get(key2, current_epoch=0) is None


if __name__ == "__main__":
    test_qdata_empty_ignores_teacher_draw()
    test_qdata_duplicate_reject()
    test_cache_version_mismatch()
    print("ok")
