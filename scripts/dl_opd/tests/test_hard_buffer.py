"""Hard Buffer TTL + dedup (§15.3 #6)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dl_opd.hard_buffer import HardBuffer, HardSample, new_sample_id


def test_ttl_eviction():
    buf = HardBuffer(ttl_cycles=2)
    s = HardSample(
        sample_id=new_sample_id(),
        image_id="i1",
        image_path="",
        prompt="p",
        outer_cycle_created=1,
        q_data=0.9,
        value=1.0,
    )
    buf.add(s)
    assert len(buf) == 1
    # age at cycle 4 with ttl=2 → age=3 > 2
    ev = buf.refresh(4)
    assert ev["evicted"] == 1 and len(buf) == 0


def test_dedup_keeps_higher_q():
    buf = HardBuffer()
    a = HardSample(new_sample_id(), "img", "", "hello", q_data=0.2, value=1.0)
    b = HardSample(new_sample_id(), "img", "", "hello", q_data=0.8, value=1.0)
    assert buf.add(a)
    assert buf.add(b)
    assert len(buf) == 1
    only = list(buf._items.values())[0]
    assert only.q_data == 0.8


def test_sample_runs():
    buf = HardBuffer()
    for i in range(5):
        buf.add(
            HardSample(
                new_sample_id(),
                f"img{i}",
                "",
                f"p{i}",
                concept="entities",
                q_data=0.5,
                value=0.5 + 0.1 * i,
                outer_cycle_created=0,
            )
        )
    got = buf.sample(3, rng=random.Random(0))
    assert len(got) == 3


if __name__ == "__main__":
    test_ttl_eviction()
    test_dedup_keeps_higher_q()
    test_sample_runs()
    print("ok")
