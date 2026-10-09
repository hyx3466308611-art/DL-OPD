#!/usr/bin/env python3
"""Stage2 / Group4 有效行统计（供 check_stage2.sh 调用）。"""
import json
import sys
from pathlib import Path

BD = Path(__file__).resolve().parent
sys.path.insert(0, str(BD))

from config import DATA_ROOT, HF_G4_NEG_JSONL, HF_G4_POS_JSONL  # noqa: E402
from hf_paths import rewrite_stage2  # noqa: E402


def count(p: Path) -> tuple[int, int]:
    total = valid = 0
    with p.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            img, msk = rewrite_stage2(item)
            if (DATA_ROOT / img).is_file() and (DATA_ROOT / msk).is_file():
                valid += 1
            total += 1
    return total, valid


def main() -> None:
    ok = True
    for name, p in [("positive", HF_G4_POS_JSONL), ("negative", HF_G4_NEG_JSONL)]:
        if not p.is_file():
            print("G4 %s: MISSING %s" % (name, p))
            ok = False
            continue
        t, v = count(p)
        print("G4 %s: jsonl=%d valid=%d" % (name, t, v))
        if v == 0:
            ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
