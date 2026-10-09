#!/usr/bin/env python3
"""Print side-by-side TRG summary: baseline author3b vs fix_v1 (Stage B/C)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def load_report(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def summarize(d: dict) -> dict:
    rows = d.get("rows") or []
    if not rows:
        return {
            "n": 0,
            "mean_trg": d.get("mean_trg"),
            "pos": d.get("positive_fraction"),
            "mean_in": None,
            "mean_out": None,
        }
    return {
        "n": len(rows),
        "mean_trg": float(np.mean([r["trg"] for r in rows])),
        "pos": float(np.mean([r["trg"] > 0 for r in rows])),
        "mean_in": float(np.mean([r["input_iou"] for r in rows])),
        "mean_out": float(np.mean([r["output_iou"] for r in rows])),
        "eval_stage": d.get("eval_stage"),
        "full_gt": d.get("full_gt"),
    }


def fmt(s: dict) -> str:
    if not s or s.get("n") == 0:
        return "  (missing)"
    return (
        f"  n={s['n']} mean_trg={s['mean_trg']:.4f} pos={s['pos']:.4f} "
        f"in={s['mean_in']:.4f} out={s['mean_out']:.4f} "
        f"stage={s.get('eval_stage')} full_gt={s.get('full_gt')}"
    )


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    pairs = [
        (
            "Stage B",
            root / "ckpts/teacher_correction/author3b/eval_recovery/trg_report.json",
            root / "ckpts/teacher_correction/author3b_fix_v1/eval_recovery_stageB/trg_report.json",
        ),
        (
            "Stage C",
            root / "ckpts/teacher_correction/author3b/eval_recovery (1)/trg_report.json",
            root / "ckpts/teacher_correction/author3b_fix_v1/eval_recovery_stageC/trg_report.json",
        ),
    ]
    # fallback if baseline C without (1)
    alt_c = root / "ckpts/teacher_correction/author3b/eval_recovery/trg_report.json"

    print("=== Correction Teacher TRG: baseline author3b vs fix_v1 ===\n")
    for name, base_p, fix_p in pairs:
        if name == "Stage C" and not base_p.is_file():
            # try without (1) only if stage field is C — user may have separate copy
            pass
        print(f"## {name}")
        if base_p.is_file():
            print("baseline:", fmt(summarize(load_report(base_p))))
            print(f"  path: {base_p}")
        else:
            print(f"baseline: MISSING {base_p}")
        if fix_p.is_file():
            print("fix_v1:  ", fmt(summarize(load_report(fix_p))))
            print(f"  path: {fix_p}")
            if base_p.is_file():
                b, f = summarize(load_report(base_p)), summarize(load_report(fix_p))
                if b["mean_trg"] is not None and f["mean_trg"] is not None:
                    print(
                        f"  delta mean_trg={f['mean_trg'] - b['mean_trg']:+.4f} "
                        f"pos={f['pos'] - b['pos']:+.4f} "
                        f"out={f['mean_out'] - b['mean_out']:+.4f}"
                    )
        else:
            print(f"fix_v1:   MISSING {fix_p}")
        print()


if __name__ == "__main__":
    main()
