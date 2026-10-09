#!/usr/bin/env python3
"""Merge per-op trg_report.json into one diagnosis table (stdout + summary JSON)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from data.mask_corrupter import PER_OP_DIAGNOSIS_OPS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help=".../eval_recovery_per_op")
    ap.add_argument("--out", default=None, help="Write summary JSON (default: root/per_op_summary.json)")
    args = ap.parse_args()
    root = Path(args.root)
    rows = []
    for op in PER_OP_DIAGNOSIS_OPS:
        p = root / op / "trg_report.json"
        if not p.is_file():
            rows.append({"op": op, "missing": True})
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        rows.append(
            {
                "op": op,
                "missing": False,
                "n": d.get("n"),
                "mean_trg": d.get("mean_trg"),
                "positive_fraction": d.get("positive_fraction"),
                "mean_input_iou": d.get("mean_input_iou"),
                "mean_output_iou": d.get("mean_output_iou"),
                "eval_stage": d.get("eval_stage"),
                "force_op": d.get("force_op"),
                "report": str(p),
            }
        )

    out = Path(args.out) if args.out else root / "per_op_summary.json"
    payload = {"root": str(root), "ops": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"{'op':<16} {'n':>6} {'TRG':>8} {'pos':>8} {'in':>8} {'out':>8}")
    print("-" * 60)
    for r in rows:
        if r.get("missing"):
            print(f"{r['op']:<16} MISSING")
            continue
        print(
            f"{r['op']:<16} {r['n']:>6} {r['mean_trg']:>8.4f} {r['positive_fraction']:>8.4f} "
            f"{r['mean_input_iou']:>8.4f} {r['mean_output_iou']:>8.4f}"
        )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
