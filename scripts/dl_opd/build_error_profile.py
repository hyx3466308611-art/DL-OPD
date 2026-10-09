#!/usr/bin/env python3
"""CLI: build error profile u from diagnostic JSONL (+ optional pred IoU fields)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from dl_opd.data_policy.error_profile import CONCEPT_KEYS, DiagnosticRecord, compute_error_profile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag_jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--outer_cycle", type=int, default=0)
    args = ap.parse_args()

    recs = []
    with open(args.diag_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            concept = r.get("concept", "entities")
            if concept not in CONCEPT_KEYS:
                concept = "entities"
            gt_empty = bool(r.get("gt_empty", r.get("is_negative", False)))
            recs.append(
                DiagnosticRecord(
                    sample_id=str(r.get("id", i)),
                    image_id=str(r.get("image", i)),
                    prompt=str(r.get("prompt", "")),
                    concept=concept,
                    is_negative=gt_empty,
                    iou=float(r.get("iou", 0.0)),
                    pred_empty=bool(r.get("pred_empty", False)),
                    gt_empty=gt_empty,
                    confidence=r.get("confidence"),
                    boundary_f=r.get("boundary_f"),
                )
            )
    profile = compute_error_profile(recs, outer_cycle=args.outer_cycle)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    profile.save(args.out)
    print(json.dumps(profile.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
