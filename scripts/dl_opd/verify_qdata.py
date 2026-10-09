#!/usr/bin/env python3
"""CLI: verify generated candidates → accept/reject with Q_data components."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from dl_opd.data_policy.verify_qdata import VerifyConfig, verify_candidate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates_jsonl", required=True)
    ap.add_argument("--out_accepted", required=True)
    ap.add_argument("--out_rejected", required=True)
    ap.add_argument("--q_min", type=float, default=0.35)
    args = ap.parse_args()
    cfg = VerifyConfig(q_min=args.q_min)

    Path(args.out_accepted).parent.mkdir(parents=True, exist_ok=True)
    n_acc = n_rej = 0
    with open(args.candidates_jsonl, "r", encoding="utf-8") as fin, open(
        args.out_accepted, "w", encoding="utf-8"
    ) as fa, open(args.out_rejected, "w", encoding="utf-8") as fr:
        for line in fin:
            r = json.loads(line)
            q = verify_candidate(
                is_negative=bool(r.get("is_negative", False)),
                label_policy=str(r.get("expected_label_policy", "hard")),
                novelty=r.get("novelty"),
                teacher_conf=r.get("teacher_conf"),
                teacher_recovery_gain=r.get("teacher_recovery_gain"),
                duplicate=bool(r.get("duplicate", False)),
                semantic_ok=r.get("semantic_ok"),
                geo_ok=r.get("geo_ok"),
                cfg=cfg,
            )
            row = {**r, "quality": q.to_dict()}
            if q.accept:
                fa.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_acc += 1
            else:
                fr.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rej += 1
    print(f"accepted={n_acc} rejected={n_rej}")


if __name__ == "__main__":
    main()
