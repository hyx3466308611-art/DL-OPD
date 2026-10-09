#!/usr/bin/env python3
"""CLI: score V(I,p) for candidates given an error profile."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from dl_opd.data_policy.data_value import CandidateFeatures, concept_sampling_probs, score_data_value
from dl_opd.data_policy.error_profile import ErrorProfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--candidates_jsonl", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    profile = ErrorProfile.load(args.profile)
    probs = concept_sampling_probs(profile)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.candidates_jsonl, "r", encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            r = json.loads(line)
            feat = CandidateFeatures(
                uncertainty=r.get("uncertainty"),
                disagreement=r.get("disagreement"),
                boundary_err=r.get("boundary_err"),
                novelty=r.get("novelty"),
                concept=r.get("concept", "entities"),
            )
            scored = score_data_value(feat, profile)
            scored["P_concept"] = probs.get(feat.concept, 0.0)
            scored.update({k: r[k] for k in r if k not in scored})
            fout.write(json.dumps(scored, ensure_ascii=False) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
