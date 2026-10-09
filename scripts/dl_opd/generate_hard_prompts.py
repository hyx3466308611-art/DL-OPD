#!/usr/bin/env python3
"""CLI: generate hard prompts WITHOUT exposing GT to the generator."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from dl_opd.data_policy.error_profile import ErrorProfile, dominant_error_dims
from dl_opd.data_policy.hard_ops import GeneratorInput, generate_candidates, select_operators


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parents_jsonl", required=True, help="image_id,prompt,concept only")
    ap.add_argument("--profile", type=str, default=None)
    ap.add_argument("--operators", type=str, default=None, help="comma list; default from profile")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_per_image", type=int, default=2)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if args.operators:
        ops = [x.strip() for x in args.operators.split(",") if x.strip()]
    elif args.profile:
        profile = ErrorProfile.load(args.profile)
        ops = select_operators(dominant_error_dims(profile, 3))
    else:
        ops = ["semantic_rewrite", "minimal_semantic_pair"]

    parents = []
    with open(args.parents_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            # strip forbidden keys
            for bad in ("gt", "mask", "answer", "gt_mask", "answer_text"):
                r.pop(bad, None)
            parents.append(
                GeneratorInput(
                    image_id=str(r.get("image_id", r.get("image", ""))),
                    image_path=str(r.get("image_path", "")),
                    original_prompt=str(r["prompt"]),
                    concept=str(r.get("concept", "entities")),
                    error_type=str(r.get("error_type", "unknown")),
                    meta={k: v for k, v in r.items() if k not in ("prompt",)},
                )
            )

    cands = generate_candidates(parents, ops, rng, max_per_image=args.max_per_image)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for c in cands:
            f.write(
                json.dumps(
                    {
                        "operator": c.operator,
                        "prompt": c.prompt,
                        "parent_prompt": c.parent_prompt,
                        "concept": c.concept,
                        "is_negative": c.is_negative,
                        "expected_label_policy": c.expected_label_policy,
                        "diff_field": c.diff_field,
                        "meta": c.meta,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"operators={ops} wrote {len(cands)} → {out}")


if __name__ == "__main__":
    main()
