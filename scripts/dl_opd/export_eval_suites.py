#!/usr/bin/env python3
"""Export Hard / Negative / Mask-Recovery eval suite manifests (plan §11)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def export_from_phase2(src: Path, out_dir: Path, seed: int = 0, n_neg: int = 200, n_hard: int = 200):
    rng = random.Random(seed)
    rows = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    rng.shuffle(rows)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Hard prompt set: keep concept diversity
    hard = rows[:n_hard]
    with (out_dir / "hard_prompt_set.jsonl").open("w", encoding="utf-8") as f:
        for r in hard:
            f.write(
                json.dumps(
                    {
                        "id": r.get("id"),
                        "image": r.get("image"),
                        "mask": r.get("mask_merged") or r.get("mask"),
                        "prompt": r.get("prompt"),
                        "concept": r.get("concept"),
                        "suite": "hard",
                        "operator": "seed_pool",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # Negative set: synthetic non-existence prompts on same images
    neg = rows[:n_neg]
    ghosts = [
        "a flying elephant that is not in the image",
        "an object with impossible attributes for this scene",
        "the missing item that does not appear here",
    ]
    with (out_dir / "negative_prompt_set.jsonl").open("w", encoding="utf-8") as f:
        for i, r in enumerate(neg):
            f.write(
                json.dumps(
                    {
                        "id": f"neg_{r.get('id', i)}",
                        "image": r.get("image"),
                        "mask": None,
                        "prompt": ghosts[i % len(ghosts)],
                        "concept": r.get("concept", "entities"),
                        "suite": "negative",
                        "gt_empty": True,
                        "parent_id": r.get("id"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # Recovery set: pointers + corruption op list (actual masks made at eval time)
    ops = [
        "morph",
        "boundary",
        "translate",
        "drop_comp",
        "false_pos",
        "elastic",
        "partial",
        "mask_dropout",
        "empty_gt",
    ]
    with (out_dir / "mask_recovery_set.jsonl").open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows[: max(n_hard, 500)]):
            f.write(
                json.dumps(
                    {
                        "id": r.get("id", i),
                        "image": r.get("image"),
                        "mask": r.get("mask_merged") or r.get("mask"),
                        "prompt": r.get("prompt"),
                        "concept": r.get("concept"),
                        "suite": "recovery",
                        "force_ops": ops,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    meta = {
        "source": str(src),
        "n_hard": len(hard),
        "n_neg": len(neg),
        "recovery_ops": ops,
        "note": "Negative GT is empty; Recovery corruptions applied at eval time via MaskCorrupter.",
    }
    (out_dir / "suite_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_jsonl", type=str, default=str(ROOT / "data/phase2/phase2_mixed.jsonl"))
    ap.add_argument("--out_dir", type=str, default=str(ROOT / "data/dl_opd/eval_suites"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    src = Path(args.src_jsonl)
    if not src.is_file():
        print(f"WARNING: {src} missing; writing empty placeholders")
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name in ("hard_prompt_set.jsonl", "negative_prompt_set.jsonl", "mask_recovery_set.jsonl"):
            (out / name).write_text("", encoding="utf-8")
        return
    export_from_phase2(src, Path(args.out_dir), seed=args.seed)


if __name__ == "__main__":
    main()
