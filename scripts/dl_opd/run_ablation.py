#!/usr/bin/env python3
"""Experiment matrix launcher (plan §13). Writes run cards; optionally submits sbatch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ABLATIONS = [
    {"name": "stage2_baseline", "outer": "fixed", "inner": "none", "hypothesis": "baseline"},
    {"name": "random_augment", "outer": "random", "inner": "none", "hypothesis": "H2-control"},
    {"name": "error_data", "outer": "error", "inner": "none", "hypothesis": "H2/I2"},
    {"name": "offpolicy_kd", "outer": "fixed", "inner": "fixed_teacher", "hypothesis": "KD-baseline"},
    {"name": "random_state_kd", "outer": "fixed", "inner": "random_state", "hypothesis": "H1-control"},
    {"name": "historical_state_kd", "outer": "fixed", "inner": "historical_state", "hypothesis": "H1-control"},
    {"name": "mask_opd", "outer": "fixed", "inner": "current_state", "hypothesis": "H1/I1"},
    {"name": "dl_opd", "outer": "error", "inner": "current_state", "hypothesis": "H3"},
    {"name": "collapse_on", "outer": "error", "inner": "current_state+collapse", "hypothesis": "H4/I4"},
    {"name": "gate_off", "outer": "fixed", "inner": "current_state_gate_off", "hypothesis": "gate-ablation"},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default=str(ROOT / "logs/dl_opd/ablation_cards"))
    ap.add_argument("--only", type=str, default=None, help="comma names to keep")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    keep = set(args.only.split(",")) if args.only else None
    for ab in ABLATIONS:
        if keep and ab["name"] not in keep:
            continue
        card = {
            **ab,
            "config": "configs/dl_opd/dl_opd.yaml",
            "train_cmd_policy": f"python -u scripts/dl_opd/train_dl_opd.py --config configs/dl_opd/dl_opd.yaml --policy_only",
            "train_cmd_inner": f"python -u scripts/dl_opd/train_mask_opd.py --config configs/dl_opd/dl_opd.yaml",
            "notes": "Fill state_source / gate.enabled overrides per name before launch.",
        }
        p = out / f"{ab['name']}.json"
        p.write_text(json.dumps(card, indent=2), encoding="utf-8")
        print("wrote", p)


if __name__ == "__main__":
    main()
