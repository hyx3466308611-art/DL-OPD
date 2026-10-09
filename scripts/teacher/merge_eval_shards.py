#!/usr/bin/env python3
"""Merge eval.py --metrics-out jsonl shards into global gIoU / cIoU (per split + by concept)."""

from __future__ import annotations

import json
import sys
from collections import defaultdict

CONCEPT_ORDER = ("entities", "spatial", "relations", "affordances", "physics")


def _bucket():
    return {"n": 0, "sum_iou": 0.0, "inter": 0.0, "union": 0.0}


def main():
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit("Usage: merge_eval_shards.py shard0.jsonl shard1.jsonl ...")

    by_split: dict[str, dict] = defaultdict(_bucket)
    by_split_concept: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(_bucket))

    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                s = str(r.get("split", "unknown"))
                by_split[s]["n"] += int(r["n"])
                by_split[s]["sum_iou"] += float(r["sum_iou"])
                by_split[s]["inter"] += float(r["total_inter"])
                by_split[s]["union"] += float(r["total_union"])
                for concept, cb in (r.get("by_concept") or {}).items():
                    b = by_split_concept[s][str(concept)]
                    b["n"] += int(cb["n"])
                    b["sum_iou"] += float(cb["sum_iou"])
                    b["inter"] += float(cb["total_inter"])
                    b["union"] += float(cb["total_union"])

    tot = _bucket()
    print("=== Merged ConVerSeg eval (shards) ===")
    for name, d in by_split.items():
        giou = d["sum_iou"] / max(d["n"], 1)
        ciou = d["inter"] / (d["union"] + 1e-10)
        print(f"{name}: n={d['n']} gIoU {giou:.3f} | cIoU {ciou:.3f}")
        cells = [f"{giou * 100:.1f}"]
        c_cells = [f"{ciou * 100:.1f}"]
        for concept in CONCEPT_ORDER:
            b = by_split_concept[name].get(concept)
            if not b or b["n"] <= 0:
                cells.append("—")
                c_cells.append("—")
                continue
            cgiou = b["sum_iou"] / max(b["n"], 1)
            cciou = b["inter"] / (b["union"] + 1e-10)
            cells.append(f"{cgiou * 100:.1f}")
            c_cells.append(f"{cciou * 100:.1f}")
            print(f"  {concept}: n={b['n']} gIoU {cgiou:.3f} | cIoU {cciou:.3f}")
        print(f"TABLE1[{name}] gIoU%: " + " ".join(cells))
        print(f"TABLE8[{name}] cIoU%: " + " ".join(c_cells))
        tot["n"] += d["n"]
        tot["sum_iou"] += d["sum_iou"]
        tot["inter"] += d["inter"]
        tot["union"] += d["union"]

    giou = tot["sum_iou"] / max(tot["n"], 1)
    ciou = tot["inter"] / (tot["union"] + 1e-10)
    print(f"OVERALL: n={tot['n']} gIoU {giou:.3f} | cIoU {ciou:.3f}")
    print(f"JSON: giou{giou:.3f}_ciou{ciou:.3f}")


if __name__ == "__main__":
    main()
