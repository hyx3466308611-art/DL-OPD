"""
Mixed-TRG / TRG report aggregation for paper-facing headlines + appendix splits.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def op_family(op: str) -> str:
    """Primary operator before '+' (e.g. boundary+morph → boundary)."""
    if not op:
        return "unknown"
    return op.split("+", 1)[0]


def input_iou_bin(input_iou: float) -> str:
    """Difficulty proxy bins for appendix tables."""
    if input_iou > 0.8:
        return "high_gt_0.8"
    if input_iou >= 0.4:
        return "mid_0.4_0.8"
    return "low_lt_0.4"


def _group_stats(rows: list[dict], key_fn) -> dict[str, dict[str, float | int]]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[str(key_fn(r))].append(r)
    out: dict[str, dict[str, float | int]] = {}
    for k, rs in sorted(buckets.items(), key=lambda x: -len(x[1])):
        trgs = [float(r["trg"]) for r in rs]
        out[k] = {
            "n": len(rs),
            "mean_trg": float(sum(trgs) / len(trgs)),
            "positive_fraction": float(sum(1 for t in trgs if t > 0) / len(trgs)),
            "mean_input_iou": float(sum(float(r["input_iou"]) for r in rs) / len(rs)),
            "mean_output_iou": float(sum(float(r["output_iou"]) for r in rs) / len(rs)),
        }
    return out


def summarize_trg_rows(rows: list[dict]) -> dict[str, Any]:
    """Headline metrics + appendix breakdowns from per-sample rows."""
    if not rows:
        raise ValueError("empty TRG rows")
    n = len(rows)
    trgs = [float(r["trg"]) for r in rows]
    in_ious = [float(r["input_iou"]) for r in rows]
    out_ious = [float(r["output_iou"]) for r in rows]
    mean_trg = float(sum(trgs) / n)
    pos_frac = float(sum(1 for t in trgs if t > 0) / n)
    mean_in = float(sum(in_ious) / n)
    mean_out = float(sum(out_ious) / n)
    return {
        "n": n,
        "mean_trg": mean_trg,
        "positive_fraction": pos_frac,
        "mean_input_iou": mean_in,
        "mean_output_iou": mean_out,
        "n_trg_negative": int(sum(1 for t in trgs if t < 0)),
        "n_trg_ge_0_3": int(sum(1 for t in trgs if t >= 0.3)),
        "by_split": _group_stats(rows, lambda r: r.get("split", "?")),
        "by_stage": _group_stats(rows, lambda r: r.get("stage", "?")),
        "by_input_iou_bin": _group_stats(rows, lambda r: input_iou_bin(float(r["input_iou"]))),
        "by_op_family": _group_stats(rows, lambda r: op_family(str(r.get("op", "")))),
    }


def print_trg_headline(summary: dict[str, Any], *, eval_stage: str) -> None:
    print("\n=== Mixed-TRG / Teacher Recovery Gain ===" if eval_stage == "MIX" else "\n=== Teacher Recovery Gain (TRG) ===")
    print(f"eval_stage:      {eval_stage}")
    print(f"Samples:         {summary['n']}")
    print(f"Mean Input IoU:  {summary['mean_input_iou']:.4f}")
    print(f"Mean Output IoU: {summary['mean_output_iou']:.4f}")
    print(f"Mean TRG:        {summary['mean_trg']:.4f}")
    print(f"TRG > 0 fraction:{summary['positive_fraction']:.4f}")
    print(f"TRG < 0 / ≥0.3:  {summary['n_trg_negative']} / {summary['n_trg_ge_0_3']}")

    if summary.get("by_stage") and (eval_stage == "MIX" or len(summary["by_stage"]) > 1):
        print("\nBy sampled stage (appendix):")
        for st, s in summary["by_stage"].items():
            print(
                f"  {st}: n={s['n']} TRG={s['mean_trg']:.4f} "
                f"pos={s['positive_fraction']:.3f} out={s['mean_output_iou']:.4f}"
            )

    print("\nBy input IoU bin (appendix):")
    for b, s in summary["by_input_iou_bin"].items():
        print(
            f"  {b}: n={s['n']} TRG={s['mean_trg']:.4f} "
            f"pos={s['positive_fraction']:.3f} out={s['mean_output_iou']:.4f}"
        )

    print("\nBy split:")
    for sp, s in summary["by_split"].items():
        print(f"  {sp}: mean TRG {s['mean_trg']:.4f} (n={s['n']})")

    print("\nBy op family (top 12, appendix):")
    for i, (fam, s) in enumerate(summary["by_op_family"].items()):
        if i >= 12:
            break
        print(f"  {fam}: mean TRG {s['mean_trg']:.4f} (n={s['n']})")
