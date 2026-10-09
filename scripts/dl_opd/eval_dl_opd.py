#!/usr/bin/env python3
"""Unified DL-OPD evaluator entry (plan §12).

Wraps:
  - CONVERSEG concept eval via eval.py
  - Recovery / Empty-Acc metrics via metrics_ops + optional MaskCorrupter
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dl_opd.metrics_ops import (
    boundary_f_score,
    empty_accuracy,
    false_positive_area,
    mask_iou,
    recovery_gain,
    summarize_recovery_batch,
)


def run_converseg_eval(args) -> None:
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "eval.py"),
        "--final_ckpt",
        args.final_ckpt,
        "--plm_ckpt",
        args.plm_ckpt,
        "--lora_ckpt",
        args.lora_ckpt,
        "--plm-model-name",
        args.plm_model,
        "--base_ckpt",
        args.base_ckpt,
        "--eval_splits",
        args.eval_splits,
        "--precision",
        args.precision,
        "--report-out",
        args.report_out,
    ]
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


def eval_negative_suite(jsonl: Path, pred_dir: Path, out_json: Path) -> dict:
    """Expect pred_dir/{id}.npy soft/hard masks; GT empty."""
    rows = []
    if jsonl.is_file():
        with jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    eas, fps = [], []
    for r in rows:
        pid = str(r.get("id"))
        pred_p = pred_dir / f"{pid}.npy"
        if not pred_p.is_file():
            continue
        pred = np.load(pred_p)
        ea = empty_accuracy(pred, True)
        fp = false_positive_area(pred, True)
        if ea is not None:
            eas.append(ea)
        if fp is not None:
            fps.append(fp)
    summary = {
        "n": len(eas),
        "empty_accuracy": float(np.mean(eas)) if eas else None,
        "false_positive_area": float(np.mean(fps)) if fps else None,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def eval_recovery_from_arrays(before_dir: Path, after_dir: Path, gt_dir: Path, ids: list[str]) -> dict:
    befores, afters, gts = [], [], []
    for i in ids:
        bp, ap, gp = before_dir / f"{i}.npy", after_dir / f"{i}.npy", gt_dir / f"{i}.npy"
        if not (bp.is_file() and ap.is_file() and gp.is_file()):
            continue
        befores.append(np.load(bp))
        afters.append(np.load(ap))
        gts.append(np.load(gp))
    return summarize_recovery_batch(befores, afters, gts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["converseg", "negative", "recovery_summary", "all_metrics_demo"], default="converseg")
    ap.add_argument("--final_ckpt", type=str, default="")
    ap.add_argument("--plm_ckpt", type=str, default="")
    ap.add_argument("--lora_ckpt", type=str, default="")
    ap.add_argument("--plm_model", type=str, default="student/Qwen2.5-VL-3B")
    ap.add_argument("--base_ckpt", type=str, default="student/sam2_hiera_large.pt")
    ap.add_argument("--eval_splits", type=str, default="sam_seeded,human_annotated")
    ap.add_argument("--precision", type=str, default="bf16")
    ap.add_argument("--report_out", type=str, default="logs/dl_opd/converseg_report.jsonl")
    ap.add_argument("--neg_jsonl", type=str, default="data/dl_opd/eval_suites/negative_prompt_set.jsonl")
    ap.add_argument("--pred_dir", type=str, default="")
    ap.add_argument("--out", type=str, default="logs/dl_opd/eval_summary.json")
    args = ap.parse_args()

    if args.mode == "converseg":
        run_converseg_eval(args)
    elif args.mode == "negative":
        s = eval_negative_suite(Path(args.neg_jsonl), Path(args.pred_dir), Path(args.out))
        print(s)
    elif args.mode == "all_metrics_demo":
        # synthetic sanity
        gt = np.zeros((64, 64), np.uint8)
        gt[16:48, 16:48] = 1
        pred = gt.copy()
        pred[20:30, 20:30] = 0
        print(
            {
                "iou": mask_iou(pred, gt),
                "boundary_f": boundary_f_score(pred, gt),
                "rg": recovery_gain(gt, pred, gt),
                "empty_acc": empty_accuracy(np.zeros_like(gt), True),
                "fp": false_positive_area(pred, True),
            }
        )
    else:
        raise SystemExit(f"mode={args.mode} needs array dirs; use recovery_summary helpers in code")


if __name__ == "__main__":
    main()
