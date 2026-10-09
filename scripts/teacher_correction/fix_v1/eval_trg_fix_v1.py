#!/usr/bin/env python3
"""TRG eval for fix_v1 (TeacherCorrectionHeadV2).

Protocols:
  1) Mixed / A/B/C — paper / appendix (no --force-op)
  2) Per-op diagnosis — --force-op <name>|ALL with --eval-stage C (or B)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

FIX_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIX_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from converseg_paths import apply_offline_env, default_sam_config_str, prepare_python_import_path, resolve_eval_data_dir

prepare_python_import_path()

from data.mask_corrupter import (
    FORCEABLE_OPS,
    PER_OP_DIAGNOSIS_OPS,
    CurriculumStage,
    MaskCorrupter,
    mask_iou,
    sample_mix_stage,
)
from eval_teacher_recovery import load_eval_items, soft_mask_tensor
from models.language_adapter import LanguageAdapter
from models.teacher_correction_model import TeacherCorrectionHeadV2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils.trg_eval_report import print_trg_headline, summarize_trg_rows
from utils.util_functions import bin_mask, erode, read_img_mask


def infer_one_v2(predictor, head, image_pe, img, gt_u8, corrupt_u8, prompt, img_path, dec_dev, dec_dtype):
    h_img, w_img = img.shape[:2]
    predictor.set_image(img)
    img_emb = predictor._features["image_embed"][-1].unsqueeze(0).to(dec_dev, dec_dtype)
    hi = [lvl[-1].unsqueeze(0).to(dec_dev, dec_dtype) for lvl in predictor._features["high_res_feats"]]
    _, _, Hm, Wm = img_emb.shape
    mask_t = soft_mask_tensor(corrupt_u8, h_img, w_img, dec_dev, dec_dtype)
    suppress = torch.ones(1, device=dec_dev, dtype=dec_dtype)
    sp, dp = head([prompt or "segment"], Hm, Wm, [img_path], mask_t, suppress_mask_branch=suppress)
    low, *_ = predictor.model.sam_mask_decoder(
        image_embeddings=img_emb,
        image_pe=image_pe,
        sparse_prompt_embeddings=sp.to(dec_dev, dec_dtype),
        dense_prompt_embeddings=dp.to(dec_dev, dec_dtype),
        multimask_output=False,
        repeat_image=False,
        high_res_features=hi,
    )
    logits = low[0, 0]
    pred = (torch.sigmoid(logits) > 0.5).float().cpu().numpy().astype(np.uint8)
    h, w = gt_u8.shape
    pred_r = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    input_iou = mask_iou((corrupt_u8 > 0.5).astype(np.uint8), gt_u8)
    output_iou = mask_iou(pred_r, gt_u8)
    return input_iou, output_iou, output_iou - input_iou


def run_one_force_op(
    *,
    force_op,
    args,
    data_root,
    predictor,
    head,
    image_pe,
    dec_dev,
    dec_dtype,
    out_dir,
    flag_path,
):
    if force_op and args.eval_stage == "MIX":
        raise SystemExit("ERROR: --force-op 不可与 MIX 同用；请用 --eval-stage C")

    fixed_stage = None if args.eval_stage == "MIX" else CurriculumStage(args.eval_stage)
    mix_rng = random.Random(42) if args.eval_stage == "MIX" else None
    # 每算子独立 RNG，保证同一 force_op 可复现，且 ALL 循环互不串扰
    corrupter = MaskCorrupter(seed=42)
    if args.eval_stage == "MIX":
        print("Mixed-TRG: A:B:C=1:2:2 full_gt=", args.full_gt)
    if force_op:
        print(f"Per-op diagnosis: force_op={force_op} stage={args.eval_stage} full_gt={args.full_gt}")

    all_rows = []
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        items, jsonl_path = load_eval_items(data_root, split, args.max_items)
        print(f"Eval {split}: {len(items)} from {jsonl_path} full_gt={args.full_gt} force_op={force_op}")
        for item in tqdm(items, desc=f"{split}/{force_op or 'mix'}"):
            img_rel = item.get("image") or item.get("image_path")
            msk_rel = item.get("mask") or item.get("mask_merged")
            prompt = item.get("prompt", "segment")
            img_p = img_rel if os.path.isabs(img_rel) else os.path.join(data_root, img_rel)
            msk_p = msk_rel if os.path.isabs(msk_rel) else os.path.join(data_root, msk_rel)
            if not os.path.isfile(img_p) or not os.path.isfile(msk_p):
                continue
            img, ann = read_img_mask(img_p, msk_p, args.max_dim)
            h_img, w_img = img.shape[:2]
            if args.full_gt:
                gt = bin_mask(ann).astype(np.uint8)
            else:
                gt = erode(bin_mask(ann), args.er_k, args.er_it).astype(np.uint8)
            if gt.shape[0] != h_img or gt.shape[1] != w_img:
                gt = cv2.resize(gt, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
            stage = sample_mix_stage(mix_rng) if mix_rng is not None else fixed_stage
            corrupt, meta = corrupter.corrupt(gt, stage, force_op=force_op)
            if meta.get("skip"):
                continue
            corrupt_aligned = corrupt.astype(np.float32)
            if corrupt_aligned.shape[0] != h_img or corrupt_aligned.shape[1] != w_img:
                corrupt_aligned = cv2.resize(corrupt_aligned, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
            inp_iou, out_iou, trg = infer_one_v2(
                predictor, head, image_pe, img, gt, corrupt_aligned, prompt, img_p, dec_dev, dec_dtype
            )
            all_rows.append(
                {
                    "id": item.get("id"),
                    "split": split,
                    "op": meta["op"],
                    "stage": meta["stage"],
                    "force_op": force_op,
                    "input_iou": inp_iou,
                    "output_iou": out_iou,
                    "trg": trg,
                }
            )

    if not all_rows:
        raise RuntimeError(f"No evaluation rows collected (force_op={force_op}).")

    summary = summarize_trg_rows(all_rows)
    mean_trg = summary["mean_trg"]
    pos_frac = summary["positive_fraction"]
    print_trg_headline(summary, eval_stage=args.eval_stage if not force_op else f"{args.eval_stage}+{force_op}")

    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "trg_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mean_trg": mean_trg,
                "positive_fraction": pos_frac,
                "mean_input_iou": summary["mean_input_iou"],
                "mean_output_iou": summary["mean_output_iou"],
                "n": summary["n"],
                "n_trg_negative": summary["n_trg_negative"],
                "n_trg_ge_0_3": summary["n_trg_ge_0_3"],
                "threshold_mean": args.trg_mean_threshold,
                "threshold_pos_frac": args.trg_positive_frac,
                "eval_stage": args.eval_stage,
                "force_op": force_op,
                "mix_weights": "A:B:C=1:2:2" if args.eval_stage == "MIX" else None,
                "full_gt": args.full_gt,
                "head_version": 2,
                "by_split": summary["by_split"],
                "by_stage": summary["by_stage"],
                "by_input_iou_bin": summary["by_input_iou_bin"],
                "by_op_family": summary["by_op_family"],
                "rows": all_rows,
            },
            f,
            indent=2,
        )
    ready = mean_trg > args.trg_mean_threshold and pos_frac >= args.trg_positive_frac
    if force_op:
        ready = False
    flag_body = {
        "ready": ready,
        "diagnosis": bool(force_op),
        "mean_trg": mean_trg,
        "positive_fraction": pos_frac,
        "mean_input_iou": summary["mean_input_iou"],
        "mean_output_iou": summary["mean_output_iou"],
        "eval_stage": args.eval_stage,
        "force_op": force_op,
        "mask_gate": head.gate_value,
        "head_checkpoint": args.resume_head_path,
        "report": report_path,
        "fix_v1": True,
    }
    os.makedirs(os.path.dirname(os.path.abspath(flag_path)), exist_ok=True)
    with open(flag_path, "w", encoding="utf-8") as f:
        json.dump(flag_body, f, indent=2)
    print(f"mean TRG {mean_trg:.4f} pos_frac {pos_frac:.4f} out={summary['mean_output_iou']:.4f} ready={ready}")
    print(f"Wrote {report_path}")
    return {
        "op": force_op or "MIX_OR_STAGE",
        "n": summary["n"],
        "mean_trg": mean_trg,
        "positive_fraction": pos_frac,
        "mean_input_iou": summary["mean_input_iou"],
        "mean_output_iou": summary["mean_output_iou"],
        "report": report_path,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--splits", default="sam_seeded,human_annotated")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument(
        "--eval-stage",
        choices=("A", "B", "C", "MIX"),
        default="MIX",
        help="A/B/C=固定难度；MIX=主协议 A:B:C=1:2:2（seed=42）",
    )
    ap.add_argument(
        "--force-op",
        default=None,
        help="诊断：算子名，或 ALL（一次加载扫 PER_OP_DIAGNOSIS_OPS）。不可与 MIX 同用。",
    )
    ap.add_argument("--model-cfg", default=default_sam_config_str())
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--plm-model-name", required=True)
    ap.add_argument("--resume-sam-path", required=True)
    ap.add_argument("--resume-plm-path", required=True)
    ap.add_argument("--resume-lora-path", default=None)
    ap.add_argument("--resume-head-path", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-dim", type=int, default=1024)
    ap.add_argument("--er-k", type=int, default=5)
    ap.add_argument("--er-it", type=int, default=1)
    ap.add_argument("--full-gt", action="store_true", help="Match fix_v1 train: no erode on GT")
    ap.add_argument("--trg-mean-threshold", type=float, default=0.03)
    ap.add_argument("--trg-positive-frac", type=float, default=0.55)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--flag-path", required=True)
    args = ap.parse_args()

    if args.force_op is not None and args.force_op != "ALL" and args.force_op not in FORCEABLE_OPS:
        raise SystemExit(f"ERROR: unknown --force-op={args.force_op}; use ALL or {sorted(FORCEABLE_OPS)}")

    apply_offline_env()
    data_root = args.data_root or str(resolve_eval_data_dir())
    os.makedirs(args.out_dir, exist_ok=True)

    predictor = SAM2ImagePredictor(build_sam2(args.model_cfg, args.checkpoint, device=args.device))
    C = predictor.model.sam_mask_decoder.transformer_dim
    plm = LanguageAdapter(
        model_name=args.plm_model_name,
        transformer_dim=C,
        use_dense_bias=True,
        use_lora=True,
        dtype=torch.bfloat16,
        device=args.device,
        use_image_input=True,
    ).to(args.device)
    head = TeacherCorrectionHeadV2(plm, transformer_dim=C).to(args.device)

    for path, load_fn in [
        (args.resume_sam_path, lambda sd: predictor.model.load_state_dict(sd["model"], strict=True)),
        (args.resume_plm_path, lambda sd: plm.load_state_dict(sd["plm"], strict=True)),
    ]:
        sd = torch.load(path, map_location="cpu")
        load_fn(sd)
    if args.resume_lora_path and plm.peft_enabled:
        plm.load_lora(args.resume_lora_path)
    sd = torch.load(args.resume_head_path, map_location="cpu")
    if sd.get("head_version") == 2:
        head.load_state_dict(sd["head"], strict=True)
    else:
        head.load_v1_head_partial(sd["head"])
    print(f"[fix_v1 eval] HeadV2 gate={head.gate_value:.4f}")

    predictor.model.eval()
    head.eval()
    dec_dev = next(predictor.model.sam_mask_decoder.parameters()).device
    dec_dtype = next(predictor.model.sam_mask_decoder.parameters()).dtype
    image_pe = predictor.model.sam_prompt_encoder.get_dense_pe().to(dec_dev, dec_dtype)

    if args.force_op == "ALL":
        suite_root = args.out_dir
        summaries = []
        for op in PER_OP_DIAGNOSIS_OPS:
            op_out = os.path.join(suite_root, op)
            op_flag = os.path.join(suite_root, f"diag_{op}.flag")
            summaries.append(
                run_one_force_op(
                    force_op=op,
                    args=args,
                    data_root=data_root,
                    predictor=predictor,
                    head=head,
                    image_pe=image_pe,
                    dec_dev=dec_dev,
                    dec_dtype=dec_dtype,
                    out_dir=op_out,
                    flag_path=op_flag,
                )
            )
        summary_path = os.path.join(suite_root, "per_op_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"root": suite_root, "eval_stage": args.eval_stage, "ops": summaries}, f, indent=2)
        print("\n=== Per-op suite done ===")
        print(f"{'op':<16} {'n':>6} {'TRG':>8} {'pos':>8} {'out':>8}")
        for s in summaries:
            print(
                f"{s['op']:<16} {s['n']:>6} {s['mean_trg']:>8.4f} "
                f"{s['positive_fraction']:>8.4f} {s['mean_output_iou']:>8.4f}"
            )
        print(f"Wrote {summary_path}")
        return

    run_one_force_op(
        force_op=args.force_op,
        args=args,
        data_root=data_root,
        predictor=predictor,
        head=head,
        image_pe=image_pe,
        dec_dev=dec_dev,
        dec_dtype=dec_dtype,
        out_dir=args.out_dir,
        flag_path=args.flag_path,
    )


if __name__ == "__main__":
    main()
