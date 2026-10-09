#!/usr/bin/env python3
"""fix_v2 TRG evaluator: Mixed main protocol and forced per-operator diagnosis."""

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
sys.path.insert(0, str(FIX_DIR))

from converseg_paths import apply_offline_env, default_sam_config_str, prepare_python_import_path, resolve_eval_data_dir

prepare_python_import_path()

from data.mask_corrupter import CurriculumStage, MaskCorrupter, mask_iou, sample_mix_stage
from eval_teacher_recovery import load_eval_items, soft_mask_tensor
from mask_corrupt_fix_v2 import FORCEABLE_OPS, PER_OP_DIAGNOSIS_OPS
from models.language_adapter import LanguageAdapter
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from teacher_correction_head_v2 import TeacherCorrectionHead
from utils.trg_eval_report import print_trg_headline, summarize_trg_rows
from utils.util_functions import bin_mask, erode, read_img_mask


@torch.inference_mode()
def infer_one(predictor, head, image_pe, img, gt, corrupt, prompt, path, dec_dev, dec_dtype):
    height, width = img.shape[:2]
    predictor.set_image(img)
    image_embedding = predictor._features["image_embed"][-1].unsqueeze(0).to(dec_dev, dec_dtype)
    high_res = [
        level[-1].unsqueeze(0).to(dec_dev, dec_dtype)
        for level in predictor._features["high_res_feats"]
    ]
    _, _, out_h, out_w = image_embedding.shape
    mask_t = soft_mask_tensor(corrupt, height, width, dec_dev, dec_dtype)
    sparse, dense = head([prompt or "segment"], out_h, out_w, [path], mask_t)
    low, *_ = predictor.model.sam_mask_decoder(
        image_embeddings=image_embedding,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse.to(dec_dev, dec_dtype),
        dense_prompt_embeddings=dense.to(dec_dev, dec_dtype),
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res,
    )
    pred = (torch.sigmoid(low[0, 0]) > 0.5).cpu().numpy().astype(np.uint8)
    pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    input_iou = mask_iou((corrupt > 0.5).astype(np.uint8), gt)
    output_iou = mask_iou(pred, gt)
    return input_iou, output_iou, output_iou - input_iou


def evaluate(args, force_op, predictor, head, image_pe, dec_dev, dec_dtype, data_root, out_dir):
    fixed_stage = None if args.eval_stage == "MIX" else CurriculumStage(args.eval_stage)
    mix_rng = random.Random(42) if args.eval_stage == "MIX" else None
    # Evaluation must stay identical across baseline/fix_v1/fix_v2.
    # The strengthened fix_v2 corruption policy is training-only.
    corrupter = MaskCorrupter(seed=42)
    rows = []

    for split_name in [value.strip() for value in args.splits.split(",") if value.strip()]:
        items, manifest = load_eval_items(data_root, split_name, args.max_items)
        print(f"Eval {split_name}: n={len(items)} manifest={manifest} op={force_op or args.eval_stage}")
        for item in tqdm(items, desc=f"{split_name}/{force_op or args.eval_stage}"):
            image_rel = item.get("image") or item.get("image_path")
            mask_rel = item.get("mask") or item.get("mask_merged")
            image_path = image_rel if os.path.isabs(image_rel) else os.path.join(data_root, image_rel)
            mask_path = mask_rel if os.path.isabs(mask_rel) else os.path.join(data_root, mask_rel)
            if not os.path.isfile(image_path) or not os.path.isfile(mask_path):
                continue
            image, annotation = read_img_mask(image_path, mask_path, args.max_dim)
            if args.full_gt:
                gt = bin_mask(annotation).astype(np.uint8)
            else:
                gt = erode(bin_mask(annotation), args.er_k, args.er_it).astype(np.uint8)
            if gt.shape != image.shape[:2]:
                gt = cv2.resize(gt, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            stage = sample_mix_stage(mix_rng) if mix_rng is not None else fixed_stage
            corrupt, meta = corrupter.corrupt(gt, stage, force_op=force_op)
            if meta.get("skip"):
                continue
            corrupt = corrupt.astype(np.float32)
            if corrupt.shape != image.shape[:2]:
                corrupt = cv2.resize(
                    corrupt, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR
                )
            input_iou, output_iou, trg = infer_one(
                predictor,
                head,
                image_pe,
                image,
                gt,
                corrupt,
                item.get("prompt", "segment"),
                image_path,
                dec_dev,
                dec_dtype,
            )
            rows.append(
                {
                    "id": item.get("id"),
                    "split": split_name,
                    "op": meta["op"],
                    "stage": meta["stage"],
                    "force_op": force_op,
                    "input_iou": input_iou,
                    "output_iou": output_iou,
                    "trg": trg,
                }
            )

    if not rows:
        raise RuntimeError(f"No evaluation rows for force_op={force_op}")
    summary = summarize_trg_rows(rows)
    print_trg_headline(summary, eval_stage=args.eval_stage if not force_op else f"{args.eval_stage}+{force_op}")
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "trg_report.json")
    report = {
        "n": summary["n"],
        "mean_trg": summary["mean_trg"],
        "positive_fraction": summary["positive_fraction"],
        "mean_input_iou": summary["mean_input_iou"],
        "mean_output_iou": summary["mean_output_iou"],
        "n_trg_negative": summary["n_trg_negative"],
        "n_trg_ge_0_3": summary["n_trg_ge_0_3"],
        "threshold_mean": args.trg_mean_threshold,
        "threshold_pos_frac": args.trg_positive_frac,
        "eval_stage": args.eval_stage,
        "force_op": force_op,
        "mix_weights": "A:B:C=1:2:2" if args.eval_stage == "MIX" else None,
        "full_gt": args.full_gt,
        "head_version": "fix_v2_spatial_3x3",
        "corruption_version": "shared_benchmark_seed42",
        "by_split": summary["by_split"],
        "by_stage": summary["by_stage"],
        "by_input_iou_bin": summary["by_input_iou_bin"],
        "by_op_family": summary["by_op_family"],
        "rows": rows,
    }
    Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {
        "op": force_op or "MIX",
        "n": summary["n"],
        "mean_trg": summary["mean_trg"],
        "positive_fraction": summary["positive_fraction"],
        "mean_input_iou": summary["mean_input_iou"],
        "mean_output_iou": summary["mean_output_iou"],
        "report": report_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--splits", default="sam_seeded,human_annotated")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--eval-stage", choices=("A", "B", "C", "MIX"), default="MIX")
    parser.add_argument("--force-op", default=None, help="One op or ALL; incompatible with MIX")
    parser.add_argument("--model-cfg", default=default_sam_config_str())
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--plm-model-name", required=True)
    parser.add_argument("--resume-sam-path", required=True)
    parser.add_argument("--resume-plm-path", required=True)
    parser.add_argument("--resume-lora-path", default=None)
    parser.add_argument("--resume-head-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-dim", type=int, default=1024)
    parser.add_argument("--er-k", type=int, default=5)
    parser.add_argument("--er-it", type=int, default=1)
    parser.add_argument("--full-gt", action="store_true")
    parser.add_argument("--trg-mean-threshold", type=float, default=0.03)
    parser.add_argument("--trg-positive-frac", type=float, default=0.55)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--flag-path", required=True)
    args = parser.parse_args()

    if args.force_op and args.eval_stage == "MIX":
        raise SystemExit("ERROR: force-op diagnosis requires EVAL_STAGE=C (or B), not MIX")
    if args.force_op not in (None, "ALL") and args.force_op not in FORCEABLE_OPS:
        raise SystemExit(f"Unknown force-op {args.force_op}; use ALL or {sorted(FORCEABLE_OPS)}")

    apply_offline_env()
    data_root = args.data_root or str(resolve_eval_data_dir())
    predictor = SAM2ImagePredictor(build_sam2(args.model_cfg, args.checkpoint, device=args.device))
    dim = predictor.model.sam_mask_decoder.transformer_dim
    plm = LanguageAdapter(
        model_name=args.plm_model_name,
        transformer_dim=dim,
        use_dense_bias=True,
        use_lora=True,
        dtype=torch.bfloat16,
        device=args.device,
        use_image_input=True,
    ).to(args.device)
    head = TeacherCorrectionHead(plm, transformer_dim=dim).to(args.device)
    predictor.model.load_state_dict(torch.load(args.resume_sam_path, map_location="cpu")["model"], strict=True)
    plm.load_state_dict(torch.load(args.resume_plm_path, map_location="cpu")["plm"], strict=True)
    if args.resume_lora_path and plm.peft_enabled:
        plm.load_lora(args.resume_lora_path)
    head_state = torch.load(args.resume_head_path, map_location="cpu")
    head.load_state_dict(head_state["head"], strict=True)
    predictor.model.eval()
    head.eval()
    dec_dev = next(predictor.model.sam_mask_decoder.parameters()).device
    dec_dtype = next(predictor.model.sam_mask_decoder.parameters()).dtype
    image_pe = predictor.model.sam_prompt_encoder.get_dense_pe().to(dec_dev, dec_dtype)
    print(f"[fix_v2 eval] gate={head.gate_value:.4f} head={head_state.get('head_version')}")

    if args.force_op == "ALL":
        summaries = []
        for op in PER_OP_DIAGNOSIS_OPS:
            summaries.append(
                evaluate(
                    args,
                    op,
                    predictor,
                    head,
                    image_pe,
                    dec_dev,
                    dec_dtype,
                    data_root,
                    os.path.join(args.out_dir, op),
                )
            )
        summary_path = os.path.join(args.out_dir, "per_op_summary.json")
        Path(summary_path).write_text(
            json.dumps({"eval_stage": args.eval_stage, "ops": summaries}, indent=2),
            encoding="utf-8",
        )
        print(f"{'op':<16} {'n':>6} {'TRG':>8} {'pos':>8} {'out':>8}")
        for value in summaries:
            print(
                f"{value['op']:<16} {value['n']:>6} {value['mean_trg']:>8.4f} "
                f"{value['positive_fraction']:>8.4f} {value['mean_output_iou']:>8.4f}"
            )
        print(f"Wrote {summary_path}")
        return

    result = evaluate(
        args,
        args.force_op,
        predictor,
        head,
        image_pe,
        dec_dev,
        dec_dtype,
        data_root,
        args.out_dir,
    )
    ready = (
        not args.force_op
        and result["mean_trg"] > args.trg_mean_threshold
        and result["positive_fraction"] >= args.trg_positive_frac
    )
    flag = {
        "ready": ready,
        "diagnosis": bool(args.force_op),
        "eval_stage": args.eval_stage,
        "force_op": args.force_op,
        "mean_trg": result["mean_trg"],
        "positive_fraction": result["positive_fraction"],
        "mean_input_iou": result["mean_input_iou"],
        "mean_output_iou": result["mean_output_iou"],
        "mask_gate": head.gate_value,
        "head_checkpoint": args.resume_head_path,
        "report": result["report"],
        "fix_v2": True,
    }
    Path(args.flag_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.flag_path).write_text(json.dumps(flag, indent=2), encoding="utf-8")
    print(f"ready={ready} flag={args.flag_path}")


if __name__ == "__main__":
    main()
