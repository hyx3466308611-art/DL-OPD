#!/usr/bin/env python3
"""
Teacher Recovery Gain (TRG) evaluation for Correction Teacher Stage 1.

TRG = mIoU(Teacher_Pred, GT) - mIoU(Corrupted_Mask, GT)
Writes teacher_frozen_ready.flag when global mean TRG passes threshold.
"""

import argparse
import json
import os
import random

import cv2
import numpy as np
import torch
from tqdm import tqdm

from converseg_paths import (
    apply_offline_env,
    default_sam_config_str,
    prepare_python_import_path,
    resolve_eval_data_dir,
)

prepare_python_import_path()

from data.mask_corrupter import (
    FORCEABLE_OPS,
    CurriculumStage,
    MaskCorrupter,
    mask_iou,
    sample_mix_stage,
)
from models.language_adapter import LanguageAdapter
from models.teacher_correction_model import TeacherCorrectionHead
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils.trg_eval_report import print_trg_headline, summarize_trg_rows
from utils.util_functions import bin_mask, erode, read_img_mask


def load_eval_items(data_root: str, split: str, max_items: int | None):
    path = os.path.join(data_root, f"{split}.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
            if max_items and len(items) >= max_items:
                break
    return items, path


def soft_mask_tensor(
    mask_arr: np.ndarray,
    target_h: int,
    target_w: int,
    dec_dev,
    dec_dtype,
) -> torch.Tensor:
    """Align mask to SAM image canvas (same H×W as `read_img_mask` output / training `mask_t`)."""
    m = mask_arr.astype(np.float32)
    if m.shape[0] != target_h or m.shape[1] != target_w:
        m = cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    m = np.clip(m, 0.0, 1.0)
    return torch.from_numpy(m).unsqueeze(0).unsqueeze(0).to(device=dec_dev, dtype=dec_dtype)


def infer_one(
    predictor,
    head,
    image_pe,
    img,
    gt_u8,
    corrupt_u8,
    prompt,
    img_path,
    dec_dev,
    dec_dtype,
):
    h_img, w_img = img.shape[:2]
    predictor.set_image(img)
    img_emb = predictor._features["image_embed"][-1].unsqueeze(0).to(dec_dev, dec_dtype)
    hi = [lvl[-1].unsqueeze(0).to(dec_dev, dec_dtype) for lvl in predictor._features["high_res_feats"]]
    _, _, Hm, Wm = img_emb.shape

    mask_t = soft_mask_tensor(corrupt_u8, h_img, w_img, dec_dev, dec_dtype)

    sp, dp = head([prompt or "segment"], Hm, Wm, [img_path], mask_t)
    model = predictor.model
    low, *_ = model.sam_mask_decoder(
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
    trg = output_iou - input_iou
    return input_iou, output_iou, trg, pred_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None, help="ConVerSeg eval root (default: data/ConverSeg)")
    ap.add_argument("--splits", default="sam_seeded,human_annotated")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument(
        "--eval-stage",
        choices=("A", "B", "C", "MIX"),
        default="MIX",
        help="A/B/C=固定课程难度；MIX=主协议，每样本按 A:B:C=1:2:2 抽 stage（seed=42）",
    )
    ap.add_argument(
        "--force-op",
        default=None,
        help="诊断：算子名（见 PER_OP_DIAGNOSIS_OPS）。不可与 MIX 同用；建议 --eval-stage C。"
        "完整 10 算子扫一遍请用 fix_v1/eval_trg_fix_v1.py --force-op ALL。",
    )
    ap.add_argument("--model-cfg", default=default_sam_config_str())
    ap.add_argument(
        "--checkpoint",
        required=True,
        help="SAM2 预训练路径 (teacher/sam2_hiera_large.pt)，供 build_sam2。",
    )
    ap.add_argument("--plm-model-name", required=True)
    ap.add_argument("--resume-sam-path", required=True, help="微调 SAM 权重（含 model 键）")
    ap.add_argument("--resume-plm-path", required=True)
    ap.add_argument("--resume-lora-path", default=None)
    ap.add_argument("--resume-head-path", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-dim", type=int, default=1024)
    ap.add_argument("--er-k", type=int, default=5)
    ap.add_argument("--er-it", type=int, default=1)
    ap.add_argument("--trg-mean-threshold", type=float, default=0.03)
    ap.add_argument("--trg-positive-frac", type=float, default=0.55)
    ap.add_argument("--out-dir", default="./ckpts/teacher_correction/eval_recovery")
    ap.add_argument("--flag-path", default="./ckpts/teacher_correction/teacher_frozen_ready.flag")
    args = ap.parse_args()

    apply_offline_env()
    data_root = args.data_root or str(resolve_eval_data_dir())
    sam_weights = args.checkpoint
    os.makedirs(args.out_dir, exist_ok=True)

    predictor = SAM2ImagePredictor(build_sam2(args.model_cfg, sam_weights, device=args.device))
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
    head = TeacherCorrectionHead(plm, transformer_dim=C).to(args.device)

    sd = torch.load(args.resume_sam_path, map_location="cpu")
    predictor.model.load_state_dict(sd["model"], strict=True)
    sd = torch.load(args.resume_plm_path, map_location="cpu")
    plm.load_state_dict(sd["plm"], strict=True)
    if args.resume_lora_path and plm.peft_enabled:
        plm.load_lora(args.resume_lora_path)
    sd = torch.load(args.resume_head_path, map_location="cpu")
    head.load_state_dict(sd["head"], strict=True)
    print(
        f"Loaded correction head from {args.resume_head_path}; "
        f"gate={head.gate_value:.4f} (saved gate={sd.get('gate_value', 'n/a')})"
    )

    predictor.model.eval()
    head.eval()
    dec_dev = next(predictor.model.sam_mask_decoder.parameters()).device
    dec_dtype = next(predictor.model.sam_mask_decoder.parameters()).dtype
    image_pe = predictor.model.sam_prompt_encoder.get_dense_pe().to(dec_dev, dec_dtype)

    if args.force_op is not None and args.force_op not in FORCEABLE_OPS:
        raise SystemExit(f"ERROR: unknown --force-op={args.force_op}; choose from {sorted(FORCEABLE_OPS)}")
    if args.force_op and args.eval_stage == "MIX":
        raise SystemExit("ERROR: --force-op 不可与 --eval-stage MIX 同用；诊断请用 --eval-stage C（或 B）")

    fixed_stage = None if args.eval_stage == "MIX" else CurriculumStage(args.eval_stage)
    mix_rng = random.Random(42) if args.eval_stage == "MIX" else None
    corrupter = MaskCorrupter(seed=42)
    if args.eval_stage == "MIX":
        print("Mixed-TRG protocol: per-sample stage ~ A:B:C = 1:2:2 (stage_rng seed=42, corrupt seed=42)")
    if args.force_op:
        print(
            f"Per-op diagnosis: force_op={args.force_op} severity_stage={args.eval_stage} "
            f"(no random dropout/compound)"
        )

    all_rows = []

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        items, jsonl_path = load_eval_items(data_root, split, args.max_items)
        print(f"Evaluating {split}: {len(items)} items from {jsonl_path}")
        for item in tqdm(items, desc=split):
            img_rel = item.get("image") or item.get("image_path")
            msk_rel = item.get("mask") or item.get("mask_merged")
            prompt = item.get("prompt", "segment")
            img_p = img_rel if os.path.isabs(img_rel) else os.path.join(data_root, img_rel)
            msk_p = msk_rel if os.path.isabs(msk_rel) else os.path.join(data_root, msk_rel)
            if not os.path.isfile(img_p) or not os.path.isfile(msk_p):
                continue
            img, ann = read_img_mask(img_p, msk_p, args.max_dim)
            h_img, w_img = img.shape[:2]
            gt = erode(bin_mask(ann), args.er_k, args.er_it).astype(np.uint8)
            if gt.shape[0] != h_img or gt.shape[1] != w_img:
                gt = cv2.resize(gt, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
            stage = sample_mix_stage(mix_rng) if mix_rng is not None else fixed_stage
            corrupt, meta = corrupter.corrupt(gt, stage, force_op=args.force_op)
            if meta.get("skip"):
                continue
            corrupt_aligned = corrupt.astype(np.float32)
            if corrupt_aligned.shape[0] != h_img or corrupt_aligned.shape[1] != w_img:
                corrupt_aligned = cv2.resize(
                    corrupt_aligned,
                    (w_img, h_img),
                    interpolation=cv2.INTER_LINEAR,
                )
            corrupt_aligned = np.clip(corrupt_aligned, 0.0, 1.0)
            inp_iou, out_iou, trg, _ = infer_one(
                predictor,
                head,
                image_pe,
                img,
                gt,
                corrupt_aligned,
                prompt,
                img_p,
                dec_dev,
                dec_dtype,
            )
            all_rows.append(
                {
                    "id": item.get("id"),
                    "split": split,
                    "op": meta["op"],
                    "stage": meta["stage"],
                    "force_op": args.force_op,
                    "input_iou": inp_iou,
                    "output_iou": out_iou,
                    "trg": trg,
                }
            )

    if not all_rows:
        raise RuntimeError("No evaluation rows collected.")

    summary = summarize_trg_rows(all_rows)
    mean_trg = summary["mean_trg"]
    pos_frac = summary["positive_fraction"]
    print_trg_headline(summary, eval_stage=args.eval_stage)

    report_path = os.path.join(args.out_dir, "trg_report.json")
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
                "force_op": args.force_op,
                "mix_weights": "A:B:C=1:2:2" if args.eval_stage == "MIX" else None,
                "by_split": summary["by_split"],
                "by_stage": summary["by_stage"],
                "by_input_iou_bin": summary["by_input_iou_bin"],
                "by_op_family": summary["by_op_family"],
                "rows": all_rows,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {report_path}")

    # 冻结门槛只用于 Mixed/主协议；单算子诊断不覆盖 ready 语义时仍写 flag 供排查
    ready = mean_trg > args.trg_mean_threshold and pos_frac >= args.trg_positive_frac
    if args.force_op:
        ready = False  # 诊断跑不作为冻结依据
    flag_body = {
        "ready": ready,
        "mean_trg": mean_trg,
        "positive_fraction": pos_frac,
        "mean_input_iou": summary["mean_input_iou"],
        "mean_output_iou": summary["mean_output_iou"],
        "eval_stage": args.eval_stage,
        "force_op": args.force_op,
        "mask_gate": head.gate_value,
        "head_checkpoint": args.resume_head_path,
        "report": report_path,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.flag_path)), exist_ok=True)
    with open(args.flag_path, "w", encoding="utf-8") as f:
        json.dump(flag_body, f, indent=2)
    if ready:
        print(f"PASS: wrote freeze flag -> {args.flag_path}")
    else:
        print(f"FAIL: TRG gate not met; flag written with ready=false -> {args.flag_path}")


if __name__ == "__main__":
    main()
