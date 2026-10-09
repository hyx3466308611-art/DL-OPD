#!/usr/bin/env python3
"""
Correction Teacher fix v1 — full defect repair per TRG analysis spec.
- Full GT supervision (no erode), negative mix, dual-mode text-only / correction
- TeacherCorrectionHeadV2, final-only checkpoints
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

FIX_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIX_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(FIX_DIR))

from converseg_paths import apply_offline_env, default_sam_config_str, prepare_python_import_path

prepare_python_import_path()

from curriculum_fix import curriculum_stage_from_step_fix
from data.load_samples import load_samples_from_jsonl
from data_mix import load_converseg_jsonl_pairs, load_negative_pairs, mix_train_pairs
from losses_fix import dice_loss_masked, identity_preserve_loss_l1, negative_suppression_loss
from mask_corrupt_fix import MaskCorrupterFixV1
from models.language_adapter import LanguageAdapter
from models.teacher_correction_model import TeacherCorrectionHeadV2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from train import PairListDataset, collate_keep_list, dice_loss_from_logits, split
from train_teacher_correction import (
    ensure_square_max_dim,
    focal_loss_from_logits,
    resize_masks_continuous,
    stack_gt_for_logits,
)
from converseg_paths import REPO_ROOT as REPO_ROOT_PATH
from utils.util_functions import bin_mask, iou, read_img_mask, seed


def configure_fix_v1_backbone(args) -> None:
    """Set PLM/SAM paths only; do not touch init_preset or resume_* from slurm."""
    bb = args.teacher_backbone
    if bb == "author3b":
        if args.plm_model_name is None:
            args.plm_model_name = str(REPO_ROOT_PATH / "student" / "Qwen2.5-VL-3B")
        if args.checkpoint is None:
            args.checkpoint = str(REPO_ROOT_PATH / "student" / "sam2_hiera_large.pt")
    elif bb == "teacher7b":
        if args.plm_model_name is None:
            args.plm_model_name = str(REPO_ROOT_PATH / "teacher" / "Qwen2.5-VL-7B")
        if args.checkpoint is None:
            args.checkpoint = str(REPO_ROOT_PATH / "teacher" / "sam2_hiera_large.pt")
    else:
        raise ValueError(f"Unknown teacher_backbone: {bb}")


def train_loop_fix_v1(
    train_pairs,
    negative_pairs,
    predictor,
    head: TeacherCorrectionHeadV2,
    corrupter: MaskCorrupterFixV1,
    steps,
    lr,
    wd,
    batch_size,
    acc,
    out,
    name,
    max_dim,
    log_every,
    save_every,
    workers,
    seed_val,
    lam_bce,
    lam_dice,
    lam_focal,
    lam_neg,
    lam_preserve,
    preserve_iou_thresh,
    negative_sample_prob,
    dual_mode_prob,
    curriculum_c_start,
    min_lr,
    warmup_updates,
):
    model = predictor.model
    plm = head.plm
    model.train()
    plm.train()
    head.train()

    for p in model.parameters():
        p.requires_grad = False
    for p in model.sam_mask_decoder.parameters():
        p.requires_grad = True
    for p in model.sam_prompt_encoder.parameters():
        p.requires_grad = True

    tb_dir = os.path.join(out, "tb", name)
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    params = [p for p in (list(model.parameters()) + list(head.parameters())) if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    total_updates = max(1, math.ceil(steps / max(1, acc)))
    warmup_updates = max(0, int(0.03 * total_updates)) if warmup_updates < 0 else warmup_updates
    cos_updates = max(1, total_updates - warmup_updates)
    if warmup_updates > 0:
        warmup = LinearLR(opt, start_factor=1e-8, total_iters=warmup_updates)
        cosine = CosineAnnealingLR(opt, T_max=cos_updates, eta_min=min_lr)
        scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_updates])
    else:
        scheduler = CosineAnnealingLR(opt, T_max=cos_updates, eta_min=min_lr)

    dec_dev = next(model.sam_mask_decoder.parameters()).device
    dec_dtype = next(model.sam_mask_decoder.parameters()).dtype
    image_pe_const = model.sam_prompt_encoder.get_dense_pe().to(dec_dev, dec_dtype)

    ds = PairListDataset(train_pairs)
    g = torch.Generator()
    g.manual_seed(seed_val)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_keep_list,
        drop_last=False,
        generator=g,
    )

    def infinite_loader():
        while True:
            for b in loader:
                yield b

    data_it = infinite_loader()
    rng = random.Random(seed_val)
    os.makedirs(out, exist_ok=True)

    opt.zero_grad(set_to_none=True)
    update_idx = grad_acc = 0
    print("[fix_v1] Training (HeadV2, full GT, dual-mode, negatives)", flush=True)
    t_loop = time.time()
    t_step = time.time()

    for step in range(1, steps + 1):
        batch = next(data_it)
        stage = curriculum_stage_from_step_fix(step, steps, c_start=curriculum_c_start)

        imgs, gts, prompts, img_paths, corrupt_masks, suppress_flags = [], [], [], [], [], []

        for _img_p, _msk_p, _pr in batch:
            use_negative = negative_pairs and rng.random() < negative_sample_prob
            if use_negative:
                img_p, msk_p, pr = rng.choice(negative_pairs)
                is_negative = True
            else:
                img_p, msk_p, pr = _img_p, _msk_p, _pr
                is_negative = False

            img, ann = read_img_mask(img_p, msk_p, max_dim)
            img = ensure_square_max_dim(img, max_dim, is_mask=False)
            ann = ensure_square_max_dim(ann, max_dim, is_mask=True)

            if is_negative:
                gt = np.zeros_like(ann, dtype=np.float32)
            else:
                gt = bin_mask(ann).astype(np.float32)
            gt = ensure_square_max_dim(gt, max_dim, is_mask=True)

            text_only = (not is_negative) and rng.random() < dual_mode_prob
            if is_negative or text_only:
                corrupted = np.zeros_like(gt, dtype=np.float32)
                meta_iou = 0.0 if not is_negative else 1.0
            else:
                corrupted, meta = corrupter.corrupt((gt > 0.5).astype(np.uint8), stage)
                corrupted = corrupted.astype(np.float32)
                meta_iou = meta["input_iou"]

            corrupted = ensure_square_max_dim(corrupted, max_dim, is_mask=True)
            suppress = 0.0 if is_negative else 1.0

            imgs.append(img)
            gts.append(np.clip(gt, 0.0, 1.0))
            corrupt_masks.append(np.clip(corrupted, 0.0, 1.0))
            prompts.append(pr or "segment")
            img_paths.append(img_p)
            suppress_flags.append(suppress)

        if not imgs:
            continue

        predictor.set_image_batch(imgs)
        img_emb_b = predictor._features["image_embed"].to(dec_dev, dec_dtype)
        hi_b = [lvl.to(dec_dev, dec_dtype) for lvl in predictor._features["high_res_feats"]]
        _, _, Hm, Wm = img_emb_b.shape

        mask_t = resize_masks_continuous(corrupt_masks, max_dim, max_dim).to(device=dec_dev, dtype=dec_dtype)
        suppress_t = torch.tensor(suppress_flags, device=dec_dev, dtype=dec_dtype)

        sp, dp = head(prompts, Hm, Wm, img_paths, mask_t, suppress_mask_branch=suppress_t)
        sp, dp = sp.to(dec_dev, dec_dtype), dp.to(dec_dev, dec_dtype)

        low, *_ = model.sam_mask_decoder(
            image_embeddings=img_emb_b,
            image_pe=image_pe_const,
            sparse_prompt_embeddings=sp,
            dense_prompt_embeddings=dp,
            multimask_output=False,
            repeat_image=False,
            high_res_features=hi_b,
        )
        logits = low[:, 0]
        h, w = logits.shape[-2:]
        gt_t = stack_gt_for_logits(gts, h, w, dec_dev)
        input_t = stack_gt_for_logits(corrupt_masks, h, w, dec_dev)

        bce = F.binary_cross_entropy_with_logits(logits, gt_t)
        dl = dice_loss_masked(logits, gt_t, dice_loss_from_logits)
        fl = dice_loss_masked(logits, gt_t, focal_loss_from_logits)
        ln = negative_suppression_loss(logits, gt_t)
        lp = identity_preserve_loss_l1(logits, input_t, gt_t, preserve_iou_thresh)

        empty_rows = gt_t.sum(dim=(1, 2)) < 0.5
        if empty_rows.all():
            loss_main = lam_bce * bce + lam_neg * ln
        elif empty_rows.any():
            loss_main = lam_bce * bce + lam_dice * dl + lam_focal * fl + lam_neg * ln
        else:
            loss_main = lam_bce * bce + lam_dice * dl + lam_focal * fl + lam_neg * ln

        loss = (loss_main + lam_preserve * lp) / acc
        loss.backward()
        grad_acc += 1

        if grad_acc == acc:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            grad_acc = 0
            update_idx += 1
            scheduler.step()
            if update_idx == 1:
                print(f"[fix_v1] First update step {step} ({time.time() - t_loop:.1f}s)", flush=True)

        if step == 1 or step <= 5 or step % log_every == 0:
            step_sec = time.time() - t_step
            t_step = time.time()
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                j = iou(preds, gt_t).item()
            tot = loss_main.item() + lam_preserve * lp.item()
            print(
                f"step {step} stage {stage.value} upd {update_idx}: loss {tot:.4f} "
                f"neg {ln.item():.4f} preserve {lp.item():.4f} IoU {j:.4f} "
                f"gate {head.gate_value:.4f} ({step_sec:.1f}s)",
                flush=True,
            )
            writer.add_scalar("train/loss", tot, step)
            writer.add_scalar("train/loss_neg", ln.item(), step)
            writer.add_scalar("train/loss_preserve", lp.item(), step)

        if save_every and step % save_every == 0 and step < steps:
            raise RuntimeError("fix_v1: use --save-every 0 (final only)")

    torch.save({"model": model.state_dict()}, os.path.join(out, f"{name}_final.torch"))
    torch.save({"plm": plm.state_dict()}, os.path.join(out, f"{name}_plm_final.torch"))
    head_final = os.path.join(out, f"{name}_head_final.torch")
    torch.save({"head": head.state_dict(), "gate_value": head.gate_value, "head_version": 2}, head_final)
    plm.save_lora(os.path.join(out, "lora_plm_adapter_final"))
    print(f"[fix_v1] Saved final -> {out} gate={head.gate_value:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--dataset-jsonl", default="phase2/phase2_mixed.jsonl")
    ap.add_argument(
        "--negative-jsonl",
        default="ConverSeg-Training-Data/converseg_stage2_data/conversational_negative_data/conversational_negative_data.jsonl",
    )
    ap.add_argument("--converseg-root", default=None)
    ap.add_argument("--human-jsonl", default="human_annotated.jsonl")
    ap.add_argument("--human-repeat", type=int, default=0)
    ap.add_argument("--negative-sample-prob", type=float, default=0.175)
    ap.add_argument("--dual-mode-prob", type=float, default=0.5)
    ap.add_argument("--model-cfg", default=default_sam_config_str())
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--plm-model-name", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--acc", type=int, default=3)
    ap.add_argument("--out", default=str(REPO_ROOT / "ckpts/teacher_correction/author3b_fix_v1"))
    ap.add_argument("--name", default="CorrectionTeacher_3B_fix_v1")
    ap.add_argument("--max-dim", type=int, default=1024)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--warmup-updates", type=int, default=-1)
    ap.add_argument("--lam-bce", type=float, default=1.0)
    ap.add_argument("--lam-dice", type=float, default=0.25)
    ap.add_argument("--lam-focal", type=float, default=0.5)
    ap.add_argument("--lam-neg", type=float, default=3.0)
    ap.add_argument("--lam-preserve", type=float, default=0.4)
    ap.add_argument("--preserve-iou-thresh", type=float, default=0.85)
    ap.add_argument("--boundary-elastic-boost", type=float, default=0.18)
    ap.add_argument("--curriculum-c-start", type=float, default=0.45)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--plm-gradient-checkpointing", action="store_true")
    ap.add_argument("--qwen-image-size", type=int, default=256)
    ap.add_argument("--resume-sam-path", default=None)
    ap.add_argument("--resume-plm-path", default=None)
    ap.add_argument("--resume-lora-path", default=None)
    ap.add_argument("--resume-head-path", default=None)
    ap.add_argument("--teacher-backbone", choices=("author3b", "teacher7b"), default="author3b")
    ap.add_argument(
        "--init-preset",
        default=None,
        help="fix_v1 使用显式 --resume-*；勿设 author3b_phase2 以免覆盖 INIT 路径",
    )
    ap.add_argument("--plm-load-strict", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    if args.human_repeat != 0:
        raise SystemExit(
            f"REFUSE: --human-repeat={args.human_repeat}. "
            "human_annotated is the ConverSeg eval split and must not be mixed into training."
        )

    apply_offline_env()
    configure_fix_v1_backbone(args)
    plm_path, sam_ckpt = args.plm_model_name, args.checkpoint
    for label, p in (
        ("resume_sam", args.resume_sam_path),
        ("resume_plm", args.resume_plm_path),
        ("resume_head", args.resume_head_path),
    ):
        if not p or not os.path.isfile(p):
            raise FileNotFoundError(f"fix_v1 missing {label}: {p}")
    print(
        f"[fix_v1] backbone={args.teacher_backbone} out={args.out} name={args.name} "
        f"batch={args.batch_size} acc={args.acc}"
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    seed(args.seed)

    pairs = load_samples_from_jsonl(
        args.data_dir, args.dataset_jsonl, num_samples=args.num_samples, repo_root=REPO_ROOT
    )
    converseg_root = args.converseg_root or str(REPO_ROOT / "data" / "ConverSeg" / "data")
    try:
        human = load_converseg_jsonl_pairs(converseg_root, args.human_jsonl, REPO_ROOT)
        train_pairs = mix_train_pairs(pairs, human, args.human_repeat, args.seed)
    except FileNotFoundError as e:
        print(f"[fix_v1] WARN: {e}; phase2 only")
        train_pairs, _ = split(pairs, 0.005, args.seed)
    else:
        train_pairs, _ = split(train_pairs, 0.005, args.seed)

    negative_pairs: list = []
    try:
        negative_pairs = load_negative_pairs(args.data_dir, args.negative_jsonl, REPO_ROOT)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[fix_v1] WARN: negative data unavailable: {e}")

    predictor = SAM2ImagePredictor(build_sam2(args.model_cfg, sam_ckpt, device=args.device))
    C = predictor.model.sam_mask_decoder.transformer_dim
    plm = LanguageAdapter(
        model_name=plm_path,
        transformer_dim=C,
        n_sparse_tokens=0,
        use_dense_bias=True,
        use_lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        dtype=torch.bfloat16,
        device=args.device,
        use_image_input=True,
        gradient_checkpointing=args.plm_gradient_checkpointing,
        qwen_image_size=args.qwen_image_size,
    ).to(args.device)
    head = TeacherCorrectionHeadV2(plm, transformer_dim=C).to(args.device)

    if args.resume_sam_path and os.path.isfile(args.resume_sam_path):
        sd = torch.load(args.resume_sam_path, map_location="cpu")
        predictor.model.load_state_dict(sd["model"], strict=True)
        print(f"[fix_v1] SAM <- {args.resume_sam_path}")
    if args.resume_plm_path and os.path.isfile(args.resume_plm_path):
        sd = torch.load(args.resume_plm_path, map_location="cpu")
        plm.load_state_dict(sd["plm"], strict=True)
        print(f"[fix_v1] PLM <- {args.resume_plm_path}")
    if args.resume_lora_path and plm.peft_enabled and os.path.isdir(args.resume_lora_path):
        plm.load_lora(args.resume_lora_path)
    if args.resume_head_path and os.path.isfile(args.resume_head_path):
        sd = torch.load(args.resume_head_path, map_location="cpu")
        if sd.get("head_version") == 2:
            head.load_state_dict(sd["head"], strict=True)
        else:
            head.load_v1_head_partial(sd["head"])
        print(f"[fix_v1] Head V2 (partial from V1 ok) <- {args.resume_head_path}")

    if args.plm_gradient_checkpointing:
        plm.prepare_gradient_checkpointing_training()

    corrupter = MaskCorrupterFixV1(seed=args.seed, boundary_elastic_boost=args.boundary_elastic_boost)

    train_loop_fix_v1(
        train_pairs,
        negative_pairs,
        predictor,
        head,
        corrupter,
        steps=args.steps,
        lr=args.lr,
        wd=args.wd,
        batch_size=args.batch_size,
        acc=args.acc,
        out=args.out,
        name=args.name,
        max_dim=args.max_dim,
        log_every=args.log_every,
        save_every=args.save_every,
        workers=args.workers,
        seed_val=args.seed,
        lam_bce=args.lam_bce,
        lam_dice=args.lam_dice,
        lam_focal=args.lam_focal,
        lam_neg=args.lam_neg,
        lam_preserve=args.lam_preserve,
        preserve_iou_thresh=args.preserve_iou_thresh,
        negative_sample_prob=args.negative_sample_prob,
        dual_mode_prob=args.dual_mode_prob,
        curriculum_c_start=args.curriculum_c_start,
        min_lr=args.min_lr,
        warmup_updates=args.warmup_updates,
    )


if __name__ == "__main__":
    main()
