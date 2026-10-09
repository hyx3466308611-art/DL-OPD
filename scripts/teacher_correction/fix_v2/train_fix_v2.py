#!/usr/bin/env python3
"""Correction Teacher fix_v2.

- full-resolution GT (no erosion)
- global + spatial confidence gate
- per-sample conditional loss for empty GT and high-IoU identity preservation
- 10% unrelated-prompt synthetic negatives
- human-annotated oversampling
- B/C mask dropout 0.15/0.30 and non-C 5% corruption area guard
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
FIX_V1_DIR = FIX_DIR.parent / "fix_v1"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(FIX_DIR))
sys.path.insert(0, str(FIX_V1_DIR))

from converseg_paths import apply_offline_env, default_sam_config_str, prepare_python_import_path

prepare_python_import_path()

from converseg_paths import REPO_ROOT as REPO_ROOT_PATH
from data.load_samples import load_samples_from_jsonl
from data.mask_corrupter import CurriculumStage
from data_mix import load_converseg_jsonl_pairs, mix_train_pairs
from mask_corrupt_fix_v2 import MaskCorrupterFixV2
from models.language_adapter import LanguageAdapter
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from teacher_correction_head_v2 import TeacherCorrectionHead
from train import PairListDataset, collate_keep_list, dice_loss_from_logits, split
from train_teacher_correction import (
    ensure_square_max_dim,
    focal_loss_from_logits,
    resize_masks_continuous,
    stack_gt_for_logits,
)
from utils.util_functions import bin_mask, iou, read_img_mask, seed


def curriculum_stage(step: int, total_steps: int) -> CurriculumStage:
    """Keep the fix_v1 hard-sample emphasis: A 20%, B 25%, C 55%."""
    p = (step - 1) / max(1, total_steps - 1)
    if p < 0.20:
        return CurriculumStage.A
    if p < 0.45:
        return CurriculumStage.B
    return CurriculumStage.C


def configure_backbone(args) -> None:
    if args.teacher_backbone == "author3b":
        args.plm_model_name = args.plm_model_name or str(REPO_ROOT_PATH / "student" / "Qwen2.5-VL-3B")
        args.checkpoint = args.checkpoint or str(REPO_ROOT_PATH / "student" / "sam2_hiera_large.pt")
    elif args.teacher_backbone == "teacher7b":
        args.plm_model_name = args.plm_model_name or str(REPO_ROOT_PATH / "teacher" / "Qwen2.5-VL-7B")
        args.checkpoint = args.checkpoint or str(REPO_ROOT_PATH / "teacher" / "sam2_hiera_large.pt")
    else:
        raise ValueError(args.teacher_backbone)


def unrelated_prompt(rng: random.Random, prompt_pool: list[str], current: str) -> str:
    """Choose a different prompt when possible; the paired GT is forced empty."""
    if not prompt_pool:
        return "an object not described by this image"
    candidate = rng.choice(prompt_pool)
    for _ in range(8):
        if candidate.strip() != (current or "").strip():
            return candidate
        candidate = rng.choice(prompt_pool)
    return candidate + " (unrelated target)"


def conditional_loss(
    logits: torch.Tensor,
    gt_t: torch.Tensor,
    input_t: torch.Tensor,
    meta_iou: list[float],
    *,
    lam_bce: float,
    lam_dice: float,
    lam_focal: float,
    empty_bce_weight: float,
    identity_weight: float,
    preserve_iou_thresh: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Exact per-sample conditional objective from the fix_v2 specification."""
    probs = torch.sigmoid(logits)
    total = logits.new_zeros(())
    empty_sum = logits.new_zeros(())
    identity_sum = logits.new_zeros(())
    base_sum = logits.new_zeros(())
    empty_n = identity_n = 0

    for i in range(logits.shape[0]):
        p_i = probs[i]
        g_i = gt_t[i]
        m_i = input_t[i]
        bce_i = F.binary_cross_entropy(p_i, g_i)

        if g_i.sum() < 1.0:
            sample_loss = empty_bce_weight * bce_i
            empty_sum = empty_sum + sample_loss
            empty_n += 1
        else:
            dl_i = dice_loss_from_logits(logits[i : i + 1], gt_t[i : i + 1])
            fl_i = focal_loss_from_logits(logits[i : i + 1], gt_t[i : i + 1])
            sample_loss = lam_bce * bce_i + lam_dice * dl_i + lam_focal * fl_i
            base_sum = base_sum + sample_loss

            if meta_iou[i] > preserve_iou_thresh:
                m_i_resized = F.interpolate(
                    m_i.unsqueeze(0).unsqueeze(0),
                    size=p_i.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
                identity = F.l1_loss(p_i, m_i_resized)
                sample_loss = sample_loss + identity_weight * identity
                identity_sum = identity_sum + identity
                identity_n += 1
        total = total + sample_loss

    batch_size = max(1, logits.shape[0])
    stats = {
        "empty": empty_sum / max(1, empty_n),
        "identity": identity_sum / max(1, identity_n),
        "base": base_sum / max(1, batch_size - empty_n),
        "empty_n": logits.new_tensor(float(empty_n)),
        "identity_n": logits.new_tensor(float(identity_n)),
    }
    return total / batch_size, stats


def train_loop(args, train_pairs, predictor, head, corrupter) -> None:
    model = predictor.model
    plm = head.plm
    model.train()
    plm.train()
    head.train()

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.sam_mask_decoder.parameters():
        parameter.requires_grad = True
    for parameter in model.sam_prompt_encoder.parameters():
        parameter.requires_grad = True

    params = [p for p in list(model.parameters()) + list(head.parameters()) if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters")
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    total_updates = max(1, math.ceil(args.steps / max(1, args.acc)))
    warmup_updates = (
        max(0, int(0.03 * total_updates)) if args.warmup_updates < 0 else args.warmup_updates
    )
    cosine_updates = max(1, total_updates - warmup_updates)
    if warmup_updates:
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=1e-8, total_iters=warmup_updates),
                CosineAnnealingLR(optimizer, T_max=cosine_updates, eta_min=args.min_lr),
            ],
            milestones=[warmup_updates],
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=cosine_updates, eta_min=args.min_lr)

    dec_dev = next(model.sam_mask_decoder.parameters()).device
    dec_dtype = next(model.sam_mask_decoder.parameters()).dtype
    image_pe = model.sam_prompt_encoder.get_dense_pe().to(dec_dev, dec_dtype)

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        PairListDataset(train_pairs),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_keep_list,
        drop_last=False,
        generator=generator,
    )

    def forever():
        while True:
            yield from loader

    data_iter = forever()
    rng = random.Random(args.seed)
    prompt_pool = [prompt or "segment" for _, _, prompt in train_pairs]
    os.makedirs(args.out, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.out, "tb", args.name))
    optimizer.zero_grad(set_to_none=True)
    grad_acc = update_idx = 0
    started = time.time()

    print(
        f"[fix_v2] start steps={args.steps} fake_negative={args.fake_negative_prob:.2f} "
        f"dropout(B/C)=0.15/0.30 full_gt=True gate={head.gate_value:.4f}",
        flush=True,
    )

    for step in range(1, args.steps + 1):
        stage = curriculum_stage(step, args.steps)
        batch = next(data_iter)
        imgs, gts, prompts, paths, corrupt_masks, meta_ious = [], [], [], [], [], []

        for img_path, mask_path, prompt in batch:
            img, ann = read_img_mask(img_path, mask_path, args.max_dim)
            img = ensure_square_max_dim(img, args.max_dim, is_mask=False)
            ann = ensure_square_max_dim(ann, args.max_dim, is_mask=True)

            fake_negative = rng.random() < args.fake_negative_prob
            if fake_negative:
                gt = np.zeros_like(ann, dtype=np.float32)
                current_prompt = unrelated_prompt(rng, prompt_pool, prompt)
                corrupted = np.zeros_like(gt, dtype=np.float32)
                current_iou = 1.0
            else:
                # Full precision original annotation: explicitly no erosion.
                gt = bin_mask(ann).astype(np.float32)
                current_prompt = prompt or "segment"
                corrupted, meta = corrupter.corrupt((gt > 0.5).astype(np.uint8), stage)
                corrupted = corrupted.astype(np.float32)
                current_iou = float(meta["input_iou"])

            gt = ensure_square_max_dim(gt, args.max_dim, is_mask=True)
            corrupted = ensure_square_max_dim(corrupted, args.max_dim, is_mask=True)
            imgs.append(img)
            gts.append(np.clip(gt, 0.0, 1.0))
            prompts.append(current_prompt)
            paths.append(img_path)
            corrupt_masks.append(np.clip(corrupted, 0.0, 1.0))
            meta_ious.append(current_iou)

        predictor.set_image_batch(imgs)
        image_embeddings = predictor._features["image_embed"].to(dec_dev, dec_dtype)
        high_res = [level.to(dec_dev, dec_dtype) for level in predictor._features["high_res_feats"]]
        _, _, height, width = image_embeddings.shape
        mask_t = resize_masks_continuous(
            corrupt_masks, args.max_dim, args.max_dim
        ).to(device=dec_dev, dtype=dec_dtype)
        sparse, dense = head(prompts, height, width, paths, mask_t)
        low, *_ = model.sam_mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse.to(dec_dev, dec_dtype),
            dense_prompt_embeddings=dense.to(dec_dev, dec_dtype),
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res,
        )
        logits = low[:, 0]
        out_h, out_w = logits.shape[-2:]
        gt_t = stack_gt_for_logits(gts, out_h, out_w, dec_dev)
        input_t = stack_gt_for_logits(corrupt_masks, out_h, out_w, dec_dev)

        batch_loss, loss_stats = conditional_loss(
            logits,
            gt_t,
            input_t,
            meta_ious,
            lam_bce=args.lam_bce,
            lam_dice=args.lam_dice,
            lam_focal=args.lam_focal,
            empty_bce_weight=args.empty_bce_weight,
            identity_weight=args.identity_weight,
            preserve_iou_thresh=args.preserve_iou_thresh,
        )
        loss = batch_loss / args.acc
        loss.backward()
        grad_acc += 1

        if grad_acc == args.acc:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            grad_acc = 0
            update_idx += 1

        if step <= 5 or step % args.log_every == 0:
            with torch.no_grad():
                pred = (torch.sigmoid(logits) > 0.5).float()
                batch_iou = iou(pred, gt_t).item()
            print(
                f"step {step} stage {stage.value} upd {update_idx}: loss {batch_loss.item():.4f} "
                f"empty {loss_stats['empty'].item():.4f}(n={int(loss_stats['empty_n'].item())}) "
                f"identity {loss_stats['identity'].item():.4f}(n={int(loss_stats['identity_n'].item())}) "
                f"IoU {batch_iou:.4f} corrupt_IoU {np.mean(meta_ious):.4f} "
                f"gate {head.gate_value:.4f} elapsed {time.time() - started:.0f}s",
                flush=True,
            )
            writer.add_scalar("train/loss", batch_loss.item(), step)
            writer.add_scalar("train/loss_empty", loss_stats["empty"].item(), step)
            writer.add_scalar("train/loss_identity", loss_stats["identity"].item(), step)
            writer.add_scalar("train/iou", batch_iou, step)
            writer.add_scalar("train/global_gate", head.gate_value, step)

    if grad_acc:
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

    torch.save({"model": model.state_dict()}, os.path.join(args.out, f"{args.name}_final.torch"))
    torch.save({"plm": plm.state_dict()}, os.path.join(args.out, f"{args.name}_plm_final.torch"))
    torch.save(
        {
            "head": head.state_dict(),
            "gate_value": head.gate_value,
            "head_version": "fix_v2_spatial_3x3",
        },
        os.path.join(args.out, f"{args.name}_head_final.torch"),
    )
    plm.save_lora(os.path.join(args.out, "lora_plm_adapter_final"))
    writer.close()
    print(f"[fix_v2] saved final checkpoints to {args.out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--dataset-jsonl", default="phase2/phase2_mixed.jsonl")
    parser.add_argument("--converseg-root", default=None)
    parser.add_argument("--human-jsonl", default="human_annotated.jsonl")
    parser.add_argument("--human-repeat", type=int, default=0)
    parser.add_argument("--fake-negative-prob", type=float, default=0.10)
    parser.add_argument("--model-cfg", default=default_sam_config_str())
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--plm-model-name", default=None)
    parser.add_argument("--teacher-backbone", choices=("author3b", "teacher7b"), default="author3b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--acc", type=int, default=3)
    parser.add_argument("--out", default=str(REPO_ROOT / "ckpts/teacher_correction/author3b_fix_v2"))
    parser.add_argument("--name", default="CorrectionTeacher_3B_fix_v2")
    parser.add_argument("--max-dim", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-updates", type=int, default=-1)
    parser.add_argument("--lam-bce", type=float, default=1.0)
    parser.add_argument("--lam-dice", type=float, default=0.25)
    parser.add_argument("--lam-focal", type=float, default=0.5)
    parser.add_argument("--empty-bce-weight", type=float, default=3.0)
    parser.add_argument("--identity-weight", type=float, default=0.5)
    parser.add_argument("--preserve-iou-thresh", type=float, default=0.85)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--plm-gradient-checkpointing", action="store_true")
    parser.add_argument("--qwen-image-size", type=int, default=256)
    parser.add_argument("--resume-sam-path", required=True)
    parser.add_argument("--resume-plm-path", required=True)
    parser.add_argument("--resume-head-path", required=True)
    parser.add_argument("--resume-lora-path", default=None)
    args = parser.parse_args()
    if args.human_repeat != 0:
        raise SystemExit(
            f"REFUSE: --human-repeat={args.human_repeat}. "
            "human_annotated is the ConverSeg eval split and must not be mixed into training."
        )

    apply_offline_env()
    configure_backbone(args)
    for label in ("resume_sam_path", "resume_plm_path", "resume_head_path"):
        path = getattr(args, label)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label}: {path}")
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
    except FileNotFoundError as exc:
        print(f"[fix_v2] WARN: {exc}; phase2 only")
        train_pairs = list(pairs)
    train_pairs, _ = split(train_pairs, 0.005, args.seed)

    predictor = SAM2ImagePredictor(build_sam2(args.model_cfg, args.checkpoint, device=args.device))
    dim = predictor.model.sam_mask_decoder.transformer_dim
    plm = LanguageAdapter(
        model_name=args.plm_model_name,
        transformer_dim=dim,
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
    head = TeacherCorrectionHead(plm, transformer_dim=dim).to(args.device)

    predictor.model.load_state_dict(torch.load(args.resume_sam_path, map_location="cpu")["model"], strict=True)
    plm.load_state_dict(torch.load(args.resume_plm_path, map_location="cpu")["plm"], strict=True)
    if args.resume_lora_path and os.path.isdir(args.resume_lora_path) and plm.peft_enabled:
        plm.load_lora(args.resume_lora_path)
    old_head = torch.load(args.resume_head_path, map_location="cpu")["head"]
    missing, unexpected = head.load_previous_head_partial(old_head)
    print(
        f"[fix_v2] transferred mask_encoder; new gates reset (global={head.gate_value:.4f}); "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    if args.plm_gradient_checkpointing:
        plm.prepare_gradient_checkpointing_training()

    train_loop(args, train_pairs, predictor, head, MaskCorrupterFixV2(seed=args.seed))


if __name__ == "__main__":
    main()
