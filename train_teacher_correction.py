#!/usr/bin/env python3
"""
Stage 1: Correction Teacher training (author 3B or 7B + SAM2-L).
Corrupts GT masks online, injects mask state into dense prompts, optimizes BCE+Dice+Focal.
"""

import argparse
import json
import math
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from converseg_paths import (
    REPO_ROOT,
    apply_offline_env,
    default_sam_config_str,
    prepare_python_import_path,
)

prepare_python_import_path()

from data.mask_corrupter import CurriculumStage, MaskCorrupter, curriculum_stage_from_step
from models.language_adapter import LanguageAdapter
from models.teacher_correction_model import TeacherCorrectionHead
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from data.load_samples import load_samples_from_jsonl
from train import (
    PairListDataset,
    collate_keep_list,
    dice_loss_from_logits,
    split,
)
from utils.util_functions import bin_mask, erode, iou, read_img_mask, seed


def focal_loss_from_logits(logits, target, gamma=2.0, alpha=0.25):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * target + (1 - p) * (1 - target)
    at = alpha * target + (1 - alpha) * (1 - target)
    loss = at * (1 - pt).clamp(min=1e-6).pow(gamma) * bce
    return loss.mean()


def ensure_square_max_dim(arr: np.ndarray, max_dim: int, *, is_mask: bool) -> np.ndarray:
    """
    ConVerSeg 标准画布：长边缩放到 max_dim，居中 padding 到 (max_dim, max_dim)。
    与 read_img_mask 一致；batch 内防御性二次对齐，避免 stack 尺寸不一致。
    """
    if arr.ndim == 2:
        h, w = arr.shape
    else:
        h, w = arr.shape[:2]
    if h == max_dim and w == max_dim:
        return arr

    scale = float(max_dim) / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if is_mask or arr.ndim == 2:
        resized = cv2.resize(arr.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        pad_val = 0.0
    else:
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(arr, (new_w, new_h), interpolation=interp)
        pad_val = (0, 0, 0)

    pad_w = max_dim - new_w
    pad_h = max_dim - new_h
    left, top = pad_w // 2, pad_h // 2
    right, bottom = pad_w - left, pad_h - top
    if arr.ndim == 2 or is_mask:
        out = cv2.copyMakeBorder(
            resized, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=pad_val
        )
        return np.ascontiguousarray(out)
    out = cv2.copyMakeBorder(
        resized, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=pad_val
    )
    return np.ascontiguousarray(out)


def resize_masks_continuous(masks: list, target_h: int, target_w: int) -> torch.Tensor:
    """Stack masks as [B,1,H,W] float32 in [0,1]; align to image canvas (same as eval soft_mask)."""
    rows = []
    for m in masks:
        mf = m.astype(np.float32)
        if mf.shape[0] != target_h or mf.shape[1] != target_w:
            mf = cv2.resize(mf, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        rows.append(np.clip(mf, 0.0, 1.0))
    stacked = np.stack(rows, axis=0)
    return torch.from_numpy(stacked).unsqueeze(1)


def stack_gt_for_logits(gts, out_h: int, out_w: int, device) -> torch.Tensor:
    """Resize each GT to decoder logits spatial size; gts must share the same canvas before this."""
    tensors = []
    for m in gts:
        mf = np.asarray(m, dtype=np.float32)
        if mf.shape[0] != out_h or mf.shape[1] != out_w:
            mf = cv2.resize(mf, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        tensors.append(torch.from_numpy(mf))
    return torch.stack(tensors, dim=0).to(device)


def _first_existing(*paths: os.PathLike) -> str | None:
    for p in paths:
        sp = str(p)
        if os.path.isfile(sp):
            return sp
    return None


def configure_teacher_backbone(args) -> None:
    """
    author3b: 作者 ConVerSeg-Net-3B + student/Qwen2.5-VL-3B（当前默认教师骨干）
    teacher7b: 自训 7B Phase2 + teacher/Qwen2.5-VL-7B（后续对比实验）
    """
    bb = args.teacher_backbone
    student_plm = REPO_ROOT / "student" / "Qwen2.5-VL-3B"
    student_sam = REPO_ROOT / "student" / "sam2_hiera_large.pt"
    teacher_plm = REPO_ROOT / "teacher" / "Qwen2.5-VL-7B"
    teacher_sam = REPO_ROOT / "teacher" / "sam2_hiera_large.pt"

    out_author = str(REPO_ROOT / "ckpts" / "teacher_correction" / "author3b")
    out_teacher7b = str(REPO_ROOT / "ckpts" / "teacher_correction" / "teacher7b")
    if not hasattr(args, "init_preset"):
        args.init_preset = None
    if bb == "author3b":
        if args.init_preset is None:
            args.init_preset = "author3b_phase2"
        if args.plm_model_name is None:
            args.plm_model_name = str(student_plm)
        if args.checkpoint is None:
            args.checkpoint = str(student_sam)
        if args.name == "CorrectionTeacher_7B":
            args.name = "CorrectionTeacher_3B"
        if args.out in (out_teacher7b, "./ckpts/teacher_correction", "./ckpts/teacher_correction/teacher7b"):
            args.out = out_author
    elif bb == "teacher7b":
        if args.init_preset is None:
            args.init_preset = "teacher7b_phase2"
        if args.plm_model_name is None:
            args.plm_model_name = str(teacher_plm)
        if args.checkpoint is None:
            args.checkpoint = str(teacher_sam)
        if args.name == "CorrectionTeacher_3B":
            args.name = "CorrectionTeacher_7B"
        if args.out in (out_author, "./ckpts/teacher_correction", "./ckpts/teacher_correction/author3b"):
            args.out = out_teacher7b
    else:
        raise ValueError(f"Unknown teacher_backbone: {bb}")


def apply_init_preset(args):
    """Fill resume_* from author Net-3B or self-trained 7B Phase2."""
    if not args.init_preset:
        return
    ck_author = REPO_ROOT / "checkpoints" / "ConVerSeg-Net-3B"
    ck_t2 = REPO_ROOT / "ckpts" / "teacher" / "stage2_phase2mix"
    name = "ConVerSeg_teacher_stage2"
    if args.init_preset == "teacher7b_phase2":
        args.resume_sam_path = args.resume_sam_path or _first_existing(
            ck_t2 / f"{name}_final.torch",
        )
        args.resume_plm_path = args.resume_plm_path or _first_existing(
            ck_t2 / f"{name}_plm_final.torch",
        )
        lora = ck_t2 / "lora_plm_adapter_final"
        if args.resume_lora_path is None and lora.is_dir():
            args.resume_lora_path = str(lora)
        if not args.resume_sam_path or not args.resume_plm_path:
            raise FileNotFoundError(
                "teacher7b_phase2 需要 ckpts/teacher/stage2_phase2mix/"
                f"{name}_final.torch 与 {name}_plm_final.torch"
            )
    elif args.init_preset == "author3b_phase2":
        args.resume_sam_path = args.resume_sam_path or _first_existing(
            ck_author / "ConVerSeg-Net_sam2_90000.torch",
            ck_author / "ConVerSeg-Net_sam2_final.torch",
            ck_author / "ConVerSeg_stage2_sam2_final.torch",
        )
        args.resume_plm_path = args.resume_plm_path or _first_existing(
            ck_author / "ConVerSeg-Net_plm_90000.torch",
            ck_author / "ConVerSeg-Net_plm_final.torch",
            ck_author / "ConVerSeg_stage2_plm_final.torch",
        )
        lora = ck_author / "lora_plm_adapter_final"
        if args.resume_lora_path is None and lora.is_dir():
            args.resume_lora_path = str(lora)
        args.plm_load_strict = True
    else:
        raise ValueError(f"Unknown init_preset: {args.init_preset}")


def resolve_teacher_plm_sam(args):
    if args.plm_model_name is None or args.checkpoint is None:
        configure_teacher_backbone(args)
    plm = args.plm_model_name
    sam = args.checkpoint
    return plm, sam


def train_loop(
    train_pairs,
    val_pairs,
    predictor,
    head: TeacherCorrectionHead,
    corrupter: MaskCorrupter,
    steps,
    lr,
    wd,
    batch_size,
    acc,
    out,
    name,
    max_dim,
    er_k,
    er_it,
    log_every,
    save_every,
    workers,
    seed_val,
    lam_bce,
    lam_dice,
    lam_focal,
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

    params = [
        p
        for p in (list(model.parameters()) + list(head.parameters()))
        if p.requires_grad
    ]
    assert len(params) > 0, "No trainable parameters found!"
    trainable_params_count = sum(p.numel() for p in params)
    print(f"[Init] Trainable parameters: {trainable_params_count / 1e6:.2f}M")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    total_updates = max(1, math.ceil(steps / max(1, acc)))
    if warmup_updates < 0:
        warmup_updates = max(0, int(0.03 * total_updates))
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
        persistent_workers=(workers > 0),
    )

    def infinite_loader():
        while True:
            for b in loader:
                yield b

    data_it = infinite_loader()
    os.makedirs(out, exist_ok=True)

    opt.zero_grad(set_to_none=True)
    update_idx = grad_acc = 0

    print("Correction teacher training started.", flush=True)
    t_loop = time.time()
    t_step = time.time()
    for step in range(1, steps + 1):
        if step <= 15 or step % log_every == 0:
            print(f"step {step} begin", flush=True)
        if step == 1:
            print("step 1: batch + Qwen/SAM forward (first step often 5–20 min)...", flush=True)
        batch = next(data_it)
        stage = curriculum_stage_from_step(step, steps)

        imgs, gts, prompts, img_paths, corrupt_masks = [], [], [], [], []
        meta_iou = []

        for img_p, msk_p, pr in batch:
            img, ann = read_img_mask(img_p, msk_p, max_dim)
            img = ensure_square_max_dim(img, max_dim, is_mask=False)
            ann = ensure_square_max_dim(ann, max_dim, is_mask=True)
            gt = erode(bin_mask(ann), er_k, er_it).astype(np.float32)
            gt = ensure_square_max_dim(gt, max_dim, is_mask=True)
            corrupted, meta = corrupter.corrupt((gt > 0.5).astype(np.uint8), stage)
            corrupted = ensure_square_max_dim(corrupted.astype(np.float32), max_dim, is_mask=True)
            meta_iou.append(meta["input_iou"])
            imgs.append(img)
            gts.append(np.clip(gt, 0.0, 1.0))
            corrupt_masks.append(np.clip(corrupted, 0.0, 1.0))
            prompts.append(pr or "segment")
            img_paths.append(img_p)

        if not imgs:
            continue

        canvas = (max_dim, max_dim)
        imgs = [ensure_square_max_dim(im, max_dim, is_mask=False) for im in imgs]
        gts = [ensure_square_max_dim(g, max_dim, is_mask=True) for g in gts]
        corrupt_masks = [ensure_square_max_dim(m, max_dim, is_mask=True) for m in corrupt_masks]
        if any(im.shape[:2] != canvas for im in imgs):
            raise RuntimeError("Batch image canvas mismatch after alignment")
        if any(g.shape[:2] != canvas for g in gts):
            raise RuntimeError("Batch GT canvas mismatch after alignment")

        predictor.set_image_batch(imgs)
        img_emb_b = predictor._features["image_embed"].to(dec_dev, dec_dtype)
        hi_b = [lvl.to(dec_dev, dec_dtype) for lvl in predictor._features["high_res_feats"]]
        B, _, Hm, Wm = img_emb_b.shape

        mask_t = resize_masks_continuous(corrupt_masks, max_dim, max_dim).to(
            device=dec_dev, dtype=dec_dtype
        )

        sp, dp = head(prompts, Hm, Wm, img_paths, mask_t)
        sp, dp = sp.to(dec_dev, dec_dtype), dp.to(dec_dev, dec_dtype)

        low, scores, *_ = model.sam_mask_decoder(
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

        bce = F.binary_cross_entropy_with_logits(logits, gt_t)
        dl = dice_loss_from_logits(logits, gt_t)
        fl = focal_loss_from_logits(logits, gt_t)
        loss = (lam_bce * bce + lam_dice * dl + lam_focal * fl) / acc
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
                print(
                    f"First optimizer update at step {step} "
                    f"(elapsed {time.time() - t_loop:.1f}s).",
                    flush=True,
                )

        step_sec = time.time() - t_step
        t_step = time.time()
        if step == 1 or step <= 5 or step % log_every == 0:
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                j = iou(preds, gt_t).item()
            mean_in = float(np.mean(meta_iou)) if meta_iou else 0.0
            print(
                f"step {step} stage {stage.value} upd {update_idx}: "
                f"loss {(lam_bce * bce + lam_dice * dl + lam_focal * fl).item():.4f} "
                f"IoU {j:.4f} corrupt_IoU {mean_in:.4f} gate {head.gate_value:.4f} "
                f"({step_sec:.1f}s/step)",
                flush=True,
            )
            writer.add_scalar("train/loss", (lam_bce * bce + lam_dice * dl + lam_focal * fl).item(), step)
            writer.add_scalar("train/iou", j, step)
            writer.add_scalar("train/corrupt_input_iou", mean_in, step)
            writer.add_scalar("train/mask_gate", head.gate_value, step)
            writer.add_scalar("train/stage", {"A": 0, "B": 1, "C": 2}[stage.value], step)

        if save_every and step % save_every == 0 and step < steps:
            torch.save({"model": model.state_dict()}, os.path.join(out, f"{name}_{step}.torch"))
            torch.save({"plm": plm.state_dict()}, os.path.join(out, f"{name}_plm_{step}.torch"))
            head_ckpt = os.path.join(out, f"{name}_head_{step}.torch")
            torch.save(
                {"head": head.state_dict(), "gate_value": head.gate_value},
                head_ckpt,
            )
            plm.save_lora(os.path.join(out, f"lora_plm_adapter_{step}"))
            print(f"Saved checkpoints at step {step} (head gate={head.gate_value:.4f})")

    # 100k 终点：仅 *_final / lora_plm_adapter_final（与 slurm_eval_trg 一致）
    torch.save({"model": model.state_dict()}, os.path.join(out, f"{name}_final.torch"))
    torch.save({"plm": plm.state_dict()}, os.path.join(out, f"{name}_plm_final.torch"))
    head_final = os.path.join(out, f"{name}_head_final.torch")
    torch.save({"head": head.state_dict(), "gate_value": head.gate_value}, head_final)
    print(f"Saved head (gate+mask_encoder) -> {head_final} gate={head.gate_value:.4f}")
    plm.save_lora(os.path.join(out, "lora_plm_adapter_final"))
    print(f"Correction teacher training complete at step {steps} (saved: 50000 + final @ {steps}).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--dataset-jsonl", default="dataset.jsonl")
    ap.add_argument("--model-cfg", default=default_sam_config_str())
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--plm-model-name", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--acc", type=int, default=3)
    ap.add_argument("--out", default="./ckpts/teacher_correction/author3b")
    ap.add_argument("--name", default="CorrectionTeacher_3B")
    ap.add_argument("--max-dim", type=int, default=1024)
    ap.add_argument("--er-k", type=int, default=5)
    ap.add_argument("--er-it", type=int, default=1)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument(
        "--save-every",
        type=int,
        default=50000,
        help="Intermediate checkpoints only (final step always saved as *_final and *_{steps}).",
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--warmup-updates", type=int, default=-1)
    ap.add_argument("--lam-bce", type=float, default=1.0)
    ap.add_argument("--lam-dice", type=float, default=0.25)
    ap.add_argument("--lam-focal", type=float, default=0.5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--plm-gradient-checkpointing", action="store_true")
    ap.add_argument("--qwen-image-size", type=int, default=256)
    ap.add_argument("--resume-sam-path", default=None)
    ap.add_argument("--resume-plm-path", default=None)
    ap.add_argument("--resume-lora-path", default=None)
    ap.add_argument("--resume-head-path", default=None)
    ap.add_argument(
        "--teacher-backbone",
        choices=("author3b", "teacher7b"),
        default="author3b",
        help="author3b=作者 Net-3B 权重；teacher7b=自训 7B Phase2（后续对比）",
    )
    ap.add_argument(
        "--init-preset",
        choices=("teacher7b_phase2", "author3b_phase2"),
        default=None,
        help="自动 resume 路径；默认随 --teacher-backbone 选择",
    )
    ap.add_argument("--plm-load-strict", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    apply_offline_env()
    configure_teacher_backbone(args)
    apply_init_preset(args)
    plm_path, sam_ckpt = resolve_teacher_plm_sam(args)
    print(
        f"[Init] teacher_backbone={args.teacher_backbone} init_preset={args.init_preset} "
        f"plm={plm_path} sam={sam_ckpt}"
    )
    print(
        f"[Init] resume_sam={args.resume_sam_path} resume_plm={args.resume_plm_path} "
        f"resume_lora={args.resume_lora_path} out={args.out} name={args.name} "
        f"batch={args.batch_size} acc={args.acc}"
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    seed(args.seed)

    pairs = load_samples_from_jsonl(
        args.data_dir,
        args.dataset_jsonl,
        num_samples=args.num_samples,
        repo_root=REPO_ROOT,
    )
    train_pairs, val_pairs = split(pairs, 0.01, args.seed)
    print(f"Train {len(train_pairs)} / val {len(val_pairs)}")

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
    head = TeacherCorrectionHead(plm, transformer_dim=C).to(args.device)

    if args.resume_sam_path and os.path.isfile(args.resume_sam_path):
        sd = torch.load(args.resume_sam_path, map_location="cpu")
        predictor.model.load_state_dict(sd["model"], strict=True)
        print(f"Loaded SAM from {args.resume_sam_path}")
    if args.resume_plm_path and os.path.isfile(args.resume_plm_path):
        sd = torch.load(args.resume_plm_path, map_location="cpu")
        incompatible = plm.load_state_dict(sd["plm"], strict=args.plm_load_strict)
        if not args.plm_load_strict:
            print(
                f"Loaded PLM (strict=False) from {args.resume_plm_path}; "
                f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
            )
        else:
            print(f"Loaded PLM from {args.resume_plm_path}")
    if args.resume_lora_path and plm.peft_enabled:
        plm.load_lora(args.resume_lora_path)
        print(f"Loaded LoRA from {args.resume_lora_path}")
    if args.resume_head_path and os.path.isfile(args.resume_head_path):
        sd = torch.load(args.resume_head_path, map_location="cpu")
        head.load_state_dict(sd["head"], strict=True)
        print(f"Loaded mask head from {args.resume_head_path}")

    if args.plm_gradient_checkpointing:
        plm.prepare_gradient_checkpointing_training()
        n_lora = sum(
            p.numel() for n, p in plm.backbone.named_parameters() if "lora_" in n and p.requires_grad
        )
        print(f"[Init] LoRA trainable params: {n_lora / 1e6:.2f}M (grad checkpointing hooks applied)")

    corrupter = MaskCorrupter(seed=args.seed)
    train_loop(
        train_pairs,
        val_pairs,
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
        er_k=args.er_k,
        er_it=args.er_it,
        log_every=args.log_every,
        save_every=args.save_every,
        workers=args.workers,
        seed_val=args.seed,
        lam_bce=args.lam_bce,
        lam_dice=args.lam_dice,
        lam_focal=args.lam_focal,
        min_lr=args.min_lr,
        warmup_updates=args.warmup_updates,
    )


if __name__ == "__main__":
    main()
