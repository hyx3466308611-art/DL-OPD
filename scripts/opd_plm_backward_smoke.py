#!/usr/bin/env python3
"""
Smoke test: OPD with frozen student SAM + trainable Qwen LoRA (backward on cuda:1).

Goal: verify dual V100 16GB can survive PLM backward in the real OPD step loop
(rollout + teacher KD + repair) without OOM / NaN.

Does NOT save checkpoints. Uses a temp dir for TensorBoard stubs only.

Usage (campus):
  cd /home/hanyuxi/ConverSeg-main
  source /home/hanyuxi/miniconda311/etc/profile.d/conda.sh && conda activate converseg
  unset CUDA_VISIBLE_DEVICES
  python scripts/opd_plm_backward_smoke.py \\
    --data-dir ./data/ConverSeg-Training-Data/converseg_stage2_data \\
    --dataset-jsonl stage2_mixed.jsonl
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

import torch

# Project root = parent of scripts/
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.language_adapter import LanguageAdapter
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from train import load_samples_from_jsonl, split
from train_opd import (
    DEFAULT_STAGE2,
    DEFAULT_TEACHER_CONTINUE,
    _cuda_empty_cache,
    _freeze_module,
    _load_plm_state,
    _load_sam_state,
    _log_cuda_env,
    _plm_prepare_for_training,
    _resolve_devices,
    _validate_cuda_devices,
    train_loop,
)
from utils.util_functions import seed


def _gpu_mem_line(label: str) -> str:
    if not torch.cuda.is_available():
        return f"{label}: (no cuda)"
    parts = [label]
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / (1024**2)
        reserved = torch.cuda.memory_reserved(i) / (1024**2)
        peak = torch.cuda.max_memory_allocated(i) / (1024**2)
        parts.append(f"cuda:{i} alloc={alloc:.0f}MB res={reserved:.0f}MB peak={peak:.0f}MB")
    return " | ".join(parts)


def _reset_peak_stats() -> None:
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)


def _resolve_student_ckpts(opd_dir: str, stage2_dir: str) -> tuple[str, str, str]:
    """Prefer OPD final weights (Phase-2 start), else Stage2."""
    opd_sam = os.path.join(opd_dir, "ConverSeg_opd_final.torch")
    if os.path.isfile(opd_sam):
        base = opd_dir
        tag = "ConverSeg_opd"
        print(f"[smoke] Student init from OPD: {opd_dir}", flush=True)
    else:
        base = stage2_dir
        tag = "ConverSeg_stage2"
        print(f"[smoke] Student init from Stage2: {stage2_dir}", flush=True)
    return (
        os.path.join(base, f"{tag}_final.torch"),
        os.path.join(base, f"{tag}_plm_final.torch"),
        os.path.join(base, "lora_plm_adapter_final"),
    )


def main() -> None:
    ap = argparse.ArgumentParser("OPD PLM backward smoke (freeze SAM, train Qwen LoRA)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--dataset-jsonl", default="stage2_mixed.jsonl")
    ap.add_argument("--num-samples", type=int, default=64, help="Subset for faster jsonl scan")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--acc", type=int, default=8, help="Gradient accumulation (8 optimizer steps in 30 iters)")
    ap.add_argument("--lr", type=float, default=1e-7, help="PLM-only smoke LR (keep tiny)")
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--model-cfg", default="configs/sam2/sam2_hiera_l.yaml")
    ap.add_argument("--checkpoint", default="/home/hanyuxi/ckpts_base/sam2_hiera_large.pt")
    ap.add_argument("--plm-model-name", default="./pretrain/Qwen2.5-VL-3B")
    ap.add_argument("--student-opd-dir", default="./ckpts/stage2_opd")
    ap.add_argument("--student-stage2-dir", default=DEFAULT_STAGE2)
    ap.add_argument("--teacher-dir", default=DEFAULT_TEACHER_CONTINUE)
    ap.add_argument("--sam-device", default="cuda:0")
    ap.add_argument("--plm-device", default="cuda:1")
    ap.add_argument("--teacher-plm-device", default="cpu")
    ap.add_argument("--max-dim", type=int, default=768)
    ap.add_argument("--qwen-image-size", type=int, default=96)
    ap.add_argument("--max-txt-len", type=int, default=128)
    ap.add_argument("--mask-scale", type=float, default=0.25)
    ap.add_argument("--lambda-kd", type=float, default=3.0)
    ap.add_argument("--kd-temperature", type=float, default=1.5)
    ap.add_argument("--bad-mask-mix", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    print("=" * 60, flush=True)
    print("OPD PLM backward smoke — freeze SAM, train Qwen LoRA", flush=True)
    print("=" * 60, flush=True)
    _log_cuda_env()

    sam_device, plm_device = _resolve_devices(args.sam_device, args.plm_device, "cuda")
    if sam_device.startswith("cuda") or plm_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for this smoke test")
        _validate_cuda_devices(sam_device, plm_device)
    if args.teacher_plm_device.startswith("cuda"):
        _validate_cuda_devices(sam_device, args.teacher_plm_device)

    if torch.cuda.device_count() < 2:
        print("WARNING: fewer than 2 GPUs visible; unset CUDA_VISIBLE_DEVICES?", flush=True)

    seed(args.seed)
    pairs = load_samples_from_jsonl(args.data_dir, args.dataset_jsonl, num_samples=args.num_samples)
    train_pairs, val_pairs = split(pairs, 0.01, args.seed)
    print(f"Smoke data: train={len(train_pairs)}, val={len(val_pairs)}", flush=True)

    s_sam, s_plm, s_lora = _resolve_student_ckpts(args.student_opd_dir, args.student_stage2_dir)
    t_sam = os.path.join(args.teacher_dir, "teacher_refine_continue_final.torch")
    t_plm = os.path.join(args.teacher_dir, "teacher_refine_continue_plm_final.torch")
    t_lora = os.path.join(args.teacher_dir, "lora_plm_adapter_final")
    for p in (s_sam, s_plm, t_sam, t_plm):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
    if not os.path.isdir(s_lora) or not os.path.isdir(t_lora):
        raise FileNotFoundError(f"LoRA dir missing: {s_lora} or {t_lora}")

    plm_dtype = torch.float32  # fp16 PLM backward NaNs on V100 after 1st opt step
    tmp_out = tempfile.mkdtemp(prefix="opd_plm_smoke_")
    print(f"Student PLM dtype: {plm_dtype} (forced float32 for smoke)", flush=True)
    print(f"Temp out (no ckpt save): {tmp_out}", flush=True)

    try:
        print(f"Building student SAM on {sam_device}...", flush=True)
        student_predictor = SAM2ImagePredictor(
            build_sam2(args.model_cfg, args.checkpoint, device=sam_device)
        )
        C = student_predictor.model.sam_mask_decoder.transformer_dim

        print(f"Building student PLM on {plm_device} (trainable LoRA)...", flush=True)
        student_plm = LanguageAdapter(
            model_name=args.plm_model_name,
            transformer_dim=C,
            n_sparse_tokens=0,
            use_dense_bias=True,
            use_lora=True,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            dtype=plm_dtype,
            device=plm_device,
            use_image_input=True,
            gradient_checkpointing=True,
            qwen_image_size=args.qwen_image_size,
            max_txt_len=args.max_txt_len,
        ).to(plm_device)

        _load_sam_state(student_predictor.model, s_sam)
        _load_plm_state(student_plm, s_plm, s_lora)
        _plm_prepare_for_training(student_plm, True)

        trainable = sum(p.numel() for p in student_plm.parameters() if p.requires_grad)
        total = sum(p.numel() for p in student_plm.parameters())
        print(f"Student PLM trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)", flush=True)

        print(f"Building frozen teacher SAM on {sam_device}...", flush=True)
        teacher_sam = build_sam2(args.model_cfg, args.checkpoint, device=sam_device)
        _load_sam_state(teacher_sam, t_sam)
        _freeze_module(teacher_sam)

        teacher_plm_dtype = torch.float32 if args.teacher_plm_device == "cpu" else _plm_dtype(args.teacher_plm_device)
        print(f"Building frozen teacher PLM on {args.teacher_plm_device}...", flush=True)
        teacher_plm = LanguageAdapter(
            model_name=args.plm_model_name,
            transformer_dim=C,
            n_sparse_tokens=0,
            use_dense_bias=True,
            use_lora=True,
            lora_r=16,
            lora_alpha=32,
            dtype=teacher_plm_dtype,
            device=args.teacher_plm_device,
            use_image_input=True,
            gradient_checkpointing=False,
            qwen_image_size=args.qwen_image_size,
            max_txt_len=args.max_txt_len,
        ).to(args.teacher_plm_device)
        _load_plm_state(teacher_plm, t_plm, t_lora)
        _freeze_module(teacher_plm)

        _cuda_empty_cache(sam_device, plm_device)
        _reset_peak_stats()
        print(_gpu_mem_line("After model load"), flush=True)

        print(
            f"\nRunning {args.steps} OPD steps: freeze_student_sam=True, train_student_plm=True\n",
            flush=True,
        )
        train_loop(
            train_pairs,
            val_pairs,
            student_predictor,
            student_plm,
            teacher_sam,
            teacher_plm,
            steps=args.steps,
            lr=args.lr,
            batch_size=1,
            acc=args.acc,
            out=tmp_out,
            name="smoke",
            max_dim=args.max_dim,
            log_every=max(1, args.steps // 6),
            save_every=0,
            val_every=0,
            val_count=4,
            val_dir=os.path.join(tmp_out, "val"),
            mask_scale=args.mask_scale,
            lambda_kd=args.lambda_kd,
            kd_temperature=args.kd_temperature,
            max_grad_norm=args.max_grad_norm,
            freeze_student_plm=False,
            freeze_student_sam=True,
            positive_only=True,
            bad_mask_mix=args.bad_mask_mix,
            save_checkpoints=False,
            run_final_val=False,
        )

        print(_gpu_mem_line("After smoke steps"), flush=True)
        print("\n" + "=" * 60, flush=True)
        print("PASS: PLM backward smoke finished (no OOM / NaN abort).", flush=True)
        print("=" * 60, flush=True)

    except RuntimeError as e:
        print(_gpu_mem_line("At failure"), flush=True)
        msg = str(e).lower()
        if "out of memory" in msg or "oom" in msg:
            print("\nFAIL (OOM): dual V100 cannot fit PLM backward in this OPD layout.", flush=True)
        elif "non-finite" in msg or "nan" in msg:
            print(
                "\nFAIL (NaN): not OOM — PLM backward unstable (try float32 + lr=1e-7; sync latest script).",
                flush=True,
            )
        else:
            print(f"\nFAIL: {e}", flush=True)
        raise SystemExit(1) from e
    finally:
        shutil.rmtree(tmp_out, ignore_errors=True)


if __name__ == "__main__":
    main()
