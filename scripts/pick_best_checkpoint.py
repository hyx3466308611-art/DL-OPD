#!/usr/bin/env python3
"""
Scan step checkpoints under --out, run quick val mean IoU, pick the best triple
(SAM + PLM + LoRA). Optional: promote copies to *_best* and prune other steps.

Example (campus V100):
  cd /home/hanyuxi/ConverSeg-main
  export CUDA_VISIBLE_DEVICES=0,1
  python scripts/pick_best_checkpoint.py \\
    --out ./ckpts/stage1_full --name ConVerSeg_stage1 \\
    --data-dir ./data/ConverSeg-Training-Data/converseg_stage2_data \\
    --dataset-jsonl open_vocabulary_regions_data/open_vocabulary_regions_data.jsonl \\
    --val-count 24 --promote-best
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.language_adapter import LanguageAdapter
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from train import (
    _select_val_subset,
    _validate_once,
    load_samples_from_jsonl,
    split,
)
from train_teacher import (
    _default_max_dim,
    _default_max_txt_len,
    _default_plm_gradient_checkpointing,
    _default_qwen_image_size,
    _plm_dtype,
    _plm_prepare_for_training,
    _resolve_devices,
    _validate_cuda_devices,
)
from utils.util_functions import seed


def _discover_steps(out: str, name: str) -> list[int]:
    steps = []
    pat = re.compile(rf"^{re.escape(name)}_(\d+)\.torch$")
    for path in glob.glob(os.path.join(out, f"{name}_*.torch")):
        base = os.path.basename(path)
        if "_plm_" in base or base.endswith("_final.torch"):
            continue
        m = pat.match(base)
        if m:
            steps.append(int(m.group(1)))
    return sorted(set(steps))


def _ckpt_triple(out: str, name: str, step: int) -> tuple[str, str, str]:
    sam = os.path.join(out, f"{name}_{step}.torch")
    plm = os.path.join(out, f"{name}_plm_{step}.torch")
    lora = os.path.join(out, f"lora_plm_adapter_{step}")
    return sam, plm, lora


def _verify_loadable(sam_p: str, plm_p: str) -> tuple[bool, str]:
    try:
        torch.load(sam_p, map_location="cpu", weights_only=False)
    except Exception as e:
        return False, f"SAM: {e}"
    try:
        torch.load(plm_p, map_location="cpu", weights_only=False)
    except Exception as e:
        return False, f"PLM: {e}"
    return True, "ok"


def _build_models(args, sam_device: str, plm_device: str):
    max_dim = args.max_dim if args.max_dim else _default_max_dim(plm_device)
    qwen_image_size = (
        args.qwen_image_size
        if args.qwen_image_size is not None
        else _default_qwen_image_size(plm_device)
    )
    max_txt_len = (
        args.max_txt_len if args.max_txt_len is not None else _default_max_txt_len(plm_device)
    )
    plm_dtype = _plm_dtype(plm_device)
    if plm_device.startswith("cuda"):
        plm_dtype = torch.float32

    predictor = SAM2ImagePredictor(
        build_sam2(args.model_cfg, args.checkpoint, device=sam_device)
    )
    c = predictor.model.sam_mask_decoder.transformer_dim
    plm = LanguageAdapter(
        model_name=args.plm_model_name,
        transformer_dim=c,
        n_sparse_tokens=0,
        use_dense_bias=True,
        use_lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        dtype=plm_dtype,
        device=plm_device,
        use_image_input=True,
        qwen_image_size=qwen_image_size,
        max_txt_len=max_txt_len,
        gradient_checkpointing=_default_plm_gradient_checkpointing(
            plm_device, plm_dtype, sam_device
        ),
    ).to(plm_device)
    _plm_prepare_for_training(plm, False)
    return predictor, plm, max_dim, plm_dtype


def _load_step(predictor, plm, out: str, name: str, step: int) -> None:
    sam_p, plm_p, lora_p = _ckpt_triple(out, name, step)
    sd = torch.load(sam_p, map_location="cpu", weights_only=False)
    predictor.model.load_state_dict(sd["model"], strict=True)
    sd = torch.load(plm_p, map_location="cpu", weights_only=False)
    plm.load_state_dict(sd["plm"], strict=True)
    if plm.peft_enabled:
        plm.load_lora(lora_p)


def main():
    ap = argparse.ArgumentParser("Pick best step checkpoint by val mean IoU")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="ConVerSeg_stage1")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--dataset-jsonl", default="open_vocabulary_regions_data/open_vocabulary_regions_data.jsonl")
    ap.add_argument("--model-cfg", default="configs/sam2/sam2_hiera_l.yaml")
    ap.add_argument("--checkpoint", default="/home/hanyuxi/ckpts_base/sam2_hiera_large.pt")
    ap.add_argument("--plm-model-name", default="./pretrain/Qwen2.5-VL-3B")
    ap.add_argument("--sam-device", default="cuda:0")
    ap.add_argument("--plm-device", default="cuda:1")
    ap.add_argument("--max-dim", type=int, default=768)
    ap.add_argument("--qwen-image-size", type=int, default=96)
    ap.add_argument("--max-txt-len", type=int, default=128)
    ap.add_argument("--val-count", type=int, default=24, help="Fixed val samples (same subset for all steps)")
    ap.add_argument("--val-dir", default=None, help="Default: <out>/pick_best_val")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--promote-best",
        action="store_true",
        help="Copy best triple to <name>_best.torch / _plm_best.torch / lora_plm_adapter_best",
    )
    ap.add_argument(
        "--promote-meta-only",
        action="store_true",
        help="Only write <name>_best_step.txt (no copy; use when disk is full)",
    )
    ap.add_argument(
        "--prune-other-steps",
        action="store_true",
        help="Delete non-best step checkpoints (SAM+PLM+LoRA dirs) after promote",
    )
    args = ap.parse_args()

    val_dir = args.val_dir or os.path.join(args.out, "pick_best_val")
    seed(args.seed)

    sam_device, plm_device = _resolve_devices(args.sam_device, args.plm_device, "cuda")
    if sam_device.startswith("cuda") or plm_device.startswith("cuda"):
        _validate_cuda_devices(sam_device, plm_device)

    steps = _discover_steps(args.out, args.name)
    if not steps:
        print(f"No step checkpoints found in {args.out} matching {args.name}_<step>.torch")
        sys.exit(1)

    print(f"Found steps: {steps}")
    valid_steps = []
    for s in steps:
        sam_p, plm_p, lora_p = _ckpt_triple(args.out, args.name, s)
        if not os.path.isfile(sam_p):
            print(f"  skip {s}: missing SAM")
            continue
        if not os.path.isfile(plm_p):
            print(f"  skip {s}: missing PLM (common after disk-full save)")
            continue
        if not (os.path.isdir(lora_p) or os.path.isfile(lora_p)):
            print(f"  skip {s}: missing LoRA {lora_p}")
            continue
        ok, msg = _verify_loadable(sam_p, plm_p)
        if not ok:
            print(f"  skip {s}: corrupt — {msg}")
            continue
        valid_steps.append(s)

    if not valid_steps:
        print("No complete loadable checkpoint triples.")
        sys.exit(2)

    pairs = load_samples_from_jsonl(args.data_dir, args.dataset_jsonl)
    _, val_pairs = split(pairs, 0.01, args.seed)
    val_subset = _select_val_subset(val_pairs, args.max_dim, args.val_count)
    print(f"Val subset: {len(val_subset)} samples (fixed seed {args.seed})")

    predictor, plm, max_dim, plm_dtype = _build_models(args, sam_device, plm_device)
    model = predictor.model
    dec_dev = next(model.sam_mask_decoder.parameters()).device
    dec_dtype = next(model.sam_mask_decoder.parameters()).dtype
    image_pe_const = model.sam_prompt_encoder.get_dense_pe().to(dec_dev, dec_dtype)

    results: list[tuple[int, float]] = []
    for s in valid_steps:
        print(f"\n=== Evaluating step {s} ===", flush=True)
        _load_step(predictor, plm, args.out, args.name, s)
        miou = _validate_once(
            predictor,
            plm,
            val_subset,
            image_pe_const,
            dec_dev,
            dec_dtype,
            max_dim,
            er_k=5,
            er_it=1,
            save_dir=os.path.join(val_dir, f"step_{s}"),
            step=s,
        )
        results.append((s, miou))

    results.sort(key=lambda x: x[1], reverse=True)
    print("\n========== Ranking ==========")
    for rank, (s, miou) in enumerate(results, 1):
        print(f"  #{rank}  step {s:6d}  mean IoU {miou:.4f}")

    best_step, best_miou = results[0]
    print(f"\nBEST: step {best_step}  mean IoU {best_miou:.4f}")

    sam_b, plm_b, lora_b = _ckpt_triple(args.out, args.name, best_step)
    print("\nResume example (continue to 100k, save every 10k):")
    print(
        f"  --resume-sam-path {sam_b} \\\n"
        f"  --resume-plm-path {plm_b} \\\n"
        f"  --resume-lora-path {lora_b} \\\n"
        f"  --start-step {best_step + 1} --save-every 10000"
    )

    meta = os.path.join(args.out, f"{args.name}_best_step.txt")
    with open(meta, "w") as f:
        f.write(f"step={best_step}\nmean_iou={best_miou:.6f}\n")
        f.write(f"sam={sam_b}\nplm={plm_b}\nlora={lora_b}\n")
    print(f"\nWrote meta -> {meta}")

    if args.promote_meta_only and not args.promote_best:
        print("(promote-meta-only: use resume paths above; no duplicate checkpoint files)")

    if args.promote_best:
        out = args.out
        name = args.name
        dst_sam = os.path.join(out, f"{name}_best.torch")
        dst_plm = os.path.join(out, f"{name}_plm_best.torch")
        dst_lora = os.path.join(out, f"lora_plm_adapter_best")
        shutil.copy2(sam_b, dst_sam)
        shutil.copy2(plm_b, dst_plm)
        if os.path.isdir(lora_b):
            if os.path.exists(dst_lora):
                shutil.rmtree(dst_lora)
            shutil.copytree(lora_b, dst_lora)
        else:
            shutil.copy2(lora_b, dst_lora)
        print(f"Promoted best copies -> {dst_sam}, {dst_plm}, {dst_lora}")

    if args.prune_other_steps:
        keep = {best_step}
        for s in valid_steps:
            if s in keep:
                continue
            sam_p, plm_p, lora_p = _ckpt_triple(args.out, args.name, s)
            for p in (sam_p, plm_p):
                if os.path.isfile(p):
                    os.remove(p)
                    print(f"removed {p}")
            if os.path.isdir(lora_p):
                shutil.rmtree(lora_p)
                print(f"removed {lora_p}")
        print("Prune done (kept best step only among evaluated).")


if __name__ == "__main__":
    main()
