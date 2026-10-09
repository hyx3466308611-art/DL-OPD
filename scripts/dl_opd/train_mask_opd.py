#!/usr/bin/env python3
"""MVP-A / Stage-2: Mask-Policy OPD training with frozen correction teacher.

Usage (from repo root):
  python -u scripts/dl_opd/train_mask_opd.py --config configs/dl_opd/dl_opd.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from converseg_paths import prepare_python_import_path

prepare_python_import_path()

from models.language_adapter import LanguageAdapter
from models.teacher_correction_model import TeacherCorrectionHead
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils.util_functions import bin_mask, erode, read_img_mask, seed as set_seed

from dl_opd.gate_teacher import GateConfig, GateInputs, decide_teacher_gate, summarize_gate_batch
from dl_opd.losses import LossWeights, assemble_opd_losses
from dl_opd.rollout import (
    decode_logits_to_prob,
    empty_mask_batch,
    soft_iou_torch,
    teacher_confidence_from_prob,
)
from dl_opd.teacher_cache import CacheEntry, TeacherCache


def _load_yaml(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    json_sidecar = p.with_suffix(".json")
    try:
        import yaml

        if p.is_file():
            with p.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except ImportError:
        pass
    if json_sidecar.is_file():
        print(f"[cfg] loading {json_sidecar}")
        return json.loads(json_sidecar.read_text(encoding="utf-8"))
    if p.is_file() and p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    print("[warn] config not loaded; use CLI overrides")
    return {}


def load_pairs(data_dir: str, jsonl: str, limit: int | None = None):
    path = os.path.join(data_dir, jsonl)
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            ip = os.path.join(data_dir, item["image"])
            mp = os.path.join(data_dir, item.get("mask_merged") or item.get("mask"))
            pr = item.get("prompt", "")
            concept = item.get("concept", "entities")
            if os.path.isfile(ip) and os.path.isfile(mp):
                out.append((ip, mp, pr, concept, item.get("id", ip)))
    if limit:
        out = out[:limit]
    if not out:
        raise RuntimeError(f"No samples in {path}")
    print(f"[mask_opd] loaded {len(out)} from {path}")
    return out


class PairDS(Dataset):
    def __init__(self, pairs):
        self.pairs = list(pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]


def collate_keep(batch):
    return batch


def ensure_square(arr, max_dim, is_mask=False):
    """Match train_teacher_correction canvas alignment."""
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
    pad_w, pad_h = max_dim - new_w, max_dim - new_h
    left, top = pad_w // 2, pad_h // 2
    right, bottom = pad_w - left, pad_h - top
    out = cv2.copyMakeBorder(
        resized, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=pad_val
    )
    return np.ascontiguousarray(out)


def build_head(plm, transformer_dim: int) -> TeacherCorrectionHead:
    return TeacherCorrectionHead(plm, transformer_dim)


def load_checkpoint_pair(predictor, plm, head, sam_path, plm_path, lora_path, head_path=None):
    if sam_path and os.path.isfile(sam_path):
        sd = torch.load(sam_path, map_location="cpu")
        key = "model" if isinstance(sd, dict) and "model" in sd else None
        predictor.model.load_state_dict(sd[key] if key else sd, strict=True)
        print(f"[load] SAM <- {sam_path}")
    if plm_path and os.path.isfile(plm_path):
        sd = torch.load(plm_path, map_location="cpu")
        plm.load_state_dict(sd if not isinstance(sd, dict) or "model" not in sd else sd.get("model", sd), strict=False)
        print(f"[load] PLM <- {plm_path}")
    if lora_path and os.path.isdir(lora_path) and getattr(plm, "peft_enabled", False):
        plm.load_lora(lora_path)
        print(f"[load] LoRA <- {lora_path}")
    if head_path and os.path.isfile(head_path):
        sd = torch.load(head_path, map_location="cpu")
        state = sd["head"] if isinstance(sd, dict) and "head" in sd else sd
        head.load_state_dict(state, strict=False)
        print(f"[load] head <- {head_path}")


@torch.no_grad()
def forward_head_decode(predictor, model, head, prompts, paths, mask_b1hw, image_pe, dec_dev, dec_dtype):
    """Shared image embeds already set on predictor for this batch."""
    img_emb_b = predictor._features["image_embed"].to(dec_dev, dec_dtype)
    hi_b = [lvl.to(dec_dev, dec_dtype) for lvl in predictor._features["high_res_feats"]]
    _, _, Hm, Wm = img_emb_b.shape
    sp, dp = head(prompts, Hm, Wm, paths, mask_b1hw)
    sp, dp = sp.to(dec_dev, dec_dtype), dp.to(dec_dev, dec_dtype)
    low, scores, *_ = model.sam_mask_decoder(
        image_embeddings=img_emb_b,
        image_pe=image_pe,
        sparse_prompt_embeddings=sp,
        dense_prompt_embeddings=dp,
        multimask_output=False,
        repeat_image=False,
        high_res_features=hi_b,
    )
    return low[:, 0], scores


def train(args):
    cfg = _load_yaml(args.config)
    train_cfg = cfg.get("train", {})
    gate_cfg_d = cfg.get("gate", {})
    loss_cfg = cfg.get("loss", {})

    device = args.device or train_cfg.get("device", "cuda")
    max_dim = int(args.max_dim or train_cfg.get("max_dim", 1024))
    batch_size = int(args.batch_size or train_cfg.get("batch_size", 1))
    acc = int(args.acc or train_cfg.get("acc", 6))
    steps = int(args.steps or train_cfg.get("steps", 2000))
    lr = float(args.lr or train_cfg.get("lr", 2e-5))
    set_seed(int(args.seed or train_cfg.get("seed", 1337)))

    data_dir = args.data_dir or cfg.get("data", {}).get("data_dir", str(ROOT / "data"))
    jsonl = args.dataset_jsonl or cfg.get("data", {}).get("dataset_jsonl", "phase2/phase2_mixed.jsonl")
    pairs = load_pairs(data_dir, jsonl, limit=args.limit)

    student_sam = args.student_sam or cfg["student"]["sam_ckpt"]
    student_plm_ckpt = args.student_plm or cfg["student"].get("plm_ckpt")
    student_lora = args.student_lora or cfg["student"].get("lora")
    student_plm_name = args.student_plm_name or cfg["student"]["plm_model"]
    student_head = args.student_head or cfg["student"].get("head_ckpt")

    teacher_sam = args.teacher_sam or cfg["teacher"]["sam_ckpt"]
    teacher_plm_ckpt = args.teacher_plm or cfg["teacher"].get("plm_ckpt")
    teacher_lora = args.teacher_lora or cfg["teacher"].get("lora")
    teacher_plm_name = args.teacher_plm_name or cfg["teacher"]["plm_model"]
    teacher_head = args.teacher_head or cfg["teacher"]["head_ckpt"]
    teacher_base = args.teacher_base_sam or cfg["teacher"].get("base_sam", teacher_sam)

    out = Path(args.out or cfg.get("out", "ckpts/dl_opd/mask_opd"))
    out.mkdir(parents=True, exist_ok=True)

    # --- build student (always init from raw SAM2, then load ConVerSeg weights) ---
    base_sam = args.base_sam or str(ROOT / "student/sam2_hiera_large.pt")
    student_sam_ckpt = student_sam if student_sam and str(student_sam).endswith(".torch") else cfg["student"].get("sam_ckpt")
    student_pred = SAM2ImagePredictor(build_sam2(args.model_cfg, base_sam, device=device))
    C = student_pred.model.sam_mask_decoder.transformer_dim
    student_plm = LanguageAdapter(
        model_name=student_plm_name,
        transformer_dim=C,
        device=device,
        use_dense_bias=True,
        use_lora=True,
    )
    student_head_m = build_head(student_plm, C)
    load_checkpoint_pair(
        student_pred,
        student_plm,
        student_head_m,
        student_sam_ckpt,
        student_plm_ckpt,
        student_lora,
        student_head,
    )

    # freeze image encoder
    for p in student_pred.model.image_encoder.parameters():
        p.requires_grad = False
    student_pred.model.sam_mask_decoder.train()
    student_head_m.train()

    # --- frozen teacher ---
    teacher_base_ckpt = teacher_base if teacher_base and os.path.isfile(str(teacher_base)) else str(ROOT / "teacher/sam2_hiera_large.pt")
    teacher_pred = SAM2ImagePredictor(build_sam2(args.model_cfg, teacher_base_ckpt, device=device))
    Ct = teacher_pred.model.sam_mask_decoder.transformer_dim
    teacher_plm = LanguageAdapter(
        model_name=teacher_plm_name,
        transformer_dim=Ct,
        device=device,
        use_dense_bias=True,
        use_lora=True,
    )
    teacher_head_m = build_head(teacher_plm, Ct)
    load_checkpoint_pair(
        teacher_pred,
        teacher_plm,
        teacher_head_m,
        teacher_sam,
        teacher_plm_ckpt,
        teacher_lora,
        teacher_head,
    )
    teacher_pred.model.eval()
    teacher_head_m.eval()
    for p in teacher_pred.model.parameters():
        p.requires_grad = False
    for p in teacher_head_m.parameters():
        p.requires_grad = False
    for p in teacher_plm.parameters():
        p.requires_grad = False

    gate_cfg = GateConfig(**{k: gate_cfg_d[k] for k in GateConfig.__dataclass_fields__ if k in gate_cfg_d})
    loss_w = LossWeights(**{k: loss_cfg[k] for k in LossWeights.__dataclass_fields__ if k in loss_cfg})

    params = [p for p in list(student_pred.model.sam_mask_decoder.parameters()) + list(student_head_m.parameters()) if p.requires_grad]
    # LoRA params
    params += [p for p in student_plm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=float(train_cfg.get("wd", 0.05)))
    total_updates = max(1, math.ceil(steps / max(1, acc)))
    warmup = max(0, int(0.03 * total_updates))
    if warmup > 0:
        sched = SequentialLR(
            opt,
            [
                LinearLR(opt, start_factor=1e-8, total_iters=warmup),
                CosineAnnealingLR(opt, T_max=max(1, total_updates - warmup), eta_min=1e-7),
            ],
            milestones=[warmup],
        )
    else:
        sched = CosineAnnealingLR(opt, T_max=total_updates, eta_min=1e-7)

    loader = DataLoader(
        PairDS(pairs),
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(train_cfg.get("workers", 2)),
        collate_fn=collate_keep,
        drop_last=False,
    )

    def infinite():
        while True:
            for b in loader:
                yield b

    it = infinite()
    writer = SummaryWriter(log_dir=str(out / "tb"))
    cache = TeacherCache(ttl_epochs=int(cfg.get("cache", {}).get("ttl_epochs", 2))) if cfg.get("cache", {}).get("enabled", True) else None

    student_ckpt_id = str(cfg["student"].get("sam_ckpt", "student"))
    teacher_ckpt_id = str(cfg["teacher"].get("head_ckpt", "teacher"))

    s_model = student_pred.model
    t_model = teacher_pred.model
    s_dec_dev = next(s_model.sam_mask_decoder.parameters()).device
    s_dec_dtype = next(s_model.sam_mask_decoder.parameters()).dtype
    s_pe = s_model.sam_prompt_encoder.get_dense_pe().to(s_dec_dev, s_dec_dtype)
    t_dec_dev = next(t_model.sam_mask_decoder.parameters()).device
    t_dec_dtype = next(t_model.sam_mask_decoder.parameters()).dtype
    t_pe = t_model.sam_prompt_encoder.get_dense_pe().to(t_dec_dev, t_dec_dtype)

    er_k, er_it = 5, 1
    opt.zero_grad(set_to_none=True)
    print("[mask_opd] training started", flush=True)
    t0 = time.time()

    for step in range(1, steps + 1):
        batch = next(it)
        imgs, gts, prompts, paths, ids, concepts = [], [], [], [], [], []
        for ip, mp, pr, concept, sid in batch:
            img, ann = read_img_mask(ip, mp, max_dim)
            img = ensure_square(img, max_dim, False)
            ann = ensure_square(ann, max_dim, True)
            gt = erode(bin_mask(ann), er_k, er_it).astype(np.float32)
            gt = ensure_square(gt, max_dim, True)
            imgs.append(img)
            gts.append(np.clip(gt, 0, 1))
            prompts.append(pr or "segment")
            paths.append(ip)
            ids.append(str(sid))
            concepts.append(concept)

        B = len(imgs)
        student_pred.set_image_batch(imgs)
        # also set teacher images (same pixels)
        teacher_pred.set_image_batch(imgs)

        gt_t = torch.from_numpy(np.stack(gts)[:, None]).to(s_dec_dev, s_dec_dtype)
        empty_state = empty_mask_batch(B, max_dim, max_dim, s_dec_dev, s_dec_dtype)

        # qs1 from empty state
        logits1, _ = forward_head_decode(
            student_pred, s_model, student_head_m, prompts, paths, empty_state, s_pe, s_dec_dev, s_dec_dtype
        )
        qs1 = decode_logits_to_prob(logits1, max_dim, max_dim)

        # teacher on sg(qs1)
        qs1_sg = qs1.detach()
        qt2 = None
        teacher_conf = None
        if cache is not None:
            # try cache per-sample; for simplicity recompute batch if any miss
            miss = False
            cached = []
            for i in range(B):
                key = cache.make_key(
                    image_id=ids[i],
                    prompt=prompts[i],
                    state_source="current",
                    student_ckpt=student_ckpt_id,
                    teacher_ckpt=teacher_ckpt_id,
                    state_mask=qs1_sg[i, 0].cpu().numpy(),
                )
                ent = cache.get(key, current_epoch=step // max(1, len(loader)))
                if ent is None:
                    miss = True
                    break
                cached.append(ent)
            if not miss:
                qt2 = torch.from_numpy(np.stack([e.teacher_mask for e in cached])[:, None]).to(
                    s_dec_dev, s_dec_dtype
                )
                teacher_conf = torch.tensor([e.teacher_confidence for e in cached], device=s_dec_dev)

        if qt2 is None:
            with torch.no_grad():
                t_mask = qs1_sg.to(t_dec_dev, t_dec_dtype)
                t_logits, _ = forward_head_decode(
                    teacher_pred, t_model, teacher_head_m, prompts, paths, t_mask, t_pe, t_dec_dev, t_dec_dtype
                )
                qt2 = decode_logits_to_prob(t_logits, max_dim, max_dim).to(s_dec_dev, s_dec_dtype)
                teacher_conf = teacher_confidence_from_prob(qt2)
            if cache is not None:
                for i in range(B):
                    key = cache.make_key(
                        image_id=ids[i],
                        prompt=prompts[i],
                        state_source="current",
                        student_ckpt=student_ckpt_id,
                        teacher_ckpt=teacher_ckpt_id,
                        state_mask=qs1_sg[i, 0].cpu().numpy(),
                    )
                    cache.put(
                        CacheEntry(
                            key=key,
                            teacher_mask=qt2[i, 0].detach().cpu().numpy(),
                            teacher_confidence=float(teacher_conf[i].item()),
                            created_epoch=step // max(1, len(loader)),
                            created_cycle=0,
                        )
                    )

        # qs2 refine
        logits2, _ = forward_head_decode(
            student_pred, s_model, student_head_m, prompts, paths, qs1_sg, s_pe, s_dec_dev, s_dec_dtype
        )
        qs2 = decode_logits_to_prob(logits2, max_dim, max_dim)

        i_s = soft_iou_torch(qs1_sg, gt_t)
        i_t = soft_iou_torch(qt2, gt_t)
        empty_mask = gt_t.flatten(1).sum(1) < 1e-4

        decisions = []
        g_kd = []
        identity = []
        improve = []
        for i in range(B):
            d = decide_teacher_gate(
                GateInputs(
                    gt_valid=True,
                    gt_empty=bool(empty_mask[i].item()),
                    input_iou=float(i_s[i].item()),
                    student_iou=float(i_s[i].item()),
                    teacher_iou=float(i_t[i].item()),
                    teacher_confidence=float(teacher_conf[i].item()),
                    recovery_gain=float((i_t[i] - i_s[i]).item()),
                ),
                gate_cfg,
            )
            decisions.append(d)
            g_kd.append(d.g_kd)
            identity.append(d.apply_identity)
            improve.append(float(i_t[i].item()) > float(i_s[i].item()) + gate_cfg.delta)

        g_kd_t = torch.tensor(g_kd, device=s_dec_dev, dtype=s_dec_dtype)
        identity_t = torch.tensor(identity, device=s_dec_dev)
        improve_t = torch.tensor(improve, device=s_dec_dev)

        bundle = assemble_opd_losses(
            qs1=qs1,
            qs2=qs2,
            gt=gt_t,
            qt2=qt2,
            g_kd=g_kd_t,
            empty_mask=empty_mask,
            identity_mask=identity_t,
            improve_mask=improve_t,
            weights=loss_w,
            pred_is_logit=False,
        )
        loss = bundle.total / max(1, acc)
        loss.backward()

        if step % acc == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

        if step <= 5 or step % int(train_cfg.get("log_every", 50)) == 0:
            gate_sum = summarize_gate_batch(decisions)
            print(
                f"step {step}/{steps} loss={bundle.total.item():.4f} "
                f"sup={bundle.l_sup.item():.4f} opd={bundle.l_opd.item():.4f} "
                f"kd_rate={gate_sum['kd_rate']:.2f} elapsed={time.time()-t0:.0f}s",
                flush=True,
            )
            for k, v in bundle.as_log_dict().items():
                writer.add_scalar(f"train/{k}", v, step)
            writer.add_scalar("train/kd_rate", gate_sum["kd_rate"], step)
            if cache is not None:
                for k, v in cache.stats().items():
                    if isinstance(v, (int, float)):
                        writer.add_scalar(f"cache/{k}", v, step)

        if step % int(train_cfg.get("save_every", 1000)) == 0 or step == steps:
            ck = out / f"mask_opd_step{step}"
            ck.mkdir(parents=True, exist_ok=True)
            torch.save({"model": s_model.state_dict()}, ck / "sam.torch")
            torch.save(student_plm.state_dict(), ck / "plm.torch")
            torch.save({"head": student_head_m.state_dict()}, ck / "head.torch")
            if getattr(student_plm, "peft_enabled", False):
                student_plm.save_lora(str(ck / "lora_plm_adapter"))
            print(f"[save] {ck}", flush=True)

    # final
    final = out / "mask_opd_final"
    final.mkdir(parents=True, exist_ok=True)
    torch.save({"model": s_model.state_dict()}, final / "sam.torch")
    torch.save(student_plm.state_dict(), final / "plm.torch")
    torch.save({"head": student_head_m.state_dict()}, final / "head.torch")
    if getattr(student_plm, "peft_enabled", False):
        student_plm.save_lora(str(final / "lora_plm_adapter"))
    meta = {"steps": steps, "gate": gate_cfg.__dict__, "loss": loss_w.__dict__}
    (final / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[mask_opd] done → {final}")


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=str(ROOT / "configs/dl_opd/dl_opd.yaml"))
    p.add_argument("--model_cfg", type=str, default="configs/sam2/sam2_hiera_l.yaml")
    p.add_argument("--base_sam", type=str, default=str(ROOT / "student/sam2_hiera_large.pt"))
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--dataset_jsonl", type=str, default=None)
    p.add_argument("--student_sam", type=str, default=None)
    p.add_argument("--student_plm", type=str, default=None)
    p.add_argument("--student_lora", type=str, default=None)
    p.add_argument("--student_plm_name", type=str, default=None)
    p.add_argument("--student_head", type=str, default=None)
    p.add_argument("--teacher_sam", type=str, default=None)
    p.add_argument("--teacher_plm", type=str, default=None)
    p.add_argument("--teacher_lora", type=str, default=None)
    p.add_argument("--teacher_plm_name", type=str, default=None)
    p.add_argument("--teacher_head", type=str, default=None)
    p.add_argument("--teacher_base_sam", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--acc", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max_dim", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
