#!/usr/bin/env python3
"""
eval.py — SAM2 + PLM evaluation on ConVerSeg / JSON / optional HF.

Reports overall gIoU/cIoU and (when `concept` is present) Table-1 style
per-concept breakdown: entities / spatial / relations / affordances / physics.
"""
from __future__ import annotations

import os
import io
import json
import argparse
import logging
import contextlib
import tempfile
from pathlib import Path
from typing import Tuple

import cv2
import torch
import numpy as np
from tqdm.auto import tqdm

from PIL import Image, ImageFile, ImageFilter, ImageChops, ImageDraw

ImageFile.LOAD_TRUNCATED_IMAGES = True  # robust against slightly broken files

# HF datasets
try:
    from datasets import load_dataset, Image as HFImage
except Exception as e:
    load_dataset = None
    HFImage = None
    _HF_IMPORT_ERROR = e
else:
    _HF_IMPORT_ERROR = None


from converseg_paths import (
    apply_offline_env,
    default_plm_dir_str,
    default_sam_ckpt_str,
    default_sam_config_str,
    EVAL_HUMAN_JSONL,
    EVAL_SAM_SEEDED_JSONL,
    prepare_python_import_path,
)

prepare_python_import_path()

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from models.language_adapter import LanguageAdapter

logger = logging.getLogger(__name__)


# ------------------------- args -------------------------

def parse_args():
    p = argparse.ArgumentParser("Evaluate SAM2+PLM on local jsonl/JSON (default) or optional HF")

    p.add_argument(
        "--plm-model-name",
        type=str,
        default=None,
        help="Local Qwen dir (default: student/Qwen2.5-VL-3B).",
    )

    # ---- HF mode (opt-in) ----
    p.add_argument(
        "--use_hf",
        action="store_true",
        help="Load eval set from Hugging Face (needs network; default is local jsonl).",
    )
    p.add_argument("--hf_dataset", type=str, default="aadarsh99/ConvSeg",
                   help="Hugging Face dataset repo id (default: aadarsh99/ConvSeg).")
    p.add_argument("--hf_config", type=str, default="default",
                   help="HF dataset config name (ConvSeg uses 'default').")
    p.add_argument("--hf_splits", type=str, default="sam_seeded,human_annotated",
                   help="Comma-separated splits to evaluate (e.g. 'sam_seeded' or 'sam_seeded,human_annotated').")
    p.add_argument("--hf_cache_dir", type=str, default=None,
                   help="Optional HF cache dir for datasets/downloads.")

    # ---- Local eval (default) ----
    p.add_argument("--input_json", type=str, default=None,
                   help="JSON with {\"items\": [...]} (see docstring).")
    p.add_argument(
        "--input_jsonl",
        type=str,
        default=None,
        help="Local .jsonl (one sample per line). Default: data/ConverSeg/sam_seeded.jsonl",
    )
    p.add_argument(
        "--eval_splits",
        type=str,
        default="sam_seeded",
        help="When --input_jsonl omitted: sam_seeded, human_annotated, or both comma-separated.",
    )

    # ---- Model / runtime ----
    p.add_argument("--model_cfg", type=str, default=default_sam_config_str())
    p.add_argument("--base_ckpt", type=str, default=default_sam_ckpt_str())
    p.add_argument("--final_ckpt", type=str, required=True,
                   help="Your fine-tuned SAM2 checkpoint (.torch)")
    p.add_argument("--plm_ckpt", type=str, required=True,
                   help="Your PLM adapter checkpoint (.torch)")
    p.add_argument("--lora_ckpt", type=str, required=True,
                   help="LoRA adapter path")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="fp32")
    p.add_argument("--num-shards", type=int, default=1, help="Split items across N independent processes (one GPU each).")
    p.add_argument("--shard-id", type=int, default=0, help="This process shard in [0, num-shards).")
    p.add_argument(
        "--metrics-out",
        type=str,
        default=None,
        help="Write per-shard JSON {n,sum_iou,total_inter,total_union,split,by_concept} for 2-GPU merge.",
    )
    p.add_argument(
        "--report-out",
        type=str,
        default=None,
        help="Write full Table-1 style JSON report (gIoU/cIoU overall + by concept).",
    )

    # ---- QoL ----
    p.add_argument("--limit", type=int, default=None,
                   help="Max number of samples (per split in HF mode; or per JSON list in JSON mode).")
    p.add_argument("--save_preds", type=str, default=None,
                   help="Directory to save predicted masks + GT + overlays.")

    return p.parse_args()


# Paper Table 1 concept order (jsonl field `concept`).
CONCEPT_ORDER = ("entities", "spatial", "relations", "affordances", "physics")
CONCEPT_SHORT = {
    "entities": "Ent.",
    "spatial": "Spat.",
    "relations": "Rel.",
    "affordances": "Aff.",
    "physics": "Phys.",
}


def _normalize_concept(raw: str | None, sample_id: str = "") -> str:
    """Map dataset concept labels onto the five paper buckets."""
    text = (raw or "").strip().lower()
    if not text and sample_id:
        # id often looks like: ...__entities__02
        for key in CONCEPT_ORDER:
            if f"__{key}__" in sample_id.lower():
                text = key
                break
    aliases = {
        "entity": "entities",
        "entities": "entities",
        "spatial": "spatial",
        "spat": "spatial",
        "relation": "relations",
        "relations": "relations",
        "affordance": "affordances",
        "affordances": "affordances",
        "physics": "physics",
        "phys": "physics",
        "physics & safety": "physics",
        "physics_and_safety": "physics",
    }
    return aliases.get(text, text if text else "unknown")


def _empty_bucket() -> dict:
    return {"n": 0, "sum_iou": 0.0, "total_inter": 0.0, "total_union": 0.0}


def _finalize_bucket(bucket: dict) -> dict:
    n = int(bucket["n"])
    giou = float(bucket["sum_iou"] / max(n, 1))
    ciou = float(bucket["total_inter"] / (bucket["total_union"] + 1e-10))
    return {
        "n": n,
        "giou": giou,
        "ciou": ciou,
        "sum_iou": float(bucket["sum_iou"]),
        "total_inter": float(bucket["total_inter"]),
        "total_union": float(bucket["total_union"]),
    }


def _accumulate(bucket: dict, inter: float, union: float, iou: float) -> None:
    bucket["n"] += 1
    bucket["sum_iou"] += float(iou)
    bucket["total_inter"] += float(inter)
    bucket["total_union"] += float(union)


def _print_table1_style(split_name: str, overall: dict, by_concept: dict[str, dict]) -> None:
    """Print a compact Table-1 style row: All + five concepts (gIoU)."""
    cells = [f"{overall['giou'] * 100:.1f}"]
    for key in CONCEPT_ORDER:
        if key in by_concept and by_concept[key]["n"] > 0:
            cells.append(f"{by_concept[key]['giou'] * 100:.1f}")
        else:
            cells.append("—")
    header = "All Ent. Spat. Rel. Aff. Phys."
    logger.info(
        "TABLE1-style [%s] gIoU(%%): %s | %s",
        split_name,
        header,
        " ".join(cells),
    )
    print(f"TABLE1[{split_name}] gIoU%: " + " ".join(cells))
    # also log cIoU overall + concepts
    c_cells = [f"{overall['ciou'] * 100:.1f}"]
    for key in CONCEPT_ORDER:
        if key in by_concept and by_concept[key]["n"] > 0:
            c_cells.append(f"{by_concept[key]['ciou'] * 100:.1f}")
        else:
            c_cells.append("—")
    print(f"TABLE8[{split_name}] cIoU%: " + " ".join(c_cells))
    for key in CONCEPT_ORDER:
        b = by_concept.get(key)
        if not b or b["n"] <= 0:
            continue
        short = CONCEPT_SHORT.get(key, key)
        logger.info(
            "  %s (%s): n=%d gIoU %.3f | cIoU %.3f",
            short,
            key,
            b["n"],
            b["giou"],
            b["ciou"],
        )


def _write_report(path: str, payload: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Wrote report -> %s", path)


# ------------------------- model build -------------------------

def _dtype_from_precision(precision: str) -> torch.dtype:
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _build_predictor_and_plm(
    cfg,
    base_ckpt,
    final_ckpt,
    plm_ckpt,
    lora_ckpt,
    device,
    precision="fp32",
    plm_model_name=None,
):
    name = str(plm_model_name or "")
    if precision == "fp32" and ("7B" in name or "7b" in name):
        logger.warning("7B eval with fp32 OOMs on 48GB; forcing precision=bf16")
        precision = "bf16"

    logger.info("Building SAM2+PLM cfg=%s device=%s precision=%s plm=%s", cfg, device, precision, name)

    model = build_sam2(cfg, base_ckpt, device=device)
    predictor = SAM2ImagePredictor(model)
    predictor.model.eval()

    if not os.path.isfile(final_ckpt):
        raise FileNotFoundError(f"Could not find fine-tuned SAM2 checkpoint: {final_ckpt}")
    logger.info("Loading fine-tuned SAM2 weights from %s (map_location=cpu)", final_ckpt)
    sd = torch.load(final_ckpt, map_location="cpu", weights_only=False)
    predictor.model.load_state_dict(sd.get("model", sd), strict=True)
    del sd

    C = predictor.model.sam_mask_decoder.transformer_dim
    plm_dtype = _dtype_from_precision(precision)

    # Assemble 7B on CPU first; .to(cuda) once. Avoids fp32 GPU duplicate of from_pretrained + ckpt.
    logger.info("Constructing LanguageAdapter on CPU dtype=%s", plm_dtype)
    plm = LanguageAdapter(
        model_name=plm_model_name,
        transformer_dim=C,
        n_sparse_tokens=0,
        use_dense_bias=True,
        use_lora=True, lora_r=16, lora_alpha=32, lora_dropout=0.05,
        dtype=plm_dtype,
        device="cpu",
        use_image_input=True,
        qwen_image_size=256,
    )

    if not os.path.isfile(plm_ckpt):
        raise FileNotFoundError(f"Could not find PLM adapter checkpoint: {plm_ckpt}")
    logger.info("Loading PLM adapter weights from %s (map_location=cpu)", plm_ckpt)
    plm_sd = torch.load(plm_ckpt, map_location="cpu", weights_only=False)
    plm.load_state_dict(plm_sd["plm"], strict=True)
    del plm_sd

    if lora_ckpt is not None:
        plm.load_lora(lora_ckpt)

    logger.info("Moving PLM to %s", device)
    plm = plm.to(device)
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
        logger.info("GPU mem alloc=%.2f GiB", torch.cuda.memory_allocated() / 1024**3)

    return predictor, plm


class SAM2PLMWrapper(torch.nn.Module):
    """
    Thin wrapper that runs predictor + PLM and returns a single best-mask logit per prompt.
    """
    def __init__(self, predictor, plm, device="cuda", precision="fp32"):
        super().__init__()
        self.predictor = predictor
        self.plm = plm
        self.device = torch.device(device)
        self.precision = precision

    @torch.no_grad()
    def infer_one(self, rgb: np.ndarray, text: str, image_path: str) -> torch.Tensor:
        """
        Args:
            rgb: HxWx3 uint8 (RGB) in ORIGINAL RESOLUTION
            text: str
            image_path: local filepath (used by PLM adapter)
        Returns:
            logit: [H, W] logits in original image size
        """
        amp_ctx = contextlib.nullcontext()
        if self.device.type == "cuda" and self.precision in ("fp16", "bf16"):
            amp_dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
            amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)

        with amp_ctx:
            # 1) set image -> get embeddings & high-res feats
            self.predictor.set_image(rgb)
            image_emb = self.predictor._features["image_embed"][-1].unsqueeze(0)  # [1,C,Hs,Ws]
            hi = [lvl[-1].unsqueeze(0) for lvl in self.predictor._features["high_res_feats"]]

            # 2) text -> (sparse, dense) via PLM
            sp, dp = self.plm([text], image_emb.shape[-2], image_emb.shape[-1], [image_path])  # sp:[1,N,C], dp:[1,C,Hs,Ws]

            # 3) SAM2 decode
            dec = self.predictor.model.sam_mask_decoder
            dev = next(dec.parameters()).device
            dtype = next(dec.parameters()).dtype

            image_pe = self.predictor.model.sam_prompt_encoder.get_dense_pe().to(dev, dtype)
            image_emb = image_emb.to(dev, dtype)
            hi = [h.to(dev, dtype) for h in hi]
            sp, dp = sp.to(dev, dtype), dp.to(dev, dtype)

            low, scores, _, _ = dec(
                image_embeddings=image_emb,
                image_pe=image_pe,
                sparse_prompt_embeddings=sp,
                dense_prompt_embeddings=dp,
                multimask_output=True,
                repeat_image=False,
                high_res_features=hi,
            )

            # Post-process to ORIGINAL image size
            logits = self.predictor._transforms.postprocess_masks(low, self.predictor._orig_hw[-1])  # [1,3,H0,W0]
            best = scores.argmax(dim=1).item()
            logit = logits[0, best]  # [H0, W0]
            return logit


# ------------------------- basic helpers -------------------------

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _resolve_path(base_dir: str, p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(base_dir, p)


@contextlib.contextmanager
def _suppress_stderr():
    """Last-resort silencer for noisy C-level libs (e.g., libpng via OpenCV)."""
    saved = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


def _pil_read_rgb(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im.info.pop("icc_profile", None)
        im = im.convert("RGB")
        return np.array(im)


def _read_mask(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return m


def _compute_iou(pred_bool: torch.Tensor, gt_bool: torch.Tensor):
    """
    pred_bool, gt_bool: torch.bool on CPU
    """
    inter = (pred_bool & gt_bool).sum().item()
    union = (pred_bool | gt_bool).sum().item()
    iou = 1.0 if union == 0 else (inter / (union + 1e-10))
    return inter, union, iou


# ------------------------- overlay helpers ----------------------------

EDGE_COLORS_HEX = ["#FF006E"]


def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


EDGE_COLORS = [_hex_to_rgb(h) for h in EDGE_COLORS_HEX]


def stable_color(key: str):
    import hashlib
    h = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16)
    return EDGE_COLORS[h % len(EDGE_COLORS)]


def tint(rgb, amt: float = 0.1):
    # soft pastel fill by mixing with white
    return tuple(int(255 - (255 - c) * (1 - amt)) for c in rgb)


def load_mask_bool(p: Path) -> np.ndarray:
    im = Image.open(p)
    try:
        if im.mode in ("RGBA", "LA"):
            arr = np.array(im.split()[-1]) > 0
        else:
            arr = np.array(im.convert("L")) > 0
    finally:
        im.close()
    return arr


def edge_map(mask_bool: np.ndarray, width_px: int = 1) -> Image.Image:
    m = Image.fromarray((mask_bool.astype(np.uint8) * 255), "L")
    edges = ImageChops.difference(
        m.filter(ImageFilter.MaxFilter(3)),
        m.filter(ImageFilter.MinFilter(3)),
    )
    for _ in range(max(0, width_px - 1)):
        edges = edges.filter(ImageFilter.MaxFilter(3))
    return edges.point(lambda p: 255 if p > 0 else 0)


def _resize_mask_to_target(m: np.ndarray, src_w: int, src_h: int, tgt_w: int, tgt_h: int) -> np.ndarray:
    # first resize to original image size if needed
    if (m.shape[1], m.shape[0]) != (src_w, src_h):
        m = np.array(
            Image.fromarray((m.astype(np.uint8) * 255), "L").resize((src_w, src_h), Image.NEAREST)
        ) > 0
    # then resize to target size
    m = np.array(
        Image.fromarray((m.astype(np.uint8) * 255), "L").resize((tgt_w, tgt_h), Image.NEAREST)
    ) > 0
    return m


def _apply_rounded_corners(img_rgb: Image.Image, radius: int) -> Image.Image:
    w, h = img_rgb.size
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=255)
    bg = Image.new("RGB", (w, h), "white")
    img_rgba = img_rgb.convert("RGBA")
    img_rgba.putalpha(mask)
    bg.paste(img_rgba.convert("RGB"), (0, 0), mask)
    return bg


def compose_overlay(
    img_path: Path,
    mask_path: Path,
    tgt_w: int,
    tgt_h: int,
    alpha_fill: float = 0.70,
    edge_w: int = 2,
    draw_box: bool = False,
    dpi: int = 300,
    no_resize: bool = False,
) -> Image.Image:
    """
    Overlay a single Boolean mask onto an image with soft pastel fill + colored contour.

    If no_resize=True, the image is kept at ORIGINAL SIZE and tgt_w/tgt_h are ignored.
    """
    base = Image.open(img_path).convert("RGB")
    src_w, src_h = base.size

    if no_resize:
        base_rgba = base.convert("RGBA")
        tgt_w, tgt_h = src_w, src_h
        m = load_mask_bool(mask_path)
        if (m.shape[1], m.shape[0]) != (src_w, src_h):
            base.close()
            raise RuntimeError(
                f"--no-resize: mask size {m.shape[1]}x{m.shape[0]} differs from image size {src_w}x{src_h}"
            )
    else:
        base_rgba = base.resize((tgt_w, tgt_h), Image.BICUBIC).convert("RGBA")
        m = load_mask_bool(mask_path)
        m = _resize_mask_to_target(m, src_w, src_h, tgt_w, tgt_h)

    color = stable_color(str(mask_path))
    fill_rgb = tint(color, 0.1)
    a = int(round(alpha_fill * 255))

    fill_layer = Image.new("RGBA", (tgt_w, tgt_h), fill_rgb + (0,))
    fill_layer.putalpha(Image.fromarray((m.astype(np.uint8) * a), "L"))
    edgesL = edge_map(m, width_px=edge_w)
    stroke = Image.new("RGBA", (tgt_w, tgt_h), color + (0,))
    stroke.putalpha(edgesL)

    out = Image.alpha_composite(base_rgba, fill_layer)
    out = Image.alpha_composite(out, stroke)

    if draw_box:
        ys, xs = np.where(m)
        if ys.size:
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            pad = max(2, int(round(min(tgt_w, tgt_h) * 0.004)))
            x0 = max(0, x0 - pad)
            y0 = max(0, y0 - pad)
            x1 = min(tgt_w - 1, x1 + pad)
            y1 = min(tgt_h - 1, y1 + pad)
            box = Image.new("RGBA", (tgt_w, tgt_h), (0, 0, 0, 0))
            d = ImageDraw.Draw(box)
            d.rectangle([x0, y0, x1, y1], outline=tint(color, 0.80) + (140,), width=2)
            out = Image.alpha_composite(out, box)

    out = out.convert("RGB")
    out = _apply_rounded_corners(out, max(12, int(0.06 * min(out.size))))
    out.info["dpi"] = (dpi, dpi)
    base.close()
    return out


# ------------------------- HF helpers -------------------------

def _mask_pil_to_u8(mask_img: Image.Image) -> np.ndarray:
    """
    Convert a PIL mask (L / 1 / RGB / RGBA) into uint8 grayscale (0..255).
    Uses alpha channel if present; otherwise converts to L.
    """
    if mask_img.mode in ("RGBA", "LA"):
        alpha = mask_img.split()[-1]
        return np.array(alpha, dtype=np.uint8)
    return np.array(mask_img.convert("L"), dtype=np.uint8)


def _hf_field_to_pil_and_path(
    field,
    tmp_dir: str,
    stem: str,
    force_rgb: bool = False,
) -> Tuple[Image.Image, str]:
    """
    HF datasets may return:
      - PIL.Image.Image
      - dict with keys like {"path": ..., "bytes": ...}
      - string path (common in JSON datasets)
    Return (PIL image, local_file_path). If no reliable path exists, write a temp PNG.
    """
    # String path
    if isinstance(field, str):
        if not os.path.isfile(field):
            # In some datasets, string paths are relative to cwd; try as Path
            if os.path.isfile(os.path.abspath(field)):
                field = os.path.abspath(field)
            else:
                raise FileNotFoundError(f"HF field path does not exist: {field}")
        img = Image.open(field)
        img = img.convert("RGB") if force_rgb else img
        return img, field

    # PIL
    if isinstance(field, Image.Image):
        img = field.convert("RGB") if force_rgb else field
        path = getattr(field, "filename", None)
        if path and os.path.isfile(path):
            return img, path
        out_path = os.path.join(tmp_dir, f"{stem}.png")
        img.save(out_path)
        return img, out_path

    # dict
    if isinstance(field, dict):
        p = field.get("path", None)
        b = field.get("bytes", None)

        if p and os.path.isfile(p):
            img = Image.open(p)
            img = img.convert("RGB") if force_rgb else img
            return img, p

        if b is not None:
            img = Image.open(io.BytesIO(b))
            img = img.convert("RGB") if force_rgb else img
            out_path = os.path.join(tmp_dir, f"{stem}.png")
            img.save(out_path)
            return img, out_path

    raise TypeError(f"Unsupported HF field type for image/mask: {type(field)}")


# ------------------------- evaluators -------------------------

def _jsonl_to_payload(jsonl_path: str) -> dict:
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return {"dataset": os.path.basename(jsonl_path), "count": len(items), "items": items}


def evaluate_jsonl(args, jsonl_path: str):
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
    shard = getattr(args, "shard_id", 0)
    payload_path = jsonl_path + f".__payload__.s{shard}.json"
    args_copy = argparse.Namespace(**vars(args))
    args_copy.input_json = payload_path
    try:
        with open(payload_path, "w", encoding="utf-8") as f:
            json.dump(_jsonl_to_payload(jsonl_path), f)
        evaluate_json(args_copy)
    finally:
        if os.path.isfile(payload_path):
            os.remove(payload_path)


def evaluate_json(args):
    if not os.path.isfile(args.input_json):
        raise FileNotFoundError(f"JSON file not found: {args.input_json}")

    with open(args.input_json, "r") as f:
        payload = json.load(f)

    items = payload.get("items", [])
    if args.limit is not None:
        items = items[: args.limit]
    n_shards = max(1, int(getattr(args, "num_shards", 1)))
    shard_id = int(getattr(args, "shard_id", 0))
    if n_shards > 1:
        if not (0 <= shard_id < n_shards):
            raise ValueError(f"--shard-id {shard_id} out of range for --num-shards {n_shards}")
        items = [it for i, it in enumerate(items) if i % n_shards == shard_id]
        logger.info("Shard %d/%d: %d item(s)", shard_id, n_shards, len(items))

    base_dir = os.path.dirname(os.path.abspath(args.input_json))
    if args.save_preds:
        _ensure_dir(args.save_preds)

    predictor, plm = _build_predictor_and_plm(
        args.model_cfg,
        args.base_ckpt,
        args.final_ckpt,
        args.plm_ckpt,
        args.lora_ckpt,
        args.device,
        precision=args.precision,
        plm_model_name=getattr(args, "plm_model_name", None),
    )
    model = SAM2PLMWrapper(predictor, plm, device=args.device, precision=args.precision)
    model.eval()

    overall_acc = _empty_bucket()
    by_concept_acc: dict[str, dict] = {k: _empty_bucket() for k in CONCEPT_ORDER}
    total_transposed = 0
    split_name = payload.get("dataset", os.path.basename(args.input_json))
    if split_name.endswith(".jsonl"):
        split_name = os.path.splitext(split_name)[0]

    logger.info("Evaluating %d item(s) from %s", len(items), split_name)
    for it in tqdm(items, desc="JSON items"):
        iid = it.get("id", "")
        img_p = _resolve_path(base_dir, it["image"])
        msk_p = _resolve_path(base_dir, it["mask"])
        prompt = it["prompt"]
        concept = _normalize_concept(it.get("concept"), str(iid))

        # read original image & mask in ORIGINAL resolution
        rgb_orig = _pil_read_rgb(img_p)
        gt = _read_mask(msk_p)

        H_img, W_img = rgb_orig.shape[:2]
        H_gt, W_gt = gt.shape[:2]
        if (H_img, W_img) != (H_gt, W_gt):
            raise RuntimeError(
                f"Shape mismatch for {iid}: image {W_img}x{H_img}, mask {W_gt}x{H_gt}"
            )

        with torch.no_grad():
            logit = model.infer_one(rgb_orig, prompt, img_p)  # [H, W] in original size

        H_pred, W_pred = logit.shape[-2], logit.shape[-1]
        if (H_pred, W_pred) != (H_img, W_img):
            raise RuntimeError(
                f"Predicted logit shape {W_pred}x{H_pred} does not match image size {W_img}x{H_img}"
            )

        gt_bool = torch.from_numpy((gt > 0).astype(np.bool_))  # CPU bool
        pred_bool = (logit.detach().to("cpu") > 0)             # CPU bool

        if pred_bool.shape == (gt_bool.shape[1], gt_bool.shape[0]) and pred_bool.shape != gt_bool.shape:
            pred_bool = pred_bool.T
            total_transposed += 1

        if pred_bool.shape != gt_bool.shape:
            raise RuntimeError(f"Shape mismatch for {iid}: pred {pred_bool.shape}, gt {gt_bool.shape}")

        inter, union, iou = _compute_iou(pred_bool, gt_bool)
        _accumulate(overall_acc, inter, union, iou)
        if concept not in by_concept_acc:
            by_concept_acc[concept] = _empty_bucket()
        _accumulate(by_concept_acc[concept], inter, union, iou)

        tqdm.write(f"{iid} [{concept}]: IoU {iou:.3f}")

        if args.save_preds:
            base_name = iid or os.path.splitext(os.path.basename(img_p))[0]

            pred_path = os.path.join(args.save_preds, f"{base_name}_pred.png")
            cv2.imwrite(pred_path, (pred_bool.numpy().astype(np.uint8) * 255))

            gt_path = os.path.join(args.save_preds, f"{base_name}_gt.png")
            cv2.imwrite(gt_path, (gt_bool.numpy().astype(np.uint8) * 255))

            panel_path = os.path.join(args.save_preds, f"{base_name}_panel.png")
            try:
                panel_img = compose_overlay(
                    img_path=Path(img_p),
                    mask_path=Path(pred_path),
                    tgt_w=1, tgt_h=1,
                    alpha_fill=0.70,
                    edge_w=2,
                    draw_box=False,
                    dpi=300,
                    no_resize=True,
                )
                panel_img.save(panel_path, dpi=(300, 300))
            except Exception as e:
                logger.warning("Failed to create overlay panel for %s: %s", iid, e)

            orig_out_path = os.path.join(args.save_preds, f"{base_name}_orig.png")
            try:
                with Image.open(img_p) as pil_img:
                    pil_img = pil_img.convert("RGB")
                    w_img, h_img = pil_img.size
                    radius = max(12, int(0.06 * min(w_img, h_img)))
                    styled_orig = _apply_rounded_corners(pil_img, radius)
                    styled_orig.save(orig_out_path, dpi=(300, 300))
            except Exception as e:
                logger.warning("Failed to save styled original for %s: %s", iid, e)

            prompt_out_path = os.path.join(args.save_preds, f"{base_name}_prompt.txt")
            try:
                with open(prompt_out_path, "w", encoding="utf-8") as f_txt:
                    f_txt.write(prompt)
            except Exception as e:
                logger.warning("Failed to save prompt txt for %s: %s", iid, e)

    if overall_acc["n"] == 0:
        logger.error("No valid items evaluated.")
        return

    logger.info("Total transposed predictions: %d out of %d", total_transposed, len(items))

    overall = _finalize_bucket(overall_acc)
    by_concept = {k: _finalize_bucket(v) for k, v in by_concept_acc.items() if v["n"] > 0}
    logger.info(
        "RESULTS [%s]: n=%d gIoU %.3f | cIoU %.3f",
        split_name,
        overall["n"],
        overall["giou"],
        overall["ciou"],
    )
    print(f"JSON[{split_name}]: giou{overall['giou']:.3f}_ciou{overall['ciou']:.3f}")
    _print_table1_style(split_name, overall, by_concept)

    metrics_out = getattr(args, "metrics_out", None)
    if metrics_out:
        rec = {
            "n": overall["n"],
            "sum_iou": overall["sum_iou"],
            "total_inter": overall["total_inter"],
            "total_union": overall["total_union"],
            "giou_shard": overall["giou"],
            "ciou_shard": overall["ciou"],
            "split": split_name,
            "by_concept": {
                k: {
                    "n": v["n"],
                    "sum_iou": v["sum_iou"],
                    "total_inter": v["total_inter"],
                    "total_union": v["total_union"],
                }
                for k, v in by_concept.items()
            },
            "shard_id": shard_id,
            "num_shards": n_shards,
        }
        os.makedirs(os.path.dirname(os.path.abspath(metrics_out)) or ".", exist_ok=True)
        with open(metrics_out, "a", encoding="utf-8") as mf:
            mf.write(json.dumps(rec) + "\n")
        logger.info("Wrote shard metrics -> %s", metrics_out)

    report_out = getattr(args, "report_out", None)
    if report_out:
        # When evaluating multiple splits in one process, append as jsonl of reports.
        report = {
            "split": split_name,
            "overall": overall,
            "by_concept": by_concept,
            "concept_order": list(CONCEPT_ORDER),
        }
        if n_shards > 1:
            report["shard_id"] = shard_id
            report["num_shards"] = n_shards
        os.makedirs(os.path.dirname(os.path.abspath(report_out)) or ".", exist_ok=True)
        with open(report_out, "a", encoding="utf-8") as rf:
            rf.write(json.dumps(report) + "\n")
        logger.info("Appended report -> %s", report_out)


def evaluate_hf(args):
    if load_dataset is None:
        raise RuntimeError(
            "Hugging Face `datasets` is not available. Install with:\n"
            "  pip install datasets\n"
            f"Import error: {_HF_IMPORT_ERROR}"
        )

    if args.save_preds:
        _ensure_dir(args.save_preds)

    tmp_dir = tempfile.mkdtemp(prefix="convseg_eval_")

    # Load HF dataset (downloads/caches automatically)
    dsdict = load_dataset(
        args.hf_dataset,
        name=args.hf_config,
        cache_dir=args.hf_cache_dir,
    )

    # Ensure image/mask are decodable as images even if the dataset stores them as strings
    for sp in list(dsdict.keys()):
        cols = set(dsdict[sp].column_names)
        if "image" in cols and HFImage is not None:
            try:
                dsdict[sp] = dsdict[sp].cast_column("image", HFImage())
            except Exception:
                pass
        if "mask" in cols and HFImage is not None:
            try:
                dsdict[sp] = dsdict[sp].cast_column("mask", HFImage())
            except Exception:
                pass

    splits = [s.strip() for s in args.hf_splits.split(",") if s.strip()]
    missing = [s for s in splits if s not in dsdict]
    if missing:
        raise ValueError(f"Requested split(s) not found: {missing}. Available: {list(dsdict.keys())}")

    # Build model once
    predictor, plm = _build_predictor_and_plm(
        args.model_cfg,
        args.base_ckpt,
        args.final_ckpt,
        args.plm_ckpt,
        args.lora_ckpt,
        args.device,
        precision=args.precision,
        plm_model_name=args.plm_model_name,
    )
    model = SAM2PLMWrapper(predictor, plm, device=args.device, precision=args.precision)
    model.eval()

    for split_name in splits:
        ds = dsdict[split_name]
        n_total = len(ds)
        n_eval = min(n_total, args.limit) if args.limit is not None else n_total

        logger.info("HF dataset=%s | config=%s | split=%s | evaluating %d/%d",
                    args.hf_dataset, args.hf_config, split_name, n_eval, n_total)

        overall_acc = _empty_bucket()
        by_concept_acc: dict[str, dict] = {k: _empty_bucket() for k in CONCEPT_ORDER}
        total_transposed = 0

        for idx in tqdm(range(n_eval), desc=f"HF:{split_name}"):
            ex = ds[idx]
            iid = ex.get("id", f"{split_name}_{idx}")
            prompt = ex["prompt"]
            concept = _normalize_concept(ex.get("concept"), str(iid))

            img_pil, img_path = _hf_field_to_pil_and_path(ex["image"], tmp_dir, stem=f"{iid}_img", force_rgb=True)
            msk_pil, _ = _hf_field_to_pil_and_path(ex["mask"], tmp_dir, stem=f"{iid}_gt", force_rgb=False)

            rgb_orig = np.array(img_pil.convert("RGB"), dtype=np.uint8)
            gt_u8 = _mask_pil_to_u8(msk_pil)

            H_img, W_img = rgb_orig.shape[:2]
            H_gt, W_gt = gt_u8.shape[:2]
            if (H_img, W_img) != (H_gt, W_gt):
                raise RuntimeError(
                    f"Shape mismatch for {iid}: image {W_img}x{H_img}, mask {W_gt}x{H_gt}"
                )

            with torch.no_grad():
                logit = model.infer_one(rgb_orig, prompt, img_path)

            H_pred, W_pred = logit.shape[-2], logit.shape[-1]
            if (H_pred, W_pred) != (H_img, W_img):
                raise RuntimeError(
                    f"Pred logit shape {W_pred}x{H_pred} does not match image size {W_img}x{H_img}"
                )

            gt_bool = torch.from_numpy((gt_u8 > 0).astype(np.bool_))
            pred_bool = (logit.detach().to("cpu") > 0)

            if pred_bool.shape == (gt_bool.shape[1], gt_bool.shape[0]) and pred_bool.shape != gt_bool.shape:
                pred_bool = pred_bool.T
                total_transposed += 1

            if pred_bool.shape != gt_bool.shape:
                raise RuntimeError(f"Shape mismatch for {iid}: pred {pred_bool.shape}, gt {gt_bool.shape}")

            inter, union, iou = _compute_iou(pred_bool, gt_bool)
            _accumulate(overall_acc, inter, union, iou)
            if concept not in by_concept_acc:
                by_concept_acc[concept] = _empty_bucket()
            _accumulate(by_concept_acc[concept], inter, union, iou)

            if args.save_preds:
                base_name = iid

                pred_path = os.path.join(args.save_preds, f"{base_name}_pred.png")
                cv2.imwrite(pred_path, (pred_bool.numpy().astype(np.uint8) * 255))

                gt_path = os.path.join(args.save_preds, f"{base_name}_gt.png")
                cv2.imwrite(gt_path, (gt_bool.numpy().astype(np.uint8) * 255))

                orig_out_path = os.path.join(args.save_preds, f"{base_name}_orig.png")
                img_pil.convert("RGB").save(orig_out_path, dpi=(300, 300))

                panel_path = os.path.join(args.save_preds, f"{base_name}_panel.png")
                try:
                    panel_img = compose_overlay(
                        img_path=Path(orig_out_path),
                        mask_path=Path(pred_path),
                        tgt_w=1, tgt_h=1,
                        alpha_fill=0.70,
                        edge_w=2,
                        draw_box=False,
                        dpi=300,
                        no_resize=True,
                    )
                    panel_img.save(panel_path, dpi=(300, 300))
                except Exception as e:
                    logger.warning("Failed overlay for %s: %s", iid, e)

                prompt_out_path = os.path.join(args.save_preds, f"{base_name}_prompt.txt")
                with open(prompt_out_path, "w", encoding="utf-8") as f:
                    f.write(prompt)

        if overall_acc["n"] == 0:
            logger.error("No samples evaluated for split %s.", split_name)
            continue

        overall = _finalize_bucket(overall_acc)
        by_concept = {k: _finalize_bucket(v) for k, v in by_concept_acc.items() if v["n"] > 0}
        logger.info(
            "RESULTS (HF split=%s): n=%d gIoU %.3f | cIoU %.3f | transposed=%d",
            split_name,
            overall["n"],
            overall["giou"],
            overall["ciou"],
            total_transposed,
        )
        print(f"HF[{split_name}]: giou{overall['giou']:.3f}_ciou{overall['ciou']:.3f}")
        _print_table1_style(split_name, overall, by_concept)

        report_out = getattr(args, "report_out", None)
        if report_out:
            report = {
                "split": split_name,
                "overall": overall,
                "by_concept": by_concept,
                "concept_order": list(CONCEPT_ORDER),
                "source": "hf",
            }
            os.makedirs(os.path.dirname(os.path.abspath(report_out)) or ".", exist_ok=True)
            with open(report_out, "a", encoding="utf-8") as rf:
                rf.write(json.dumps(report) + "\n")


# ------------------------- main -------------------------

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    apply_offline_env()
    if args.plm_model_name is None:
        args.plm_model_name = default_plm_dir_str()

    if args.use_hf:
        evaluate_hf(args)
        return
    if args.input_json is not None:
        evaluate_json(args)
        return
    if args.input_jsonl is not None:
        evaluate_jsonl(args, args.input_jsonl)
        return

    split_map = {
        "sam_seeded": str(EVAL_SAM_SEEDED_JSONL),
        "human_annotated": str(EVAL_HUMAN_JSONL),
    }
    names = [s.strip() for s in args.eval_splits.split(",") if s.strip()]
    if not names:
        raise ValueError("No eval split requested.")
    for name in names:
        path = split_map.get(name)
        if path is None:
            raise ValueError(f"Unknown eval split {name!r}. Choose from {list(split_map)}")
        evaluate_jsonl(args, path)


if __name__ == "__main__":
    main()
