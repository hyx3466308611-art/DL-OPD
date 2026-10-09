"""Phase2 + human + conversational negative manifests for fix v1."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

from data.load_samples import load_samples_from_jsonl, resolve_media_path, _make_bases


def load_converseg_jsonl_pairs(
    converseg_root: str | Path,
    jsonl_rel: str,
    repo_root: str | Path,
) -> list[tuple[str, str, str]]:
    root = Path(converseg_root)
    manifest = root / jsonl_rel if not os.path.isabs(jsonl_rel) else Path(jsonl_rel)
    if not manifest.is_file():
        manifest = root / "data" / os.path.basename(jsonl_rel)
    if not manifest.is_file():
        raise FileNotFoundError(f"ConverSeg manifest not found: {jsonl_rel} under {root}")

    bases = _make_bases(root, repo_root)
    out: list[tuple[str, str, str]] = []
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            img_rel = item.get("image") or item.get("image_path")
            msk_rel = item.get("mask") or item.get("mask_merged")
            pr = item.get("prompt", "") or "segment"
            ip = resolve_media_path(str(img_rel), bases) if img_rel else None
            mp = resolve_media_path(str(msk_rel), bases) if msk_rel else None
            if ip and mp:
                out.append((ip, mp, pr))
    print(f"[fix_v1] Loaded {len(out)} ConverSeg pairs from {manifest}")
    return out


def load_negative_pairs(data_dir: str | Path, jsonl_rel: str, repo_root: str | Path) -> list[tuple[str, str, str]]:
    pairs = load_samples_from_jsonl(data_dir, jsonl_rel, repo_root=repo_root)
    print(f"[fix_v1] Loaded {len(pairs)} negative-dialogue pairs (GT forced zero in train)")
    return pairs


def mix_train_pairs(
    phase2_pairs: list[tuple[str, str, str]],
    human_pairs: list[tuple[str, str, str]],
    human_repeat: int,
    seed: int,
) -> list[tuple[str, str, str]]:
    if human_repeat <= 0 or not human_pairs:
        boosted = list(phase2_pairs)
    else:
        boosted = list(phase2_pairs)
        for _ in range(human_repeat):
            boosted.extend(human_pairs)
    rng = random.Random(seed)
    rng.shuffle(boosted)
    print(f"[fix_v1] Mixed positive train size {len(boosted)}")
    return boosted
