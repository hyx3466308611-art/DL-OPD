"""Resolve HF jsonl paths relative to repo data/ root."""
from __future__ import annotations

from config import DATA_ROOT, HF_STAGE1, HF_STAGE2


def stage1_data_rel() -> str:
    return HF_STAGE1.relative_to(DATA_ROOT).as_posix()


def stage2_data_rel() -> str:
    return HF_STAGE2.relative_to(DATA_ROOT).as_posix()


def rewrite_under_data_root(item: dict, stage_rel_prefix: str) -> tuple[str, str]:
    """HF manifest paths are relative to converseg_stage{1,2}_data/, not leaf split folders."""
    img = item["image"].replace("\\", "/")
    msk = item["mask_merged"].replace("\\", "/")
    if not img.startswith("ConverSeg-Training-Data/"):
        img = "%s/%s" % (stage_rel_prefix, img.lstrip("/"))
        msk = "%s/%s" % (stage_rel_prefix, msk.lstrip("/"))
    return img, msk


def rewrite_stage1(item: dict) -> tuple[str, str]:
    return rewrite_under_data_root(item, stage1_data_rel())


def rewrite_stage2(item: dict) -> tuple[str, str]:
    return rewrite_under_data_root(item, stage2_data_rel())
