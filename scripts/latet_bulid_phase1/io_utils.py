"""JSONL 读写与「文件是否存在」过滤（相对 data/ 根目录）。"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from config import DATA_ROOT


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def row_files_exist(row: dict, data_root: Path = DATA_ROOT) -> bool:
    for key in ("image", "mask_merged"):
        p = data_root / str(row[key]).replace("\\", "/")
        if not p.is_file():
            return False
    return bool(str(row.get("prompt", "")).strip())


def filter_existing_rows(rows: Iterable[dict], data_root: Path = DATA_ROOT) -> List[dict]:
    out: List[dict] = []
    for row in rows:
        if row_files_exist(row, data_root):
            out.append(row)
    return out


def cap_shuffle(rows: List[dict], cap: Optional[int], seed: int) -> List[dict]:
    if cap is None or len(rows) <= cap:
        return rows
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:cap]


def train_row(row: dict) -> dict:
    return {
        "image": row["image"],
        "mask_merged": row["mask_merged"],
        "prompt": row["prompt"],
    }
