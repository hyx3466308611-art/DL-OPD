import json
import random
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np
from PIL import Image


def rel_to_data_root(path: Path, data_root: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(data_root.resolve()).as_posix()
    except ValueError:
        return str(path)


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


def save_mask_png(path: Path, mask_bool: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L").save(path)


def reservoir_sample(iterable: Iterable[dict], k: int, seed: int = 42) -> List[dict]:
    """Reservoir sample k items from unknown-length stream."""
    rng = random.Random(seed)
    res: List[dict] = []
    for i, item in enumerate(iterable):
        if i < k:
            res.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                res[j] = item
    return res


def shuffle_jsonl(in_path: Path, out_path: Path, seed: int = 42) -> int:
    rows = list(read_jsonl(in_path))
    rng = random.Random(seed)
    rng.shuffle(rows)
    return write_jsonl(out_path, rows)
