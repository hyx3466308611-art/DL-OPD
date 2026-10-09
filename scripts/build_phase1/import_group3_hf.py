#!/usr/bin/env python3
"""Group 3: rewrite HF open_vocab jsonl paths relative to data/ root."""
import argparse
import json
from pathlib import Path

from config import DATA_ROOT, HF_G3_JSONL, HF_G3_ROOT, MANIFEST_DIR
from io_utils import write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-jsonl", type=Path, default=MANIFEST_DIR / "group3_open_vocab.jsonl")
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    if not HF_G3_JSONL.is_file():
        raise SystemExit("Missing HF Group3: %s" % HF_G3_JSONL)

    # HF jsonl paths are relative to converseg_stage1_data/ (see train_stage1_100k.sh)
    stage1_base = (
        DATA_ROOT / "ConverSeg-Training-Data" / "converseg_stage1_data"
    ).relative_to(DATA_ROOT).as_posix()
    rows = []
    with HF_G3_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            img = item["image"].replace("\\", "/")
            msk = item["mask_merged"].replace("\\", "/")
            if not img.startswith("ConverSeg-Training-Data/"):
                img = "%s/%s" % (stage1_base, img.lstrip("/"))
                msk = "%s/%s" % (stage1_base, msk.lstrip("/"))
            rows.append(
                {
                    "image": img,
                    "mask_merged": msk,
                    "prompt": item["prompt"],
                    "source": "hf_open_vocab",
                }
            )
            if args.max_rows and len(rows) >= args.max_rows:
                break

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    n = write_jsonl(args.out_jsonl, rows)
    print("Wrote %d rows -> %s" % (n, args.out_jsonl))


if __name__ == "__main__":
    main()
