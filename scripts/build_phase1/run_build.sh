#!/usr/bin/env bash
# Phase-1 build (Linux). From repo root:
#   pip install pycocotools tqdm pillow numpy
#   bash scripts/build_phase1/run_build.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
BD="scripts/build_phase1"
PY="${PYTHON:-python}"

"$PY" "$BD/import_group3_hf.py"
"$PY" "$BD/export_group2_ref.py"
"$PY" "$BD/export_group1_coco.py" --max-rows 200000
"$PY" "$BD/export_group1_coconut.py" \
  --target-rows 140000 --min-area 768 --min-area-small 512 --small-quota 8000
"$PY" "$BD/export_group1_paco.py" --max-rows 100000
"$PY" "$BD/mix_phase1.py" --ref-cap 321327 --coconut-cap 140000
"$PY" "$BD/check_phase1.py" --sample-paths 500

echo "Done. train.py --data-dir $ROOT/data --dataset-jsonl phase1/phase1_mixed.jsonl"
