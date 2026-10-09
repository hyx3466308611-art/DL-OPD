#!/usr/bin/env bash
# Rebuild Group 1 (paper Sec. 6.2.1 literal): COCO + COCONut + PACO = 440K
# From repo root. Requires data/raw (coco, coconut, paco).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
BD="scripts/build_phase1"
PY="${PYTHON:-python}"

echo "==> Removing old Group1 manifests and masks (Group2/3 untouched)"
rm -f data/phase1/manifests/group1_coco.jsonl
rm -f data/phase1/manifests/group1_coconut.jsonl
rm -f data/phase1/manifests/group1_paco.jsonl
rm -rf data/phase1/masks/group1_coco
rm -rf data/phase1/masks/group1_coconut
rm -rf data/phase1/masks/group1_paco

"$PY" "$BD/export_group1_coco.py" --max-rows 200000
"$PY" "$BD/export_group1_coconut.py" \
  --target-rows 140000 --min-area 768 --min-area-small 512 --small-quota 8000
"$PY" "$BD/export_group1_paco.py" --max-rows 100000

echo "==> Group1 done. Re-run mix_phase1.py when Group2/3 ready."
