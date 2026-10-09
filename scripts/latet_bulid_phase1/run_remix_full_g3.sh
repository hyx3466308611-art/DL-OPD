#!/usr/bin/env bash
# Stage1 mask 已补全后：不重跑 Group1/2/4，只刷新 G3 manifest + Phase1/2 混合。
# 若 group1_paco.jsonl 不足 10 万行，会先补导 PACO。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BD="$ROOT/scripts/latet_bulid_phase1"
PY="${PYTHON:-python3}"
if [[ -x "$HOME/miniconda311/envs/converseg/bin/python" ]]; then
  PY="$HOME/miniconda311/envs/converseg/bin/python"
fi
cd "$ROOT"

G3_CAP=48529
PACO_CAP=100000

echo "=== Remix: G3 full + phase1/2 mixed (skip heavy Group1 coco/coconut/ref) ==="

PACO_LINES=0
if [[ -f data/phase1/manifests/group1_paco.jsonl ]]; then
  PACO_LINES=$(wc -l < data/phase1/manifests/group1_paco.jsonl)
fi
if [[ "$PACO_LINES" -lt "$PACO_CAP" ]]; then
  echo "--- PACO manifest $PACO_LINES < $PACO_CAP, re-export PACO only ---"
  "$PY" "$BD/build_group1.py" --skip-coco --skip-coconut --paco-max-rows "$PACO_CAP" --paco-min-area 256
else
  echo "--- PACO manifest OK ($PACO_LINES lines), skip PACO export ---"
fi

echo "--- Group3 (all valid HF rows) ---"
"$PY" "$BD/build_group3.py"

echo "--- Phase1 mix ---"
"$PY" "$BD/mix_phase1_train.py" --g3-cap "$G3_CAP" --coconut-cap 140000 --paco-cap "$PACO_CAP"

echo "--- Phase2 mix ---"
"$PY" "$BD/mix_phase2_train.py" --g3-cap "$G3_CAP"

echo "=== Done ==="
wc -l data/phase1/phase1_mixed.jsonl data/phase2/phase2_mixed.jsonl
echo "See phase1_mixed.meta.json / phase2_mixed.meta.json"
