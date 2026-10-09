#!/usr/bin/env bash
# 固定 PACO：min-area=256，mix 使用 69614 行；刷新 phase1/2 mixed 并全量检查 jsonl。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BD="$ROOT/scripts/latet_bulid_phase1"
PY="${PYTHON:-python3}"
if [[ -x "$HOME/miniconda311/envs/converseg/bin/python" ]]; then
  PY="$HOME/miniconda311/envs/converseg/bin/python"
fi
cd "$ROOT"

PACO_ROWS=69614
PACO_AREA=256
G3=48529

echo "=== Finalize PACO $PACO_ROWS (min-area $PACO_AREA) + remix + check ==="

"$PY" "$BD/build_group1.py" \
  --skip-coco --skip-coconut \
  --paco-max-rows "$PACO_ROWS" \
  --paco-min-area "$PACO_AREA"

"$PY" "$BD/build_group3.py"

"$PY" "$BD/mix_phase1_train.py" \
  --g3-cap "$G3" --coconut-cap 140000 --paco-cap "$PACO_ROWS"

"$PY" "$BD/mix_phase2_train.py" --g3-cap "$G3"

"$PY" "$BD/check_all_jsonl.py" --sample 3000

echo "=== Finalize done ==="
