#!/usr/bin/env bash
# 一键从零构建 Group1–4 与 Phase1/2 混合清单（在仓库根目录执行）。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BD="$ROOT/scripts/latet_bulid_phase1"
PY="${PYTHON:-python3}"

if [[ -x "${CONDA_PY:-}" ]]; then
  PY="$CONDA_PY"
elif [[ -x "$HOME/miniconda311/envs/converseg/bin/python" ]]; then
  PY="$HOME/miniconda311/envs/converseg/bin/python"
fi

echo "=== ConVerSeg dataset rebuild (latet_bulid_phase1) ==="
echo "REPO=$ROOT"
echo "PYTHON=$PY"

cd "$ROOT"

echo "--- Group1 (COCO 200k + COCONut 140k + PACO 100k) ---"
"$PY" "$BD/build_group1.py" --coconut-target 140000

echo "--- Group2 (RefCOCO*) ---"
"$PY" "$BD/build_group2.py"

echo "--- Group3 (HF Stage1, all valid masks ~48529) ---"
"$PY" "$BD/build_group3.py"

echo "--- Group4 (HF Stage2 conversational +/-) ---"
"$PY" "$BD/build_group4.py"

echo "--- Phase1 mix ---"
"$PY" "$BD/mix_phase1_train.py" --g3-cap 48529 --coconut-cap 140000 --paco-cap 100000

echo "--- Phase2 mix (compose: G4 + replay + G3 full) ---"
"$PY" "$BD/mix_phase2_train.py" --g3-cap 48529

echo "=== All done ==="
echo "Phase1: data/phase1/phase1_mixed.jsonl"
echo "Phase2: data/phase2/phase2_mixed.jsonl"
