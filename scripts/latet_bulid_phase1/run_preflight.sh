#!/usr/bin/env bash
# 构建前自检（通过后再跑 run_build_all.sh）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BD="$ROOT/scripts/latet_bulid_phase1"
PY="${PYTHON:-python3}"

if [[ -x "${CONDA_PY:-}" ]]; then
  PY="$CONDA_PY"
elif [[ -x "$HOME/miniconda311/envs/converseg/bin/python" ]]; then
  PY="$HOME/miniconda311/envs/converseg/bin/python"
fi

cd "$ROOT"
echo "PYTHON=$PY"
exec "$PY" "$BD/preflight_before_build.py" "$@"
