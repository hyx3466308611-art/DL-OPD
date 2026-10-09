#!/usr/bin/env bash
# Source this in nohup/non-interactive jobs:  source scripts/_converseg_env.sh
# Sets PYTHON to converseg env (avoids broken `conda activate` under nohup).

if [[ -n "${PYTHON:-}" && -x "$PYTHON" ]]; then
  :
elif [[ -x /home/hanyuxi/miniconda3/envs/converseg/bin/python ]]; then
  PYTHON=/home/hanyuxi/miniconda3/envs/converseg/bin/python
elif [[ -x /home/hanyuxi/miniconda311/envs/converseg/bin/python ]]; then
  PYTHON=/home/hanyuxi/miniconda311/envs/converseg/bin/python
elif [[ -f /home/hanyuxi/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/hanyuxi/miniconda3/etc/profile.d/conda.sh
  conda activate converseg
  PYTHON="$(command -v python)"
elif [[ -f /home/hanyuxi/miniconda311/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/hanyuxi/miniconda311/etc/profile.d/conda.sh
  conda activate converseg
  PYTHON="$(command -v python)"
else
  echo "ERROR: cannot find converseg python. Set PYTHON=/path/to/python"
  return 1 2>/dev/null || exit 1
fi

export PYTHON
export PYTHONUNBUFFERED=1
# 权重与数据已在服务器本地，训练/评测不访问 Hugging Face Hub
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
echo "[env] PYTHON=$PYTHON"
"$PYTHON" -V
