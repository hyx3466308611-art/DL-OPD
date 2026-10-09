#!/usr/bin/env bash
# 在国内/校内 GPU 机器上一键拉作者训练数据（HF 国内镜像，非 COCO 原始 zip）
# 用法: bash scripts/download_training_data_cn.sh
# 目标目录: $REPO/data/ConverSeg-Training-Data  (~19GB)

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$REPO/data}"
OUT="$DATA_ROOT/ConverSeg-Training-Data"

# Hugging Face 国内常用镜像（TUNA 已停 HF 全量同步，见 mirrors.tuna 2021 公告）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

echo "REPO=$REPO"
echo "OUT=$OUT"
echo "HF_ENDPOINT=$HF_ENDPOINT"

# 勿 pip install -U huggingface_hub 到 2.x，会与 transformers<5 冲突（见 requirements.txt）
python -m pip install "huggingface-hub>=0.34.0,<1.0" "huggingface_hub[cli]" -i "$PIP_INDEX"

mkdir -p "$OUT"

huggingface-cli download aadarsh99/ConverSeg-Training-Data \
  --repo-type dataset \
  --local-dir "$OUT" \
  --resume-download

echo "Done. Check:"
wc -l "$OUT/converseg_stage1_data/open_vocabulary_regions_data/open_vocabulary_regions_data.jsonl" || true
wc -l "$OUT/converseg_stage2_data/stage2_mixed.jsonl" || true
