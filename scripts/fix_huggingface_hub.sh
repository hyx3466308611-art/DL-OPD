#!/usr/bin/env bash
# 将 huggingface-hub 降回 transformers<5 兼容范围（勿用 2.x）
set -euo pipefail
pip install "huggingface-hub>=0.34.0,<1.0"
python -c "import huggingface_hub as h; print('huggingface-hub', h.__version__)"
