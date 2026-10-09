#!/usr/bin/env bash
# Cloud Phase-1: verify layout -> build (if needed) -> full path check
# Usage (repo root, in tmux):
#   pip install pycocotools tqdm pillow numpy
#   bash scripts/build_phase1/cloud_phase1.sh verify
#   bash scripts/build_phase1/cloud_phase1.sh build
#   bash scripts/build_phase1/cloud_phase1.sh check
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
BD="scripts/build_phase1"
PY="${PYTHON:-python}"
MODE="${1:-verify}"

case "$MODE" in
  verify)
    "$PY" "$BD/verify_data_layout.py"
    ;;
  build)
    "$PY" "$BD/verify_data_layout.py" || {
      echo "Fix layout before build."
      exit 1
    }
    bash "$BD/run_build.sh"
    ;;
  check)
    "$PY" "$BD/verify_data_layout.py"
    "$PY" "$BD/check_phase1.py" --sample-paths 500
    ;;
  fullscan)
    # Slow: every manifest row vs disk (run after sample check PASS)
    (cd "$BD" && "$PY" -c "
from config import DATA_ROOT, MANIFEST_DIR
from io_utils import read_jsonl
for name in ['group1_coco.jsonl','group1_coconut.jsonl','group1_paco.jsonl','group2_ref.jsonl','group3_open_vocab.jsonl']:
    p = MANIFEST_DIR / name
    mi = mm = n = 0
    for row in read_jsonl(p):
        n += 1
        if not (DATA_ROOT / row['image']).is_file(): mi += 1
        if not (DATA_ROOT / row['mask_merged']).is_file(): mm += 1
    print('%s rows=%d missing_img=%d missing_mask=%d' % (name, n, mi, mm))
")
    ;;
  *)
    echo "Usage: $0 {verify|build|check|fullscan}"
    exit 1
    ;;
esac
