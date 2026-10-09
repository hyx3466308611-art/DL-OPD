#!/usr/bin/env bash
# 提交 7B fix_v1 / fix_v2 之前运行。
# 确认 ConverSeg 测评集（sam_seeded + human_annotated）不会进入训练清单，
# 并检查 Stage-1 / fix_v1 权重是否已经就位。
#
#   cd /home/hanyuxi/ConverSeg-main
#   bash scripts/teacher_correction/check_eval_not_in_train.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export ROOT
export TEACHER_BACKBONE=teacher7b

FAIL=0
note() { printf '%s\n' "$*"; }
bad() { printf 'FAIL: %s\n' "$*"; FAIL=1; }
ok() { printf 'OK: %s\n' "$*"; }

note "ROOT=$ROOT"
note "TEACHER_BACKBONE=$TEACHER_BACKBONE"

if [[ "${HUMAN_REPEAT:-0}" != "0" ]]; then
  bad "当前 shell 的 HUMAN_REPEAT=${HUMAN_REPEAT}。sbatch --export=ALL 会把它带进作业。请先 unset HUMAN_REPEAT，提交时显式写 HUMAN_REPEAT=0"
fi

FILES=(
  scripts/teacher_correction/fix_v1/_env_fix_v1.sh
  scripts/teacher_correction/fix_v1/slurm_train_fix_v1.slurm
  scripts/teacher_correction/fix_v1/train_fix_v1.py
  scripts/teacher_correction/fix_v1/data_mix.py
  scripts/teacher_correction/fix_v2/_env_fix_v2.sh
  scripts/teacher_correction/fix_v2/slurm_train_fix_v2.slurm
  scripts/teacher_correction/fix_v2/train_fix_v2.py
)
for f in "${FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    bad "缺少 $f"
    continue
  fi
  if grep -q $'\r' "$f"; then
    bad "$f 含 CRLF，先执行: sed -i 's/\\r\$//' $f"
  fi
done

if grep -nE -- '--human-repeat[[:space:]]+[1-9]' \
  scripts/teacher_correction/fix_v1/slurm_train_fix_v1.slurm \
  scripts/teacher_correction/fix_v2/slurm_train_fix_v2.slurm \
  scripts/teacher_correction/fix_v1/_env_fix_v1.sh \
  scripts/teacher_correction/fix_v2/_env_fix_v2.sh >/dev/null; then
  bad "slurm/env 里把 --human-repeat 写成了正整数"
fi

if ! grep -q 'HUMAN_REPEAT="${HUMAN_REPEAT:-0}"' scripts/teacher_correction/fix_v1/_env_fix_v1.sh; then
  bad "fix_v1 环境默认不是 HUMAN_REPEAT=0"
fi
if ! grep -q 'HUMAN_REPEAT="${HUMAN_REPEAT:-0}"' scripts/teacher_correction/fix_v2/_env_fix_v2.sh; then
  bad "fix_v2 环境默认不是 HUMAN_REPEAT=0"
fi
if ! grep -q -- '--human-repeat", type=int, default=0' scripts/teacher_correction/fix_v1/train_fix_v1.py; then
  bad "train_fix_v1.py 未把 --human-repeat 默认设为 0"
fi
if ! grep -q -- '--human-repeat", type=int, default=0' scripts/teacher_correction/fix_v2/train_fix_v2.py; then
  bad "train_fix_v2.py 未把 --human-repeat 默认设为 0"
fi
for f in \
  scripts/teacher_correction/fix_v1/train_fix_v1.py \
  scripts/teacher_correction/fix_v2/train_fix_v2.py
do
  if ! grep -q 'human_annotated is the ConverSeg eval split' "$f"; then
    bad "$f 缺少 human_repeat!=0 时的拒绝退出（请重新同步本地最新文件）"
  fi
done
for f in \
  scripts/teacher_correction/fix_v1/slurm_train_fix_v1.slurm \
  scripts/teacher_correction/fix_v2/slurm_train_fix_v2.slurm
do
  if ! grep -q 'REFUSE: HUMAN_REPEAT' "$f"; then
    bad "$f 缺少 HUMAN_REPEAT!=0 时的拒绝退出（请重新同步本地最新文件）"
  fi
done

# 路径必须是 ConverSeg（小写 v）。下面只扫训练入口，不扫本检查脚本的说明文字。
if grep -nE 'data/ConVerSeg|/ConVerSeg/|ConVerSeg-Training-Data' \
  scripts/teacher_correction/fix_v1/_env_fix_v1.sh \
  scripts/teacher_correction/fix_v2/_env_fix_v2.sh \
  scripts/teacher_correction/fix_v1/train_fix_v1.py \
  scripts/teacher_correction/fix_v2/train_fix_v2.py >/dev/null; then
  bad "训练入口仍写着错误大小写 ConVerSeg；Linux 上应使用 ConverSeg / ConverSeg-Training-Data"
fi

PY_BIN="$(command -v python3 || command -v python)"

(
  export ROOT TEACHER_BACKBONE=teacher7b
  unset HUMAN_REPEAT || true
  # shellcheck disable=SC1091
  source "$ROOT/scripts/teacher_correction/fix_v1/_env_fix_v1.sh"
  [[ "$HUMAN_REPEAT" == "0" ]] || { echo "FAIL: sourced fix_v1 HUMAN_REPEAT=$HUMAN_REPEAT"; exit 1; }
  [[ "$PHASE2_JSONL_REL" == "phase2/phase2_mixed.jsonl" ]] \
    || { echo "FAIL: fix_v1 PHASE2_JSONL_REL=$PHASE2_JSONL_REL"; exit 1; }
  case "$CONVERSEG_EVAL_ROOT" in
    */data/ConverSeg/data|*/data/ConverSeg) ;;
    *) echo "FAIL: fix_v1 CONVERSEG_EVAL_ROOT=$CONVERSEG_EVAL_ROOT（应为 data/ConverSeg/...）"; exit 1 ;;
  esac
  printf 'OK: fix_v1 env HUMAN_REPEAT=%s jsonl=%s out=%s eval_root=%s\n' \
    "$HUMAN_REPEAT" "$PHASE2_JSONL_REL" "$CORR_OUT" "$CONVERSEG_EVAL_ROOT"
  "$PY_BIN" - "${TRAIN_FIX_COMMON[@]}" <<'PY'
import sys
args = sys.argv[1:]
if "--human-repeat" not in args:
    raise SystemExit("FAIL: TRAIN_FIX_COMMON 没有 --human-repeat")
val = args[args.index("--human-repeat") + 1]
if val != "0":
    raise SystemExit(f"FAIL: TRAIN_FIX_COMMON --human-repeat {val}")
print(f"OK: fix_v1 TRAIN_FIX_COMMON --human-repeat {val}")
PY
)
(
  export ROOT TEACHER_BACKBONE=teacher7b
  unset HUMAN_REPEAT || true
  # shellcheck disable=SC1091
  source "$ROOT/scripts/teacher_correction/fix_v2/_env_fix_v2.sh"
  [[ "$HUMAN_REPEAT" == "0" ]] || { echo "FAIL: sourced fix_v2 HUMAN_REPEAT=$HUMAN_REPEAT"; exit 1; }
  [[ "$PHASE2_JSONL_REL" == "phase2/phase2_mixed.jsonl" ]] \
    || { echo "FAIL: fix_v2 PHASE2_JSONL_REL=$PHASE2_JSONL_REL"; exit 1; }
  case "$CONVERSEG_EVAL_ROOT" in
    */data/ConverSeg/data|*/data/ConverSeg) ;;
    *) echo "FAIL: fix_v2 CONVERSEG_EVAL_ROOT=$CONVERSEG_EVAL_ROOT（应为 data/ConverSeg/...）"; exit 1 ;;
  esac
  printf 'OK: fix_v2 env HUMAN_REPEAT=%s jsonl=%s out=%s eval_root=%s\n' \
    "$HUMAN_REPEAT" "$PHASE2_JSONL_REL" "$CORR_OUT" "$CONVERSEG_EVAL_ROOT"
)

"$PY_BIN" - <<'PY'
import json
import sys
from pathlib import Path

root = Path(".").resolve()
sys.path.insert(0, str(root))
from converseg_paths import resolve_eval_data_dir, TRAINING_DATA_ROOT

eval_dir = resolve_eval_data_dir()
print(f"OK: resolved eval_dir={eval_dir}")
# Linux 大小写敏感：目录名必须是 ConverSeg，不是 ConVerSeg
parts = {p.lower(): p for p in eval_dir.parts}
if "converseg" in parts and parts["converseg"] != "ConverSeg":
    print(f"FAIL: eval 目录大小写错误: {parts['converseg']}（需要 ConverSeg）")
    sys.exit(1)

eval_files = {
    "sam_seeded": eval_dir / "sam_seeded.jsonl",
    "human_annotated": eval_dir / "human_annotated.jsonl",
}
phase2 = root / "data" / "phase2" / "phase2_mixed.jsonl"
negative = (
    TRAINING_DATA_ROOT
    / "converseg_stage2_data"
    / "conversational_negative_data"
    / "conversational_negative_data.jsonl"
)
if not negative.is_file():
    negative = (
        root
        / "data"
        / "ConverSeg-Training-Data"
        / "converseg_stage2_data"
        / "conversational_negative_data"
        / "conversational_negative_data.jsonl"
    )

missing = [str(p) for p in (*eval_files.values(), phase2) if not p.is_file()]
if missing:
    print("FAIL: 无法核对清单，缺少:")
    for p in missing:
        print(" ", p)
    print("  提示: 正确路径是 data/ConverSeg/data/*.jsonl（小写 v），不是 ConVerSeg")
    sys.exit(1)

def load_keys(path: Path):
    images, ids = set(), set()
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            item = json.loads(line)
            img = item.get("image") or item.get("image_path") or ""
            img = str(img).replace("\\", "/").lstrip("./")
            if img:
                images.add(img)
                images.add(Path(img).name)
            if item.get("id"):
                ids.add(str(item["id"]))
    return n, images, ids

eval_images, eval_ids = set(), set()
for name, path in eval_files.items():
    n, images, ids = load_keys(path)
    eval_images |= images
    eval_ids |= ids
    print(f"OK: eval {name} lines={n} unique_image_keys={len(images)}")

def report(label, path: Path):
    n, images, ids = load_keys(path)
    hit_img = sorted(images & eval_images)
    hit_id = sorted(ids & eval_ids)
    print(f"OK: train {label} lines={n} file={path}")
    if hit_img or hit_id:
        print(f"FAIL: {label} 与测评集相交 image={len(hit_img)} id={len(hit_id)}")
        for item in (hit_img + hit_id)[:12]:
            print(" ", item)
        return 1
    print(f"OK: {label} 与 sam_seeded/human_annotated 无图像或 id 交集")
    return 0

rc = 0
rc |= report("phase2", phase2)
if negative.is_file():
    rc |= report("conversational_negative", negative)
else:
    print(f"FAIL: 缺少负样本清单 {negative}")
    rc = 1
sys.exit(rc)
PY

# shellcheck disable=SC1091
source "$ROOT/scripts/teacher_correction/fix_v1/_env_fix_v1.sh"
FIX_V1_READY=yes
for f in "$SAM_FIN" "$PLM_FIN" "$HEAD_FIN"; do
  if [[ -f "$f" ]]; then
    ok "fix_v1 init $f"
  else
    bad "fix_v1 缺少 Stage-1 权重 $f"
    FIX_V1_READY=no
  fi
done

# shellcheck disable=SC1091
source "$ROOT/scripts/teacher_correction/fix_v2/_env_fix_v2.sh"
FIX_V2_READY=yes
for f in "$SAM_FIN" "$PLM_FIN" "$HEAD_FIN"; do
  if [[ -f "$f" ]]; then
    ok "fix_v2 init $f"
  else
    note "WAIT: fix_v2 还不能提交，缺少 $f"
    FIX_V2_READY=no
  fi
done

note ""
note "FIX_V1_READY=$FIX_V1_READY"
note "FIX_V2_READY=$FIX_V2_READY"
note "训练清单只允许 data/phase2/phase2_mixed.jsonl 和 conversational_negative_data.jsonl。"
note "测评集路径: data/ConverSeg/data/{sam_seeded,human_annotated}.jsonl（HUMAN_REPEAT 必须为 0）。"

if [[ "$FAIL" != "0" || "$FIX_V1_READY" != "yes" ]]; then
  note "不要提交。先处理上面的 FAIL。"
  exit 1
fi

note ""
note "fix_v1 可以提交（必须带 HUMAN_REPEAT=0，且不要与 fix_v2 并行）："
note "sbatch --export=ALL,TEACHER_BACKBONE=teacher7b,HUMAN_REPEAT=0 scripts/teacher_correction/fix_v1/slurm_train_fix_v1.slurm"
if [[ "$FIX_V2_READY" == "yes" ]]; then
  note "fix_v2 可以在 fix_v1 作业结束后提交："
  note "sbatch --export=ALL,TEACHER_BACKBONE=teacher7b,HUMAN_REPEAT=0 scripts/teacher_correction/fix_v2/slurm_train_fix_v2.slurm"
else
  note "fix_v2 现在不能提交。等 teacher7b_fix_v1 的 *_final.torch 出现后，重新运行本脚本。"
fi
exit 0
