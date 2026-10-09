"""
latet_bulid_phase1 统一路径与配额（与仓库 data/ 布局一致）。

所有构建产物：
  Phase1 清单与 mask：data/phase1/manifests/、data/phase1/masks/
  Phase1 训练混合：   data/phase1/phase1_mixed.jsonl
  Phase2 清单：       data/phase2/manifests/
  Phase2 训练混合：   data/phase2/phase2_mixed.jsonl

原始数据不复制：COCO/Ref/COCONut/PACO 在 data/raw/；HF 在 data/ConverSeg-Training-Data/
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"

# --- raw ---
RAW = DATA_ROOT / "raw"
COCO_TRAIN2017 = RAW / "coco" / "train2017"
COCO_TRAIN2014 = RAW / "coco" / "train2014"
COCO_INST2017 = RAW / "coco" / "annotations2017" / "instances_train2017.json"
COCONUT_JSON = RAW / "coconut" / "coconut_s.json"
COCONUT_MASK_DIR = RAW / "coconut" / "coconut_s"
PACO_TRAIN = RAW / "paco" / "annotations" / "paco_lvis_v1_train.json"
REFER_ROOT = RAW / "refer"

# --- Hugging Face Stage1 (Group3) ---
HF_STAGE1 = DATA_ROOT / "ConverSeg-Training-Data" / "converseg_stage1_data"
HF_G3_ROOT = HF_STAGE1 / "open_vocabulary_regions_data"
HF_G3_JSONL = HF_G3_ROOT / "open_vocabulary_regions_data.jsonl"

# --- Hugging Face Stage2 (Group4 + 可选作者预混) ---
HF_STAGE2 = DATA_ROOT / "ConverSeg-Training-Data" / "converseg_stage2_data"
HF_G4_POS_ROOT = HF_STAGE2 / "conversational_positive_data"
HF_G4_NEG_ROOT = HF_STAGE2 / "conversational_negative_data"
HF_G4_POS_JSONL = HF_G4_POS_ROOT / "conversational_positive_data.jsonl"
HF_G4_NEG_JSONL = HF_G4_NEG_ROOT / "conversational_negative_data.jsonl"
HF_STAGE2_MIXED = HF_STAGE2 / "stage2_mixed.jsonl"

# --- 构建输出 ---
PHASE1_ROOT = DATA_ROOT / "phase1"
PHASE1_MANIFESTS = PHASE1_ROOT / "manifests"
PHASE1_MASKS = PHASE1_ROOT / "masks"
PHASE1_MIXED = PHASE1_ROOT / "phase1_mixed.jsonl"

PHASE2_ROOT = DATA_ROOT / "phase2"
PHASE2_MANIFESTS = PHASE2_ROOT / "manifests"
PHASE2_MIXED = PHASE2_ROOT / "phase2_mixed.jsonl"

# Group1 分片清单（论文 Sec. 6.2.1）
G1_COCO_JSONL = PHASE1_MANIFESTS / "group1_coco.jsonl"
G1_COCONUT_JSONL = PHASE1_MANIFESTS / "group1_coconut.jsonl"
G1_PACO_JSONL = PHASE1_MANIFESTS / "group1_paco.jsonl"
G1_COCO_MASKS = PHASE1_MASKS / "group1_coco"
G1_COCONUT_MASKS = PHASE1_MASKS / "group1_coconut"
G1_PACO_MASKS = PHASE1_MASKS / "group1_paco"

G2_REF_JSONL = PHASE1_MANIFESTS / "group2_ref.jsonl"
G2_REF_MASKS = PHASE1_MASKS / "group2_ref"

G3_OPEN_VOCAB_JSONL = PHASE1_MANIFESTS / "group3_open_vocab.jsonl"

G4_POS_JSONL = PHASE2_MANIFESTS / "group4_conversational_positive.jsonl"
G4_NEG_JSONL = PHASE2_MANIFESTS / "group4_conversational_negative.jsonl"

# 论文规模目标
QUOTA_G1_COCO = 200_000
QUOTA_G1_COCONUT = 140_000
QUOTA_G1_PACO = 100_000  # 论文目标；本地 export 在 min-area=256 下取 69614
QUOTA_G1_PACO_USED = 69_614
PACO_MIN_AREA = 256
QUOTA_G2_REF = 321_327
QUOTA_G3_PAPER = 48_529

# Group3：与论文一致；build/mix 默认用满「磁盘上齐全」的有效行（通常 48529）
QUOTA_G3_EFFECTIVE_MASKS = QUOTA_G3_PAPER

# Group4 论文规模（以 HF jsonl 行为准，混合时可再 cap）
QUOTA_G4_POS = 106_127
QUOTA_G4_NEG = 106_125

# Phase2 自建混合：从 Group1+2 回放的大致规模（不含 G3；G3 单独 cap）
QUOTA_PHASE2_REPLAY_COCO = 26_000
QUOTA_PHASE2_REPLAY_COCONUT = 26_000
QUOTA_PHASE2_REPLAY_PACO = 26_000
QUOTA_PHASE2_REPLAY_REF = 26_000

# 调用 scripts/build_phase1 里已验证的 export 脚本
LEGACY_BUILD_DIR = REPO_ROOT / "scripts" / "build_phase1"

DEFAULT_SEED = 42
