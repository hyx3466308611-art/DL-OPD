"""Default paths for Phase-1 dataset build (Group1+2+3 → train.py jsonl)."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"

RAW = DATA_ROOT / "raw"
COCO_TRAIN2017 = RAW / "coco" / "train2017"
COCO_TRAIN2014 = RAW / "coco" / "train2014"
COCO_INST2017 = RAW / "coco" / "annotations2017" / "instances_train2017.json"
COCONUT_JSON = RAW / "coconut" / "coconut_s.json"
COCONUT_MASK_DIR = RAW / "coconut" / "coconut_s"
PACO_TRAIN = RAW / "paco" / "annotations" / "paco_lvis_v1_train.json"
REFER_ROOT = RAW / "refer"

HF_G3_ROOT = (
    DATA_ROOT
    / "ConverSeg-Training-Data"
    / "converseg_stage1_data"
    / "open_vocabulary_regions_data"
)
HF_G3_JSONL = HF_G3_ROOT / "open_vocabulary_regions_data.jsonl"

# Built artifacts (masks exported here; images stay under raw/ or HF)
PHASE1_ROOT = DATA_ROOT / "phase1"
MANIFEST_DIR = PHASE1_ROOT / "manifests"
MASK_ROOT = PHASE1_ROOT / "masks"

# Paper-scale targets (tunable via CLI)
QUOTA_GROUP1_TOTAL = 440_000
QUOTA_GROUP1_COCO = 200_000
QUOTA_GROUP1_COCONUT = 140_000
QUOTA_GROUP1_PACO = 100_000
QUOTA_GROUP2 = 321_000  # ~all Ref train sentences
QUOTA_GROUP3 = 48_529

PROMPT_COCO = "name"  # name | segment_the
