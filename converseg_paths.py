"""
ConVerSeg 本地路径约定（服务器已下载权重与数据，不访问 Hugging Face Hub）。

默认布局（相对仓库根目录 REPO_ROOT）:
  student/Qwen2.5-VL-3B/          学生 Qwen 基座（需含 config.json）
  student/sam2_hiera_large.pt       SAM2 Large 权重
  teacher/Qwen2.5-VL-7B/          教师（OPD 等，可选）
  teacher/sam2_hiera_large.pt       可与 student 软链
  sam2/                             SAM2 官方 clone（pip install -e sam2）
                                    yaml 在 sam2/sam2/configs/sam2/*.yaml
                                    训练参数 --model-cfg 用 Hydra 名，非根目录 configs/
  data/phase1/phase1_mixed.jsonl    Phase1 训练清单
  data/phase2/phase2_mixed.jsonl    Phase2 训练清单
  data/ConverSeg-Training-Data/     作者 HF 训练包（已 wget/clone 到本地）
  data/ConverSeg/                   评测：sam_seeded.jsonl、images/、masks/
                                    （亦兼容 data/ConverSeg/data/）
  ckpts/                            训练产出
  pretrain/Qwen2.5-VL-3B-Instruct/  备选 Qwen 目录（若未用 student/）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

STUDENT_DIR = REPO_ROOT / "student"
TEACHER_DIR = REPO_ROOT / "teacher"
DATA_ROOT = REPO_ROOT / "data"
CKPTS_ROOT = REPO_ROOT / "ckpts"

DEFAULT_STUDENT_PLM = STUDENT_DIR / "Qwen2.5-VL-3B"
DEFAULT_STUDENT_SAM = STUDENT_DIR / "sam2_hiera_large.pt"
# build_sam2() / Hydra 配置名（相对已安装的 sam2 Python 包，不是 ConVerSeg 根下 configs/）
DEFAULT_SAM_CONFIG_NAME = "configs/sam2/sam2_hiera_l.yaml"

DEFAULT_TEACHER_PLM = TEACHER_DIR / "Qwen2.5-VL-7B"
DEFAULT_TEACHER_SAM = TEACHER_DIR / "sam2_hiera_large.pt"

PHASE1_JSONL = DATA_ROOT / "phase1" / "phase1_mixed.jsonl"
PHASE2_JSONL = DATA_ROOT / "phase2" / "phase2_mixed.jsonl"
TRAINING_DATA_ROOT = DATA_ROOT / "ConverSeg-Training-Data"

def resolve_eval_data_dir() -> Path:
    """ConVerSeg benchmark 根目录（jsonl 与 images/、masks/ 同级）。"""
    candidates = (
        DATA_ROOT / "ConverSeg",
        DATA_ROOT / "ConverSeg" / "data",
    )
    for root in candidates:
        if (root / "sam_seeded.jsonl").is_file():
            return root.resolve()
    return candidates[0].resolve()


EVAL_DATA_DIR = resolve_eval_data_dir()
EVAL_SAM_SEEDED_JSONL = EVAL_DATA_DIR / "sam_seeded.jsonl"
EVAL_HUMAN_JSONL = EVAL_DATA_DIR / "human_annotated.jsonl"

# 旧 Hub ID -> 本地候选目录（按优先级）
_PLM_HUB_ALIASES: dict[str, list[Path]] = {
    "Qwen/Qwen2.5-VL-3B-Instruct": [
        DEFAULT_STUDENT_PLM,
        REPO_ROOT / "pretrain" / "Qwen2.5-VL-3B-Instruct",
        REPO_ROOT / "pretrain" / "Qwen2.5-VL-3B",
    ],
    "Qwen/Qwen2.5-VL-7B-Instruct": [
        DEFAULT_TEACHER_PLM,
        REPO_ROOT / "pretrain" / "Qwen2.5-VL-7B-Instruct",
    ],
}


def prepare_python_import_path(repo_root: Path | None = None) -> None:
    """
    ConVerSeg 根目录下的 sam2/ 是 SAM2 官方 clone，会与 pip 安装的 sam2 包同名冲突。
    去掉 sys.path 里的 cwd（''），保留显式的项目根路径，以便 import 走 editable 安装。
    """
    root = (repo_root or REPO_ROOT).resolve()
    root_s = str(root)
    shadow_clone = root / "sam2"
    if shadow_clone.is_dir() and (shadow_clone / "setup.py").is_file():
        cleaned: list[str] = []
        for p in sys.path:
            if p in ("", "."):
                continue
            try:
                if Path(p).resolve() == root:
                    continue
            except OSError:
                pass
            cleaned.append(p)
        sys.path[:] = cleaned
    if root_s not in sys.path:
        sys.path.insert(0, root_s)


def apply_offline_env() -> None:
    """禁止 transformers/datasets 访问 Hub（本地已有文件时）。"""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def _is_model_dir(path: Path) -> bool:
    return (path / "config.json").is_file()


def resolve_plm_dir(model_name: str | None = None) -> Path:
    """
    解析 PLM 本地目录。若传入 Hub 名或相对路径，映射到已下载目录；
    找不到则抛出 FileNotFoundError（离线模式）。
    """
    apply_offline_env()

    if not model_name:
        model_name = str(DEFAULT_STUDENT_PLM)

    candidates: list[Path] = []
    if model_name in _PLM_HUB_ALIASES:
        candidates = list(_PLM_HUB_ALIASES[model_name])
    else:
        p = Path(model_name)
        if not p.is_absolute():
            p = REPO_ROOT / p
        candidates = [p]

    for c in candidates:
        if _is_model_dir(c):
            return c.resolve()

    tried = ", ".join(str(x) for x in candidates)
    raise FileNotFoundError(
        "找不到本地 Qwen 目录（需要 config.json）。"
        f" 请求: {model_name!r}；已尝试: {tried}。"
        f" 请将模型放到 {DEFAULT_STUDENT_PLM}/"
    )


def default_plm_dir_str() -> str:
    return str(resolve_plm_dir(None))


def default_sam_ckpt_str() -> str:
    if DEFAULT_STUDENT_SAM.is_file():
        return str(DEFAULT_STUDENT_SAM.resolve())
    legacy = REPO_ROOT / "checkpoints" / "sam2_hiera_large.pt"
    if legacy.is_file():
        return str(legacy.resolve())
    return str(DEFAULT_STUDENT_SAM)


def resolve_sam2_repo_root() -> Path:
    """SAM2 源码根目录（优先 sam2_repo，避免与 import sam2 冲突）。"""
    for name in ("sam2_repo", "sam2"):
        root = REPO_ROOT / name
        if (root / "setup.py").is_file() or (root / "pyproject.toml").is_file():
            return root.resolve()
        if (root / "sam2" / "configs" / "sam2" / "sam2_hiera_l.yaml").is_file():
            return root.resolve()
    return (REPO_ROOT / "sam2").resolve()


def sam2_hiera_l_yaml_on_disk() -> Path | None:
    """Large 模型 yaml 在磁盘上的实际路径（自检用）。"""
    root = resolve_sam2_repo_root()
    candidates = [
        root / "sam2" / "configs" / "sam2" / "sam2_hiera_l.yaml",
        root / "configs" / "sam2" / "sam2_hiera_l.yaml",
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    return None


def default_sam_config_str() -> str:
    return DEFAULT_SAM_CONFIG_NAME
