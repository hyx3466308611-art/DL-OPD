# latet_bulid_phase1 — 从零构建 ConVerSeg 训练数据

在清空 `data/phase1/` 后，用本目录脚本 **按论文 Group1–4 规则** 重建清单、mask（Group1/2）与 Phase1/Phase2 混合 jsonl。

> **Group3 临时策略**：HF Stage1 的 open_vocab **mask 目前不全**，`build_group3.py`、`mix_phase1_train.py`、`mix_phase2_train.py` 默认 **最多使用 10 000 条「文件齐全」样本**。补全 mask 后把 `--max-rows` / `--g3-cap` 改为 `48529` 并重新 mix 即可。

---

## 前置依赖

| 类型 | 路径 |
|------|------|
| COCO train2017 + instances | `data/raw/coco/` |
| COCO train2014（Ref） | `data/raw/coco/train2014/` |
| COCONut-S | `data/raw/coconut/` |
| PACO | `data/raw/paco/` |
| RefCOCO 系 | `data/raw/refer/` |
| HF Stage1 open_vocab | `data/ConverSeg-Training-Data/converseg_stage1_data/open_vocabulary_regions_data/` |
| HF Stage2 对话 ± | `data/ConverSeg-Training-Data/converseg_stage2_data/conversational_*_data/` |

Python 环境需有：`tqdm`、`numpy`、`Pillow`（Group1/2 由 `scripts/build_phase1` 导出）。

---

## 构建前自检（强烈建议先跑）

确认 raw、HF、Python 依赖、磁盘空间、G3/G4 有效行数等，**避免 Group1/2 跑几小时后才因缺文件失败**：

```bash
cd ~/ConverSeg-main
export PATH=$HOME/miniconda311/envs/converseg/bin:$PATH
bash scripts/latet_bulid_phase1/run_preflight.sh
```

退出码 `0` 后再执行 `run_build_all.sh`。快速只查目录、不扫 HF jsonl：`bash scripts/latet_bulid_phase1/run_preflight.sh --skip-hf-scan`。

---

## 一键运行

**Linux / 云端（推荐 tmux）：**

```bash
cd ~/ConverSeg-main
export PATH=$HOME/miniconda311/envs/converseg/bin:$PATH   # 按你的环境调整
bash scripts/latet_bulid_phase1/run_build_all.sh
```

**Windows：**

```powershell
cd D:\Python\Program\ConverSeg-main
.\scripts\latet_bulid_phase1\run_build_all.ps1
```

顺序：`build_group1` → `build_group2` → `build_group3` → `build_group4` → `mix_phase1_train` → `mix_phase2_train`。

---

## 脚本说明

| 脚本 | 作用 |
|------|------|
| `config.py` | 统一路径、论文配额、G3 有效 cap=10000 |
| `build_group1.py` | COCO 200k + **COCONut 140k** + PACO 100k |
| `build_group2.py` | RefCOCO / + / g train ~321k 句 |
| `build_group3.py` | HF Stage1 open_vocab，过滤缺失文件，默认 cap 10k |
| `build_group4.py` | HF Stage2 对话正/负 manifest（**不进 Phase1**） |
| `mix_phase1_train.py` | Phase1 训练混合 jsonl |
| `mix_phase2_train.py` | Phase2 训练混合 jsonl（默认 G4+回放+G3@10k） |
| `run_build_all.sh` / `.ps1` | 一键执行上述全部 |

每个 `.py` 文件 **开头有长注释**，说明构建/融合逻辑与 CLI 参数。

---

## 各 Group 产物位置

### Group 1（Literal，自建 mask）

| 子源 | 清单 | Mask 目录 | 命名规则 |
|------|------|-----------|----------|
| COCO | `data/phase1/manifests/group1_coco.jsonl` | `data/phase1/masks/group1_coco/` | `g1_coco_{imageId}_{catId}.png` |
| COCONut | `data/phase1/manifests/group1_coconut.jsonl` | `data/phase1/masks/group1_coconut/` | `g1_cnut_{imageId}_{catId}.png` |
| PACO | `data/phase1/manifests/group1_paco.jsonl` | `data/phase1/masks/group1_paco/` | `g1_paco_{imageId}_{catId}.png` |

原图：`data/raw/coco/train2017/`（及 COCONut panoptic 在 `data/raw/coconut/coconut_s/`）。

Prompt（论文风格）：`Segment all the {category/part} in the image`。

默认规模：**200 000 + 140 000 + 100 000**。

### Group 2（Ref，自建 mask）

| 清单 | Mask 目录 |
|------|-----------|
| `data/phase1/manifests/group2_ref.jsonl` | `data/phase1/masks/group2_ref/` |

原图：`data/raw/coco/train2014/`。Prompt 为 Ref 数据集原句。

### Group 3（HF Stage1，不复制 png）

| 清单 | 图像 / Mask |
|------|-------------|
| `data/phase1/manifests/group3_open_vocab.jsonl` | `data/ConverSeg-Training-Data/converseg_stage1_data/open_vocabulary_regions_data/images/` 与 `masks/` |

### Group 4（HF Stage2，不复制 png）

| 清单 | 图像 / Mask |
|------|-------------|
| `data/phase2/manifests/group4_conversational_positive.jsonl` | `.../conversational_positive_data/` |
| `data/phase2/manifests/group4_conversational_negative.jsonl` | `.../conversational_negative_data/` |

---

## Phase1 / Phase2 混合结果

| 训练阶段 | 输出文件 | 元数据 | 默认组成 |
|----------|----------|--------|----------|
| **Phase1** | `data/phase1/phase1_mixed.jsonl` | `phase1_mixed.meta.json` | G1 三源 + G2 + **G3≤10k** |
| **Phase2** | `data/phase2/phase2_mixed.jsonl` | `phase2_mixed.meta.json` | G4 正/负 + G1/2 回放 + **G3≤10k** |

训练命令（相对 `data/`）：

```bash
# Phase1
python train.py --data-dir ./data --dataset-jsonl phase1/phase1_mixed.jsonl ...

# Phase2（自建混合）
python train.py --data-dir ./data --dataset-jsonl phase2/phase2_mixed.jsonl ...
```

若 HF **`stage2_mixed.jsonl` 已完整**，Phase2 也可：

```bash
python scripts/latet_bulid_phase1/mix_phase2_train.py --use-hf-stage2-mixed --g3-cap 10000
```

或直接按仓库 `scripts/train_stage2_90k.sh` 使用  
`data/ConverSeg-Training-Data/converseg_stage2_data/stage2_mixed.jsonl`。

---

## 常用单独命令

```bash
# 只重建 Group1，COCONut 断点续导
python scripts/latet_bulid_phase1/build_group1.py --coconut-resume

# Group3 导出全部「文件齐全」行（不 cap）
python scripts/latet_bulid_phase1/build_group3.py --no-cap

# Phase1 mix，HF mask 补全后
python scripts/latet_bulid_phase1/mix_phase1_train.py --g3-cap 48529
```

---

## 与旧 `scripts/build_phase1` 的关系

- **Group1/2 的 mask 几何与论文 prompt** 仍由 `scripts/build_phase1/export_group1_*.py`、`export_group2_ref.py` 实现；本目录通过 subprocess 调用，避免两套逻辑分叉。
- **Group3/4、混合、一键脚本** 在本目录维护；路径常量见 `config.py`。

---

## 预估耗时（云端双 V100 同机，仅供参考）

- Group1 COCO + PACO：数小时量级（视磁盘）
- COCONut 140k：最长，建议 tmux + `--coconut-resume`
- Group2：1–2 小时量级
- Group3/4 + mix：分钟级
