# ConverSeg Model Configuration Guide

This document explains the full training/evaluation pipeline, which modules are trained, how SAM2 Small/Large is selected, and how the final model is composed in code.

---

## 1. End-to-End Pipeline (What You Run)

A complete train-and-test workflow has **four main parts** (plus optional smoke test):


| Step         | Name              | Purpose                                                                                           |
| ------------ | ----------------- | ------------------------------------------------------------------------------------------------- |
| 0 (optional) | Smoke run         | Short training (e.g. 200–1000 steps) to verify GPU, data paths, and checkpoint saving             |
| 1            | Stage-1 training  | Pretrain on literal / open-vocabulary region data                                                 |
| 2            | Stage-2 training  | Fine-tune on conversational positive (and optionally negative) data, usually resumed from Stage-1 |
| 3            | Evaluation / Demo | Test with `eval.py` and/or `demo.py` using the Stage-2 checkpoints                                |


Paper reference (author settings):

- Stage-1 (Phase 1): **100,000** steps  
- Stage-2 (Phase 2): **90,000** steps  
- Batch size 6, gradient accumulation 8, lr `1e-4`, one A100 ~96 hours total

Practical course / first full run recommendation:

- Stage-1: **10,000–20,000** steps  
- Stage-2: **10,000–20,000** steps  
- Use smaller batch + larger accumulation if VRAM is limited (e.g. `--batch-size 1 --acc 8`)

---



## 2. What Is the “Full Model”?

ConverSeg-Net is **not** a single new network from scratch. It is a fusion of:

1. **SAM2** (image encoder + prompt encoder + mask decoder)
2. **Language adapter (PLM)** built on **Qwen2.5-VL-3B-Instruct** (+ LoRA)
3. Lightweight projections that map language features into SAM2 prompt space



### Data flow (inference / one training step)

```text
Image ──► SAM2 image encoder (frozen during training) ──► image embeddings
                                                              │
Text (+ image to Qwen) ──► Qwen2.5-VL + LoRA + adapters ──────┼──► sparse / dense prompts
                                                              │
                                                              ▼
                                                    SAM2 mask decoder
                                                              │
                                                              ▼
                                                           mask logits
```



### Trainable vs frozen (as implemented in `train.py`)


| Module                                          | Trainable?                                 |
| ----------------------------------------------- | ------------------------------------------ |
| SAM2 image encoder                              | **Frozen**                                 |
| SAM2 prompt encoder                             | **Trainable**                              |
| SAM2 mask decoder                               | **Trainable**                              |
| Qwen backbone                                   | Mostly frozen; **LoRA** adapters trainable |
| LanguageAdapter projection heads (sparse/dense) | **Trainable**                              |


So you are not “training three separate products”. You train **one ConverSeg system** whose learnable parts are: SAM2 prompt encoder + mask decoder + PLM adapters/LoRA.

---



## 3. Checkpoints You Save / Load

After training, typical outputs under `--out` look like:


| File / folder              | Meaning                                                                                                      |
| -------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `<name>_<step>.torch`      | Fine-tuned SAM2 weights (prompt encoder + mask decoder updated; full SAM2 state dict as saved by the script) |
| `<name>_plm_<step>.torch`  | Language adapter (PLM) state dict                                                                            |
| `lora_plm_adapter_<step>/` | LoRA adapter weights for Qwen                                                                                |
| `tb/<name>/`               | TensorBoard logs                                                                                             |
| `val/step_*.png`           | Validation visualization panels                                                                              |


For demo / eval you need all three:

- `--final_ckpt` → SAM2 `.torch`  
- `--plm_ckpt` → PLM `.torch`  
- `--lora_ckpt` → LoRA folder  
- plus base SAM2 init `--base_ckpt` / `--checkpoint` (e.g. `sam2_hiera_large.pt`) and local Qwen path

Released official weights example:

- `ConverSeg-Net_sam2_90000.torch`  
- `ConverSeg-Net_plm_90000.torch`  
- `lora_plm_adapter_90000/`

---



## 4. SAM2 Small vs Large (Command Choice, No Code Rewrite)

This chooses the **SAM2 backbone size**. YAML and `.pt` must match.


| Variant | Hydra config name                | Weight file           |
| ------- | -------------------------------- | --------------------- |
| Small   | `configs/sam2/sam2_hiera_s.yaml` | `sam2_hiera_small.pt` |
| Large   | `configs/sam2/sam2_hiera_l.yaml` | `sam2_hiera_large.pt` |


Notes:

- `train.py` defaults to **Small**; override via CLI if you use Large.  
- Official ConverSeg-Net-3B release is aligned with **Large**.  
- You do **not** need to edit Python to switch sizes—only CLI flags.  
- Small uses less VRAM; Large is usually stronger and matches the released demo weights.

Example (Large):

```bash
--model-cfg configs/sam2/sam2_hiera_l.yaml \
--checkpoint ./new_checkpoints/sam2_hiera_large.pt
```

Example (Small):

```bash
--model-cfg configs/sam2/sam2_hiera_s.yaml \
--checkpoint ./new_checkpoints/sam2_hiera_small.pt
```

---



## 5. Code Layout (Where the “Model” Lives)


| Path                         | Role                                                                |
| ---------------------------- | ------------------------------------------------------------------- |
| `train.py`                   | Training loop, freeze/unfreeze SAM2 parts, optimizer, checkpointing |
| `models/language_adapter.py` | Qwen2.5-VL backbone + LoRA + projection to SAM2 prompt dims         |
| `sam2_repo/sam2/`            | Upstream SAM2 package (`build_sam2`, predictors, decoder, etc.)     |
| `demo.py`                    | Single-image / interactive inference                                |
| `eval.py`                    | Benchmark evaluation (HF dataset or JSON)                           |
| `utils/util_functions.py`    | I/O, IoU, visualization helpers                                     |


Important: the repository folder should be named `sam2_repo` (or similar), not `sam2`, to avoid Python package shadowing when running from the project root.

---



## 6. Dataset Layout vs Training Stages

Hugging Face training data keeps this structure (do not flatten unless you want to):

```text
ConverSeg-Training-Data/
  converseg_stage1_data/
    open_vocabulary_regions_data/
      open_vocabulary_regions_data.jsonl
      images/
      masks/
  converseg_stage2_data/
    conversational_positive_data/
      conversational_positive_data.jsonl
      images/
      masks/
    conversational_negative_data/
      ...
    open_vocabulary_regions_data/
      ...
```

Each JSONL line matches `train.py` fields:

```json
{"image":"images/....png","mask_merged":"masks/....png","prompt":"..."}
```

On AutoDL, data may live under `/root/autodl-fs/data/...` while code lives under `/root/autodl-tmp/ConverSeg-main`. A symlink is enough:

```bash
mkdir -p data
ln -sfn /root/autodl-fs/data/ConverSeg-Training-Data ./data/ConverSeg-Training-Data
```

---



## 7. Recommended Commands (Skeleton)

Assume project root and Large SAM2 + local Qwen under `./pretrain/Qwen2.5-VL-3B-Instruct`.

### Stage-1

```bash
python train.py \
  --data-dir ./data/ConverSeg-Training-Data/converseg_stage1_data/open_vocabulary_regions_data \
  --dataset-jsonl open_vocabulary_regions_data.jsonl \
  --model-cfg configs/sam2/sam2_hiera_l.yaml \
  --checkpoint ./new_checkpoints/sam2_hiera_large.pt \
  --plm-model-name ./pretrain/Qwen2.5-VL-3B-Instruct \
  --device cuda \
  --steps 10000 \
  --batch-size 1 \
  --acc 8 \
  --out ./ckpts/stage1 \
  --name ConverSeg_stage1
```



### Stage-2 (resume Stage-1)

```bash
python train.py \
  --data-dir ./data/ConverSeg-Training-Data/converseg_stage2_data/conversational_positive_data \
  --dataset-jsonl conversational_positive_data.jsonl \
  --model-cfg configs/sam2/sam2_hiera_l.yaml \
  --checkpoint ./new_checkpoints/sam2_hiera_large.pt \
  --plm-model-name ./pretrain/Qwen2.5-VL-3B-Instruct \
  --resume-sam-path ./ckpts/stage1/ConverSeg_stage1_<STEP>.torch \
  --resume-plm-path ./ckpts/stage1/ConverSeg_stage1_plm_<STEP>.torch \
  --resume-lora-path ./ckpts/stage1/lora_plm_adapter_<STEP> \
  --device cuda \
  --steps 10000 \
  --batch-size 1 \
  --acc 8 \
  --out ./ckpts/stage2 \
  --name ConverSeg_stage2
```



### Demo / Eval

Point `demo.py` / `eval.py` `model_name` to the local Qwen folder (or ensure network HF access). Then pass Stage-2 checkpoints via `--final_ckpt`, `--plm_ckpt`, `--lora_ckpt`.

---



## 8. One-Sentence Summary

**Full training** = Stage-1 then Stage-2 on one ConverSeg stack (trainable SAM2 prompt/mask heads + Qwen LoRA/adapters).  
**Full model in code** = `LanguageAdapter` + SAM2 (`build_sam2`) wired in `train.py` / `demo.py` / `eval.py`.  
**Small/Large** = SAM2 size selected only by CLI config + checkpoint paths.

---



## 9. Checklist Before Training

- [ ] `conda activate converseg` and `torch.cuda.is_available() == True`  
- [ ] `pip install -e ./sam2_repo` works (`from sam2.build_sam import build_sam2`)  
- [ ] Local Qwen at `./pretrain/Qwen2.5-VL-3B-Instruct` (or your actual path)  
- [ ] Matching SAM2 yaml + `.pt` (Large recommended if matching official release)  
- [ ] Dataset symlink or absolute `--data-dir` pointing to a folder that contains the JSONL + `images/` + `masks/`  
- [ ] Enough disk for `./ckpts/`  

For questions, see also the project root `README.md` and the paper abstract file in the repository root.

---
---

# 中文注释（Chinese Annotations）

> 说明：以下为上文英文内容的汉语对照注释，**不替代、不修改**上方英文原文。阅读时可英文与中文分段对照。

---

## 中文 · 文档用途

本文说明：完整训练/评测流程、哪些模块会训练、如何选择 SAM2 的 Small/Large，以及最终模型在代码中如何组成。

---

## 中文 · 1. 端到端流程（你实际要跑什么）

完整的「训练 + 测试」流程主要有 **四部分**（外加可选冒烟测试）：

| 步骤 | 名称 | 作用 |
|------|------|------|
| 0（可选） | 冒烟运行 Smoke run | 短训（如 200–1000 step），检查 GPU、数据路径、能否存 checkpoint |
| 1 | Stage-1 训练 | 在字面/开放词表区域数据上预训练 |
| 2 | Stage-2 训练 | 在对话式正样本（可选负样本）上微调，通常从 Stage-1 权重继续（resume） |
| 3 | 评测 / Demo | 用 Stage-2 的权重跑 `eval.py` 和/或 `demo.py` |

论文作者设定：

- Stage-1（Phase 1）：**100,000** steps  
- Stage-2（Phase 2）：**90,000** steps  
- batch size 6，梯度累积 8，学习率 `1e-4`，单卡 A100 全程大约 96 小时  

课程作业 / 第一次完整跑的建议：

- Stage-1：**10,000–20,000** steps  
- Stage-2：**10,000–20,000** steps  
- 显存不够时：减小 batch、增大累积（例如 `--batch-size 1 --acc 8`）

---

## 中文 · 2. 「完整模型」是什么？

ConverSeg-Net **不是**从零写的一个全新大网络，而是融合：

1. **SAM2**（图像编码器 + prompt 编码器 + mask 解码器）  
2. **语言适配器（PLM）**：基于 **Qwen2.5-VL-3B-Instruct**（加 LoRA）  
3. **轻量投影层**：把语言特征映射到 SAM2 的 prompt 空间  

### 数据流（推理 / 训练一步）

```text
图像 ──► SAM2 图像编码器（训练时冻结） ──► 图像特征
                                              │
文本（+ 图像进 Qwen）──► Qwen2.5-VL + LoRA + 适配器 ──┼──► 稀疏/稠密 prompt
                                              │
                                              ▼
                                        SAM2 mask 解码器
                                              │
                                              ▼
                                           mask 预测
```

### 哪些训练、哪些冻结（以 `train.py` 为准）

| 模块 | 是否训练 |
|------|----------|
| SAM2 图像编码器 | **冻结** |
| SAM2 prompt 编码器 | **训练** |
| SAM2 mask 解码器 | **训练** |
| Qwen 主干 | 基本冻结；**LoRA** 可训练 |
| LanguageAdapter 投影头（稀疏/稠密） | **训练** |

所以你不是在「分别训练三个产品」，而是训练 **一套 ConverSeg 系统**；真正更新的主要是：SAM2 的 prompt encoder + mask decoder，以及 PLM 的适配器/LoRA。

---

## 中文 · 3. 保存 / 加载的权重文件

训练后，`--out` 目录下典型输出：

| 文件 / 目录 | 含义 |
|-------------|------|
| `<name>_<step>.torch` | 微调后的 SAM2 权重（脚本保存的 state dict） |
| `<name>_plm_<step>.torch` | 语言适配器（PLM）权重 |
| `lora_plm_adapter_<step>/` | Qwen 的 LoRA 适配器 |
| `tb/<name>/` | TensorBoard 日志 |
| `val/step_*.png` | 验证可视化拼图 |

跑 demo / eval 时这三样都要：

- `--final_ckpt` → SAM2 的 `.torch`  
- `--plm_ckpt` → PLM 的 `.torch`  
- `--lora_ckpt` → LoRA 文件夹  
- 另外还要 SAM2 初始权重 `--base_ckpt` / `--checkpoint`（如 `sam2_hiera_large.pt`），以及本地 Qwen 路径  

官方发布权重示例：

- `ConverSeg-Net_sam2_90000.torch`  
- `ConverSeg-Net_plm_90000.torch`  
- `lora_plm_adapter_90000/`  

---

## 中文 · 4. SAM2 Small 与 Large（命令选择，不用改代码）

这是选 **SAM2 骨干尺寸**。YAML 配置和 `.pt` 权重必须配对。

| 型号 | Hydra 配置名 | 权重文件 |
|------|--------------|----------|
| Small | `configs/sam2/sam2_hiera_s.yaml` | `sam2_hiera_small.pt` |
| Large | `configs/sam2/sam2_hiera_l.yaml` | `sam2_hiera_large.pt` |

注意：

- `train.py` **默认是 Small**；用 Large 时要在命令行改掉。  
- 官方 ConverSeg-Net-3B 与 **Large** 对齐。  
- 切换尺寸 **不用改 Python**，只改命令行参数。  
- Small 更省显存；Large 通常更强，也更贴官方 demo 权重。  

Large 示例 / Small 示例见上方英文第 4 节命令块（保持英文原样）。

---

## 中文 · 5. 代码布局（「模型」在哪些文件里）

| 路径 | 作用 |
|------|------|
| `train.py` | 训练循环、冻结/解冻 SAM2、优化器、存盘 |
| `models/language_adapter.py` | Qwen2.5-VL + LoRA + 投到 SAM2 维度 |
| `sam2_repo/sam2/` | 上游 SAM2 包（`build_sam2`、预测器、解码器等） |
| `demo.py` | 单图 / 交互推理 |
| `eval.py` | 基准评测（HF 数据集或 JSON） |
| `utils/util_functions.py` | 读写、IoU、可视化 |

重要：仓库目录建议叫 `sam2_repo`（或类似名字），**不要**叫 `sam2`，否则在工程根目录运行时容易和 Python 包名冲突（shadow）。

---

## 中文 · 6. 数据集结构与训练阶段

Hugging Face 训练数据保持如下结构（除非你想改，否则不要强行压成单层）：

```text
ConverSeg-Training-Data/
  converseg_stage1_data/          # 阶段1：偏字面、开放词表
  converseg_stage2_data/          # 阶段2：对话式正/负样本等
```

每行 JSONL 字段与 `train.py` 一致：`image`、`mask_merged`、`prompt`。

在 AutoDL 上：数据常在 `/root/autodl-fs/data/...`，代码在 `/root/autodl-tmp/ConverSeg-main`。用软链接即可（见英文第 6 节命令），不必把数据复制进工程。

---

## 中文 · 7. 推荐命令骨架

前提：工程根目录；用 Large SAM2；本地 Qwen 在 `./pretrain/Qwen2.5-VL-3B-Instruct`。

- **Stage-1**：用 stage1 的 `open_vocabulary_regions_data` 开训，输出到 `./ckpts/stage1`  
- **Stage-2**：用 stage2 的 `conversational_positive_data`，并通过 `--resume-sam-path` / `--resume-plm-path` / `--resume-lora-path` 加载 Stage-1 权重，输出到 `./ckpts/stage2`  
- **Demo / Eval**：把 `demo.py` / `eval.py` 里的 `model_name` 指到本地 Qwen（或保证能连 HF），再传入 Stage-2 的三个 checkpoint  

具体命令块见上方英文第 7 节（请保持英文命令原样复制使用）。

---

## 中文 · 8. 一句话总结

- **完整训练** = 先 Stage-1 再 Stage-2，训的是同一套 ConverSeg（可训的 SAM2 prompt/mask 头 + Qwen LoRA/适配器）。  
- **代码里的完整模型** = `LanguageAdapter` + SAM2（`build_sam2`），在 `train.py` / `demo.py` / `eval.py` 中接起来。  
- **Small/Large** = 只通过命令行配置与权重路径选择 SAM2 尺寸。  

---

## 中文 · 9. 开训前检查清单

- [ ] 已 `conda activate converseg`，且 `torch.cuda.is_available() == True`  
- [ ] `pip install -e ./sam2_repo` 成功，能 `from sam2.build_sam import build_sam2`  
- [ ] 本地 Qwen 在 `./pretrain/Qwen2.5-VL-3B-Instruct`（或你的实际路径）  
- [ ] SAM2 的 yaml 与 `.pt` 匹配（对齐官方发布时建议 Large）  
- [ ] 数据软链接或绝对路径 `--data-dir` 指向含 JSONL + `images/` + `masks/` 的目录  
- [ ] `./ckpts/` 有足够磁盘空间  

更多信息见工程根目录 `README.md` 与论文双语文本文件。