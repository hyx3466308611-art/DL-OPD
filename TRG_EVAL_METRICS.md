# 纠错教师 TRG 测评说明

本文档说明 **Stage 1 纠错教师（Correction Teacher）** 在 `eval_teacher_recovery.py` 中的测评流程、**TRG（Teacher Recovery Gain）** 的含义与计算方式，以及输出文件里各字段的含义。实现代码见 `eval_teacher_recovery.py` 与 `data/mask_corrupter.py`。

---

## 1. 测评在测什么？

纠错教师的任务不是「从零分割」，而是：**在已有（可能很差的）掩码提示下，把分割结果纠正得更接近真值（GT）**。

因此测评流程是：

1. 使用 **ConVerSeg 官方测评集**（与学生/主模型 benchmark 相同的数据根目录，默认 `data/ConVerSeg/`）。
2. 对每条样本读取 **原图 + GT 掩码 + 文本 prompt**。
3. 在内存中 **在线腐化 GT**，得到「错误输入掩码」，**不**把腐化结果写入磁盘。
4. 把腐化掩码作为 dense prompt 输入纠错教师，得到预测掩码。
5. 比较 **输入 IoU**、**输出 IoU**，并计算单样本 **TRG = 输出 IoU − 输入 IoU**。

TRG 越大，表示教师相对「腐化输入」恢复得越好；TRG 为负表示预测比腐化输入还差。

---

## 2. 测评集与数据划分

| 项目 | 说明 |
| --- | --- |
| 默认数据根 | `data/ConVerSeg/`（可通过 `--data-root` 覆盖；未指定时由 `resolve_eval_data_dir()` 解析） |
| 默认 splits | `sam_seeded,human_annotated`（对应 `{split}.jsonl`） |
| 规模（全量） | `sam_seeded.jsonl` 约 1194 条 + `human_annotated.jsonl` 约 493 条 ≈ **1687** 条 |
| JSONL 字段 | `image` / `image_path`、`mask` / `mask_merged`、`prompt`（缺省为 `"segment"`） |
| 子集调试 | `--max-items N` 只读每个 split 的前 N 条 |

与 **Phase 2 训练数据**（如 `data/phase2/phase2_mixed.jsonl`）无关；TRG 专用测评集是 **ConVerSeg 目录下的 jsonl**，不是训练 manifest。

---

## 3. 测评时是否「随机一种腐化算子」？

**是**。随机性受 **`--eval-stage`（A / B / C / MIX）** 约束；与训练时「按 step 切换课程」不同。

### 3.0 论文主协议：Mixed-TRG（推荐 / 默认）

| 项 | 设定 |
| --- | --- |
| 开关 | `--eval-stage MIX`（Slurm 默认 `EVAL_STAGE=MIX`） |
| 抽样 | 每条样本先用独立 `Random(42)` 按 **A:B:C = 1:2:2** 抽一个 stage，再对该 stage 调用 `MaskCorrupter.corrupt` |
| 腐化 RNG | `MaskCorrupter(seed=42)`（与 stage 抽样流分离，均可复现） |
| **主表只报** | **mean TRG**、**positive_fraction**、**mean output IoU**（可选 mean input IoU） |
| 附录 | JSON 内 `by_stage` / `by_input_iou_bin`（>0.8 / 0.4–0.8 / <0.4）/ `by_op_family` / `by_split` |
| 冻结门槛 | 同第 7 节：mean TRG > 0.03 且 pos_frac ≥ 0.55 |
| 产出 | `eval_recovery_stageMIX/trg_report.json`、`teacher_frozen_ready_stageMIX.flag` |

固定 A/B/C 单档测评仍可用于附录难度扫描，**不作为论文主结论**。

### 3.0.1 第二套协议：Per-op 诊断（改进模型用）

| 项 | 设定 |
| --- | --- |
| 开关 | `--force-op <op>` 或 `--force-op ALL`（与 `MIX` **互斥**） |
| severity | 建议 `--eval-stage C`（固定难度，不抽 MIX） |
| 行为 | 强制单一算子；**关闭**随机 `mask_dropout` 与 compound |
| 算子表 | `morph, boundary, translate, drop_comp, partial, false_pos, invert_halluc, elastic, mask_dropout, empty_gt` |
| `empty_gt` | 仅评 **GT 为空** 的样本 |
| 产出 | `eval_recovery_per_op/<op>/trg_report.json` + `per_op_summary.json` |
| 脚本 | `scripts/teacher_correction/fix_v1/slurm_eval_trg_per_op.slurm` |

说明文档：[scripts/teacher_correction/TRG_TWO_PROTOCOLS.md](scripts/teacher_correction/TRG_TWO_PROTOCOLS.md)。

### 3.1 固定随机种子

- 测评脚本构造 `MaskCorrupter(seed=42)`；MIX 另建 `Random(42)` 只负责抽 stage。
- 腐化器内部使用 **单一顺序推进** 的 `random.Random(42)`：按 **jsonl 遍历顺序** 每调用一次 `corrupt()` 消耗一段随机状态。
- 因此：**相同代码版本、相同 splits、相同 `--eval-stage`、相同样本顺序** 下可复现；改 split 顺序或 `--max-items` 会改变后续样本的腐化。

### 3.2 单条样本的腐化流程（`MaskCorrupter.corrupt`）

对每条样本，在 **已对齐的 GT 二值掩码** 上执行：

1. **（仅 MIX）** 按 1:2:2 抽 stage ∈ {A,B,C}。
2. **采样 severity**：在 Stage 配置的 `[severity_lo, severity_hi]` 上均匀随机。
3. **mask_dropout**（可选）：以 `mask_dropout_prob` 概率 **整 mask 置零**（算子记为 `mask_dropout`）；Stage A 为 0%，B 为 8%，C 为 20%。
4. **空 GT**：若 GT 全空，不腐化，记 `empty_gt`，meta 中 `input_iou=1.0`。
5. **主算子**：从该 Stage 的 `allowed_ops` 中 **`rng.choice` 均匀随机选 1 个**，施加到 GT。
6. **复合腐化**：以 `compound_prob` 再随机选 **第二个** 算子（severity 略降），算子名记为 `op1+op2`。
7. **兜底**：非 Stage C 且腐化后全空时，用较轻的 `morph` 避免空 mask。

测评脚本 **不会** 使用训练用的 `curriculum_stage_from_step()`；固定档用 `CurriculumStage(args.eval_stage)`，MIX 用 `sample_mix_stage`。

### 3.3 各 Stage 的算子池与超参

| Stage | severity 范围 | mask_dropout | compound_prob | allowed_ops |
| ---: | --- | ---: | ---: | --- |
| **A** | 0.15 – 0.45 | 0% | 15% | morph, boundary, translate |
| **B** | 0.35 – 0.72 | 8% | 35% | + drop_comp, partial |
| **C** | 0.55 – 0.95 | 20% | 55% | + false_pos, invert_halluc, elastic |

### 3.4 八种腐化算子（简要语义）

| 算子名 | 作用（直观） |
| --- | --- |
| `morph` | 随机腐蚀或膨胀，核大小随目标面积与 severity 自适应 |
| `boundary` | 开/闭运算破坏边界；severity 较高时可能叠加轻度 elastic |
| `translate` | 按 bbox 尺度平移 mask |
| `drop_comp` | 随机去掉若干连通域 |
| `partial` | 在凸包区域内擦掉一块，模拟局部缺失 |
| `false_pos` | 在背景或合成区域贴假阳性块（Stage C） |
| `invert_halluc` | 假阳性 blob 或强烈收缩（Stage C） |
| `elastic` | 弹性形变（Stage C；boundary 高 severity 时也可能出现） |
| `mask_dropout` | 整 mask 清零（非 allowed_ops 内，由 dropout 分支触发） |
| `empty_gt` | GT 为空时的占位 meta |

训练与测评 **共用同一套** `MaskCorrupter` 实现；测评 **不落盘** 腐化 mask，仅用于当次 forward。

---

## 4. GT 与输入掩码在评测中的预处理

与 `infer_one()` 一致：

1. **读图**：`read_img_mask(..., max_dim=1024)`（默认 `--max-dim 1024`）。
2. **GT**：`gt = erode(bin_mask(ann), er_k=5, er_it=1)`（默认 `--er-k 5 --er-it 1`），与主评测管线中对 GT 的侵蚀约定一致。
3. **腐化**：在 **eroded GT** 上调用 `corrupter.corrupt(gt, stage)`，得到 `corrupt` 与 `meta`（含 `op`、`severity`、`stage`、腐化器内部的 `input_iou` 估计）。
4. **对齐画布**：腐化 mask 若与 SAM 读图后的 H×W 不一致，用 **双线性** resize 到与图像相同尺寸，再作为 **软 mask**（0–1 float）送入 `TeacherCorrectionHead`。
5. **输入 IoU（正式指标）**：在 **与 GT 相同分辨率** 下，对腐化 mask 二值化 `(corrupt > 0.5)` 与 `gt_u8` 算 IoU（见下节公式）。**以评测脚本重算为准**，不必等于 `meta["input_iou"]`（resize 前后可能有细微差别）。

---

## 5. 教师前向与预测掩码

对每条样本：

1. SAM2 image encoder 提特征；**腐化掩码**经 correction head（PLM + 门控融合）生成 sparse/dense prompt。
2. SAM mask decoder 输出 **单 mask** logits，`sigmoid > 0.5` 二值化。
3. 预测 mask **最近邻** resize 到 GT 的 `(h, w)`，再与 GT 算 **输出 IoU**。

---

## 6. IoU 与 TRG 的定义与计算

### 6.1 掩码 IoU（`data.mask_corrupter.mask_iou`）

对二值掩码 \(A, B\)（前景为 1）：

\[
\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}
\]

实现为像素级 `logical_and` / `logical_or` 计数。若 **并集为空**（两者皆空），定义为 **1.0**；若仅一方为空则 **0.0**。

### 6.2 单样本指标（每条 jsonl 一行）

| 符号 / 字段 | 计算 |
| --- | --- |
| **input_iou** | \(\text{IoU}(\text{CorruptedMask}_{>0.5}, \text{GT})\) |
| **output_iou** | \(\text{IoU}(\text{TeacherPred}_{>0.5}, \text{GT})\) |
| **trg** | \(\text{output\_iou} - \text{input\_iou}\) |

代码对应（`eval_teacher_recovery.py`）：

```python
input_iou = mask_iou((corrupt_u8 > 0.5).astype(np.uint8), gt_u8)
output_iou = mask_iou(pred_r, gt_u8)
trg = output_iou - input_iou
```

### 6.3 TRG 反映什么？

- **TRG 是「相对增益」**：在 **同样腐化输入** 下，教师输出比「直接把腐化 mask 当结果」好多少（用 IoU 衡量）。
- **mean TRG**：全测评集 `trg` 的算术平均；整体纠错能力。
- **positive_fraction**：`trg > 0` 的样本比例；是否 **大多数样本** 都有正增益（避免少数难例拉低均值时仍「偶尔有用」）。
- **Mean Input / Output IoU**（终端打印，**未写入** `trg_report.json` 顶层）：分别对所有样本的 `input_iou`、`output_iou` 求平均，便于看「腐化有多狠」以及「教师绝对质量」。

示例解读（Stage C，示意）：

- mean input IoU ≈ 0.46、mean output IoU ≈ 0.77 → 腐化很重，但教师显著拉回。
- mean TRG ≈ 0.31 → 平均每条样本 IoU 提升约 0.31。
- positive_fraction ≈ 80% → 约八成样本 TRG > 0。

某些算子（如 `elastic`、`boundary`）上 mean TRG 可能略负，表示该腐化类型仍难纠正；终端会按算子聚合打印 **top 12**（按样本数），**同样不在 JSON 顶层**。

---

## 7. 全局汇总指标与「是否冻结教师」门槛

脚本在全部样本跑完后计算：

| 指标 | 含义 | 默认门槛 |
| --- | --- | --- |
| **mean_trg** | 所有样本 `trg` 的均值 | `> --trg-mean-threshold`（默认 **0.03**） |
| **positive_fraction** | `mean(trg > 0)` | `>= --trg-positive-frac`（默认 **0.55**） |

**同时满足** 则 `ready = true`，否则 `ready = false`。门槛可通过命令行调整。

---

## 8. 输出文件有哪些？每个字段什么意思？

测评 **主要写两个文件**（路径可配置）：

- `--out-dir` → **`trg_report.json`**
- `--flag-path` → **`teacher_frozen_ready.flag`**

终端还会打印：样本数、mean input/output IoU、mean TRG、positive fraction、**按 split** 与 **按腐化算子（top 12）** 的 mean TRG——这些 **除 rows 内 per-sample 外，split/op 聚合未单独写入 JSON**。

### 8.1 `trg_report.json`（完整字段）

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `mean_trg` | float | 全集 TRG 均值 |
| `positive_fraction` | float | TRG > 0 的比例，∈ [0, 1] |
| `threshold_mean` | float | 本次运行使用的 mean TRG 门槛（默认 0.03） |
| `threshold_pos_frac` | float | 本次运行使用的 positive fraction 门槛（默认 0.55） |
| `eval_stage` | string | `"A"` / `"B"` / `"C"` / `"MIX"` |
| `mean_input_iou` / `mean_output_iou` | float | 全集均值（主表用） |
| `mix_weights` | string\|null | MIX 时为 `"A:B:C=1:2:2"` |
| `by_stage` / `by_input_iou_bin` / `by_op_family` / `by_split` | object | 附录分层汇总 |
| `rows` | array | 每条有效样本一条记录（见下表） |

**`rows[]` 每条记录：**

| 字段 | 含义 |
| --- | --- |
| `id` | jsonl 中的样本 id（若有） |
| `split` | `sam_seeded` 或 `human_annotated` |
| `op` | 腐化算子名，可能为复合名如 `morph+boundary` |
| `stage` | 与 `eval_stage` 相同（冗余记录，便于按行过滤） |
| `input_iou` | 腐化输入 vs GT 的 IoU |
| `output_iou` | 教师预测 vs GT 的 IoU |
| `trg` | `output_iou - input_iou` |

**未写入 JSON 但可自算：**

- 全集 mean input/output IoU：对 `rows` 中 `input_iou` / `output_iou` 求平均。
- 按 `split` / `op` 分组 mean TRG：对 `rows` 分组聚合。

### 8.2 `teacher_frozen_ready.flag`

| 字段 | 含义 |
| --- | --- |
| `ready` | 是否通过 TRG 双门槛（见第 7 节） |
| `mean_trg` | 与 report 相同 |
| `positive_fraction` | 与 report 相同 |
| `mask_gate` | 加载的 correction head 当前 **门控值** `head.gate_value`（掩码分支权重） |
| `head_checkpoint` | `--resume-head-path` 路径 |
| `report` | 指向本次 `trg_report.json` 的路径 |

该 flag 用于后续流水线（如 Stage 2 OPD）判断 **纠错教师是否达到「可冻结」标准**；`ready=false` 时仍会写 flag，便于排查。

---

## 9. 与主 benchmark `eval.py`（gIoU / cIoU）的区别

| 维度 | TRG（`eval_teacher_recovery.py`） | 主评测（`eval.py`） |
| --- | --- | --- |
| 输入 | **腐化后的 mask** + 文本 | 通常为 SAM seed / 点框等 **标准 prompt**，非故意腐化 |
| 核心指标 | **TRG**、input/output IoU | **gIoU**（样本 IoU 均值）、**cIoU** 等 |
| 目的 | 衡量 **纠错增益** | 衡量 **绝对分割质量** |
| 数据 | ConVerSeg jsonl + 在线腐化 | 同数据根但协议不同 |

二者 **不可直接数值对比**；TRG 高不等价于 gIoU 最高，因为任务设定不同。

---

## 10. 常用命令

**本地 / 集群 Python：**

```bash
python eval_teacher_recovery.py \
  --checkpoint /path/to/sam2_hiera_large.pt \
  --plm-model-name Qwen/Qwen2.5-VL-3B-Instruct \
  --resume-sam-path ckpts/teacher_correction/author3b/CorrectionTeacher_3B_final.torch \
  --resume-plm-path ckpts/teacher_correction/author3b/CorrectionTeacher_3B_plm_final.torch \
  --resume-head-path ckpts/teacher_correction/author3b/CorrectionTeacher_3B_head_final.torch \
  --resume-lora-path ckpts/teacher_correction/author3b/lora_plm_adapter_final \
  --eval-stage C \
  --splits sam_seeded,human_annotated \
  --out-dir ckpts/teacher_correction/eval_recovery \
  --flag-path ckpts/teacher_correction/teacher_frozen_ready.flag
```

**Slurm（仓库内）：**

```bash
# 默认 EVAL_STAGE=B；Stage C 可：EVAL_STAGE=C sbatch scripts/teacher_correction/slurm_eval_trg.slurm
sbatch scripts/teacher_correction/slurm_eval_trg.slurm
```

---

## 11. 指标清单速查

| 层级 | 指标名 | 出现在 |
| --- | --- | --- |
| 单样本 | `input_iou`, `output_iou`, `trg` | `trg_report.json` → `rows[]` |
| 单样本 | `op`, `split`, `stage`, `id` | `rows[]`（描述用） |
| 全局 | `mean_trg`, `positive_fraction` | JSON 顶层 + flag |
| 全局 | `threshold_mean`, `threshold_pos_frac`, `eval_stage` | JSON 顶层 |
| 门槛 | `ready` | flag |
| 诊断 | `mask_gate`, `head_checkpoint`, `report` | flag |
| 仅终端 | mean input/output IoU；按 split/op 的 mean TRG | 标准输出 |

---

## 12. 相关源码索引

| 文件 | 职责 |
| --- | --- |
| `eval_teacher_recovery.py` | 测评主程序、TRG 汇总、写 JSON/flag |
| `data/mask_corrupter.py` | Stage 配置、八种算子、IoU、`MaskCorrupter` |
| `models/teacher_correction_model.py` | Correction head 与 mask 门控 |
| `scripts/teacher_correction/slurm_eval_trg.slurm` | 集群 TRG 任务模板 |

如有新门槛、新 split 或新算子，请同步更新本文档与 `eval_teacher_recovery.py` 中的 argparse 默认值说明。
