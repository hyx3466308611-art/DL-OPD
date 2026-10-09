# 实验

**源文件：** `4_Experiments.tex`  
**标签：** `sec:experiments`  
**命名：** 全文统一使用 **DL-OPD**（与摘要 / 方法一致）；不另引入未在提案中定义的方法别名。

---

实验围绕提案假设 **H1–H4** 组织，回答四个问题：（1）Mask-Policy OPD 是否支持有效的掩码纠正；（2）纠正向训练是否保留单遍 CIS 能力；（3）Data-Policy 与 Mask-Policy 是否各自有效且互补；（4）多步纠错能否在可接受延迟下折叠为单遍。

> **数值说明：** 下表结构来自提案实验设计。当前仓库主结果与消融**尚未全部跑完**；本文档**不填写编造数字**，待实测后填入。

---

## 1. 实验设置

**标签：** `sec:exp_setup`

### 1.1 数据集与测试任务

| 任务 | 内容 |
| --- | --- |
| **ConVerSeg** | SAM-seeded / human-annotated；报告总体及五类概念的 gIoU（主）与 cIoU（附录） |
| **RefCOCO / + / g、ReasonSeg** | 零样本，检验指代与推理分割能力是否保持 |
| **Hard Prompt Set** | 语义保持改写、最小语义对、组合约束、物理反事实 |
| **Mask Recovery Set** | 控制腐化类型与严重度，评价学生 / 教师纠错 |
| **Negative Prompt Set** | 不存在对象、错误属性 / 关系 / 功能 / 状态 |

#### 掩码纠正协议（Mask Recovery）

每样本含图像 $I$、提示 $p$、真值掩码 $M$。错误初始掩码有两种互补来源：

| 协议 | 初始掩码 | 用途 |
| --- | --- | --- |
| **On-policy** | $q_s^1=\pi_\theta(I,p,\emptyset)$ | 贴近部署时的学生错误 |
| **Corrupted-GT** | 对 $M$ 做受控腐化（腐蚀 / 膨胀 / 缺块 / 假区域 / 边界抖动 / 位移等） | 与纠正教师训练设定对齐，控制误差严重度 |

两种协议使用相同 image–prompt–target 对，仅错误掩码来源不同。

### 1.2 基线与消融矩阵

| 方法 | 外层数据策略 | 内层掩码状态 | 用途 |
| --- | --- | --- | --- |
| ConVerSeg（Stage-2 学生） | 固定 | 无 | 原始基线 / Ours-init |
| Random-Augment | 随机生成 | 无 | 控制新增数据量 |
| Error-Data | 错误驱动 | 无 | 验证外层贡献 |
| Off-policy KD | 固定 | 固定教师输出 | 标准蒸馏基线 |
| Random-State KD | 固定 | 随机腐化状态 | 控制状态增强 |
| Historical-State KD | 固定 | 历史学生状态 | 验证状态陈旧影响 |
| Mask-OPD | 固定 | **当前学生状态** | 验证内层贡献 |
| **DL-OPD（完整）** | **错误驱动** | **当前学生状态** | 完整方法 |
| Correction Teacher（参考） | — | — | 纠正任务上界；推理不部署 |

外部分割方法（LISA、UniLSeg、EVF-SAM、Seg-Zero 等）仅作单遍 CIS 的外部参考；未为掩码纠正任务重训。作者公开 ConVerSeg-Net 结果可作为参考，但若训练预算不匹配，须在正文中明确标注。

### 1.3 评价指标

记初始错误掩码为 $M_b$，纠正后为 $M_r$，真值为 $M$。

| 维度 | 指标 | 说明 |
| --- | --- | --- |
| 分割精度 | gIoU、cIoU | 总体与五类概念分项 |
| 边界质量 | Boundary F、HD、Over/Under ratio | 功能区域 / 部件 / 表面几何 |
| 负向提示 | Empty Accuracy、False Positive Area | 条件不成立时的幻觉抑制 |
| 纠错能力 | Recovery Gain $=\mathrm{IoU}(M_r,M)-\mathrm{IoU}(M_b,M)$ | 纠正增益；并报 $\mathrm{IoU}_{\mathrm{after}}$、Gain$>0$ 比例 |
| 校准 | ECE、No-mask ECE | 置信度与正确性匹配 |
| 数据效率 | $\Delta\mathrm{gIoU}$ / accepted samples | 单位有效新增样本收益 |
| 部署效率 | Params、Latency、GPU memory | One-pass / Two-pass / 教师对比 |

单遍 CIS 默认二值化阈值 $0.5$。修复评测同时报告 Text IoU（$q_s^1$）与纠正后绝对质量。

### 1.4 实现细节（与提案 / 当前实现一致处）

- **学生：** SAM2-Large + Qwen2.5-VL-3B；LoRA $r=16$，$\alpha=32$  
- **纠正教师：** Qwen2.5-VL-7B + SAM2 Hiera-L；腐化–恢复训练后冻结  
- **学生可训：** Mask Decoder、MaskStateEncoder、语言侧 LoRA  
- **OPD 学习率：** 约 $1\times10^{-5}$–$5\times10^{-5}$（低于 Stage-0）  
- **批次配比（建议）：** Base 25% / Conv+ 25% / Conv− 20% / Hard 30%  
- **Outer cycles：** 4–6；Hard Buffer TTL：1–2 周期  
- **数据：** Stage-2 对话混合 + Data-Policy 困难样本；无额外人工 repair 标注  
- **推理：** 支持纯文本 grounding 与掩码条件纠正；两种模式共享缓存的 SAM 图像 embedding  

完整超参、腐化配置与硬件细节放入附录；以实际训练配置为准，不在此虚构。

---

## 2. 主结果（待填数）

**标签：** `sec:main_results`

### 2.1 掩码纠正

在 On-policy 与 Corrupted-GT 双协议下报告 $\mathrm{IoU}_{\mathrm{after}}$、Recovery Gain 与 Gain$>0$ 比例。对比 Stage-2 init、Mask-OPD 与完整 DL-OPD；并以冻结纠正教师作为上界参考。

### 2.2 单遍会话分割（对齐作者 Table 1 风格）

报告 SAM-seeded / human-annotated 的 All 与五概念 gIoU（附录补 cIoU）。目标：引入纠正监督**不损害**原有 grounding；若单遍可比或更好且纠正提升，则说明掩码条件纠正可与对话 grounding 兼学。

> 已具备可填入对照的实测（自训 Stage-2 3B 等）见仓库根目录 `AUTHOR_PAPER_EVAL_RESULTS.md`；正式论文表须在统一评测协议下重新汇总后写入，避免混用未完成设置。

### 2.3 迁移评估

零样本 RefCOCO/+/g 与 ReasonSeg：重点观察相对 Stage-2 init 是否显著改变一般指代 / 推理行为，而非单纯刷 SOTA。

### 2.4 定性结果

对比初始文本预测、错误掩码状态、DL-OPD refine 与 GT；覆盖过分割、欠分割、空间偏移、缺失区域与干扰物，并包含失败案例。

---

## 3. 消融与因果验证

**标签：** `sec:ablations`

按提案「关键因果验证」组织，每组**只改一个因素**：

| 验证 | 控制 | 变化因素 | 对应假设 |
| --- | --- | --- | --- |
| 状态因果 | 固定训练数据 | Random / Historical / **Current** mask state | H1 |
| 数据因果 | 固定教师与步数 | Random-Augment vs **Error-driven** | H2 |
| 双层协同 | — | 仅 Data / 仅 Mask / **DL-OPD** | H3 |
| 损失因果 | — | 逐项加入 pixel / boundary / representation / collapse | 方法设计 |
| Collapse | — | Two-pass vs **One-pass Collapse** 延迟与精度 | H4 |
| 成本因果 | 固定教师查询预算 | 在线 / 周期缓存 / 纯离线 | 效率 |
| 模型因果 | — | 7B→3B、同规模自蒸馏等 | 泛化 |

补充可报告：掩码融合强度、on-policy 与 corrupted-GT 混合比例、去掉教师 KD 后的师生 gap。

---

## 4. 预期主结论（可检验表述，非实测声称）

在相同新增数据量与教师查询预算下，完整 DL-OPD 应同时提升 Physics/Safety、Affordance、负向提示抑制与边界质量，并证明 **Current-state 监督**与 **Error-driven 数据策略**分别提供独立且互补的收益。是否成立，以填表后的实测为准。
