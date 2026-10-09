# DL-OPD 双层在策略蒸馏：完整详细实现方案

**依据文档：**《双层OPD驱动的会话式图像分割研究Proposal》（V1.0，2026年8月）  
**修订基础：** 原 `DL_OPD_IMPLEMENTATION_PLAN.zh.md`  
**方法简称：** DL-OPD；应用场景名可用 ConverOPD  
**文档性质：** 可执行的研究与工程实施规格。文中“已完成”仅沿用现有计划中的资产状态；所有尚未运行的效果均标记为“待实测”，不将预期结果写成实验事实。

---

## 0. 总目标、范围与实施原则

本项目面向会话式图像分割（Conversational Image Segmentation, CIS），研究如何让模型根据自然语言中的实体、空间布局、关系与事件、功能与可供性、物理与安全约束，输出可靠的像素级掩码。项目以 ConVerSeg-Net 为基础，沿用 SAM2 图像编码器、Qwen2.5-VL 提示编码器、Language Adapter 与 SAM2 Mask Decoder，并在训练中加入双层在策略蒸馏机制。

核心假设是：比起对固定样本反复使用静态教师输出，更有价值的监督来自教师对“当前学生真实会犯的错误”的纠正；与此同时，学生当前的错误分布应决定下一轮生成和采样哪些数据。因此整个方法包含两个相互作用但职责分离的闭环：

- **内层 Mask-Policy OPD：** 以 mini-batch / inner step 为时间尺度。学生先产生当前掩码状态，冻结纠错教师在该状态上给出纠正目标，学生再预测，并在 GT 监督、像素蒸馏、边界蒸馏、表征蒸馏与单步折叠目标下更新。
- **外层 Data-Policy OPD：** 以 outer cycle（多个 epoch 或一个训练周期）为时间尺度。固定诊断集形成学生错误画像，数据策略据此选择概念类别、生成操作、样本和训练配比，经过多级质量验证后刷新短生命周期 Hard Buffer。

两个闭环必须协同，但不能混淆职责：外层只改变训练数据分布与样本权重，不直接对学生反向传播；内层更新学生，不更新冻结教师，也不直接决定全局数据分布。教师可靠性门控贯穿内层和数据验证阶段。

本方案同时覆盖研究计划书的全部内容：背景问题、五项创新、H1–H4、Mask-Policy、Data-Policy、五类提示生成算子、多级验证、训练 Stage 0–4、批次配比与超参数、教师查询效率、完整实验矩阵、专项数据集与评价指标、五类因果验证、资源可行性、风险控制、预期成果、论文组织、研究里程碑及附录 MVP。为避免首版引入过多不稳定因素，工程实现按 MVP-A、MVP-B、完整版本逐步推进；这只是工程拆分，不改变研究计划书定义的完整方法范围。

### 0.1 状态标签约定

- **已有资产：** 现有计划中记录已训练、已评测或可复用的模型、代码、数据。
- **待实现：** 尚需新增代码或流程。
- **待验证：** 已有初步诊断依据，但必须通过本方案的统一测试确认。
- **待实测：** 尚无可引用的实验结果；不得在论文或报告中提前声称有效。

### 0.2 五项创新与代码映射

| 创新 | 方法内容 | 必须实现的模块 | 最低验收证据 |
|---|---|---|---|
| I1：非自回归分割的 OPD 定义 | 学生掩码被定义为可被教师纠正的策略状态 | `MaskStateEncoder`、current-state rollout、state ablation | Current-state 对比 Random-State、Historical-State |
| I2：错误条件化 Data-Policy | 数据类别、生成操作、采样配比随学生错误改变 | error profile、data value、concept sampler、Hard Buffer | 同数据预算下 Error-Data 对比 Random-Augment |
| I3：多粒度联合蒸馏 | GT、像素概率、边界、表征联合优化，并控制教师偏差 | 六项损失、教师门控、置信度与双教师一致性检查 | loss ablation 与风险门控消融 |
| I4：Iterative-to-Single-Pass Collapse | 把多步纠错能力折叠到单次前向 | `L_collapse`、One-pass/Two-pass 评测 | 精度、延迟和显存的共同对比 |
| I5：专项评测体系 | 困难提示、负提示、掩码恢复及校准评测 | 三类专项测试集、统一 evaluator | 数据清单、指标定义与可复现实验表 |

---

## 1. 问题定义、符号与科学假设

### 1.1 任务问题

标准 CIS 模型可以处理实体、属性与简单空间关系，但对抽象意图、功能可供性、关系事件、物理稳定性和安全约束仍可能出现选错实例、边界不完整、关系理解错误或不存在目标却输出非空掩码等问题。固定数据分布不能自动追踪学生当前弱项；固定教师输出也不能完整覆盖部署时由学生自身预测产生的错误状态。

DL-OPD 将训练过程建模为两个相互关联的策略问题：

1. **预测层状态问题：** 对给定图像 `I` 和提示 `p`，学生第一次输出概率掩码 `q_s^1`。这个输出不是最终不可更改的答案，而是后续纠错的状态。冻结教师根据该状态生成 `q_t^2`，学生则以同一状态为条件生成 `q_s^2`。
2. **数据层分布问题：** 训练数据并非只从静态池均匀采样。模型在固定诊断集上的错误向量 `u` 决定后续优先采样的概念、生成操作和困难样本。

### 1.2 主要符号

| 符号 | 含义 |
|---|---|
| `I` | 输入图像 |
| `p` | 会话提示或自然语言指令 |
| `M` | Ground-truth 掩码；负提示的正确目标可以是全零掩码 |
| `q_s^k` | 学生第 `k` 步的概率掩码 |
| `q_t^2` | 冻结教师在学生当前掩码状态下的纠错输出 |
| `pi_theta` | 学生策略 |
| `pi_T` | 冻结纠错教师策略 |
| `B` | 基础与会话式训练数据池 |
| `U` | 可用未标注图像池或待生成候选池 |
| `H` | Hard Buffer，保存经过验证的短生命周期困难样本 |
| `u` | 学生错误画像向量 |
| `V(I,p)` | 候选训练样本价值分数 |
| `Q_data` | 样本综合质量分数 |
| `g_kd` | 样本级教师蒸馏开关或权重 |

### 1.3 H1–H4 假设与科学问题

| 假设 | 内容 | 对照实验 | 证据要求 |
|---|---|---|---|
| H1 | 学生诱导状态比固定随机腐化或历史状态更接近部署分布 | Random-State KD、Historical-State KD、Current-State Mask-OPD | 固定数据、训练步数、教师与损失，只改变状态来源 |
| H2 | 错误驱动数据策略具有更高边际训练价值 | Random-Augment、Error-Data | 固定新增样本预算、训练步数与验证协议 |
| H3 | 数据层与预测层互补而非冗余 | Mask-OPD、Error-Data、DL-OPD | 报告单层和双层增益，并按五类概念分解 |
| H4 | 多步纠错知识可压缩至单步推理 | One-pass、Two-pass、Collapse off/on | 比较分割质量、延迟、参数量与 GPU 显存 |

对应科学问题为：当前学生状态是否比随机或历史状态有效；错误驱动数据是否优于随机扩增；两层机制是否互补；训练时多步纠错是否能在部署时保持单次前向效率。实验应围绕这些问题组织，不应仅罗列总体指标。

---

## 2. 代码与模型资产盘点

原计划记录的资产如下。开始训练前应逐项检查路径、配置、模型结构和权重键，避免把文档记录误当作当前运行环境已经验证。

| 资产 | 原计划中的路径/说明 | 在 DL-OPD 中的用途 | 开始前检查 |
|---|---|---|---|
| 3B 学生 Phase 1 | `ckpts/stage1_phase1mix` | 基础能力对照、初始化回退 | 模型可加载、推理输出与基线一致 |
| 完整 3B 学生 Phase 1+2 | `ckpts/stage2_phase2mix` | OPD 学生初始化 | 作为主学生 init，保存原始配置和校验信息 |
| 7B 基础教师骨干 | `ckpts/teacher/stage2_phase2mix` | 基础教师参考、必要时构建第二教师 | 与纠错教师区分目录和用途 |
| 7B 纠错教师 Stage-1 | `ckpts/teacher_correction/teacher7b` | 主冻结纠错教师 | 重新跑统一 Recovery Set，确认 Teacher Recovery Gain |
| 3B fix_v1/v2 | `author3b_fix_v*` | 门控、条件损失和小规模诊断参考 | 不直接默认优于 7B；逐版本验证 |
| 教师侧 MaskStateEncoder / gated head | `models/teacher_correction_model.py` 等 | 复用模块设计、核对张量接口 | 对齐维度、归一化、掩码阈值和 checkpoint 键 |
| 现有评测入口 | `eval.py`、教师 TRG 评测脚本 | 基线、分概念评估、恢复评估 | 统一数据划分和指标定义 |
| 腐化工具 | `MaskCorrupter` 等 | 生成 Mask Recovery Set 与教师训练状态 | 固定算子配置和随机种子 |

### 2.1 进入 OPD 前的门槛

纠错教师不能仅因为模型更大就默认可信。它必须在独立的 Mask Recovery Set 上验证，并按腐化算子、严重度、输入 IoU、GT 是否为空分别报告结果。原有诊断记录提示：大块错误和 `mask_dropout` 可能是教师强项；`empty_gt`、高输入 IoU、`boundary`/`elastic` 可能存在假阳性或过修风险。这些结果作为门控设计的先验，但所有阈值都应在统一评测集上复核。

进入学生蒸馏阶段前至少应产出：

1. 教师与学生在同一恢复集上的 IoU、Recovery Gain、Empty Accuracy、False Positive Area。
2. 按腐化类型、严重度和输入 IoU 区间拆分的结果。
3. 教师是否复制输入掩码的检查，包括输入输出掩码重叠、变化区域和恢复增益。
4. 教师置信度是否校准，以及低置信样本的错误率。
5. 若采用双教师，两个教师的独立性、输出一致性和各自错误边界。

---

## 3. 总体架构与完整数据流

```text
固定诊断集 D_diag ──> 学生评估 ──> 错误画像 u
                                      │
未标注图像池 U + 基础数据池 B ────────┤
                                      v
                          Data-Policy / 样本价值 V
                                      │
                 选择概念类别、生成算子、采样配比
                                      │
                         生成候选困难提示/样本
                                      │
           语义验证 -> 几何验证 -> 教师稳定性 -> 去重/新颖性
                                      │
                               Q_data 质量门控
                                      │
                         Hard Buffer H（TTL 1–2 cycles）
                                      │
                Base 25% / Conv+ 25% / Conv- 20% / Hard 30%
                                      │
                                      v
             q_s^1 = pi_theta(I,p,empty state)
                                      │
                     当前学生状态 sg(q_s^1)
                         ┌────────────┴────────────┐
                         v                         v
            冻结教师 q_t^2                 学生 q_s^2
                         └────────────┬────────────┘
                                      v
                  GT 优先 + 教师置信/一致性/改进门控
                                      │
                  L_sup + L_opd + L_bd + L_rep
                         + L_collapse + L_anchor
                                      │
                               更新学生 theta
                                      │
                    下一 outer cycle 重新评估与刷新数据
```

完整前向形式为：

\[
q_s^1=\pi_\theta(I,p,\emptyset),\quad
q_t^2=\pi_T(I,p,\operatorname{sg}(q_s^1)),\quad
q_s^2=\pi_\theta(I,p,\operatorname{sg}(q_s^1)).
\]

`sg` 表示 stop-gradient。教师与第二次学生预测均以学生当前状态为条件，但教师参数冻结；学生的 `q_s^1` 作为状态输入时不通过该路径反向传播，避免将状态条件变成不稳定的循环梯度。

---

## 4. Stage 0–4 训练阶段与实施分层

### 4.1 Stage 0：基础 CIS 学生

**目标：** 建立可复现的 ConVerSeg 基线，并保留基础指代分割与开放词汇能力。

- 沿用 Literal、RefCOCO、Open-vocabulary 与 Conversational 两阶段课程。
- 固定官方或当前项目使用的数据版本、预处理、训练配置、随机种子和 checkpoint 选择规则。
- 评估 ConVerSeg SAM-seeded 与 human-annotated 设置；记录总体与五类概念的 gIoU/cIoU。
- 保存基础学生权重、配置、评测日志和模型推理延迟，作为 Anchor 参考。

**产出：** Stage-0/Stage-2 学生基线、训练配置、基线评测表和数据清单。原计划中的 `ckpts/stage2_phase2mix` 用作主学生初始化，实际使用前需确认与基线评测对应。

### 4.2 Stage 1：纠错教师

**目标：** 训练一个能够根据输入掩码恢复正确分割的教师，而不是仅把掩码当作普通提示后复述输入。

教师建议采用 Qwen2.5-VL-7B 与 SAM2 Hiera-L。训练样本从可靠 GT 掩码构造语义或几何上有意义的腐化状态，覆盖：

- 腐蚀、膨胀与形态变化；
- 组件缺失、部分组件缺失；
- 错误实例混入、false positive；
- 边界抖动、弹性形变、孔洞或局部缺失；
- 位置偏移、平移；
- 部分满足条件、属性或关系错误；
- 正负提示翻转；
- mask dropout（输入状态为空但 GT 非空）；
- empty-GT 负样本（正确目标为空，必须单独处理）。

腐化训练应由轻到重，记录每个算子的概率、强度、种子和生成参数。教师完成训练后冻结参数。进入 Stage 2 前，独立运行 Teacher Recovery Set，验证不同算子的 Recovery Gain；如果某一算子或输入 IoU 区间教师没有可靠正增益，则该区域不得无条件蒸馏。

**关键区别：** `mask_dropout` 指输入掩码被清空但 GT 是非空目标；`empty_gt` 指 GT 本身为空。两者不能混为一谈。

### 4.3 Stage 2：内层 Mask-Policy OPD

**目标：** 在学生当前状态上进行 GT、像素和边界监督，并引入可靠教师的纠错信号。

- 学生初始化：完整 3B Stage-2 checkpoint。
- 教师：冻结 7B 纠错教师；可配置在线、近在线缓存或双教师模式。
- 新增/复用 MaskStateEncoder，把概率掩码映射到 SAM2 dense prompt 空间。
- 初始 rollout 深度 `K=1`，即从空状态产生 `q_s^1`，然后在 `q_s^1` 条件下产生 `q_s^2`；验证稳定后才试 `K=2,3`。
- 初期必做 `L_sup`、门控后的 `L_opd`、`L_bd`。`L_rep`、`L_collapse` 与完整 `L_anchor` 按 MVP 分层逐步加入，但完整方法必须纳入损失消融。

**产出：** Mask-OPD 学生、状态来源消融、门控日志、损失日志和恢复评测。

### 4.4 Stage 3：外层 Data-Policy 闭环

**目标：** 依据学生在固定诊断集上的错误画像，调整样本采样、提示生成算子和训练配比。

每个 outer cycle 执行：诊断集评估 → 更新错误画像 `u` → 候选样本打分 → 选择概念与生成操作 → 生成困难提示 → 多级验证 → 更新 Hard Buffer → 在下一 inner cycle 训练。

MVP 先采用规则化错误统计、按错误率重采样、语义保持改写和最小语义对；后续再启用完整数据价值函数、更多算子、Contextual Bandit 或其他学习型数据策略。

### 4.5 Stage 4：Iterative-to-Single-Pass Collapse 与收敛

**目标：** 训练阶段允许多步纠错，部署阶段仍可只执行一次模型前向。

- 保留 `q_s^1` 作为 One-pass 输出，`q_s^2` 作为 Two-pass / refine 输出。
- 只有当 `q_s^2` 相对 `q_s^1` 在 GT 指标上确实改善，且教师目标可靠时，才用第二步知识监督第一步路径。
- 随 outer cycle 或训练进度逐步提高 `lambda_col`，避免早期学生尚未学会基本分割时就过度压缩。
- 同时保存 One-pass 与 Two-pass checkpoint/推理配置，报告质量、延迟、参数量和 GPU memory。

### 4.6 MVP 分层：避免把工程 MVP 与研究计划书 MVP 混为一谈

原提案附录 A 的 MVP 明确包括六项：复现官方 ConVerSeg 并固定基线；训练能够修复随机与语义腐化的 7B 教师；实现 MaskStateEncoder、Current-state rollout 与像素/边界 OPD；以规则统计五类概念及负提示错误并重采样已有数据；加入语义保持改写和最小语义对；完成 ConVerSeg、Off-policy KD、Random-State KD、Mask-OPD、DL-OPD 五组实验。

为了便于工程调试，执行上拆成以下三个里程碑：

| 工程阶段 | 内容 | 目的 | 与提案 MVP 的关系 |
|---|---|---|---|
| MVP-A：内层可运行 | 固定现有数据，跑通 `MaskStateEncoder`、current-state rollout、GT/pixel/boundary loss、教师门控和日志 | 先定位前向、显存、梯度和门控问题 | 是提案 MVP 的前置调试子集，不可宣称已经完成提案 MVP |
| MVP-B：提案最小完整版本 | 增加五类概念/负提示错误统计、规则重采样、语义保持改写、最小语义对，并完成五组主实验 | 满足附录 A 的最小实现范围 | **达到提案 MVP 的最低完整范围** |
| 完整 DL-OPD | 五类算子、多级质量门控、双教师一致性、置信度机制、完整损失、查询成本因果实验、跨规模实验 | 支撑完整研究论文与系统验证 | 对应完整 Proposal，而非仅附录 MVP |

不能把“先只做内层”表述为“已经覆盖提案全部 MVP”；它是合理的工程阶段，但外层规则重采样和两种低噪声生成算子仍必须进入 MVP-B。

---

## 5. 学生架构与 MaskStateEncoder

### 5.1 模块配置

| 模块 | 输入/输出 | OPD 阶段状态 | 说明 |
|---|---|---|---|
| SAM2 Image Encoder | `I -> z_img` | 冻结为默认设置 | 保留分割先验，减少训练变量 |
| Qwen2.5-VL Prompt Encoder | `(I,p) -> token states / EOS` | LoRA 微调 | 低学习率继续适配当前任务 |
| Language Adapters | 文本状态到 sparse/dense prompts | 训练 | 与现有 ConVerSeg 接口对齐 |
| MaskStateEncoder | `q_s^k -> dense mask prompt` | 新增并训练 | 将当前概率掩码转换为 SAM2 可用的 dense prompt 表示 |
| SAM2 Mask Decoder | 图像/文本提示/掩码状态到输出掩码 | 训练 | 输出 `q_s^{k+1}` |

### 5.2 MaskStateEncoder 第一版

第一版遵循提案：使用三层卷积将学生概率掩码编码至 SAM2 dense prompt 空间，并与 Qwen EOS 对应的全局图文表示融合，先采用直接相加。输入应使用概率掩码而非仅二值化结果，以保留置信度与不确定性信息。

实施步骤：

1. 明确 `q_s` 的空间尺寸、数值范围、通道数和是否经过 sigmoid。
2. 将 `q_s` resize 到 decoder dense prompt 所需分辨率；resize 方法在配置中固定。
3. 使用三层卷积提取空间结构，输出通道与 SAM2 dense prompt 对齐。
4. 将全局图文表示映射到相同通道/空间维度后融合；首版直接相加，并明确广播与归一化规则。
5. 通过单元测试验证全空掩码、全满掩码、单点、边界形状和随机概率图的输出尺寸、有限值与梯度。
6. 与教师侧实现对齐张量契约，但学生模块应明确独立训练与 checkpoint 管理方式。

后续变体按消融顺序加入：门控融合、Mask Token、正负点采样。不要在 MVP-A 同时加入所有变体。

### 5.3 防止学生状态泄漏和训练不稳定

- 传给教师和学生第二步的 `q_s^1` 默认执行 stop-gradient。
- 教师全程 `eval()` 与 `no_grad()`，确认没有 BatchNorm/Dropout 状态被意外更新。
- 训练日志应同时保存 `q_s^1`、教师目标 `q_t^2`、`q_s^2` 的可视化样例及其 IoU，不能只保存最终损失。
- 记录模型版本、状态来源（current/random/historical）、腐化类型和状态年龄，确保消融可追溯。

---

## 6. 内层 Mask-Policy OPD 训练流程

### 6.1 单个 batch 的步骤

对每个样本 `(I,p,M)`：

1. **初始预测：** `q_s^1 = pi_theta(I,p,empty_state)`。
2. **教师纠错：** 如当前样本符合查询策略，在 `no_grad` 下计算 `q_t^2 = pi_T(I,p,sg(q_s^1))`；否则使用合法缓存目标或该样本不计算教师 KD。
3. **学生 refine：** `q_s^2 = pi_theta(I,p,sg(q_s^1))`。
4. **样本可信度判定：** 计算学生/教师相对 GT 的指标、教师置信度、双教师一致性（若启用）、输入状态 IoU、GT 是否为空及边界风险。
5. **GT 监督：** 对 `q_s^1` 与 `q_s^2` 分别计算监督损失。
6. **蒸馏损失：** 仅对通过教师门控的像素/样本计算 `L_opd`；低置信教师目标不能直接作为硬标签。
7. **边界损失：** 在有效非空 GT 上计算 Boundary Dice 或 signed-distance loss；负提示单独处理。
8. **表征/折叠/锚定损失：** 仅在相应阶段启用，记录各项原始值与加权值。
9. **反向传播：** 仅更新学生参数；教师始终冻结。
10. **日志：** 记录每个门控分支数量、教师调用率、师生 IoU、Recovery Gain、置信度、各项损失、显存与 step 时间。

### 6.2 教师门控：必须区分“教师是否改进”与“教师是否可信”

原方案的 `empty_gt`、高 IoU 关断和教师优于学生门控继续保留；同时补齐提案风险章节要求的 GT 优先、双教师一致性、置信度门控和低置信样本仅作 soft supervision。不能只以“教师比学生 IoU 高”就认定教师输出可靠。

建议把门控拆成以下独立条件：

- `valid_gt`：GT 标签是否有效，是否为空目标，是否属于多解样本。
- `state_risk`：当前学生输入状态是否已很接近 GT（高 IoU），是否属于容易过修的 boundary/elastic 区间。
- `teacher_improvement`：教师输出是否比当前学生输出更好。
- `teacher_confidence`：教师输出置信度是否达到硬目标阈值。
- `teacher_consistency`：两名独立教师是否在区域、语义或预测分布上达成约定的一致程度；如果没有第二教师，则记录为未启用，不伪造一致性结果。
- `target_mode`：选择 GT-only、teacher-hard、teacher-soft、candidate-set 或 reject。

建议优先级如下：

| 优先级 | 条件 | 处理方式 | 目的 |
|---:|---|---|---|
| 1 | GT 无效、数据损坏或提示/掩码对齐错误 | 拒绝样本，不参与训练并记录原因 | 防止坏数据污染 |
| 2 | GT 是明确的空目标/负提示 | 不使用教师掩码作为目标；用全零 GT 做监督 | 防止教师在空目标上画出假阳性 |
| 3 | 多解功能/物理提示且无法唯一确定区域 | 不构造错误硬标签；使用候选掩码集合、soft mask 或人工复核 | 避免合理多解被压成单一标签 |
| 4 | 输入状态 IoU 高于 `tau_hi`（默认初始值 0.85，需验证）且教师可能过修 | 关闭 pixel KD；可选小权重 identity 稳定项 | 防止高 IoU 过修 |
| 5 | 教师置信度低或双教师不一致 | 禁止 hard KD；如分布本身仍有信息，仅作为低权重 soft supervision，否则跳过 KD | 落实低置信样本策略 |
| 6 | 教师未超过学生至少 `delta`，或教师 Recovery Gain 非正 | 关闭 KD，仅保留 GT 监督 | 避免学生学习到较差目标 |
| 7 | 教师置信度合格、教师一致性合格（若启用）且有足够改进 | 启用标准 pixel KD 与适用的边界/表征监督 | 使用可靠教师纠错信号 |

`tau_hi=0.85`、`delta=0` 或 `0.02` 只是可配置初始值，不是已经证明的最优阈值。建议在验证集上比较 `tau_hi` 取 0.80、0.85、0.90 的门控覆盖率、Recovery Gain 与 Boundary F；最终阈值应依据验证结果固定，不能根据测试集调参。

### 6.3 空 GT 与负提示的统一约定

- `empty_gt` 指正确答案为空，监督目标为全零掩码；教师输出不得覆盖 GT。
- `mask_dropout` 指输入状态为空但正确答案非空，此时教师有机会从图像与提示恢复目标；只有教师相对学生有可靠改进并通过置信门控时才 KD。
- 空 GT 的 IoU 存在不同实现约定，必须在整个项目统一。建议主评测使用 Empty Accuracy 与 False Positive Area；在计算 IoU 时明确“空预测且空 GT”的处理，避免不同脚本结果不可比较。
- 空 GT 的监督损失默认使用对全零目标的 BCE；Dice 项应明确跳过或使用经过验证的平滑定义，不允许除零或 NaN。
- `Empty Accuracy`、`False Positive Area` 和 `No-mask ECE` 必须单独报告，不能只用总体 gIoU 代表负提示能力。

### 6.4 联合损失总式

\[
\mathcal L=\mathcal L_{sup}+\lambda_{opd}g_{kd}\mathcal L_{opd}+\lambda_{bd}\mathcal L_{bd}+\lambda_{rep}\mathcal L_{rep}+\lambda_{col}\mathcal L_{collapse}+\lambda_{anc}\mathcal L_{anchor}.
\]

各项损失的具体实现如下。

#### 6.4.1 GT 监督损失 `L_sup`

对 `q_s^1` 和 `q_s^2` 都计算 BCE 与 Dice，以确保教师错误不会取代可靠 GT：

\[
\mathcal L_{sup}=\sum_{k\in\{1,2\}} w_k\left(\mathcal L_{BCE}(q_s^k,M)+\lambda_{dice}\mathcal L_{Dice}(q_s^k,M)\right).
\]

`w_1,w_2` 作为配置项；初版可以等权，后续验证是否需要提高第二步或第一步权重。空 GT 样本至少保留 BCE 到全零目标，并可设置负样本权重，但权重必须通过验证集确定。

#### 6.4.2 像素 OPD 损失 `L_opd`

对教师与学生的逐像素 Bernoulli 分布做温度化 KL，并按学生不确定性、师生分歧与边界位置加权。温度初始值按提案设为 `T=2`。

**方向必须在代码和文档中固定：** 常规 teacher-to-student 蒸馏使用 `KL(p_teacher || p_student)`，即教师分布作为 target、学生分布作为被优化分布。原 MD 草稿中写的 `KL(q_s || q*)` 与这一方向不同，不能直接照抄；实现时明确 `q*` 是教师目标，检查梯度确实更新学生而不是反向匹配教师。若采用非标准方向，必须给出方法动机并作为独立消融。

可将像素权重写为：

\[
w_i = w_{entropy,i}\,w_{disagree,i}\,w_{boundary,i},\quad
\mathcal L_{opd}=\frac{\sum_i w_i\,KL\big(\mathrm{Bern}(p_{t,i}^{T})\,\|\,\mathrm{Bern}(p_{s,i}^{T})\big)}{\sum_i w_i+\epsilon}.
\]

具体权重形式、归一化和 clipping 应集中在配置中，做单元测试并通过消融验证，避免权重尺度让少数像素主导梯度。教师置信度不足时使用 soft supervision 或跳过，而不是强行把教师输出二值化成硬标签。

#### 6.4.3 边界蒸馏 `L_bd`

可选 Boundary Dice 或 signed-distance transform loss。第一版选定一种并固定评估流程，之后再比较另一种。目标是提升部件、表面、功能区域边界质量，尤其关注 Boundary F、HD、Over/Under ratio。

- 非空 GT：对 GT 边界和预测边界计算损失。
- 空 GT：不得以不存在的边界参与常规边界匹配；单独使用空目标 BCE/假阳性惩罚。
- 高输入 IoU 且教师存在过修风险：可以保留 GT 边界监督，但不应因此重新开启教师 pixel KD。

#### 6.4.4 表征蒸馏 `L_rep`

对齐提案指定的中间表征：Qwen 选定层文本 token、EOS、Language Adapter 表征及 SAM2 mask token。需要为每个特征对定义层索引、张量形状、投影层、归一化和 mask/padding 规则。不同教师/学生规模的特征维度不一致时，使用可训练投影头对齐维度；投影头是否参与推理应记录清楚。

此项在 MVP-B 可先不启用，但完整版本必须实现并进行“仅像素/边界”与“加入表征”的损失因果对比。

#### 6.4.5 单步折叠 `L_collapse`

当且仅当 `q_s^2` 相对 `q_s^1` 对 GT 的表现确实改善、目标可靠且样本不是不适用的负提示/多解异常样本时，才把第二步纠错知识蒸馏回空状态的一步输出路径。可以使用 `q_s^2` 作为 soft target 监督 One-pass 分支；需要明确 stop-gradient，防止两个分支互相追逐。

- 早期训练 `lambda_col` 为 0 或较小值。
- 后期逐步提高到提案初始建议 `lambda_col=0.5` 附近，再根据验证结果调整。
- 同时保留 Two-pass 评测结果，不能只展示折叠后的单步指标。

#### 6.4.6 基础能力锚定 `L_anchor`

在 COCO、RefCOCO、RefCOCO+、RefCOCOg、开放词汇和基础指代样本上，约束学生不要因困难会话样本而遗忘基础分割能力。实现可以是 Stage-0 输出蒸馏、特征约束或主要依赖基础数据回放；具体选择应在配置与实验中说明。

提案要求基础数据回放比例不低于 25%。建议初始 batch composition 中 Base 占 25%，并将 `lambda_anc` 初始建议设为 0.2，后续按验证表现调节。

### 6.5 初始超参数与调参顺序

| 参数 | 提案/计划初始值 | 调整规则 |
|---|---:|---|
| rollout 深度 K | 1 | H1 验证稳定后试 2、3 |
| distillation temperature T | 2 | 固定验证集比较，避免同时调太多项 |
| `lambda_opd` | 1 | 检查门控后有效 KD 样本比例与梯度尺度 |
| `lambda_bd` | 0.5 | 以 Boundary F、HD、Over/Under ratio 验证 |
| `lambda_rep` | 0.1 | MVP 后加入，做单独损失消融 |
| `lambda_col` | 0.5（后期目标附近） | 从 0/小值开始逐步增加 |
| `lambda_anc` | 0.2 | 监控 RefCOCO 等基础能力是否退化 |
| OPD learning rate | `1e-5`–`5e-5` | 低于基础训练；结合 LoRA 与 decoder 学习率分组 |
| 高 IoU 门控 `tau_hi` | 0.85 初始值 | 验证 0.80/0.85/0.90，不以测试集选阈值 |
| 教师改进阈值 `delta` | 0 或 0.02 初始值 | 根据验证集噪声和实际增益选择 |

推荐调参顺序：先验证前向和监督损失；再验证教师门控；然后加入像素 KD；再加入边界；再加入表征；最后加入 collapse 与 anchor。每次只增加一个主要变量。

---

## 7. Data-Policy OPD：错误画像、数据价值与采样策略

### 7.1 固定诊断集与候选池必须分离

每个 outer cycle 结束后，在固定诊断集 `D_diag` 上评估学生。诊断集必须跨周期保持一致，用于衡量模型错误如何变化；不能把不断变化的 Hard Buffer 当成唯一诊断集，否则各周期错误率不可直接比较。

候选池 `U` 则用于发现可扩展图像和提示。它可以包含授权可用的 COCO、SA-1B 子集和自建场景图像。固定诊断集负责测量；候选池负责生成与选择；训练池 `B` 与 `H` 负责提供训练样本。三者应在代码与数据 manifest 中明确区分，避免验证/测试数据泄漏。

### 7.2 错误画像向量

提案定义：

\[
\mathbf u=[E_{ent},E_{spa},E_{rel},E_{aff},E_{phy},E_{neg},E_{bd},E_{cal}].
\]

| 分量 | 主要错误 | 应驱动的数据动作 |
|---|---|---|
| `E_ent` | 选错实例、属性混淆、漏选或多选同类目标 | 属性最小对、同类干扰、语义保持改写 |
| `E_spa` | 左右/前后/布局关系错误 | 空间关系翻转、组合约束、多目标提示 |
| `E_rel` | 支撑、交互、事件关系错误 | 关系最小对、组合关系提示 |
| `E_aff` | 仅按类别选择，而非按用途/功能/可供性选择 | 用途改写、功能候选对、反事实使用 |
| `E_phy` | 稳定性、危险性、受力关系和物理安全错误 | 物理反事实、危险条件、支撑关系提示 |
| `E_neg` | 不存在对象或不满足条件时仍输出非空掩码 | 不存在对象、错误属性/关系/功能/状态负提示 |
| `E_bd` | 过分割、欠分割、孔洞、部分覆盖和边界不完整 | 边界困难样本、部件/表面区域样本 |
| `E_cal` | 置信度与实际正确性不匹配 | 置信度分层样本、校准训练和置信门控样本 |

每个诊断样本至少保存：图像 ID、提示 ID、概念类别、错误类型、预测掩码路径或可重建引用、GT/候选掩码引用、学生置信度、教师分歧、边界误差、提示类型、难度、预测版本、评测时间与指标。

错误分量的具体计算需要统一尺度。建议在固定诊断集上先计算各类别的错误率或归一化损失，再用固定基线/范围进行归一化。不能直接把不同量纲的 gIoU 差、边界误差和 ECE 原值相加。归一化方式、缺失值处理和每类样本量必须写入配置并记录在日志中。

### 7.3 数据价值函数

提案定义：

\[
V(I,p)=\alpha U+\beta D_{ST}+\gamma E_{bd}+\delta Novelty+\eta ConceptWeakness.
\]

各项含义：

- `U`：学生预测不确定性，可从像素熵、样本级置信度或不确定性估计得到。
- `D_ST`：学生与教师分歧，可基于概率掩码差异、区域差异或校准后的 disagreement 计算。
- `E_bd`：样本相关的边界误差或该类边界薄弱度。
- `Novelty`：相对训练池/Hard Buffer 的视觉或语义新颖性，避免重复生成近似提示。
- `ConceptWeakness`：该样本概念所属类别在固定诊断集上的弱项权重。

每个分量先归一化，再加权求和。权重 `alpha` 至 `eta` 必须通过验证或明确的规则设定；MVP 可用等权归一化作为起点，但应记录这是工程初始值，不是提案已验证的最优权重。对缺少教师输出的候选样本，`D_ST` 应标记缺失并使用明确的回退规则，而不是将缺失自动当作 0。

概念采样概率遵循提案形式：

\[
P(c)\propto(E_c+\epsilon)^\rho(N_c+\epsilon)^{-\mu}.
\]

其中 `E_c` 是概念错误率或薄弱度，`N_c` 是当前可用/已采样样本数量。`epsilon` 防止零值问题，`rho` 控制错误率影响，`mu` 控制样本数量反比惩罚。采样概率归一化后必须执行：

1. 每类最低采样比例，防止 Entity 等高频类别占满预算。
2. 概念采样熵或分布集中度监控，避免数据策略坍缩。
3. 每轮最大生成预算、每图像/提示重复次数限制。
4. 与基础数据回放和 Conv+/Conv− 配比共同约束，不能让 Hard Buffer 取代基础分布。

### 7.4 五类困难提示生成算子

所有算子都根据当前 `u` 选择，输出必须有可追踪来源和验证记录。

#### 算子 A：语义保持改写

- **目标：** 改变措辞、句法或表达方式，但不改变指代对象、约束和正确掩码。
- **适用：** 数据不足、提示表述敏感、通用泛化不足。
- **标签策略：** 通过语义等价验证后复用原始可靠掩码。
- **禁止：** 生成器直接查看完整 GT 掩码或答案文本；它只接收图像、原提示、概念类别和结构化错误类型。
- **验证：** 语义蕴含/等价检查、实体与约束保持检查、去重。

#### 算子 B：最小语义对

- **目标：** 只改变一个属性、关系、状态或物理条件，形成可解释的正负对。
- **适用：** `E_ent`、`E_spa`、`E_rel`，也可扩展到功能或物理条件。
- **标签策略：** 如果目标仍是原对象，可复用标签；如果目标改变，必须重新定位/分割并验证。
- **验证：** 检查是否确实只改变一个关键语义因素；对成对样本记录差异字段与预期目标变化。

#### 算子 C：组合约束

- **目标：** 在实体条件上叠加空间、关系、用途或安全限制，考察多条件组合理解。
- **适用：** `E_rel`、`E_aff` 及复杂组合提示失败。
- **标签策略：** 多目标或多解场景保留候选掩码集或软标签，不强行生成单一硬标签。
- **验证：** 逐项检查约束是否成立，确认目标不是由模板生成器臆造。

#### 算子 D：功能与物理反事实

- **目标：** 生成“可以做什么”“若条件改变会怎样”“哪里更安全”等功能、可供性、事件和物理约束问题。
- **适用：** `E_aff`、`E_phy`。
- **标签策略：** 允许多解；必要时采用候选区域、soft mask 或人工抽检。
- **验证：** 对物理、安全、支撑与稳定性判断使用独立的语义/视觉复核；不能把语言模型生成的合理性直接等同于真值。

#### 算子 E：幻觉定向生成

- **目标：** 针对学生容易激活的区域，生成不存在对象、错误属性、错误关系、错误功能或错误状态的负提示。
- **适用：** `E_neg` 高或 Negative Prompt Set 上 False Positive Area 高。
- **标签策略：** 只有经过语义验证、确实不存在满足条件的目标时，才把 GT 设为全零。
- **验证：** `C_sem` 必须确认“应为空”；不能因为纠错教师输出了一个掩码，就把该负样本判定为有效正目标。

### 7.5 数据生成器的答案隔离

生成器输入仅包含：图像、原始提示、概念类别、结构化错误类型和必要的非答案元数据。不得把完整 GT 掩码、答案文本或目标区域坐标直接交给提示生成器，否则会产生答案泄漏或模板化提示。标签验证器可以在生成完成后读取必要的 GT/候选掩码，用于验证与过滤；必须在代码接口上把“生成”和“验证”拆开。

---

## 8. 多级数据验证、质量分数与 Hard Buffer

### 8.1 综合质量分数

提案定义：

\[
Q_{data}=C_{sem}\cdot C_{geo}\cdot C_{teacher}\cdot C_{novel}.
\]

四项分数均应落在统一范围（建议 `[0,1]`），并定义清楚拒绝条件、阈值和缺失值规则。乘积结构意味着任一项很低都可能显著降低总分；实现中需记录每一项，而不是只存总分。对某些不适用项（例如明确空 GT 的教师稳定性）不能简单填成 0 或 1，应有任务特定规则。

- `C_sem`：提示语义是否成立、目标是否存在、原提示与改写是否等价、组合条件是否全部满足。
- `C_geo`：掩码是否与目标对应，是否存在明显漏分/过分割，边界和组件是否合理。
- `C_teacher`：教师输出是否稳定、置信度是否足够、与第二教师是否一致（启用双教师时）。
- `C_novel`：与训练集及当前 Hard Buffer 的近重复程度；新颖性越低，分数应越低或直接去重。

`Q_data` 的各项阈值必须在验证集上校准，并记录阈值版本。对于负提示，语义空目标验证优先于教师掩码；对于多解功能/物理样本，若没有可靠单一真值，应转为候选集/软标签或人工抽检，而不是硬性接受。

### 8.2 多级验证流程

1. **输入完整性检查：** 图像、提示、ID、原始来源与许可信息完整；无损坏文件。
2. **语义成立性检查：** 验证提示与图像一致，实体、属性、关系、功能和状态约束清楚；负提示确实不存在满足条件的目标。
3. **几何与掩码检查：** 语义保持改写复用原掩码；目标变化时执行开放词汇定位、SAM2 分割与 VLM 复核；计算面积、连通组件、孔洞和边界合理性。
4. **教师稳定性检查：** 在适用的非空目标上检查教师置信度、Recovery Gain 与双教师一致性；空 GT 不以教师是否“画出东西”作为通过条件。
5. **新颖性检查：** 文本近重复、图像近重复、掩码近重复检测；防止重复样本占用预算。
6. **人工抽检：** 对多解、物理/安全反事实和高影响负样本进行分层抽检，记录审核结果。
7. **入池或拒绝：** 依据 `Q_data`、硬性规则和配额限制写入 Hard Buffer；每个拒绝样本也记录拒绝原因，供后续诊断生成器改进。

### 8.3 数据 manifest 必备字段

每条生成样本至少包含：

- `sample_id`、`image_id`、`parent_sample_id`；
- 原始数据集/图像来源、许可或可用性标记；
- 原提示、生成提示、概念类别、错误类型、生成算子与参数；
- 正/负提示标记、GT/候选掩码引用、标签类型（hard/soft/candidate/empty）；
- 生成模型/版本、验证模型/版本、教师 checkpoint 版本；
- `C_sem`、`C_geo`、`C_teacher`、`C_novel`、`Q_data`；
- 学生 checkpoint/outer cycle、产生该样本的错误画像版本；
- 接受/拒绝状态与理由、创建时间、最近训练时间、TTL 到期周期；
- 人工审核状态及审核结论（如适用）。

### 8.4 Hard Buffer 生命周期

- Hard Buffer 容量与抽样权重由样本价值、概念配额和质量分数共同决定。
- TTL 为 **1–2 个 outer cycles**；过期条目淘汰，不允许历史困难样本无限期留存。
- 每个 cycle 末依次执行评估、生成、验证、去重、刷新和统计。
- 保存每轮入池数、拒绝率、各拒绝原因、类别直方图、样本年龄、平均价值和有效训练次数。
- 如果某概念错误已明显改善，应允许降低其生成比例；但仍受每类最低采样比例约束，防止策略完全遗忘该概念。

---

## 9. 采样配比、内外层交替与教师查询效率

### 9.1 初始 batch composition

按提案的初始比例执行：

| 数据桶 | 比例 | 内容与用途 |
|---|---:|---|
| Base | 25% | Literal、RefCOCO、开放词汇与基础指代数据；保持 Anchor 能力 |
| Conv+ | 25% | 会话式正样本，覆盖五类概念 |
| Conv− | 20% | 负提示、空 GT 与条件不成立样本，抑制幻觉 |
| Hard | 30% | 来自 Hard Buffer 的高价值困难样本 |

以上是提案初始配比，不是固定不变的最优值。任何动态调整都应保留 Base 至少 25% 的回放要求，并监控 Conv+ 与 Conv− 的实际比例、概念覆盖和训练稳定性。

### 9.2 Outer cycles 与 Inner steps

- 初始 outer cycles：4–6 轮。
- 每轮开始/结束的具体评估时点应固定。推荐每个 outer cycle 开始时使用最近一轮训练结束的 checkpoint 评估诊断集，再更新本轮采样策略。
- 每个 outer cycle 内执行固定数量的 inner steps 或 epoch，避免不同数据策略获得不公平的训练步数。
- 记录每个 outer cycle 的学生 checkpoint、错误画像、数据策略、Hard Buffer manifest、训练曲线和教师调用预算。
- 后期逐步启用/增加 collapse loss；不要在早期同时大幅改变采样策略、教师门控与损失权重。

### 9.3 在线、近在线缓存与离线 KD

| 模式 | 机制 | 优点 | 风险/限制 | 在实验中的位置 |
|---|---|---|---|---|
| 在线 OPD | 当前 step 即时调用冻结教师 | 状态最新，最接近严格 on-policy | 计算昂贵 | 高价值样本和严格在线对照 |
| 周期性近在线缓存 | 用当前学生快照生成状态，再查询教师；缓存只使用 1–2 epoch/cycle 后刷新 | 兼顾成本与状态新鲜度 | 学生更新后缓存逐渐陈旧 | 计划主路径之一，需明确刷新策略 |
| 纯离线 KD | 固定教师对固定输入/状态预先输出 | 成本低、容易复现 | 不覆盖当前学生真实状态 | Off-policy KD 基线 |

完整 Proposal 的成本因果实验要求在相同教师查询预算下比较在线、周期缓存和纯离线蒸馏。比较时同时报告查询次数、缓存命中率、训练 wall-clock、显存、精度与 Recovery Gain；不能只比较总训练时间而不控制查询预算。

### 9.4 高价值查询策略

- 对高 `V(I,p)`、高学生不确定性或高教师分歧样本优先在线查询教师。
- 其他样本可使用周期性缓存；缓存 key 必须包含图像/提示 ID、学生状态来源、学生 checkpoint 版本、教师版本和腐化/状态参数。
- 不得跨学生 checkpoint 无条件复用旧 `q_t^2`，否则会把历史状态误当成当前状态。
- 缓存 TTL 为 1–2 epoch/cycle；记录生成时间、命中情况、状态年龄和缓存失效率。
- 可探索教师量化、张量并行、选层特征缓存，但这些属于效率优化，必须保证对照实验的查询预算定义清晰。

---

## 10. 双层协同训练伪代码

```text
Input:
    student pi_theta initialized from Stage-2 checkpoint
    frozen correction teacher pi_T (optional second teacher pi_T2)
    base training pool B, unlabeled/candidate pool U
    fixed diagnostic set D_diag
    validation/test suites kept separate

Initialize:
    Hard Buffer H = empty
    teacher cache = empty
    lambda_col = 0 or a small warm-up value

for outer_cycle c = 1 ... N_outer:
    # A. Diagnose student on a fixed set; do not use H as the only diagnostic set
    evaluate pi_theta on D_diag
    compute u = [E_ent, E_spa, E_rel, E_aff, E_phy, E_neg, E_bd, E_cal]
    normalize component errors using the fixed configured rule

    # B. Build or update data policy
    score candidates using V(I,p)
    compute concept sampling probabilities P(c)
    enforce per-concept minimums, sample caps, and sampling-entropy constraints
    choose prompt operators based on u
    generate candidate prompts without exposing complete GT/answer to generator
    verify semantics, geometry, teacher reliability, and novelty
    reject invalid candidates with reason codes
    refresh H; evict items older than TTL=1–2 outer cycles
    optionally rebuild near-online teacher cache for this student snapshot

    # C. Inner Mask-Policy updates
    for each configured inner step / batch:
        sample Base / Conv+ / Conv- / Hard according to configured ratios
        q_s1 = student(I, p, empty_state)
        if query policy/cache allows:
            q_t2 = frozen_teacher(I, p, stop_gradient(q_s1))
        q_s2 = student(I, p, stop_gradient(q_s1))

        evaluate GT validity, empty-GT flag, input IoU, teacher confidence,
                 teacher improvement, teacher consistency (if dual teacher), and ambiguity
        select target mode: GT-only / teacher-hard / teacher-soft / candidate-set / reject
        compute L_sup; compute gated L_opd; compute L_bd
        if enabled: compute L_rep, L_collapse, L_anchor
        update student parameters only
        log losses, gates, teacher queries, confidence, recovery, and resource use

    # D. Stage schedule
    if in later cycles and validation supports it:
        gradually increase lambda_col
    run validation suites; save checkpoint and complete manifests

Return:
    One-pass student checkpoint
    Two-pass/refinement evaluation checkpoint or inference mode
    all config files, logs, data manifests, and experiment reports
```

---

## 11. 测试集与专项评测数据构建

### 11.1 主 CIS 评测

使用 ConVerSeg SAM-seeded 与 human-annotated 两种设置，报告总体和五类概念分项结果。两种设置的含义、数据版本和样本划分需在实验记录中写清楚；不得将同一图像的派生提示跨训练与测试集泄漏。

### 11.2 基础能力保持测试

至少包含：

- RefCOCO；
- RefCOCO+；
- RefCOCOg；
- ReasonSeg。

这些集合用于检测 DL-OPD 是否以牺牲普通指代分割或推理分割能力为代价换取困难提示收益。必须与 Stage-0 基线使用相同评测协议。

### 11.3 Hard Prompt Set

由下列样本构成：

- 语义保持改写；
- 最小语义对；
- 组合约束；
- 功能与物理反事实。

每条样本保存概念类别、错误类型、生成算子、质量置信度、来源、验证记录和标签类型。主结果按五类概念、生成算子和难度分组。

### 11.4 Mask Recovery Set

系统控制输入掩码腐化类型和严重度，覆盖腐蚀、膨胀、组件缺失、错误实例混入、边界抖动、弹性变化、平移、部分满足、正负提示翻转、mask dropout 等。按输入 IoU 分桶，比较学生第一次预测、教师纠错、学生第二步预测的质量。

至少报告：

\[
Recovery\ Gain=IoU(q_s^2,M)-IoU(q_s^1,M).
\]

对教师也报告相同定义下的 Recovery Gain，并对空 GT 单独使用 Empty Accuracy 和 False Positive Area，避免空目标的 IoU 约定掩盖假阳性。

### 11.5 Negative Prompt Set

至少覆盖：

- 不存在对象；
- 错误属性；
- 错误关系；
- 错误功能；
- 错误状态。

需要确认提示确实不满足目标条件。测试集标签应经独立验证，不能仅由待评测模型或同一个教师自动决定真值。报告 Empty Accuracy、False Positive Area、No-mask ECE，并按负提示类型拆分。

### 11.6 数据泄漏与版本管理

- 训练数据、诊断集、验证集和最终测试集应分开管理。
- 生成 Hard Prompt 时保留父样本 ID，按父图像/场景分组划分，避免改写样本泄漏到测试集。
- 任何阈值选择、门控调参和生成器选择只能使用训练/验证数据；最终测试集只用于最终报告。
- 数据版本、模型版本、生成模型版本和质量门控阈值必须在 manifest 中固定。

---

## 12. 指标定义与报告规范

| 评估维度 | 必报指标 | 解释与注意事项 |
|---|---|---|
| CIS 分割精度 | gIoU、cIoU | 总体与五类概念分别报告；明确聚合方法 |
| 边界质量 | Boundary F、HD、Over/Under ratio | 评价功能区域、部件、表面边界完整性；明确边界阈值/距离定义 |
| 负提示抑制 | Empty Accuracy、False Positive Area | 对正确答案为空的提示衡量假阳性；单独按负提示类型报告 |
| 纠错能力 | Recovery Gain | `IoU(q_s^2,M)-IoU(q_s^1,M)`，按腐化类型/严重度/输入 IoU 分组 |
| 校准 | ECE、No-mask ECE | ECE 分箱和概率定义需固定；No-mask ECE 针对空目标/负提示置信度 |
| 数据效率 | `Delta gIoU / accepted samples` | 明确 `Delta gIoU` 相对哪个基线、accepted samples 的定义与计数范围 |
| 部署效率 | Params、Latency、GPU memory | 比较 One-pass、Two-pass 和教师；注明硬件、batch size、精度模式、预热与测量区间 |
| 教师可靠性 | Teacher Recovery Gain、教师置信度、双教师一致性 | 用于解释门控是否必要，不应只在附录给总体平均值 |
| 数据策略 | 各概念采样分布、Buffer 年龄、接受率、拒绝原因、采样熵 | 证明外层策略确实随学生弱项改变，且未坍缩 |
| 训练效率 | 教师查询次数、缓存命中率、wall-clock、峰值显存 | 支持成本因果实验 |

### 12.1 统计与可视化

- 所有主要结果应报告运行配置、数据版本、随机种子和 checkpoint 选择方式。
- 若计算预算允许，关键结果采用多个随机种子并报告均值与离散程度；如果只有单次运行，应明确标记，不得写成统计显著。
- 输出训练曲线、按概念类别柱状/折线图、错误画像随 outer cycle 变化图、Hard Buffer 年龄和类别分布、门控开启率与输入 IoU 的关系图。
- 必须分析失败案例：教师偏差、边界过修、负提示假阳性、多解硬标签错误、数据策略坍缩和基础能力遗忘。

---

## 13. 主实验矩阵与五项因果验证

### 13.1 主消融矩阵

| 方法 | 外层数据策略 | 内层状态/监督 | 固定项 | 主要回答的问题 |
|---|---|---|---|---|
| ConVerSeg Stage-2 | 固定原始数据 | 无 OPD | 原始训练配置 | 原始基线 |
| Random-Augment | 随机生成/扩增 | 无 OPD | 新增样本预算、训练步数 | 数据量本身是否带来提升 |
| Error-Data | 错误驱动采样 | 无 current-state OPD | 数据预算、训练步数 | H2 / I2：数据策略贡献 |
| Off-policy KD | 固定数据 | 固定教师输出/固定状态 | 教师查询预算需记录 | 标准离线 KD 基线 |
| Random-State KD | 固定数据 | 随机腐化状态 | 状态数量和腐化预算 | 随机状态增强是否足够 |
| Historical-State KD | 固定数据 | 历史学生状态 | 历史状态存储和采样规则 | 过时学生状态的影响 |
| Mask-OPD | 固定数据 | 当前学生状态 | 训练预算与教师配置 | H1 / I1：内层贡献 |
| DL-OPD | 错误驱动数据 | 当前学生状态 | 与单层方法匹配预算 | H3：两层互补收益 |
| Collapse off/on | 固定同一训练方案 | 折叠损失关闭/开启 | 其余损失一致 | H4 / I4：单步折叠作用 |
| Teacher gate off/on | 固定数据与教师 | 门控关闭/开启 | 查询预算尽量匹配 | 门控是否防止教师弱项污染 |

### 13.2 五个关键因果验证

**实验 1：状态因果。** 固定训练数据、学生初始化、教师、训练步数与损失，仅改变状态来源：Random-State、Historical-State、Current-State。主要比较 Recovery Gain、gIoU/cIoU、Boundary F 与不同概念的结果。验证 H1。

**实验 2：数据因果。** 固定教师、学生初始化、训练步数、新增样本总量和评测集，仅改变 Random-Augment 与 Error-Data。报告总 gIoU 增益、困难类别增益、`Delta gIoU / accepted samples` 和样本接受率。验证 H2。

**实验 3：损失因果。** 逐项加入 pixel OPD、boundary loss、representation loss、collapse loss；GT 监督作为共同基准。每一步仅新增一个主要损失，报告总体与困难概念、边界质量、负提示和部署成本。教师门控另做 on/off 对照。

**实验 4：成本因果。** 在相同教师查询预算下比较在线 OPD、周期性缓存和纯离线 KD。报告性能、查询数、缓存命中率、wall-clock、GPU memory 与每次教师查询的边际收益。缓存策略必须固定 TTL 与状态版本，不能把无限新鲜的在线目标和过期缓存不加区分地比较。

**实验 5：模型规模因果。** 按提案验证 7B→3B、3B→更小模型和同规模自蒸馏。固定数据划分、查询预算和评估指标，比较不同规模下的状态蒸馏收益与成本。该项可以在 MVP-B 后执行，但完整论文路线必须保留。

### 13.3 对照公平性

- Random-Augment 与 Error-Data 的新增样本数量、训练步数和数据许可范围应一致。
- Current/Historical/Random state 实验除状态来源外尽量保持其余配置相同。
- teacher gate on/off 应报告实际 KD 样本数量；如果门控关闭使有效教师监督量明显变化，应增加匹配监督量的对照或明确限制。
- 在线/缓存/离线实验必须明确“教师查询预算”的计数单位，区分一次 batch 查询、单样本查询与缓存命中。
- 不可只报告最佳 checkpoint；checkpoint 选择规则应在实验前固定。

---

## 14. 风险分析与具体应对

| 风险 | 可能表现 | 工程应对 | 必须监控的证据 |
|---|---|---|---|
| 教师复制输入掩码 | 输出与输入高度一致，Recovery Gain 接近 0 | 教师独立图文编码、mask dropout、独立恢复评测、Teacher Recovery Gain 门槛 | 输入/输出 IoU、变化区域、Recovery Gain |
| 教师过修 | 高 IoU 状态边界变差 | `tau_hi` 关断、identity 轻约束、教师目标不覆盖 GT | 按输入 IoU 的 Boundary F/HD 与 KD 开启率 |
| 教师假阳性，尤其空 GT | 负提示产生非空掩码 | GT 优先、空 GT 不蒸馏教师掩码、负样本监督 | Empty Accuracy、False Positive Area、No-mask ECE |
| 教师偏差放大 | 伪标签错误进入闭环 | 双教师一致性、置信度门控、低置信样本只作 soft supervision | 一致率、置信度分桶错误率、低置信样本处理统计 |
| 数据策略坍缩 | 采样长期集中少数概念 | 概念覆盖下限、采样熵约束、Base 回放、配额上限 | 每 cycle 概念直方图、采样熵、类别覆盖 |
| Hard Buffer 陈旧 | 历史困难样本偏离当前学生状态 | TTL 1–2 cycles，每轮刷新，记录状态版本 | Buffer 年龄、样本重复率、旧样本收益 |
| 多解提示硬标签错误 | 合理区域被当成错误/负样本 | 候选掩码集合、soft mask、人工抽检 | 多解样本比例、抽检错误率、拒绝原因 |
| 基础能力遗忘 | RefCOCO/ReasonSeg 下降 | Anchor loss、Base 不低于 25% 回放 | 基础数据指标与训练前后差值 |
| 教师调用成本过高 | 无法完成足够消融 | 高价值在线查询、周期缓存、量化/并行、成本因果实验 | 查询次数、缓存命中率、wall-clock、显存 |
| 数据泄漏 | 近重复提示跨训练/测试集 | 按父图像/场景分组划分、manifest 追踪 | 重复检查报告、划分 ID 审计 |
| 损失数值不稳定 | NaN、梯度爆炸、某损失支配训练 | 单元测试、loss normalization、梯度裁剪/异常检测 | 每项 loss、梯度范数、NaN 计数 |
| 评价定义不一致 | 空 GT IoU、边界指标不同脚本不可比 | 统一 evaluator 与单元测试 | 同一预测在不同入口的数值一致性 |

---

## 15. 工程目录、配置与日志

### 15.1 建议目录

```text
scripts/dl_opd/
  README.zh.md
  DL_OPD_IMPLEMENTATION_PLAN.zh.md
  train_mask_opd.py             # 内层 Mask-Policy
  train_dl_opd.py               # outer/inner 总调度
  gate_teacher.py               # GT/置信度/一致性/改进门控
  build_error_profile.py        # 固定诊断集错误画像
  score_data_value.py           # V(I,p) 与 P(c)
  generate_hard_prompts.py      # 五类生成算子
  verify_qdata.py               # C_sem/C_geo/C_teacher/C_novel
  hard_buffer.py                # TTL、去重、采样和刷新
  build_teacher_cache.py        # 近在线缓存
  export_eval_suites.py         # Hard/Negative/Recovery 数据导出
  eval_dl_opd.py                # 统一评估入口
  run_ablation.py               # 实验矩阵调度
  slurm_*.slurm
configs/
  dl_opd.yaml                   # 主配置
  teacher_gate.yaml             # 门控阈值与目标模式
  data_policy.yaml              # 错误画像、价值函数、概念配额
  eval_suites.yaml              # 数据集路径和指标设置
```

可复用现有教师推理 `scripts/teacher_correction/**/eval_trg*.py`、学生评测 `eval.py` 和 `MaskCorrupter`，但应先统一输入输出协议，不应复制多个彼此不一致的指标实现。

### 15.2 主配置至少包含

- 模型：学生/教师 checkpoint、Qwen/SAM2 版本、LoRA 配置、MaskStateEncoder 配置；
- 训练：batch size、gradient accumulation、学习率、warmup、训练步数、精度模式、随机种子；
- rollout：K、状态来源、stop-gradient 开关；
- loss：各 `lambda`、温度、pixel 权重、边界损失类型、空 GT 策略；
- gate：`tau_hi`、`delta`、置信度阈值、双教师一致性阈值、低置信样本处理模式；
- data policy：错误归一化、价值函数权重、概念配额、五类算子概率、最大生成预算；
- quality：`Q_data` 分量阈值、拒绝规则、人工抽检比例；
- buffer：容量、TTL、去重阈值、采样规则；
- cache：启用模式、TTL、key 字段、刷新周期；
- evaluation：数据版本、指标实现、边界阈值、延迟测量设置；
- resource logging：教师查询次数、GPU memory、训练时间和吞吐。

### 15.3 关键单元测试

1. `MaskStateEncoder` 对不同输入尺寸、全空/全满/单点/随机概率图输出合法形状与有限数值。
2. `gate_teacher.py` 覆盖空 GT、高 IoU、教师未改进、低置信、双教师不一致、教师可靠改进等全部分支。
3. KL 方向测试：教师分布作为 target 时，优化应降低 `KL(p_teacher || p_student)`，而不是反向。
4. 空 GT loss 测试：无 NaN、假阳性受到惩罚、Dice 空集合处理符合统一定义。
5. `Q_data` 测试：各分量范围合法、缺失值有明确定义、总分与拒绝规则一致。
6. Hard Buffer TTL 测试：超过 1–2 个 outer cycles 的条目被淘汰；ID 去重与父样本追踪有效。
7. 数据划分测试：父图像/场景派生提示不会同时进入训练和测试。
8. evaluator 测试：已知合成掩码的 gIoU、Boundary F、HD、Empty Accuracy 与 False Positive Area 结果符合预期。
9. cache 测试：学生 checkpoint 或教师版本变化后，不会错误复用不匹配缓存。

---

## 16. 资源需求与可行性

按 Proposal 的资源估计：

- 基础学生训练：单卡 A100/H100 80GB 或等效多卡环境。
- 7B 纠错教师训练：建议 2–4 张 80GB GPU；推理阶段可采用量化或张量并行。
- 存储：原始图像、掩码、教师 soft logits 和中间特征建议预留 3–5TB。
- 未标注图像池：授权可用的 COCO、SA-1B 子集及自建场景图像。
- 教师查询：优先使用开放权重 VLM 保证复现；闭源 VLM 仅作为数据验证增强，而不是不可复现的唯一主路径。

以上是提案中的资源规划值，实际资源消耗需在目标服务器上通过短跑测量确认。先进行小 batch、少量样本的端到端 profiling，再决定缓存、量化、梯度累积和训练周期。记录峰值显存、每步耗时、教师查询耗时、缓存命中率与存储增长速度。

方案的技术可行性依赖于 ConVerSeg 的公开训练代码、数据格式和权重，以及 SAM2 dense prompt/Mask Decoder 对掩码状态的接口。Qwen 提示编码器使用 LoRA 继续微调可以控制可训练参数规模。新增 MaskStateEncoder 相对轻量，主要成本预计来自冻结教师在线查询；但成本结论应以实际 profiling 为准。

---

## 17. 研究里程碑与交付物

| 阶段 | 计划月份 | 主要任务 | 阶段交付/验收 |
|---|---|---|---|
| I | M1–M2 | 复现 ConVerSeg，建立统一训练、评测和错误诊断管线 | 可重复基线、五类概念错误画像、数据版本清单 |
| II | M3–M4 | 训练 Mask Refinement Teacher，验证各类腐化恢复 | 按算子/严重度报告教师 Recovery Gain，明确可靠与不可靠区间 |
| III | M5–M6 | 实现 MaskStateEncoder 与内层 Mask-OPD | Current-state 对比 Random/Historical 的状态因果结果 |
| IV | M7–M8 | 实现外层错误驱动采样、困难提示生成和多级验证 | 同新增数据预算下 Error-Data 对比 Random-Augment |
| V | M9–M10 | 完成双层协同、collapse 和跨模型规模实验 | 完整 DL-OPD、主实验、成本因果与模型规模实验 |
| VI | M11–M12 | 泛化、统计分析、代码整理、论文撰写 | 论文、训练代码、模型、数据与测试集（按授权允许公开） |

每一阶段都应有明确停止条件。若教师在某类腐化上没有稳定正 Recovery Gain，则先完善教师或禁用该类 KD，不应为了赶进度直接进入完整蒸馏。若 Current-state 不优于对照状态，先分析状态输入接口、教师是否复制掩码和状态分布，而不是直接开启 Data-Policy 掩盖问题。

---

## 18. 预期成果、验收标准与论文组织

### 18.1 预期成果

| 成果 | 内容 | 验收标准 |
|---|---|---|
| 算法 | DL-OPD 完整框架及 One-pass/Two-pass 模型 | ConVerSeg 总体与困难类别结果优于原模型和 Off-policy KD；差异须由实测支持 |
| 数据 | Hard Prompt、Minimal Pair、Negative Prompt、Mask Recovery 集合 | 每条样本有类别、错误类型、质量置信度、生成来源和验证记录 |
| 代码 | 训练、rollout、教师纠错、数据策略、缓存、评测工具 | 配置化运行，核心实验可复现，关键分支有单测 |
| 论文 | 方法论文与系统实验 | 完成主实验、核心因果消融、效率分析和失败案例 |
| 开放资产 | 模型权重、错误分类规范、数据引擎接口 | 在数据和模型许可允许范围内公开 |

建议主结论的验证目标是：在相同新增数据量与教师查询预算下，DL-OPD 同时改善 Physics/Safety、Affordance、负提示抑制与边界质量；Current-state 监督和 Error-driven 数据策略分别提供独立且互补的收益。此处是研究目标，不是预先成立的结果。

### 18.2 论文组织建议

- **Introduction：** 静态数据与静态蒸馏不能充分覆盖学生部署错误；提出双层闭环动机。
- **Method：** 先定义 Mask-Policy OPD，再定义 Data-Policy OPD，最后给出协同算法与教师可靠性机制。
- **Experiments：** 围绕 H1–H4 组织主实验和五项因果验证，不只罗列指标。
- **Analysis：** 展示错误画像随 outer cycle 迁移、Hard Buffer 的类别/年龄变化、门控覆盖率和教师查询成本。
- **Limitations：** 讨论教师偏差、数据生成成本、多解提示、严格在线程度和模型规模迁移。

Proposal 建议论文题目：*Learning from Its Own Mistakes: Dual-Level On-Policy Distillation for Conversational Image Segmentation*。方法简称使用 DL-OPD；如需突出应用场景，可使用 ConverOPD。

---

## 19. 最终验收清单

### 方法与训练

- [ ] ConVerSeg 基线可重复，Stage-0 checkpoint 与评测配置可追踪。
- [ ] 纠错教师在独立 Recovery Set 上完成按算子、严重度和输入 IoU 分析。
- [ ] `MaskStateEncoder` 接入学生并通过张量、梯度与边界案例单测。
- [ ] Current-state rollout 明确执行 `q_s^1 -> q_t^2, q_s^2`，教师冻结、状态 stop-gradient。
- [ ] GT、pixel、boundary、representation、collapse、anchor 六项损失均有明确接口和消融计划。
- [ ] KL 方向、温度、空 GT 约定和边界指标实现统一。
- [ ] 教师门控包括 GT 优先、empty GT、high-IoU、teacher improvement、confidence、dual-teacher consistency 和 low-confidence soft supervision。

### Data-Policy

- [ ] 固定诊断集与候选图像池明确分离。
- [ ] `u` 包含五类概念、负提示、边界和校准八个分量。
- [ ] 数据价值函数每一项完成归一化、缺失值处理与配置记录。
- [ ] 五类生成算子全部实现或在计划中标记阶段，并遵守生成器答案隔离。
- [ ] 多级验证实现 `C_sem/C_geo/C_teacher/C_novel` 与 `Q_data`。
- [ ] manifest 记录来源、生成操作、模型版本、质量分数、接受/拒绝原因和 TTL。
- [ ] Hard Buffer TTL 为 1–2 outer cycles，基础数据回放不少于 25%。
- [ ] 提案 MVP 所要求的规则重采样、语义保持改写、最小语义对均进入 MVP-B。

### 实验与交付

- [ ] ConVerSeg、Random-Augment、Error-Data、Off-policy KD、Random-State KD、Historical-State KD、Mask-OPD、DL-OPD 主矩阵完整。
- [ ] Collapse on/off 与 Teacher gate on/off 作为补充消融。
- [ ] 五项因果验证覆盖状态、数据、损失、教师查询成本和模型规模。
- [ ] ConVerSeg SAM-seeded/human-annotated、RefCOCO/+/g、ReasonSeg、Hard Prompt、Mask Recovery、Negative Prompt 均有明确评测入口。
- [ ] 指标覆盖 gIoU/cIoU、Boundary F、HD、Over/Under ratio、Empty Accuracy、False Positive Area、Recovery Gain、ECE、No-mask ECE、数据效率、Params、Latency、GPU memory。
- [ ] 资源 profiling、失败案例、统计配置、模型与数据版本记录完整。
- [ ] 结果中的数值均来自实际运行；未实测项明确标记，不预写结论。

---

## 20. 修订说明

本版在原实现方案的基础上补齐了原 Proposal 中未被明确落实或覆盖不充分的内容：

1. 将工程 MVP-A 与 Proposal 附录 A 的完整 MVP 区分，明确规则重采样、语义保持改写和最小语义对必须进入 MVP-B。
2. 将教师“相对学生有改进”与“教师本身可信”拆成不同门控条件，补入双教师一致性、置信度门控和低置信 soft supervision。
3. 明确固定诊断集与未标注候选池的职责分离，并补齐样本 manifest、质量分数归一化和拒绝原因追踪。
4. 补齐五类提示生成算子、多级质量验证、Hard Buffer TTL 与完整采样约束。
5. 补齐五个因果验证，特别是相同教师查询预算下的在线/周期缓存/离线比较，以及 7B→3B、3B→小模型和同规模自蒸馏。
6. 补齐所有专项数据集、评价指标、资源规划、里程碑、预期成果与论文组织。
7. 修正并澄清 pixel KD 的 KL 方向要求：常规 teacher-to-student KD 以教师分布为 target，即 `KL(p_teacher || p_student)`；原 MD 草稿中的反向写法需要在代码实现前纠正或作为有动机的单独变体验证。
8. 对 `tau_hi`、`delta`、loss weights 和其他超参数均保留为初始配置或待验证值，不将它们描述为已证实的最优设置。

**实施原则：先完成可复现基线和教师可靠性验证，再完成内层 MVP-A；随后按 Proposal 附录 A 完成 MVP-B；最后扩展为包含完整 Data-Policy、表征蒸馏、Collapse、成本因果与跨规模验证的完整 DL-OPD。**
