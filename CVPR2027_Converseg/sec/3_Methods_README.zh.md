# 方法

**源文件：** `3_Methods.tex`  
**标签：** `sec:method`

---

为克服静态监督与静态数据分布的局限，我们提出 **双层在策略蒸馏（DL-OPD）**。内层 **Mask-Policy OPD** 将分割表述为掩码状态策略，并在学生当前状态下实施教师纠正；外层 **Data-Policy OPD** 依据学生误差画像动态重塑训练分布。两层在不同时间尺度上协同（见图示设想：外层塑形数据，内层纠错预测）。下文依次给出形式化、模块设计、损失与协同算法。

---

## 1. Mask-Policy OPD：状态条件纠正

**标签：** `subsec:mask_opd`

### 1.1 掩码状态策略

将单遍分割重新表述为 Markov 掩码状态策略。第 $k$ 步状态定义为

$$
s^k=(I,p,q_s^k)
$$

其中 $I$ 为输入图像，$p$ 为对话提示，$q_s^k\in[0,1]^{H\times W}$ 为学生预测的概率掩码。初始状态取空掩码 $\emptyset$。学生先给出第一次预测 $q_s^1$；冻结教师在 stop-gradient 的学生状态上产生纠正目标 $q_t^2$；学生在同一状态下 refine 得到 $q_s^2$：

$$
\begin{aligned}
q_s^1 &= \pi_\theta(I,p,\emptyset),\\
q_t^2 &= \pi_T\!\big(I,p,\mathrm{sg}(q_s^1)\big),\\
q_s^2 &= \pi_\theta\!\big(I,p,\mathrm{sg}(q_s^1)\big).
\end{aligned}
$$

$\pi_\theta$ 与 $\pi_T$ 分别表示学生与教师策略，$\mathrm{sg}(\cdot)$ 为停止梯度。教师监督随学生参数更新而改变，从而覆盖学生真实访问的错误状态。

### 1.2 MaskStateEncoder 与可训练组件

在 ConVerSeg 的 Language Adapter 与 SAM2 Mask Decoder 之间引入 **MaskStateEncoder**，将概率掩码编码到 SAM2 dense prompt 空间，并与 Qwen EOS 对应的全局视觉–语言表示融合。第一版采用轻量卷积 + 直接相加；后续可比较门控融合、Mask Token 与正负点采样等变体。

| 组件 | 输入 / 输出 | 训练策略 |
| --- | --- | --- |
| SAM2 Image Encoder | $I \rightarrow z_{\mathrm{img}}$ | 冻结 |
| Qwen2.5-VL Prompt Encoder | $(I,p)\rightarrow$ token / EOS | LoRA 微调 |
| Language Adapters | 文本状态 $\rightarrow$ sparse / dense prompts | 全量训练 |
| MaskStateEncoder | $q_s^k \rightarrow$ dense mask prompt | 新增并训练 |
| SAM2 Mask Decoder | 图像 / 提示 / 掩码状态 $\rightarrow q_s^{k+1}$ | 全量训练 |

### 1.3 纠正教师构建

教师采用更强的 **Qwen2.5-VL-7B** 与 **SAM2 Hiera-L**。为避免教师简单复制输入掩码，先从真值掩码构造具有语义意义的腐化状态（腐蚀、膨胀、部件缺失、假实例混入、边界抖动、位置偏移、部分满足、正负提示翻转等），按由轻到重的课程训练教师恢复目标掩码，然后冻结。

进入 OPD 前须**独立验证 Teacher Recovery Gain**。若教师未经掩码纠正训练、仅将学生掩码当作普通 SAM 提示，则可能放大而非修复错误。

### 1.4 联合蒸馏目标

总损失由六项构成：

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{sup}}
+\lambda_{\mathrm{opd}}\mathcal{L}_{\mathrm{opd}}
+\lambda_{\mathrm{bd}}\mathcal{L}_{\mathrm{bd}}
+\lambda_{\mathrm{rep}}\mathcal{L}_{\mathrm{rep}}
+\lambda_{\mathrm{col}}\mathcal{L}_{\mathrm{col}}
+\lambda_{\mathrm{anc}}\mathcal{L}_{\mathrm{anc}}.
$$

各项含义如下：

| 符号 | 作用 |
| --- | --- |
| $\mathcal{L}_{\mathrm{sup}}$ | 对 $q_s^1$ 与 $q_s^2$ 均施加相对 GT 的 BCE 与 Dice，避免教师错误覆盖可靠真值 |
| $\mathcal{L}_{\mathrm{opd}}$ | 像素级 Bernoulli KL；按学生熵、师生分歧与边界位置加权 |
| $\mathcal{L}_{\mathrm{bd}}$ | 边界 Dice 或有符号距离约束，强化部件 / 表面 / 功能区域几何质量 |
| $\mathcal{L}_{\mathrm{rep}}$ | 对齐选定文本 token、EOS、Language Adapter 输出与 SAM2 mask token |
| $\mathcal{L}_{\mathrm{col}}$ | **Iterative-to-Single-Pass Collapse**：仅当 $q_s^2$ 优于 $q_s^1$ 时，将第二步纠错蒸馏回第一步输出 |
| $\mathcal{L}_{\mathrm{anc}}$ | 在 COCO、RefCOCO 等基座数据上锚定，减轻灾难性遗忘 |

建议初始权重：$\lambda_{\mathrm{opd}}=1$，$\lambda_{\mathrm{bd}}=0.5$，$\lambda_{\mathrm{rep}}=0.1$，$\lambda_{\mathrm{col}}=0.5$，$\lambda_{\mathrm{anc}}=0.2$；蒸馏温度 $\tau=2$。最终取值以验证集与梯度尺度为准。

---

## 2. Data-Policy OPD：自适应分布塑形

**标签：** `subsec:data_opd`

随学生能力演化，静态训练分布逐渐次优：均匀采样对已掌握概念收益有限，错误则集中在稀有关系、负向提示、几何边界与校准不良。Data-Policy OPD 按**当前失败模式**重塑训练分布。

每个 **outer cycle** 末，在固定诊断集与未标注图像池上评估学生，得到误差画像；据此驱动选样与困难提示生成；生成样本经冻结教师验证后进入 Hard Buffer，形成「学生错误 → 定向数据」闭环。

### 2.1 学生误差画像

**标签：** `subsec:error_profile`

结构化误差向量定义为

$$
\mathbf{u}
=
\big[
E_{\mathrm{ent}},\,
E_{\mathrm{spa}},\,
E_{\mathrm{rel}},\,
E_{\mathrm{aff}},\,
E_{\mathrm{phy}},\,
E_{\mathrm{neg}},\,
E_{\mathrm{bd}},\,
E_{\mathrm{cal}}
\big].
$$

| 分量 | 含义 |
| --- | --- |
| $E_{\mathrm{ent}}$–$E_{\mathrm{phy}}$ | 实体/属性、空间、关系、可供性、物理与安全推理错误 |
| $E_{\mathrm{neg}}$ | 负向提示下的幻觉 |
| $E_{\mathrm{bd}}$ | 边界几何错误 |
| $E_{\mathrm{cal}}$ | 预测校准误差 |

样本级额外记录概念类别、错误类型、置信度、师生分歧、边界误差、提示类型与难度，以区分「选错实例 / 欠分割 / 负向幻觉」等不同失败模式。

错误族与对应数据动作：

| 错误族 | 典型表现 | 数据动作 |
| --- | --- | --- |
| 实体 / 属性 | 选错实例、属性混淆、漏掉同类对象 | 属性最小对、同类干扰、语义保持改写 |
| 空间 / 关系 | 左右、前后、支撑或交互关系错误 | 关系翻转、组合约束、多目标提示 |
| 功能 / 可供性 | 仅按类别而非用途选择区域 | 用途改写、反事实使用、功能候选对 |
| 物理 / 安全 | 稳定性、危险性、受力判断错误 | 物理反事实、危险条件、支撑关系提示 |
| 负向提示 | 对不存在或不满足条件的对象产生非空掩码 | 不存在物体、错误属性 / 状态提示 |
| 几何边界 | 过分割、欠分割、部分覆盖、孔洞 | 边界困难样本、部件 / 表面区域数据 |

### 2.2 自适应数据价值

**标签：** `subsec:data_value`

候选样本 $(I,p)$ 的数据价值为

$$
V(I,p)
=
\alpha U
+\beta D_{\mathrm{ST}}
+\gamma E_{\mathrm{bd}}
+\delta N
+\eta W_{\mathrm{weak}},
$$

其中 $U$ 为学生不确定性，$D_{\mathrm{ST}}$ 为师生分歧，$E_{\mathrm{bd}}$ 为边界误差，$N$ 为视觉/语义新颖性，$W_{\mathrm{weak}}$ 为对 $\mathbf{u}$ 中高误差概念的加权。

概念级采样引入逆频率修正，避免高频 Entity 持续占用预算：

$$
P(c)\propto (E_c+\varepsilon)^{\rho}\,(N_c+\varepsilon)^{-\mu}.
$$

每个概念设置最低采样比例；Hard Buffer $\mathcal{H}$ **最多保留约 1–2 个 outer cycle**（TTL），以限制分布坍缩与陈旧策略数据。

### 2.3 定向困难提示生成

**标签：** `subsec:hard_prompt`

五类互补算子，按 $\mathbf{u}$ 中主导误差维度选择：

1. **语义保持改写** — 改变表达、复用原掩码（低噪声首选）  
2. **最小语义对** — 只改一个属性 / 关系 / 状态 / 物理条件  
3. **组合约束** — 在实体条件上叠加空间、关系、用途或安全限制  
4. **功能与物理反事实** — 「可以做什么 / 条件改变会怎样」类查询  
5. **幻觉定向生成** — 围绕学生错误激活区域生成不存在对象、错误属性与错误关系  

### 2.4 多级验证与质量门控

**标签：** `subsec:data_verification`

生成样本依次通过语义成立性、掩码几何、教师稳定性与新颖性检查：

$$
Q_{\mathrm{data}}
=
C_{\mathrm{sem}}\cdot C_{\mathrm{geo}}\cdot C_{\mathrm{teacher}}\cdot C_{\mathrm{novel}}.
$$

语义保持改写可复用已验证掩码；目标变化时重新执行开放词汇定位、SAM2 分割与 VLM 验证。对多解功能 / 物理提示，优先使用 soft mask 或候选掩码集合，避免将合理多解压成错误硬标签。

**质量原则：** 数据生成器**不得**直接读取完整 GT 掩码或答案文本；仅接收图像、原提示、概念类别与结构化错误类型，以降低答案泄漏与模板化提示。

---

## 3. 双层协同训练

**标签：** `subsec:data_policy_algorithm` / `alg:dlopd`

### 3.1 训练阶段

| 阶段 | 目标 | 核心操作 |
| --- | --- | --- |
| Stage 0 | 获得基础 ConVerSeg 学生 | Literal / RefCOCO / Open-vocab / Conversational 两阶段课程 |
| Stage 1 | 获得纠正教师 | GT 腐化恢复训练后冻结，验证 Recovery Gain |
| Stage 2 | 启动内层 OPD | 在当前学生掩码状态上做像素 / 边界 / 表征蒸馏 |
| Stage 3 | 启动外层数据闭环 | 误差画像、困难数据生成、教师验证、刷新 Hard Buffer |
| Stage 4 | 能力折叠与收敛 | 逐步提高 collapse loss，输出 One-pass / Two-pass 版本 |

### 3.2 协同算法（概要）

对每个 outer cycle $c=1,\ldots,N_{\mathrm{outer}}$：

1. **Data-Policy OPD**  
   评估 $\pi_\theta$，更新 $\mathbf{u}$；计算 $V(I,p)$ 与 $P(c)$；从未标注池采样并生成困难提示；按 $Q_{\mathrm{data}}$ 验证；刷新 $\mathcal{H}$。  
2. **Mask-Policy OPD**（内层 $k=1,\ldots,N_{\mathrm{inner}}$）  
   从 $\mathcal{H}\cup\mathcal{B}$ 采样；计算 $q_s^1$、$q_t^2$、$q_s^2$；用 $\mathcal{L}$ 更新 $\theta$。

返回优化后的 $\pi_\theta$。

### 3.3 批次配比与效率

建议初始配比：Base 25% / Conv+ 25% / Conv− 20% / Hard 30%；outer cycles 约 4–6；内层 rollout 自 $K=1$ 起步。OPD 阶段学习率低于基础训练（约 $1\times10^{-5}$–$5\times10^{-5}$）。

严格逐 batch 在线查询教师代价较高。采用**在线小比例 + 周期性近在线缓存**：对高价值样本即时查询教师；其余样本使用当前学生快照 rollout，缓存仅保留 1–2 个 epoch 后刷新，以平衡成本与 on-policy 新鲜度。

### 3.4 风险控制（摘要）

| 风险 | 应对 |
| --- | --- |
| 教师复制输入掩码 | 掩码 dropout、独立图文编码、Recovery Gain 门槛 |
| 教师偏差放大 | GT 优先、置信门控、低置信仅 soft 监督 |
| 数据策略坍缩 | 类别覆盖下限、采样熵约束、基础数据回放 |
| Hard Buffer 陈旧 | TTL 限制、每轮重新生成 |
| 基础能力遗忘 | Anchor loss、不低于约 25% 基础数据回放 |
