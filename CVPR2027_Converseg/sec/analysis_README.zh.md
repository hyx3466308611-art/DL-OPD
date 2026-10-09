# 分析

**源文件：** `analysis.tex`  
**标签：** `sec:analysis`

---

DL-OPD 的核心主张之一，是从**静态训练范式**转向**动态、共演化闭环**。本节给出分析框架：如何在 outer Data-Policy 周期上追踪学生误差分布迁移，以及 Hard Buffer 组成如何随之变化。具体曲线与统计量待实验日志汇总后填入，**下文描述预期动力学模式，不作为已观测事实**。

---

## 1. 学生误差分布的迁移

**标签：** `subsec:error_migration`

在固定诊断集上，于每个 outer cycle 末更新误差画像 $\mathbf{u}$（见方法第 2.1 节）。若闭环按设计工作，误差分布应呈**非平稳**迁移，例如：

| 阶段（示意） | 可能现象 |
| --- | --- |
| 早期 cycle | 错误主要集中在基础 grounding：$E_{\mathrm{ent}}$、$E_{\mathrm{spa}}$ 较高 |
| 内层 Mask-Policy 生效后 | $E_{\mathrm{ent}}$、$E_{\mathrm{spa}}$ 下降，暴露更深推理短板 |
| 中后期 | $E_{\mathrm{phy}}$、$E_{\mathrm{neg}}$ 等成为主导瓶颈 |
| 外层数据策略跟进后 | 复杂推理错误与幻觉率被抑制，$\mathbf{u}$ 趋于更均衡 |

该分析用于论证：静态数据分布会在已掌握概念上浪费算力，同时对**新出现的弱点**欠采样。绘图时建议对 $E_{\mathrm{ent}}$、$E_{\mathrm{phy}}$、$E_{\mathrm{neg}}$ 等分项画折线（对应设想图 `fig:evolution_dynamics`(a)）。

---

## 2. Hard Buffer 组成的演化

**标签：** `subsec:buffer_evolution`

外层数据引擎应随 $\mathbf{u}$ 更新生成算子与采样概率 $P(c)$。示意轨迹：

- **早期：** Buffer 中较多**最小语义对**与空间约束，应对 grounding 混淆。  
- **学生掌握后：** 因频率惩罚，$P(c)$ 对已缓解类别衰减。  
- **后期：** 预算转向**物理反事实**与**幻觉定向生成**；Buffer 以复杂查询为主。  
- **TTL（1–2 个 outer cycle）：** 淘汰陈旧困难样本，避免历史策略数据长期滞留。

堆叠面积图可展示 Buffer 概念/算子占比随 cycle 的变化（对应设想图 `fig:evolution_dynamics`(b)）。

---

## 3. 失败模式与局限（写作提纲）

正文 Limitations 建议覆盖提案风险表中的关键项，并配定性失败例：

1. **教师偏差：** 伪标签进入闭环时的放大风险，以及置信门控 / GT 优先的缓解效果。  
2. **多解提示：** 功能 / 物理查询的合理多解被硬标签误伤。  
3. **计算成本：** 教师查询预算与近在线缓存对严格 on-policy 程度的折中。  
4. **基础能力：** Anchor 与基础数据回放能否稳住 RefCOCO 等指标。

---

## 4. 图示占位

**Figure `fig:evolution_dynamics` — DL-OPD 共演化动力学（待替换实测图）**

- **(a)** 各 cycle 上误差分量的下降轨迹  
- **(b)** Hard Buffer 组成从「语义对」转向「物理 / 负向」的占比变化  

路径建议：`figures/evolution_dynamics.pdf`
