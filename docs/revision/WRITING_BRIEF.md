# 写作任务书（交 Gemini）

- 起草：Claude，2026-11-25。依据：作者 2026-11-25 决定 G1 走 S1 线，论文由 Gemini 写、图由 Codex 画、Claude 审查合并。
- 开工前先读 [AGENTS.md](../../AGENTS.md)、[PROGRESS.md](../PROGRESS.md)、[DECISIONS.md](../DECISIONS.md)、冻结协议 [mathematics_revision_protocol.md](../../fractional_motion_repair/protocols/mathematics_revision_protocol.md)、[修订计划](MATHEMATICS_REVISION_PLAN.md)第 4 节。
- 图的要求见 [FIGURE_BRIEF.md](FIGURE_BRIEF.md)。正文里引用图时用那份清单里的编号和文件名。

## 0. 工作方式

1. 分支命名 `gemini/<任务>`，一个写作任务一个 PR，交 Claude 审查。不要自己合并。
2. 只改 `submission_mathematics/` 和 `docs/PROGRESS.md`（工作记录）。不改代码、结果文件、协议和 `submission/`（旧稿归档）。
3. **论文里的每个数字都必须能在 `fractional_motion_repair/results/revision/` 的某个文件里找到。** 在 PR 描述里列出"数字 → 文件 → 行或列"。还没有结果的地方写 `\todo{...}`，不要估计、不要从旧稿抄。
4. 每个 PR 附第 7 节格式的"主张–证据"表和自查结果。
5. 正文用英文；PR 描述和工作记录用中文。

## 1. 期刊格式（MDPI *Mathematics*）

- 模板：把 `submission/manuscript_source/Definitions/` 复制到 `submission_mathematics/Definitions/`，主文件 `submission_mathematics/manuscript.tex`，类选项 `\documentclass[mathematics,article,submit,moreauthors,pdftex]{Definitions/mdpi}`；参考文献 `submission_mathematics/references.bib`。
- 摘要不超过 200 词，一段，不引用文献。关键词 3–10 个。
- 结构：1 Introduction；2 Related Work；3 Problem Setting and Preliminaries；4 Endpoint-Constrained Kernel Bridges（算子与命题）；5 Learning Architecture；6 Experimental Protocol；7 Results；8 Discussion（含 Limitations）；9 Conclusions；Appendix A 证明；Appendix B 早期固定划分研究；Appendix C 补充表。
- 结尾声明：Author Contributions、Funding、Data Availability、Conflicts of Interest 用占位符 `\todo{作者填写}`，由作者自己填。Data Availability 的正文写明 AIST++（GitHub 发布页 `google/aistplusplus_dataset`，遵守 AIST Dance Video Database 条款）和代码仓库，作者确认后定稿。

## 2. 故事（关键内容图）

| 关键内容 | 本文的回答 |
|---|---|
| 任务 | 骨架动作序列的长缺口补全：某个肢体或全身在连续数十帧里缺失，要从可见部分恢复 |
| 目标量 | 缺失帧上的 MPJPE（按窗口、种子、条件的顺序平均），同时看速度误差、骨长误差、端点速度跳变 |
| 现有方法的表面问题 | 学习式补全在缺口两端与观测不连续；带时间记忆的设计说不清该用多长的记忆，对照往往不公平 |
| 根本原因 | 端点连续性靠网络自己学，没有结构保证；记忆核的形状和尺度被当作固定超参数，没有放进同一族里比较 |
| 我们的方案 | 端点约束核桥：对线性插值的残差增量做核特征，再投影到"缺口两端修正为零"的子空间；核取回火 Grünwald–Letnikov 族，β、λ 可学习，幂律、指数、均匀窗都是它的特例 |
| 为什么有效 | 投影是有界、幂等的线性算子，端点精确；特征恒等于多尺度对称二阶差分的加权和，核只决定尺度权重，可学的回火参数让模型自己选尺度 |
| 新认知 | 在预先登记、5 折、聚类 bootstrap 的检验下，可学尺度的回火核优于所有固定形状的核；但学到的记忆极短（平均滞后 0.04–0.52 帧），收益来自局部曲率尺度的特征，长程记忆没有带来额外收益 |

**一句话主张**：端点约束的核桥加上可学尺度的回火分数阶核，在预先登记的受控比较中稳定优于继续训练和各种固定记忆核；它的作用机制是学习局部二阶差分特征的尺度，而不是利用长程记忆。

**题目（S1 线，作者已同意）**：*Learnable Tempered Fractional Kernels with Exact Endpoint Constraints for Long-Gap Skeletal Motion Imputation*

## 3. 术语与命名（全文统一，首次出现时定义）

| 代码名 | 论文名 | 缩写 |
|---|---|---|
| 线性插值 `X_lin` | linear interpolation | — |
| B0 `continued` | continued training baseline | CT |
| B3 | endpoint-constrained kernel bridge（输入侧加输出侧端点投影） | ECB |
| K0 | instantaneous cross-joint velocity regression（不是核族成员，见 DECISIONS 2026-11-12） | ECB-K0 |
| K1 | uniform windows 9/33/97 | ECB-U |
| K2 | uniform windows 51/65/97 | ECB-U′ |
| K3 | fixed-order GL kernels（β = 0.225/0.5/0.775） | ECB-GL |
| K4 | exponential kernels matched to K3 mean lags | ECB-Exp |
| K5 | learnable tempered GL kernels, three bands（本文方法） | ECB-TGL |
| K6 | free nonnegative 97-tap kernels | ECB-Free |
| K7 | single-band learnable tempered GL kernel | ECB-TGL1 |
| L 终点 | long single-limb gaps (60–96 frames) | — |
| W 终点 | whole-body gaps (32, 48 frames) | — |

基础架构（短缺口专家、长缺口专家、时长路由）作为本文的基础系统完整描述，称为 duration-routed two-expert TCN。**不要使用、也不要引用"DanceRouter"。**

## 4. 数学部分（第 3、4 节和附录 A）

命题陈述、证明要点见[修订计划](MATHEMATICS_REVISION_PLAN.md)第 4 节，数值核对脚本见 [check_kernel_claims.py](check_kernel_claims.py)。每个命题写成：陈述 → 完整证明（放附录 A）→ 正文一句"这说明了什么"。

- **P1 回火 GL 核的闭式性质**：生成函数、归一化常数、平均滞后、方差（未截断时为负二项分布）；截断无回火时的和与平均滞后；渐近幂律 a_k(β) ~ k^{β−1}/Γ(β)。说明 β = 1、λ = 0 退化为均匀窗，β = 1、λ > 0 为指数核。并注明 a_k(β) 是分数差分 (1 − B)^{−β} 的 MA(∞) 系数。
- **P2 端点桥 = 投影加累加**：P = I − 11ᵀ/L；端点为零；幂等；L2 和 L∞ 上界。
- **P3 零空间与均匀窗的盲区**：零空间是常数增量；W 抽头均匀窗的盲区帧数 max(0, g + 3 − 2W)。
- **P4 无目标泄漏**：适配器输出只依赖 (M, M ⊙ X)。
- **P6（新增，2026-11-25）多尺度二阶差分恒等式**：可见性全为 1 时，掩码双向上下文减去原增量等于

  ```math
  (\mathcal{K}-\delta)*d\,(t)=\sum_{k\ge 1}\frac{c_k}{2C}\,\big(d_{t+k}-2d_t+d_{t-k}\big),\qquad C=\sum_{k\ge0}c_k .
  ```

  证明：分子含 k = 0 项 2c_0 d_t，分母为 2C，减去 d_t 后 k = 0 项抵消，其余项重排即得。推论：(i) β → 0 时 c_k / C → 0（k ≥ 1），特征恒为零，适配器退化为 CT，这就是 K0 不能取核族极限的原因；(ii) 核只决定各尺度二阶差分的权重，平均滞后刻画了特征的空间尺度。有遮挡时把 c_k 换成可见性加权，结论在可见区域内成立。
- P5 不写。

**措辞边界**：P6 是代数事实，可以放在第 4 节作为结果。第 7 节里"增益随最短一带平均滞后单调变化"是看过测试结果后的探索性观察，必须写明 *post hoc*。

## 5. 各节要点

### 1 Introduction（按 V3 开头）

1. 长缺口补全的应用（动作捕捉清洗、遮挡下的姿态估计），本文聚焦"数十帧的肢体或全身缺口"。
2. 现有方法：经典插值和低秩方法在长缺口上误差快速增长；学习式补全和缺失时序插补（GRU-D、BRITS、SAITS）把端点连续性交给网络，记忆形状隐式或固定（GRU-D 只有指数衰减）。由此引出两个问题：端点没有结构保证；记忆核该多长，没有在同一族里公平比较。
3. 本文方案：段落模板见故事表"我们的方案"和"为什么有效"。
4. 贡献列表（3 条，每条可验证）：
   - 端点约束核桥及其性质（P1–P4、P6）；
   - 统一核族下的预先登记比较（5 折、3 种子、聚类 bootstrap、Holm），ECB-TGL 优于全部对照；
   - 机制解释：学到的尺度极短，收益来自局部曲率尺度特征（P6 加探索性分析）。
5. 结果摘要：见第 6 节表格里 H1 和 H2 的数字。

**禁止**：先讲一个朴素方案再一步步打补丁；写"first""novel"。

### 2 Related Work（4 个主题）

经典补缺（样条、低秩/PCA、卡尔曼）；学习式运动补全与中间帧生成；缺失时序插补；分数阶与长记忆模型（ARFIMA、回火分数阶微积分、分数阶在人体运动中的应用）。每个主题写：范围 → 代表方法 → 与本问题相关的局限 → 本文的区别。文献线索见修订计划第 12 节，**每条引用都要核对出处后才能进 `references.bib`**，在 PR 里列出核对结果。

### 3–5 预备知识、算子、架构

按修订计划第 4、5 节写。架构一节写清：线性插值初值 → 增量与可见性 → 三带核特征（每带混合、门控、1×1 投影）→ 输入侧端点桥 → TCN → 输出侧端点投影（单侧缺口只约束可见端，见 DECISIONS 2026-11-16）→ 时长路由。训练细节（8 轮上限、早停 5/4 轮、记忆参数 10 倍学习率、float32 计算掩码上下文的数值稳定性修正）放在第 6 节末尾。

### 6 Experimental Protocol

按冻结协议写：数据（AIST++，剔除 `ignore_list.txt` 的 45 条后 1,363 条序列、197 个编舞组）、5 折分组划分、内层验证、3 个训练种子、10 种测试条件和掩码种子、测试集冻结、H1–H3 判据、Holm 分族、等效界 ±0.2%、非劣界 −0.5%。写明协议在测试前冻结，以及冻结后的两处更正（早停口径、种子数），见 DECISIONS 2026-11-16。

### 7 Results（小节标题写成结论句）

1. *The endpoint-constrained bridge improves on continued training*（H1）。
2. *Learnable tempering outperforms every fixed kernel shape*（H2a、H2b，加 K0、K2、K3、K6、K7 的描述性比较）。
3. *Learned memory scales are short*（核形状图，K5/K6 平均滞后，P6 的解释；探索性分析，标 post hoc）。
4. *Comparison with classical and learned imputation baselines* —— `\todo{里程碑 D}`。
5. *Gap-duration response and efficiency* —— `\todo{里程碑 D}`。
6. *Second dataset and realistic occlusion* —— `\todo{里程碑 E}`。

### 8 Discussion

新认知；适用边界：K7（单带）只比 K5 差 0.05%/0.49%，多带的价值很小；L 组上相对均匀窗只好 0.28%；8 轮训练预算（B0 10/15、K5 14/15、K6 12/15 选在第 8 轮，英文草稿已在 `fractional_motion_repair/README.md`）；学到的核形状依赖参数化（PR #17 审查结论），不能当作数据本身记忆长度的证据；单数据集（除非里程碑 E 完成）；128 帧窗口。

### 附录 B

旧固定划分上的研究和复现（T0.1b、T0.2、T0.3），注明"重建管线与归档有系统偏差"，旧数字只以归档值身份出现。

## 6. 已有的确证性数字（来源 `kernel_hypotheses.csv`、`kernel_ablation_summary.csv`）

| 比较 | 终点 | 增益 | 95% 区间 | Holm 差异 p | 结论 |
|---|---|---:|---|---:|---|
| H1：ECB-TGL 对 CT | L | +3.521% | [2.954, 4.170] | 0.0004 | 更好 |
| H1：ECB-TGL 对 CT | W | +7.862% | [7.051, 8.736] | 0.0004 | 更好 |
| H2a：对 ECB-U | L | +0.279% | [0.194, 0.375] | 0.0008 | 更好，效应小 |
| H2a：对 ECB-U | W | +1.940% | [1.683, 2.210] | 0.0008 | 更好 |
| H2b：对 ECB-Exp | L | +0.699% | [0.579, 0.821] | 0.0008 | 更好 |
| H2b：对 ECB-Exp | W | +5.754% | [5.204, 6.312] | 0.0008 | 更好 |
| H3：对 ECB-U | W | 下限 +1.683% | — | — | 不劣 |
| K0 等效（探索） | L / W | 90% 区间 [0.570, 0.822] / [6.144, 7.300] | — | — | 不等效，ECB-TGL 更好 |

样本：1,363 条序列，197 个编舞组，5 折各 273/310/213/295/272 条；毫米误差覆盖 411 条。论文里所有百分数保留两位小数，区间与点估计同精度。

## 7. 每个 PR 的交付格式

1. 本节提纲（3–7 条，中文）。
2. 英文正文（LaTeX）。
3. 主张–证据表：`主张 | 证据（文件、行或列） | 状态：supported / needs evidence / weaken`。
4. 自查：每段一个中心意思；句间衔接；术语与第 3 节一致；没有无证据的主张；没有"first/novel"。

## 8. 任务顺序

| ID | 内容 | 依赖 |
|---|---|---|
| W1 | `submission_mathematics/` 骨架（模板、主文件、各节空文件、`references.bib`），能编译 | 无 |
| W2 | 第 3、4 节和附录 A（P1–P4、P6 及证明） | W1 |
| W3 | 第 5、6 节 | W1 |
| W4 | 第 1、2 节（文献逐条核对） | W2、W3 |
| W5 | 第 7 节 1–3 小节，第 8 节初稿，附录 B | W3、图 F4–F6 |
| W6 | 第 7 节 4–6 小节 | 里程碑 D、E |
| W7 | 摘要、题目、关键词、结论；对照第 9 节自审清单逐项写 pass/needs revision | W1–W6 |

## 9. 投稿前自审清单（W7 时逐项回答）

贡献（读者学到什么；是否只是可预见的小增益）、清晰度（能否复现；每个模块有没有动机；术语是否一致）、结果强度（增益是否有意义；跨折跨种子是否一致；失败是否报告）、完整性（基线、消融、指标是否齐全）、可靠性（设置是否现实；假设是否合理；是否无逐例调参）。**硬规则：摘要和引言里的每个主张都要有结果支持，否则补证据、减弱或删除。**
