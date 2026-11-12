# 项目进度

## 当前状态

仓库已整理 2026-09-03 投稿版和相应的代码、实验协议及汇总结果。原投稿包的提交检查和引用核对见 `submission/`。

该稿被 *Fractal and Fractional* 拒稿（意见为创新点不足），现改投 *Mathematics*，常规投稿（非特刊），全文重写。修订按 [修订计划](revision/MATHEMATICS_REVISION_PLAN.md) 执行，目前处于阶段 0。AIST++ 缓存已重建，旧固定划分上的 T0.1b 三种子复现通过 MPJPE 容差检查。验收结果见 [pipeline_rebuild_check.csv](../fractional_motion_repair/results/revision/pipeline_rebuild_check.csv)。

**期限：2026-11-17 前投稿。** 实验机为 NVIDIA RTX A6000（作者 2026-11-12 确认）；[一周排期](revision/ONE_WEEK_SCHEDULE.md)原按 RTX 4070 Super 估算，A6000 只会更宽裕。

**合并规则**：执行智能体开 PR，Claude 审查通过后直接合并，作者不需要操作。详见 `AGENTS.md` 第 7 条。

## 待办

- **先看一周排期第 4 节的当日安排，再去任务表领任务。** 排期里标为"这一周不做"的任务不要领。
- T0.1、T0.1b 已完成（PR #3）。旧划分下 `continued` 和 `fractional_loss` 各有三个种子的检查点。排期 D2（11-12）继续 T0.2（验证集归因比较）和 T0.3（逐带诊断），随后进入决策门 G0。
- 不需要数据和 GPU、可立即并行的任务：T0.4、T1.1、T2.1、T2.4、T2.5、T2.9、T5.1。
- 作者对 U1–U6 的答复已写入修订计划第 11 节；修订计划和 `DECISIONS.md` 里的决策已由作者确认并合并（PR #1）。下一个需要作者拍板的地方是 G0。

## 已知条件

- **重建管线与归档存在系统偏差，新实验只在重建管线内部比较。** `limb72` 上，重建的有限窗桥和分数阶桥 MPJPE 比归档低 1.28%、1.29%（约为归档种子标准差的 22 倍、34 倍），基础路由器低 0.55%。按测试汇总均值之比，路由器到有限窗桥的增益从归档 5.19% 变为 5.88%，有限窗到分数阶的增益分别为归档 0.261%、重建 0.271%。T0.2 的归因分析和新稿里的所有比较只用重建结果，归档数字在附录注明来源。
- **桥模型偏差原因未明。** 三个重建种子的基础专家各跑满 24 轮，桥变体各跑满 8 轮，训练历史无提前结束。三个入口脚本与归档提交中的版本一致，CUDA 训练分支启用 AMP。实测环境默认矩阵运算 TF32 关闭、cuDNN TF32 开启，脚本未覆盖这些设置。归档只记载 RTX 3090 推理计时，旧训练的实际轮数、早停、GPU、PyTorch/CUDA 和精度设置无法核对。
- [五种掩码对照](../fractional_motion_repair/results/revision/pipeline_rebuild_scope_check.csv)按归档口径先逐序列平均三个种子，再按序列等权计算 MPJPE 增益。`temporal32` 和 `temporal48` 的有限窗到分数阶增益重建为 −2.6386%、−2.5836%，归档为 −2.7179%、−2.5972%。
- 完整复现需要自行取得 AIST++ 数据并生成运行中间文件。图 1、图 2 的最终投稿版来自作者手绘导出。
- 归档 PDF 与本次重编均为 17 页。原投稿说明中的 16 页记录与归档 PDF 页数不同。
- 现稿的有限窗对照在长缺口内存在结构性退化（修订计划 D3、命题 P3）。在 T0.3 用真实数据量化之前，不要在任何新文本里把"分数阶优于有限窗"当作已确立的结论。
- 修订计划里的解析结论只在合成信号上做过数值核对，真实数据上的结论都还没有。
- 数据集只用能直接下载、不需要注册的：AIST++ 标注（GitHub 发布页，约 2 GB）和 LAFAN1（GitHub 仓库经 Git LFS，144 MB）。数据和缓存不进仓库。
- "DanceRouter"没有发表过，新稿不用这个名称，也不引用。

## 工作记录

### 2026-11-12 · claude · claude/merge-rule-and-gpu

- **完成**：按作者要求在 `AGENTS.md` 加第 7 条合并规则：执行智能体开 PR 不自行合并，Claude 审查通过后直接合并。把实验机更新为 RTX A6000（排期、本文件）。合并了 PR #4。
- **注意 / 坑**：这条规则只在本仓库生效，没有改 `ai-workflow` 共享规则里的"不要自己合并"。


### 2026-11-12 · claude · codex/t0-1b-review-follow-up（审查与合并）

- **完成**：审查 PR #4。用已提交的 45 行汇总表独立重算了验收表的三种误差，全部一致；汇总表 SHA-256 与记录一致；五种掩码对照的口径与 `aggregate_raw.py` 的 `paired_bootstrap`（序列均值之差除以基线序列均值）一致；T0.0 标签指向 `4f5724e`。
- **更正**：`keypoints3d.zip` 的 SHA-256 原记录少了一位（63 个字符）。已从 GitHub 发布页重新下载并流式计算，字节数 876,142,511 与记录相同，正确值已写入下表。
- **注意 / 坑**：
  - T0.1b 实际在 RTX A6000 上训练和评估，不是排期估算所用的 4070 Super。以后的效率表（T3.8）必须注明 GPU，且不能和归档的 RTX 3090 计时直接比较。
  - 五种掩码对照依赖服务器上的逐序列结果，仓库里只能复算点加权的验收表，不能复算对照表。
  - 速度误差和骨长误差与归档的偏差都在 0.5% 以内，只有两个桥模型的 MPJPE 偏差约 1.28%。

### 2026-11-11 · Codex · codex/t0-1b-review-follow-up

- 补齐 PR #3 的验收记录。[验收表](../fractional_motion_repair/results/revision/pipeline_rebuild_check.csv)增加速度误差和相对骨长误差，2% 判定沿用 MPJPE。核对了[服务器 45 行汇总](../fractional_motion_repair/results/revision/fractional_test_results.csv)的 SHA-256，按旧协议计算了五种掩码的逐序列增益对照。
- 在服务器重算 AIST++ v1.0 下载文件的大小和 SHA-256。`ignore_list.txt` 有 45 个不同条目，T0.1b 按旧协议保留全部 1,408 条序列。

  | 文件 | 字节数 | SHA-256 |
  |---|---:|---|
  | `keypoints3d.zip` | 876142511 | `8b2a3bfcea233b8d1859a0dc93c7800a8f9e832136ef6828d0b3ddff640ddcfd` |
  | `keypoints2d.zip` | 1316187461 | `e108a10718b805f8a909bdb590778934483e7208251ec2aae58693803b0a98d8` |
  | `ignore_list.txt` | 1217 | `be44c54baa91a1aff18306da2f3fc44abd1adfbb32317ac9f9b1362685415339` |

- 复现使用提交 `dc4662e` 中的 `raw_suite.py`（基础专家）、`fractional_screen.py`（桥和损失对照）及 `fractional_test.py`（冻结测试），运行环境为 PyTorch `2.6.0+cu124`、CUDA 12.4、NVIDIA RTX A6000。
- 完成 T0.0。GitHub 标签 `v1-fractalfract-submission` 指向投稿归档提交 `4f5724e09957bd2cbba65fcb382fdd6556807cae`。

### 2026-11-12 · claude · codex/t0-1b-pipeline-rebuild-check（审查与合并）

- **完成**：审查 PR #3，逐项重算验收表，数值全部一致，方法映射正确。解决与 PR #2 的合并冲突，更新任务表状态，记录下面的系统偏差。审查意见见 PR #3。
- **注意 / 坑**：见"已知条件"第一条。T0.1b 的遗留记录项列在"待办"里。

### 2026-11-11 · Codex · codex/t0-1b-pipeline-rebuild-check

- 核对服务器的 `T0.1b_DONE` 和 `runs/confirmatory_test/fractional_test_results.csv`。测试覆盖 3 个训练种子、3 种方法和 5 种掩码条件，共 45 组汇总记录。`limb72` 测试包含 221 条序列、1,419 个窗口。
- 对每种方法的 `limb72` MPJPE 取三种子均值，并与 `results/main_results.csv` 的 `confirmatory` 记录计算绝对相对偏差。原始路由器、有限窗桥和分数阶桥的偏差依次为 0.5538%、1.2768%、1.2870%，均在计划规定的 2% 容差内。
- 重建缓存包含 1,408 条序列、1,123,873 帧，与归档计数一致。归档未提供 `data/sequence_cache_128.npz`，`verify_raw_cache.py` 的旧缓存对照无法执行，由 T0.1b 的结果级复现替代。

### 2026-11-10 · claude · claude/one-week-schedule

- **完成**：按作者给的两个约束（4070 Super 12 GB、一周内投稿）写出 `docs/revision/ONE_WEEK_SCHEDULE.md`：必须做、限时做、不做三类范围，三条并行通道，逐日安排，种子数规则，决策门在期限内的处理。
- **未完成**：没有下载数据，没有运行任何实验。
- **注意 / 坑**：
  - 排期里"10–20 个 GPU 小时"是按代码默认的轮数和步数推算的，没有实测。D1 夜间计时后按排期第 4 节的规则修正。
  - T0.1b 在排期里从 3 个种子减为 1 个种子，完整计划的任务表没有改，以排期为准。
  - 一周没有返工余量。G0 或 G2 不过时按排期第 5 节停止成稿，不要硬投。

### 2026-11-10 · claude · claude/mathematics-revision-plan（第二次提交）

- **完成**：把作者对 U1–U6 的答复写进修订计划（v1.1）。新增 5.1 节数据获取；阶段 0 改为从下载数据开始重建，增加 T0.1b 复现性检查（与归档汇总值相对偏差不超过 2%）；第二数据集定为 LAFAN1，增加 T2.9（骨架配置抽象）、T2.10（LAFAN1 读入）；协议里加入 AIST++ `ignore_list.txt` 的处理。
- **未完成**：没有下载任何数据，没有运行任何实验。
- **注意 / 坑**：
  - AIST++ 的下载地址和大小取自官方下载页；LAFAN1 的文件大小取自其仓库的 Git LFS 指针（144,051,503 字节），序列数、帧数、帧率取自其 README。压缩包内部目录结构没有核对，下载后以实际为准。
  - LAFAN1 的许可是 CC BY-NC-ND 4.0，不能再分发数据或改编物；AIST++ 受 AIST Dance Video Database 使用条款约束。
  - 现有代码没有处理 AIST++ 官方建议剔除的 `ignore_list.txt`，旧稿用的是全部 1,408 条序列。新协议主分析剔除、敏感性分析保留。
  - LAFAN1 是 30 fps、22 关节，AIST++ 是 60 Hz、17 关节。同样的帧数缺口对应的秒数差一倍，论文里两个都要写。

### 2026-11-10 · claude · claude/mathematics-revision-plan

- **完成**：通读定稿、协议、汇总结果和 `fractional_models.py`，写出面向 *Mathematics* 的修订计划 `docs/revision/MATHEMATICS_REVISION_PLAN.md`（诊断、新定位、命题 P1–P5、代码改动、预设协议、任务表、决策门）。新增 `docs/revision/check_kernel_claims.py`，用纯 NumPy 核对命题 P1–P3，运行通过。
- **未完成**：没有运行任何训练或评估，本次会话的环境没有 GPU、PyTorch 和 AIST++ 数据。计划里所有实验任务都是待办。命题的完整证明还没写（T1.1）。
- **注意 / 坑**：
  - `python -m compileall -q fractional_motion_repair/src` 通过。`smoke_fractional.py` 因环境没有 PyTorch 未能运行；本次没有改动 `fractional_motion_repair/` 和 `submission/` 下的任何文件。
  - 97 抽头均匀窗的本通道修正量只在缺口居中时精确为零，一般起点下是变小而不是为零；跨通道贡献不受影响。引用这一点时以计划 D3 的说明为准。
  - 计划第 12 节的候选文献是检索线索，未经核对，不能直接进参考文献。
  - 期刊指标取自第三方页面，投稿前由 T5.6 复核。

### 2026-11-10 · Codex · main

- 按 `project-onboarding` 整理定稿、复现资料和协作文档。
- 核对原投稿包 34 个文件和源文件 ZIP 内 26 个文件的 SHA-256，均一致。复现文件与原发布提交一致。
- 通过 Python 语法检查及核心模型 CPU 烟雾测试。源文件 ZIP 全新解包后完成 LaTeX 和 BibTeX 编译，引用已解析。
