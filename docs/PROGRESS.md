# 项目进度

## 当前状态

仓库已整理 2026-09-03 投稿版和相应的代码、实验协议及汇总结果。原投稿包的提交检查和引用核对见 `submission/`。

该稿被 *Fractal and Fractional* 拒稿（意见为创新点不足），现改投 *Mathematics*。修订按 [docs/revision/MATHEMATICS_REVISION_PLAN.md](revision/MATHEMATICS_REVISION_PLAN.md) 执行，目前处于阶段 0，尚未开始任何新实验。

## 待办

- 按修订计划第 7 节任务表领取任务。最先要做的是 T0.1（确认原运行目录）、T0.2（`continued` 归因对照）、T0.3（逐带诊断）、T0.4（冻结新协议），它们决定决策门 G0。
- 不需要数据和 GPU、可立即并行的任务：T0.4、T1.1、T2.1、T2.4、T2.5、T5.1。
- 需要作者答复的事项见修订计划第 11 节 U1–U6。
- 从头复现时，按协议取得 AIST++ 关键点，构建缓存并重跑三随机种子实验。

## 已知条件

- 完整复现需要自行取得 AIST++ 数据并生成运行中间文件。图 1、图 2 的最终投稿版来自作者手绘导出。
- 归档 PDF 与本次重编均为 17 页。原投稿说明中的 16 页记录与归档 PDF 页数不同。
- 现稿的有限窗对照在长缺口内存在结构性退化（修订计划 D3、命题 P3）。在 T0.3 用真实数据量化之前，不要在任何新文本里把"分数阶优于有限窗"当作已确立的结论。
- 修订计划里的解析结论只在合成信号上做过数值核对，真实数据上的结论都还没有。

## 工作记录

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
