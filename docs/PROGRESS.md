# 项目进度

## 当前状态

仓库已整理 2026-09-03 投稿版和相应的代码、实验协议及汇总结果。原投稿包的提交检查和引用核对见 `submission/`。

该稿被 *Fractal and Fractional* 拒稿（意见为创新点不足），现改投 *Mathematics*，常规投稿（非特刊），全文重写。修订按 [修订计划](revision/MATHEMATICS_REVISION_PLAN.md) 执行，目前处于阶段 0。AIST++ 缓存已重建，旧固定划分上的 T0.1b 三种子复现通过 MPJPE 容差检查。验收结果见 [pipeline_rebuild_check.csv](../fractional_motion_repair/results/revision/pipeline_rebuild_check.csv)。

## 待办

- T0.1b 验收表待审查。阶段 0 后续任务为 T0.2（验证集归因比较）和 T0.3（逐带桥修正量诊断），两项完成后进入决策门 G0。
- 不需要数据和 GPU、可立即并行的任务：T0.4、T1.1、T2.1、T2.4、T2.5、T2.9、T5.1。
- 作者对 U1–U6 的答复已写入修订计划第 11 节；修订计划和 `DECISIONS.md` 里的决策已由作者确认并合并（PR #1）。下一个需要作者拍板的地方是 G0。

## 已知条件

- 完整复现需要自行取得 AIST++ 数据并生成运行中间文件。图 1、图 2 的最终投稿版来自作者手绘导出。
- 归档 PDF 与本次重编均为 17 页。原投稿说明中的 16 页记录与归档 PDF 页数不同。
- 现稿的有限窗对照在长缺口内存在结构性退化（修订计划 D3、命题 P3）。在 T0.3 用真实数据量化之前，不要在任何新文本里把"分数阶优于有限窗"当作已确立的结论。
- 修订计划里的解析结论只在合成信号上做过数值核对，真实数据上的结论都还没有。
- 数据集只用能直接下载、不需要注册的：AIST++ 标注（GitHub 发布页，约 2 GB）和 LAFAN1（GitHub 仓库经 Git LFS，144 MB）。数据和缓存不进仓库。
- "DanceRouter"没有发表过，新稿不用这个名称，也不引用。

## 工作记录

### 2026-11-11 · Codex · 汇总 T0.1b 复现验收结果

- 核对服务器的 `T0.1b_DONE` 和 `runs/confirmatory_test/fractional_test_results.csv`。测试覆盖 3 个训练种子、3 种方法和 5 种掩码条件，共 45 组汇总记录。`limb72` 测试包含 221 条序列、1,419 个窗口。
- 对每种方法的 `limb72` MPJPE 取三种子均值，并与 `results/main_results.csv` 的 `confirmatory` 记录计算绝对相对偏差。原始路由器、有限窗桥和分数阶桥的偏差依次为 0.5538%、1.2768%、1.2870%，均在计划规定的 2% 容差内。
- 重建缓存包含 1,408 条序列、1,123,873 帧，与归档计数一致。归档未提供 `data/sequence_cache_128.npz`，`verify_raw_cache.py` 所需的旧缓存对照尚未执行。

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
