# 架构说明

## 目录结构

```text
submission/                    2026-09-03 论文、投稿文件和核查记录
  manuscript_source/           LaTeX、参考文献、MDPI 模板和图
fractional_motion_repair/      实验复现
  src/                         数据准备、模型、训练、测试与统计
  protocols/                   开发、确认性测试和事后分析的协议
  results/                     论文汇总表与时长分析汇总
docs/                          协作交接和设计记录
```

## 模块职责

| 入口 | 职责 |
|---|---|
| `build_raw_cache.py`、`verify_raw_cache.py` | 从 AIST++ 发布的关键点构建并检查 60 Hz 缓存 |
| `pilot.py`、`raw_suite.py`、`run_suite.py` | 骨架插值、遮挡掩码、基础模型与分组实验 |
| `fractional_models.py`、`fractional_screen.py` | 分数阶与有限记忆模型定义、轨迹损失及长缺口专家训练 |
| `fractional_test.py`、`aggregate_fractional_test.py` | 冻结的分组留出测试及配对序列 bootstrap |
| `fractional_gap_sweep.py`、`aggregate_fractional_gap_sweep.py`、`fractional_order_ablation.py` | 事后时长响应与阶数消融 |
| `analyze_fractional_memory.py`、`benchmark_fractional_latency.py`、`make_fractional_paper_figures.py` | 记忆特性、效率和结果图分析 |

代码入口位于 `fractional_motion_repair/src/`。

## 数据流

AIST++ 关键点经过原采样率缓存与按编舞分组的数据划分，生成带遮挡掩码的 128 帧窗口。有限或分数阶记忆提取可见增量的上下文。端点约束和时序卷积给出长缺口修正，缺口长度路由结合短缺口专家并保留已观测坐标。测试脚本输出位置、速度和骨长误差，统计脚本按序列配对汇总。

主结论对应冻结的 72 帧单肢缺口测试。时长扫描、损失匹配对照和阶数消融对应事后分析。协议目录保存各阶段的比较条件。

## 外部依赖与来源

完整训练需要 AIST++ 关键点和 PyTorch 运行环境。生成结果图需要训练与测试产生的中间统计，论文中采用的六张图已随 LaTeX 源码归档。

复现目录基于 [StreetStyle 原发布的固定提交](https://github.com/liangdagongjue-code/StreetStyle/tree/461bdabd4a83447798cfcc29445f836c6ceddba2/fractional_motion_repair)。投稿文件来自 `Fractional_Dance_Memory_FractalFract_20260903_full_autowriting_audit` 及同版 `MDPI_Fractional_Dance_Manuscript_Source.zip`。
