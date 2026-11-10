# AGENTS.md

Codex、Cursor、Claude 等工具接手前先阅读本文件，然后按顺序查看 [PROGRESS.md](docs/PROGRESS.md)、[ARCHITECTURE.md](docs/ARCHITECTURE.md) 和 [DECISIONS.md](docs/DECISIONS.md)。

## 项目与环境

本项目研究 AIST++ 舞蹈骨架的长肢体遮挡修复，保留投稿定稿、实验协议、复现代码和汇总数据。复现代码使用 Python、NumPy、PyTorch 和 Matplotlib。论文使用 MDPI LaTeX 模板和 BibTeX。

| 操作 | 仓库根目录命令 |
|---|---|
| 安装 Python 依赖 | `python -m pip install -r requirements.txt` |
| 检查 Python 语法 | `python -m compileall -q fractional_motion_repair/src` |
| 运行核心模型烟雾测试 | `python fractional_motion_repair/src/smoke_fractional.py` |
| 复现训练和评估 | 按 [fractional_motion_repair/README.md](fractional_motion_repair/README.md) 在复现目录运行 |
| 编译论文 | 按 [README.md](README.md) 的 LaTeX 命令运行 |

训练和评估脚本通过命令行参数接收数据、运行目录和种子。

## 工作规则

1. 从最新 `main` 建立 `<工具名>/<任务简述>` 分支。通过 PR 交付后续改动。
2. 在请求范围内编辑。`submission/` 是 2026-09-03 版本归档，后续修订需要明确区分版本。
3. 保持确认性测试和事后探索的标签、数据划分和随机种子。修改论断前核对协议和汇总表。
4. 提交前核查密钥、原始数据、逐样本结果、模型权重和工具缓存。按数据提供方条款处理 AIST++。
5. 按改动运行检查，将验证范围和仍待处理的事项写入 `docs/PROGRESS.md`。设计取舍记入 `docs/DECISIONS.md`。
6. 用中文向用户汇报，代码、论文和参考文献沿用已有术语。
