# Multi-Band Fractional Memory

**Endpoint-Constrained Multi-Band Fractional Memory for Repairing Long-Duration Limb Occlusions in Dance Motion Capture**

Lei Liu and Zhantu Liang. 本仓库归档 *Fractal and Fractional* 投稿材料和实验复现资料。方法将三个截断分数阶积分核用于舞蹈骨架的可见运动增量，再用端点约束构造缺失区间的修正量。

| 内容 | 入口 |
|---|---|
| 2026-09-03 定稿 PDF | [论文](submission/Fractional_Dance_Memory_FractalFract_full_autowriting_audit.pdf) |
| 投稿文件 | [源文件 ZIP](submission/MDPI_Fractional_Dance_Manuscript_Source.zip)、[LaTeX 源码与图](submission/manuscript_source/)、[标题页](submission/Title_Page.txt)、[投稿信](submission/Cover_Letter_Special_Issue.pdf) |
| 引用、版本及提交核对 | [引用核查](submission/CITATION_VERIFICATION.md)、[修订记录](submission/REVISION_NOTES.md)、[投稿说明](submission/SUBMISSION_README.txt) |
| 实验复现 | [运行说明](fractional_motion_repair/README.md)、[协议](fractional_motion_repair/protocols/)、[汇总结果](fractional_motion_repair/results/RESULTS.md) |
| 后续协作 | [AGENTS.md](AGENTS.md)、[架构](docs/ARCHITECTURE.md)、[进度](docs/PROGRESS.md)、[决策记录](docs/DECISIONS.md) |

## 快速检查

在仓库根目录运行。烟雾测试使用合成骨架，检查端点约束、已观测坐标保留、遮挡目标隔离和有限梯度。

```bash
python -m pip install -r requirements.txt
python fractional_motion_repair/src/smoke_fractional.py
```

完整实验的数据下载、缓存构建、训练和统计分析步骤见[复现说明](fractional_motion_repair/README.md)。AIST++ 原始数据从[官方项目](https://google.github.io/aistplusplus_dataset/)获取。投稿归档包含代码和汇总值。逐序列误差、训练权重和生成缓存需要通过复现流程生成。

## 重新编译论文

安装 LaTeX 和 BibTeX 后，在仓库根目录运行下列命令。图 1、图 2 使用归档中的作者绘制版本。

```bash
cd submission/manuscript_source
pdflatex -interaction=nonstopmode -halt-on-error Fractional_Dance_Memory_FractalFract.tex
bibtex Fractional_Dance_Memory_FractalFract
pdflatex -interaction=nonstopmode -halt-on-error Fractional_Dance_Memory_FractalFract.tex
pdflatex -interaction=nonstopmode -halt-on-error Fractional_Dance_Memory_FractalFract.tex
```

复现代码沿用[原发布的 MIT 许可](fractional_motion_repair/LICENSE)。AIST++ 由数据集提供方授权。论文和 MDPI 模板按原投稿材料归档。
