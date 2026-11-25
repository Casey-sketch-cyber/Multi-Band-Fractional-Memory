# 作图任务书（交 Codex）

- 起草：Claude，2026-11-25。依据：作者决定论文图由 Codex 画，Claude 审查。正文由 Gemini 按 [WRITING_BRIEF.md](WRITING_BRIEF.md) 写，图号以本文件为准。
- **优先级：里程碑 D、E 的实验优先于作图。** 作图不占 GPU 的任务可以在训练间隙做。

## 0. 工作方式

1. 分支 `codex/fig-<编号>`，一张图或一组相关的图一个 PR，交 Claude 审查。
2. 作图脚本放 `fractional_motion_repair/figures/`，输出放 `submission_mathematics/figures/`。数据只从 `fractional_motion_repair/results/revision/` 读，脚本里不写死任何结果数字。
3. 每张图交付：矢量 PDF（嵌入字体）、300 dpi PNG 预览、生成脚本或可编辑源文件，以及一句英文图注草稿（设置、记号，最多一句结果）。
4. PR 里附渲染后的 PNG，并说明已人工看过：文字不重叠、不出界，色盲可分辨。

## 1. 风格规范（作者要求）

- **框架图和示意图（F1、F2、F3）**：参照 CVPR/ECCV 论文 Figure 1 的做法。用 [FigGenie](https://github.com/Sleepy-Avacado/FigGenie-paper-diagram-skill) 手写 SVG（以 pt 为单位），运行它的 `check.py` 直到 0 错误，再导出 PDF（`embed_fonts.py`、`export_pdf.py`）和可编辑的 PowerPoint 副本（`svg2pptx.py`，去掉继承的主题阴影）。没有 FigGenie 时直接在 PowerPoint 里用原生形状、文本框和箭头画，`.pptx` 源文件提交进仓库。**不要用 matplotlib 或 Mermaid 画框架图。**
- **框架图几乎不放文字和数字**：只有图形、骨架小图和 2–3 个词的模块名，不写解释句，不写数值。框架图是用来突出新意的，不是让读者看懂的，理解靠正文。新的部分（核特征带、端点投影）用强调色，其余用白和灰。
- **配色**：参照近年 CVPR/ECCV 论文图的配色，低饱和、协调；全文所有图用同一套色板。方法色固定：CT 灰色，ECB-TGL 用强调色，其余核用同一色系的不同明度，K0 用中性色。红绿对比要同时配勾叉或形状区分。
- **数据图（F4–F8）**：用 matplotlib 或同类绘图库，色板与框架图一致；无衬线字体；印刷字号正文 6.5 pt，最多三种字号，不小于 6 pt；坐标轴写明单位和方向（例如 "MPJPE gain over CT (%) ↑"）；误差线或区间要在图注里说明是什么区间。
- **尺寸**：在 MDPI 模板里量 `\linewidth`，按最终尺寸出图，不靠 LaTeX 缩放。通栏图用 MDPI 的 `adjustwidth` 环境，宽度同样实测。

## 2. 图清单

| 编号 | 内容 | 类型 | 数据来源 | 依赖 |
|---|---|---|---|---|
| F1 | 方法框架：带缺口的骨架序列 → 线性插值 → 多带回火核特征 → TCN → 端点约束投影 → 补全结果 | FigGenie / PowerPoint | 无 | 无 |
| F2 | 核族示意：幂律、回火、指数、均匀窗、瞬时核的权重曲线放在同一族里，标出 β、λ 的特例关系（P1） | FigGenie 或数据图均可，按示意图规范 | 由 P1 公式计算，不读结果 | 无 |
| F3 | 端点桥示意：一条残差曲线经投影后两端为零（P2），可叠一条单侧缺口的情形 | 示意图 | 合成数据 | 无 |
| F4 | 主结果：各方法相对 CT 在 10 种测试条件上的增益，L、W 两组单独标出；ECB-TGL 突出 | 数据图 | `kernel_ablation_summary.csv`、`kernel_hypotheses.csv` | 无 |
| F5 | W 组增益对最短一带平均滞后的散点（K1–K7；K0 单独标注为不同特征类型）。**图注必须写明这是测试后的探索性分析（post hoc）** | 数据图 | `kernel_ablation.csv` 的滞后列与汇总表 | 无 |
| F6 | 学到的核形状：K5、K6 入选检查点与各自初值的权重曲线（对数纵轴），按 PR #17 的解读边界写图注 | 数据图（可改造现有 `kernel_shapes.pdf`） | `kernel_shape_fit.csv`、`kernel_shape_manifest.json` | 无 |
| F7 | 时长响应：limb24–96 七个长度上各方法相对 CT 的增益 | 数据图 | 里程碑 D 的 T3.4 结果 | 里程碑 D |
| F8 | 定性示例：同一测试缺口上真值、CT、ECB-U、ECB-TGL 的骨架帧序列；选例规则在看图前写进 PR（例如按 ECB-TGL 误差的中位数选，不挑最好的） | 数据图，骨架线框 | 冻结检查点在测试集上的推理输出（只在实验机上生成，不提交逐帧数据） | 里程碑 D |

F4–F6 现在就能做；F7、F8 等里程碑 D 合并后再做。

## 3. 审查要点（Claude 会逐项看）

- 图里的每个数都能从结果文件复算；脚本没有写死数字。
- 方法名、缩写和颜色与 [WRITING_BRIEF.md](WRITING_BRIEF.md) 第 3 节一致。
- 探索性分析的图注标明 post hoc。
- 框架图没有解释句和数字，新意部分一眼可见。
