# Visualization Best Practices for ML/AI Papers

论文级可视化的最佳实践指南。

> **Implementation**: 绘图代码实现由 `scientific-figure-making` skill 全权负责。
> 本文档仅保留 **active policy rules** 和 **决策指南**，不重复 API / 配色 / 字号等细节。
>
> 详细参考：
> - API & PALETTE → `skills/scientific-figure-making/references/api.md`
> - Design theory → `skills/scientific-figure-making/references/design-theory.md`
> - Layout patterns → `skills/scientific-figure-making/references/common-patterns.md`
> - Tutorials → `skills/scientific-figure-making/references/tutorials.md`

## Active Policy Rules（仅 2 条）

| Rule | 要求 |
|------|------|
| `FIG.NO_IN_FIGURE_TITLE` | 禁止 `plt.title()` / `ax.set_title()` / `fig.suptitle()`。标题只放 LaTeX `\caption{}` | <!-- policy:FIG.NO_IN_FIGURE_TITLE -->
| `FIG.ONE_FILE_ONE_FIGURE` | 1 文件 = 1 图。禁止 `plt.subplots(n, m)` 拼多个独立图。复合布局用 LaTeX `\subfigure` | <!-- policy:FIG.ONE_FILE_ONE_FIGURE -->

其余 fig-* rules（FONT_GE_24PT、VECTOR_FORMAT_REQUIRED、COLORBLIND_SAFE_PALETTE、SELF_CONTAINED_CAPTION、SYSTEM_OVERVIEW_ASPECT_RATIO）已退役为 `severity: info`，由 `scientific-figure-making` 的 FigureStyle / finalize_figure / PALETTE 接管。

## 图表 vs 表格选择

- **Figures（Python plots）**: 数据稀疏、需展示趋势/分布/关系、< 20 个数据点、空间编码有意义
- **Tables（`booktabs`）**: 密集数值结果、≥5 指标或 ≥5 baseline、读者需精确数值 <!-- policy:TABLE.BOOKTABS_FORMAT -->

## 误差表示

- 柱状图 → 垂直误差条（SD / SE / CI）
- 折线图 → 误差带（`alpha=0.2-0.3`）
- Caption 中必须说明误差类型（SD / SE / 95% CI）
- 多次运行（≥3-5 runs）报告 <!-- policy:EXP.ERROR_BARS_REQUIRED -->

## 常见错误速查

| 错误 | 正确做法 |
|------|---------|
| 图内加标题 | 标题放 `\caption{}` |
| 多图拼一个文件 | 每图独立文件，LaTeX `\subfigure` 组合 |
| 只显示均值 | 加误差条/误差带 |
| Y 轴从 80% 开始 | 从 0 开始或说明原因 |
| 一张图 10+ 条曲线 | 拆分多图 |

## 参考资源

- [figures4papers](https://github.com/ChenLiu-1996/figures4papers) — upstream skill source
- [Ten Simple Rules for Better Figures](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003833)
