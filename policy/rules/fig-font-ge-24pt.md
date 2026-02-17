---
id: FIG.FONT_GE_24PT
slug: fig-font-ge-24pt
severity: error
locked: false
layer: core
artifacts: [figure]
phases: [writing-experiments, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {min_font_pt: 24, min_line_width_pt: 2.5}
conflicts_with: []
---

## Requirement

图表中所有文字（轴标签、图例、刻度标签、注释）字号不小于 `min_font_pt`（默认 24pt）。线宽不小于 `min_line_width_pt`（默认 2.5pt）。

## Rationale

学术论文中图表通常被缩放到单栏或双栏宽度。小字号在打印或投影时不可读，顶会 reviewer 常因此扣分。24pt 是确保缩放后仍可读的经验阈值。

## Check

- **LLM 检查**: 审查 matplotlib/seaborn 代码中的 `fontsize` 参数，确认 >= `min_font_pt`
- **要点**: 检查 `ax.set_xlabel(fontsize=...)`, `ax.set_ylabel(fontsize=...)`, `ax.tick_params(labelsize=...)`, `ax.legend(fontsize=...)`, `plt.rcParams['font.size']`
- **线宽**: 检查 `linewidth` / `lw` 参数 >= `min_line_width_pt`

## Examples

### Pass

```python
ax.set_xlabel("Epoch", fontsize=28)
ax.set_ylabel("Accuracy (%)", fontsize=28)
ax.tick_params(labelsize=24)
ax.legend(fontsize=24)
ax.plot(x, y, linewidth=2.5)
```

### Fail

```python
ax.set_xlabel("Epoch", fontsize=12)  # 违规：字号 < 24pt
ax.plot(x, y, linewidth=1.0)         # 违规：线宽 < 2.5pt
```
