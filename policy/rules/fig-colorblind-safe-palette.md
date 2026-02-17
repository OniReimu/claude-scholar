---
id: FIG.COLORBLIND_SAFE_PALETTE
slug: fig-colorblind-safe-palette
severity: warn
locked: false
layer: core
artifacts: [figure, code]
phases: [writing-experiments, self-review, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: []
---

## Requirement

所有多色图表必须使用色盲安全调色板（推荐 Okabe-Ito 或 Paul Tol）。避免红绿组合。图表需在灰度模式下仍可辨读。用线型（实线/虚线/点线）辅助区分，不仅依赖颜色。

## Rationale

约 8% 男性有色觉缺陷。色盲不安全的配色方案导致部分审稿人和读者无法正确解读图表。

## Check

- **LLM 检查**: 代码中是否使用了红绿直接组合
- **Colormap 检查**: 是否使用了 jet/rainbow 等不安全 colormap
- **线型检查**: 验证是否有线型区分（而非仅靠颜色）

## Examples

### Pass

```python
# 使用 seaborn 色盲安全调色板 + 不同线型
import seaborn as sns
palette = sns.color_palette("colorblind")
ax.plot(x, y1, color=palette[0], linestyle="-",  label="Ours")
ax.plot(x, y2, color=palette[1], linestyle="--", label="Baseline A")
ax.plot(x, y3, color=palette[2], linestyle=":",  label="Baseline B")
```

```python
# Okabe-Ito 手动色值
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
             "#0072B2", "#D55E00", "#CC79A7", "#000000"]
```

### Fail

```python
# 使用 jet colormap（色盲不安全）
ax.imshow(data, cmap=plt.cm.jet)

# 红绿直接组合，仅靠颜色区分
ax.plot(x, y1, color='red',   label="Method A")
ax.plot(x, y2, color='green', label="Method B")
```
