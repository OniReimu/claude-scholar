---
id: FIG.SYSTEM_OVERVIEW_ASPECT_RATIO_GE_2TO1
slug: fig-system-overview-aspect-ratio-ge-2to1
severity: error
locked: true
layer: core
artifacts: [figure]
phases: [ideation, writing-system-model, writing-methods, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: manual
enforcement: doc
params: {}
conflicts_with: []
---

## Requirement

对 AutoFigure-Edit 生成的系统概览类图（`system-overview`、`pipeline`、`architecture`），画布宽高比（width:height）必须不小于 2:1。允许 2.1:1、3:1 等更宽比例；禁止小于 2:1 的比例（如 1.9:1、1.5:1、1:1）。

## Rationale

系统概览图通常包含多模块与箭头关系，若画布过“高”或接近方形，在双栏论文排版中会导致字体与连线拥挤，信息层次变差。采用至少 2:1 的横向比例可以保持流程方向清晰，提升在缩放后的可读性。

## Check

- **人工检查**: 在导出成稿（`final.svg`/`figure.pdf`）上确认宽高比 `width/height >= 2.0`
- **SVG 检查要点**: 使用 `viewBox` 或导出尺寸计算比例
- **PDF 检查要点**: 使用页面边界（MediaBox/CropBox）计算比例
- **LLM 审查**: 检查是否存在明显“竖向堆叠”导致的窄宽图布局

## Examples

### Pass

```text
system-overview: 2520 x 1200  -> 2.10:1 (pass)
pipeline:        3000 x 1000  -> 3.00:1 (pass)
```

### Fail

```text
system-overview: 1900 x 1000  -> 1.90:1 (fail)
architecture:    1600 x 1200  -> 1.33:1 (fail)
```
