---
id: EXP.ERROR_BARS_REQUIRED
slug: exp-error-bars-required
severity: error
locked: false
layer: core
artifacts: [table, figure]
phases: [writing-experiments, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {min_runs: 3}
conflicts_with: []
---

## Requirement

所有定量结果（表格和图）必须包含误差范围（mean +/- std/stderr），并明确说明计算方法（如 "n=5 runs, std dev over seeds"）。单次运行结果不可作为主结论依据。

## Rationale

误差范围展示统计严谨性，所有顶会都期望看到。缺失误差范围意味着结果不可靠。

## Check

- **LLM 检查**: 定量表格中是否存在 +/- 符号或 CI 标记
- **Figure 检查**: 图表中是否有 error bar
- **方法论检查**: 方法论描述中是否说明了运行次数和统计方法

## Examples

### Pass

```latex
% 表格中包含误差范围，正文说明统计方法
\begin{table}
  ...
  Our Method & $83.2 \pm 1.3$ & $91.7 \pm 0.8$ \\
  ...
\end{table}
Results averaged over 5 random seeds with standard deviation.
```

### Fail

```latex
% 缺失误差范围，无运行次数说明
\begin{table}
  ...
  Our Method & $83.2$ & $91.7$ \\
  ...
\end{table}
```
