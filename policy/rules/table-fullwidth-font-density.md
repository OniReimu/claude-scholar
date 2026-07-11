---
id: TABLE.FULLWIDTH_FONT_DENSITY
slug: table-fullwidth-font-density
severity: warn
locked: false
layer: core
artifacts: [table]
phases: [writing-experiments, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: []
constraint_type: guardrail
autofix: none
---

## Requirement

全宽（`table*` / 双栏）表格若使用 `\resizebox`，缩放后表内字号必须 **≤ 正文字号**——即 `\resizebox` 只允许缩小、不允许放大。

判据：一张表内容稀疏到 `\resizebox{\textwidth}{!}{...}` 会把字体放大到超过正文字号，就证明它不配占双栏。此时只有两条出路：

1. **降为单栏 `table`**（用 `\resizebox{\columnwidth}{!}{...}`）；或
2. **增加信息密度**——补更多有信号的 metrics/列，使表格真正需要全宽承载。

不允许"内容不足却硬拉全宽、靠 resizebox 把字放大占满"这种做法。

## Rationale

`\resizebox` 按目标宽度等比缩放，宽度撑满而列少行少时会把字号放大到比正文还大，视觉上突兀、浪费双栏这一稀缺版面，且暴露"这张表数据不够"。全宽版面应留给信息密度真正高的表。本条给 `TABLE.RESIZEBOX_COLUMN_FIT`（默认对齐栏宽）补上方向性约束（只缩不放），给 `TABLE.DIMENSION_BUDGET`（默认单栏、先剪列再上 `table*`）补上"加 metrics"这条对称出路。

## Check

- **LLM 检查**：对每个 `table*`/全宽表，核对列数×行数的信息密度是否足以撑满 `\textwidth` 而无需放大字体。判断信号：列数很少（如 ≤3 数据列）却用 `\resizebox{\textwidth}`；`\resizebox` 高度参数为 `!` 且自然宽度远小于 `\textwidth`。命中即建议降单栏或加 metrics。

## Examples

### Pass

```latex
% 全宽表信息密度高（8 个 metric 列），resizebox 缩小到 textwidth，字号 ≤ 正文
\begin{table*}[t]
  \centering
  \caption{Full benchmark across eight metrics.}\label{tab:full}
  \resizebox{\textwidth}{!}{%
  \begin{tabular}{lcccccccc}
    \toprule
    Method & Acc & F1 & AUC & Latency & Mem & FLOPs & Params & Energy \\
    \midrule
    ...
    \bottomrule
  \end{tabular}}
\end{table*}
```

### Fail

```latex
% 只有 3 个数据列却硬占双栏，resizebox 把字体放大到超过正文 → 应降单栏或加 metrics
\begin{table*}[t]
  \centering
  \resizebox{\textwidth}{!}{%
  \begin{tabular}{lccc}
    \toprule
    Method & Accuracy & Latency & Memory \\
    \midrule
    Ours & 0.91 & 12ms & 1.2GB \\
    Base & 0.88 & 15ms & 1.5GB \\
    \bottomrule
  \end{tabular}}
\end{table*}
```
