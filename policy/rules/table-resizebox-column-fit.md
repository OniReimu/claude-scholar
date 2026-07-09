---
id: TABLE.RESIZEBOX_COLUMN_FIT
slug: table-resizebox-column-fit
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
constraint_type: guidance
autofix: assisted
---

## Requirement

LaTeX 表格默认用 `\resizebox` 对齐目标栏宽：单栏表用 `\resizebox{\columnwidth}{!}{...}`，`table*` / 全宽环境用 `\resizebox{\textwidth}{!}{...}` 或 `\resizebox{\linewidth}{!}{...}`。

**仅当以下条件全部满足时才可省略** `\resizebox`：

1. 表格自然宽度已贴合目标栏宽 / 页宽；
2. 无 overfull hbox 风险；
3. 列间距与字号可读；
4. 不缩放的渲染效果确实更好。

## Rationale

不加 `\resizebox` 的表格是 overfull hbox 和"表格戳出栏外"的最常见来源；反之无脑缩放会把窄表的字压到不可读。默认缩放 + 显式豁免条件，比"看情况"更可执行。（改编自 DELONG-L/Academic-Paper-Skills 的 resizebox rule，MIT。）

## Check

- **LLM 检查**：`.tex` 中每个 `tabular` 环境，若外层无 `\resizebox` 包裹，核对上述 4 条豁免条件是否成立（重点：列数多、表头长、编译日志有 overfull hbox 的表）
- 单栏环境误用 `\textwidth`、全宽环境误用 `\columnwidth` 也属违规

## Examples

### Pass

```latex
\begin{table}[t]
  \centering
  \caption{Comparison of related approaches.}\label{tab:related}
  \resizebox{\columnwidth}{!}{%
  \begin{tabular}{lccc}
    \toprule
    ...
    \bottomrule
  \end{tabular}}
\end{table}
```

### Fail

```latex
% 8 列宽表未包 \resizebox，右缘戳出栏外（overfull hbox）
\begin{table}[t]
  \begin{tabular}{lccccccc}
    \toprule
    Method & Dim1 & Dim2 & Dim3 & Dim4 & Dim5 & Dim6 & Dim7 \\
    ...
  \end{tabular}
\end{table}
```
