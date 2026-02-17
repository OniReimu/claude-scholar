---
id: REF.CROSS_REFERENCE_STYLE
slug: ref-cross-reference-style
severity: warn
locked: false
layer: core
artifacts: [text]
phases: [writing-background, writing-methods, writing-experiments, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: []
---

## Requirement

论文正文中引用图表、方程、章节时，使用 LaTeX 交叉引用命令而非硬编码数字。项目规范格式：

- 图表：`Fig.~\ref{fig:xxx}`
- 表格：`Table~\ref{tab:xxx}`
- 章节：`\S\ref{sec:xxx}`
- 附录：`\textbf{Appendix~\ref{app:xxx}}`
- 方程：`\eqref{eq:xxx}`
- 算法：`Algorithm~\ref{alg:xxx}`
- 代码清单：`Listing~\ref{lst:xxx}`

使用 `~`（非断行空格）连接名称和编号。

## Rationale

硬编码数字在图表/方程增删时需要手动更新全文，极易遗漏导致引用错误。`\ref` 和 `\eqref` 自动同步编号。统一的引用格式（`Fig.` 而非 `Figure`，`\S` 而非 `Section`）保持全文一致性并节省版面空间。

## Check

- **LLM 检查**: 搜索正文中的 "Figure 1"、"Fig 1"（缺点号）、"Table 2"、"Eq. 3"、"Section 4" 等硬编码引用模式
- **regex pattern**: `(Figure|Fig\s|Table|Eq\.|Section|Algorithm|Listing)\s+\d+` 检测硬编码引用（排除 caption 内的描述）
- **格式检查**: 确认使用 `Fig.~\ref`（非 `Figure~\ref`），`\S\ref`（非 `Section~\ref`），`\eqref`（非 `Eq.~\ref`）

## Examples

### Pass

```latex
As shown in Fig.~\ref{fig:architecture}, our model consists of...
The objective function (\eqref{eq:loss}) minimizes...
Results are summarized in Table~\ref{tab:main-results}.
We describe the details in \S\ref{sec:methods}.
See \textbf{Appendix~\ref{app:proofs}} for full proofs.
Algorithm~\ref{alg:training} outlines the procedure.
```

### Fail

```latex
As shown in Figure 1, our model consists of...  % 硬编码
As shown in Figure~\ref{fig:arch}...            % 应为 Fig.~\ref
The objective function (Eq. 3) minimizes...      % 硬编码
We describe details in Section~\ref{sec:methods} % 应为 \S\ref
Results are summarized in Table 2.               % 硬编码
```
