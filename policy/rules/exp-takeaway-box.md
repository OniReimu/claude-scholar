---
id: EXP.TAKEAWAY_BOX
slug: exp-takeaway-box
severity: warn
locked: false
layer: core
artifacts: [text, table]
phases: [writing-experiments, self-review, revision]
domains: [core]
venues: [all]
check_kind: llm_style
enforcement: doc
params: {}
conflicts_with: []
---

## Requirement

每个实验结果表格或关键实验图之后，添加一个 takeaway box 或加粗总结句，明确提炼该实验的核心发现。格式可以是：

1. `\textbf{Takeaway:}` 加粗行
2. 自定义 `tcolorbox` / `mdframed` 环境
3. `\paragraph{Key Finding}` 段落

## Rationale

Reviewer 快速扫描论文时，takeaway box 帮助他们在不细读数据的情况下理解每个实验的核心发现。这也强制作者明确每个实验的目的和结论，避免"展示数据但不解释"的问题。

## Check

- **LLM 检查**: 每个实验表格（`\begin{table}`）或关键图（`\begin{figure}`）之后是否有明确的 takeaway 总结
- **要点**: 检查实验章节中每个结果呈现后是否有加粗总结或 box 环境

## Examples

### Pass

```latex
\begin{table}[t]
\caption{Main results on three benchmarks.}
\label{tab:main}
\centering
\begin{tabular}{lcc}
\toprule
Method & CIFAR-10 & ImageNet \\
\midrule
Baseline & 93.2 & 76.1 \\
Ours & \textbf{95.8} & \textbf{79.3} \\
\bottomrule
\end{tabular}
\end{table}

\textbf{Takeaway:} Our method consistently outperforms the baseline
across all benchmarks, with the largest gain (+3.2\%) on ImageNet.
```

### Fail

```latex
\begin{table}[t]
\caption{Main results on three benchmarks.}
\label{tab:main}
% ... table content ...
\end{table}

As shown in Table~\ref{tab:main}, our method performs better.
% 缺少明确的 takeaway，读者需要自行解读数据
```
