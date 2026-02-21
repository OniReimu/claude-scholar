---
id: EXP.RESULTS_STATUS_DECLARATION_REQUIRED
slug: exp-results-status-declaration-required
severity: warn
locked: false
layer: core
artifacts: [text, figure, table]
phases: [writing-experiments, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: []
---

## Requirement

默认所有实验结果状态为 ACTUAL（无需额外声明）。  
仅当某个 `\subsubsection` 含有 fabricated/synthetic/dummy 结果时，必须在该小节开头增加异常状态声明注释，明确说明其为占位并给出原因。

- 推荐格式：`% [FABRICATED] Results in this subsection are placeholder/synthetic; awaiting HPC execution.`
- 状态声明应与对应 caption 的 `[FABRICATED]` 披露一致。

## Rationale

caption 披露解决“单图单表可见性”，而 subsection 级状态声明提供上下文范围，帮助读者理解该节内哪些结论尚未由真实执行支持。

## Check

- **LLM 语义检查**:
  - 若某 `\subsubsection` 含 `[FABRICATED]` caption，是否在小节开头有状态声明注释
  - 状态声明是否解释“为何未执行”（例如 waiting for HPC / long GPU job）
  - 是否错误地把 fabricated 声明扩散到实际结果小节
- **适用边界**:
  - 有 fabricated 结果 -> 必须声明
  - 无 fabricated 结果 -> 不要求声明（默认 ACTUAL）

## Examples

### Pass

```latex
\subsubsection{Scalability Analysis}
% [FABRICATED] Results in this subsection are placeholder projections for draft review; awaiting HPC execution.
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figs/coming-soon.png}
\caption{\textcolor{red}{\textbf{[FABRICATED] NOT EXECUTED.}} Placeholder scalability figure.}
\label{fig:scalability}
\end{figure}
```

### Fail

```latex
\subsubsection{Scalability Analysis}
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figs/coming-soon.png}
\caption{\textcolor{red}{\textbf{[FABRICATED]}} Placeholder scalability figure.}
\label{fig:scalability}
\end{figure}
% 问题：有 fabricated caption，但小节开头缺少状态声明注释
```
