---
id: EXP.FABRICATED_RESULTS_CAPTION_DISCLOSURE
slug: exp-fabricated-results-caption-disclosure
severity: error
locked: false
layer: core
artifacts: [figure, table, text]
phases: [writing-experiments, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: []
---

## Requirement

只要 Experimental Results 中出现非实际执行产出的结果（例如 fabricated/synthetic/dummy/placeholder/projection），其对应 figure/table caption 必须显式披露，并使用红色大写标记：

- 推荐格式：`\textcolor{red}{\textbf{[FABRICATED] ...}}`
- 披露内容应至少包含“非实际执行产出”这一事实（例如 `NOT EXECUTED`、`PLACEHOLDER`、`SYNTHETIC`）。

禁止在 caption 中隐去这一状态或用模糊表述替代。

## Rationale

实验结果真实性是学术写作底线。占位结果并非绝对禁止，但必须被明确标注，避免读者将其误读为真实执行结果。

## Check

- **LLM 语义检查**:
  - 若结果为占位/合成/推测，caption 是否含有红色大写的显式披露
  - 披露是否出现在 figure/table caption（而非仅正文一句带过）
  - 是否存在“看起来像真实结果但无披露”的风险表达
- **判定原则**: 只要不是实际执行产出，就必须在 caption 直接披露

## Examples

### Pass

```latex
\begin{figure}[t]
\centering
\fbox{\parbox{0.9\linewidth}{\centering\vspace{1.2cm}
\textbf{[EXPERIMENT PENDING --- AWAITING HPC EXECUTION]}
\vspace{1.2cm}}}
\caption{\textcolor{red}{\textbf{[FABRICATED] NOT EXECUTED.}} Placeholder
scalability trend for draft layout only; actual results pending HPC runs.}
\label{fig:scale-placeholder}
\end{figure}
```

### Fail

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figs/coming-soon.png}
\caption{Scalability analysis across model sizes.}
\label{fig:scale}
\end{figure}
% 问题：占位图/非真实结果未在 caption 中披露 fabricated 状态
```
