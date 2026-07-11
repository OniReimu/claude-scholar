---
id: FIG.COLUMN_WIDTH_JUSTIFICATION
slug: fig-column-width-justification
severity: warn
locked: false
layer: core
artifacts: [figure]
phases: [writing-system-model, writing-experiments, self-review, revision, camera-ready]
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

图默认**单栏优先**，system overview 图尤其如此。上全宽（`figure*` / 双栏）之前必须自问：单栏是否放得下且清晰？

- 若单栏能清晰容纳，用双栏即为**扣分项**。
- 全宽只有在信息密度高到单栏无法承载（组件多、数据流复杂、必须横向展开）时才成立，作者须能说出这个密度理由。

这是 `TABLE.DIMENSION_BUDGET`（表格默认单栏、先剪再上 `table*`）在图上的对应物。

## Rationale

双栏是稀缺版面。system overview 初稿常反射性地上全宽求"气派"，但单栏能放下时全宽只会摊薄信息密度、浪费页数、逼审稿人扫过大片留白。先证明密度、再要宽度，比排版阶段硬撑更有说服力。注意与（已弃用的）`FIG.SYSTEM_OVERVIEW_ASPECT_RATIO_GE_2TO1` 正交：那条讲"若为宽图，宽高比 ≥ 2:1"，本条讲"是否该做宽图"——单栏图同样可以是 2:1 的宽扁比例，宽高比不是上双栏的理由。

## Check

- **LLM 检查**：对每个 `figure*`/全宽图，核对其信息密度是否真的需要 `\textwidth`。判断信号：system overview/pipeline/architecture 图用 `figure*` 但组件稀疏、大片留白、单栏明显放得下。命中即建议降为单栏 `figure`，除非作者给出显式密度理由。

## Examples

### Pass

```latex
% 单栏放得下的 system overview，就用单栏
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{overview}
  \caption{System overview.}\label{fig:overview}
\end{figure}

% 组件多、数据流复杂，单栏无法承载 → 全宽有密度理由
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{pipeline_dense}
  \caption{End-to-end pipeline across five interacting modules.}\label{fig:pipe}
\end{figure*}
```

### Fail

```latex
% 稀疏的 overview 硬上双栏、大片留白，单栏本可放下 → 扣分
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{overview_sparse} % 三个方框一条箭头
  \caption{System overview.}\label{fig:overview}
\end{figure*}
```
