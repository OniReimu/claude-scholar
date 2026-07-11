---
id: FIG.RESEARCH_GAP_TEASER
slug: fig-research-gap-teaser
severity: warn
locked: false
layer: core
artifacts: [figure]
phases: [writing-background, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: manual
enforcement: doc
params: {}
conflicts_with: []
constraint_type: guidance
autofix: none
---

## Requirement

可选的 **research-gap 图**（teaser）：一张专门图示化研究 gap 的图，若使用，须满足：

1. **位置**：首页右上角，位于 system overview 图**之前**（通常是 `\begin{figure}[t]` 放在第一栏顶部或用 teaser 惯例置于顶部）；
2. **单栏**：必须单栏 `figure`，不占双栏；
3. **触发条件**：仅当 gap 用文字说不清、需要图才能让读者一眼看懂时才画。

**不是每篇论文都需要这张图。** gap 简单、一两句话能讲清的论文不应硬凑此图。

## Rationale

有些研究 gap（如"现有方法覆盖 A、B 却在 A∩B 的交叉场景全部失效"、多维权衡的空白区）用文字描述冗长且不直观，一张 gap 图能在首页就把"别人到哪、我们补哪"讲清，是强开场。但它是**可选增强**，不是义务：gap 本就清晰时强加此图反而稀释重点、挤占版面。单栏保证它作为轻量 teaser 而非喧宾夺主。与 system overview（`FIG.COLUMN_WIDTH_JUSTIFICATION`）分工：gap 图讲"为什么做"，overview 讲"怎么做"。

## Check

- **人工判断**：这张图是可选项，不做硬性 lint。评审时问三点：(1) 本文的 gap 是否复杂到文字难以说清？若否，不需要此图；(2) 若有此图，是否单栏、是否置于 overview 之前的首页顶部；(3) 图是否真的在图示 gap（对手覆盖/空白区），而非提前画了方法。

## Examples

### Pass

```latex
% 首页顶部、单栏、图示 gap（现有工作的覆盖空白）
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{gap}
  \caption{Existing defenses cover single-agent (A) and static (B) threats,
    but the multi-agent dynamic regime (A$\cap$B, shaded) is unaddressed.}
  \label{fig:gap}
\end{figure}
```

### Fail

```latex
% gap 本可一句话讲清（"缺少高效方法"），却硬画一张图，且拉成双栏
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{gap_trivial}
  \caption{Prior work is slow; we are fast.}
  \label{fig:gap}
\end{figure*}
```
