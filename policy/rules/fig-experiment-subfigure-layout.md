---
id: FIG.EXPERIMENT_SUBFIGURE_LAYOUT
slug: fig-experiment-subfigure-layout
severity: warn
locked: false
layer: core
artifacts: [figure]
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

实验图（数据图/结果图）的 subfigure 布局不得出现"单行单个"：任何一行都不允许只放一个 subfigure。合法布局：

1. **单行并排 2 个**（1×2）；
2. **双行及以上并排 4 个或更多**（2×2、2×3、3×2 …）。

若某个实验图找不到并排对象、只能孤零零占一行，说明它要么应与相邻图合并进同一网格，要么应补一个成对的度量图（如同一实验的另一 metric），要么降级为正文内联描述。孤图不单独成行。

## Rationale

单个 subfigure 独占一行会浪费纵向版面、割裂读者对同组实验的对照阅读，也常是"这张图其实不够重要/信息不足"的信号。成对或成网格排布强制作者把相关 metric 放在一起对照，信息密度和叙事连贯性都更高。与 `FIG.ONE_FILE_ONE_FIGURE`（1 脚本=1 图，复合布局在 LaTeX `\subfigure` 完成）互补：那条管文件粒度，本条管排布。

## Check

- **LLM 检查**：定位 Results 章节的 `figure`/`figure*` 环境，统计每个 `\subfigure`/`\subfloat` 的行内数量。任何一行只含 1 个 subfigure 即违规；总数为 2 时须并排同一行，总数 ≥4 时须排成 ≥2 行的网格。单个 `\includegraphics`（无 subfigure）的独立实验图，核对是否应与同实验的另一 metric 成对。

## Examples

### Pass

```latex
% 单行并排 2 个
\begin{figure}[t]
  \centering
  \subfigure[Accuracy vs. epochs]{\includegraphics[width=0.48\columnwidth]{acc}}
  \hfill
  \subfigure[Loss vs. epochs]{\includegraphics[width=0.48\columnwidth]{loss}}
  \caption{Training dynamics.}\label{fig:dyn}
\end{figure}

% 双行并排 4 个（2x2 网格）
\begin{figure*}[t]
  \centering
  \subfigure[]{\includegraphics[width=0.24\textwidth]{a}}
  \subfigure[]{\includegraphics[width=0.24\textwidth]{b}}
  \subfigure[]{\includegraphics[width=0.24\textwidth]{c}}
  \subfigure[]{\includegraphics[width=0.24\textwidth]{d}}
  \caption{Ablation across four settings.}\label{fig:abl}
\end{figure*}
```

### Fail

```latex
% 3 个 subfigure 排成 2 行 → 第二行只剩 1 个孤图（违规）
\begin{figure}[t]
  \subfigure[]{\includegraphics[width=0.48\columnwidth]{a}}
  \subfigure[]{\includegraphics[width=0.48\columnwidth]{b}} \\
  \subfigure[]{\includegraphics[width=0.48\columnwidth]{c}} % 孤零一行
\end{figure}
```
