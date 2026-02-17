---
id: PAPER.CONCLUSION_SINGLE_PARAGRAPH
slug: paper-conclusion-single-paragraph
severity: warn
locked: false
layer: core
artifacts: [text]
phases: [writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_style
enforcement: doc
params: {}
conflicts_with: []
---

## Requirement

Conclusion 部分写成一个连贯的单段落。不使用子标题、项目列表或多段落结构。内容依次覆盖：问题回顾 → 方法概述 → 关键结果 → 局限性 → 未来工作。

## Rationale

顶会论文的 Conclusion 通常篇幅有限（0.5-1 页），单段落结构更紧凑、连贯。分散的列表或多段落会打断论证流程，显得不够成熟。单段落强制作者提炼最核心的信息。

## Check

- **LLM 检查**: Conclusion section 是否为单段落（允许段落内换行但不允许空行分段）
- **要点**: 检查是否有 `\subsection`、`\paragraph`、`\begin{itemize}`、`\begin{enumerate}` 出现在 Conclusion 中

## Examples

### Pass

```latex
\section{Conclusion}

In this paper, we addressed the challenge of ... by proposing ...,
a novel approach that leverages ... Our extensive experiments on ...
demonstrate that ... achieves state-of-the-art performance, with ...
improvement over ... While our method shows limitations in ...,
we believe this work opens promising directions for future research,
particularly in ...
```

### Fail

```latex
\section{Conclusion}

In this paper, we proposed a novel method for ...

\paragraph{Key Contributions}
Our main contributions are:
\begin{itemize}
  \item We proposed ...
  \item We demonstrated ...
\end{itemize}

\paragraph{Future Work}
In the future, we plan to ...
```
