---
id: LATEX.NOTATION_CONSISTENCY
slug: latex-notation-consistency
severity: error
locked: true
layer: core
artifacts: [equation, text]
phases: [writing-system-model, writing-methods, writing-experiments, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: []
---

## Requirement

Methods、Experiments、Discussion 中使用的所有数学符号必须与 notation table（`Table~\ref{tab:notation}`）中定义的符号一致。不得在正文中引入未在 notation table 中声明的新符号。

## Rationale

符号不一致让读者困惑。Notation table 是符号定义的单一真相源，所有后续章节引用必须匹配。

## Check

- **LLM 检查**: 正文中的数学符号是否在 notation table 中出现
- **新符号检测**: 是否有未在 notation table 中声明的新符号被引入

## Examples

### Pass

```latex
% Notation table 定义
\begin{table}
  \caption{Notation}\label{tab:notation}
  $\mathcal{D}$ & Dataset \\
  $\theta$      & Model parameters \\
\end{table}

% 正文中一致使用
We train on dataset $\mathcal{D}$ with parameters $\theta$.
```

### Fail

```latex
% Notation table 定义 $\mathcal{D}$
% 正文中交替使用不同写法
We train on dataset $D$ ...          % 应为 $\mathcal{D}$
We optimize $\mathcal{D}$ ...        % 正确
The data $\mathbf{D}$ is split ...   % 又换了写法
```
