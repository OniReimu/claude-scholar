---
id: LATEX.EQ.DISPLAY_STYLE
slug: latex-eq-display-style
severity: error
locked: true
layer: core
artifacts: [equation]
phases: [writing-background, writing-system-model, writing-methods, writing-experiments, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: regex
enforcement: doc
params: {}
conflicts_with: []
lint_patterns:
  - pattern: "\\$\\$"
    mode: match
  - pattern: "(?<!\\\\)\\\\\\["
    mode: match
lint_targets: "**/*.tex"
---

## Requirement

Display 公式统一使用 `\begin{equation}...\end{equation}`。禁止使用 `$$...$$` 或 `\[...\]` 作为 display 公式写法。Inline 公式可以使用 `$...$`。

## Rationale

`equation` 环境自动编号，便于交叉引用（`\eqref{}`）。`$$` 是 Plain TeX 遗留语法，在 LaTeX 中会导致间距不一致。`\[...\]` 虽然是 LaTeX 原生语法但不支持自动编号。统一使用 `equation` 环境确保编号一致性和可引用性。

## Check

- **regex pattern**: `\$\$[^$]+\$\$` 检测 display math `$$`
- **regex pattern**: `\\\[[\s\S]*?\\\]` 检测 `\[...\]`
- **LLM 检查**: 确认所有 display 数学公式使用 `\begin{equation}` 或其变体（`equation*`, `align`, `align*`）

> 注：`equation*`、`align`、`align*` 等 amsmath 环境也可接受。核心约束是禁止 `$$` 和 `\[...\]`。

## Examples

### Pass

```latex
The loss function is defined as:
\begin{equation}
\mathcal{L} = -\sum_{i=1}^{N} y_i \log(\hat{y}_i)
\label{eq:loss}
\end{equation}
```

### Pass（多行公式）

```latex
\begin{align}
\mathcal{L}_{\text{total}} &= \mathcal{L}_{\text{cls}} + \lambda \mathcal{L}_{\text{reg}} \\
&= -\sum y_i \log \hat{y}_i + \lambda \|\theta\|^2
\label{eq:total-loss}
\end{align}
```

### Fail

```latex
The loss function is:
$$\mathcal{L} = -\sum_{i=1}^{N} y_i \log(\hat{y}_i)$$
```

### Fail

```latex
The loss function is:
\[\mathcal{L} = -\sum_{i=1}^{N} y_i \log(\hat{y}_i)\]
```
