---
id: LATEX.VAR.LONG_TOKEN_USE_TEXT
slug: latex-var-long-token-use-text
severity: warn
locked: false
layer: core
artifacts: [equation]
phases: [writing-system-model, writing-methods, writing-experiments, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {min_token_length: 4}
conflicts_with: []
---

## Requirement

数学模式中，变量名长度 >= `min_token_length`（默认 4）个字母时，必须使用 `\text{}` 或 `\mathrm{}` 包裹，避免被渲染为多个斜体单字母变量的乘积。

## Rationale

LaTeX 数学模式默认将连续字母视为独立变量的乘积（如 `$loss$` 渲染为 *l·o·s·s*）。长标识符（如 `total`, `train`, `pred`）必须用 `\text{}` 包裹以正确显示。短变量名（1-3 字母如 `x`, `lr`, `dim`）可保持数学斜体。

## Check

- **LLM 检查**: 在 `$...$` 或 equation 环境内，识别连续字母序列长度 >= `min_token_length`（默认 4），确认是否用 `\text{}` 或 `\mathrm{}` 包裹
- **常见违规**: `$L_{total}$`（应为 `$L_{\text{total}}$`）、`$loss$`（应为 `$\text{loss}$`）

## Examples

### Pass

```latex
$L_{\text{total}} = L_{\text{cls}} + \lambda L_{\text{reg}}$

\begin{equation}
\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
\end{equation}
```

### Pass（短变量名无需包裹）

```latex
$x_i$, $y$, $\theta$, $lr$, $dim$
```

### Fail

```latex
$L_{total}$           % total 有 5 个字母，需要 \text{}
$accuracy = 0.95$     % accuracy 有 8 个字母
$loss_{train}$        % loss 和 train 都 >= 4 个字母
```
