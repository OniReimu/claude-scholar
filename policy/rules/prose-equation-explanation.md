---
id: PROSE.EQUATION_EXPLANATION
slug: prose-equation-explanation
severity: warn
locked: false
layer: domain
artifacts: [text, equation]
phases: [writing-methods, writing-experiments, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_style
enforcement: doc
params: {}
conflicts_with: []
---

## Requirement

关键公式（非辅助性公式）必须遵循三步解释模式：

1. **引入左侧概念**: 在公式前用一句话解释公式要表达什么
2. **给出公式**: 公式本身
3. **解释右侧各项**: 在公式后逐项说明右侧的变量和符号含义

"where" 子句用于解释 RHS 各项时，每个符号一条。

## Rationale

公式不能孤立存在。读者需要知道公式的目的（LHS 概念）和构成（RHS 解释）。这是工程类论文的标准写法，确保可理解性和可复现性。

## Check

- **LLM 检查**:
  1. 关键公式（`equation` / `align` 环境）前是否有引导句
  2. 公式后是否有 "where" 子句或等效的符号解释
  3. 是否有公式直接出现在两个公式之间，没有任何文字解释

## Examples

### Pass

```latex
The overall loss function combines the task loss and the unlearning
regularization term:
\begin{equation}
  \mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \mathcal{L}_{\text{forget}}
\end{equation}
where $\mathcal{L}_{\text{task}}$ is the standard cross-entropy loss on
the remaining data, $\mathcal{L}_{\text{forget}}$ penalizes the model's
ability to predict the forgotten samples, and $\lambda$ controls the
trade-off between utility and forgetting.
```

### Fail

```latex
\begin{equation}
  \mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \mathcal{L}_{\text{forget}}
\end{equation}
\begin{equation}
  \theta^* = \arg\min_\theta \mathcal{L}
\end{equation}
```
