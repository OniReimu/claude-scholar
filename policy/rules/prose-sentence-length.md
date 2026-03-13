---
id: PROSE.SENTENCE_LENGTH
slug: prose-sentence-length
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-background, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: regex
enforcement: lint_script
params: {max_words: 35}
conflicts_with: []
lint_patterns:
  - pattern: "[^.!?]*\\s[^.!?]*"
    mode: count
    threshold: 35
    threshold_param: max_words
lint_targets: "**/*.tex"
---

## Requirement

单句不超过 **35 词**。典型句长区间为 25-35 词。超过时拆成多个短句，或使用分号连接两个独立子句。

## Rationale

过长的句子降低可读性，增加理解负担。Pre-GPT 时期的论文典型句长为 25-35 词，这是工程类学术写作的自然节奏。

## Check

- **计数检查**: 统计每句词数（以句号/问号/感叹号为句子边界），超过 35 词的句子标记为违规
- **排除**: LaTeX 命令 token 不计入词数（如 `\cite{}`、`\ref{}`）
- **排除**: 公式环境内的内容

## Examples

### Pass

```latex
The proposed method formulates the unlearning problem as an optimization
task. It minimizes the influence of target samples while preserving
model utility on the remaining data.
```

### Fail

```latex
The proposed method formulates the machine unlearning problem as a
constrained optimization task that minimizes the influence of target
training samples on the model parameters while simultaneously preserving
the overall model utility and performance on the remaining non-target
training data through a carefully designed regularization term.
```
