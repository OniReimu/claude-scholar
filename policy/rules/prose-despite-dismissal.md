---
id: PROSE.DESPITE_DISMISSAL
slug: prose-despite-dismissal
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-background, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: regex
enforcement: lint_script
params: {}
conflicts_with: []
constraint_type: guardrail
autofix: none
lint_patterns:
  - pattern: "\\b[Dd]espite (these|its|their|such) (challenges|limitations|drawbacks|shortcomings)"
    mode: match
lint_targets: "**/*.tex"
---

## Requirement

避免 "Despite these challenges, ... continues to thrive/show promise" 的公式化 dismissal 模式。AI 用这个句式先承认问题再立刻否定问题的重要性，实际上跳过了对问题的真正分析。

如果存在 limitations，应该具体说明 limitation 是什么、影响有多大、有什么可能的应对方案。

## Rationale

"Despite these challenges" 是 AI 生成文本的经典结尾公式。它给读者一种"所有问题都不重要"的印象，削弱论文的严谨度。审稿人期望看到对 limitations 的真诚讨论，不是公式化的 dismissal。

## Check

- **regex 搜索**: 匹配 "Despite these/its/their challenges/limitations/drawbacks/shortcomings"
- **检查范围**: `.tex` 文件正文区域

## Examples

### Pass

```latex
The main limitation of our approach is the quadratic memory cost of
the attention mechanism, which restricts batch size to 32 on a single
A100 GPU. Sparse attention variants~\cite{sparse} could reduce this
cost, which we leave as future work.
```

### Fail

```latex
Despite these challenges, the proposed method continues to show
promising results and represents a significant step forward in the
field of federated unlearning.
```
