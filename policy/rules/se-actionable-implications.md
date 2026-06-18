---
id: SE.ACTIONABLE_IMPLICATIONS
slug: se-actionable-implications
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-conclusion, self-review]
domains: [se]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: []
constraint_type: guidance
autofix: none
---

## Requirement

Discussion/Implications 须给出**面向利益相关方的可执行建议**，按角色组织成 run-in 三段（典型：`For tool builders / For standards bodies / For practitioners`），每段一行、具体可执行，并回指对应 RQ/finding。

## Rationale

SE 论文的评审显式奖励实用 payoff——“谁该据此做什么”。空泛的“本工作有意义、未来可拓展”不构成 implication。把建议按角色拆成 run-in 三段，可在几乎不增加篇幅的前提下大幅提升 skimmability 与实践价值。**面向 empirical-SE 论文，通过 `se-*` profile 激活。**

## Check

- **LLM 语义检查**：
  - 是否按利益相关方角色组织（≥2 个角色，典型为 builders / standards / practitioners）
  - 每条建议是否具体可执行、是否回指 RQ/finding
- **通过标准**：每个角色都能读出“我据此该做的下一步”

## Examples

### Pass

```latex
\smallskip\noindent\textbf{For tool builders.} Add an identity-layer extractor that
resolves model/dataset/service handles, since classic SBOM tools collapse to zero recall
there (RQ4).
\smallskip\noindent\textbf{For standards bodies.} Add a fidelity-status relation to
ML-BOM so card-code divergence is expressible (RQ3).
\smallskip\noindent\textbf{For practitioners.} Treat card-derived ML-BOMs as biased at
source and cross-check against loaded artifacts.
```

### Fail

```latex
Our findings have important implications for the community and open up promising
directions for future research on software supply chains.
% 空泛、无角色拆分、不可执行、不回指 RQ/finding
```
