---
id: PROSE.SHORT_PUNCHY_FRAGMENTS
slug: prose-short-punchy-fragments
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-background, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_style
enforcement: doc
params: {}
conflicts_with: []
constraint_type: guidance
autofix: none
---

## Requirement

避免用极短句（≤5 词）或单句独立成段来制造戏剧效果。学术论文的每个段落应包含完整的论证，不用短句碎片制造节奏感。

## Rationale

极短句独立成段（"Platforms do." "This changes everything."）是 RLHF 训练推动的 AI 写作模式——追求"可读性"而牺牲信息密度。人类学术写作中，段落是论证的基本单位，不会只有一句话。

## Check

- **LLM 检查**:
  1. 是否存在 ≤5 词的独立段落（非标题、非公式）
  2. 是否存在连续多个单句段落
  3. 段落是否因追求戏剧效果而被过度拆分
- **排除**: LaTeX 命令行、公式环境、figure/table caption

## Examples

### Pass

```latex
The proposed method outperforms all baselines by at least 5.2\% in
terms of accuracy while maintaining comparable computational cost.
This improvement is consistent across all four benchmark datasets,
suggesting that the approach generalizes well beyond the training
distribution.
```

### Fail

```latex
The results speak for themselves.

A 5.2\% improvement. Across all datasets.

This changes everything.
```
