---
id: PROSE.RULE_OF_THREE
slug: prose-rule-of-three
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-background, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_style
enforcement: doc
params: {max_per_paragraph: 1}
conflicts_with: []
constraint_type: guidance
autofix: none
---

## Requirement

避免在同一段落中反复使用三项并列修辞（"X, Y, and Z"）。每段最多出现 1 次三项并列。如果需要列举三个以上概念，使用 enumerate 环境或拆成多个句子。

## Rationale

三项并列（Rule of Three）是 LLM 生成文本的强烈信号。人类写作中偶尔使用三项并列是自然的，但同一段落反复出现则不自然。

## Check

- **LLM 风格检查**: 扫描每个段落，计算 "A, B, and C" 或 "A, B, or C" 模式的出现次数
- **阈值**: 每段 ≤ 1 次
- **排除**: enumerate/itemize 环境内的列表项

## Examples

### Pass

```latex
Our method improves accuracy, latency, and scalability.
We achieve this through a novel attention mechanism that
reduces memory consumption while maintaining model quality.
```

### Fail

```latex
Our method improves accuracy, latency, and scalability.
It handles diverse, heterogeneous, and noisy data sources.
The framework is robust, efficient, and generalizable.
```
