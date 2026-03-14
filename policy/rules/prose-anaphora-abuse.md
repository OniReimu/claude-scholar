---
id: PROSE.ANAPHORA_ABUSE
slug: prose-anaphora-abuse
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-background, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_style
enforcement: doc
params: {max_consecutive: 2}
conflicts_with: []
---

## Requirement

避免同一句式开头连续重复 3 次或以上。AI 倾向于用重复句首来制造节奏感或强调感，在学术写作中这是不自然的。

## Rationale

排比反复（anaphora）是修辞手法，偶尔使用在演讲或文学中有效，但在学术论文中连续 3+ 次使用同一句式开头是明显的 AI 信号。人类学术写作自然会变换句式结构。

## Check

- **LLM 检查**: 扫描连续句子的开头词/短语，检测 3+ 次重复
- **常见 AI 模式**:
  - "They could... They could... They could..."
  - "This enables... This enables... This enables..."
  - "We propose... We design... We evaluate..." (Contribution list 除外)
- **排除**: `enumerate` 环境中的条目

## Examples

### Pass

```latex
The first component handles data partitioning. Next, a gradient
aggregation module computes the global update. Finally, the server
distributes the updated model to all participants.
```

### Fail

```latex
They could expose internal model states to external auditors.
They could offer fine-grained access control over training data.
They could provide cryptographic proofs of deletion.
They could create verifiable audit trails for compliance.
```
