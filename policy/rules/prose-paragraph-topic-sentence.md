---
id: PROSE.PARAGRAPH_TOPIC_SENTENCE
slug: prose-paragraph-topic-sentence
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
---

## Requirement

每段的首句必须是 **topic sentence**，能独立概括本段主旨。读者只读每段首句就应能理解论文的逻辑脉络。

## Rationale

Topic sentence 是学术写作的基本功。审稿人通常先扫首句来把握论文结构。首句模糊的段落会被认为逻辑不清。

## Check

- **LLM 检查**: 提取每段首句，验证：
  1. 首句是否是一个完整的陈述（不是过渡短语片段）
  2. 首句是否概括了本段的核心观点
  3. 所有首句连读是否构成论文的逻辑骨架

## Examples

### Pass

```latex
Federated learning enables collaborative model training without
sharing raw data. Each participant trains a local model on its
private dataset and only shares model updates with the server.
The server aggregates these updates to produce a global model.
```

### Fail

```latex
However, there are also some other considerations. For example,
the privacy of the training data is important. In federated
learning, participants do not share raw data.
```
