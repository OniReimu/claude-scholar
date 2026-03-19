---
id: PROSE.HEDGING_DISCIPLINE
slug: prose-hedging-discipline
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
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

Hedging 词（may, might, could, possibly, potentially）仅用于以下场景：

1. **真正不确定的推测**: 尚无实验证据的假设
2. **泛化声明**: 超出实验范围的推论
3. **未来方向**: 可能的后续工作

**已有实验证据支撑的结论禁止 hedge。** 实验结果用确定性语言陈述。

## Rationale

过度 hedging 削弱论文说服力，让读者质疑作者对自己结果的信心。但完全不 hedge 又可能导致过度 claim。关键是匹配 hedging 程度和证据强度。

## Check

- **LLM 检查**:
  1. Results section 中是否对有数据支撑的结论使用了 may/might/could
  2. Conclusion 中对已完成工作的总结是否过度 hedge
  3. Method section 中对设计选择是否不必要地用了 potentially/possibly

## Examples

### Pass

```latex
% Results - 有数据，不 hedge
The proposed method outperforms all baselines by at least 5.2\%.

% Discussion - 推测，合理 hedge
This improvement may stem from the regularization term, which could
prevent catastrophic forgetting in non-target classes.
```

### Fail

```latex
% Results - 有数据却 hedge
The proposed method may potentially outperform the baselines.
Our results could possibly suggest an improvement of 5.2\%.
```
