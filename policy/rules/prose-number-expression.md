---
id: PROSE.NUMBER_EXPRESSION
slug: prose-number-expression
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

数字表达遵循以下规则：

1. **句首**: 一律拼写（"Thirty participants were recruited"）
2. **≤ 10**: 拼写（"three baselines", "five datasets"）
3. **> 10**: 阿拉伯数字（"15 epochs", "128 dimensions"）
4. **单位前**: 一律阿拉伯数字（"5 GB", "3 hours", "2 GPUs"）
5. **百分比**: 阿拉伯数字 + `\%`（"15.3\%"）
6. **范围**: 用 en-dash（"10--20 epochs"）

## Rationale

数字表达的一致性是学术出版的基本规范。大多数 IEEE/ACM 期刊和会议遵循类似的数字拼写规则。

## Check

- **LLM 检查**:
  1. 句首是否有未拼写的阿拉伯数字
  2. ≤ 10 的数字是否用了阿拉伯数字（单位前除外）
  3. 单位前是否拼写了数字（如 "five GB"）

## Examples

### Pass

```latex
We evaluate on three benchmark datasets with 15 different configurations.
Each experiment runs for 5 hours on 2 NVIDIA A100 GPUs.
Twenty participants were recruited for the user study.
```

### Fail

```latex
We evaluate on 3 benchmark datasets with fifteen different configurations.
5 participants were recruited. Each run takes five hours on two GPUs.
```
