---
id: PROSE.COMMA_OVERUSE
slug: prose-comma-overuse
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-background, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: regex
enforcement: lint_script
params: {max_commas: 3}
conflicts_with: []
constraint_type: guardrail
autofix: none
lint_patterns:
  - pattern: "[^.!?]*,[^.!?]*,[^.!?]*,[^.!?]*,[^.!?]*[.!?]"
    mode: match
lint_targets: "**/*.tex"
---

## Requirement

单句逗号不超过 **3 个**（≥4 个逗号触发）。逗号过多的句子通常是 meandering 的从句堆叠，应拆成多个短句或重组。

本规则与 `PROSE.SENTENCE_LENGTH`（>35 词）互补：一个句子可能词数不超标但逗号成灾（密集插入语、并列限定），那种"分句套分句"的曲折感同样是 AI 味。

## Rationale

LLM 倾向把多个限定、插入、并列塞进一个长句，用逗号串起来，读起来像永远不收尾的从句链。Pre-GPT 工程类写作偏好短句和分号分隔的独立子句，单句很少超过 3 个逗号。

## Check

- **regex 匹配**: 单个句子（以 `.!?` 为边界）内出现 ≥4 个逗号即标记
- **排除**: 列举型句子中由 `\item` / `enumerate` 承载的并列（应改用列表环境，由 `PROSE.COLON_LIST_OVERUSE` 管）
- **排除**: 公式环境、表格内容
- **注意**: 触发后优先考虑拆句或用分号，而非简单删逗号

## Examples

### Pass

```latex
The method first selects query states. It then computes influence
scores and flags the top-ranked samples for removal.
```

### Fail

```latex
The method, which selects query states, computes influence scores,
flags the top-ranked samples, and removes them, operates in a single
pass.
% 一句话塞了 5 个逗号，从句套从句，应拆成多句
```
