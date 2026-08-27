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
conflicts_with: [PROSE.CAUSAL_CONNECTIVE, PROSE.RULE_OF_THREE, PROSE.SEMICOLON_RESTRICTION, PROSE.TRAILING_AFTERTHOUGHT]
constraint_type: guardrail
autofix: none
lint_patterns:
  # Each segment excludes the comma as well as sentence punctuation. The comma
  # is itself a member of [^.!?], so the original form gave the engine an
  # exponential number of ways to split a comma run: a 60-comma line with no
  # sentence punctuation — an ordinary concatenated bibliography or table row —
  # did not finish in 15 minutes. Excluding the comma makes each split unique.
  - pattern: "[^.!?,]*,[^.!?,]*,[^.!?,]*,[^.!?,]*,[^.!?]*[.!?]"
    mode: match
lint_targets: "**/*.tex"
---

## Requirement

单句逗号不超过 **3 个**（≥4 个逗号触发）。逗号过多的句子通常是 meandering 的从句堆叠，应拆成多个短句或重组。

本规则与 `PROSE.SENTENCE_LENGTH`（>35 词）互补：一个句子可能词数不超标但逗号成灾（密集插入语、并列限定），那种"分句套分句"的曲折感同样是 AI 味。

## Rationale

LLM 倾向把多个限定、插入、并列塞进一个长句，用逗号串起来，读起来像永远不收尾的从句链。Pre-GPT 工程类写作偏好短句，单句很少超过 3 个逗号。

## Check

- **regex 匹配**: 单个句子（以 `.!?` 为边界）内出现 ≥4 个逗号即标记
- **排除**: 列举型句子中由 `\item` / `enumerate` 承载的并列（应改用列表环境，由 `PROSE.COLON_LIST_OVERUSE` 管）
- **排除**: 公式环境、表格内容
- **注意**: 触发后优先**拆句**，而非简单删逗号。分号曾是本卡的首选修法，`PROSE.SEMICOLON_RESTRICTION` 生效后不再是——用分号满足本卡会立刻触发那一条

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

## Conflicts

- `PROSE.RULE_OF_THREE`：合规的四项短列表（`expand, duplicate, reorder, or rescale`）会自然带 4 个逗号并触发本卡——那是**副作用命中**。先按那条确认列表本身合规，然后改用重述或拆句来满足本卡，**不得为了降逗号数而删列表项**
- `PROSE.TRAILING_AFTERTHOUGHT`：实测同一句常同时命中——句末逗号甩片段本身也把逗号数推过阈值。**先修 `PROSE.TRAILING_AFTERTHOUGHT`**（把尾片段折回主句），逗号数往往随之落回阈值内；反过来先拆句会把甩尾留在其中一半里。两条各报一次，不合并计数
