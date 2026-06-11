---
id: PROSE.NEGATION_CONTRAST
slug: prose-negation-contrast
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
  - pattern: ",\\s+not\\s+\\w"
    mode: match
lint_targets: "**/*.tex"
---

## Requirement

避免 `X, not Y` 的逗号否定对比句式，如 `promoted by Dutch institutions, not by the people themselves`、`a property of the data, not the model`。直接陈述肯定的一方，或用 `rather than` 重组。

这是 `PROSE.NEGATIVE_PARALLELISM` 的短变体：`NEGATIVE_PARALLELISM` 抓 `It's not X, it's Y` / `not just X, but Y` 的整句排比，本规则抓 `X, not Y` 的逗号挂尾否定。两者都靠"先否定再确认"制造伪强调。

## Rationale

`X, not Y` 是 LLM 用来制造对比张力的廉价手法——通过否定一个对照项来强调主张，而非直接给出证据。Pre-LLM 学术写作偶尔用，但密集出现就是 AI 痕迹。优先用 `rather than`，或干脆只陈述肯定项。

## Check

- **regex 搜索**: 匹配 `, not ` 后接词
- **排除**: 列举/枚举中的合法否定（`A, B, not C, and D` 这类罕见，需人工判断）
- **排除**: `, not only ... but also ...`（由 `PROSE.NEGATIVE_PARALLELISM` 限频管理，每篇≤2 次）
- **改写优先级**: ① 只留肯定项；② `X rather than Y`；③ 拆成两句各自陈述
- **检查范围**: `.tex` 文件正文区域

## Examples

### Pass

```latex
The term is primarily promoted by Dutch institutions rather than by
the people themselves.
% 用 rather than，不用逗号否定对比
```

### Fail

```latex
The term is primarily promoted by Dutch institutions, not by the
people themselves.
% X, not Y 否定对比
```
