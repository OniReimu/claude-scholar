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

避免用不必要的对比构造去陈述一个本可以直接正面说出的事实。三种同源句式都在范围内：

- `X, not Y`（逗号否定）
- `X rather than Y`
- `X instead of Y`

**默认改法：直接正面陈述 `X is A`，把对比整个去掉。** 不要把 `, not Y` 反射性地换成 `rather than Y` / `instead of Y`——那只是换件衣服的同一个对比，没有解决问题。

**仅当"排除 Y"本身承载信息时才保留对比**，例如：

- 反过度声称（`the gain comes from the data, not the architecture`——排除架构是一个实证主张）
- 安全 / 威胁模型里需要明确划界（`the adversary controls the client, not the server`）
- 纠正读者大概率会有的误解

判断标准：删掉 `not Y / rather than Y / instead of Y` 后，句子是否丢了**实质信息**？没丢 → 删对比，只留正面；丢了 → 保留，这才是"必要的 highlight"。

与 `PROSE.NEGATIVE_PARALLELISM` 的关系：后者抓 `It's not X, it's Y` / `not just X, but Y` 的整句排比，本规则抓 `X, not/rather than/instead of Y` 的短对比。

## Rationale

LLM 习惯用"否定一个对照项"来制造强调或深度感，即使那个对照项根本没人会误以为。Pre-LLM 学术写作默认直接正面陈述，对比是稀缺的、用在刀刃上的工具。把 `, not Y` 反射性改成 `rather than Y` 不是去 AI 味，只是换一种 AI 味——作者偏好是**能正面说就正面说，对比只留给真正需要 highlight 的地方**。

## Check

- **regex 搜索**: 匹配 `, not ` 后接词（最机械的形式）
- **LLM 补充**: `rather than` / `instead of` 是同源构造，但合法用法太多，不做硬 regex；self-review 时逐句判断「排除项是否 load-bearing」
- **改写优先级**: ① 删对比、只留正面 `X is A`；② 确认排除 Y 是实质主张时才保留，并在三种形式里选语义最自然的一种
- **排除**: `, not only ... but also ...`（由 `PROSE.NEGATIVE_PARALLELISM` 限频管理，每篇≤2 次）
- **检查范围**: `.tex` 文件正文区域

## Examples

### Pass

```latex
% 默认：直接正面陈述，无对比
The improvement comes from the larger training set.

% 必要时保留——排除"架构"是实证主张，load-bearing
Our ablation shows the gain comes from the data, not the architecture.
```

### Fail

```latex
The method is efficient, not slow.               % → The method is efficient.
The framework is modular rather than monolithic. % → The framework is modular.
% 把 ", not Y" 反射性换成 "rather than Y" 只是换衣服，不是去对比
```
