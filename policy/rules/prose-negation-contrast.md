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
  # Tier A (guardrail) — copular contrast, listed first and kept separate from
  # the bare comma form so the finding carries the stronger reading: this is the
  # `It's not X, it's Y` family PROSE.NEGATIVE_PARALLELISM also owns.
  - pattern: "\\b(is|are|was|were|remains?|becomes?)\\s+[\\w-]+(\\s+[\\w-]+){0,2},\\s+not\\s+\\w"
    mode: match
  - pattern: "\\bneither\\b[^.!?]{1,60}\\bnor\\b"
    mode: match
  # Reversed copular order: `is not A, but B`. The original single pattern was
  # anchored on the byte sequence `, not `, which only exists in positive-first
  # order — in the reversed form the comma sits before `but`, so the family's
  # most common negative-first member was invisible to both this card and
  # PROSE.NEGATIVE_PARALLELISM (whose patterns require a doubled pronoun or
  # `not just/only`). The lookahead hands `not just/only/merely/simply` to that
  # card. A concessive but-CLAUSE (`is not complete, but the gap is small`)
  # also matches: the regex locates, the judgment layer clears it — same
  # contract as tier B.
  - pattern: "\\b(is|are|was|were|remains?|becomes?)\\s+not\\s+(?!just\\b|only\\b|merely\\b|simply\\b)[\\w-]+(\\s+[\\w-]+){0,3},\\s+but\\s+\\w"
    mode: match
  - pattern: ",\\s+not\\s+\\w"
    mode: match
  # Tier B (guidance) — previously left to the judgment layer because legitimate
  # uses are common. Measured on one full manuscript, `rather than` alone was 17
  # of 38 instances: the largest class, entirely invisible mechanically, while
  # the card warned in prose against reflexively rewriting `, not Y` into exactly
  # this form. Tiering resolves the precision objection — a tier-B hit is
  # advisory. The regex LOCATES; the tier and the verdict are judgment.
  - pattern: "\\brather than\\b"
    mode: match
  - pattern: "\\binstead of\\b"
    mode: match
coverage_note: "tier B (rather than / instead of) is located but not judged here — the exclusion may be load-bearing. Tier A (copular contrast both orders, neither...nor) is zero-tolerance. Not enumerable: the comma-free reversed form (is not A but B) and fronted Not A, but B — check negated copulas by hand."
lint_targets: "**/*.tex"
---

## Requirement

避免用不必要的对比构造去陈述一个本可以直接正面说出的事实。**按句式分两档，强度不同**：

| 档 | 句式 | 强度 |
|---|---|---|
| **A — guardrail（零容忍）** | `X is/are A, not B` · `not A, but B` · `X is neither A nor B`（**系动词 + 表语对比**） | 强制改正面 |
| **B — guidance（偶尔可用）** | `rather than` · `instead of`（**挂在动词/动名词上**） | 排除项本身承载主张时可用 |

分档的理由：系动词型是 `PROSE.NEGATIVE_PARALLELISM` 的 `It is not X, it is Y` 近亲，属同一族假深刻构造；而 `rather than` 挂在动词上时表达的往往是方法学取舍（`we mark X as unavailable rather than guessing its timing`），那是另一回事。

⚠️ **反向护栏：单纯的否定谓语不在禁令内。** 同句内没有正面对项的否定——`that difference is not an effect estimate` / `visible events are not yet evidence for action`——是正常陈述，不是对比构造。实测一份全稿的 16 处命中里有 **5 处**属此类。不划这条界会把正常否定谓语全铲掉。

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

- **regex 搜索**: 五条 pattern 覆盖档 A 的系动词型与 `neither…nor`、任意逗号否定，以及档 B 的 `rather than` / `instead of`。**正则只定位，分档与判决归判断层**
- **无正面对项的否定谓语不报**：`is not` / `are not` / `does not` 后面没有跟一个被对照的正面项时，不是本卡的对象。档 A 的 pattern 已经把这类排除在外（它要求 `is <表语>, not <X>` 的完整形状），审阅时不要手工补报
- **改写优先级**: ① **正面陈述**——`only` / `alone` / 一个更准的正面形容词常常能把排除项吸收进去；② 排除项本身即主张时，降级用档 B 的 `rather than` 保留
- **删排除项前必须确认它在别处有落点**。实测的两处删除分别核实于同段末句；排除项承载的是实证主张时，删掉等于丢一个 claim
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
