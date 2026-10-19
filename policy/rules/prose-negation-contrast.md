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
coverage_note: "tier B (rather than / instead of) is located but not judged here — keep it only under (a) concrete default, (b) named and cited holder, or (c) tested here with the result at the same place (PROSE.EXPLAINING_REGISTER principle 6). Tier A (copular contrast both orders, neither...nor) is zero-tolerance. Not enumerable: the comma-free reversed form (is not A but B) and fronted Not A, but B — check negated copulas by hand."
lint_targets: "**/*.tex"
---

## Requirement

避免用不必要的对比构造去陈述一个本可以直接正面说出的事实。**按句式分两档，强度不同**：

| 档 | 句式 | 强度 |
|---|---|---|
| **A — guardrail（零容忍）** | `X is/are A, not B` · `not A, but B` · `X is neither A nor B`（**系动词 + 表语对比**） | 强制改正面 |
| **B — guidance（按三条判据）** | `rather than` · `instead of`（**挂在动词/动名词上**） | 排除项满足 ①②③ 之一才保留，③ 必须保留 |

分档的理由：系动词型是 `PROSE.NEGATIVE_PARALLELISM` 的 `It is not X, it is Y` 近亲，属同一族假深刻构造；而 `rather than` 挂在动词上时表达的往往是方法学取舍（`we mark X as unavailable rather than guessing its timing`），那是另一回事。

⚠️ **反向护栏：单纯的否定谓语不在禁令内。** 同句内没有正面对项的否定——`that difference is not an effect estimate` / `visible events are not yet evidence for action`——是正常陈述，不是对比构造。实测一份全稿的 16 处命中里有 **5 处**属此类。不划这条界会把正常否定谓语全铲掉。

**默认改法：直接正面陈述 `X is A`，把对比整个去掉。** 不要把 `, not Y` 反射性地换成 `rather than Y` / `instead of Y`——那只是换件衣服的同一个对比，没有解决问题。

**档 B 的判据（= `PROSE.EXPLAINING_REGISTER` 第 6 条，最高优先级）：对照只在对手真实存在时保留。**排除项必须满足下面三条之一：

| | 排除项是什么 | 怎么处理 |
|---|---|---|
| **①** | 读者原本会默认采用的具体做法（`we take the median instead of the mean`、Engram 默认的 raw token IDs） | 保留 |
| **②** | 有人主张或正在使用的做法，**在原处点名并引用**（`Unlike RETRO~\cite{…}, we …`） | 保留；没有引用就补上，补不上就按「三条都不满足」处理 |
| **③** | 本文实验检验过的可能（`the gain comes from the data, not the architecture`，有消融支撑） | **必须保留，并在同一处给出结果**：同一句、紧接的一句，或者明确指向某个表或图。不许删掉排除项，只留正面陈述，那样等于丢掉一个比较性论断 |

三条都不满足，排除项就是假想对手：删掉对照，只留正面陈述。两边都有实际内容时，也可以拆成两句，各说一边。

安全 / 威胁模型里的划界（`the adversary controls the client, not the server`）属于 ③ 的一种：划界是威胁模型的一部分，必须保留，并紧跟它的依据。

**换句式不算修改。**把 `, not Y`、`rather than Y`、`instead of Y`、`whereas`、`unlike` 互相替换，只是给同一个对照换了件衣服。

与 `PROSE.NEGATIVE_PARALLELISM` 的关系：后者抓 `It's not X, it's Y` / `not just X, but Y` 的整句排比，本规则抓 `X, not/rather than/instead of Y` 的短对比。

## Rationale

LLM 习惯用"否定一个对照项"来制造强调或深度感，即使那个对照项根本没人会误以为。

**实测**（280 篇 arXiv 论文，7 个领域，每个领域 ChatGPT 之前 20 篇、2025 年以后 20 篇；单位为每千词）：

- 「与另一项对照」的句式合计从 1.00 翻倍到 2.06；
- `rather than` 从 0.10 升到 0.75（×7.3），用到它的论文从 36% 升到 85%；
- `instead of` 反而从 0.23 降到 0.11，`on the other hand` 从 0.16 降到 0.04。

带解释和归因词（suggest、stem from、merely……）的 `rather than` 句，从每百万词 26 句升到 233 句；而带解释词的句子本身只从 12.6% 升到 14.1%。也就是说，多出来的不是解释，而是**给每个解释配上的假想对手**。人写的论文用对照，主要是说明「换掉了哪个具体的默认做法」，或者点名引用一个真实存在的对手。Pre-LLM 学术写作默认直接正面陈述，对比是稀缺的、用在刀刃上的工具。把 `, not Y` 反射性改成 `rather than Y` 不是去 AI 味，只是换一种 AI 味——作者偏好是**能正面说就正面说，对比只留给真正需要 highlight 的地方**。

## Check

- **regex 搜索**: 五条 pattern 覆盖档 A 的系动词型与 `neither…nor`、任意逗号否定，以及档 B 的 `rather than` / `instead of`。**正则只定位，分档与判决归判断层**
- **无正面对项的否定谓语不报**：`is not` / `are not` / `does not` 后面没有跟一个被对照的正面项时，不是本卡的对象。档 A 的 pattern 已经把这类排除在外（它要求 `is <表语>, not <X>` 的完整形状），审阅时不要手工补报
- **档 B 逐条判 ①②③**: 三条都不满足 → **正面陈述**（`only` / `alone` / 一个更准的正面形容词常能把排除项吸收进去）；属于 ③ → 保留，并在同一处给结果；属于 ② → 确认原处有引用
- **删排除项前必须确认它在别处有落点**。实测的两处删除分别核实于同段末句；排除项承载的是实证主张时，删掉等于丢一个 claim
- **排除**: `, not only ... but also ...`（由 `PROSE.NEGATIVE_PARALLELISM` 限频管理，每篇≤2 次）
- **检查范围**: `.tex` 文件正文区域

## Examples

### Pass

```latex
% 默认：直接正面陈述，无对比
The improvement comes from the larger training set.

% ③ 保留，结果在同一处
Our ablation shows the gain comes from the data, not the architecture (Table~\ref{tab:abl}).

% ① 具体默认做法
We take the median instead of the mean, because two runs diverge.
```

### Fail

```latex
The method is efficient, not slow.               % → The method is efficient.
The framework is modular rather than monolithic. % → The framework is modular.
The failures stem from annotation noise rather than reasoning errors.
% 假想对手：没人主张、正文也没检验 → 只留正面，或在同一处给出排除 reasoning errors 的检验
Readers drive performance.  % 原文 "readers, rather than backbone identity, drive performance" 是 ③（实验换过 backbone）
% → 删掉排除项丢了比较性论断；应保留并给结果
% 把 ", not Y" 反射性换成 "rather than Y" 只是换衣服，不是去对比
```
