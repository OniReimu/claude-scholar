---
id: PROSE.OVER_DEFENSIVE
slug: prose-over-defensive
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-intro, writing-background, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: lint_script
params: {max_homes_per_caveat: 1}
conflicts_with: [PROSE.RHYTHM_VARIANCE, PROSE.SELF_UNDERMINING]
constraint_type: guidance
autofix: none
lint_targets: "**/*.tex"
coverage_note: "the pattern locates author-subject refusals only (We / Our X / The rule-procedure-...). Paper-specific product names (TRACT does not ...) and pronoun subjects (It does not ...) are not enumerable — check them by hand. A/B triage and the three keep classes are judgment; every hit is a candidate, not a verdict."
lint_patterns:
  # Sentence-layer locator for disclaimer-style negative predicates: an author
  # or authored-artifact subject followed by a refusal predicate. Locate-only:
  # the same shape carries A-class disclaimers (We do not select an optimal
  # policy), keep-class methodological choices (Because ..., we do not use
  # survival models) and B-class mathematical facts (The bound does not hold) —
  # the subject list cannot split those, the predicate semantics can, and that
  # is the judgment layer's call. Measured on one manuscript: 33 negative
  # constructions, 16 A-class, at 1.7 per 1000 words.
  - pattern: "\\b(?:[Ww]e|[Oo]ur (?:method|approach|analysis|framework|procedure|rule|diagnostic|comparison|bounds?)|The (?:rule|procedure|analysis|comparison|diagnostic|method|framework|bounds?))\\s+(?:do(?:es)? not|did not|cannot|neither\\b|reports? no|makes? no)"
    mode: match
---

## Requirement

每条 scope 限定、排除、"we do not claim X" 只能有**一个 canonical home**：要么是**设计描述**（operating model / threat model / 定义 / 定理假设），要么是**Limitations 块**。同一条 caveat 在第二处出现即为违规。

此外禁止四种放置错误：

1. **认怂前置**——让步、限定、hedge 出现在段落的主题句或 run-in 小标题之后的第一句，读者还没拿到结论就先拿到代价。
2. **免责收尾**——贡献段、方法段、结果段以"做不到什么"结束。读者离开该段时手里应该是主张，不是例外。
3. **预防性辩解**——为正文从未提出过的质疑辩护，且不在 Limitations 块内。
4. **负向重述**——正面陈述已经蕴含的内容，再用否定说一遍。
5. **贡献未立先谈不足（文档域）**——Abstract 与 Introduction 中，在贡献落地之前出现的 caveat / 局限 / 排除声明。读者应当先拿到"这篇做成了什么"，再拿到边界。判定只看**位置**：该句是否出现在贡献陈述（contribution 列表或等价的主张句）之前？

> 第 1–4 条是**段落域**（一个段落内部的落点），第 5 条是**文档域**（Abstract/Intro 的先后顺序）。第 5 条只判"贡献之前有没有"，**不做跨节搬迁规划**——跨节冗余聚类与 canonical home 选择归 `claim-architecture-review`（它的 P2 pass 拥有 relocation-map）。本卡在 Abstract/Intro 内部给出"移到 Limitations 或删除"的建议，由掌握全局的人裁决。

### 句子层：免责式否定谓语

以上五条全是**位置与次数**的判据——一句免责句只出现一次、落点合法时，五条全放行。但十六句这样的句子叠起来，通篇是在报备。所以本卡另管**句子形状**本身：作者或本文产物做主语 + 拒绝某项主张做谓语。

**先分诊，只有 A 类在射程内：**

| 类 | 主语 / 性质 | 例 | 处置 |
|---|---|---|---|
| **A 免责式** | 主语是作者或本文产物，谓语是**拒绝某项主张** | `We do not select an optimal policy` · `TRACT does not count X as a benefit` | 在射程内 |
| **B 事实性否定** | 定义、发现、数值结果 | `whose object cannot be recovered from reviewer text`（定义）· `neither is contained inside $[-10,10]$`（结果）· `Equalizing the available window does not restore the regular-year pattern`（发现） | 不在射程内 |

实测一份全稿：33 处否定构造里只有 **16 处**属 A 类。不划这条界会把定义句和实验发现一起铲掉。

**regex 覆盖不到的主语**：论文专属产物名（`TRACT does not ...`）与代词主语（`It does not ...`）无法枚举进 pattern——人工检查所有否定谓语的主语指代。pattern 只覆盖 `We / Our X / The rule|procedure|...` 这组可枚举形式，且**只定位不裁决**：同一形状同时承载 A 类免责、保留类方法取舍与 B 类数学事实（`The bound does not hold`），劈开它们靠谓语语义，那是判断层的活。

**修法一步，不是删：把「我们不做 X」翻成「我们做的是 Y」**——这是 `PROSE.SELF_UNDERMINING` 三步阶梯的第三步（收缩主张到证据实际支持的范围）落到句子层，信息一字不减。实测（12 处）：

| 免责式 | 正面式 |
|---|---|
| `It does not determine closure, workload, or policy benefit.` | `Closure, workload, and policy benefit are computed without it.` |
| `We do not select an optimal policy or predict reviewer behavior.` | `The comparison is descriptive, and policy selection remains with the venue.` |
| `The procedure neither repairs semantic outputs nor assigns fallback labels.` | `Malformed and indeterminate outputs pass through unchanged.` |
| `We report no significance tests or causal coefficients.` | `We report weighted contrasts and resampling intervals only.` |

副产品是真实的：多数正面式**信息量更大**——`policy selection remains with the venue` 说出了谁来选；`describe worst-case allocations at a fixed budget` 说出了这些界到底算什么。免责式只说了不是什么。

**三类必须保留（反向护栏）：**

1. **发现本身是否定命题**——结论就是「关联不支持因果」时，翻正面 = over-claim。例：`It does not show that the comments caused ratings to rise`；
2. **带理由的方法学取舍**——`Because the administrative freeze is part of the process being studied, we do not use survival models that assume censoring is unrelated to that process.` 这是可核查的做法陈述，翻正面要凭空补出「我们用了什么」；
3. **点名具体误读方向的边界**——当否定句是**唯一列出被排除项**的地方时，翻成 `The estimand is X` 会把名字全丢掉。

**不得机械全翻。** 十六句全部改成 `X is Y only` / `Z remains outside` 会造出新的均质化（撞 `PROSE.RHYTHM_VARIANCE`）。实测 12 处用了六种不同结构：`are computed without` / `remains with` / `pass through unchanged` / `describe X` / `applies to X alone` / `fall outside it`。**优先沿用稿件已有句式。**

⚠️ 单篇实测 1.7/千词**不足以当阈值**——没有语料基线就没有高低可言。当前只按逐条 A/B 分诊执行，不做密度判定。

## Rationale

这是 **结构性**问题，不是词级问题，所以既有的 30 条 PROSE 规则一条都抓不到。`PROSE.HEDGING_DISCIPLINE` 管的是 hedge 词是否匹配证据；这条管的是**辩护放在哪里、放了几次**。一篇稿子可以逐句通过全部行文规则，读起来仍然通篇在道歉。

过度防御通常不是原始草稿的毛病，而是**修 overclaim 时的过冲**。作者被指出"这个主张太强"，于是在每一处相关位置都补一句限定，结果同一条 caveat 出现三到四次，且每次都落在最伤读感的位置。见 [[feedback_rewrite_for_punch_upgrades_claims]] 的反方向。

代价是真实的：审稿人读到的第一印象决定基调。同样的事实，"Staged gradient isolation withholds the deletable units, so exact deletion is provisioned at a utility tax" 和 "The backbone is competitive, and the isolation tax stays within noise at the operating point" 诚实度完全相同，但前者让读者带着"有代价"进入数据，后者带着"有结论"。

**边界很重要，不要过度执行。** threat model 的边界写在 threat model 里、定理假设写在定理里、operating model 说明系统提供什么——这些是**设计**，不是辩护，必须保留。Limitations 块本身也不是违规。venue 要求的 Limitations 章节是硬性的。

**最危险的误删**：某条 caveat 可能是某条审稿意见的**唯一可见答复**。删之前必须确认它不是。这类判断无法由只看单节的检查者完成。**第 5 条（文档域）同样受这条约束**：Intro 里一句看似多余的局限，可能正是上一轮 reviewer 要求前置的声明；确认它在 Limitations 有完整落点之前，只允许"移"不允许"删"。

## Conflicts

- `PROSE.SELF_UNDERMINING` 管词级措辞与责任范围（`unfortunately` / 把局部写成普遍）；本卡管辩护的**落点与次数**，以及免责句式的**形状**（句子层与那条的三步阶梯共用第三步）。同一句可同时触发，各报一次
- `PROSE.RHYTHM_VARIANCE`：句子层修法禁止机械全翻——全部免责句统一翻成同一种正面结构，只是把报备指纹换成模板指纹
- **`claim-architecture-review`（skill，非规则）** 拥有跨节冗余聚类与 canonical home 的搬迁规划（P2 relocation-map）。本卡第 5 条只在 Abstract/Intro 内部判"贡献之前是否出现 caveat"，给建议不做全局重排；两者顺序是 `claim-architecture-review`（结构编辑）先于 `writing-anti-ai`（线编）
- `ETHICS.LIMITATIONS_SECTION_MANDATORY` 优先：把 caveat 从 Intro 移走的前提是 Limitations 已完整承担它，**不得因移动而削薄披露**

## Check

- **提取 `.tex` 正文的正确方法**：见 `policy/references/tex-prose-extraction.md`。手搓扫描器的四个典型错误（`split('%')` 在 `$95\%$` 处截断、剔数学时 `$` 奇数配对吞掉整段、逐行扫描漏掉被硬换行劈开的短语、两遍大小写策略不一致）都会产生**假的「已清零」结论**

- **LLM 检查**（逐节隔离扫描，每节一个检查者，禁止越界）：
  1. 本节每条 scope 限定 / 排除 / 否定式主张，在全文其他位置是否已经出现？
  2. 是否有段落的主题句就是让步或限定？
  3. 是否有贡献 / 方法 / 结果段以"做不到什么"收尾？
  4. 是否在回答正文从未提出的质疑？
  5. 是否有正面句已蕴含、又用否定重说一遍的内容？
- **裁决必须由掌握全局的人做**，不能由单节检查者直接执行删除。每条建议删除的句子，先回答：**它是不是某条 reviewer comment 的唯一落点？** 是则改写不删。
- **多 home caveat 的删除顺序**（实测补充）：当一条 caveat 有多个落点、且其中一个在 Limitations（合法 canonical home）时，**先确认 Limitations 里那个 home 确实存在且完整地承担了这条边界，再删其余落点**。倒过来做——先删正文那处、指望 Limitations 兜住——会在 Limitations 表述不完整时静默丢掉一条真实的 scope 声明。
  实测形态：`no exact-deletion guarantee` 同时出现在 §1 段末与 §7.3 Limitations。§1 那处是典型「免责收尾」（方法段以"做不到什么"结束，紧接 `\textbf{Contributions.}`），删它是对的——但前提是先读过 §7.3 并确认它完整。
- **排除**：threat model 边界；定义 / 定理 / 引理 / 命题的假设；operating model 对系统能力的陈述；Limitations 块内**只出现一次**的每条限制；诚实报告的负结果；统计口径（置信区间、噪声水平）。

粗筛（只定位候选，不作判定）：

```bash
# 同一 caveat 的多处落点
rg -n "we (do not|don't) claim|no .* guarantee|is not reliably|out of scope|remains open|covers only|only for" sections/

# 段落主题句里的让步（run-in 小标题后紧跟限定词）
rg -n "textbf\{[^}]*\}\s*$" -A1 sections/ | rg -i "however|although|while|but |cannot|does not|only|tax|cost"
```

## Examples

Pass:

```text
The continuous-time backbone is itself competitive, and the utility tax of staged
gradient isolation stays within noise at the operating point (Appendix B).

[Limitations 块，全文唯一一处]
The guarantee has limits tied to advance provisioning and to co-located inputs.
CISU provisions the editable class in advance, as do all the design-for-deletion
methods we survey (Table 1). Data outside this class falls back to the retrospective
regime and its obstruction.
```

Fail:

```text
[小节开场即代价声明，结论被推到后面]
Staged gradient isolation withholds the deletable units from the backbone, so exact
O(1) deletion is provisioned at a utility tax, which we quantify in Appendix B.

[强结果后以「做不到什么」收尾]
...deleting the attack windows recovers the majority of that loss, 52 to 74%, at
O(1). The remainder would require clean replacements the operator does not have.

[同一 caveat 的第 2、3、4 次出现]
§1: Unisolated data remains in the retrospective regime and receives no
    exact-deletion guarantee.
§3: We therefore make no claim of exact post-hoc unlearning for an arbitrary
    pre-trained CT model.
§3: ...data reaching the shared backbone without prior isolation stays in the
    retrospective regime and carries no exact-deletion guarantee.
§7: Data outside this class falls back to the retrospective regime...

[负向重述：前三句的 non-goals 列表已经建立了完全相同的边界]
Our privacy claim is therefore limited to membership inference from the sanitized
post-deletion artifact.
```

## Conflicts

- `PROSE.SELF_UNDERMINING` 管**词级措辞与责任范围**（情绪副词、自贬词表、把局部结果升格为普遍缺陷），本卡管**结构与落点**（辩护放在哪里、放了几次）。一处文本可能同时违反两条，各报各的；走完那条的三步处置后仍要写的 limitation，落点由本卡裁决
