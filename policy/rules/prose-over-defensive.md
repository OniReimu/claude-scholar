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
enforcement: doc
params: {max_homes_per_caveat: 1}
conflicts_with: []
constraint_type: guidance
autofix: none
---

## Requirement

每条 scope 限定、排除、"we do not claim X" 只能有**一个 canonical home**：要么是**设计描述**（operating model / threat model / 定义 / 定理假设），要么是**Limitations 块**。同一条 caveat 在第二处出现即为违规。

此外禁止四种放置错误：

1. **认怂前置**——让步、限定、hedge 出现在段落的主题句或 run-in 小标题之后的第一句，读者还没拿到结论就先拿到代价。
2. **免责收尾**——贡献段、方法段、结果段以"做不到什么"结束。读者离开该段时手里应该是主张，不是例外。
3. **预防性辩解**——为正文从未提出过的质疑辩护，且不在 Limitations 块内。
4. **负向重述**——正面陈述已经蕴含的内容，再用否定说一遍。

## Rationale

这是 **结构性**问题，不是词级问题，所以既有的 30 条 PROSE 规则一条都抓不到。`PROSE.HEDGING_DISCIPLINE` 管的是 hedge 词是否匹配证据；这条管的是**辩护放在哪里、放了几次**。一篇稿子可以逐句通过全部行文规则，读起来仍然通篇在道歉。

过度防御通常不是原始草稿的毛病，而是**修 overclaim 时的过冲**。作者被指出"这个主张太强"，于是在每一处相关位置都补一句限定，结果同一条 caveat 出现三到四次，且每次都落在最伤读感的位置。见 [[feedback_rewrite_for_punch_upgrades_claims]] 的反方向。

代价是真实的：审稿人读到的第一印象决定基调。同样的事实，"Staged gradient isolation withholds the deletable units, so exact deletion is provisioned at a utility tax" 和 "The backbone is competitive, and the isolation tax stays within noise at the operating point" 诚实度完全相同，但前者让读者带着"有代价"进入数据，后者带着"有结论"。

**边界很重要，不要过度执行。** threat model 的边界写在 threat model 里、定理假设写在定理里、operating model 说明系统提供什么——这些是**设计**，不是辩护，必须保留。Limitations 块本身也不是违规。venue 要求的 Limitations 章节是硬性的。

**最危险的误删**：某条 caveat 可能是某条审稿意见的**唯一可见答复**。删之前必须确认它不是。这类判断无法由只看单节的检查者完成。

## Check

- **LLM 检查**（逐节隔离扫描，每节一个检查者，禁止越界）：
  1. 本节每条 scope 限定 / 排除 / 否定式主张，在全文其他位置是否已经出现？
  2. 是否有段落的主题句就是让步或限定？
  3. 是否有贡献 / 方法 / 结果段以"做不到什么"收尾？
  4. 是否在回答正文从未提出的质疑？
  5. 是否有正面句已蕴含、又用否定重说一遍的内容？
- **裁决必须由掌握全局的人做**，不能由单节检查者直接执行删除。每条建议删除的句子，先回答：**它是不是某条 reviewer comment 的唯一落点？** 是则改写不删。
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
