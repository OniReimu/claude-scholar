---
id: PROSE.THEATRICAL_SPLIT
slug: prose-theatrical-split
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
constraint_type: guidance
autofix: none
---

## Requirement

禁止把"铺垫 + 反驳"拆成两句、并让反驳句缩成无内容的短促一击。典型违规：`One might expect X. It does not.` / `The obvious fix is Y. It fails.` / `We tried Z. Nothing.`

反驳必须与铺垫合并为一句（用 `but` / `yet` / `, and`），并由**证据**承担，而不是由停顿承担。

## Rationale

停顿后落锤是口语和演讲的节奏（书面上等同于 mic drop），不是学术书面语域。三词的 "It does not." 不携带任何信息，它的全部作用是制造戏剧感；把它并回上一句，读者失去的只有那个戏剧停顿。

反例来源值得记录：`One might expect ...` 这个反预期框架本身是好的，NDSS 投稿 UniProve 就用了 —— 但它写成一整句：*"One might expect this gap to be closed by stronger SNARK security notions, **but** knowledge soundness, witness indistinguishability, and simulation extractability each address a different problem, and none of them pins which valid witness was used."* 反驳由三个具名概念和一句实质结论承担。把同一个框架拆成两拍并配三词反驳，就滑进了口语语域。

与 `PROSE.RHETORICAL_SELF_ANSWER`（禁 "The result? X." 自问自答）同族但不同：那条是问号形式的自问自答，这条是陈述形式的"设预期—击碎"。与 `PROSE.HYPOTHETICAL_FOIL` 也相邻：那条禁虚构对照物，这条允许 "One might expect"，只要求它别拆拍。

## Check

**合并测试**：把两句用 `but` / `, and` 连成一句。

- 信息**零损失** → 原来的拆分只是在演戏 → 合并
- 合并后逻辑关系变糊 → 断句是必要的 → 保留

- **LLM 检查**:
  1. 是否存在 ≤5 词、且主语为代词或指示词的独立句，紧跟在一个"预期/常识/直觉"陈述之后
  2. 该短句是否只做否定或确认，不引入新信息
  3. 常见形态：`It does not.` / `It fails.` / `Not quite.` / `That is wrong.` / `Nothing changed.`
- **排除**: 直接回答一个显式提出的研究问题（如紧跟 RQ 后的一句结论），且该结论带具体限定词

## Examples

### Pass

```latex
One might expect extra capacity to open room, but widening the network to
$35\times$ more parameters than retain constraints never lifts $\rho_\perp$ past
the $0.2$ usable-slack threshold, and training to convergence does not lift it
either.
```

反驳由 35×、0.2 阈值、训练到收敛三项证据承担，语域是书面的。

### Fail

```latex
One might expect extra capacity to open room. It does not. Widening the network
to $35\times$ more parameters than retain constraints never lifts $\rho_\perp$
past the $0.2$ usable-slack threshold.
```

`It does not.` 合并后信息零损失，属于纯戏剧停顿。
