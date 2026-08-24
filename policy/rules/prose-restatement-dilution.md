---
id: PROSE.RESTATEMENT_DILUTION
slug: prose-restatement-dilution
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-background, writing-system-model, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: [EXP.EXPERIMENT_ROLE, PAPER.CONCLUSION_SINGLE_PARAGRAPH, PROSE.ELEGANT_VARIATION, PROSE.RULE_OF_THREE]
constraint_type: guardrail
autofix: none
---

## Requirement

一个命题在同一节内只陈述一次，放在证据最强的位置。禁止：

- 段首 topic sentence 与段末句是同一命题的两种说法；
- subsection 收尾句把该 subsection 首句换词重说；
- Abstract / Introduction / Conclusion 之外出现"迷你摘要"段落。

重复出现时保留信息量最大的那一处（通常是紧挨证据的那一处），删掉其余，不做合并改写。

## Rationale

模型生成的文本平均把每件事说 1.5 遍：先抽象陈述一次，给出证据，再用不同措辞总结一次。单看每一句都是合格的学术句子，问题只在把段落作为整体读时才浮现——读者被迫读两遍才发现第二遍没有新信息。

这条在顶会语境下有直接成本。页数是硬约束（`SUBMIT.PAGE_LIMIT_STRICT`），复述占掉的行本可以用来放一个消融或一句 threat-to-validity。审稿人对"内容稀释"很敏感，`the paper could be substantially shortened` 是常见的低分理由。

复述还会掩盖论证的薄弱处：同一个主张说两遍会产生"已经论证过"的错觉，而实际上证据只有一份。

## Check

- **提取 `.tex` 正文的正确方法**：见 `policy/references/tex-prose-extraction.md`。手搓扫描器的四个典型错误（`split('%')` 在 `$95\%$` 处截断、剔数学时 `$` 奇数配对吞掉整段、逐行扫描漏掉被硬换行劈开的短语、两遍大小写策略不一致）都会产生**假的「已清零」结论**

**删除测试**（逐段执行）：

1. 列出本段每句陈述的命题；
2. 找出命题相同、措辞不同的句对；
3. 删掉后出现的那一句，重读本段——信息零损失即确认违规，删除定稿。

**节级检查**：读 subsection 的首句和末句，若二者可互换而不影响该节，删末句。

**检查范围**：`.tex` 正文。Abstract 与 Conclusion 对全文主张的复述是**结构性要求**，不计入本卡。

**边界**：本卡管段内与节内。跨节的重复主张（同一 claim 在 Introduction、Method、Discussion 各出现一次）属结构问题，交 `claim-architecture-review` 处理，不在本卡范围。

## Examples

### Pass

```latex
Folding removes the per-step R1CS extraction, which accounts for 92.9\% of
single-step cost. The folded prover therefore runs 14.6$\times$ faster at
$n=64$, and the speedup grows linearly with the number of steps.
```

### Fail

```latex
Folding provides a substantial efficiency benefit by removing redundant work.
Specifically, it removes the per-step R1CS extraction, which accounts for
92.9\% of single-step cost, yielding a 14.6$\times$ speedup at $n=64$.
Overall, these results demonstrate that folding substantially improves
efficiency by eliminating redundant per-step work.
```

## Conflicts

- `PROSE.RULE_OF_THREE` 拥有**列举层**的同一集合重复（一组对象被列举两次）。本卡管命题层，那条管列举层；同一处可能两条都成立，修法以那条的「命名 + 引用」为准，不重复计数
- `PROSE.ELEGANT_VARIATION` 管**术语层**换词（同一概念用不同名字），本卡管**命题层**复述（同一主张用不同句子）。一段文字可能同时犯两条
- `PAPER.CONCLUSION_SINGLE_PARAGRAPH` 允许 Conclusion 复述全文主张，本卡不适用于 Abstract 与 Conclusion
- `PROSE.PARAGRAPH_TOPIC_SENTENCE` 要求段首为 topic sentence；topic sentence 与段内展开不是复述，只有段**末**的同义回归才是
- `PROSE.SUBSECTION_COMPLETENESS` 要求 subsection ≥2 段——删复述后不足 2 段的，补内容而不是把复述留着凑数
- `EXP.EXPERIMENT_ROLE` 管**实验层**的冗余（一张不承担职责的表），本卡管**命题层**的文字复述。成本同源（页数），对象不同，可以同时发生
