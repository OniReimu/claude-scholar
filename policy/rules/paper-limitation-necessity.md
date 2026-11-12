---
id: PAPER.LIMITATION_NECESSITY
slug: paper-limitation-necessity
severity: warn
locked: false
layer: core
artifacts: [text]
phases: [writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: [ETHICS.LIMITATIONS_SECTION_MANDATORY, PROSE.OVER_DEFENSIVE, PROSE.SELF_UNDERMINING]
constraint_type: guardrail
autofix: none
---

## Requirement

Limitations 小节（以及 Discussion 里承担同样职能的段落）里的每一条，都要先回答一个问题：**它在解释发现的边界，还是在解释作者的边界？** 只有前者是 limitation。

逐条分三类：

| 类 | 它在做什么 | 判据 | 处置 |
|---|---|---|---|
| **A. 发现的边界** | 说明某个结果在什么条件下成立、在哪里未经检验、依赖什么假设、在什么情形下失效 | 它点得出**被它限定的那个结果或主张**，读者读完会改变对那个结果的使用方式 | 保留。写平实，带锚点（数据集、规模、表号） |
| **B. 作者的边界** | 解释为什么没做更多：算力不够、时间不够、"留待 future work"、"更大规模的研究会更好" | 点不出被限定的结果，只说明工作量可以更大 | 先试着改写成 A：它背后若藏着一个真实的适用边界，就丢掉理由、只写边界（"due to compute we only test 7B models" → "Results are measured on 7B–13B models; behaviour at larger scale is untested."）。改写不出 A 就删，或在 Conclusion 里压成一行 future work |
| **C. 没有证据的辩护** | 用断言或别人的做法挡一个批评："one might argue …; however, X is widely used and we believe …" | 去掉辩护之后，剩下的部分是否限定了某个结果？辩护里有没有本文自己的证据？ | 剩下的是 A 就写成 A；什么都不剩就删。**例外**：它若是某条真实审稿意见唯一可见的答复，只许移、不许删（与 `PROSE.OVER_DEFENSIVE` 相同的保护） |

**留下的每一条都要写全三件事：顾虑、事实、影响。** Discussion 是论文里专门放限制、边界和潜在审稿意见的地方，这些内容放在这里，就是为了不进正文前面。所以每一条都要说清：

1. **顾虑**：读者会因为本文哪个结果或哪个设计选择提出这个问题，也就是这条为什么值得在这里讨论；
2. **事实**：本文与之相关的事实或检查；
3. **影响**：它改变了读者对哪个结果的理解或使用方式，稿内有数字就带上数字。

只有事实的一条（"the labels were checked by independent model review and by reference runs"）是半成品：读者读完会问"然后呢"。它不删，要补齐顾虑和影响；补的内容只能来自稿内已有的证据（表、附录里的数字），稿内没有就列给作者。

**提前回应潜在审稿意见不等于 C 类。** 一个认真读本文的审稿人会因本文某个设计选择提出的疑问，用本文自己的证据回答、并写出影响，是 Discussion 的正当内容，按 A 类写。C 类是**没有证据的辩护**：靠断言（"we believe it is representative"）或别人的做法（"widely used in prior work"）来挡一个批评。

**条数由真实存在的边界决定，不设下限。** 凑满几条的配额会直接诱导 B 类和 C 类。反过来，本卡也不是删减披露的许可：A 类一条都不能少，删掉一条真实的适用边界比多写一条 B 类更伤。

venue 强制要求 Limitations 小节时，小节本身保留（`ETHICS.LIMITATIONS_SECTION_MANDATORY`）。本卡只判小节里的每一条该不该存在。

## Rationale

把论文当产品发布书来写，这个直觉一半是对的：B 类和 C 类对论文有害，而且不带来任何信息。B 类告诉审稿人"作者自己也觉得做得不够"，C 类替审稿人想出了一条审稿人本来没想到的批评。两者都在回答"作者为什么没有做到更多"，而读者需要知道的是"这个结果能用到哪里为止"。

另一半不对：A 类不是论文的弱点，是论文的使用说明。一个没有写明适用范围的结果会被读者用到范围之外，出错时归咎于论文；审稿人自己发现一个未写明的边界，会把它当成作者隐瞒，到 rebuttal 阶段代价更大。所以本卡的方向是**减 B 和 C、保 A**，而不是减 limitation 本身。

只保 A 还不够。一条只列事实的 A（"labels were checked by model review"）在审稿人眼里和没写差不多：他看不出作者为什么要说这个，也看不出这对结果意味着什么。NFTBench 的 Discussion 里，作者把这类条目逐条退回，要求每条写出为什么讨论、影响是什么；改写后每条都落到了稿内一个具体数字或表上。

`PROSE.SELF_UNDERMINING` 的三步处置（是否必须讨论 → 能否换目标解释 → 能否收缩主张）决定一个**不利结果**要不要变成 limitation；本卡审的是已经写成 limitation 的**每一条**是否属于 A 类。两张卡在同一个小节上前后接力。

## Check

- **LLM 检查**：把 Limitations 小节（及 Discussion 里的同类段落）拆成条目，逐条：
  1. 先挑出只列事实的条目：它陈述本文设计、数据或检查的一个事实，却没说为什么讨论、影响是什么。这类条目不进 B/C 判定，直接按第 5 步补齐，不删。
  2. 其余条目，写出它限定的结果或主张（带表号/节号）。写不出 → 不是 A 类。
  3. 归入 A / B / C，给出处置；B 类先尝试改写成 A。
  4. C 类删除前，确认它不是某条审稿意见唯一可见的答复。
  5. 留下的每一条（含第 1 步挑出的），标出顾虑、事实、影响三件事各在哪一句；缺哪件就补哪件，补的内容指向稿内证据，稿内没有就列给作者。
- **反向检查**：Results 和 Discussion 里有没有 A 类边界**没有**出现在 Limitations 或设计描述里（结果只在某个规模、某种语言、某类数据上测过，却被写成普遍结论）？有则补，补的是边界，不是道歉。
- **结构执行**：`claim-architecture-review` 在 P1 审 Limitations 小节时逐条分类，只有 A 类条目标 `required_caveat=true`。

## Examples

### Pass

```latex
\section*{Limitations}
All results are measured on English-language benchmarks; whether the
calibration gap persists in morphologically rich languages is untested
(Table~\ref{tab:main}). The bound in Theorem~\ref{thm:main} assumes
i.i.d.\ samples and does not cover the sequential setting of
Section~\ref{sec:online}.
```

### Pass（Discussion 条目，三件事齐全）

```latex
\item The labels were proposed and reviewed by models and checked by reference
runs. Without a human reviewer, an error the reviewing models share could pass
unnoticed. Every \textsc{feasible} task has a passing reference trajectory and
every \textsc{infeasible} label rests on on-chain evidence, whereas a
\textsc{clarify} label rests on the author's judgment that a decision is open.
Removing one \textsc{clarify} template raises $S$ by at most 0.062
(Appendix~\ref{app:results}).
```

顾虑：模型出题、模型审核，共有的错误会漏掉。事实：三类标签各自受什么检查约束。影响：最弱的一类去掉一个模板，分数最多变 0.062。

### Fail

```latex
\section*{Limitations}
% B：解释作者的边界，点不出被限定的结果
Due to limited computational resources, we could not run more
experiments. We leave a more comprehensive study to future work.
% C：没有证据的辩护
One might argue that our benchmark is synthetic. However, synthetic
data is widely used in prior work and we believe it is representative.
% 只有事实：读者会问"然后呢"——为什么在这里讨论，影响是什么
The labels of both tiers were checked by independent model review and by
reference runs.
```

## Conflicts

- `ETHICS.LIMITATIONS_SECTION_MANDATORY`：那张卡要求小节存在，本卡决定小节里每一条该不该存在。小节因 venue 要求保留，条目按本卡筛。
- `PROSE.OVER_DEFENSIVE`：那张卡管一条 caveat 放在哪、出现几次；本卡管它属于哪一类。C 类的"真实审稿意见唯一答复"保护沿用那张卡。
- `PROSE.SELF_UNDERMINING`：那张卡的三步处置在前，决定不利结果要不要写成 limitation；本卡在后，审写成的每一条。
