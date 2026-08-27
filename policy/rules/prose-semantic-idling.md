---
id: PROSE.SEMANTIC_IDLING
slug: prose-semantic-idling
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
conflicts_with: [PROSE.RESTATEMENT_DILUTION, PROSE.ANNOUNCEMENT_SENTENCE, PROSE.FRACTAL_SUMMARY, PROSE.FILLER_PHRASES, PROSE.CAUSAL_CONNECTIVE]
constraint_type: guardrail
autofix: none
---

## Requirement

每个句子必须新增一条**可证伪的内容**。两种失败形态：

**A. 零命题（元叙述空转）** — 句子描述"我们在做分析 / 这很重要 / 这带来了理解"，却不给出任何具体的变量、数值、机制或结论。典型标志是句子可以原样搬到另一篇完全不同的论文里而不需要改一个词。

**B. 因果回环** — `because` / `since` / `which enables` 之后的解释项是被解释项的换词重述。理由和结论是同一件事。

修法只有两条：**具体化**（换成真正的变量、数字、机制）或**删除**。不要改写成更好听的空话。

## Rationale

这是最被感知为"AI 味"的写法，成因也清楚：模型被训练成"详尽=周全"，而当输入本身缺乏具体事实时，唯一能扩展篇幅的方式就是扩展语义外延——句法在前进，命题在原地。

它与复述（`PROSE.RESTATEMENT_DILUTION`）是两回事，这一点决定了它需要单独一张卡：复述是**同一个命题说了两遍**，空转是**一个命题都没有**。删除测试对复述有效（删掉第二遍，信息零损失），对空转无效——空转的句子删掉后信息同样零损失，但原因不是"别处说过"，而是"这里从来没说过"。诊断不同，修法也不同：复述删后一句即可，空转必须补事实，否则删完这一段就空了，而那说明这一段本来就不该独立存在（转 `claim-architecture-review`）。

成本与复述相同——顶会页数是硬约束，空转句占掉的行本可以放一个消融——但审稿人的反应更差：复述让人觉得啰嗦，空转让人怀疑作者没有东西可写。

## Check

- **提取 `.tex` 正文的正确方法**：见 `policy/references/tex-prose-extraction.md`。

**命题提取测试**（逐句执行，这是本卡唯一的检查方法）：

1. 用一句话写出这句断言了什么，**必须包含至少一个具体对象**（变量名、数值、模块名、数据集、机制）；
2. 写不出来 → 形态 A；
3. 若句中含因果连接词，分别写出解释项与被解释项。二者是同一命题的两种措辞 → 形态 B。

**不要用指标代替判断。** 不做嵌入余弦相似度、不做 padding-token 占比、不做命题密度阈值。这些代理量在本仓库的实测中全部塌陷：结构性重复信号自评 16x，换 fresh 模型盲评后降到 1.22x，组内离散度（2.11–6.81）比组间差异还大。凡是"给判断套一个阈值"的仪器，测的都是仪器自己。

**豁免**（这些不是空转）：
- **定义句与形式化陈述**——"A deletable unit is a designated time segment." 命题是定义本身；
- **必要的铺垫**：一句抽象陈述紧跟着在同段内被具体兑现的，属于 topic sentence（`PROSE.PARAGRAPH_TOPIC_SENTENCE`），不是空转。判据是**兑现距离**：下一句就落地 → 放行；隔了三句还在抽象层 → 违规；
- **领域惯例的套式**：Ethics / Threats to Validity / Broader Impact 段落的规定动作，以及 `\paragraph{}` 标题句；
- **直接引语、reviewer comment 原文、被批评的对象文本**。

**不豁免**：Future Work / Conclusion 的套话**不在豁免之列**。`paving the way for progressive improvements in subsequent research endeavors` 与任何一篇论文的未来工作段可以互换，正是形态 A——未来工作必须点名一个具体的限制或一个具体的下一步配置。（Ethics / Threats to Validity 之所以豁免，是因为那里的规定动作本身就是被要求的内容；Future Work 没有这个豁免依据。）

**报告已放行项。** 只报违规会让作者分不清"查过合格"与"没看到"。每段至少报出该段被检查的句数与放行数。

## Boundary

本卡管**句内**：这一句有没有命题。

**整段不推进**是另一回事，归 `claim-architecture-review` **P1 逐段审计**——判"这段该不该独立存在"需要看相邻段落各自承载什么信息，那是 `merge` / `delete` 的 verdict 集合，本卡的 verdict 只有"具体化 / 删句"。

分流规则：一段之内**多数句子**都判为形态 A，不要逐句报，**整段转 `claim-architecture-review` 跑 P1**（若 `architecture-review/spine.md` 不存在，先跑 P0）。这不是把问题推走——逐句"具体化"一个本身没有内容的段落，产出的是更好听的空话。

## Limitations

**形态 B 会在"标准机制"上过判。** 盲评中一段 OOD robustness 文本被判 `escalate`，理由行称 `because the representations remain invariant across distribution shifts` 是回环。这一条不成立——invariance → OOD reliability 是真实的因果机制，不是同一命题的换词。该段整体判 `escalate` 仍然正确（另外三句确实零具体对象），但**形态 B 的判据在"机制是标准的、因而听起来像同义"的地方会失准**。

判 B 之前先问：解释项是否引入了一个**独立可测的量**？引入了就不是回环，哪怕这个因果关系在领域内是常识。未被测量的机制主张是含糊（归 `PROSE.VAGUE_QUANTIFIERS` / 证据强度问题），不是空转。

## Examples

### Pass

```latex
The kernel spills to global memory at sequence length 4096, where the working
set exceeds the 228\,KB shared-memory budget of an H100 SM. Throughput drops
by 38\% at that point and stays flat beyond it.
```

第二句的 `because` 关系由第一句的具体机制承担，不是重述。

### Fail

```latex
To provide a comprehensive understanding of the underlying dynamics, we
carefully examine the various factors that influence the overall behavior of
the system, thereby gaining valuable insights. The model achieves low latency
because the execution time is reduced, which effectively enables faster
processing.
```

第一句形态 A（30 词，零具体对象，可原样搬到任何论文）；第二句形态 B（low latency = reduced execution time = faster processing，三种措辞一个命题）。

## Conflicts

- `PROSE.RESTATEMENT_DILUTION` 管**命题重复**（说了两遍），本卡管**命题缺失**（一遍都没有）。同一段可能两条都成立，但修法相反：那条删后一句，本卡补事实。先判本卡——若句子根本没有命题，就不存在"它和谁重复"的问题
- `PROSE.ANNOUNCEMENT_SENTENCE` 管**短**标签句（`The difficulty is structural.`），本卡管**长**的元叙述句。二者是同一病灶的两种长度，边界按句长与句型：抽象名词 + 系动词 + 抽象形容词归那条，其余归本卡
- `PROSE.FRACTAL_SUMMARY` 管**结构位置**上的预告与回顾（节首节尾），本卡不看位置，只看句子有没有命题。节首的预告句两条都会命中，按那条删
- `PROSE.FILLER_PHRASES` 管在册的固定短语（regex 可判），本卡管**不在任何列表上**的空转句。凡是能被那条的 pattern 抓到的，归那条
- `PROSE.CAUSAL_CONNECTIVE` 管 `, so` 这一个**连接词形式**，本卡形态 B 管因果关系的**内容**是否为回环。`X, so X` 两条同时成立；`X because X` 只有本卡
