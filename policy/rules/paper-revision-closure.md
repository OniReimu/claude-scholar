---
id: PAPER.REVISION_CLOSURE
slug: paper-revision-closure
severity: warn
locked: false
layer: core
artifacts: [text]
phases: [self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: []
constraint_type: guidance
autofix: none
---

## Requirement

self-review 完整跑完之后，必须对**全稿**输出一条判决，四档取其一。判决是 self-review 的最后一件事，不是可选的收尾语。

- **STOP_REVISING** — 不存在足以重启修订的实质根因。允许仍有 findings：非根本性的问题进方向性建议，或进投稿准备轴（页数、匿名化、模板合规），**它们不构成再开一轮改写的理由**。
- **ONE_BOUNDED_ROUND** — 存在**一个**局部实质问题。判决必须同时给出三样东西：点名该根因、限定修改范围（哪些节 / 哪些段）、限定一轮。该轮结束后重新判决；**同一根因不得第二次触发 ONE_BOUNDED_ROUND 或 REOPEN_SUBSTANTIVE_REVISION**——若一轮限定改写没能解决它，那说明它不是局部问题，要么按新证据判 REOPEN，要么承认它超出改稿能力并作为已知限制留在稿上。这是防循环条款。
- **REOPEN_SUBSTANTIVE_REVISION** — 存在中央性实质根因：claim 不成立、实验支撑有缺口、结构性论证断裂。这三类需要真正重开修订，不是改写措辞能收的。
- **UNASSESSED** — 稿件不完整（缺节、占位文本、正文截断）。此时拒绝给出整稿判决，指明缺什么即可。对半份稿子说 STOP_REVISING 是伪造判决。

**判决依据只能是实质根因。以下三类理由明文禁止：**

1. **findings 数量**——"还有 N 处违规"永远不构成 REOPEN 的理由。数量是 lint 的输出，不是稿件质量的读数；
2. **抽象完美标准**——"还能更好"对任何一份稿子都成立，因此对任何一份稿子都不提供信息；
3. **接收概率猜测**——"这样审稿人可能不买账"是在预测一个本卡观察不到的量。

**不设任何数值阈值。** 不写"≤N findings → STOP"，不写"违规密度低于 X"。分档判定完全归判断层。

**输出格式**（紧凑判决块，不复述完整 violation report）：

```
判决：<档位>
根因：<1–2 句，说清楚是什么问题、为什么它够格 / 不够格重启修订>
建议：<≤3 条方向性建议>
范围：<仅 ONE_BOUNDED_ROUND 需要：哪些节 / 段，一轮>
```

## Rationale

polish 循环此前的出口是「全部 pass？」——零 findings 才停。这是数量口径，而数量口径在本仓库自己的证据面前站不住：`PROSE.CAUSAL_CONNECTIVE` 记录了连接词单一化是**跨 pass 累积**出来的，三条标点清理规则各自把标点隐含的因果赶到词汇层，每一遍引入一两个 `therefore`，逐处判都合理，实测三遍之后 26 个正式连接词里 22 个是 `therefore`。没有任何一条单规则在任何一遍上失效，崩的是总量。

这件事的推论是：**pass 次数本身是风险变量**。每多跑一轮修补，都会把新的统计特征攒进稿子——修补是有代价的动作，而"零 findings 才停"这个口径把代价定义成了零。所以封顶不能靠给每条规则各加一道护栏（护栏是逐处的，累积是跨处的），只能在所有单条规则之上放一个"还要不要再跑一轮"的判决。closure 是比逐条规则更上游的结构性动作，这也是它不与任何单条规则争管辖、`conflicts_with` 为空的原因。

**拒绝阈值与本仓库既有原则同构。** `PROSE.SEMANTIC_IDLING` 在 Check 里明确拒绝用嵌入相似度、padding 占比、命题密度阈值代替判断，理由是那些代理量在实测中全部塌陷（组内离散度比组间差异还大）；同一条卡在 Rewrite 里拒绝设压缩率目标，理由是把观察量当指标会制造为凑比例而删内容的压力。closure 的分档若挂上 findings 计数，就是同一个错误的第三次：把判断套进一个仪器，然后测量仪器自己。更糟的是这个仪器有明确的作弊方向——降低 findings 数最省力的方式是让规则更难触发，而不是让稿子更好。

**轴线分离：修订截止 ≠ 投稿准备。** 页数超限、匿名化未清、模板不合规都是必须解决的问题，但它们不是"论证有洞"，改它们不会改变稿件的主张结构。这类问题归 Step 8 的投稿合规检查，作为待办事项与 STOP_REVISING 共存。把它们算作重开修订的根因，会让一份论证已经成立的稿子因为页边距再走一遍全稿改写——那恰恰是上一段说的、有代价的一轮。

## Check

纯判断层，没有可机械执行的部分。

判决前依次问三个问题：

1. **稿件完整吗？** 有缺节、占位文本、截断正文 → `UNASSESSED`，到此为止，不再往下判。
2. **findings 里有没有中央性实质根因？** 逐条问"这条如果不改，稿件的主张还成立吗"。答案为否的是实质根因；答案为是的（措辞、局部冗余、格式、投稿准备）不是。有多个实质根因、或单个根因牵动主张结构 → `REOPEN_SUBSTANTIVE_REVISION`。
3. **只有一个、且局部吗？** 是 → `ONE_BOUNDED_ROUND`，写清范围；否 → `STOP_REVISING`。

判 `ONE_BOUNDED_ROUND` 之前先查上一轮判决：**同一根因已经吃过一轮限定改写的，禁止再判**（防循环条款）。判决记录在 `self_review` 阶段的 note / violation report 里，重新判决时读它。

**不加 lint_patterns，不进 corpus。** 本卡的检查对象是判决过程而不是稿件文本，语料测试的每一条 case 都会被记成假 MISSED——`PROSE.SEMANTIC_IDLING` 同为 `enforcement: doc` 的判断层规则，处理方式相同。

## Examples

### Pass

```
判决：STOP_REVISING
根因：三条 claim 各有实验支撑，Section 4 的论证链在上一轮补齐后闭合，剩余 11 处
      findings 全是措辞层（连接词、量词）与投稿准备（参考文献页超出 0.3 页）。
建议：1) 提交前按 Step 8 压缩参考文献版式；2) Section 5.2 的 therefore 分布可在
      终稿再看一眼；3) 无。
```

11 处 findings 未清而判 STOP，理由是逐条问过"不改这一条，主张还成立吗"，答案全是成立。数量没有参与判决。

```
判决：ONE_BOUNDED_ROUND
根因：Section 3.2 的威胁模型允许攻击者访问梯度，但 Section 5 的防御评估假设攻击者
      只见输出——两处设定不一致，读者无法判断实验测的是哪个模型。
建议：1) 以 3.2 为准统一到 5.1 的评估描述；2) 检查 Table 3 的 caption 是否沿用旧设定。
范围：Section 3.2、5.1、Table 3 caption，一轮。
```

### Fail

```
判决：REOPEN_SUBSTANTIVE_REVISION
根因：self-review 仍报出 23 处违规，距离全部 pass 还有距离，稿子还能再打磨得更好，
      按目前的完成度提交风险偏高。
```

三条禁止理由集齐：数量（23 处）、抽象完美标准（还能更好）、接收概率猜测（风险偏高）。没有点名任何一条"不改则主张不成立"的根因，因此这不是判决，是把停止的决定推迟到下一轮。

```
判决：ONE_BOUNDED_ROUND
根因：Section 4 的论证仍然不够清楚（上一轮已限定改写过 Section 4，问题依旧）。
范围：Section 4，一轮。
```

违反防循环条款：同一根因第二次触发限定改写。正确做法是按新证据判 `REOPEN_SUBSTANTIVE_REVISION`（若确认它牵动主张结构），或判 `STOP_REVISING` 并把它写进已知限制——一轮限定改写没能解决的问题，第二轮限定改写同样解决不了。

## Conflicts

本卡不与任何单条规则争管辖。单条规则判"这一处是不是违规"，本卡判"整稿还要不要再跑一轮"，二者在不同的层上，同时成立不构成冲突。唯一的顺序约束是本卡**最后**跑：所有逐条检查产出的 findings 是它的输入。
