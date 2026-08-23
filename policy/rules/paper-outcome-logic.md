---
id: PAPER.OUTCOME_LOGIC
slug: paper-outcome-logic
severity: warn
locked: false
layer: core
artifacts: [text]
phases: [ideation, writing-background, writing-system-model, writing-methods, writing-experiments, writing-conclusion, self-review, revision]
domains: [core]
venues: [all]
check_kind: llm_style
enforcement: doc
params: {}
conflicts_with: [EXP.RESULTS_STATUS_DECLARATION_REQUIRED, EXP.FABRICATED_RESULTS_CAPTION_DISCLOSURE, PROSE.RELATED_WORK_EVOLUTION, EXP.ABLATION_IN_RESULTS]
constraint_type: guidance
autofix: none
---

## Requirement

论文写**成果逻辑**（为什么这个问题重要 → 现有方法为何不够 → 本文提供什么解法 → 证据如何支持），不写**过程流水账**（先做了什么、后尝试了什么、哪些中途放弃了）。时间顺序属于 lab notebook，不属于 manuscript。这条与 `policy/style-guide.md` §5.2 的五步叙事（Background → Problem → Gap → Method → Evaluation）是同一件事：五步叙事就是成果逻辑的骨架，本卡加的是它未言明的禁令与授权。

**禁令**——以下几种是典型的流水账泄漏：

- `we first tried X, which did not work, so we then ...`
- `initially we used A but later switched to B`
- `in an earlier version of this work / in our preliminary implementation`
- Method 节按实现历史排序（先讲最初的朴素版本，再讲怎么一步步改成最终设计）
- Results 节按实验跑的时间排序，而不是按每组实验**确立了什么**排序
- 只为交代"我们也试过"而存在的段落，不服务于任何最终主张

**授权**（本卡的另一半，其余规则都没有给）——**当结果撑不住原始叙事时，允许彻底重定义问题、重排贡献、重写结构。故事服务于核心证据，而不是忠于最初设想。** 把 framing 改成证据真正支持的样子是**正确动作**，不是让步、不是失败、不需要在文中致歉或解释改动过程。若最强的证据指向的命题与最初 outline 的命题不同，改命题，不要改证据的呈现权重去迁就 outline。

**边界**（授权的限制，先读这三条再动结构）：

1. **重排叙事 ≠ 隐藏失败实验。** 已经跑过且与主张矛盾的实验仍须报告（见 `EXP.RESULTS_STATUS_DECLARATION_REQUIRED`）；预注册的承诺、审稿人要求补的实验、决定性的负面结果，都不因重排而消失。重排改的是**顺序与 framing**，不是**报告集合**。
2. **重定义必须由证据驱动，不是由"结果好看"驱动。** 判据：重定义之后，本文的主张是否被本文的数据**更好地**支持？如果只是换了一个更容易赢的说法而证据本身没变强，那是包装，不是重构——退回原命题并如实报告其强度。
3. **消融与负面结果不在禁令范围内。** 解释"优势来自哪里"的 ablation、划定方法失效边界的负面结果，是成果逻辑的组成部分（见 `EXP.ABLATION_IN_RESULTS`），不得以"这是过程"为由删除。区分标准：该结果是否支撑或限定了一个最终主张——支撑则留，只交代作者行踪则删。

## Rationale

Agent 在这条上失败得比人更系统。三个原因：**对初始 outline 的忠诚**——outline 是被交付的指令，重写 framing 感觉像抗命；**过程顺序伪装成诚实**——把"我们试了 A 再试 B"写进去像是在坦白，实际上是把作者的搜索路径当成读者的阅读路径；**已写 outline 的 sunk cost**——重排意味着已生成的段落作废，模型倾向于保留已有产出。

对论文的代价是双份的。按时间排序的 Method 节把"重建最终设计"这件事推给读者：读者要读完三个废弃版本才知道你到底提出了什么，而审稿人往往在读完之前就已经形成印象。而证据撑不住的叙事，是本可以通过 reframing 避免的 rejection 中最常见的一种——论文在为一个自己的数据不支持的主张辩护，审稿人只需指出这一点，全篇的技术工作就一起陪葬。

## Check

- **提取 `.tex` 正文的正确方法**：见 `policy/references/tex-prose-extraction.md`。手搓扫描器的四个典型错误（`split('%')` 在 `$95\%$` 处截断、剔数学时 `$` 奇数配对吞掉整段、逐行扫描漏掉被硬换行劈开的短语、两遍大小写策略不一致）都会产生**假的「已清零」结论**

**无 `lint_patterns`（有意为之）**：时间顺序是段落级判断，不是表面形式。本仓库已测得 judgement-class 规则的 regex 精度为 0.00——`first`/`then`/`initially` 在正常技术写作中的合法用法远多于违规用法。本卡只做 LLM 判定。

**检查一（禁令侧，逐节执行）**：

1. 问：如果这项工作以另一种顺序完成，本节的段落顺序会改变吗？会改变 = 本节按过程组织。
2. 把段落重排成最终逻辑顺序（问题 → 设计 → 为何这样设计 → 证据），重读：若信息零损失，原顺序就是流水账，采用重排稿。
3. 逐段问：删掉这段，哪个**最终主张**失去支撑？答不上来且该段只交代"我们试过什么"，删。

**检查二（授权侧，全文一次）**：

1. 列出 Abstract 的核心主张；
2. 独立地列出本文**最强证据**实际支持的命题（不看 Abstract，只看表和图）；
3. 二者不一致时判断：Abstract 匹配的是最初的计划还是现有的证据？匹配计划 = 违规，改 Abstract / Contribution / 节序去匹配证据，并按上文边界 1 保留所有已跑实验的报告。

## Examples

### Pass

```latex
\subsection{Adaptive Shard Selection}
Unlearning cost is dominated by shard retraining, so the design goal is to
minimize the number of shards touched per request. We therefore select shards
by request locality rather than uniformly at random. Uniform assignment
spreads a single request across $O(k)$ shards; locality-aware assignment
confines $93\%$ of requests to a single shard (Table~\ref{tab:locality}).
The ablation in Table~\ref{tab:ablation} isolates this effect: removing
locality-aware assignment raises per-request cost by $4.1\times$, confirming
that the speedup comes from shard confinement and not from the cheaper
gradient estimator.
```

### Fail

```latex
\subsection{Adaptive Shard Selection}
We first implemented uniform random shard assignment, following the original
SISA design. This did not work well, so we then tried a hash-based variant,
which in an earlier version of this work gave inconsistent results across
seeds. After several iterations we finally switched to locality-aware
assignment, which is what we report here. We ran the locality experiments
before the ablation study, and later added the gradient-estimator comparison
when a reviewer of a previous submission asked about it.
```

## Conflicts

- `EXP.RESULTS_STATUS_DECLARATION_REQUIRED` — 本卡授权重排叙事，但**不**授权删减已跑实验的报告；状态声明义务不因重排改变。二者冲突时以该卡为准
- `EXP.FABRICATED_RESULTS_CAPTION_DISCLOSURE` — 重定义问题后若新叙事需要尚未执行的实验，必须走 fabricated 披露，不得用"叙事已重排"掩盖缺失证据
- `PROSE.RELATED_WORK_EVOLUTION` — 该卡要求 Related Work 按**领域**的 intellectual evolution 组织；本卡禁止按**本文作者**的工作时序组织。领域演进 = 成果逻辑，作者行踪 = 流水账，二者不矛盾
- `EXP.ABLATION_IN_RESULTS` — 消融是成果逻辑的一部分，本卡的禁令不得用来删消融
- `claim-architecture-review` skill — 分工：该 skill 在**草稿之后**审计段落归属与 claim spine，且是 propose-only；本卡陈述的是**一开始就该按什么顺序写**的原则。本卡在写作期生效，该 skill 在 `architecture_review` stage 收尾核查
