---
id: EXP.EXPERIMENT_ROLE
slug: exp-experiment-role
severity: warn
locked: false
layer: core
artifacts: [text, table, figure]
phases: [writing-experiments, self-review, revision]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: [EXP.RESULTS_STATUS_DECLARATION_REQUIRED, EXP.FABRICATED_RESULTS_CAPTION_DISCLOSURE, EXP.ABLATION_IN_RESULTS, EXP.RESULTS_SUBSECTION_STRUCTURE, PROSE.RESTATEMENT_DILUTION]
constraint_type: guidance
autofix: none
---

## Requirement

论文中每一个实验——每张结果表、每张结果图、每个 Experimental Results 的 `\subsubsection`——必须能被指派到下面**四种职责之一**（封闭集合，没有第五种）：

1. **证明核心方法有效** — 主结果：在标准 setting 下，本文方法相对 baseline 达到所声明的效果。
2. **说明优势从何而来** — 机制归因：消融、组件替换、中间量测量，回答"这个 gain 是由哪一部分产生的"。
3. **在目标场景展示价值** — 部署证据：在论文声称要服务的场景（规模、数据分布、硬件、对抗强度）下方法仍然成立。
4. **排除最可能的替代解释** — 反事实控制：点名一个 reviewer 最可能提出的、与本文主张竞争的解释，并用实验消掉它。

**职责 4 必须落到具体**。它不是"补充实验"的同义词，写法是：先用一句话写出竞争解释（例如"gain 来自更长的训练预算 / 更大的模型 / 调过的超参 / 更宽松的评测协议 / 数据泄漏"），再给出使该解释不成立的对照实验。如果全文没有任何实验消掉那个解释，这是**证据缺口**，要补实验或在 Limitations 中显式承认，不能靠改措辞掩盖。

一个实验若四种职责都对不上，按下列**优先级顺序**处置：

1. **重设计** — 改成能承担某个职责的形式（例如把"超参扫描"改成"证明 gain 对超参不敏感，从而排除调参解释"，即转成职责 4）。
2. **弱化** — 降级为附录表格，正文只留一句话指过去。
3. **删除** — 最后手段。真正跑过的实验通常重新 framing 后能承担某个职责，先穷尽前两步再删。

同一个实验可以同时承担多项职责，但至少要有一项，且写作时必须明确其中的主职责。

## Rationale

页数是硬约束（`SUBMIT.PAGE_LIMIT_STRICT`）。一个不承担职责的实验消耗论文最稀缺的资源，同时稀释那些承担职责的实验：审稿人读 Experimental Results 时是在找"主张—证据"的连线，一张挂不上任何主张的表读起来就是 padding，还会让读者怀疑作者自己也不清楚哪个结果重要。

职责 4 单列，是因为作者最常跳过它。主结果、消融、场景实验都是"证明我行"，而 reviewer 拒稿的高频理由是"gain 可能来自别的原因"。主动点名并消掉最强的那个竞争解释，收益远高于再加一组 baseline。

**Agentic 角度**：agent 增加实验的默认倾向是"更完整更安全"——多一张表看上去没有代价，于是 Results 变成一份实验清单而非一条论证线。现有 EXP 规则（error bar、多次运行聚合、fabricated 披露、状态声明、消融位置、子节结构、takeaway box）全部是**格式与诚信**约束：它们保证实验被正确地呈现，但没有任何一条问"这个实验是干什么用的"。本卡补的就是这个缺口。

**layer 选 core**：判据不依赖领域或会议——任何有主张、有页数限制的论文都要求实验服务于主张。其余 EXP 规则同样是 core，保持一致。

## Check

**本卡不提供 regex 检测器**。"实验是否承担职责"是判断类问题，本仓库已实测：为判断类规则发 regex 会得到 precision 0.00（能匹配的表面特征与违规不相关），因此 `lint_patterns` 留空，只做 LLM/人工逐项判定。

- **提取 `.tex` 正文的正确方法**：见 `policy/references/tex-prose-extraction.md`。手搓扫描器的四个典型错误（`split('%')` 在 `$95\%$` 处截断、剔数学时 `$` 奇数配对吞掉整段、逐行扫描漏掉被硬换行劈开的短语、两遍大小写策略不一致）都会产生**假的「已清零」结论**

**逐实验判定表**（枚举每个 `\begin{table}` / 结果 `\begin{figure}` / Results 下的 `\subsubsection`，各填一行）：

| 实验 | 主职责（1/2/3/4） | 支撑的具体主张（原文句子或 spine 中的 claim id） | 处置 |
|------|------------------|--------------------------------------------|------|

判定规则：

1. **说出职责**——从四项封闭集合中选，不许写"提供额外证据""增强完整性"这类非职责说法；
2. **说出主张**——必须能指到论文里的一句具体主张（Abstract/Intro 的 contribution 句，或该节 takeaway），不能是"说明方法性能"这种同义反复；
3. 两者任一说不出来，进入处置阶梯：重设计 → 弱化 → 删除，并在表中记录选了哪一步及理由。

**建议基于 claim spine 执行**：若 `claim-architecture-review` 已产出 `architecture-review/spine.md`，第 2 列直接引用其中的 paper-level claim，而不是临时从正文里找句子——spine 才是论文级主张的所在地，临时找句子容易把"某段的描述句"误当作主张。若无 spine，先从 Abstract 抽出 1–3 条主张再开始填表。

**职责 4 的专项检查**：全文至少要有一个实验承担职责 4。若一行都没有，写下"reviewer 最可能提出的竞争解释是什么"，再判断是缺实验（补跑或写入 Limitations）还是缺呈现（实验已有但没有被 framing 成排除性证据）。

## Examples

### Fail

```latex
\subsection{Hyperparameter Sensitivity}
Table~\ref{tab:hparam} reports performance across learning rates
$\{1\!\times\!10^{-4}, 3\!\times\!10^{-4}, 1\!\times\!10^{-3}\}$ and batch sizes
$\{32, 64, 128\}$ for completeness.

Performance varies across the grid, with the best configuration reaching 95.8\%
and the worst 94.9\%. We use $3\!\times\!10^{-4}$ and batch size 64 in all
other experiments.

\fbox{Takeaway: We report the full hyperparameter grid for completeness.}
% 问题：占一张全宽表 + 半栏正文，但四种职责一项都不承担——
% 不是主结果、不解释 gain 来源、不是目标场景、也没排除任何竞争解释。
% takeaway 写的是"为了完整性"，这本身就是无职责的自白。
```

### Pass

```latex
% 处置阶梯第 1 步：重设计——同一批数据改成承担职责 2/4
\subsubsection{The Gain Does Not Come from Tuning}
Table~\ref{tab:hparam} sweeps learning rate and batch size for both our method
and the strongest baseline under an identical grid and an identical budget.

A reviewer may reasonably attribute our 3.2\% improvement to a better-tuned
configuration rather than to the proposed routing module. The sweep rules this
out: our worst configuration (94.9\%) still exceeds the baseline's best
(93.4\%), so the ordering is invariant to the choice of hyperparameters within
the grid. The remaining 0.9\% spread across our own configurations is an order
of magnitude smaller than the gap to the baseline.

\fbox{Takeaway: Our advantage survives the full hyperparameter grid --- the
worst tuned variant of our method still beats the best tuned baseline,
excluding tuning effort as an explanation for the gain.}
```

```latex
% 处置阶梯第 2 步：弱化——无法重设计时降为附录一句话
\subsubsection{Main Comparison}
...
We fix the learning rate to $3\!\times\!10^{-4}$ and batch size to 64
throughout; the full sweep is reported in Appendix~\ref{app:hparam}.
```

## Conflicts

- `EXP.RESULTS_STATUS_DECLARATION_REQUIRED` / `EXP.FABRICATED_RESULTS_CAPTION_DISCLOSURE` 划定本卡的**硬边界**。本卡管的是一个实验**是否配占论文的版面**，不授权因为结果不利而删实验。已经跑过、且与主张相左的实验是**证据**：为了故事干净把它拿掉是学术不端，不是编辑决策。此类结果的正确处置是照 `EXP.RESULTS_STATUS_DECLARATION_REQUIRED` 如实呈现并在正文解释，或写入 Limitations。**预注册实验与 reviewer 要求补的实验永远保留席位**，不进处置阶梯。判据只有一个：处置理由是"这个实验不服务于任何主张"（本卡适用），还是"这个实验的数字不好看"（本卡不适用）。
- `EXP.ABLATION_IN_RESULTS` 管消融**放在哪一节**，本卡管这个消融**是否该存在**。先用本卡确认它承担职责 2，再按该卡放进 Experimental Results。
- `EXP.RESULTS_SUBSECTION_STRUCTURE` 要求每个结果子节引用图表、≥2 段、以 takeaway 收尾。那是**呈现格式**；本卡在其之前生效——一个不承担职责的子节，补齐三项格式只是把 padding 包装得更整齐，应先走处置阶梯。若处置结果是"弱化"，该子节整体消失，格式要求自然不再适用。
- `PROSE.RESTATEMENT_DILUTION` 管命题层的**文字**复述，本卡管**实验层**的冗余。两者的成本来源相同（页数），但对象不同：删掉一句复述与删掉一张无职责的表可以同时发生。
