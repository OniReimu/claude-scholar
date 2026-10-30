---
id: PAPER.ARGUMENT_STRUCTURE
slug: paper-argument-structure
severity: warn
locked: false
layer: core
artifacts: [text]
phases: [writing-background, writing-system-model, writing-methods, writing-experiments, self-review, revision, camera-ready]
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

按下面六条搭论证。本卡管**论断和支撑怎么摆、理由讲什么、靠什么撑**；句子怎么向读者解释归 `PROSE.EXPLAINING_REGISTER`，章节按什么顺序讲归 `PAPER.OUTCOME_LOGIC`。结构先于句子：先按本卡定下搭法，再做句子层的改写。

1. **先论断，后理由。**一条论断和支撑它的材料放在一起时，默认先下论断，再给理由。证明和推导逐步得出结论、Introduction 的叙事顺序（`style-guide.md` §5.2 的 Background → Problem → Gap）、Method 里先讲难点或先给动机性定理再引出设计（`PAPER.METHOD_NOVELTY_WEIGHT`）、Results 里先交代设置再报结果，都不算违规，因为那些铺垫不是在支撑一条已有的论断。
2. **设计选择讲动机。**读者要知道为什么需要这个选择：它解决什么问题，不这样做会怎样。把做法换个说法再展开一遍不算理由。标准组件仍按 `PAPER.METHOD_NOVELTY_WEIGHT` 一句话加引用，只在偏离标准做法时说明理由。
3. **能用文献或背景撑的，就用它撑。**前人结果、领域共识、已知约束都是正当的支撑，论断不必全靠本文自己的材料互相支撑。每个挂具体断言的引用仍过 `CITE.CLAIM_SUPPORT_REQUIRED`，不为凑支撑加引用。
4. **支撑关系要点明。**读者要靠推断才能看出一句在支撑另一句时，用 `because`、`since`、`therefore`、`thus` 这类词或从属结构点明。关系已经一目了然的不加。`Moreover` 这类加法词不表示支撑，不算在内，仍归 `PROSE.AI_LEXICON`；选哪个因果词归 `PROSE.CAUSAL_CONNECTIVE`。连接词不能把相关写成因果。
5. **支撑尽量放近。**默认放在论断的同一段。证据的归属在别节时（结果在 Results，定理在形式化一节），回指即可，不要把整段证据复制过来，但读者在本处理解论断所需的分母、取值和限定要留下（`PROSE.EXPLAINING_REGISTER`）；重复仍归 `claim-architecture-review` 的 P2。
6. **不为完整加层。**论断已被直接支撑时，不再补一层中间论断把论证拉长。判据：删掉这一层，读者是否仍看得出论断为何成立。看得出，这一层就多余。

### 两种入口

**从头写**（`ml-paper-writing`）：六条都在起草时执行。此时内容还没定，作者的理由和要引的文献都可以写进来。

**改已有稿子**（先 `claim-architecture-review`，后 `writing-anti-ai`）：只能重排稿子里已有的内容，不能补作者没写的东西。

| 条目 | 改稿时怎么做 |
|---|---|
| 1 先论断 | 段内调序归 `writing-anti-ai`，跨段调序由 `claim-architecture-review` 提方案 |
| 2 讲动机 | 先按设计的名称搜全文。别处已写明理由：搬过来或回指。没搜到：列入 `relocation-map.md` 的「Author input needed」并写明搜过哪里，不替作者编理由 |
| 3 文献支撑 | 列入「Author input needed」，不在改写时加引用 |
| 4 点明关系 | 关系在稿内已明确或可直接推出时，保留或补上连接词（`writing-anti-ai`）；不确定是否成立就报告 |
| 5 放近 | `claim-architecture-review` 提搬动方案 |
| 6 不加层 | `claim-architecture-review` 指出，由作者决定；不自动删除中间论断 |

### 不做什么

- **不设数值门槛或占比目标。**不写「链不超过 N 层」「N% 的理由要讲动机」，也不按这些比例改到某个值。
- **不以论证分为改写目标。**按分数或特征改写就是 `writing-anti-ai` 证据 E10 说的 reward hacking。改动要能说出读者因此少推断了什么。
- **不判质量。**这六条描述的是人写和 AI 写的差异，不是好坏标准。论证链深的段落可以是对的，例外按读者是否看得清判断。

## Rationale

一篇论文可以论证完整却搭得不像人写。对 18 篇论文的论证图（人写 8 篇、原始 AI 稿 6 篇、清理后 AI 稿 4 篇）算 28 个结构特征，下列方向在两组 AI 稿上一致（中位数依次为人写、原始 AI、清理后 AI）：

| 特征 | 人写 | 原始 AI | 清理后 AI | 对应条目 |
|---|---|---|---|---|
| 支撑放在论断前面 | 8% | 15% | 16% | 1 |
| 设计理由讲动机 | 35% | 24% | 21% | 2 |
| 用文献或背景作支撑 | 28% | 18% | 14% | 3 |
| 用连接词点明支撑关系 | 50% | 38% | 30% | 4 |
| 支撑与论断在同一段 | 75% | 65% | 64% | 5 |
| 跨章节支撑 | 9% | 16% | 16% | 5 |
| 论证链深度（层） | 2.5 | 3.4 | 2.9 | 6 |

两点限定了本卡的用法。

**清理后的稿子里这些差距仍在。**论证链深度的差距从 0.9 层缩到 0.4 层，其余特征和原始 AI 稿接近，连接词更少。两组篇数不同，这不是同一批稿子的前后对比，不能证明清理有没有效果。这些差异属于结构，句子层的规则本来就不针对它们，所以在结构层处理；改稿时缺内容的部分只能由作者补，这是保真要求（`PROSE.EXPLAINING_REGISTER` 第 3 条：不许补论文没有的解释）的直接结果。

**这是观察，不是实验。**样本小，只说明两类稿子不同，没有测过照这六条改写后读者是否更容易读懂，或稿子是否更接近人写。单个特征的区分力差别很大（连接词 AUC 0.98 / 1.0，文献支撑 0.71 / 0.91，同段 0.81 / 0.72）；把 7 个过程特征合成相似度分时，只有 7 篇人写论文作参考，AUC 为 0.73 / 0.70（两个数分别对原始 AI 稿和清理后 AI 稿）。因此本卡定为 `warn` 级指引，不进 lint。

## Check

- **LLM 逐段判**（`claim-architecture-review` P1，起草后由 `paper-self-review` 抽查）：
  1. 支撑是否排在它所支撑的论断之前，且不属于第 1 条的例外；
  2. 设计选择的说明是在讲为什么需要，还是在复述做法；
  3. 论断是否全靠本文内部材料撑，而有现成的文献或背景可用；
  4. 读者是否要靠推断才能看出支撑关系；
  5. 支撑是否无必要地放在别段；
  6. 是否有删掉后不影响理解的中间论断。
- **报告方式**：指明位置、问题和依据，不报计数或比例。缺作者内容的只列出来，不代写。

## Examples

自拟例，数字仅作示意。

### Pass

```text
Fixed-rate sampling misses short bursts. Earlier audits report 12% missed events on bursty workloads [7],
and our replay finds 11% on the same traces.
```

```text
We use a sliding window because event rates drift within a session, and a global average would hide that
drift. The window keeps the last $w$ events.
```

### Fail

```text
Earlier audits report 12% missed events on bursty workloads [7]. Our replay finds 11% on the same traces.
Fixed-rate sampling misses short bursts.
```

支撑排在论断之前，读者读完才知道这些数字在证明什么。

```text
We use a sliding window of size $w$. The window moves one event at a time and keeps the last $w$ events.
```

第二句只是把做法再说一遍，读者不知道为什么需要窗口。
