---
id: PROSE.CAUSAL_CONNECTIVE
slug: prose-causal-connective
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-intro, writing-background, writing-system-model, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: regex
enforcement: lint_script
params: {}
conflicts_with: [PROSE.COMMA_OVERUSE, PROSE.EM_DASH_RESTRICTION, PROSE.HEDGING_DISCIPLINE, PROSE.INFORMAL_VOCABULARY, PROSE.MIDSENTENCE_COLON, PROSE.RHYTHM_VARIANCE, PROSE.SEMANTIC_IDLING, PROSE.SEMICOLON_RESTRICTION]
constraint_type: guidance
autofix: none
lint_patterns:
  - pattern: ",\\s+so\\s+(?!that\\b|far\\b|as\\b|long\\b|much\\b|many\\b|called\\b)[a-z]"
    mode: match
  - pattern: "(?:^|\\.\\s+)So\\s+(?!that\\b|far\\b|as\\b|long\\b|much\\b|many\\b|what\\b|how\\b|why\\b|when\\b|where\\b|who\\b|which\\b)[a-z]"
    mode: match
lint_targets: "**/*.tex"
---

## Requirement

`X, so Y` 是口语里默认的因果连接——**正因为默认，用它的时候通常没做过选择**。学术散文有一组标记因果**类型**的连接词，`so` 把这个区别整个抹掉。

### 先说反向护栏

**目标不是清零。** Pre-GPT 语料（两组不重叠的类别，共 56 篇 51 万词）本身有 0.18–0.28/千词的 `, so`。当后果显而易见到 `therefore` 会显得端着时，`so` 就是对的语域。

**这条护栏是可执行的，不是提醒**：实测显示逐条裁决分不出年代——pre-GPT 论文里的 `so` 实例，按"能不能更精确"去判，应改率不低于当代 draft。所以判据不能是"能否改进"，只能是上面那三个子类。

**也不要机械替换。** 把一篇里 40 个 `so` 全换成 `therefore`，只是把一种指纹换成另一种，并制造出新的均质化（见 `PROSE.RHYTHM_VARIANCE`）。

**连接词单一化是跨 pass 累积出来的，上面那条护栏守不住它。** 三条标点清理规则（`PROSE.EM_DASH_RESTRICTION` / `PROSE.SEMICOLON_RESTRICTION` / `PROSE.MIDSENTENCE_COLON`）的修法都把标点隐含的关系（"所以""即""因此"）赶到词汇层，而本卡的菜单里 `therefore` 是最安全的默认——它排第一、定义最宽，学术散文里绝大多数因果都能算逻辑蕴含，于是"说不出是哪一种 = 没想清楚"这条判据在它身上失效：`therefore` 永远说得出。每一遍清理引入一两个，逐处判都合理，只在总量上崩——实测一份稿子三遍标点清理后 26 个正式连接词里 22 个是 `therefore`，每一处单看都过。

**修法不是同义替换，按优先级：**

1. **删**——因果已由上一句承担，或与 `However` / `Because` 双重标记（实测 10 处里 8 处走这条）；
2. **从属化**——`Because A, B`，顺带少一个逗号；
3. **按语义换**——只有当那个因果确实属于另一类时（经验后果换 `consequently`），不是为了凑分布硬塞 `hence`。

**复核判据一句话：逐处追问这是哪一种因果，答案是否真的全是同一种？** 是——合规，全篇 `therefore` 占多数就是对的（全为逻辑蕴含的稿子本该如此）；不是——说明当初有几处没想清楚、退回了默认词。**不设占比阈值**：lint 打印一张 `therefore / thus / hence / consequently / accordingly` 计数表，只报分布不判违规，供这个追问对着真实数字进行。

### 只有三个子类触发改写，其余一律保留

逐条判别力实测为零（见 Rationale），因此**不要**因为"换成 `hence` 会更精确"就改。只有能归入下面三类之一才动：

1. **设计选择伪装成推论**——`The adversary is adaptive, so we sample fresh randomness each round.` 这不是推论，是动机。改成 `To defend against an adaptive adversary, we sample…`
2. **因果未被证据支持**——证据只给出相关，或那只是个猜测。判定权交 `PROSE.HEDGING_DISCIPLINE`
3. **证明 / 推导步骤**——数学蕴含里 `hence` / `thus` 是本领域惯例，`so` 在此处是语域失配

**其余全部保留。** 后果显而易见的 `so`、解释性的 `so`、`so it remains to show` 这类证明惯用语——都不动。

### 归入三类之后，再走三问



**问一：这是哪一种因果？**

| 连接词 | 因果类型 |
|--------|----------|
| `therefore` | **逻辑蕴含**——从已陈述的前提必然得出 |
| `hence` | **就近承接**——从刚刚建立的那个结论继续往下推 |
| `thus` | **以此方式 / 据此**——构造性推论，"由此得到" |
| `consequently` | **经验后果**——实测、部署或运行中观察到的结果 |

说不出是哪一种，说明**这个因果关系本身没想清楚**。先想清楚，再选词——不要因为说不清就退回 `so`。

**问二：证据支持这个因果吗？**

`, so` 允许句子**断言因果而不承诺是哪一种**。当证据其实只支持相关、或那只是个设计选择时，`so` 让它读起来像推导出来的。这时不是 register 问题，是 over-claim，归 `PROSE.HEDGING_DISCIPLINE`——改法是降级动词（`is consistent with`）或把两件事拆开陈述，不是换连接词。

**问三：需要连接词吗？**（首选修法往往在这里）

两个独立子句用逗号 + `so` 粘起来，多数时候正确的修法是把因果写进**句法**而不是靠连接词：

1. **从属化**：`Because A, B` / `Since A, B` / `As A, B`
2. **关系从句**：`, which tightens the bound`——当后果是对前一句的直接展开
3. **断句 + 显式动词**：`A. This implies B.` / `It follows that B.` / `B follows from A.`

从属化同时降低逗号数（见 `PROSE.COMMA_OVERUSE`），这是本卡与那条的协同点。

## Rationale

### 稳健的部分：pre-GPT 基线

六个语料、约 120 万词，同一口径（排除 `so that` / `so far` / `so as` / `so long` / `so much` / `so many` / `so-called`；剥离注释、数学环境、引用宏）：

| 语料 | 篇数 | 词数 | `, so` /千词 | formal /千词 | 比 |
|------|------|------|-------------|-------------|-----|
| arXiv 2019–2021 · cs.LG/CR/CL | 35 | 273k | 0.18 | 2.16 | 1 : 11.8 |
| arXiv 2019–2021 · cs.SE/stat.ML/cs.DB（样本外） | 21 | 241k | 0.28 | 2.18 | 1 : 7.7 |

**pre-GPT 基线在两组完全不重叠的类别与月份窗口上复现**：`, so` 0.18–0.28/千词，formal 约 2.2/千词，比例 8–12:1。本卡的反向护栏取这个值。

### 不稳健的部分：当代漂移的幅度是领域特异的，且月窗方差很大

| 语料 | 篇数 | `, so` /千词 | formal /千词 | 比 |
|------|------|-------------|-------------|-----|
| arXiv 2026-06 · cs.LG/CR/CL | 23 | 0.80 | 1.01 | 1 : 1.3 |
| arXiv 2026-02 · cs.LG/CR/CL（同类别，另一窗口） | 20 | 0.43 | 1.19 | 1 : 2.8 |
| arXiv 2026 · cs.SE/stat.ML/cs.DB（样本外） | 23 | 0.36 | 1.95 | 1 : 5.4 |

同类别两个月窗差近 2 倍（0.43 vs 0.80），**单窗口数字不可当点估计引用**。合并后 cs.LG/CR/CL 2026 约 0.61/千词，即 pre-GPT 的约 3.4 倍；而 cs.SE/stat.ML/cs.DB 只有约 1.3 倍，正式连接词几乎没降。**漂移集中在 ML / NLP / 安全这几个方向**，不是全学科现象。

### 本卡不是 AI 检测器——这一点经实测确认

从三个来源各抽 14 条被命中的真实句子（pre-GPT / arXiv 2026 / 本地 draft），**打乱编号、隐去来源后逐条裁决**，再回连标签：

- **定位精度 42/42**：没有一条是非因果的 `so`，负向前瞻工作正常
- **逐条判别力为零**：宽判据下"应改率" pre-GPT 71% / 2026 64% / draft 93%；收紧到三个可诊断子类后 **pre-GPT 64% / 2026 43% / draft 29%——顺序反转**

也就是说，**任取一个 `, so` 实例，无论出自哪个年代，可改进的比例都差不多**。差别只在**有多少个**。本卡因此是一条**因果精度**规则，不是 AI 痕迹检测规则；与 AI 写作的关联只成立在**密度**层面。

把两层合起来才是真实收益——每千词的**可执行** finding（密度 × 严判应改率）：

| 来源 | `, so` /千词 | 严判应改率 | 可执行 finding /千词 |
|------|-------------|-----------|---------------------|
| pre-GPT | 0.18 | 64% | **0.12** |
| arXiv 2026 | 0.61 | 43% | 0.26 |
| 本地 draft 抽样 5 篇 | 3.76 | 29% | **1.09** |

本地 draft 每千词的可执行 finding 约为 pre-GPT 基线的 **9 倍**——**这个收益全部来自密度**。逐条裁决贡献为负。

**方法学限制**：裁决者与规则作者是同一个（我），盲评只隔断了来源标签，隔不断文体本身的年代线索；每源 14 条，样本小。结论方向可信，具体百分比不可当精确值用。

## Check

- **regex 定位**：`,\s+so\s+` 后接小写词，排除 `so that` / `so far` / `so as` / `so long` / `so much` / `so many` / `so called`；另抓句首 `So `（pre-GPT 语料中仅 0.01/千词，几乎总是口语残留）
- **regex 只负责定位，不负责裁决**——命中后逐处走上面的三问。与 `PROSE.NEGATION_CONTRAST` 同形态：机械层给候选，语义层给结论
- **不抓 `X, and so Y`**：与 `and so on` / `and so forth` 在正则上不可分，交语义层
- **句首 `So` + 疑问词不归本卡**（`So what changes at scale?`）：那是设问句开场，归 `PROSE.RHETORICAL_SELF_ANSWER`，正则已排除
- **`so far` 不归本卡**：它是习语性状语，归 `PROSE.INFORMAL_VOCABULARY` 的 `LEXIS` 类，正则已排除，不重复计数
- **程度副词 `so large` / `so many` 不归本卡**：那是强度词问题
- **`so that`（目的从句）完全合法**，不在范围内
- **检查范围**：`.tex` 正文；`.tex` 注释已由 lint 的 comment-blanked view 排除
- **不发 `fix_patterns`（`autofix: none`）**：四个正式连接词的选择取决于那个因果是逻辑蕴含还是经验后果，而首选修法（从属化 / 关系从句 / 断句）根本不是替换。机械替换会把语义改错，那比留着 `so` 更糟

## Conflicts

- `PROSE.HEDGING_DISCIPLINE` 拥有**因果主张与证据不匹配**的情形。本卡命中处若三问的第二问失败，判定权交那条：改法是降级动词或拆开陈述，**不是换连接词**
- `PROSE.COMMA_OVERUSE` 与本卡在**从属化**这一修法上协同：`Because A, B` 比 `A, so B` 少一个逗号且因果更明确。先按本卡判类型，再看逗号数
- `PROSE.RHYTHM_VARIANCE` 是本卡修法的约束：正式连接词必须**按语义分布**，不得把所有 `so` 统一换成同一个词——那会造出新的均质化指纹
- `PROSE.INFORMAL_VOCABULARY` 拥有 `so far`（`LEXIS` 类习语性状语）与 `so large` 式程度副词；本卡只管作因果连接词的 `so`，三者不重叠
- `PROSE.AI_LEXICON` 的句首连接词密度（`Moreover` / `Furthermore` / `Additionally` / `In addition` 全文 ≤4）是同族但不同家的问题：那条管**连接词堆砌**，本卡管**因果连接词的语域与精度**。修本卡时不得把密度转移到那一族去

## Examples

### Pass

```latex
% 逻辑蕴含：从已陈述的单调性必然得出。连接词放在新句句首，不用分号——
% 分号只是把同一个长句换个标点，见 PROSE.SEMICOLON_RESTRICTION
The map is monotone in $p$ and $q$. Therefore any downstream continuation
bound tightens under the same substitution.

% 从属化——因果写进句法，同时少一个逗号
Because the mask density is known at aggregation time, the unbiased estimator
requires no additional communication.

% 关系从句——后果是前一句的直接展开
We rescale by the inverse mask density, which restores unbiasedness without
a second pass.

% 经验后果，不是推论
The kernel spills to global memory at sequence length 4096. Consequently,
throughput drops by 38 percent.

% 反向护栏：后果显然，therefore 会显得端着，so 就是对的语域
The server never sees raw features, so it cannot invert them directly.

% 目的从句，完全合法，不在本卡范围
We normalise the embedding so that the similarity is scale-free.
```

### Fail

```latex
% 逻辑蕴含被降级成随口一提的因果
The map is monotone in $p$ and $q$, so any downstream continuation bound
tightens.

% 实为设计选择的动机，却写成推论
The adversary is adaptive, so we sample fresh randomness each round.
% → To defend against an adaptive adversary, we sample fresh randomness
%   each round.

% 因果只被相关性支持，so 让它读起来像推导出来的（交 HEDGING_DISCIPLINE）
Attention entropy drops in layer 7, so the model has learned to route.

% 句首口语残留
So the bound is tight only in the balanced regime.
```
