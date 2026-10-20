---
id: PROSE.EXPLAINING_REGISTER
slug: prose-explaining-register
severity: warn
locked: true
layer: core
artifacts: [text]
phases: [writing-intro, writing-background, writing-system-model, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
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

**最高优先级。**在所有 PROSE 规则和 `writing-anti-ai` 的规则里，本卡排第一：其他规则和这六条冲突时，按这六条执行。保护的是作者、关系、条件、情态和真实对照，不是禁用句式。命题换成合规结构，句中冒号、正文分号、em-dash 和档 A 对照仍须清理，不能借保真原样放行。MSR 曾把违规形式与命题一起锁住（writing-anti-ai/references/evidence.md E12）。

正文要写成**作者在向读者解释**（解释式语域），不要写成一串结论摘要（摘要式语域）。逐句按六条原则检查：

1. **主语按句子在做的事来选，让作者出场。**ChatGPT 之前的论文在句子主语上有固定的分工，按下表选：

   | 句子在做什么 | 主语怎么写 | 例子 |
   |---|---|---|
   | 作者的决定、选择、设计取舍 | `we`，并写出理由 | `We freeze the table so that …` |
   | 本文的贡献 | `This paper` 或 `we` 都可以（style-guide §4.3） | `This paper proposes …` |
   | 流程、计算、实验操作 | 被动句（style-guide §4.2） | `The hash is computed over …` |
   | 定义、性质、状态 | 系动词 `X is Y` | `The reader is a gated projection.` |
   | 一个东西字面上真的在做的事 | 名词作主语 | `The network consists of …`、`The model predicts …` |

   **要改的只有一类：名词作主语，配的却是它做不到的动作。**模块、方法、结果、数据不能自己「揭示」「确立」「保证」「承载」「保留」什么，这些动作是作者做的，或者是作者从结果中得出的判断。改法是：作者的判断用 `we` 写出（`we find …`、`we take this to mean …`），流程改成被动句，性质改成 `X is Y`。**不要把被动句、系动词句机械地改成 `we`**，它们本来就是人写论文的特征。
2. **拆开藏着关系的名词串。**几个名词和修饰语堆在中心词前面时，用从句或表示具体关系的介词短语写清楚谁对谁做了什么：从哪到哪、在什么上训练、属于谁。拆开后关系必须**比原来更具体**。如果拆完只剩「两者有关」或「范围很广」，就等于没拆。例如 `cross-model transfer` 应拆成「把 memory 从 source model 转到 target model」，写出方向；拆成 `transfer across models` 方向就丢了。领域通用术语（`hash head`、`memory table`、`language model`）不拆。**所有格回指同理**：一段里同时出现好几个实体（比如解释器、控制器、cell）时，「它的 X」「其 X」让读者去猜所属者，要写出所属者是谁（`the latency tail of the interpreter`）。只有一个可能的所属者时保留。
3. **情态和证据一致。**哪些是可能的，在什么条件下成立，后文会给出什么，都要如实写出来，不要把每句话都写成没有条件的断言。证据不需要的地方不加 hedge，见 `PROSE.HEDGING_DISCIPLINE`。**句子之间的关系也要写出来**：连着几句只报数字，哪句是原因、哪句是例子、哪句是限定、这个数说明了什么，都留给读者自己补，这就是摘要式语域。相邻两句是原因和结果、论断和例子、结果和它的限定时，用从句或连接词把关系写出来，或者合成一句。一个结果的含义论文在同一节已经说出来了，就紧挨着结果写（`This means …`、`so …`）。只写论文已经给出、或者由相邻两句直接推出的关系，**不许补论文没有的解释**。
   - **全称和必要性论断要交代范围和依据，重述要有作用。**经验性的全称结论（某个结果对一整组都成立）要写明覆盖的是哪个集合，并指向证据；必要性论断（某件事必须如此）要交代论文已有的推导或依据。同一个这样的论断再次出现时，先判断它在这里做什么：定义、推理的一步、衔接，还是总结。只是重复强调的，合并掉，或者用明确的回指承接（见 `PROSE.RESTATEMENT_DILUTION`）。
   - **一句话带上它的关系，不要一句一个事实。**ChatGPT 之前的论文在方法和实验段落里也是在陈述事实，但一个事实常和它的条件、用途、后果或补充说明写在同一句里，用从句连起来（`which …`、`where …`、`when …`、`because …`、`so …`、`, and this …`）。逐条并列的短句把这些关系全留给读者。按下面的步骤改：
     1. 看连着的两三句：它们是不是在讲同一个对象、同一个设置或同一个步骤？
     2. 后一句和前一句是什么关系：条件、原因、结果、用途、对前一句某个名词的补充说明，还是顺序？关系要么原文写了，要么从这两句、同一段或论文别处已有的内容直接看得出来。
     3. 关系清楚，就合成一句，用能表达这个关系的从句或连接词；只是顺序关系、又各自完整的，可以不合。
     - 合并后一句一般不超过 35 词，不跨段，不把三句以上并成一句。原句的数字、引用、条件和量化都保留。
     - 段落里连着三句以上都是不带从句的短陈述句，就是这一款要改的地方，不能用「每句都合格」放过。
不改变必要条件和充分条件的关系；不补论文没有的原因。定义、算法步骤、定理和证明需要的量化原样保留。**换一个同义词不算按本款修改**；原句已经交代了范围和依据，就不改。
4. **目的用动词说。**为了达到某个目的而做某件事时，把目的写成作者的动作（`To test X, we …`），不要压缩成一个名词（`for X evaluation`）。
   - **方法和实验里的每个步骤，都问一句「这一步是为了什么」。**答案在论文里能找到（同一段、同一节、引言或方法里说过这一步的用途，或者下一句直接用到了这一步的产出），就用 `to` 加动词写在这一步上（`We record activations immediately before cross-feature mixing to estimate …`）。ChatGPT 之前的论文在实验章节里这样的目的不定式每千词约 8 个，方法章节约 12 个。
   - 论文里找不到用途的，不补；在回复里列出来，交给作者。

5. **条件和范围要写成读者能核对的条件，不要压成短语标签。**附在论断上的每一个条件、范围或量化短语（某个工作点、某种负载、某个预算、某个集合里的每一个、某几种设置），先判断它是什么，再选写法：
   - **先问删掉它，论断还成立吗？**成立，就是装饰（泛泛的「在各种设置下」「在多个维度上」，没有限定作用的「每一个」「全部」）：删掉。不成立，就是真条件，按下面选写法。
   - **介词短语就够了**：条件是一个具体取值、一个对象或一个已经命名的设置，而且只管这一个结果，读者一眼能核对。比如 `at 4 requests/s`、`with eight fields`、`on ImageNet`、`in Part~B`。
   - **要写成从句**：条件是一种状态或一件事，得有自己的主语和动词才说得清，即什么东西处于什么状态、发生了什么、超过或低于什么。比如 `when the catalog changes between requests`、`if a call misses its deadline`、`when offered load exceeds slot capacity`。把它压成名词标签（`under churn`、`under overload`、`under bursty arrivals`）会把这个状态藏起来，读者得自己去还原。
   - **要写成段首的设置句**：好几句话共用同一组设置时，在段首或这几句之前交代一次（`In the steady-arrival experiment, …`），后面的句子直接报结果，不再挂条件。设置变了，再用一个从句或一句话标一次。一节的结果大多在同一个设置下时，同理在节首交代一次。
   - **不要矫枉过正**：同一段里连着几句都用 `When` 开头，和每句都挂一个 `under` 是同一个毛病，说明这组设置应该提到前面交代一次。
   - 本条只要求把条件写清楚，**不是换掉某几个介词**。把 `under X` 换成 `with X`，不算修改。
   - **结果计数只在提出论断的地方精确给一次。**`X of Y`（`1,199 of 1,200`）和 `all` 加数字（`all 120 runs`）放在论断提出的地方：摘要、引言、一节的总结句。后面各个设置的计数，表或图里已经有的，正文写出规律（`in all settings except one`）并指向那张表；表和图里都没有的计数照常写在正文，**一个都不能丢**。`all` 加数字留给小的枚举（`all three datasets`）。
   - **全称、必要性和范围的措辞，按 ChatGPT 之前的论文在同一类句子里的习惯选。**这是习惯，不是禁令；先判断句子属于哪一类，再看措辞：
     - 实验结果对一整组都成立：多写 `all`（`all runs pass`、`in all settings`），`every` 很少用；一个都不成立写 `none` 或 `no`。
     - 方法、定义或步骤适用于每个成员：多写 `each`（`for each edge`），`every` 和 `any` 也用，但 `each` 大约是 `every` 的 3 倍。
     - 必须、需要：`requires`、`must`、`needs`、`is required` 换着用，没有哪个占多数；主语是方法、条件或性质时，常写成 `X requires Y`。连着几处都是 `must`，就是没换。
     - 范围和界限：数值容差写 `within`、`at most`、`less than` 都可以；限定在某个集合或位置时多写 `only`（`only channels of the same layer are merged`），少用 `within` 加名词作标签。
     - **护栏**：只在意思完全不变时换。`every run` 和 `all runs` 等价；`must` 不能换成 `should` 或 `can`，那会弱化必要性；`only` 会加上排他的意思，原句没有排他就不加。定理、定义和算法步骤里的量化词不动。
6. **对照只在对手真实存在时才保留。**判据见 `PROSE.NEGATION_CONTRAST` 档 B，排除项必须满足下面三条之一：
   - ① 排除的是读者原本会默认采用的具体做法；
   - ② 排除的是有人主张或在用的做法，并在原处点名、引用；
   - ③ 排除的是本文实验检验过的可能。**这一种必须保留对照，并在同一处给出结果**：同一句、紧接的一句，或者明确指向某个表或图。
   - 三条都不满足，就只留正面陈述。把一种对照句式换成另一种，不算按本条修改。

**顺序：先按六条原则改一遍，再按 anti-AI 规则改一遍，然后停手。**已测稿件整篇重跑这两步没有继续提高，不能推广到所有稿件，见 Rationale。整篇论文要多轮改，用 `writing-anti-ai-loop`：逐章改，第 2 轮起以六原则为改写主任务，硬约束和新增改动每轮复核，并核对保真，停不停看扣掉超出人写范围的那部分加分以后的分数。

章节粒度按 `writing-anti-ai-loop`「准备／章节功能与粒度」。摘要交代核心发现必需的信息，结论不复演实验；减少重复前确认同一事实仍有准确落点，不设词数或数字配额（E12）。

## Rationale

这六条来自一次对比测量：280 篇 arXiv 论文，覆盖 7 个领域，每个领域 ChatGPT 之前 20 篇、2025 年以后 20 篇，用 spaCy 做句法分析。下面每项差异在 7 个领域里方向都一致：

- **主语分工变了**（每份论文取前 3,000 词统计主语，7 个领域方向都一致）：`we` 作主语 ×0.76，被动句 ×0.75，系动词作主干 ×0.71；名词作主语配主动动词则是 ×1.39，增量集中在把结论归给事物的动词上（「揭示」「确立」「承载」「保留」一类）。按全文统计，作者作主语少了约 40%；
- 名词串多了约 22%，名词化多了约 20%，表示所属和来源的结构少了约三分之一；
- 情态和助动词少了约 1/3；「这说明」「这意味着」一类把结果和含义连起来的句子，每千词从 0.39 降到 0.23；
- 目的不定式少了约 30%；
- 范围介词明显增多：条件的总量两个时代差不多，变的是写法。ChatGPT 之前的论文更多把条件写成 `when/if` 从句，或者先用一句交代实验设置；2025 年以后的论文更多把条件压成 `under` / `across` / `within` 加名词的标签。正文里数字的密度两个时代没有差别，所以要改的是条件的写法，不是把数字挪进表格；
- 「与另一项对照」的句式翻倍（每千词 1.00 → 2.06），多出来的几乎都是给解释句配一个假想对手；
- 同一类句子的措辞（17 篇 ChatGPT 之前的论文：作者本人 8 篇、cs.CR 6 篇、获奖论文 3 篇）：实验结果全部成立的句子里，`all` 占 37%，`every` 占 2%；规则适用于每个成员的句子里，`each` 占 34%，`every` 占 12%；必须、需要的句子里，`requires` 13%、`must` 13%、`should` 11%、`needs`/`need` 18%、`required` 8%；范围和界限的句子里，`only` 占 37%，`within` 占 7%；
- 句子结构（ChatGPT 之前 60 篇 cs.CR、安全和 cs.LG 论文，按章节对比，spaCy 句法分析）：方法和实验章节的平均句长约 21–22 词，从句（关系从句、状语从句、补语从句）每千词约 47–52 个，目的不定式约 8–12 个。一篇写成一句一个事实的 2026 年投稿稿：句长约 17.5 词，从句约 24–36 个，目的不定式约 1.6–2.5 个。名词串和形容词修饰的密度两者相同，所以差别在句子怎么连，不在信息密度；
- 结果计数：ChatGPT 之前的论文把各个设置的计数放进表里，正文写 `in all cases`、`in all but one case`。`X of Y` 每千词 0.04（95% 的论文不超过 0.21），`all` 加数字 0.12（不超过 0.40）。正文数字的总密度两个时代相近（10.8 / 12.3），所以第 5 条只要求同一个计数不在正文里反复报，不是把结果挪走。

这些差异指向同一件事：AI 时代的论文把「谁、为什么、在什么条件下、相对什么」都压进了名词和限定语，读起来像结论清单。人写的论文则把推理一步步说出来。六条原则针对的就是这些被压缩掉的东西。它们首先是**为读者服务的写作要求**：作者出场能交代设计理由，情态写真能防止 overclaim，目的写成动作能立住论证。

**本卡原则上不给词表。**统计差异只用来验证，不用来改写。按词频去增删单词，是对检测器做 reward hacking，读者一眼就能看出来。第 5 条最后一款是唯一的例外。它的依据不是检测器的权重，而是同一类句子在 ChatGPT 之前的论文里怎么写：从 17 篇论文里按句子意思抽出四类句子，不按词搜索，再统计承担全称、必要性或范围意思的词（`skills/writing-anti-ai/references/evidence.md` E11）。验证时，要看分数和句法层面的变化（作者作主语、名词串、情态、目的从句）是否一起动。只有分数动、句法不动，说明只是换了词。

**实测**（一篇 2026 年 arXiv 论文 2608.17050 的 Introduction、Background 和 Methods 三节，约 2,500 词；每组由一个看不到检测器的全新子代理改写一次；用 p2g 文风分衡量，ChatGPT 之前的论文落在 82–96）：

- 原文：33；
- 只做通用润色：33；
- 只按前 5 条：39；
- 5 条 + anti-AI：56；
- **6 条 + anti-AI：71**；
- 5 条 + anti-AI 再改两轮：54 / 54，不再提高。

内容保真全部通过，只有一处例外：6 条版本把一个属于第 ③ 种情况的对比删掉了。这就是第 6 条第 ③ 种情况要写成「必须保留」的原因。详见 `skills/writing-anti-ai/references/evidence.md` E8。

## Check

- 逐句读，不能靠 regex 判定。每条原则在每一句上只问一个问题：
  1. 这个设计或步骤是谁做的？为什么？
  2. 这串名词（或这个「它的 X」）背后藏着什么关系、属于谁？拆开以后更具体了吗？
  3. 这个断言的条件和证据强度写出来了吗？它和上一句是什么关系，写出来了吗？连着几句都是不带从句的短句吗？全称或必要性论断的范围和依据清楚吗？这次重述有什么作用？我的修改改变了原命题吗？
  4. 目的写成作者的动作了吗？这一步是为了什么，论文里说过吗？说过就写在这一步上。
  5. 这个条件或范围标签，是论断成立的条件吗？是，就写成可核对的条件；不是，就删掉。这个计数，表里已经有了吗？全称、必要性和范围的措辞，是 ChatGPT 之前的论文在这一类句子里常用的写法吗？换了以后意思变了吗？
  6. 这个对手属于 ①、②、③ 哪一种？
- **先改原则这一遍，再跑 anti-AI 规则**，最后用 `policy/lint.sh` 兜底。anti-AI 规则如果要撤回原则带来的改动（把 we 改回无主句、把拆开的名词串重新堆起来、删掉有证据支持的情态、删掉目的从句、删掉第 ③ 种对照），就跳过那条规则，并在回复里写明。
- **一次调用只改一遍。**不要为了让某个分数上涨而反复改写。多轮改写走 `writing-anti-ai-loop`，它的停止判据不计超出人写范围的加分。

## Conflicts

- **`policy/style-guide.md` §4.2、§4.3**：两者一致。第 1 条的主语分工表就是按这两节和测量结果写的。
- **`PROSE.ABSTRACT_AGENCY`**：第 1 条把它推广到所有「名词作主语配主动动词」的句子，并给出改成哪种主语的规则。
- **`PROSE.COPULA_DODGE`**：方向相同，性质和定义都用 `is`。
- **`PROSE.RESTATEMENT_DILUTION`**：去重时，保留写出「谁做的」的那一句。
- **`PROSE.NEGATION_CONTRAST`**：档 B 的判据就是第 6 条。档 A 照旧零容忍。
- **`PROSE.INTENSIFIERS_ELIMINATION`**：表示证据强度的程度词（如 `substantially`，附近有数字或结果）按第 3 条处理，不当空洞强调词删除。

## Examples

### Pass

```latex
% 1 主语分工：决定用 we，流程用被动，性质用 is，字面动作用名词
We freeze the memory table so that any gain on the target must come from the reader.
The hash is computed over canonicalized $n$-grams.
The reader is a gated projection of the retrieved vector.
The network consists of two branches.

% 2 拆开名词串，写出方向
We transfer the frozen memory from the source model to the target model.

% 3 情态与证据一致
Under a shared tokenizer, this scheme suffices. Section~\ref{sec:canon} shows what breaks without it.

% 4 目的写成动作
To test whether the table is portable, we remove the source backbone.

% 5 计数在论断处给一次，后面写规律并指向表
Recovery succeeds in 1,199 of 1,200 attacked configurations. ... It succeeds in all settings except one (Table~\ref{tab:main}).

% 6③ 对照保留，结果就在同一处
Swapping in a stronger reader closes most of the gap, while changing the source backbone does not (Table~\ref{tab:qa}).
```

### Fail

```latex
% 1 名词主语配了它做不到的动作
The framework ensures portability by design.            % → We design X so that …
The results reveal that the reader is the bottleneck.   % → We find that … / From Table~\ref{…}, the reader is …
% 1 反向错误：把正常的被动句和系动词句机械改成 we
We compute the hash over canonicalized $n$-grams.        % 原句 "The hash is computed over …" 本来就合格
% 2 名词串藏关系，或拆完更含糊
The cross-model frozen-memory transfer evaluation ...    % → 写出谁转到谁、测什么
We evaluate transfer across models.                      % → 方向丢了
% 4 目的压成名词
For portability evaluation, the backbone is removed.     % → To test whether ..., we remove ...
% 5 计数反复报
It accepts 599 of 600 on CNNs and all 120 runs on ViT.   % → 表里已有：写规律并指向表
% 6 假想对手（没人主张、没检验）
The failures stem from annotation noise rather than reasoning errors.  % → 只留正面陈述，或给出检验
% 6 换句式不算修改
... instead of reasoning errors.                          % → 同一个对照，换了件衣服
```
