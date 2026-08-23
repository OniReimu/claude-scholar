---
id: PROSE.SELF_UNDERMINING
slug: prose-self-undermining
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-background, writing-system-model, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: regex
enforcement: lint_script
params: {}
conflicts_with: [PROSE.OVER_DEFENSIVE, PROSE.HEDGING_DISCIPLINE, ETHICS.LIMITATIONS_SECTION_MANDATORY, EXP.FABRICATED_RESULTS_CAPTION_DISCLOSURE, CITE.CLAIM_SUPPORT_REQUIRED, EXP.RESULTS_STATUS_DECLARATION_REQUIRED]
constraint_type: guardrail
autofix: none
lint_patterns:
  - pattern: "(?i)\\b(unfortunately|regrettably|sadly|disappointingly)\\b"
    mode: match
  - pattern: "(?i)\\badmittedly\\b"
    mode: match
  - pattern: "(?i)\\bf(?:alls?|ell|alling) short\\b(?!\\s+of\\s+(?:the\\s+)?(?:information[- ]theoretic|theoretical|optimal|minimax|Bayes|entropy|lower|upper|Cram))"
    mode: match
  - pattern: "(?i)\\b(?:lags?|lagged|lagging) behind\\b(?!\\s+by\\s+(?:one|two|three|\\d+)\\s+(?:time\\s+)?(?:step|sample|frame|cycle|round|epoch)s?\\b)"
    mode: match
  - pattern: "(?i)\\b(?:do(?:es)?|did) not outperform\\b"
    mode: match
  - pattern: "(?i)\\bfail(?:s|ed)? to (?:outperform|match|beat|surpass|exceed)\\b"
    mode: match
  - pattern: "(?i)\\bwe (?:are|were) unable to\\b"
    mode: match
  - pattern: "(?i)\\b(?:suffers?|suffered|suffering|incurs?|incurred|exhibits?) (?:from )?(?:a |an )?(?:\\w+ ){0,2}performance drop\\b"
    mode: match
  - pattern: "(?i)\\bwe (?:only )?managed to\\b"
    mode: match
  - pattern: "(?i)\\b(?:is|are|remains?)\\s+(?:still\\s+)?far\\s+from\\s+(?:being\\s+)?(?:optimal|perfect|satisfactory|ideal|competitive|mature|practical|state[- ]of[- ]the[- ]art)\\b"
    mode: match
  - pattern: "(?i)\\bleaves? much to be desired\\b"
    mode: match
  - pattern: "(?i)\\bnot competitive (?:with|against)\\b"
    mode: match
  - pattern: "(?i)\\b(?:we|our\\s+[a-z-]+)\\s+merely\\b"
    mode: match
  - pattern: "(?i)\\bmerely\\s+(?:match\\w*|achiev\\w+|attain\\w+|reach\\w+|on par)\\b"
    mode: match
  - pattern: "(遗憾的是|仅仅|本文弱于|未能超过|仍明显落后)"
    mode: match
lint_targets: "**/*.tex"
---

## Requirement

**不主动示弱，不递刀子。** 一个不利事实该怎么写，由证据决定；写成一次自我攻击，则是措辞选择。

### 完整性边界（先读这条）

> 本规则管**措辞与位置**，不管**披露**。真实的负面结果、失败模式、不利比较**必须照常报告**——删除或掩盖一个真实结果是学术不端，不在本规则的授权范围内。本规则禁止的是把一个诚实的事实写成一次自我攻击。

任何以本卡为由删掉一个真实数字、隐去一次不利比较、或抽掉 venue 要求的 Limitations 内容的改动，都是误用。发现这种改动，回滚。

### 1. 自我削弱词表（闭集，regex 判定）

`unfortunately` · `regrettably` · `sadly` · `disappointingly` · `admittedly` · `merely`（`we merely` / `our X merely` / `merely matches|achieves|attains|reaches`）· `falls short` · `lags behind` · `does not outperform` · `fails to outperform|match|beat|surpass|exceed` · `we are unable to` · `we managed to` · `suffers/incurs a performance drop` · `is far from optimal|competitive|practical|…` · `leaves much to be desired` · `not competitive with` · `遗憾的是` · `仅仅` · `本文弱于` · `未能超过` · `仍明显落后`

**判据**：删掉这个词，命题是否不变？不变则删——`Unfortunately, our recall is 3.1 points lower` 与 `Our recall is 3.1 points lower` 披露完全相同的事实，前者额外附赠了一个作者情绪。情绪不是数据。

**开集部分交 LLM 层**（词表抓不到，见 Check）：`significantly worse` 这类串**主语在谁**决定它是自贬还是自证（"our method is significantly worse" vs "the baseline is significantly worse than ours" 完全相反），regex 判不了；`significantly` 本身归 `PROSE.INTENSIFIERS_ELIMINATION`，本卡不重复收词。

### 2. 责任范围自查（每句两问）

对每一句涉及不利事实的话问：

1. **这句话是否扩大了我要承担的责任？** 写下的每一个属性都要在 rebuttal 里守住。承诺越宽，可攻击面越大。
2. **是否把局部描述成普遍缺陷？** 一个局部结果（"on dataset D our recall is 3 points lower"）不得写成一般性质（"our method is weaker at recall"）。前者是一个可核对的观测，后者是一条作者自己签发的、覆盖全部数据集与全部设定的负面通用主张——审稿人会照后者引用。

局部→普遍的升格几乎总是无意的：agent 在"概括一下"时把 dataset 名、metric 名、operating point 全部丢掉，剩下一句光秃秃的能力判决。**不利事实的锚点（数据集 / 指标 / 幅度 / 表号）比有利事实的锚点更不能省。**

### 3. 不利结果的三步处置（必须按序）

拿到一个真实的不利结果，依次问：

1. **是否必须讨论？** 它是否支撑本文的任一主张、是否被 venue / checklist 要求、是否是审稿人必然会查的对照？都不是则不进正文（**注意：这是"不主动展开"，不是"删数据"**——表里的数字照常在）。
2. **能否换目标解释？** 换评价口径（该 metric 是否是本文优化目标？）、换比较对象（与之比较的是不是同一 operating point / 同一预算？）、或直接说明该指标非本文目标（"we do not optimize for throughput; the comparison is reported for completeness"）。
3. **能否收缩主张到证据实际支持的范围？** 把"我们更弱"改回"在 D 上、在该预算下，差 3.1 点"。收缩的是**主张**，不是**证据**。

三步全部失败，才写成一条平实的 limitation——写成之后按 `PROSE.OVER_DEFENSIVE` 送到它**唯一的 canonical home**（Limitations 块或设计描述），不得在正文里再出现第二次。

## Rationale

agent 写论文时有一条稳定的失效路径：它预期自己会因"不够严谨 / 不够全面"被批评，于是**代替一个想象中的审稿人先攻击稿子**——把每一处可以设想的弱点都标注出来，宁可平庸也不担风险。结果是一篇处处防御性措辞的稿子：没有一句话是假的，但通篇读起来像作者自己不信这份工作。

机制值得说清楚，因为它决定了修法：

- **优化目标错位。** agent 优化的是"不被指责遗漏"，不是"准确"。这两个目标在有利结果上重合，在不利结果上分叉——多加一句自贬永远降低"被指责遗漏"的风险，所以自贬会**单调累积**，没有自然停点。停点必须由规则给。
- **第一印象设定基调。** 审稿人读到的前几句决定了他带着什么预期看数据。同一张表，前面写 `Unfortunately, our method does not outperform the strongest baseline` 和写 `On ImageNet-LT our recall is 3.1 points below the strongest baseline (Table 4); on the other four benchmarks we lead by 1.2 to 4.0 points` 会被读成两份不同的工作，而两句披露的是同一件事。
- **自贬会被原样引用。** 审稿意见里最省力的一句话是引用作者自己的判词。你写下"our method is weaker at recall"，这句话就成了 meta-review 里的既成事实，而你在 rebuttal 里已经没有立场反驳它——反驳自己的原文比反驳审稿人贵得多。

**本卡的存在是为了给一股正确的压力**画边界**，不是取消它。** `ETHICS.LIMITATIONS_SECTION_MANDATORY` 硬性要求一个实质性的 Limitations section；`paper-self-review` 的检查是**加法的**（它找缺什么，找到就要求补）。这两股力都指向"多说自己的问题"，而且都是对的。没有上界时，它们的合力就是上面那条失效路径。本卡提供上界：**披露量由 ETHICS / self-review 决定，措辞与落点由本卡和 `PROSE.OVER_DEFENSIVE` 决定。** 两件事不冲突，因为它们管的不是同一个维度。

## Check

- **提取 `.tex` 正文的正确方法**：见 `policy/references/tex-prose-extraction.md`。手搓扫描器的四个典型错误（`split('%')` 在 `$95\%$` 处截断、剔数学时 `$` 奇数配对吞掉整段、逐行扫描漏掉被硬换行劈开的短语、两遍大小写策略不一致）都会产生**假的「已清零」结论**

- **regex 层（闭集词表）**：`policy/lint.sh --rule PROSE.SELF_UNDERMINING <dir>` 自动执行，见 `lint_patterns`。命中即**候选**，不是判决——按 Requirement 的"删掉是否命题不变"判据逐处裁决。

- **LLM 层（regex 判不了的部分）**，逐句执行：
  1. **责任范围问一**：这句话是否扩大了我要承担的责任？它承诺的属性是否宽于证据？
  2. **责任范围问二**：是否把局部描述成普遍缺陷？句中是否丢失了数据集 / 指标 / 幅度 / 表号锚点？（丢锚点的负面句一律回填锚点，不是删句）
  3. **三步处置**：每一处不利结果，检查作者是否走完了 ① 是否必须讨论 → ② 能否换目标解释 → ③ 能否收缩主张，还是直接跳到了"写成 limitation"。跳步是本卡最常见的实际违规形态。
  4. **主语归属**：`worse` / `lower` / `below` 类比较句，主语是本文还是 baseline？主语是本文且无锚点的，按问二处理。
  5. **落点**：走完三步后确实要写的 limitation，交 `PROSE.OVER_DEFENSIVE` 检查它是否只有一个 canonical home。

- **排除**（命中不算违规）：
  - venue / checklist 要求的 Limitations 内容本身，以及 `ETHICS.LIMITATIONS_SECTION_MANDATORY` 覆盖的一切披露
  - `EXP.FABRICATED_RESULTS_CAPTION_DISCLOSURE` / `EXP.RESULTS_STATUS_DECLARATION_REQUIRED` 要求的状态声明
  - 带完整锚点的中性负面陈述（"on D, recall is 3.1 points lower (Table 4)"）——这是本卡要保护的形态，不是要清除的
  - 描述**他人方法**弱点的 `lags behind` / `falls short`（主语不是本文）
  - 形式化陈述中相对**理论界**的 `falls short of the information-theoretic limit`（pattern 已排除常见界名，其余人工裁决）
  - 直接引语、被引文献标题、注释掉的历史草稿

粗筛（只定位候选，不作判定）：

```bash
rg -ni "unfortunately|regrettably|admittedly|falls short|lags behind|does not outperform|unable to|far from (optimal|practical)" sections/
```

## Examples

### Pass

同一个事实，两种写法——**披露的内容完全相同**，区别只在是否自我攻击：

```latex
% 自我削弱版（Fail）：情绪 + 无锚点 + 局部升格为普遍
% Unfortunately, our method does not outperform the strongest baseline;
% our approach is simply weaker at recall.

% 准确中性版（Pass）：同一事实，锚点齐全，责任范围与证据等宽
On ImageNet-LT, recall is 3.1 points below RIDE at the same inference
budget (Table 4); on the remaining four benchmarks our method leads by
1.2 to 4.0 points.
```

```latex
% 三步处置的第 2 步：换目标解释——数字照常报告，但说明它不是本文目标
Throughput is 0.7$\times$ that of the static baseline. We do not optimize
for throughput; the comparison is reported for completeness, and the
operating point is fixed by the deletion latency budget (\S5.2).
```

```latex
% 三步全部失败后的写法：平实、有锚点、只有一个 canonical home（Limitations 块）
\section*{Limitations}
\paragraph{Coverage of long-tail classes.} On classes with fewer than 20
training samples, recall remains 3.1 points below RIDE (Table 4). We do
not have a diagnosis for this gap and leave it open.
```

### Fail

```latex
% 词表层：情绪副词 + 自我攻击式比较
Unfortunately, our method does not outperform the strongest baseline, and
recall falls short of prior work. Admittedly, we merely match the 2021
results, and we are unable to close the gap.

% 责任范围问二：局部结果升格为普遍能力判决，锚点全部丢失
Our method is weaker at recall.

% 责任范围问一：为正文从未量化的属性签发一条通用负面主张
Our approach is far from practical for real-world deployment.

% 跳过三步处置：一个可以换口径解释的结果被直接写成自贬
Regrettably, our model suffers a performance drop under distribution shift.
```

## Conflicts

五条规则与本卡相邻，分工如下。**凡遇分歧，"必须披露"一侧永远赢**——本卡从不用于减少披露量。

- **`PROSE.OVER_DEFENSIVE` — 结构 vs 词汇。** 那条管**辩护放在哪里、放了几次**（认怂前置 / 免责收尾 / 预防性辩解 / 负向重述、每条 caveat 只许一个 canonical home），它的 Rationale 明说"这是结构性问题，不是词级问题"。因此**词级层面与责任范围判断此前无人认领**，那正是本卡的领地：同一句话，放错位置归那条，写成自我攻击归本卡。一处文本可能同时违反两条（位置错 + 措辞自贬），各报各的，不合并。走完本卡三步处置后仍要写的 limitation，**落点由那条裁决**。
- **`PROSE.HEDGING_DISCIPLINE` — 动词强度 vs 证据强度，它在校准过的 hedge 上压过本卡。** 那条是双向的：既禁 over-claim 也禁 over-hedge。当一句 hedge 是**校准正确**的（`suggests` / `is consistent with` / `we hypothesize`，用于真不确定的主张），它就是正确形态，本卡不得以"听起来示弱"为由把它改强——那是制造 over-claim。本卡管的是与证据强度无关的**情绪与自贬措辞**（`unfortunately`、`merely`、`far from practical`），删掉它们不改变任何动词的强度。
- **`ETHICS.LIMITATIONS_SECTION_MANDATORY` — 强制披露压过本卡。** venue 要求的 Limitations section 必须存在且实质（3+ 具体局限点）。本卡**只改这些局限点的措辞**（去掉情绪副词、回填锚点、把普遍判决收回局部），**不减少条数、不削弱内容、不允许因本卡而使该 section 变得敷衍**。若某次改写的净效果是 Limitations 变短变空，该改写违反那条，回滚。
- **`EXP.FABRICATED_RESULTS_CAPTION_DISCLOSURE` — 强制披露压过本卡。** fabricated / 占位结果的 caption 红色大写披露、以及 `EXP.RESULTS_STATUS_DECLARATION_REQUIRED` 的状态注释，是合规声明，不是自我攻击。本卡对其**零管辖**：不得弱化、不得移位、不得以"读起来不好看"为由改写。
- **`CITE.CLAIM_SUPPORT_REQUIRED` — 收缩主张不得留下无支撑的断言。** 本卡第三步"收缩主张到证据实际支持的范围"会重写比较句；重写后的句子仍须携带它自己的支撑（表号 / 图号 / 引用）。把"我们弱于 X"改成"在 D 上差 3.1 点"却不给出 Table 号，是用一条本卡违规换一条那条违规。**收缩主张的同时必须补锚点**，二者是同一次编辑。
