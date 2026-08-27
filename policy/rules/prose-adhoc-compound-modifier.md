---
id: PROSE.ADHOC_COMPOUND_MODIFIER
slug: prose-adhoc-compound-modifier
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-intro, writing-background, writing-system-model, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: regex
enforcement: lint_script
params: {include_based: false}
conflicts_with: [PROSE.INVENTED_CONCEPT_LABEL, PROSE.ELEGANT_VARIATION, PROSE.ABBREVIATION_FIRST_USE]
constraint_type: guidance
autofix: none
lint_targets: "**/*.tex"
---

## Requirement

连字符复合修饰语（`X-based` · `X-aware` · `X-driven` · `X-guided` · `X-centric` · `X-oriented` · `X-agnostic` · `X-preserving` …）在技术散文里**是标准构造**。本卡不管这个构造，管的是**临时造一个、只用一次**。

### 判定是三值的，不是二值

二值判定（报 / 不报）在中间地带必然失准。实测：`gradient-norm-dependent`（表意清晰但笨重）被报重，`pre-training/fine-tuning`（标准但可优化）被漏掉——**一次错两个方向**。

| 判定 | 用于 | 输出 |
|------|------|------|
| **flag** | 造词，读者必须停下来解码 | 诊断 + 两个具体改法 |
| **hint** | 语法合法、语义清晰，但笨重或可优化 | 一句提示，不给改法清单，作者可以不理 |
| **clear** | 领域既有术语 | **必须写出领域先验出处** |

`hint` 这一档是给"知道它不好但改不改随你"的情形留位置。没有它，这类词只能被塞进 flag（作者觉得工具吵）或 clear（作者觉得工具瞎）。

### flag 档的判据：两条必须同时成立

1. **全文只出现一次**（hapax）——领域已有的术语会反复出现；临时造的用完就扔
2. **不是本领域已有的术语**——`blockchain-based` 在区块链论文里是标准说法，`gradient-based` 在优化论文里是标准说法

**只满足第 1 条不触发。** 一个标准术语在某篇论文里恰好只出现一次，不是问题。

### 为什么"只用一次"是关键

读者遇到 `community-shift-aware signals` 必须**在句中现场解码**这个复合词：是"对 community shift 敏感"还是"知晓 community 发生了 shift"？解码成本付掉之后，这个词全文再不出现——**成本花了，收益为零**。

反过来，一个复合词若在全文反复出现，第一次的解码成本被后续每一次使用摊薄，它已经成为本文的术语。那时它归 `PROSE.INVENTED_CONCEPT_LABEL` 管（要不要显式声明命名），不归本卡。

### 修法：先判意图，再选改法

不要按优先级逐个试。**先问作者当时想干什么**，两条路的改法完全不同：

**场景 A — 图省事，把修饰语压缩进连字符。** 特征：这个词不承载本文的任何主张，只是省了几个词。

> `norm-dependent coordinate escaping behavior`
> → `coordinates that depend on the norm escape ...`

改法是**动词化或拆成介词短语/从句**。顺带消除密集名词化，可读性净增。

**场景 B — 想立一个概念，但只用了一次。** 特征：这个词指向本文真正关心的对象。

> `community-shift-aware signal`
> → 换常规搭配：`signals robust to community shifts`
> → 或**显式命名**：`We define a community-shift-aware signal as ...`，此后全文一致使用

选哪条取决于它是不是本文的命名贡献。是 → 声明 + 定义 + 全文一致（转 `PROSE.INVENTED_CONCEPT_LABEL` 与 `PROSE.ABBREVIATION_FIRST_USE`）；不是 → 换常规搭配。

**两个场景共同的红线**：**不要把一次性造词换成另一个一次性造词**（`exposure-aware` → `visibility-conditioned`），那只是换个壳。

## Rationale

**测量。** 40 篇 arXiv 源码（领域配平，各 20 篇），统计以高产后缀结尾的连字符复合修饰语：

| 语料 | 出现/千词 | 不同类型/千词 | **只出现一次的类型/千词** |
|------|----------:|--------------:|-------------------------:|
| arXiv 2019–2021 | 0.57 | 0.27 | **0.16** |
| arXiv 2025–2026 | 2.00 | 0.88 | **0.48** |
| | 3.5x | 3.3x | **3.1x** |

2025–2026 语料里的一次性造词实例：`community-shift-aware` · `brokerage-oriented` · `compass-guided` · `concatenation-based` · `burst-oriented` · `api-augmented` · `context-conditioned`。

**为什么用「只出现一次」而不是总量。** 一份本地区块链方向的 pre-GPT 稿件出现率为 4.28/千词，是 arXiv pre-GPT 基线的 7 倍——但其类型率仅 0.93、一次性率仅 0.38，即**同几个领域术语反复使用**（`blockchain-based` · `sharding-based` · `PBFT-based`）。按总量判会把整个领域误伤；按一次性判则不会。这条反例是本卡判据的直接来源。

**本卡在本仓库的测量史上的地位。** 同期检验的另外两个候选信号——`, so` 因果连接词密度、结构性重复（命题复述 / 集合重列）——在同一批语料上分别得到 1.00x 与 1.22x，即**不可分**；且两者都依赖人的判断，实测存在实验者期望效应（自判 16x，换独立盲判后降至 1.22x）。本卡是三者中唯一**不依赖任何判断**的：hapax 计数是纯机械的，换谁跑都是同一个数。3.1x 因此是这三个信号里唯一可信的效应量。

⚠️ **仍不构成 AI 检测器。** 3.1x 的分布重叠严重，个案无判别力；且本测量未做主题配平之外的混淆控制。本卡的用途是**改稿**，不是判定作者。

## Check

- **机械层（builtin，非 regex pattern）**：`policy/lint.sh` 的 `lint_prose_adhoc_compound_modifier`。逐文件统计后缀复合修饰语，仅报**在该文件中恰好出现一次**、位于前置定语位置、且不在 allowlist / 豁免内的
- **后缀分层——`-based` 默认不参与**。`X-based` 是 "based on X" 的压缩写法，是**构式不是造词**，可自由组合，两个年代都在大量使用：40 篇语料实测，pre-GPT 的 25 个 hapax 里 **22 个是 `-based`**，2026 的 83 个里 43 个是。含 `-based` 时两组差 4.05x，去掉后 16.27x。默认后缀集：
  `aware|driven|guided|centric|oriented|enabled|agnostic|informed|preserving|grounded|conditioned|specific|augmented|enhanced`
  `LINT_ADHOC_INCLUDE_BASED=1` 可折回——`concatenation-based` / `euclidean-based` 说明 `-based` 仍可能被造得生涩，只是造词不集中在那里
  ⚠️ **16.27x 不可作为效应量引用**：pre-GPT 侧非 `-based` 的 hapax 只有 **3 个**（`document-specific` · `image-aware` · `report-specific`），分母过小、方差极大。可靠的只有方向
- ⚠️ **后缀集是廉价兜底，不是判据边界**。后缀是开集：`score-blinded` · `research-bearing` · `throughput-hardened` 都是本卡要抓的对象，而机械层一个都命中不了。实测过形态泛化（`-ed` / `-ing` 分词形式）：召回从 43 条升到 287 条，但两组分离度从 15.4x 塌到 2.75x，多抓的是 `decision-making` · `cross-validated` · `layer-wise` · `english-speaking` 这类标准英语。**因此机械层刻意选高精度低召回**：召回由阅读承担，机械层只做不会疲劳的兜底。曾试过做一个不限后缀的候选生成器交由 LLM 批量裁决，在一篇 2 万词的真实稿件上产出 415 条候选，其中 `channel-wise` · `end-to-end` · `block-diagonal` · `union-find` 这类标准技术英语占大多数——**那个量级的清单比没有清单更糟，它制造「已经查过了」的假象**。已删除
- **句法过滤：仅前置定语**。`a model-agnostic estimator`（修饰后接名词）才让读者在句中解码；`the estimator is model-agnostic`（表语）时主语已解析完毕，成本低得多。实测：该过滤保留 95 个命中里的 87 个，分离度 3.26x → 3.17x 基本不变，**误报少 8%**
- **机械豁免（两条，均为显式命名行为）**：① 复合词后紧跟缩略语定义 `Latency-Budget-Aware (LBA)`；② 各段首字母全大写 `Delay-Tolerant-Enabled`（句首大写只抬升第一段，不会被误吞）
- **风险标记（不过滤，只分级）**：左项本身为复合结构（≥2 个连字符，`out-of-distribution-driven`）的命中附 `[multi-part left element]`。实测该形态分离度最高（3.50x）但绝对量极小（PRE 2 / POST 6），**当过滤器会漏掉九成命中，只能当权重**
- **机械层在真实工作流中不是入口**。`writing-anti-ai` 逐节执行时不调用 `lint.sh`（skill 内无该指令，hooks 也不触发），识别完全由阅读完成。builtin 的职责是**给不读全文的场合一个廉价兜底**，以及给逐节阅读补一层不会疲劳的覆盖——**它不定义本卡的范围**
- **为什么必须是 builtin**：判据是**频次**，逐行正则无法表达。`lint_patterns` 只能判「这一行有没有」，判不了「全文出现几次」
- **allowlist 是地板不是边界**：脚本内置的 26 个（`agent-based` · `data-driven` · `privacy-preserving` …）只挡最通用的。**某个 hapax 是不是本文领域的既有术语，是语义层的判断**，机械层只负责给候选
- **语义层逐条问两件事**：① 这个词在本领域文献里已有吗？有 → 保留；② 没有的话，它在本文出现几次？只有一次 → 拆开或改写；反复出现 → 转 `PROSE.INVENTED_CONCEPT_LABEL` 走命名声明
- **排除**：数学模式内（builtin 已剥离 `$...$`）；`\texttt{}` 内的标识符；作者姓名与专有名词
- **提取口径**：见 `policy/references/tex-prose-extraction.md`

### 「是不是既有术语」这一步的性质与局限（必读）

这一步**由 LLM 依据其训练知识判断**，不是查表，也没有可验证的权威来源。三条局限必须随规则一起交付：

1. **不可复现性集中在边界，不是均匀分布**——清晰案例跨模型高度一致：`research-bearing` 与 `score-blinded` 经三个独立模型判定均为造词，连改写方向都收敛到"拆成介词短语"。不可靠的是**边界案例**（`brokerage-oriented` 就是本卡自己判错的那个）。因此"不可复现"不能笼统地说；准确的表述是**清晰案例可复现、边界案例不可靠**，而判定方向保守正是为了让边界案例落在"不报"一侧
2. **有知识截止**——训练截止之后出现的术语、或截止前但极冷门子领域的术语，会被误判为造词
3. **领域不均衡**——判官对自己熟悉领域的术语识别率明显更高

**已知的自身误判实例**：本卡 Rationale 引用的 2026 语料样本中，`brokerage-oriented` 被列为一次性造词，但 *brokerage* 是社会网络分析的既有概念（Burt 的结构洞理论）。该判定**很可能是错的**，保留在此作为局限的证据。

**放行必须举证。** 判定一个词是既有术语时，**必须同时说出它的领域先验出处**——文献、经典方法名、或该领域公认的标准搭配（`sharpness-aware minimisation` → Foret et al., SAM）。**说不出出处就不能放行。**

这条把一个主观判断变成**带举证责任的判断**：读者可以核查那个出处，而不是只能相信一句断言。它不能消除不可复现性，但把不可复现的部分压缩到"举证是否成立"，那是可争论、可推翻的。

**反事实测试**（判定时问自己）：把这个词从本文语境里抽出来，直接放进该领域顶会（NeurIPS / OSDI / S&P）的一篇论文里，**同行审稿人能否不停顿地读过去**？会停顿 → 不是既有术语。

**因此判定方向必须保守**：不确定时**不报**。误报一个真实领域术语的代价（作者认定工具不懂本领域，整条规则被关掉）远高于漏报一个造词。

**评估过的替代方案**：
- *以 pre-GPT 语料建通用词表* —— 已实测，**不可行**：56 篇仅提取到 89 个类型，`agent-based` / `model-agnostic` / `sharpness-aware` 均不在内，规模远不够
- *查论文自身参考文献标题* —— 有吸引力（本地、可复现），但**未完成测量**，不作结论；作为后续候选

## Report

**已放行的也要报。** 只报违规时，`read/write` 到底是"查过、判定合格"还是"根本没看到"，作者无从分辨——**沉默不等于干净**。

每条一行，`clear` 档必须带出处：

```
research-bearing          sec1.tex:12   flag    -bearing 的既有搭配是 load-/interest-，
                                                接抽象名词后读者需在「承担科研任务」与
                                                「产出科研成果」之间猜
                                        → A  university teams conducting active research
                                        → B  research-intensive university teams
sharpness-aware           sec3.tex:41   clear   Foret et al., SAM
gradient-norm-dependent   sec3.tex:08   hint    表意清晰，略笨重
```

**两个改法必须是不同类型**（拆成从句 / 换常规搭配），不能两个都是新造的复合词——`cache-miss-aware` → `cache-miss-sensitive` 不是修复，是换壳。

**诊断不作语法裁决**：写「这个复合词增加读者的解码成本」，不写「这是语法错误」——它在语法上通常没错。

## Conflicts

- `PROSE.INVENTED_CONCEPT_LABEL` 拥有**反复使用**的造词（要不要显式声明命名、给定义）。本卡拥有**只用一次**的造词。同一个词不会同时落入两边：判据是出现次数。本卡修法 3（确实是核心概念）就是把它交给那条
- `PROSE.ELEGANT_VARIATION` 是本卡修法的约束：拆开或改写之后，全文必须一致使用同一个说法，不得每次换一种拆法
- **斜杠并列（`A/B noun`）是本卡的同一现象**：同样把修饰关系压进标点、让读者现场解码。`load-balancing/routing module` 是"做两件事的一个模块"还是"两个模块之一"，读者判不了。**但不要按密度报**——实测斜杠密度不是年代信号：arXiv 两组 0.52 → 1.04/千词（2.0x，弱），而一份本地 pre-GPT 稿件 1.44、其当前 draft 1.16（**反向**）。标准对偶（`read/write` · `GPU/TPU` · `input/output` · `client/server`）一律 clear；**只有两侧都是修饰语且并列关系不明时才 flag**，改法是展开为 `A and B` 或 `A-B`
- `PROSE.ABBREVIATION_FIRST_USE` 在修法 3 生效：若造词最终保留并缩写，首次出现处定义

## Examples

### Pass

```latex
% 领域标准术语，且全文反复使用
We compare against gradient-based attribution methods throughout, and the
gradient-based baseline remains the strongest competitor in every setting.

% 拆成从句，读者不需要现场解码
The dataset records signals that indicate what each user was shown, which
lets us separate exposure from engagement.

% 确实是本文概念：显式声明 + 定义 + 全文一致
We refer to a mask whose density is fixed at aggregation time as a
\emph{static mask}. Static masks admit the unbiased estimator of
\Cref{eq:est}, and every experiment below uses static masks.
```

### Fail

```latex
% 造一次，用一次，读者付了解码成本却没有复用
We build a community-shift-aware sampler over the interaction graph.

% 同一句里两个一次性造词，且都未定义
The brokerage-oriented encoder feeds a concatenation-based fusion network.

% 把一次性造词换成另一个一次性造词，不是修复
% exposure-aware signals → visibility-conditioned signals   ✗
```
