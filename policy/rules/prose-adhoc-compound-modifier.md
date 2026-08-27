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
params: {}
conflicts_with: [PROSE.INVENTED_CONCEPT_LABEL, PROSE.ELEGANT_VARIATION, PROSE.ABBREVIATION_FIRST_USE]
constraint_type: guidance
autofix: none
lint_targets: "**/*.tex"
---

## Requirement

连字符复合修饰语（`X-based` · `X-aware` · `X-driven` · `X-guided` · `X-centric` · `X-oriented` · `X-agnostic` · `X-preserving` …）在技术散文里**是标准构造**。本卡不管这个构造，管的是**临时造一个、只用一次**。

### 判据只有两条，必须同时成立

1. **全文只出现一次**（hapax）——领域已有的术语会反复出现；临时造的用完就扔
2. **不是本领域已有的术语**——`blockchain-based` 在区块链论文里是标准说法，`gradient-based` 在优化论文里是标准说法

**只满足第 1 条不触发。** 一个标准术语在某篇论文里恰好只出现一次，不是问题。

### 为什么"只用一次"是关键

读者遇到 `community-shift-aware signals` 必须**在句中现场解码**这个复合词：是"对 community shift 敏感"还是"知晓 community 发生了 shift"？解码成本付掉之后，这个词全文再不出现——**成本花了，收益为零**。

反过来，一个复合词若在全文反复出现，第一次的解码成本被后续每一次使用摊薄，它已经成为本文的术语。那时它归 `PROSE.INVENTED_CONCEPT_LABEL` 管（要不要显式声明命名），不归本卡。

### 修法

按优先级：

1. **拆成介词短语或从句**——`exposure-aware signals` → `signals that record what each user was shown`
2. **直接用普通语言说清楚**——多数一次性造词删掉后句子更清楚
3. **若它确实是本文的核心概念**，不要一次性使用：显式声明命名（`we refer to this as ...`）、给可判定的定义、全文一致使用。此时按 `PROSE.INVENTED_CONCEPT_LABEL` 与 `PROSE.ABBREVIATION_FIRST_USE` 走

**不要把一次性造词换成另一个一次性造词**——那只是换个壳。

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

- **机械层（builtin，非 regex pattern）**：`policy/lint.sh` 的 `lint_prose_adhoc_compound_modifier`。逐文件统计以下后缀结尾的复合修饰语，仅报**在该文件中恰好出现一次**且不在 allowlist 内的：
  `based|aware|driven|guided|centric|oriented|enabled|agnostic|informed|preserving|grounded|conditioned|specific|augmented|enhanced`
- **为什么必须是 builtin**：判据是**频次**，逐行正则无法表达。`lint_patterns` 只能判「这一行有没有」，判不了「全文出现几次」
- **allowlist 是地板不是边界**：脚本内置的 26 个（`agent-based` · `data-driven` · `privacy-preserving` …）只挡最通用的。**某个 hapax 是不是本文领域的既有术语，是语义层的判断**，机械层只负责给候选
- **语义层逐条问两件事**：① 这个词在本领域文献里已有吗？有 → 保留；② 没有的话，它在本文出现几次？只有一次 → 拆开或改写；反复出现 → 转 `PROSE.INVENTED_CONCEPT_LABEL` 走命名声明
- **排除**：数学模式内（builtin 已剥离 `$...$`）；`\texttt{}` 内的标识符；作者姓名与专有名词
- **提取口径**：见 `policy/references/tex-prose-extraction.md`

### 「是不是既有术语」这一步的性质与局限（必读）

这一步**由 LLM 依据其训练知识判断**，不是查表，也没有可验证的权威来源。三条局限必须随规则一起交付：

1. **不可复现**——换模型、换版本，同一个词可能判得不一样。本卡的机械部分（hapax 计数）换谁跑都一样，**这一步不是**
2. **有知识截止**——训练截止之后出现的术语、或截止前但极冷门子领域的术语，会被误判为造词
3. **领域不均衡**——判官对自己熟悉领域的术语识别率明显更高

**已知的自身误判实例**：本卡 Rationale 引用的 2026 语料样本中，`brokerage-oriented` 被列为一次性造词，但 *brokerage* 是社会网络分析的既有概念（Burt 的结构洞理论）。该判定**很可能是错的**，保留在此作为局限的证据。

**因此判定方向必须保守**：不确定时**不报**。误报一个真实领域术语的代价（作者认定工具不懂本领域，整条规则被关掉）远高于漏报一个造词。

**评估过的替代方案**：
- *以 pre-GPT 语料建通用词表* —— 已实测，**不可行**：56 篇仅提取到 89 个类型，`agent-based` / `model-agnostic` / `sharpness-aware` 均不在内，规模远不够
- *查论文自身参考文献标题* —— 有吸引力（本地、可复现），但**未完成测量**，不作结论；作为后续候选

## Conflicts

- `PROSE.INVENTED_CONCEPT_LABEL` 拥有**反复使用**的造词（要不要显式声明命名、给定义）。本卡拥有**只用一次**的造词。同一个词不会同时落入两边：判据是出现次数。本卡修法 3（确实是核心概念）就是把它交给那条
- `PROSE.ELEGANT_VARIATION` 是本卡修法的约束：拆开或改写之后，全文必须一致使用同一个说法，不得每次换一种拆法
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
