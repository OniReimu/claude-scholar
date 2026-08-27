---
name: writing-anti-ai
description: This skill should be used when the user asks to "remove AI writing patterns", "humanize this text", "make this sound more natural", "remove AI-generated traces", "fix robotic writing", "polish this paragraph/section", or needs sentence-level cleanup of AI patterns in prose. Supports both English and Chinese. Based on Wikipedia's "Signs of AI writing" guide plus the local policy PROSE rules — detects and fixes inflated symbolism, promotional language, intensifiers, em-dash abuse, superficial -ing analyses, vague attributions, AI vocabulary, negative parallelisms, copula dodges, rhetorical self-answers, hyphenated compound modifiers coined once and never reused, causal connectives defaulting to ", so" instead of therefore/hence/thus/consequently, and excessive conjunctive phrases. Academic cleanup preserves technical density and the author voice (policy/style-guide.md) — no casual "humanizer" tone. Also handles questions about statistical AI detectors (Pangram, GPTZero, Turnitin AI, "会不会被检测出来") — the skill separates reader-facing tells from detector-facing generation dynamics and never promises detector evasion. This is a LINE edit; for whether a paragraph should exist/move/merge at all, run claim-architecture-review FIRST; for drafting new content use ml-paper-writing.
version: 1.4.2
author: gaoruizhang
license: MIT
tags: [Writing, AI, Anti-AI, Humanizer]
---

# Writing Anti-AI

Remove AI-generated writing patterns from text to make it sound natural and human-written. Supports both English and Chinese.

## ⚠️ Author Style Guide <!-- style:author-voice -->

> **MANDATORY**: 编辑前必须先读 `policy/style-guide.md`。去除 AI 痕迹的目标不只是"不像 AI"，更是要回归作者真实的写作风格。style-guide 定义了那个风格。

## Policy Rules

> 本 skill 执行以下论文写作规则。权威定义在 `policy/rules/`。
> 行内出现处以 HTML 注释标记引用。**冲突时以 `policy/rules/` 为准。**
> 紧凑版 guardrail checklist 见 `policy/guardrail-checklist.md`（32 条禁止模式）。
> **本表是执行清单，不是索引** <!-- policy-table:checklist -->——表里每一条 `enforcement: doc` 的规则
> 都必须在正文有对应的执行块（`validate.sh` 9b 节机器检查），否则它没有任何东西会执行它。

| Rule ID | 摘要 |
|---------|------|
| `LATEX.EQ.DISPLAY_STYLE` | Display 公式用 equation 环境 |
| `LATEX.VAR.LONG_TOKEN_USE_TEXT` | 长变量名用 \text{} |
| `PROSE.AI_LEXICON` | AI 高频词表（tier-1 零容忍 + tier-2 密度阈值） |
| `PROSE.FRACTAL_SUMMARY` | 禁止逐层预告/回顾（"In this section we…"） |
| `PROSE.INVENTED_CONCEPT_LABEL` | 禁止自造术语冒充既有概念 |
| `PROSE.RESTATEMENT_DILUTION` | 同一命题一节内只说一次 |
| `PROSE.SEMANTIC_IDLING` | 每句必须新增可证伪内容：零命题的元叙述句 / 理由复述结论的因果回环 |
| `PROSE.INTENSIFIERS_ELIMINATION` | 删除空洞强调词 |
| `PROSE.HEDGING_DISCIPLINE` | 动词强度匹配证据强度（双向：不 over-hedge 也不 over-claim） |
| `PROSE.EM_DASH_RESTRICTION` | 禁止em-dash（零容忍）；替换标点不算修法，结构必须变 |
| `PROSE.SEMICOLON_RESTRICTION` | 正文段落禁止分号，两个独立子句拆成两句 |
| `PROSE.FILLER_PHRASES` | 删除冗余填充短语 |
| `PROSE.COLON_LIST_OVERUSE` | 禁止正文内联编号列表 |
| `PROSE.RULE_OF_THREE` | 并列列举的密度与重复（不止三项并列）；同一集合不得枚举两次 |
| `PROSE.PROMOTIONAL_LANGUAGE` | 禁止推销性/情绪化用词 |
| `PROSE.FORMATTING_RESTRAINT` | 格式克制（不滥用bold/list） |
| `PROSE.INFORMAL_VOCABULARY` | 口语语域五类分类表（习语状语/短语动词/判断形容词/具象比喻/工作痕迹动词） |
| `PROSE.IDIOM_COLLISION` | 技术短语与常用习语同形（歧义问题，非语域问题） |
| `PROSE.REGISTER_PRESERVATION` | 简化/压缩不得降语域（判 diff 不判 document） |
| `PROSE.ELEGANT_VARIATION` | 术语全文一致 |
| `PROSE.COPULA_DODGE` | 禁止"serves as"替代"is" |
| `PROSE.NEGATIVE_PARALLELISM` | 禁止"It's not X, it's Y"假深刻 |
| `PROSE.NEGATION_CONTRAST` | 禁止"X, not Y"逗号否定对比 |
| `PROSE.TRAILING_AFTERTHOUGHT` | 禁止句末逗号甩短片段（"..., as editable."） |
| `PROSE.COMMA_OVERUSE` | 单句逗号≤3（≥4 触发） |
| `PROSE.MIDSENTENCE_COLON` | 禁止句中解释性冒号（非小标题） |
| `PROSE.SUPERFICIAL_ING_SUFFIX` | 禁止句末-ing浮浅分析 |
| `PROSE.DESPITE_DISMISSAL` | 禁止"Despite challenges"公式化dismissal |
| `PROSE.VAGUE_ATTRIBUTIONS` | 禁止"experts argue"模糊归因 |
| `PROSE.RHETORICAL_SELF_ANSWER` | 禁止"The result? X."自问自答 |
| `PROSE.CLEFT_CONSTRUCTION` | 禁止分裂句前置强调 |
| `PROSE.HYPOTHETICAL_FOIL` | 禁止虚构对照物/第二人称设问 |
| `PROSE.ABSTRACT_AGENCY` | 抽象名词不做行动者、不配比喻动词 |
| `PROSE.ANAPHORA_ABUSE` | 禁止同一句首重复3+次 |
| `PROSE.GERUND_FRAGMENT_LITANY` | 禁止分词片段堆叠 |
| `PROSE.SHORT_PUNCHY_FRAGMENTS` | 禁止极短句独立成段 |
| `PROSE.RHYTHM_VARIANCE` | 句长必须有落差（sd≥10 词），上限规则不是目标值 |
| `PROSE.ANNOUNCEMENT_SENTENCE` | 短句要承载主张，不做预告标签 |
| `PROSE.THEATRICAL_SPLIT` | 禁止"设预期—短促击碎"两拍式反驳 |
| `PROSE.OVER_DEFENSIVE` | 一条 caveat 只准一个 canonical home；禁认怂前置/免责收尾；Abstract/Intro 贡献未立不谈不足 |
| `PAPER.OUTCOME_LOGIC` | 删句级过程流水账（we first tried…）；结构级重排归 claim-architecture-review |
| `PROSE.SELF_UNDERMINING` | 不主动示弱：删情绪副词与自贬措辞，不利结果按「必须讨论→换目标解释→收缩主张」三步处置；只管措辞不减披露 |
| `PROSE.ADHOC_COMPOUND_MODIFIER` | 临时造的连字符复合修饰语（`X-aware`/`X-driven`…）且**全文只用一次**；领域既有术语不算 |
| `PROSE.CAUSAL_CONNECTIVE` | 因果连接词按类型选，不默认用 `, so`；**只改三个可诊断子类**（设计选择伪装成推论 / 因果无证据支持 / 证明步骤），其余保留 |
| `PROSE.UNICODE_ARROWS` | 禁止Unicode箭头，用LaTeX命令 |

## Overview

This skill identifies and eliminates predictable AI writing patterns from prose, based on [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing), maintained by WikiProject AI Cleanup.

**Core insight**: LLMs use statistical algorithms to predict what should come next. The result tends toward the most statistically likely outcome that applies to the widest variety of cases—creating detectable patterns.

## When to Use This Skill

**Trigger phrases:**
- "Humanize this text" / "人性化处理这段文字"
- "Remove AI writing patterns" / "去除 AI 写作痕迹"
- "Make this sound more natural" / "让这段文字更自然"
- "This sounds robotic/AI-generated" / "这听起来像机器写的"
- "Fix the AI patterns" / "修复 AI 模式"

**Use cases:**
- Editing AI-generated content to sound human
- Reviewing text for AI patterns before publication
- Polishing academic or professional writing
- Removing "slop" from prose

## Core Rules (快速检查清单)

### 1. Cut Filler Phrases <!-- policy:PROSE.INTENSIFIERS_ELIMINATION -->
Remove throat-clearing openers and emphasis crutches.

**English examples**:
- "In order to achieve this goal" → "To achieve this"
- "Due to the fact that" → "Because"
- "It is important to note that" → (delete)

**中文示例**:
- "为了实现这一目标" → "为了实现这一点"
- "值得注意的是" → (删除)
- "基于……的事实" → "因为"

### 2. Break Formulaic Structures
Avoid binary contrasts, dramatic fragmentation, rhetorical setups.

**Patterns to avoid**:
- **段落长度异常 → 转诊，不要自己判**：只有一两句的段落（可以并进上下文的）、或明显超长的段落，是 `merge` / `split` 的问题——判它需要看相邻段落各自承载什么信息，那是 `claim-architecture-review` **P1 逐段审计**的 verdict 集合（`{keep, tighten, merge, move, split, delete, escalate}`），不是线编能定的。最小调用见下方转诊说明。
  ⚠️ **段长本身不是 AI 痕迹**：实测 40 篇语料，2019–2021 与 2025–2026 的段落长度中位数同为 **57 词**，均值 74 vs 71，CV 0.86 vs 0.80——**没有差别**。所以**不要按词数抓段落**，触发转诊的是「这段该不该独立存在」，不是它有多长。（句长分布是另一回事，那条有依据，见 §3 `PROSE.RHYTHM_VARIANCE`。）
- Negative parallelisms: "It's not just X, it's Y" <!-- policy:PROSE.NEGATIVE_PARALLELISM -->
- Unnecessary contrast: "X, not Y" / "X rather than Y" / "X instead of Y" — default to plain positive "X is A"; keep the contrast only when ruling out Y carries real information (don't reflexively swap "not Y" → "rather than Y") <!-- policy:PROSE.NEGATION_CONTRAST -->
- 并列列举的密度与重复 <!-- policy:PROSE.RULE_OF_THREE -->：**不要**把三项列表改成四项来破坏三段式指纹——那在学术散文里只会造出更长的列举墙（本行此前写的就是 "prefer two or four items"，它是这个问题的生产者）。四条判据：同段三项并列 ≤1 次；**同一集合不得枚举两次**（首次列举时命名，之后引用名字）；短项内联 ≤4 项、长项（>3 词的名词短语）内联 ≤2 项；一段中带列表的句子 ≤2 句。⚠️ 反向护栏：技术散文里列举合法且常见，**首选修法是命名+引用，不是删项**。**跨节转诊**：本条只判本段/本节内的重复列举；若怀疑同一组对象在别节也列过，转 `claim-architecture-review`——`information-ledger.md` 以集合为 info-key，第二次列举在 `lookup-before-create` 时撞上（最小调用见下方转诊说明）
- Em-dash (zero allowed — not even one): "X---Y---Z" parentheticals. **替换标点不算修法**——`X --- a 12.5$\times$ reduction` 改成 `X, a 12.5$\times$ reduction` 只换了标点，同位语仍然是尾巴，而本条存在的理由正是这种挂载方式。按尾巴性质选：完整命题→拆成新句；对前面名词的限定→关系从句 `, which ...`；真旁白→括号或删。判据是"原来挂着的内容是否已经进入某个语法主结构" <!-- policy:PROSE.EM_DASH_RESTRICTION -->
- **正文段落禁止分号** <!-- policy:PROSE.SEMICOLON_RESTRICTION -->：分号连起来的两个独立子句**拆成两句**，第二句首字母大写。不要把分号换成逗号——那还是标点替换。豁免：行内数学里的条件记号 `p(y \mid x; \theta)`、`\;` 细空格宏、列表项分隔、algorithm/verbatim 环境（lint 的 builtin 会先剥掉这三类再判，并报出剥掉了多少）。⚠️ **这是作者选定的风格约束，不是被验证过的 AI 痕迹**：作者 pre-GPT 2.62/千词 vs 当前 draft 5.60（2.14x），arXiv 同期 1.32→1.80（1.36x），但两侧语料不可比且未做盲评——不要在报告里把它写成"AI 味"证据
- Colon-list overuse: "X: A, B, and C" inline enumeration (restructure into separate sentences or use "such as"/"including") <!-- policy:PROSE.COLON_LIST_OVERUSE -->
- Mid-sentence colon: "key observation: the model fails" — rewrite as a full sentence or split; only heading colons (`\textbf{X:}`) are exempt <!-- policy:PROSE.MIDSENTENCE_COLON -->
- Trailing afterthought: "..., as editable." comma + short tag tacked onto a sentence end (fold into the main clause) <!-- policy:PROSE.TRAILING_AFTERTHOUGHT -->
- Cleft construction: "That is what sets X" / "which is what makes X" / "What X is is Y" — front the real subject instead: "X sets Y" <!-- policy:PROSE.CLEFT_CONSTRUCTION -->
- Hypothetical foil: "A method that only described the data would stop there. Ours predicts." — the invented opponent adds nothing the evidence does not; also "Once you view it as X" second-person staging <!-- policy:PROSE.HYPOTHETICAL_FOIL -->
- Abstract agency: "the analogy's job", "the estimator carries decades of validation", "built to catch" — abstractions do not act; use literal verbs, and never reuse the same figurative verb twice in one document <!-- policy:PROSE.ABSTRACT_AGENCY -->
- 句首重复 ≥3 次 <!-- policy:PROSE.ANAPHORA_ABUSE -->：`We show… We show… We show…` — 靠重复句首造节奏在学术散文里不自然。修法是让第二、三句从各自的**内容**起句，不是换个同义动词
- 分词片段堆叠 <!-- policy:PROSE.GERUND_FRAGMENT_LITANY -->：`Improving throughput. Reducing memory. Enabling longer contexts.` — 每句必须有主语和谓语。改法是合成一句带并列宾语的完整句，或各自补主语
- 极短句独立成段 <!-- policy:PROSE.SHORT_PUNCHY_FRAGMENTS -->：≤5 词的句子单独成段制造戏剧效果。⚠️ **短句本身没问题**——问题只在两种（见 §3 的两道门槛：预告而非主张、两拍式反驳）；一个承载具体结论的短句该留
- Fractal summary: "In this section, we present…" / "As we have seen…" / "Having discussed X, we now…" — 同一信息在标题、预告句、回顾句里讲三遍。节的首句直接进内容，末句停在最后一个具体结论上；前向引用交给 `\Cref{}`，不要用叙述句预告 <!-- policy:PROSE.FRACTAL_SUMMARY -->

### 3. Vary Rhythm
Mix sentence lengths. End paragraphs differently.

**⚠️ 均质化本身就是 AI 痕迹。** 逐句读全部合格、但句长压在同一区间的散文，读者一眼认出是机器写的。执行 `PROSE.SENTENCE_LENGTH`（≤35 词）时不要把所有句子拉到同一长度——那是上限不是目标值。目标是句长标准差 ≥10 词，15–30 词区间占比 ≤55%。 <!-- policy:PROSE.RHYTHM_VARIANCE -->

**Check**:
- Three consecutive sentences same length? Break one.
- Paragraph ends with punchy one-liner? Vary it.
- 整节找不到 <12 词的句子？或找不到 >32 词的句子？→ 分布已被压平，双向修复（合并被拆碎的从句 + 把关键论断压短）
- Sentence with ≥4 commas? Split it or use semicolons—comma-chained clauses read as AI meandering. <!-- policy:PROSE.COMMA_OVERUSE -->

**短句的两道门槛**（短句本身没问题，这两种短句有问题）：
- **预告而非主张**："The difficulty is structural." / "This can be made precise." → 删除测试：删掉后信息是否零损失？是则改写成承载内容的句子 <!-- policy:PROSE.ANNOUNCEMENT_SENTENCE -->
- **两拍式戏剧反驳**："One might expect X. It does not." → 合并测试：用 `but` 连成一句，若信息零损失则原拆分只是 mic drop，合并并让证据承担反驳 <!-- policy:PROSE.THEATRICAL_SPLIT -->

### 4. Trust Readers
State facts directly. Skip softening, justification, hand-holding.

**Bad**: "It could potentially be argued that the policy might have some effect."
**Good**: "The policy may affect outcomes."

### 5. Cut Quotables
If it sounds like a pull-quote, rewrite it.

**Bad**: "This represents a major step in the right direction."
**Good**: "The company plans to open two more locations."

### 5b. 格式克制 <!-- policy:PROSE.FORMATTING_RESTRAINT -->

排版装饰是最容易被误当成"强调"的 AI 习惯。五条：

1. `\textbf{}` 只给**首次定义的核心术语**，不给普通词做强调。一段里三处 bold 等于没有 bold
2. `\textit{}` 只给术语引入和外来语，不做情绪强调
3. 正文段落用连贯散文，**不在段落中间插 `itemize`**。Contribution 列表与算法描述除外
4. `\texttt{}` 只给代码字面量、命令名、以及确属贡献一部分的文件名。方法名和系统名定义一次后用正体，不持续 `\texttt{}`
5. 用模板的原生格式，不加装饰性排版

### 6. Preserve LaTeX Math Rules (Academic Manuscripts)
When editing paper text, preserve math-style constraints instead of "humanizing" them away.

**Required:**
- Display equations must use `\begin{equation}...\end{equation}` <!-- policy:LATEX.EQ.DISPLAY_STYLE -->
- Do not rewrite display equations into `$$...$$` or `\[...\]`
- Inline equations can use `$...$`
- In math mode, variable-like tokens longer than 3 letters must use `\text{}` <!-- policy:LATEX.VAR.LONG_TOKEN_USE_TEXT -->

### 7. No Over-Defensive Placement <!-- policy:PROSE.OVER_DEFENSIVE -->
每条 scope 限定只能有**一个** canonical home（设计描述 或 Limitations 块）。禁止：让步放在段落主题句、贡献/结果段以「做不到什么」收尾、为正文未提出的质疑预先辩解、正面句已蕴含还用否定重说。
这是**结构**问题，逐句读全部合格也可能整篇在道歉——须按节隔离扫描，且裁决前确认该句不是某条 reviewer comment 的唯一落点。

**文档域第 5 条：贡献未立先谈不足。** 扫 Abstract 与 Introduction，任何出现在**贡献陈述之前**的 caveat / 局限 / 排除声明都是违规——读者应当先拿到"这篇做成了什么"，再拿到边界。判定只看位置，不看措辞。
处置**只允许"移"不允许"删"**，且移之前先确认 Limitations 已完整承担该条（`ETHICS.LIMITATIONS_SECTION_MANDATORY` 优先，不得因移动削薄披露）；那句也可能是上一轮 reviewer 要求前置的唯一答复。
**分工**：本条只在 Abstract/Intro 内部判先后，**不做跨节搬迁规划**——那归 `claim-architecture-review` 的 P2 relocation-map，且它在本 skill 之前跑。

### 7b. 不主动示弱，不递刀子 <!-- policy:PROSE.SELF_UNDERMINING -->
删掉情绪副词与自贬措辞（`unfortunately` · `regrettably` · `admittedly` · `merely` · `falls short` · `lags behind` · `does not outperform` · `far from practical` · `遗憾的是` · `仅仅`）——判据是"删掉这个词，命题是否不变"，不变则删；同时回填不利句丢失的锚点（数据集/指标/幅度/表号），不得把一个局部结果（"on D recall is 3.1 points lower"）写成普遍能力判决（"our method is weaker at recall"）。每个不利结果按序走三步处置：**是否必须讨论 → 能否换目标解释 → 能否收缩主张到证据实际支持的范围**，三步全失败才写成平实的 limitation。**落点**：先按 §7 判本段/本节；若这条 caveat 在别节也出现过（同一限定的多个 home），那是跨节问题，转 `claim-architecture-review` 的 P2 relocation-map 定 canonical home（最小调用见下方转诊说明）。
**这条只管措辞与位置，不管披露**：真实的负面结果、失败模式、不利比较照常报告；`ETHICS.LIMITATIONS_SECTION_MANDATORY` 与 `EXP.RESULTS_STATUS_DECLARATION_REQUIRED` 要求的内容优先，任何以本条为由删数字、隐去不利比较或削薄 Limitations 的改动都是误用，回滚。

### 8. 词汇指纹与内容稀释

**词表分两层执行** <!-- policy:PROSE.AI_LEXICON -->
- **Tier 1 零容忍**（delve · leverage · underscore · harness · foster · streamline · showcase · seamless · intricate · meticulous · nuanced · multifaceted · pivotal · tapestry · realm · myriad · plethora · intricacies · "paving the way for" · "valuable insights" · "at its core" …）：一次都不出现。替换原则是用你会读出声的那个词，更优先用本文语域里的具体名词。
- **Tier 2 看密度**（comprehensive · essential · vital · innovative · powerful · facilitate · enhance · ensure · explore · highlight · insights · perspective · interplay · paradigm …）：单个合法，聚集违规。同句两个即 cluster，单文件累计 >5 触发。别逐个替换——找最密的那一段改。
- 词表已按学术语域裁剪：`robust` / `optimize` / `trajectory` / `loss landscape` / `framework` / `approach` 是术语，**不在表内**，不要误伤。
- **套话开场（零容忍，本条此前在本 skill 里不可见）**：`In recent years,` · `has attracted increasing attention` · `With the rapid development of`。lint 已在抓，但执行者读本 skill 时看不到它，等于扫描时不会去查开篇句。**正面要求**：Abstract / Introduction 的第一句必须携带**具体的张力或 gap**（一个可验证的结构性事实），不能是泛化趋势陈述——趋势应由那个事实自己带出来。与 `PROSE.FRACTAL_SUMMARY` 分工：那条管**节内**的预告/回顾（"In this section we…"），本条管**论文/章节的开篇**是不是套话。
- **句首连接词密度**：`Moreover,` / `Furthermore,` / `Additionally,` / `In addition,` 全文合计 ≤4。超出说明逻辑靠连接词粘贴而非论证顺序承载；修法是重排句序，不是换个同义连接词。

**因果连接词要选，不能默认用 `so`** <!-- policy:PROSE.CAUSAL_CONNECTIVE -->
`X, so Y` 是口语里默认的因果连接——**正因为默认，用它时通常没做过选择**。Pre-GPT 基线（两组不重叠类别，56 篇 51 万词）：`, so` 0.18–0.28/千词，`therefore/hence/thus/consequently` 约 2.2/千词，比例 8–12:1。

⚠️ **这不是 AI 痕迹检测**。42 条真实句子的盲评实测：任取一个 `, so`，无论出自 pre-GPT 论文还是当代 draft，"可改进"的比例都差不多（收紧判据后 pre-GPT 64% vs draft 29%，顺序还反转）。差别只在**有多少个**——本地 draft 抽样是基线的 15–25 倍。所以**判据不能是「能否改得更精确」**，否则会把 pre-GPT 水准的散文一起改掉。

**只有三个子类触发改写，其余一律保留：**

1. **设计选择伪装成推论**——`The adversary is adaptive, so we sample fresh randomness each round.` 这是动机不是推论 → `To defend against an adaptive adversary, we sample…`
2. **因果未被证据支持**——只有相关或只是猜测 → 交 `PROSE.HEDGING_DISCIPLINE`
3. **证明 / 推导步骤**——数学蕴含里 `hence` / `thus` 是本领域惯例

后果显而易见的 `so`、解释性的 `so`、`so it remains to show` 这类证明惯用语——**不动**。

归入三类之后，再走三问定修法：

1. **这是哪一种因果？** `therefore` 逻辑蕴含（从前提必然得出）· `hence` 就近承接刚建立的结论 · `thus` 以此方式/据此（构造性推论）· `consequently` 经验后果（实测/运行中观察到）。**说不出是哪一种 = 这个因果本身没想清楚**，先想清楚再选词，不要因为说不清就退回 `so`
2. **证据支持这个因果吗？** 只支持相关、或那其实是个设计选择 → 不是语域问题，是 over-claim，交 `PROSE.HEDGING_DISCIPLINE`：降级动词或拆开陈述，**不是换连接词**
3. **需要连接词吗？**（首选修法在这里）两个独立子句用逗号+`so` 粘着，多数时候应把因果写进**句法**：从属化 `Because A, B` · 关系从句 `, which tightens the bound` · 断句 `A. It follows that B.` 从属化同时少一个逗号（协同 `PROSE.COMMA_OVERUSE`）

⚠️ **反向护栏**：**不要机械替换**，把全篇 `so` 统一换成 `therefore` 只是把一种指纹换成另一种，并造出新的均质化（撞 `PROSE.RHYTHM_VARIANCE`）。正式连接词必须按语义分布。

不在范围内：`so that`（目的从句，合法）· `so far`（习语性状语，归 `PROSE.INFORMAL_VOCABULARY` 第 1 类）· `so large` 式程度副词。

**口语语域按五个具名类别查，不是按词表查** <!-- policy:PROSE.INFORMAL_VOCABULARY -->
词表只够得着第 1 类。实测一篇 author-original 稿件的 30 处语域问题里，**29 处是多词构造**，词表命中 0/26。
1. `LEXIS` **习语性状语**（regex 可判）：`at all` · `in the first place` · `ahead of time` · `up front` · `at the end of the day` · `so far` · `more or less` · `pretty much`。删除测试：删掉命题是否不变？⚠️ `from scratch` 是术语（`retrain from scratch`），**不在此列**
2. `PHRASAL-VERB` **短语动词顶替拉丁语源动词**：`comes with`→`entails`、`sits in`→`lies in`、`gives up`→`forfeits`、`bears this out`→`confirms this`。⚠️ allowlist：`carries out` / `rules out` / `follows from` / `falls back` 等是本领域标准用法
3. `JUDGMENT-ADJ` **判断性形容词**（最易过度执行）：`hard` / `cheap` / `easy` / `huge`。判据只有一条——**该词是否已是本领域既有术语？** `cheap unlearning` 是术语，强改会撞 `ELEGANT_VARIATION`
4. `PREDICATE-METAPHOR` **具象名词比喻**：`the wall is a property` → 换回本文正式术语（`the obstruction`）。与 `ABSTRACT_AGENCY`（抽象名词做施事）分工
5. `WORK-TRACE` **内部工作痕迹动词**：`what quarantine buys` / `survives the conditioning`。判据：把主语换成不会有体验的对象，句子还成立吗？

**一个概念全文只用一个词** <!-- policy:PROSE.ELEGANT_VARIATION -->
为"避免重复"而换同义词是中学作文的要求，学术散文相反：术语重复是**精确**，换词是**制造新指称**。最常见的四组混用：`model / framework / architecture / system` · `method / approach / technique / scheme` · `dataset / data / corpus / benchmark` · `training / learning / optimization`。首次引入时定义，此后全文一致。

**这条是另外两条修法的约束**，执行它们时会撞上：`PROSE.RULE_OF_THREE` 要求"首次列举时给集合命名，之后引用名字"——名字必须全文同一个；`PROSE.INFORMAL_VOCABULARY` 要求"替换用稿件已有的措辞"——同样是这个词。所以改任何一条时，先查该概念在别处叫什么，不要新造。

**技术短语撞习语要换掉** <!-- policy:PROSE.IDIOM_COLLISION -->
`A fair bit selects one member` —— `fair bit` 指无偏比特，技术正确，但读者第一遍读成"相当多"。不是语域问题也不是准确性问题，是歧义。同类：`a good deal` · `on the order of` · `significant`（统计 vs 重要）。改法是把隐含限定词显式写出来：`an unbiased random bit`。

**造一次、用一次的连字符复合词要拆开** <!-- policy:PROSE.ADHOC_COMPOUND_MODIFIER -->
连字符复合修饰语这个构造**本身是标准的**，本条不管它。管的是**临时造一个、全文只用一次**：读者要在句中现场解码 `community-shift-aware signals`，解码完这个词再不出现——**成本付了，收益为零**。

⚠️ **后缀是开集，不要按列表找。** 下面 lint 用的那张后缀表是**廉价兜底**，不是判据边界——`score-blinded` · `research-bearing` · `throughput-hardened` 这类它一个都抓不到，而它们正是本条要抓的。**你读稿时的判据是「这个复合词读者要不要停下来解码」，不是「它的后缀在不在表里」。** `research-bearing` 尤其典型：`-bearing` 的既有搭配是 `load-bearing`（承重）、`interest-bearing`（计息），接到抽象名词 `research` 上，读者必须现场猜是"承担科研任务的"还是"产出科研成果的"。

判据两条必须同时成立：**① 全文只出现一次 ② 不是本领域既有术语**。`blockchain-based` 在区块链论文里、`gradient-based` 在优化论文里都是标准说法，反复出现即合规。

⚠️ **反向护栏**：一份区块链方向的 pre-GPT 稿件复合词出现率是基线的 7 倍，但几乎全是同几个领域术语反复使用。**按总量判会把整个领域误伤，只能按「一次性」判。**

**判定三值，不是二值**：**flag**（造词，须停下解码 → 给两个不同类型的改法）· **hint**（合法且清晰，只是笨重或可优化 → 一句提示，作者可以不理）· **clear**（领域既有术语 → **必须写出出处**）。二值在中间地带必然失准，实测同时报重 `gradient-norm-dependent` 和漏掉 `pre-training/fine-tuning`。

**已放行的也要报**——只报违规时，作者分不清 `read/write` 是"查过合格"还是"没看到"。**沉默不等于干净。**

**斜杠并列同属本条**：`load-balancing/routing module` 是"一个模块做两件事"还是"两个模块之一"？标准对偶（`read/write` · `GPU/TPU` · `client/server`）clear；两侧都是修饰语且关系不明才 flag。⚠️ **不要按密度报**——斜杠密度实测不是年代信号（arXiv 2.0x，而本地稿件反向）。

**放行必须举证**：判定它是既有术语时，**同时说出领域先验出处**（`sharpness-aware minimisation` → Foret et al., SAM）。说不出出处就不能放行。反事实测试：把这个词抽离本文语境、直接放进该领域顶会论文里，**同行审稿人会不会停顿**？

**修法先判意图，不要按优先级试**：
- **场景 A｜图省事**（这个词不承载任何主张）：动词化或拆成从句 —— `norm-dependent coordinate escaping behavior` → `coordinates that depend on the norm escape ...`
- **场景 B｜想立概念但只用了一次**：换常规搭配 `signals robust to community shifts`；**或**显式命名 `We define a community-shift-aware signal as ...` 并全文一致使用（转 `PROSE.INVENTED_CONCEPT_LABEL`）

红线：**不要把一次性造词换成另一个一次性造词**（`exposure-aware` → `visibility-conditioned`），那只是换壳。

`lint.sh` 的 builtin（**只是兜底，不是入口**）已做四件事：**只报前置定语位置**、**放行缩略语定义与全段大写的命名约定**、**给左项复合的命中打 `[multi-part left element]` 风险标记**、**`-based` 默认不报**（`X-based` 是 "based on X" 的构式不是造词，两个年代都在用；含它信号从 16x 稀释到 4x）。造词集中在 `-aware` / `-oriented` / `-centric` / `-guided` / `-conditioned` 这几个后缀上。

机械层由 `policy/lint.sh` 的 builtin 给候选（只报 hapax），**是不是领域术语由你判**——这一步依据你的训练知识，**不可复现、有知识截止、冷门子领域识别率低**。因此**不确定时不报**：误报一个真实领域术语会让作者判定工具不懂本领域、整条规则被关掉，代价远高于漏一个造词。

**不要给现象起名** <!-- policy:PROSE.INVENTED_CONCEPT_LABEL -->
"the supervision paradox" / "workload creep" 这类标签只在两种情况下允许：有文献出处并引用，或是本文明确的命名贡献（显式声明 + 定义 + 全文一致）。两者都不满足就用普通语言描述现象。审稿人会去检索你造的"既有概念"。

**同一命题一节内只说一次** <!-- policy:PROSE.RESTATEMENT_DILUTION -->
AI 平均把每件事说 1.5 遍：抽象陈述一次 → 给证据 → 换措辞再总结一次。删除测试：删掉后出现的那句，本段信息零损失即确认是复述，删掉定稿（不要合并改写）。跨节的重复主张不归这里，交 `claim-architecture-review` 的 **P2 relocation-map**——同一命题有多个 home 时由它定 canonical home（最小调用见下方转诊说明）。

**每句必须新增可证伪的内容（"轱辘话"）** <!-- policy:PROSE.SEMANTIC_IDLING -->
与上一条正好相反：那条是**同一个命题说了两遍**，这条是**一个命题都没有**。所以删除测试在这里不成立——空转句删掉后信息同样零损失，但原因不是"别处说过"，而是"这里从来没说过"，修法因此不是删后一句而是补事实。

两种形态：
- **A 零命题（元叙述空转）**：句子在说"我们在做分析 / 这很重要 / 这带来了理解"，却没有任何变量、数值、机制或结论。判据是**可移植性**——这句能不能原样搬进一篇完全不同的论文而不改一个词，能就是空转。
  例：*To provide a comprehensive understanding of the underlying dynamics, we carefully examine the various factors that influence the overall behavior of the system, thereby gaining valuable insights.*（30 词，零具体对象）
- **B 因果回环**：`because` / `since` / `which enables` 之后的解释项是被解释项的换词重述，理由和结论是同一件事。
  例：*The model achieves low latency because the execution time is reduced, which effectively enables faster processing.*（low latency = reduced execution time = faster processing）

**检查方法只有一个**：逐句写出这句断言了什么，**必须包含至少一个具体对象**（变量名/数值/模块名/数据集/机制）；写不出来即形态 A。含因果连接词的，分别写出解释项与被解释项，同一命题两种措辞即形态 B。

**修法只有两条：具体化或删除。** 不要改写成更好听的空话。

**输出改写的前提是这一段还有命题存活**，且存活的命题必须 100% 保留（可压措辞，不可丢断言），方向是**强动词 + 紧凑谓语**，删 `thereby` / `which serves to` / `allowing us to` 这类虚词接从句。**一句都没存活就不要给改写**——那是 escalate，正确输出是删除或转诊。实测依据：对一份十段测试集，出题方给的十条 gold rewrite 盲判后 **6 条仍然违规**，且失败的全部集中在无命题存活的段落（`Our framework consistently outperforms baseline methods on standard benchmarks` 仍无 baseline/benchmark/幅度；`Minimizing intermediate computation time reduces overall end-to-end inference latency` 两个量仍是同一个量）。**压缩对有内容的段落是提纯，对没内容的段落是把空话变短**，而短的空话读起来更像结论。

**不设压缩率目标**：75–85% 是删完填充后观察到的结果，不是该瞄准的指标；当指标会制造为凑比例而删内容的压力。可以报告，不可以当判据。

**逐句提取，逐段判定**：单句脱离上下文判会系统性高估——实测段落粒度对真实已发表论文零误报（30 段），孤立单句粒度误报 3/10，被误报的正是「下一句就兑现的 topic sentence」和领域惯例的动机句。

⚠️ **不要用指标代替判断**：不做嵌入余弦、不做 filler-token 占比、不做命题密度阈值。本仓库实测过这类代理量——结构性重复信号自评 16x，换 fresh 模型盲评后塌到 1.22x，组内离散度比组间差异还大。给判断套阈值，测的是仪器自己。

**豁免**：定义句与形式化陈述；**下一句就落地兑现**的抽象铺垫（那是 topic sentence，判据是兑现距离——隔三句还在抽象层才违规）；Ethics / Threats to Validity / Broader Impact 的规定动作与 `\paragraph{}` 标题句；直接引语与被批评的对象文本。
**Future Work / Conclusion 套话不豁免**——`paving the way for progressive improvements in subsequent research endeavors` 和任何一篇论文的未来工作段可以互换，正是形态 A。未来工作必须点名一个具体限制或一个具体的下一步配置。

⚠️ **判形态 B 之前先问：解释项有没有引入一个独立可测的量？** 有就不是回环，哪怕这个因果在领域内是常识。实测过判案例：`because the representations remain invariant across distribution shifts` 被判成回环，但 invariance → OOD reliability 是真机制。**没被测量的机制主张是含糊，不是空转。**

**报告已放行项**：每段报出被检查句数与放行数。只报违规，作者分不清"查过合格"和"没看到"。

**转诊分流**：一段之内**多数句子**都判为形态 A 时，不要逐句报——整段转 `claim-architecture-review` 跑 **P1 逐段审计**（`spine.md` 不存在时先跑 P0）。逐句"具体化"一个本身没有内容的段落，产出的是更好听的空话；该判的是这段该不该存在（最小调用见下方转诊说明）。

### 8b. 删掉过程流水账（句级） <!-- policy:PAPER.OUTCOME_LOGIC -->

论文写**最终成立的逻辑**，不写**做事的顺序**。时间顺序属于 lab notebook。线编能处理的是其中三种**纯句级**泄漏：

```
we first tried X, which did not work, so we then ...
initially we used A but later switched to B
in an earlier version of this work / in our preliminary implementation
```

**判据（删除测试）**：删掉这段行踪，**最终设计是否仍然完整**？完整则删，直接陈述最终设计；不完整说明那不是行踪，是设计理由，保留并改写成理由。

**不要为这条写 regex。** `first` / `then` / `initially` 在学术散文里合法用法远多于违规用法（"We first define the threat model" 完全正常），规则卡因此刻意不发 `lint_patterns`。逐处判断，不做模式匹配。

**边界（这条最容易被误用成删证据）**：
- 只删**实现弯路**。一个**跑过并被报告的实验**即使当初是"试了一下"，它的结果一旦限定了主张，就是证据不是行踪——保留（`EXP.RESULTS_STATUS_DECLARATION_REQUIRED`）。解释"优势来自哪里"的消融、划定失效边界的负面结果同理，禁令不适用。
- 与 §7b 的分工：`we first tried X, which did not work` 可能同时触发两条——**§7b 管这个不利结果怎么措辞**（别写成自贬），**本条管这段时间顺序该不该出现**。两条各报一次，不合并。

**结构级不归本 skill**：Method 按实现史排序、Results 按跑的时间排序、只为交代"我们也试过"而存在的整段——这些要重排章节，归 `claim-architecture-review` 的 **P3 narrative closure**——它判 spine 是否闭环、以及这个 spine 是不是最强证据指向的那个命题（最小调用见下方转诊说明）。规则卡里那半条**重定义问题、重排贡献的授权**同样是写作/结构层的事，线编不执行。

### 9. Claim–Evidence Calibration（动词对证据） <!-- policy:PROSE.HEDGING_DISCIPLINE -->

逐个实证主张查两件事：**(a) 有没有锚点**（数字/图表/引用在同句或紧邻句），**(b) 动词是否匹配证据强度**。

- 无锚点的比较主张 → 补锚点或收窄："Our method is more robust." → "Our method's accuracy drops by 2 points under distribution shift, versus 11 for the baseline (Fig. 3)."
- 动词强于证据 → 降级：prove/demonstrate/establish/guarantee 只留给数学证明或穷尽验证；实验支撑用 show / provide evidence / improve by N
- 模糊幅度 → 区间 + comparator："a large improvement" → "a 2--6\% improvement over the strongest baseline"（对比对象取最强者，不是 trivial baseline）
- **反向同样违规**：把校准过的 suggest/may indicate 改成确定性动词是制造 over-claim（见 Do NOT Over-Correct §1）

## Common AI Patterns (常见 AI 模式)

### Content Patterns (内容模式)

| Pattern | Description | 中文描述 |
|---------|-------------|----------|
| **Undue emphasis** | "stands as a testament", "crucial role" | "作为……的证明"，"关键作用" |
| **Promotional language** | "vibrant", "rich heritage", "breathtaking" | "充满活力的"，"丰富遗产"，"令人叹为观止" |
| **Vague attributions** | "Experts believe", "Observers note" | "专家认为"，"观察者指出" |
| **Superficial -ing analyses** | "highlighting the importance", "ensuring that" | "强调……的重要性"，"确保……" |
| **Formulaic "challenges" sections** | "Despite X, faces challenges" | "尽管……面临挑战" |

### Language Patterns (语言模式)

| Pattern | Description | 中文描述 |
|---------|-------------|----------|
| **AI vocabulary** | Additionally, crucial, delve, enhance, landscape | 此外，至关重要，深入探讨，增强，格局 |
| **Copula avoidance** | "serves as", "stands for", "represents" | "作为"，"代表"，"充当" |
| **Em dash (zero allowed)** | "X---Y---Z" parenthetical insertions | 破折号一个都不允许（做插入语） | <!-- policy:PROSE.EM_DASH_RESTRICTION -->
| **Colon-list overuse** | "X: A, B, and C" inline enumeration | 冒号引出内联列表 |
| **Rule of three / 列举堆积** | Forcing ideas into groups of three; the same set enumerated twice; multi-word noun phrases stacked inline | 强行三段式；同一集合列两遍；长名词短语内联堆叠 |
| **Elegant variation** | Excessive synonym substitution | 过度换词 |

For comprehensive pattern lists, see:
- **`references/patterns-english.md`** - Complete English pattern reference
- **`references/patterns-chinese.md`** - Complete Chinese pattern reference

## Voice, by Register（语域决定"人味"是什么）

去掉 AI 模式只是一半；另一半取决于语域——**两种语域的"人味"定义相反，用错方向就是制造新的返工**。

### 学术语域（论文、rebuttal、技术报告）——默认

**中性、精确、证据绑定本身就是人类学者的声音。** 不要注入观点、幽默、情绪或第一人称"个性"：

- ❌ 不加 "I think" / "honestly" / 俏皮话 / 情绪反应（这会触发 `PROSE.INFORMAL_VOCABULARY`，且违背 `policy/style-guide.md`）
- ✅ 人味来自：具体的数字和命名的对象、承载主张的短句与展开论证的长句交错（`PROSE.RHYTHM_VARIANCE`）、作者立场通过**主张的选择和证据的排布**表达
- ✅ "we" 是学术标准用法，保留；校准过的 hedge 是学者声音的一部分，保留

### 随意语域（博客、社交帖、newsletter、个人邮件）

此时才适用"注入灵魂"：

- **Have opinions.** "I genuinely don't know how to feel about this" 比中立列利弊更有人味
- **Acknowledge complexity.** "This is impressive but also kind of unsettling" 胜过 "This is impressive."
- **Use "I".** "I keep coming back to..." 是真人在思考的信号
- 中文同理："我真的不知道该怎么看待这件事"比中立地列出利弊更有人味

## ⚠️ Do NOT Over-Correct（学术语域反向护栏）

通用 humanizer 会把合法的学术构造当 AI 痕迹铲掉——**过度矫正制造的问题和 AI 痕迹一样严重**。

护栏有两个方向，**第 1 条是替换方向，其余是删除方向**。这个顺序是实测逼出来的：一次压缩 pass 产生的九处语域违规，没有一处删掉了下面 2–7 条里的任何东西，它们全都是**用低语域措辞替换了合语域措辞**——只讲"不要删"的护栏对这类问题完全不设防。

### 1. 不要在简化/压缩时降低语域 <!-- policy:PROSE.REGISTER_PRESERVATION -->

**语域是编辑动作的属性，不是词的属性**，所以这条判的是 **diff**：一个替换即使**准确**，只要语域低于它替换掉的措辞，就是违规。

| 改前 | 改后（违规） | 问题 |
|---|---|---|
| route micropayments to peers | pay peers | 精确度损失——指称对象没了 |
| a sanction set too high deters | too heavy a one drives off | 名词被代词顶替 + 短语动词顶替拉丁语源动词 |
| permanently excluding a verifier | excluding a verifier for good | 口语惯用语 |
| Institutional instruments impose a cost | Institutions hold levers | 谓语位置的比喻 |

注意前两行**根本不口语**，只是更含糊——词表永远抓不到它们。

**修复规则（先做这个，再考虑自己造措辞）**：**用这篇稿子在别处已经使用的措辞**。实测九处里有七处的正确改法已经存在于更早草稿或另一节中。报告格式必须四列：`original → replacement → suggested wording → source of the suggestion`，写出 source 才可核对。

**易懂 ≠ 低语域。** 降低阅读门槛的正确手段是理顺句子结构、讲清术语，不是换上口语措辞。实测中一处 `at all` 正是 agent 在「让句子更好懂」这一步主动加进去的——**anti-AI pass 本身就会触发这个失效模式**，不只是压缩 pass。

**压缩为什么必然触发这个**：压缩优化词数，而最便宜可砍的恰恰是精确的多音节词。所以 drift 不是失误，是这个优化目标的预期输出，每次压缩 pass 都会复发，除非语域被单独计分。`PROSE.RHYTHM_VARIANCE`（拉句长方差）推的是同一个方向——**拉方差不得靠写短口语句**。

### 以下内容保留，不要"修复"（删除方向）

2. **校准过的 hedging**：suggests / is consistent with / we hypothesize / may indicate 用在真不确定的主张上是**必需的**。把 "the results suggest X" 改成 "the results prove X" 是制造 over-claim（见 `PROSE.HEDGING_DISCIPLINE` 校准红线） <!-- policy:PROSE.HEDGING_DISCIPLINE -->
3. **行动者无关时的被动语态**："Samples were normalized to total protein." 不改主动
4. **第一人称复数 "we"**：学术标准，不为"去 AI 味"改写规避
5. **正式定义、命名的方法/指标、术语、公式、符号**：逐字保留
6. **数字、结果、引用**：永不发明、删除或改动；cite key 一个不丢
7. **分号、偶发的三项并列、真实需要的列举**：适度即合法。`PROSE.RULE_OF_THREE` 管的是**密度与重复**，不是禁绝列举——一个方法确实作用于四种维度就该说出那四种；它的修法是给集合命名后引用，**不是删项**

**Funding proposal 是另一个语域**：proposal 靠 vision + feasibility 卖，论文语域要削的 ambition 语言（"long-term goal"、"transformative"）在 proposal 里是预期形态。改 grant/fellowship 文本走 `grant-application-writing` skill，不要用本 skill 的论文标准去压平 vision。

## 两个不同的目标：读者 vs 统计检测器

本 skill 上面所有内容针对的是**读者**——让熟悉 AI 模式的人（现在包括审稿人）读不出机器痕迹。这和**骗过统计检测器**（Pangram、GPTZero、Originality.ai、Turnitin 的 AI 模块）不是同一件事，两者的失败模式互不相关。

| | 读者层（reader tells） | 检测器层（statistical detectors） |
|---|---|---|
| 判据 | 词汇、句式、结构、语气 | 生成动态（planned-text dynamics）+ 主导作者身份 |
| 本 skill 的规则 | 全部有效 | **基本无效** |
| 实测 | — | 同一段文字用 11 种风格重写（punchy / rambling / 全小写 / 带错别字 / ESL 腔 / 模仿特定人笔迹）**全部 100% AI，无梯度** |

**必须说清楚的三件事：**

1. **改完所有 tell ≠ 检测器会放过。** 逐条清完本 skill 的规则只保证人读不出来。如果用户关心的是投稿系统的 AI 检测，直说这两件事不是一回事，别让他以为改完就安全了。
2. **学术语域下唯一可用的检测器手段是 interleave protocol**（见下节）。让文本"看起来乱"的那套办法（全小写、掉撇号、自我打断、`!!`）在论文里是荒谬的，不要用——那是社交/newsletter 语域的技术。
3. **所有阈值都是快照，不是承诺。** 证据来源、日期和样本量见 `references/evidence.md`；检测器会 retrain，任何"多少比例能过"的数字都要重测。

## Interleave Protocol（人类底稿 + AI 补写）

唯一在学术语域下可用、且有实测支撑的检测器层手段。前提是**存在一份用户自己写的底稿**。

**机制**：分类器判的是主导作者身份，不是句子质量。真人写的底稿能吸收相当比例的 AI 句子而仍被判为人类，前提是 AI 句子始终被真人句子夹住。

**执行规则（五条，缺一条就退化）：**

1. **逐字保留用户的句子。** 不改语法、不改错别字、不改跑题的长句——这些是载荷信号。**改写一句人类句子等于把它变成 AI token。**
2. **只加不换。** 你的句子插在他的句子**之间**，绝不连续两句都是你写的。
3. **AI 词占比 ≤ ~40%。**
4. **开头句必须是用户的，逐字保留。** 首批 token 决定分类器的初始判断。
5. **交付时标注哪些句子是他的、哪些是你的**，否则他下一轮编辑会把人类信号改掉。

**实测数据**（Pangram v3.3.2，2026-07-22，n 极小，社交/newsletter 语域）：13% AI 交错 → Human；35% → Human；45% 严格交替 → Human；70% 且含一段连续 AI 块 → 100% AI。

**用在论文上的做法**：让用户口述或速记一段自己的话（idea 底稿、口述转录、实验当天的笔记都行，质量无所谓，作者身份才是关键），你在他的句子之间补技术表述、公式引导、引文衔接。**这条路径的产物仍需过一遍本 skill 的读者层规则**——interleave 保的是检测器，不保证句子写得好。

> ⚠️ 上述比例来自社交语域的小样本，**学术长文没有被测过**。当作 hypothesis 用，不要当作保证；用户真的关心结果时让他自己在目标检测器上复测，并把结果记回 `references/evidence.md`。

## Workflow (工作流程)

> **Order — line edit is LAST.** Run `claim-architecture-review` (paragraph necessity / placement / cross-section redundancy + claim spine) and then `paper-self-review` BEFORE this skill. Fix the architecture before polishing sentences; polishing a paragraph that should be moved or cut is wasted work.

> **📄 转诊说明（定点调用本 skill 时读这条）**
> 逐节定点跑本 skill 是**合法用法**，不是流水线的错误入口。但本 skill 判不了跨节的事，遇到上述转诊点时，
> **不必重跑整篇 arch-review**——它的 P1 本来就是逐节的：
> - `architecture-review/spine.md` 已存在 → **只对当前这一节跑 P1**，读 ledger 判跨节重复
> - spine 不存在 → 先跑 **P0**（只读 abstract + intro + 各节标题 + 每段主题句，不读正文，很便宜），再 P1 单节
> - 只有需要全局搬迁规划时才跑 P2/P3
> 转诊要说清**最小需要跑什么**，只丢一个 skill 名字等于没转。

> **🚦 压缩闸门（compression / de-jargon / simplify pass 必须遵守）** <!-- policy:PROSE.REGISTER_PRESERVATION -->
> **register check 未通过之前，不得报告词数或压缩百分比。**
> 实测教训：agent 在作者读到正文之前三次报告「968 → 728 words, −25%」作为成功指标，而那一版正文里有九处语域塌陷。**数字先于质量出现，就会替代质量成为验收标准。**
> 顺序固定为：压缩 → 逐个改动 span 跑替换测试（见 Do NOT Over-Correct §1）→ 用稿件已有措辞修复 → **然后**才报数字。

> **📏 输入尺度会静默削掉覆盖率。** 一段（~200 词）和一整个 `.tex`（1.2 万–6 万词）不是同一件事，但一遍读完再报告，两种情况读起来一样自信。**给整篇时不要一次通读就出结论**——按 section 逐个过，或者明说这一轮只覆盖了哪些规则、哪些部分。给一段或一节时不需要这条。

> **🔁 收尾必做：跑一次机械兜底**
> ```bash
> bash policy/lint.sh <本节所在目录>
> ```
> 本 skill 的 101 条规则由**阅读**执行，其中 33 条同时有正则实现。读一节两千词，第 12 段的一个破折号、一句 36 词的长句，**人会看漏，正则不会**。两层是互补的：判断力来自阅读，不疲劳的覆盖来自 lint。
> 报出来的每一条逐个确认——**不要**因为「我刚才读过了」就跳过。若某条是本 skill 判定后**刻意保留**的（例如 `PROSE.CAUSAL_CONNECTIVE` 三个子类之外的 `, so`），在回复里写明保留理由，不要默默忽略。
> ⚠️ `--constraint-type guardrail` **看不到 guidance 类规则**，收尾要跑不带过滤的完整版。

### For English Text:

1. **Identify patterns** - Scan for AI patterns listed above
2. **Rewrite sections** - Replace AI-isms with natural alternatives
3. **Preserve meaning** - Keep core message intact
4. **Maintain voice** - Match intended tone (formal, casual, technical)
5. **Add soul** - Inject personality and opinions

### For Chinese Text (中文文本):

1. **识别 AI 模式** - 扫描上述列出的模式
2. **重写问题片段** - 用自然替代方案替换
3. **保留含义** - 保持核心信息完整
4. **维持语调** - 匹配预期的语气（正式、随意、技术）
5. **注入灵魂** - 添加个性和观点

## Quick Scoring (快速评分)

Rate the text 1-10 on each dimension (总分 50):

| Dimension | Question | 问题 | Score |
|-----------|----------|------|-------|
| **Directness** | Direct statements or announcements? | 直接陈述还是绕圈宣告？ | /10 |
| **Rhythm** | Varied or metronomic? | 节奏变化还是机械重复？ | /10 |
| **Trust** | Respects reader intelligence? | 尊重读者智慧吗？ | /10 |
| **Authenticity** | Sounds human? | 听起来像真人吗？ | /10 |
| **Density** | Anything cuttable? | 有可删减的内容吗？ | /10 |

**Standard**:
- 45-50: Excellent, AI patterns removed
- 35-44: Good, room for improvement
- Below 35: Needs revision

## Examples (示例)

See **`examples/`** for before/after transformations:
- **`examples/english.md`** - English text examples
- **`examples/chinese.md`** - Chinese text examples

## Quick Reference (快速参考)

### English - Common Fixes:
| Before | After |
|--------|-------|
| "serves as a testament to" | "shows" |
| "Moreover, it provides" | "It adds" |
| "It's not just X, it's Y" | "X does Y" |
| "Industry experts believe" | "According to [specific source]" |

### 中文 - 常见修复：
| 改写前 | 改写后 |
|--------|--------|
| "作为……的证明" | "表明" |
| "此外，……提供了" | "……增加了" |
| "这不仅仅是……而是……" | "……是……" |
| "专家认为" | "根据[具体来源]" |

## Additional Resources

### Reference Files
- **`references/patterns-english.md`** - Complete English pattern reference
- **`references/patterns-chinese.md`** - 完整中文模式参考
- **`references/evidence.md`** - 证据日志：每条主张的来源/日期/仪器/样本量/失效条件。改规则或被问"凭什么"时读这个

### Evals
- **`evals/evals.json`** - 6 个学术语域回归用例（词表误伤、fractal summary、自造术语、复述稀释、检测器预期管理、中文摘要）。改完 skill 用 `skill-creator` 跑一遍再交付

### Example Files
- **`examples/english.md`** - English before/after examples
- **`examples/chinese.md`** - 中文改写示例

## Best Practices

✅ **DO**:
- Combine pattern detection with soul injection
- Support both English and Chinese
- Use progressive disclosure (core rules here, details in references/)
- Vary sentence structure and rhythm
- Add specific details instead of vague claims
- Use simple constructions (is/are/have) where appropriate

❌ **DON'T**:
- Just remove patterns without adding voice
- Leave stereotypic structures intact
- Over-correct and lose the original meaning
- Ignore language-specific patterns
- Make all sentences the same length

## License

MIT

## Attribution

Based on [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia/Signs_of_AI_writing), maintained by WikiProject AI Cleanup. Merges content from `humanizer`, `humanizer-zh`, and `stop-slop` skills.
