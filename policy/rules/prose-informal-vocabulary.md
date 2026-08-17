---
id: PROSE.INFORMAL_VOCABULARY
slug: prose-informal-vocabulary
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-background, writing-system-model, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: regex
enforcement: lint_script
params: {idiom_allowlist: "from scratch", phrasal_verb_allowlist: "carries out|carried out|rules out|ruled out|points out|pointed out|sets up|set up|holds at|holds with|stems from|follows from|reads off|falls back|falls back to", judgment_adjective_allowlist: "", concrete_metaphor_allowlist: ""}
conflicts_with: [PROSE.VAGUE_QUANTIFIERS, PROSE.REGISTER_PRESERVATION, PROSE.ABSTRACT_AGENCY, PROSE.ELEGANT_VARIATION, PROSE.IDIOM_COLLISION]
constraint_type: guardrail
autofix: safe
lint_patterns:
  - pattern: "\\b(a lot of|lots of)\\b"
    mode: match
  - pattern: "\\b(things|stuff)\\b"
    mode: match
  - pattern: "\\bkind of\\b"
    mode: match
  - pattern: "\\bsort of\\b"
    mode: match
  - pattern: "\\bbigger\\b"
    mode: match
  - pattern: "(?i)\\bfor good\\b(?!\\s+(reason|measure|cause|practice|approximation))"
    mode: match
  - pattern: "(?i)\\bon heads\\b"
    mode: match
  - pattern: "(?i)\\bdrives? off\\b"
    mode: match
  - pattern: "(?i)\\bholds? levers\\b"
    mode: match
  - pattern: "(?i)\\bat all\\b(?!\\s+(times|levels|scales|costs|stages|points|rounds|frequencies|orders|sites|nodes))"
    mode: match
  - pattern: "(?i)\\bin the first place\\b"
    mode: match
  - pattern: "(?i)\\b(ahead of time|up ?front)\\b"
    mode: match
  - pattern: "(?i)\\bat the end of the day\\b"
    mode: match
  - pattern: "(?i)\\b(needless to say|to be fair|more or less|pretty much|a fair bit)\\b"
    mode: match
  - pattern: "(?i)\\b(that said|after all|of course)[,.]"
    mode: match
  - pattern: "(?i)\\bat (a|first) glance\\b"
    mode: match
  - pattern: "(?i)\\bon par\\b"
    mode: match
  - pattern: "(?i)\\bso far\\b"
    mode: match
fix_patterns:
  - find: "\\bbigger\\b"
    replace: "larger"
  - find: "\\bkind of\\b"
    replace: "somewhat"
  - find: "\\bsort of\\b"
    replace: "somewhat"
  - find: "(?i)\\bahead of time\\b"
    replace: "in advance"
  - find: "(?i)\\bup ?front\\b"
    replace: "in advance"
lint_targets: "**/*.tex"
---

## Requirement

**易懂 ≠ 低语域。** 降低阅读门槛的正确手段是把句子结构理顺、把术语定义清楚，不是换上口语措辞。这两件事被系统性地混为一谈，是本卡存在的理由。

学术语域的违规分**五类**，只有第 1 类是词表能覆盖的。

### 类 1 — 习语性状语 / 语气短语（闭集，regex 判定）

`at all` · `in the first place` · `ahead of time` · `up front` · `of course` · `after all` · `at the end of the day` · `that said` · `to be fair` · `needless to say` · `so far` · `on par` · `at a glance` · `at first glance` · `more or less` · `pretty much` · `a fair bit`

**判据**：删掉之后命题是否不变？不变则删；语义有损则换正式对应词（`ahead of time → in advance`，`after the fact → post hoc`，`so far → to date`）。

**allowlist（`params.idiom_allowlist`）**：`from scratch` **不在禁用之列**——`retrain from scratch` 是 ML / unlearning 的固定说法。这类"看起来口语、其实是术语"的项必须进 allowlist，不得机械替换。

### 类 2 — 短语动词顶替拉丁语源动词（半开集，LLM 判定）

| 原 | 建议替换 |
|---|---|
| comes with | entails |
| lands in / sits in | lies at / lies in |
| lifts X past Y | raises X above Y |
| gives up | forfeits |
| folded into | absorbed into |
| bears this out | confirms this |
| stay within reach of | remain close to |
| have no route to | do not provide |
| turns X into Y | makes X … |
| is gone | remains / is removed |

**判据**：本领域是否已有一个单词动词在用？有就用那个。

**allowlist（`params.phrasal_verb_allowlist`）**：`carries out` · `rules out` · `points out` · `sets up` · `holds at` · `holds with` · `stems from` · `follows from` · `reads off` · `falls back`（`falls back to the retrospective regime` 这类是本领域术语）。**不配 allowlist 的短语动词检查会严重误伤。**

**不做 autofix**：正确替换依赖领域上下文。

### 类 3 — 判断性形容词（**必须查 allowlist，不可硬禁**）

`cheap` · `cheaper` · `hard` · `easy` · `easier` · `nice` · `neat` · `tricky` · `messy` · `huge` · `big` · `tiny` · `good` · `bad` · `dramatic` · `impressive` · `striking`

**这是最容易过度执行的一类。** 判据只有一条：**该词是否已经是本领域的既有术语？**

- 是 → **保留**，并在改动记录里注明理由。实测：`cheap unlearning` / `make deletion cheap` 属 Bourtoule 那一支的既有术语，某稿中 `inexpensive` / `low-cost` / `costly` 出现 **0 次**，强改等于新造术语并要求全稿统一，会撞 `PROSE.ELEGANT_VARIATION`
- 否，且稿件他处已用正式对应词 → **改**（实测：`is hard → is difficult`，因为该稿他处已用 `difficult` 两次）
- 无对照物的比较级（`cheaper` 比谁便宜？）→ 那不是语域问题，交 `PROSE.HEDGING_DISCIPLINE` 的无锚点比较

### 类 4 — 谓语位置的具象名词比喻（LLM 判定）

`wall` · `lever` · `knob` · `corner` · `route` · `room` · `story` · `picture`

实测：`the wall is a property of …` → `the obstruction is …`（`obstruction` 是该稿的正式术语）；`the cheaper capacity lever` → `recovers capacity at lower cost`。

**修法优先级**：先换回**本文已有的正式术语**，再考虑重述。这类比喻往往同时是 `PROSE.ELEGANT_VARIATION` 违规——`wall` 就是 `obstruction` 的同义替换。

与 `PROSE.ABSTRACT_AGENCY` 的分工：那条管**抽象名词做施事**（"the estimator carries decades of validation"），本类管**具象名词做表语/宾语的比喻**（"the wall is a property"）。交叉引用，不重复判定。

### 类 5 — 内部工作痕迹动词（LLM 判定）

`buys`（"what quarantine **buys**"、"the invariant object this **buys**"）、`survives`（"the residual **survives** the conditioning"）。这类词把"我们做实验时的体验"写进了论文。

**判据**：把主语换成一个不会有体验的对象，句子是否还成立？不成立即违规。

实测改法：`what quarantine buys` → `separate quarantine from keyed capacity`；`survives the conditioning` → `is not an artifact of the conditioning`。

### 遗留词表（类 1 之外的单词级项）

| 禁用 | 替代 |
|------|------|
| a lot of / lots of | 具体数字，或 many（"many studies" 类组合会触发 `PROSE.VAGUE_QUANTIFIERS`，最好直接量化） |
| things / stuff | factors / components / elements；data / material |
| get | obtain / achieve / acquire |
| kind of / sort of | 删除，或用 approximately / somewhat |
| bigger | larger（`smaller` **不禁**，它本身就是规范的比较级学术用词） |
| for good | permanently（`for good reason/measure/cause` 是合法搭配，pattern 已排除） |
| on heads | on a positive draw（或该实验实际的事件名） |
| drives off | deters / discourages |
| holds levers | imposes a cost / has instruments（并查 `PROSE.ABSTRACT_AGENCY`） |

## Rationale

口语化词汇降低论文的正式程度，在同行评审中会被视为不够严谨。但本卡真正的教训是**范围与工具的匹配**。

**实证**（CISU / ct_unlearning 稿件，NDSS 投稿，20 页，2026-08-17，逐节 anti-AI 清理）：抓出约 30 处口语语域问题，把其中 **26 条**代表性字符串喂给本卡当时的 9 条 lint pattern，命中 **0/26**。

关键在于这批文本的性质：**稿件不是 agent 生成的**，此前已跑过 `writing-anti-ai`，tier-1 AI 词表、em-dash、cleft、negation-contrast 均已清零。**正因为词表层早就干净，这一轮暴露出来的才全是词表抓不到的那一类**——30 处里 **29 处是多词构造**（习语 / 短语动词 / 名词性比喻），不是单词。只看词表命中率，这份稿子会被判为"已达标"。

本卡当时的正文写着「本卡负责：词表层命中，**以及未被本次 pass 改动的文本**」——它**声明**了后一块领地，但工具只有 9 条单词级 regex，其中 `for good` / `on heads` / `drives off` 三条还是从上一次实测案例逆向补进来的单例，没有泛化。**声明范围与工具能力不匹配，等于把一块地圈起来无人看守。** 五类分类表就是补上这个缺口：类 1 交给 regex，类 2–5 交给带判据和 allowlist 的 LLM 判定。

用户给出的判据值得原样记下：

> 「`at all` 这种词不该出现在论文里。我们要易懂的文字不代表用这种 verbal 的语言。」

同一轮里有一处 `at all` 是 agent 在上一步「让句子更好懂」时**主动加进去的**——即 `PROSE.REGISTER_PRESERVATION` 描述的失效模式，但它发生在一次 **anti-AI pass** 里，而不是压缩 pass 里。

## Check

- **regex（类 1 + 遗留词表）**：`policy/lint.sh` 自动执行，见 `lint_patterns`
- **LLM 判定（类 2–5）**：无 regex，按各类判据逐处裁决，**先查该类的 `params` allowlist**
- **提取正文的正确方法**：见 `policy/references/tex-prose-extraction.md`。不要手搓 `.tex` 扫描器——用 `line.split('%')[0]` 剔注释会在 `$95\%$` 处截断整行后半段，用 `re.sub(r'\$[^$]*\$','',t)` 剔数学会在 `$` 计数为奇数时吞掉成段散文，逐行扫描会漏掉被硬换行劈开的短语。这三种写法都会产生**假的"已清零"结论**
- **排除**：直接引语（quote 环境）、被引文献标题、注释掉的历史草稿（`rg` 直扫原始 `.tex` 会命中它们）

**修法通则**（实测有效）：替换词优先从**稿件别处已有的措辞**里取。某轮 `hard→difficult`、`folded into→absorbed`、`sit in→lie in`、`after the fact→post hoc`、`up front→in advance` 五处的替换词**全部**取自该稿他处，没有一个是新造的。这同时满足 `PROSE.ELEGANT_VARIATION`。

## Examples

### Pass

```latex
% 类 1 allowlist：术语，不是口语
We retrain the model from scratch as the exact-deletion reference.

% 类 2 allowlist：本领域标准短语动词
Theorem 3 rules out unilateral deviations; the bound follows from Lemma 2,
and the estimator falls back to the retrospective regime when $t < t_0$.

% 类 3：既有术语，保留并注明
Cheap unlearning is the design goal: deletion must cost less than retraining.

% 修法通则：替换词取自稿件他处
The residual is not an artifact of the conditioning.
```

### Fail

```latex
% 类 1：习语性状语
The frozen-state baseline does not help at all, and in the first place
the operator must decide ahead of time which segments to quarantine.

% 类 2：短语动词顶替拉丁语源动词
Segment-level deletion comes with a capacity cost, and the guarantee
gives up its exact form once the update is folded into the live backbone.

% 类 4：具象名词比喻（同时是 ELEGANT_VARIATION，wall = obstruction）
The wall is a property of the continuous-time dynamics.

% 类 5：内部工作痕迹动词
What quarantine buys is an invariant object that survives the conditioning.
```

## Conflicts

- `PROSE.REGISTER_PRESERVATION` 判 **diff**（一次编辑 pass 改动过的 span 是否降了语域）；本卡判 **document**（作者原文本身的语域），且只在有明确判据和 allowlist 的五类范围内。**不要因为本卡而扩大 REGISTER_PRESERVATION 的范围**——它的 diff-only 收窄有 precision 0.00 / recall 0.00 的实测支撑
- `PROSE.ABSTRACT_AGENCY` 管抽象名词做施事；本卡类 4 管具象名词做表语/宾语的比喻
- `PROSE.ELEGANT_VARIATION` 是类 3 与类 4 的**修法约束**：换词前先确认稿件他处用的是什么，避免新造术语
- `PROSE.HEDGING_DISCIPLINE` 拥有无锚点比较级（`cheaper` 比谁便宜），那不是语域问题
- `PROSE.VAGUE_QUANTIFIERS` 拥有 `several` / `various` / `a number of`，不得重复报告
- `PROSE.IDIOM_COLLISION` 拥有"技术短语与常用习语同形"（`a fair bit` 指无偏比特却被读成"相当多"）。本卡把 `a fair bit` 当口语量词收在类 1；当它是技术义时交由那条规则处理
