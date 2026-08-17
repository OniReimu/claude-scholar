---
id: PROSE.AI_LEXICON
slug: prose-ai-lexicon
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-background, writing-system-model, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: regex
enforcement: lint_script
params: {tier2_max_per_file: 5, connectives_max_per_file: 4}
conflicts_with: [PROSE.FILLER_PHRASES, PROSE.INTENSIFIERS_ELIMINATION, PROSE.PROMOTIONAL_LANGUAGE, PROSE.VAGUE_QUANTIFIERS]
constraint_type: guardrail
autofix: assisted
lint_patterns:
  - pattern: "(?i)\\b(delv\\w+|leverag\\w+|underscor\\w+|harness\\w*|foster\\w*|streamlin\\w+|bolster\\w*|illuminat\\w+|showcas\\w+|embark\\w*|empower\\w*|unleash\\w*|garner\\w*|resonat\\w+|transcend\\w*|reimagin\\w+|intertwin\\w+|espous\\w+)\\b"
    mode: match
  - pattern: "(?i)\\b(tapestry|realms?|myriad|plethora|beacon|symphony|kaleidoscope|whimsy|intricacies|advancements)\\b"
    mode: match
  - pattern: "(?i)\\b(seamless(ly)?|vibrant|intricate(ly)?|meticulous(ly)?|nuanced|multifaceted|invaluable|indelible|poignant|profound(ly)?|relentless(ly)?|tireless(ly)?|unwavering|unyielding|timeless|ever-evolving|fast-paced|pivotal)\\b"
    mode: match
  - pattern: "(?i)(in today's [a-z-]+ (world|landscape)|stands? as a testament|rich tapestry|navigat\\w+ the (complexities|challenges|landscape)|paving the way for|valuable insights|a deeper understanding of|when it comes to|shed(s|ding)? light on|grappl\\w+ with|at its core|(the|a|this|current|evolving|technological|research|competitive) landscape\\b)"
    mode: match
  - pattern: "(?i)\\b(comprehensive|essential|vital|dynamic|innovative|powerful|notabl[ey]|facilitat\\w+|elucidat\\w+|enhanc\\w+|ensur\\w+|explor\\w+|highlight\\w*|reveal\\w*|engag\\w+|embrac\\w+|insights?|perspectives?|impactful|genuinely|truly|arguably|thought-provoking|interplay|paradigms?)\\b"
    mode: count
    threshold: 5
    threshold_param: tier2_max_per_file
  - pattern: "(?i)\\bIn recent (years|decades),|has (attracted|received|gained|garnered) (increasing|considerable|significant|growing|substantial|widespread) (attention|interest)|With the (rapid|recent|increasing|growing) (development|advancement|growth|progress|proliferation|adoption|rise) of"
    mode: match
  - pattern: "\\b(Moreover|Furthermore|Additionally|In addition),\\s"
    mode: count
    threshold: 4
    threshold_param: connectives_max_per_file
lint_targets: "**/*.tex"
---

## Requirement

**Tier 1（零容忍）**——下列词在正文中一次都不出现，除非它是被讨论对象的字面名称（引文标题、被引方法名、数据集字段名）：

- 动词：delve · leverage · underscore · harness · foster · streamline · bolster · illuminate · showcase · embark · empower · unleash · garner · resonate · transcend · reimagine · intertwine · espouse · shed light on · grapple with
- 名词：tapestry · realm · myriad · plethora · beacon · symphony · kaleidoscope · whimsy · intricacies · advancements
- 形容词/副词：seamless(ly) · vibrant · intricate(ly) · meticulous(ly) · nuanced · multifaceted · invaluable · indelible · poignant · profound(ly) · relentless(ly) · tireless(ly) · unwavering · unyielding · timeless · ever-evolving · fast-paced · pivotal
- 短语：in today's ... world/landscape · stands as a testament · rich tapestry · navigate the complexities · paving the way for · valuable insights · a deeper understanding of · when it comes to · at its core · 比喻义的 the/current/evolving landscape

**Tier 2（单个合法，聚集即违规）**——comprehensive · essential · vital · dynamic · innovative · powerful · notable/notably · facilitate · elucidate · enhance · ensure · explore · highlight · reveal · engage · embrace · insights · perspective · impactful · genuinely · truly · arguably · thought-provoking · interplay · paradigm。单文件累计命中 > `tier2_max_per_file`（默认 5）触发；同一句出现两个即为 cluster，无论总数。

**Formulaic openers（零容忍）**——`In recent years,` / `has attracted increasing attention` / `With the rapid development of`。这些开场白是 AI 起草 Introduction 的默认模板，替换方式是直接从具体的 gap 或结构性事实开篇（"Tabular deep learning has a structural limitation: most models discard feature-type metadata."）。

**Sentence-initial 连接词（密度阈值）**——句首 `Moreover,` / `Furthermore,` / `Additionally,` / `In addition,` 全文合计 ≤ `connectives_max_per_file`（默认 4）。超出说明逻辑靠连接词粘贴而非靠论证顺序承载；修复是重排句序让逻辑自显，不是把 Moreover 换成 Furthermore。

替换原则：优先用你会读出声的那个词（`use` 而非 `leverage`），更优先用本文自身语域里的具体名词（`the retrieval index` 而非 `the ecosystem`）。

## Rationale

这些词是当前 LLM 输出的高频词汇指纹。它们在人类论文里的基频远低于模型输出，因此一段文字里出现三四个就足以让熟悉这套模式的读者（现在包括审稿人）判定为机器生成——即使每一句单独看都无可指摘。

Tier 1 / Tier 2 的分层是必要的：`comprehensive`、`ensure`、`explore` 这类词单独出现时经常是**正确的**词，禁绝它们会把文字逼成不自然的同义词沙拉，那本身是另一种可识别的痕迹。真正的信号是密度，所以 Tier 2 用阈值判定而非黑名单判定。

词表已按学术语域裁剪：`robust`、`optimize`、`trajectory`、`synthesize`、`landscape`（`loss landscape`）、`ecosystem`、`critical`、`significant`、`framework`、`approach`、`challenges` 全部**不入表**——它们在本领域是术语或被其他规则管辖，误伤代价高于收益。

## Check

- **regex 搜索**：Tier 1 四条 `match` 模式；Tier 2 一条 `count` 模式，阈值 `tier2_max_per_file`
- **检查范围**：`.tex` 正文
- **豁免**：
  - 被引文献标题、被引方法/系统的专有名（`\cite` 邻近的原文名称、`\emph{}` 内的他人术语）
  - 术语用法而非比喻用法：`loss landscape` / `energy landscape` / `fitness landscape`（pattern 已限定为带限定词的比喻形态）
  - 直接引语、reviewer comment 原文、附录中的问卷/提示词原文
  - 数学环境与代码块内的标识符
- **人工复核线**：Tier 1 命中一律改写；Tier 2 超阈值时不要逐个替换，先找**最密的那一段**改，密度降下来即可

## Examples

### Pass

```latex
We use the pilot query to rank candidates, which cuts index lookups by 41\%.
The remaining cost comes from the verification pass, not from ranking.
```

### Fail

```latex
We leverage a comprehensive pilot query to harness the intricate interplay
between candidates, showcasing seamless integration that underscores the
pivotal role of ranking in today's retrieval landscape.
```

## Conflicts

- `PROSE.PROMOTIONAL_LANGUAGE` 管情绪化/推销性用词（exciting、revolutionary、groundbreaking），本卡管 AI 高频词汇指纹，两表不交叉；`pivotal` 归本卡，`crucial` 归 PROMOTIONAL 的偏好词裁定，不重复收词
- `PROSE.INTENSIFIERS_ELIMINATION` 管空洞强调副词（very、significantly、remarkably），本卡 Tier 2 不再收这些词
- `PROSE.FILLER_PHRASES` 管句首铺垫短语（in order to、it is important to note that），本卡短语表不与其重叠
- `PROSE.INFORMAL_VOCABULARY` 管口语化下限，本卡管 AI 腔上限，方向相反但可同时触发
