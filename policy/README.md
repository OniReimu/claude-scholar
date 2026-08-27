# Policy Engine — 论文写作规则注册中心

## ⚠️ MANDATORY: Author Style Guide

**所有论文写作任务开始前，必须先读 `policy/style-guide.md`。**
这是作者的个人写作风格指纹（基于 Pre-GPT 时期论文分析），与 `policy/rules/` 同级权威。
不读 style-guide 写出的文字不是作者的风格。引用标记：`<!-- style:author-voice -->`

## 权威定义优先级

```
policy/style-guide.md (整体性写作风格) ≡ policy/rules/ (单条可判定规则) > CLAUDE.md/AGENTS.md (指引入口) > skills/*/SKILL.md (上下文引用)
```

**`style-guide.md` 和 `rules/` 是同级权威，二者缺一不可。**

区分标准：
- **`style-guide.md`** — 整体性写作风格身份：偏好动词、句式模板、段落组织、叙事逻辑。写作时整体浸入，无法拆成单条 pass/fail
- **`rules/`** — 单条可判定的规则（1 rule = 1 file）：每条有明确的 pass/fail 判定标准，可用 regex 或 LLM 逐条检查

技能文件通过 `<!-- policy:RULE_ID -->` 标记引用规则，通过 `<!-- style:author-voice -->` 标记引用风格指南。M3 已完成去重，`policy/rules/` 为唯一真相源。

---

## Rule Card 规范

**1 rule = 1 file**，位于 `policy/rules/`，文件名 kebab-case。

### Frontmatter（必填字段）

```yaml
---
id: FIG.NO_IN_FIGURE_TITLE          # 唯一 ID，大写点分隔
slug: fig-no-in-figure-title         # 文件名（kebab-case）
severity: error | warn               # 统一用 warn（不用 warning）
locked: true | false                 # locked=true 时 severity 和 params 均不可被 profile 覆盖
layer: core | domain | venue         # M2 新增：规则分层
artifacts: [figure, equation, text, table, code, bibtex]
phases: [ideation, writing-intro, writing-background, writing-system-model,
         writing-methods, writing-experiments, writing-conclusion, self-review,
         revision, camera-ready]
domains: [core] | [security, hci, se, is]
venues: [all] | [neurips, icml, iclr, ccs, usenix, ndss, sp, chi, icse, fse, ase, misq, isr, ...]
check_kind: regex | ast | llm_semantic | llm_style | manual
enforcement: doc | lint_script       # lint_script=由 policy/lint.sh 执行（默认 regex；个别规则可有内置脚本检查）
constraint_type: guardrail | guidance  # 规则语义：guardrail=约束性（不要做 X）, guidance=构建性（要做 Y）
autofix: safe | assisted | none      # 修复策略：safe=无人值守替换, assisted=展示 diff 待确认, none=需重写
params: {}                           # 可选，profile 可覆盖（locked=false 时）
conflicts_with: []                   # 可选
lint_patterns: []                    # M2 新增：机器可读 regex（仅 check_kind=regex 时）
lint_targets: ""                     # M2 新增：glob pattern 指定检查目标
fix_patterns: []                     # 可选：自动修复映射（仅 autofix=safe 时），条目含 find/replace
---
```

### Body Sections（必须）

1. `## Requirement` — 祈使句，可执行的约束声明
2. `## Rationale` — 为什么有这条规则（帮 LLM 在边界情况判断）
3. `## Check` — 验证方法（LLM 检查要点 / regex pattern / lint 命令）
4. `## Examples` — **Pass** 和 **Fail** 各至少一个，用代码块

### Body Sections（可选）

5. `## Conflicts` — 与其他规则的张力
6. `## Template Ref` — 指向模板文件的链接

### 字段值说明

- **severity**: `error`（必须修复）| `warn`（建议修复）
- **locked**: `true` 时 profile 不可覆盖 severity 和 params
- **layer**: `core`（所有论文必须遵守）| `domain`（领域/风格相关）| `venue`（会议/期刊特定）
- **enforcement**: `doc`（仅文档约束）| `lint_script`（已有自动检查脚本）
- **constraint_type**: `guardrail`（约束性——"不要做 X"，如禁用特定词汇/模式）| `guidance`（构建性——"要做 Y"，如要求特定结构/内容）。与 `check_kind` 正交，不可互推
- **autofix**: `safe`（有封闭替换表，零已知 exception，可无人值守执行）| `assisted`（可生成 diff，但有已知 exception，必须展示 diff 待确认）| `none`（需要理解上下文/重写，不可自动修复）。与 `constraint_type` 独立判断
- **params**: 声明所有可覆盖参数的默认值，Profile override 引用的 param key 必须在此存在
- **lint_patterns**: 机器可读 regex 模式列表（仅 `check_kind: regex` 时填写），每项含：
  - `pattern`: 正则表达式
  - `mode`: `match`（匹配即违规）| `count`（超阈值违规）| `negative`（缺失即违规）
  - `threshold`: count 模式时的阈值（可选）
  - `threshold_param`: 关联的 `params` 键名（可选，Profile 可通过 `params.<key>` 覆盖阈值）
- **lint_targets**: glob pattern 指定检查目标文件（如 `**/*.tex`、`**/*.bib`、`**/*.py`）
- **deprecated_by**（可选）: 该规则已被某 skill/规则接管时填写继任者名（如 `scientific-figure-making`）。填写后规则卡保留作历史参考，但引用它的 skill 必须同步（见下节「规则弃用 / 变更流程」）

---

## 规则弃用 / 变更流程

当一条规则被弃用（打 `deprecated_by:`）或其 `params` 值发生变更（如 `FIG.FONT_GE_24PT` 的 24pt 阈值改为自适应）时，**规则卡与引用它的 skill 会漂移**——skill 的 quick-ref 表往往仍在断言旧值。改规则时必须同步：

1. **规则卡**：打 `deprecated_by:` 或改 `params`，并在 body 顶部加 `> **⚠️ Deprecated**` banner 说明继任者做什么。
2. **同步引用**：`grep -rn 'policy:<RULE_ID>' skills/ commands/` 找出所有 `<!-- policy:X -->` 引用，逐处更新措辞——弃用类加 inline successor note（如「已弃用，交 `scientific-figure-making`」），值变更类改成新值或转指继任者。
3. **校验**：跑 `policy/validate.sh`。**Section 8b（Deprecated-Rule Citation Acknowledgment）** 会列出所有引用弃用规则却缺 successor note 的位置。这是 **WARNING（非阻塞）**——部分弃用规则合法保留为写作指引（如 `FIG.SELF_CONTAINED_CAPTION` 交 `writing-convention`），逐条确认即可，不必强制加注。

---

## Phase 词汇表

| Phase | 描述 |
|-------|------|
| `ideation` | 研究构思、选题、大纲、Figure 1 |
| `writing-intro` | Introduction（问题陈述、贡献列表、叙事开篇） |
| `writing-background` | Background & Related Work |
| `writing-system-model` | System Model |
| `writing-methods` | Methods / Our Approach |
| `writing-experiments` | 实验计划、执行、分析、撰写 |
| `writing-conclusion` | Conclusion |
| `self-review` | 论文自审 |
| `revision` | Rebuttal / 修改 |
| `camera-ready` | 终稿准备 |

---

## Step→Phase 映射表

以 `ml-paper-writing` workflow 为参考基准：

| ml-paper-writing Step | Phase |
|----------------------|-------|
| Step 1-3 (选题/大纲/Figure 1) | ideation |
| Step 3-4 (Abstract / Introduction) | writing-intro |
| Step 5 (Background & Related Work) | writing-background |
| Step 6 (System Model) | writing-system-model |
| Step 7 (Methods / Our Approach) | writing-methods |
| Step 8a-8c (实验计划/执行/分析) | writing-experiments |
| Step 9 (Write Experiments Section) | writing-experiments |
| Step 10 (Conclusion) | writing-conclusion |
| Step 11 (Self-review) | self-review |
| Rebuttal / Camera-ready | revision, camera-ready |

其他 workflow（survey、workshop paper）可建立各自的 Step→Phase 映射，Phase 是稳定的跨 workflow 抽象层。

---

## SoK Scope（v1）

SoK（Systematization of Knowledge）在 v1 中不新增 frontmatter `scope` 字段，而是通过 profile 激活：

- 示例 profile：`policy/profiles/security-sok-sp.md`
- 激活方式：`policy/lint.sh --profile <profile-file>`
- 当前限制：一次仅加载一个 profile（无 inheritance/composition）

SoK 规则集合（语义规则）：

| Rule ID | 主要映射 Phase |
|---------|-----------------|
| `SOK.TAXONOMY_REQUIRED` | `writing-background` |
| `SOK.METHODOLOGY_REPORTING` | `writing-methods` |
| `SOK.BIG_TABLE_REQUIRED` | `writing-experiments` |
| `SOK.RESEARCH_AGENDA_REQUIRED` | `writing-conclusion` |

> 规则卡元数据采用现有 schema：`layer: domain`、`domains: [security, se, is]`、`venues: [all]`；并在 Rationale 中明确“仅在 SoK profile 激活时生效”。

---

## Experiment Status Scope（v1）

实验结果默认状态为 ACTUAL（无需额外声明）。
仅在使用占位/合成/虚拟结果时触发异常状态规则：

| Rule ID | 作用 |
|---------|------|
| `EXP.FABRICATED_RESULTS_CAPTION_DISCLOSURE` | 在 figure/table caption 中红色大写披露 fabricated 状态 |
| `EXP.RESULTS_STATUS_DECLARATION_REQUIRED` | 在对应 `\subsubsection` 开头声明 `% [FABRICATED] ...` 状态注释 |

> 轻/重任务判定属于 workflow 决策，不作为 policy rule schema 字段；policy 只约束论文产出物披露合规性。

---

## Rule ID Registry

| Rule ID | slug | layer | severity | locked | enforcement | constraint_type | autofix |
|---------|------|-------|----------|--------|-------------|-----------------|---------|
| FIG.NO_IN_FIGURE_TITLE | fig-no-in-figure-title | core | error | true | doc | guidance | none |
| FIG.FONT_GE_24PT | fig-font-ge-24pt (deprecated → scientific-figure-making) | core | error | false | doc | guidance | none |
| FIG.ONE_FILE_ONE_FIGURE | fig-one-file-one-figure | core | error | true | doc | guidance | none |
| FIG.COLORBLIND_SAFE_PALETTE | fig-colorblind-safe-palette (deprecated → scientific-figure-making) | core | warn | false | doc | guidance | none |
| FIG.SELF_CONTAINED_CAPTION | fig-self-contained-caption (deprecated → paper-self-review) | core | warn | false | doc | guidance | none |
| FIG.SYSTEM_OVERVIEW_ASPECT_RATIO_GE_2TO1 | fig-system-overview-aspect-ratio-ge-2to1 (deprecated → paper-figure-generator) | core | error | true | doc | guidance | none |
| FIG.VECTOR_FORMAT_REQUIRED | fig-vector-format-required (deprecated → scientific-figure-making) | core | error | false | lint_script | guidance | none |
| FIG.EXPERIMENT_SUBFIGURE_LAYOUT | fig-experiment-subfigure-layout | core | warn | false | doc | guardrail | none |
| FIG.HEATMAP_LABEL_ABBREVIATION | fig-heatmap-label-abbreviation | core | warn | false | doc | guidance | none |
| FIG.COLUMN_WIDTH_JUSTIFICATION | fig-column-width-justification | core | warn | false | doc | guardrail | none |
| FIG.RESEARCH_GAP_TEASER | fig-research-gap-teaser | core | warn | false | doc | guidance | none |
| TABLE.BOOKTABS_FORMAT | table-booktabs-format | core | warn | false | lint_script | guardrail | assisted |
| TABLE.DIRECTION_INDICATORS | table-direction-indicators | core | warn | false | doc | guidance | none |
| TABLE.RESIZEBOX_COLUMN_FIT | table-resizebox-column-fit | core | warn | false | doc | guidance | assisted |
| TABLE.DIMENSION_BUDGET | table-dimension-budget | core | warn | false | doc | guidance | none |
| TABLE.FULLWIDTH_FONT_DENSITY | table-fullwidth-font-density | core | warn | false | doc | guardrail | none |
| LATEX.CMARK_XMARK_PMARK_MACROS | latex-cmark-xmark-pmark-macros | core | error | false | doc | guidance | none |
| LATEX.EQ.DISPLAY_STYLE | latex-eq-display-style | core | error | true | lint_script | guardrail | none |
| LATEX.VAR.LONG_TOKEN_USE_TEXT | latex-var-long-token-use-text | core | warn | false | doc | guidance | none |
| LATEX.NOTATION_CONSISTENCY | latex-notation-consistency | core | error | true | doc | guidance | none |
| REF.CROSS_REFERENCE_STYLE | ref-cross-reference-style | core | warn | false | doc | guidance | none |
| REF.WOVEN_CROSS_REFERENCE | ref-woven-cross-reference | core | warn | false | doc | guardrail | assisted |
| PAPER.SECTION_HEADINGS_MAX_6 | paper-section-headings-max-6 | core | error | false | lint_script | guidance | none |
| PAPER.CONCLUSION_SINGLE_PARAGRAPH | paper-conclusion-single-paragraph | core | warn | false | doc | guidance | none |
| PAPER.OUTCOME_LOGIC | paper-outcome-logic | core | warn | false | doc | guidance | none |
| CITE.VERIFY_VIA_API | cite-verify-via-api | core | error | true | lint_script | guidance | none |
| CITE.CLAIM_SUPPORT_REQUIRED | cite-claim-support-required | core | warn | false | lint_script | guidance | none |
| EXP.ERROR_BARS_REQUIRED | exp-error-bars-required | core | error | false | doc | guidance | none |
| EXP.TAKEAWAY_BOX | exp-takeaway-box | core | warn | false | doc | guidance | none |
| EXP.ABLATION_IN_RESULTS | exp-ablation-in-results | core | warn | false | doc | guidance | none |
| EXP.RESULTS_SUBSECTION_STRUCTURE | exp-results-subsection-structure | core | warn | false | doc | guidance | none |
| EXP.FABRICATED_RESULTS_CAPTION_DISCLOSURE | exp-fabricated-results-caption-disclosure | core | error | false | doc | guidance | none |
| EXP.RESULTS_STATUS_DECLARATION_REQUIRED | exp-results-status-declaration-required | core | warn | false | doc | guidance | none |
| EXP.MULTIRUN_AGGREGATE_CONSISTENCY | exp-multirun-aggregate-consistency | core | error | false | doc | guardrail | none |
| EXP.EXPERIMENT_ROLE | exp-experiment-role | core | warn | false | doc | guidance | none |
| REPRO.RANDOM_SEED_DOCUMENTATION | repro-random-seed-documentation | core | error | false | doc | guidance | none |
| REPRO.COMPUTE_RESOURCES_DOCUMENTED | repro-compute-resources-documented | core | warn | false | doc | guidance | none |
| SUBMIT.SECTION_NUMBERING_CONSISTENCY | submit-section-numbering-consistency | core | warn | false | lint_script | guidance | none |
| SOK.TAXONOMY_REQUIRED | sok-taxonomy-required | domain | error | false | doc | guidance | none |
| SOK.METHODOLOGY_REPORTING | sok-methodology-reporting | domain | warn | false | doc | guidance | none |
| SOK.BIG_TABLE_REQUIRED | sok-big-table-required | domain | error | false | doc | guidance | none |
| SOK.RESEARCH_AGENDA_REQUIRED | sok-research-agenda-required | domain | error | false | doc | guidance | none |
| SE.RESEARCH_QUESTIONS_EXPLICIT | se-research-questions-explicit | domain | error | false | doc | guidance | none |
| SE.RQ_SECTION_BINDING | se-rq-section-binding | domain | warn | false | doc | guidance | none |
| SE.THREATS_TO_VALIDITY_STRUCTURED | se-threats-to-validity-structured | domain | error | false | doc | guidance | none |
| SE.ACTIONABLE_IMPLICATIONS | se-actionable-implications | domain | warn | false | doc | guidance | none |
| PROSE.CRYPTO_CONSTRUCTION_TEMPLATE | prose-crypto-construction-template | domain | warn | false | doc | guidance | none |
| PROSE.PSEUDOCODE_ABSTRACTION | prose-pseudocode-abstraction | domain | warn | false | doc | guidance | none |
| PROSE.INTENSIFIERS_ELIMINATION | prose-intensifiers-elimination | domain | warn | false | lint_script | guardrail | assisted |
| PROSE.EM_DASH_RESTRICTION | prose-em-dash-restriction | domain | warn | false | lint_script | guardrail | assisted |
| PROSE.FILLER_PHRASES | prose-filler-phrases | domain | warn | false | lint_script | guardrail | safe |
| PROSE.COLON_LIST_OVERUSE | prose-colon-list-overuse | domain | warn | false | lint_script | guardrail | none |
| PROSE.RULE_OF_THREE | prose-rule-of-three | domain | warn | false | doc | guidance | none |
| PROSE.OVER_DEFENSIVE | prose-over-defensive | domain | warn | false | doc | guidance | none |
| PROSE.SELF_UNDERMINING | prose-self-undermining | domain | warn | false | lint_script | guardrail | none |
| PROSE.PROMOTIONAL_LANGUAGE | prose-promotional-language | domain | warn | false | lint_script | guardrail | assisted |
| PROSE.FORMATTING_RESTRAINT | prose-formatting-restraint | domain | warn | false | doc | guidance | none |
| PROSE.NO_INTERNAL_PROVENANCE | prose-no-internal-provenance | core | error | false | lint_script | guardrail | assisted |
| PROSE.TENSE_CONSISTENCY | prose-tense-consistency | domain | warn | false | doc | guidance | none |
| PROSE.ABBREVIATION_FIRST_USE | prose-abbreviation-first-use | domain | warn | false | doc | guidance | none |
| PROSE.VAGUE_QUANTIFIERS | prose-vague-quantifiers | domain | warn | false | lint_script | guardrail | assisted |
| PROSE.SENTENCE_LENGTH | prose-sentence-length | domain | warn | false | lint_script | guardrail | none |
| PROSE.PARAGRAPH_TOPIC_SENTENCE | prose-paragraph-topic-sentence | domain | warn | false | doc | guidance | none |
| PROSE.SUBSECTION_COMPLETENESS | prose-subsection-completeness | domain | warn | false | doc | guidance | none |
| PROSE.EQUATION_EXPLANATION | prose-equation-explanation | domain | warn | false | doc | guidance | none |
| PROSE.INFORMAL_VOCABULARY | prose-informal-vocabulary | domain | warn | false | lint_script | guardrail | safe |
| PROSE.HEDGING_DISCIPLINE | prose-hedging-discipline | domain | warn | false | doc | guidance | none |
| PROSE.NUMBER_EXPRESSION | prose-number-expression | domain | warn | false | doc | guidance | none |
| PROSE.ELEGANT_VARIATION | prose-elegant-variation | domain | warn | false | doc | guidance | none |
| PROSE.RELATED_WORK_EVOLUTION | prose-related-work-evolution | domain | warn | false | doc | guidance | none |
| PROSE.COPULA_DODGE | prose-copula-dodge | domain | warn | false | lint_script | guardrail | safe |
| PROSE.NEGATIVE_PARALLELISM | prose-negative-parallelism | domain | warn | false | lint_script | guardrail | assisted |
| PROSE.SUPERFICIAL_ING_SUFFIX | prose-superficial-ing-suffix | domain | warn | false | lint_script | guardrail | assisted |
| PROSE.DESPITE_DISMISSAL | prose-despite-dismissal | domain | warn | false | lint_script | guardrail | none |
| PROSE.VAGUE_ATTRIBUTIONS | prose-vague-attributions | domain | warn | false | lint_script | guardrail | safe |
| PROSE.RHETORICAL_SELF_ANSWER | prose-rhetorical-self-answer | domain | warn | false | lint_script | guardrail | none |
| PROSE.CLEFT_CONSTRUCTION | prose-cleft-construction | domain | warn | false | lint_script | guardrail | none |
| PROSE.HYPOTHETICAL_FOIL | prose-hypothetical-foil | domain | warn | false | lint_script | guardrail | none |
| PROSE.ABSTRACT_AGENCY | prose-abstract-agency | domain | warn | false | lint_script | guardrail | none |
| PROSE.ANAPHORA_ABUSE | prose-anaphora-abuse | domain | warn | false | doc | guidance | none |
| PROSE.GERUND_FRAGMENT_LITANY | prose-gerund-fragment-litany | domain | warn | false | doc | guidance | none |
| PROSE.SHORT_PUNCHY_FRAGMENTS | prose-short-punchy-fragments | domain | warn | false | doc | guidance | none |
| PROSE.RHYTHM_VARIANCE | prose-rhythm-variance | domain | warn | false | doc | guidance | none |
| PROSE.ANNOUNCEMENT_SENTENCE | prose-announcement-sentence | domain | warn | false | doc | guidance | none |
| PROSE.THEATRICAL_SPLIT | prose-theatrical-split | domain | warn | false | doc | guidance | none |
| PROSE.UNICODE_ARROWS | prose-unicode-arrows | domain | warn | false | lint_script | guardrail | safe |
| PROSE.TRAILING_AFTERTHOUGHT | prose-trailing-afterthought | domain | warn | false | lint_script | guardrail | none |
| PROSE.COMMA_OVERUSE | prose-comma-overuse | domain | warn | false | lint_script | guardrail | none |
| PROSE.MIDSENTENCE_COLON | prose-midsentence-colon | domain | warn | false | lint_script | guardrail | none |
| PROSE.NEGATION_CONTRAST | prose-negation-contrast | domain | warn | false | lint_script | guardrail | none |
| PROSE.CAUSAL_CONNECTIVE | prose-causal-connective | domain | warn | false | lint_script | guidance | none |
| PROSE.AI_LEXICON | prose-ai-lexicon | domain | warn | false | lint_script | guardrail | assisted |
| PROSE.FRACTAL_SUMMARY | prose-fractal-summary | domain | warn | false | lint_script | guardrail | assisted |
| PROSE.INVENTED_CONCEPT_LABEL | prose-invented-concept-label | domain | warn | false | doc | guardrail | none |
| PROSE.ADHOC_COMPOUND_MODIFIER | prose-adhoc-compound-modifier | domain | warn | false | lint_script | guidance | none |
| PROSE.RESTATEMENT_DILUTION | prose-restatement-dilution | domain | warn | false | doc | guardrail | none |
| PROSE.SEMANTIC_IDLING | prose-semantic-idling | domain | warn | false | doc | guardrail | none |
| PROSE.REGISTER_PRESERVATION | prose-register-preservation | domain | warn | false | doc | guardrail | none |
| PROSE.IDIOM_COLLISION | prose-idiom-collision | domain | warn | false | doc | guardrail | none |
| ETHICS.LIMITATIONS_SECTION_MANDATORY | ethics-limitations-section-mandatory | venue | error | false | doc | guidance | none |
| ANON.DOUBLE_BLIND_ANONYMIZATION | anon-double-blind-anonymization | venue | error | true | doc | guidance | none |
| SUBMIT.PAGE_LIMIT_STRICT | submit-page-limit-strict | venue | error | false | doc | guidance | none |
| BIBTEX.CONSISTENT_CITATION_KEY_FORMAT | bibtex-consistent-citation-key-format | venue | warn | false | lint_script | guardrail | none |

---

## 词汇类规则归属表（Lexicon Ownership）

八条规则都在管"哪些词不能用"，词表**互斥分工**如下。新增禁用词时先查此表，归属唯一；一个 surface form 只允许出现在一条规则的 lint_patterns 里。

| 规则 | 管辖范围 | 例词 |
|------|---------|------|
| `PROSE.AI_LEXICON` | AI 高频词汇指纹（tier-1/tier-2）、formulaic openers、句首连接词密度 | delve, pivotal, tapestry; "In recent years,"; Moreover 密度 |
| `PROSE.PROMOTIONAL_LANGUAGE` | 推销性/情绪化用词 + novelty padding | groundbreaking, revolutionary; novel 超频, "for the first time" |
| `PROSE.INTENSIFIERS_ELIMINATION` | 空洞强调**副词** | very, extremely, significantly |
| `PROSE.FILLER_PHRASES` | 可删除的铺垫**短语** | "in order to", "it is worth noting that" |
| `PROSE.INFORMAL_VOCABULARY` | 口语化下限，**五类分类表**：习语状语(regex) / 短语动词 / 判断形容词 / 具象比喻 / 工作痕迹动词（后四类 LLM 判定 + allowlist） | a lot of, at all, in the first place, comes with, cheap, wall, buys |
| `PROSE.IDIOM_COLLISION` | 技术短语与常用习语同形（歧义，非语域） | a fair bit, on the order of, significant |
| `PROSE.VAGUE_QUANTIFIERS` | 模糊量词（仅「量词+文献名词」组合与恒模糊短语，不抓裸词） | "many studies", "a wide range of", "extensive experiments" |
| `PROSE.SELF_UNDERMINING` | 自我削弱词汇（情绪副词与自贬式比较措辞）；边界：`PROSE.OVER_DEFENSIVE` 管辩护的结构与放置，本条只管词级措辞 | unfortunately, regrettably, admittedly, merely, falls short, lags behind |

**两条硬性不变量**（由 `validate.sh` Section 5c 机器检查第 2 条）：

1. **词表互斥**：同一 token 不得出现在两条规则的 lint_patterns 中（重叠 = 同一违规双报，噪声）
2. **修复无循环**：任何规则的 `fix_patterns` replace 产物（以及 Requirement 替代表建议的词）不得命中其他规则的 lint pattern——否则应用 A 的修复会制造 B 的违规

---

## 测试套件

四套，问的是四个不同的问题。改规则卡或改 skill 后全部跑一遍。

| 脚本 | 问的问题 |
|------|----------|
| `policy/validate.sh` | Registry 不变量成立吗？（字段完整、conflicts_with 互相声明、phase 词汇合规、无孤儿规则、pattern 块语法合法） |
| `policy/test-lint.sh` | lint **机制**对吗？（flag 过滤、fix 发射与回验、退出码、边界情形） |
| `policy/test-corpus.sh` | 规则报在**对的句子**上吗？逐 case 报漏报与误报 |
| `policy/test-referrals.sh` | 转诊图完整吗？（目的地存在 / 目的地真的执行那条规则 / 转诊写明了最小操作） |
| `policy/test-pipeline.sh` | 两阶段流水线（arch-review → anti-ai）**端到端**做对了吗？该删的删了、该留的留了、顺序没颠倒 |

前两套测的是**结构与机制**，都不读散文；后两套补的正是这个洞。最后一套测的不是任何单条规则，而是**规则之间的协作**。

**流水线验收**（`policy/test-pipeline/`，格式见该目录 README）：每个 fixture 是一份埋了缺陷的 draft + 一份**运行前封存**的答案。agent 那一半是手动的（`PROMPT.md`），脚本这一半是确定性的：`GONE` / `KEPT` / `VERBATIM` 三个断言词，外加一条"输出里的每个数字都必须在输入里出现过"的伪造检查。CI 跑不了 agent，因此它给**录制的参考输出**打分——规则改动若会改变流水线的删留行为，这里会红，fixture 必须被有意识地重跑重录，而不是悄悄漂移。

**必须先跑反向控制**：`./test-pipeline.sh <case> <case>/input.tex`（未加工的原稿）必须**大声失败**。只通过过的 harness 不携带信息——本仓库第一个 fixture 就有两条断言的标签声称能抓顺序颠倒，喂进"只做词级编辑"的输出后发现它们照样通过，真正的守卫是段落自己的主题短语。**标签会朝着你的本意漂移，反向控制才告诉你实际写下的是哪个。**

**语料测试**（`policy/test-corpus/`，格式见该目录 README）：precision case 与 recall case 同等重要，甚至更重要——过度触发是本引擎的失效模式，一条把每个 threat-model 句子都报出来的规则会被作者关掉，代价远大于它带来的收益。新增或收紧任何 `lint_patterns` 时，**同时补一条会误命中的近似句**，把边界钉死。

**引擎分歧**：`grep -P` 与 perl 不可互换。macOS 解析到 perl，Linux CI 解析到 GNU grep，因此只在一侧跑的测试看不到另一侧的缺陷。`LINT_ENGINE=ggrep|grep|perl` 强制指定，`test-corpus.sh --engine <name>` 用它在同一份语料上对比两个引擎。

**转诊图**（`test-referrals.sh`）：一条转诊边 = 某条规则的 marker 管辖范围内，出现了显式转诊动词（转 / 归 / 交 / refer to）加一个反引号包住的真实 skill 名。仅仅**提到**另一个 skill 不算转诊。R2 的豁免必须带理由登记在脚本里；vendor submodule symlink skill 自动豁免（marker 不写进 vendor 树）。

## 与 `rules/` 目录的边界

- `rules/` = 开发运维规则（代码风格、安全、agent 编排、实验可复现性）
- `policy/` = 论文写作规则（LaTeX 格式、图表规范、论文结构）
- `rules/experiment-reproducibility.md` 保留原位，profile 中用 Cross-References 引用

---

## 去重状态（M3）

M3 清理了 CLAUDE.md、AGENTS.md 和 skill 文件中的重复规则文本。
`policy/rules/` 现在是所有论文写作规则的唯一真相源。

**引用约定**：
- **CLAUDE.md / AGENTS.md**: 仅包含 policy engine 入口指引 + 强约束语句
- **SKILL.md**: 工作流内使用 one-liner + `<!-- policy:RULE_ID -->` 标记
- **references/*.md**: 使用 blockquote pointer 指向 `policy/rules/`
- **硬规则**：只删规则定义重复文本，不删模板示例/可执行参数/具体颜色值
