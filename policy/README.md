# Policy Engine — 论文写作规则注册中心

## 权威定义优先级

```
policy/rules/ (单一真相源) > CLAUDE.md/AGENTS.md (指引入口) > skills/*/SKILL.md (上下文引用)
```

技能文件通过 `<!-- policy:RULE_ID -->` 标记引用规则。M3 已完成去重，`policy/rules/` 为唯一真相源。

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
phases: [ideation, writing-background, writing-system-model, writing-methods,
         writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core] | [security, hci, se, is]
venues: [all] | [neurips, icml, iclr, ccs, usenix, ndss, sp, chi, icse, fse, ase, misq, isr, ...]
check_kind: regex | ast | llm_semantic | llm_style | manual
enforcement: doc | lint_script       # doc=M1 无独立脚本, lint_script=M1 已有脚本。lint.sh 按 check_kind=regex 运行，不区分 enforcement
params: {}                           # 可选，profile 可覆盖（locked=false 时）
conflicts_with: []                   # 可选
lint_patterns: []                    # M2 新增：机器可读 regex（仅 check_kind=regex 时）
lint_targets: ""                     # M2 新增：glob pattern 指定检查目标
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
- **params**: 声明所有可覆盖参数的默认值，Profile override 引用的 param key 必须在此存在
- **lint_patterns**: 机器可读 regex 模式列表（仅 `check_kind: regex` 时填写），每项含：
  - `pattern`: 正则表达式
  - `mode`: `match`（匹配即违规）| `count`（超阈值违规）| `negative`（缺失即违规）
  - `threshold`: count 模式时的阈值（可选）
  - `threshold_param`: 关联的 `params` 键名（可选，Profile 可通过 `params.<key>` 覆盖阈值）
- **lint_targets**: glob pattern 指定检查目标文件（如 `**/*.tex`、`**/*.bib`、`**/*.py`）

---

## Phase 词汇表

| Phase | 描述 |
|-------|------|
| `ideation` | 研究构思、选题、大纲、Figure 1 |
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

## Rule ID Registry

| Rule ID | slug | layer | severity | locked | enforcement |
|---------|------|-------|----------|--------|-------------|
| FIG.NO_IN_FIGURE_TITLE | fig-no-in-figure-title | core | error | true | lint_script |
| FIG.FONT_GE_24PT | fig-font-ge-24pt | core | error | false | doc |
| FIG.ONE_FILE_ONE_FIGURE | fig-one-file-one-figure | core | error | true | doc |
| FIG.COLORBLIND_SAFE_PALETTE | fig-colorblind-safe-palette | core | warn | false | doc |
| FIG.SELF_CONTAINED_CAPTION | fig-self-contained-caption | core | warn | false | doc |
| FIG.SYSTEM_OVERVIEW_ASPECT_RATIO_GE_2TO1 | fig-system-overview-aspect-ratio-ge-2to1 | core | error | true | doc |
| FIG.VECTOR_FORMAT_REQUIRED | fig-vector-format-required | core | error | false | doc |
| TABLE.BOOKTABS_FORMAT | table-booktabs-format | core | warn | false | lint_script |
| TABLE.DIRECTION_INDICATORS | table-direction-indicators | core | warn | false | doc |
| LATEX.CMARK_XMARK_PMARK_MACROS | latex-cmark-xmark-pmark-macros | core | error | false | doc |
| LATEX.EQ.DISPLAY_STYLE | latex-eq-display-style | core | error | true | doc |
| LATEX.VAR.LONG_TOKEN_USE_TEXT | latex-var-long-token-use-text | core | warn | false | doc |
| LATEX.NOTATION_CONSISTENCY | latex-notation-consistency | core | error | true | doc |
| REF.CROSS_REFERENCE_STYLE | ref-cross-reference-style | core | warn | false | doc |
| PAPER.SECTION_HEADINGS_MAX_6 | paper-section-headings-max-6 | core | error | false | lint_script |
| PAPER.CONCLUSION_SINGLE_PARAGRAPH | paper-conclusion-single-paragraph | core | warn | false | doc |
| CITE.VERIFY_VIA_API | cite-verify-via-api | core | error | true | doc |
| EXP.ERROR_BARS_REQUIRED | exp-error-bars-required | core | error | false | doc |
| EXP.TAKEAWAY_BOX | exp-takeaway-box | core | warn | false | doc |
| EXP.ABLATION_IN_RESULTS | exp-ablation-in-results | core | warn | false | doc |
| EXP.RESULTS_SUBSECTION_STRUCTURE | exp-results-subsection-structure | core | warn | false | doc |
| REPRO.RANDOM_SEED_DOCUMENTATION | repro-random-seed-documentation | core | error | false | doc |
| REPRO.COMPUTE_RESOURCES_DOCUMENTED | repro-compute-resources-documented | core | warn | false | doc |
| SUBMIT.SECTION_NUMBERING_CONSISTENCY | submit-section-numbering-consistency | core | warn | false | lint_script |
| PROSE.INTENSIFIERS_ELIMINATION | prose-intensifiers-elimination | domain | warn | false | lint_script |
| PROSE.EM_DASH_RESTRICTION | prose-em-dash-restriction | domain | warn | false | lint_script |
| ETHICS.LIMITATIONS_SECTION_MANDATORY | ethics-limitations-section-mandatory | venue | error | false | doc |
| ANON.DOUBLE_BLIND_ANONYMIZATION | anon-double-blind-anonymization | venue | error | true | doc |
| SUBMIT.PAGE_LIMIT_STRICT | submit-page-limit-strict | venue | error | false | doc |
| BIBTEX.CONSISTENT_CITATION_KEY_FORMAT | bibtex-consistent-citation-key-format | venue | warn | false | lint_script |

---

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
