# Rule Card Template

创建新 rule card 时复制此模板到 `policy/rules/<slug>.md`。

## Frontmatter

```yaml
---
id: CATEGORY.RULE_NAME
slug: category-rule-name
severity: error
locked: false
layer: core
artifacts: [text]
phases: [writing-methods, self-review]
domains: [core]
venues: [all]
check_kind: regex
enforcement: lint_script
params: {}
conflicts_with: []
lint_patterns: []
lint_targets: ""
---
```

## Body

```markdown
## Requirement

[祈使句，一句话描述可执行约束。]

## Rationale

[为什么需要这条规则。帮助 LLM 在边界情况做出判断。]

## Check

- **检查方式**: [regex / LLM 语义 / 人工]
- **要点**: [具体检查逻辑]

## Examples

### Pass

\```latex
% 符合规则的示例
\```

### Fail（违规描述）

\```latex
% 违规的示例
% 解决方式：...
\```
```

## Frontmatter 字段速查

### id（必填）

大写点分隔的唯一标识符。命名空间：

| Prefix | 领域 | 示例 |
|--------|------|------|
| `FIG.` | 图表 | `FIG.NO_IN_FIGURE_TITLE` |
| `TABLE.` | 表格 | `TABLE.BOOKTABS_FORMAT` |
| `LATEX.` | LaTeX 公式/格式 | `LATEX.EQ.DISPLAY_STYLE` |
| `REF.` | 交叉引用 | `REF.CROSS_REFERENCE_STYLE` |
| `PAPER.` | 论文结构 | `PAPER.SECTION_HEADINGS_MAX_6` |
| `CITE.` | 引文 | `CITE.VERIFY_VIA_API` |
| `EXP.` | 实验 | `EXP.ERROR_BARS_REQUIRED` |
| `REPRO.` | 可复现性 | `REPRO.RANDOM_SEED_DOCUMENTATION` |
| `SUBMIT.` | 投稿 | `SUBMIT.PAGE_LIMIT_STRICT` |
| `PROSE.` | 行文风格 | `PROSE.INTENSIFIERS_ELIMINATION` |
| `ETHICS.` | 伦理 | `ETHICS.LIMITATIONS_SECTION_MANDATORY` |
| `ANON.` | 匿名化 | `ANON.DOUBLE_BLIND_ANONYMIZATION` |
| `BIBTEX.` | BibTeX | `BIBTEX.CONSISTENT_CITATION_KEY_FORMAT` |

新领域可自定义前缀，保持大写点分隔。

### slug（必填）

= 文件名（不含 `.md`），kebab-case，如 `fig-no-in-figure-title`。

### severity

| 值 | 含义 |
|----|------|
| `error` | 必须修复，lint 报错 |
| `warn` | 建议修复，lint 报警告 |

### locked

| 值 | 含义 |
|----|------|
| `true` | severity 和 params 均不可被 profile 覆盖 |
| `false` | 允许 profile 覆盖 |

### layer

| 值 | 含义 |
|----|------|
| `core` | 所有论文必须遵守 |
| `domain` | 领域/风格相关 |
| `venue` | 会议/期刊特定 |

### artifacts

可选值：`figure`, `equation`, `text`, `table`, `code`, `bibtex`

### phases

可选值：`ideation`, `writing-background`, `writing-system-model`, `writing-methods`, `writing-experiments`, `writing-conclusion`, `self-review`, `revision`, `camera-ready`

### domains

- `[core]` = 通用
- 或具体领域列表：`[security, hci, se, is]`

### venues

- `[all]` = 所有会议
- 或具体列表：`[neurips, icml, iclr, ccs, usenix, ndss, sp, chi, icse, fse, ase, misq, isr]`

### check_kind

| 值 | 含义 | lint 支持 |
|----|------|-----------|
| `regex` | 正则匹配 | lint.sh 自动检查 |
| `ast` | AST 分析 | 未实现 |
| `llm_semantic` | LLM 语义检查 | 无，由 LLM 执行 |
| `llm_style` | LLM 风格检查 | 无，由 LLM 执行 |
| `manual` | 人工检查 | 无 |

### enforcement

| 值 | 含义 |
|----|------|
| `doc` | 仅文档约束 |
| `lint_script` | 已有自动检查 |

### params

声明可覆盖参数的默认值。Profile 通过 `params.<key>` 覆盖。

```yaml
params: {max_sections: 6}          # 单参数
params: {min_font_pt: 24}          # 单参数
params: {}                          # 无参数
```

### lint_patterns

仅 `check_kind: regex` 时填写。

```yaml
lint_patterns:
  - pattern: "\\\\section\\{"       # 正则（YAML 双转义）
    mode: count                      # match | count | negative
    threshold: 6                     # count 模式阈值
    threshold_param: max_sections    # 关联 params 键（可选）
```

**mode 说明**：
- `match`: 匹配即违规
- `count`: 匹配数 > threshold 才违规
- `negative`: 在目标文件中未匹配 = 违规（用于"必须存在某模式"的规则）

### lint_targets

glob pattern，如 `**/*.tex`、`**/*.bib`、`**/*.py`。
