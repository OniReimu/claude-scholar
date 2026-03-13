---
id: PROSE.EM_DASH_RESTRICTION
slug: prose-em-dash-restriction
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-background, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: regex
enforcement: lint_script
params: {max_per_paragraph: 0}
conflicts_with: []
lint_patterns:
  - pattern: "---"
    mode: match
lint_targets: "**/*.tex"
---

## Requirement

禁止使用 em-dash（`---`）。一个都不允许。用以下替代方案：(1) 拆成新句子，(2) 关系从句（`, which...`），(3) 逗号插入语，(4) 括号。Em-dash 是强烈的 AI 写作信号。

## Rationale

Em-dash 过度使用是 LLM 生成文本的典型特征。人类学术写作极少在一段内使用多个 em-dash。

## Check

- **regex 匹配**: `.tex` 文件正文区域中出现 `---` 即违规
- **注意区分**: LaTeX em-dash `---` 和 YAML frontmatter `---`，仅检查正文区域

## Examples

### Pass

```latex
% 全段无 em-dash
Our method builds on prior work and outperforms all baselines, including recent
state-of-the-art approaches, by a large margin.

% 用逗号插入语替代 em-dash
Our method, the first to combine both techniques, outperforms all baselines.
```

### Fail

```latex
Our method---which builds on prior work---outperforms baselines---including
recent state-of-the-art approaches---by a large margin.
```
