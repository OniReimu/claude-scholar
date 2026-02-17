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
params: {max_per_paragraph: 1}
conflicts_with: []
lint_patterns:
  - pattern: "---"
    mode: count
    threshold: 1
lint_targets: "**/*.tex"
---

## Requirement

不使用 em-dash（`---`）做括号式插入。用以下替代方案：(1) 拆成新句子，(2) 关系从句（`, which...`），(3) 逗号插入语。每段 em-dash 不超过 1 个。每段出现 >=2 个 em-dash 是强烈的 AI 写作信号。

## Rationale

Em-dash 过度使用是 LLM 生成文本的典型特征。人类学术写作极少在一段内使用多个 em-dash。

## Check

- **regex 计数**: 每段中 `---` 出现次数，超过 1 个则违规
- **注意区分**: LaTeX em-dash `---` 和 YAML frontmatter `---`，仅检查 `.tex` 文件正文区域
- **阈值**: 每段最多 1 个 em-dash

## Examples

### Pass

```latex
% 全段无 em-dash
Our method builds on prior work and outperforms all baselines, including recent
state-of-the-art approaches, by a large margin.

% 仅一处 em-dash 用于强调
Our method---the first to combine both techniques---outperforms all baselines.
```

### Fail

```latex
Our method---which builds on prior work---outperforms baselines---including
recent state-of-the-art approaches---by a large margin.
```
