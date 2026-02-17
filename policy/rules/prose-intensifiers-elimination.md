---
id: PROSE.INTENSIFIERS_ELIMINATION
slug: prose-intensifiers-elimination
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-background, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: regex
enforcement: lint_script
params: {}
conflicts_with: []
lint_patterns:
  - pattern: "\\b(very|extremely|highly|significantly|remarkably|substantially)\\b"
    mode: match
lint_targets: "**/*.tex"
---

## Requirement

删除空洞的强调词（very、extremely、highly、significantly、remarkably、substantially），除非在统计显著性语境中使用（如 "statistically significant"）。用具体数据替代（"improves accuracy by 15%" 而非 "significantly improves accuracy"）。

## Rationale

空洞强调词削弱论文可信度，顶级论文用具体数据说话。这些词也是 AI 写作的常见信号。

## Check

- **regex 搜索**: `\b(very|extremely|highly|significantly|remarkably|substantially)\b`
- **排除合法用法**: "statistically significant"、"significantly different (p < 0.05)" 等统计显著性语境
- **检查范围**: 所有 `.tex` 文件正文区域

## Examples

### Pass

```latex
Our method improves accuracy by 15.3\% over the baseline.
```

### Fail

```latex
Our method very significantly outperforms the highly competitive baseline.
```
