---
id: PROSE.VAGUE_ATTRIBUTIONS
slug: prose-vague-attributions
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
  - pattern: "\\b([Ee]xperts|[Oo]bservers|[Rr]esearchers|[Ss]cholars)\\s+(argue|believe|suggest|note|have cited|have noted|contend|maintain)\\b"
    mode: match
  - pattern: "\\b([Ii]ndustry|[Rr]ecent) reports (suggest|indicate|show)\\b"
    mode: match
lint_targets: "**/*.tex"
---

## Requirement

禁止将观点归因于未命名的权威（"experts argue", "researchers believe", "industry reports suggest"）。学术论文中所有归因必须有具体的 citation。

| 禁用 | 替代 |
|------|------|
| Experts argue that... | Zhang et al.~\cite{zhang} argue that... |
| Researchers believe... | 删除，或提供 citation |
| Industry reports suggest... | A report by McKinsey~\cite{mck} shows... |
| Observers have noted... | 删除，直接陈述 + citation |

## Rationale

模糊归因在学术写作中不可接受——审稿人会直接要求补充引用。这也是 AI 写作的常见特征：AI 没有真实来源，所以用 "experts" 来伪造权威性。

## Check

- **regex 搜索**: 匹配 "experts/observers/researchers/scholars" + 动词模式
- **检查范围**: `.tex` 文件正文区域

## Examples

### Pass

```latex
Zhang et al.~\cite{zhang2023} demonstrated that gradient-based
unlearning achieves comparable accuracy to full retraining.
```

### Fail

```latex
Experts argue that gradient-based unlearning is a promising approach.
Recent industry reports suggest that privacy regulations will drive
adoption of machine unlearning techniques.
```
