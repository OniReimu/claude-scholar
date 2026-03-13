---
id: PROSE.FILLER_PHRASES
slug: prose-filler-phrases
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
  - pattern: "\\b[Ii]n order to\\b"
    mode: match
  - pattern: "\\b[Ii]t is important to note that\\b"
    mode: match
  - pattern: "\\b[Ii]t is worth noting that\\b"
    mode: match
  - pattern: "\\b[Aa]s a matter of fact\\b"
    mode: match
  - pattern: "\\bplays a crucial role in\\b"
    mode: match
  - pattern: "\\b[Ii]t should be noted that\\b"
    mode: match
  - pattern: "\\b[Ii]t is noteworthy that\\b"
    mode: match
  - pattern: "\\b[Ii]n the context of\\b"
    mode: match
lint_targets: "**/*.tex"
---

## Requirement

删除冗余填充短语。这些短语不携带信息，只增加字数。

禁用列表：
- "In order to" → "To"
- "It is important to note that" → 删除，直接陈述
- "It is worth noting that" → 删除，直接陈述
- "As a matter of fact" → 删除
- "plays a crucial role in" → "is critical for" 或更具体的描述
- "It should be noted that" → 删除，直接陈述
- "It is noteworthy that" → 删除，直接陈述
- "In the context of" → "in" 或 "for"

## Rationale

填充短语浪费页面空间，稀释论点密度。顶级论文每个词都有信息量。这些短语也是 AI 生成文本的常见特征。

## Check

- **regex 搜索**: 逐条匹配禁用短语
- **检查范围**: 所有 `.tex` 文件正文区域

## Examples

### Pass

```latex
To improve efficiency, we propose a new scheduling algorithm.
```

### Fail

```latex
In order to improve efficiency, it is important to note that we propose a new scheduling algorithm.
```
