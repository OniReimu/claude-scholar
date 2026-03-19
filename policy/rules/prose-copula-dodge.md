---
id: PROSE.COPULA_DODGE
slug: prose-copula-dodge
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
constraint_type: guardrail
autofix: safe
lint_patterns:
  - pattern: "\\b(serves as|stands as|marks a|represents a)\\b"
    mode: match
fix_patterns:
  - find: "\\bserves as\\b"
    replace: "is"
  - find: "\\bstands as\\b"
    replace: "is"
  - find: "\\bmarks a\\b"
    replace: "is a"
lint_targets: "**/*.tex"
---

## Requirement

不要用 "serves as"、"stands as"、"marks a"、"represents a" 来替代简单的 "is/are"。AI 因为重复惩罚倾向于避开基本系动词，产生这些浮夸替代。

| 禁用 | 替代 |
|------|------|
| serves as | is |
| stands as | is |
| marks a pivotal moment | is an important step |
| represents a shift | is a change |

**例外**：当 "represents" 用于数学/统计含义（"$x$ represents the input"）时允许。

## Rationale

系动词回避是 AI 生成文本的已知特征。人类学术写作中 "is" 完全正常，不需要用花哨的替代来避免重复。

## Check

- **regex 搜索**: 匹配 "serves as", "stands as", "marks a", "represents a"
- **排除**: 数学定义语境中的 "represents"（如 "where $x$ represents..."）
- **检查范围**: `.tex` 文件正文区域

## Examples

### Pass

```latex
Gallery 825 is LAAA's exhibition space for contemporary art.
The gallery has four rooms totaling 3,000 square feet.
```

### Fail

```latex
Gallery 825 serves as LAAA's exhibition space for contemporary art.
The gallery boasts four separate spaces and features over 3,000 square feet.
```
