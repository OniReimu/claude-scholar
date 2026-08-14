---
id: PROSE.INFORMAL_VOCABULARY
slug: prose-informal-vocabulary
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
conflicts_with: [PROSE.VAGUE_QUANTIFIERS]
constraint_type: guardrail
autofix: safe
lint_patterns:
  - pattern: "\\b(a lot of|lots of)\\b"
    mode: match
  - pattern: "\\b(things|stuff)\\b"
    mode: match
  - pattern: "\\bkind of\\b"
    mode: match
  - pattern: "\\bsort of\\b"
    mode: match
  - pattern: "\\bbigger\\b"
    mode: match
fix_patterns:
  - find: "\\bbigger\\b"
    replace: "larger"
  - find: "\\bkind of\\b"
    replace: "somewhat"
  - find: "\\bsort of\\b"
    replace: "somewhat"
lint_targets: "**/*.tex"
---

## Requirement

禁止在学术论文中使用口语化、非正式的词汇。

| 禁用 | 替代 |
|------|------|
| a lot of / lots of | 具体数字，或 many（注意 "many studies" 类组合会触发 `PROSE.VAGUE_QUANTIFIERS`，最好直接量化） |
| things | factors / components / elements |
| stuff | data / material / content |
| get | obtain / achieve / acquire |
| big | large / substantial |
| kind of / sort of | 删除，或用 approximately / somewhat |
| bigger | larger |

`smaller` 不在禁用列表：它本身就是规范的比较级学术用词。

## Rationale

口语化词汇降低论文的正式程度，在同行评审中会被视为不够严谨。

## Check

- **regex 搜索**: 匹配禁用词列表
- **检查范围**: `.tex` 文件正文区域
- **排除**: 直接引用（quote 环境）中的口语化词汇

## Examples

### Pass

```latex
We obtain the optimal parameters by solving the constrained optimization problem.
A large number of factors influence the convergence rate.
```

### Fail

```latex
We get the parameters by solving a lot of things in the optimization.
```
