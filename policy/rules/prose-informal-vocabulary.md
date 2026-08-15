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
conflicts_with: [PROSE.VAGUE_QUANTIFIERS, PROSE.REGISTER_PRESERVATION]
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
  - pattern: "\\bfor good\\b(?!\\s+(reason|measure|cause|practice|approximation))"
    mode: match
  - pattern: "\\bon heads\\b"
    mode: match
  - pattern: "\\bdrives? off\\b"
    mode: match
  - pattern: "\\bholds? levers\\b"
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
| for good | permanently（`for good reason/measure/cause` 是合法搭配，已在 pattern 中排除） |
| on heads | on a positive draw（或该实验实际的事件名） |
| drives off | deters / discourages |
| holds levers | imposes a cost / has instruments（并查 `PROSE.ABSTRACT_AGENCY`） |

`smaller` 不在禁用列表：它本身就是规范的比较级学术用词。

> ⚠️ **这份词表是地板，不是覆盖范围。** 实测中一次压缩 pass 产生的九处语域违规，本卡的 lint pattern 命中 **0/9**——因为语域是**编辑动作**的属性，不是**词**的属性：`pay peers`、`too heavy a one` 都不口语，只是比它们替换掉的措辞更不精确，任何以"口语词"为范围的规则永远看不见它们。
>
> **真正生效的检查是 `PROSE.REGISTER_PRESERVATION`**（diff 范围，比对改动 span 与其改前措辞）。本卡负责：词表层命中，以及**未被本次 pass 改动**的文本。不要把这份加长的列表误当成问题已解决。

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
