---
id: PROSE.UNICODE_ARROWS
slug: prose-unicode-arrows
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
  - pattern: "[→←↔⇒⇐⇔➜➔]"
    mode: match
fix_patterns:
  - find: "→"
    replace: "$\\rightarrow$"
  - find: "←"
    replace: "$\\leftarrow$"
  - find: "↔"
    replace: "$\\leftrightarrow$"
  - find: "⇒"
    replace: "$\\Rightarrow$"
  - find: "⇐"
    replace: "$\\Leftarrow$"
lint_targets: "**/*.tex"
---

## Requirement

LaTeX 正文中禁止使用 Unicode 箭头字符（→, ←, ↔, ⇒ 等）。使用 LaTeX 数学命令替代：`$\rightarrow$`, `$\leftarrow$`, `$\Rightarrow$` 等。

## Rationale

Unicode 箭头在 LaTeX 编译时可能因编码/字体问题出错，且不是标准 LaTeX 用法。AI（尤其是 Claude）倾向于在文本中直接使用 → 箭头。正确做法是使用 LaTeX 数学模式的箭头命令。

## Check

- **regex 搜索**: 匹配 Unicode 箭头字符
- **检查范围**: `.tex` 文件
- **排除**: 注释行

## Examples

### Pass

```latex
The input $x$ is mapped to the output $y$ via $f: x \rightarrow y$.
The pipeline proceeds as follows: preprocessing $\rightarrow$ training
$\rightarrow$ evaluation.
```

### Fail

```latex
The input x is mapped to the output y via f: x → y.
The pipeline proceeds as follows: preprocessing → training → evaluation.
```
