---
id: PROSE.COLON_LIST_OVERUSE
slug: prose-colon-list-overuse
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
  - pattern: ":\\s*\\(1\\)"
    mode: match
  - pattern: ":\\s*\\(i\\)"
    mode: match
  - pattern: ":\\s*1\\)"
    mode: match
lint_targets: "**/*.tex"
---

## Requirement

正文段落中禁止使用"句子：(1)...(2)...(3)..."的模板化内联列表写法。如需列举，使用 LaTeX `enumerate` 或 `itemize` 环境，或改写为连贯散文。

Contribution section 中的编号列表除外（使用 `enumerate` 环境）。

## Rationale

冒号后直接跟编号列表是 AI 生成文本的典型模式。人类学术写作要么用正式的 LaTeX 列表环境，要么用散文形式的 "First, ... Second, ... Third, ..." 表达。

## Check

- **regex 搜索**: 正文中出现 `:(1)` 或 `:1)` 或 `:(i)` 模式
- **排除**: `\begin{enumerate}` 环境内的编号
- **检查范围**: `.tex` 文件正文区域

## Examples

### Pass

```latex
Our method has three advantages. First, it reduces computation cost.
Second, it improves accuracy. Third, it scales to large datasets.
```

```latex
Our method has three advantages:
\begin{enumerate}
  \item It reduces computation cost.
  \item It improves accuracy.
  \item It scales to large datasets.
\end{enumerate}
```

### Fail

```latex
Our method has three advantages: (1) it reduces computation cost,
(2) it improves accuracy, and (3) it scales to large datasets.
```
