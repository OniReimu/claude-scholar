---
id: PROSE.SENTENCE_LENGTH
slug: prose-sentence-length
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-background, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: regex
enforcement: lint_script
params: {max_words: 35}
conflicts_with: []
constraint_type: guardrail
autofix: none
lint_patterns:
  - pattern: "(?:[^.!?\\s]+\\s+){35,}[^.!?\\s]+[.!?]"
    mode: match
lint_targets: "**/*.tex"
---

## Requirement

单句不超过 **35 词**。典型句长区间为 25-35 词。超过时拆成多个短句，或使用分号连接两个独立子句。

## Rationale

过长的句子降低可读性，增加理解负担。Pre-GPT 时期的论文典型句长为 25-35 词，这是工程类学术写作的自然节奏。

## Check

- **regex（逐行）**: 匹配「≥36 个不含句读的 token 连排 + 终止标点」的行内长句。**已知局限**：grep 逐行匹配，硬换行（每 80 列断行）的 `.tex` 中跨行长句不会被抓到——regex 层只保证零误报的部分召回，软换行文件（一段一行，Overleaf 默认）召回完整
- **完整检查**: 用 `prose-rhythm-variance.md` 附带的 Python 脚本剥离公式/命令后按句统计，超 35 词逐句标记；`params.max_words` 是该语义检查的阈值（regex 的 35 硬编码在 pattern 中，profile 覆盖 `max_words` 只影响语义检查）
- **排除**: LaTeX 命令 token 不计入词数（如 `\cite{}`、`\ref{}`）；公式环境内的内容

## Examples

### Pass

```latex
The proposed method formulates the unlearning problem as an optimization
task. It minimizes the influence of target samples while preserving
model utility on the remaining data.
```

### Fail

```latex
The proposed method formulates the machine unlearning problem as a
constrained optimization task that minimizes the influence of target
training samples on the model parameters while simultaneously preserving
the overall model utility and performance on the remaining non-target
training data through a carefully designed regularization term.
```
