---
id: PROSE.MIDSENTENCE_COLON
slug: prose-midsentence-colon
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
autofix: none
lint_patterns:
  - pattern: "[a-z]{3,}:\\s+[a-z]"
    mode: match
lint_targets: "**/*.tex"
---

## Requirement

禁止在正文句子中间使用冒号引出解释、定义或转折，如 `We make a key observation: the model collapses.`、`We define the threat model: an adversary controls...`。改写为完整句子，或用句号断开。

**唯一例外**：段落开头的小标题（`\paragraph{Setup.}`、`\textbf{Security:}`）和 bullet/`\item` 的内联标题（`\item \textbf{Term:} ...`）——这些是结构标记，不是句中冒号。

注意本规则比 `PROSE.COLON_LIST_OVERUSE` 更宽：后者只抓「冒号 + 编号列表」，本规则抓**任意句中解释性冒号**，并撤销对 definition-style 冒号（`We define X: ...`）的豁免。

## Rationale

句中冒号引出解释是 AI 写作的典型句法——先抛一个宽泛主句，再用冒号补一个"重点"。人类正式写作更倾向把这层关系写进句子语法（用 that 从句、where、新句子），而不是靠冒号制造停顿。作者个人风格里，句中冒号（非小标题）一律视为待改写。

## Check

- **regex 搜索**: 匹配「小写词（≥3 字母）+ 冒号 + 空格 + 小写字母」，即句中解释性冒号
- **排除**（regex 已自然规避）: `\textbf{X:}` / `\paragraph{X:}` 等以 `}` 结尾的标题冒号（冒号后非「空格+小写」）
- **排除**: `\section`/`\subsection` 标题、`url`/`http:` 协议、时间 `12:30`、`\item` 内联标题
- **同时检查**: `references/patterns-english.md` 13b 节不应再把 definition-style 冒号列为 acceptable
- **检查范围**: `.tex` 文件正文区域

## Examples

### Pass

```latex
We make a key observation. The model collapses when the learning
rate exceeds 0.1.

\paragraph{Threat model.} The adversary controls a subset of clients.
% 段落小标题冒号是结构标记，不受本规则约束
```

### Fail

```latex
We make a key observation: the model collapses when the learning
rate exceeds 0.1.

We define the threat model: the adversary controls a subset of
clients.
% 句中解释性冒号，应改写为完整句子或断句
```
