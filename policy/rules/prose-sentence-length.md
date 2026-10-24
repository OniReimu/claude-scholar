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
conflicts_with: [PROSE.SEMICOLON_RESTRICTION]
constraint_type: guardrail
autofix: none
lint_patterns:
  # Upper-bounded repetition. `{35,}` is greedy and unbounded, so on a long line
  # with no sentence punctuation the engine expands to the end of the line and
  # backtracks from every start position. A 35-60 word window is enough to
  # locate a review candidate: a 200-word sentence still matches on its final window.
  #
  # A word is any run of characters in which `.`/`!`/`?` appear only INSIDE the
  # token: `[.!?](?!\s|$)`. The previous form used `[^.!?\s]+`, which treated
  # every dot as a sentence boundary, so `$\beta=0.9$` cut a 55-word sentence
  # into runs of 30 and 24 and the rule went silent. Decimals sit in exactly the
  # sentences this rule is meant to catch, so the under-count was systematic.
  # The word group is atomic — `(?>...)` inside `{35,60}` is a nested quantifier,
  # and without it this is the same backtracking hazard that already hit
  # PROSE.COMMA_OVERUSE. Benchmarked at 44ms on 400 tokens with no terminator.
  # Abbreviations (`i.e.`, `et al.`) still split a run; their dot really is
  # followed by a space and no regex can tell them from a full stop.
  - pattern: "(?:(?>(?:[^\\s.!?]|[.!?](?!\\s|$))+)\\s+){35,60}(?>(?:[^\\s.!?]|[.!?](?!\\s|$))+)[.!?](?=\\s|$)"
    mode: match
lint_targets: "**/*.tex"
---

## Requirement

单句超过 **35 词**时触发复查，**不是硬拆线**。按指代是否清楚、从句是否嵌套、主语到主干谓语的距离和理解负担判断，读者能一次解析主干及必要限定就可保留。允许把「事实 + 原因／条件／用途／限定」合成一句，不限于修复语法残句；拆合都应让关系更清楚，不追求特定句长。（来源：v1.38.1，nso 实验 077207a 分析）

不要用分号连接两个独立子句来"缩短"句子——分号不产生新句子，只是把同一个长句换个标点，而 `PROSE.SEMICOLON_RESTRICTION` 禁止这个构造。本卡此前把它列为合法修法，那是错的。

## Rationale

句长可以提示理解负担，但不能代替阅读判断。复杂的指代、嵌套和过远的主干关系需要修，事实及其必要解释则可以放在同一句里。

主语到主干谓语等依存关系拉长时的加工负担，见 Gibson（1998，*Cognition*, 68(1):1–76），相关模型研究和跨时代复测见 [解释式语域卡的 Rationale](prose-explaining-register.md#rationale)。

## Check

- **regex（逐行）**: 匹配「≥36 个不含句读的 token 连排 + 终止标点」的行内复查候选，不直接判违规。**已知局限**：grep 逐行匹配，硬换行（每 80 列断行）的 `.tex` 中跨行长句不会被抓到；软换行文件（一段一行，Overleaf 默认）仍须人工判断理解负担
- **完整检查**: 用 `prose-rhythm-variance.md` 附带的 Python 脚本剥离公式/命令后按句统计，超 35 词逐句标记复查；`params.max_words` 是复查阈值（regex 的 35 硬编码在 pattern 中，profile 覆盖 `max_words` 只影响语义检查）。最终按指代、嵌套、主干距离和理解负担决定是否拆句
- **排除**: LaTeX 命令 token 不计入词数（如 `\cite{}`、`\ref{}`）；公式环境内的内容

## Examples

### Pass

```latex
The proposed method formulates the unlearning problem as an optimization
task. It minimizes the influence of target samples while preserving
model utility on the remaining data.
```

### Fail

下面的问题在于目标、保留条件和实现手段层层挂靠，读者需要回溯关系，不是单因超过 35 词。

```latex
The proposed method formulates the machine unlearning problem as a
constrained optimization task that minimizes the influence of target
training samples on the model parameters while simultaneously preserving
the overall model utility and performance on the remaining non-target
training data through a carefully designed regularization term.
```
