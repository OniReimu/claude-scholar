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
conflicts_with: [PROSE.CAUSAL_CONNECTIVE]
constraint_type: guardrail
autofix: none
lint_patterns:
  - pattern: "[a-z]{3,}:\\s+[a-z]"
    mode: match
lint_targets: "**/*.tex"
---

## Requirement

正文段落（含脚注）里的句中冒号一律禁止，含中英文冒号，不限于解释、定义或转折。改写为完整句子，或用句号断开。

**适用范围与优先级**：正文段落（含脚注、列表中的散文和直接引语）一律执行。锁、六原则和保真都不能豁免；保留命题并改成合规结构，不再逐条判断是否放行。

**仅非散文内容豁免**：数学、代码（含 `\texttt{...}`、`\verb`）、`\;` 细空格、URL、时间、引用键；结构标题中的冒号也豁免，含段首 `\textbf{...}` / `\paragraph{...}` run-in 小标题，冒号可在标题中间，例如 `\textbf{TV-1: Static extraction scope. }`。普通句中的加粗文字不是标题，标题中的分号也不因此豁免。

**排除范围**：caption 和表格（`table`、`tabular` 及其变体，包括 `\input` 进来的表格文件）不计候选、不计违规，只在报告中给「已排除」计数。表格按实际环境识别，不因文件在 `tables/` 目录就跳过其中的正文。

**清理后必须复测连接词分布。** 本卡的修法把标点隐含的逻辑关系（"所以""即""因此"）赶到词汇层，而 `PROSE.CAUSAL_CONNECTIVE` 菜单里 `therefore` 是最安全的默认——跨多遍清理会累积出连接词单一化，逐处判都合理、只在总量上崩。动过一批之后，看一眼 lint 的 connective distribution 计数表，逐处追问"这是哪一种因果、答案是否真的全是同一种"。

本规则比 `PROSE.COLON_LIST_OVERUSE` 更宽，禁止范围内的所有句中冒号。

## Rationale

作者 2026-10-21 定为硬规则，用来取代逐条判断；MSR 一稿因为逐条判断加上锁，违规句一路存活了 11 组。

## Check

- 用 `skills/writing-anti-ai-loop/scripts/guardrails.py` 扫描摘要到参考文献前的正文，展开静态 input/include，保留文件与行号。
- 非豁免的正文冒号全部记「违规」，没有「待判」；caption / table 只报「已排除」计数。
- 原有 lint regex 仅作定位，不能代替完整枚举；大写字母、数字或宏之前的冒号同样违规。

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
