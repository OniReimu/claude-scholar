---
id: PAPER.DEFINE_BEFORE_USE
slug: paper-define-before-use
severity: warn
locked: false
layer: core
artifacts: [text, equation, figure, table]
phases: [writing-intro, writing-background, writing-system-model, writing-methods, writing-experiments, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: []
constraint_type: guidance
autofix: none
---

## Requirement

按阅读顺序，一个术语第一次被**使用**的地方，读者必须已经拿到它的含义——要么前文已定义，要么就在这一处给出一句 gloss。形式化定义可以放在后面，但读者在首次使用处必须已经能读懂这句话。

"术语"指本文引入或赋予特定含义的构造：自造术语、方法与组件名、RQ / claim 标签、威胁模型角色、被收窄了含义的常用词（本文里的 "consistency" 不是一般意义上的 consistency），以及正文里出现的数学符号。

判定分三步：

1. **是不是使用。** 这句话的主张依赖这个术语的含义，才算使用。只是宣告"某节会引入它"的路线图提名（"§4 presents the drift detector"）不算，那是 `PROSE.FRACTAL_SUMMARY` 管的预告。
2. **首次使用处能否读懂。** 前文有定义、或本处有一句说清它是什么的 gloss，都算读得懂。**章节指针不是 gloss**："cascade drift (§4) accumulates …" 等于要读者翻到后面再回来，仍然判违规。
3. **读者基线。** 目标 venue 的读者本来就懂的标准术语（transformer、differential privacy 之于安全会议）不需要定义。但标准词被收窄了含义时，按本文术语处理。

阅读单元与 `PROSE.ABBREVIATION_FIRST_USE` 一致：Abstract 与正文各自独立，Abstract 里的 gloss 不算正文的定义。图表 caption 按它被首次引用的位置计入阅读顺序——Figure 1 常在第一页，它的 caption 往往就是术语的首次使用。

缩写的展开形式归 `PROSE.ABBREVIATION_FIRST_USE`，符号与 notation table 的一致性归 `LATEX.NOTATION_CONSISTENCY`；本卡只管**顺序**：定义有没有出现在第一次使用之前。

## Rationale

"全文某处有定义"不等于"读者读到这里时有定义"。`PROSE.NO_INTERNAL_PROVENANCE` 的未定义标识符子检查问的是前者，定义放在 §4、却在 §1 和 §3 已经用了三次的术语能顺利通过。审稿人按顺序读，读到一个不认识的构造，只能三选一：猜、往后翻、跳过。三种都会让论证在他那里断掉，而且作者自己看不到——作者脑子里早就有这个定义。

这类问题落在节与节之间，逐段润色看不见：句级检查只看得到一段，看不出三节之后才出现定义。所以它的主执行点是 `claim-architecture-review` 的 P1 逐节扫描，那里的 ledger 本来就是按阅读顺序建的。

## Check

- **提取 `.tex` 正文的正确方法**：见 `policy/references/tex-prose-extraction.md`。

- **LLM 检查（按阅读顺序，Abstract 单独一轮）**：
  1. 记下每个术语的**首次使用**位置与**定义**位置（定义、定理/定义环境、notation table 行、或一句显式 gloss）。
  2. 首次使用早于定义、且首次使用处没有 gloss → forward use，违规。
  3. 全文从未定义 → undefined，违规（与 `PROSE.NO_INTERNAL_PROVENANCE` 子检查的结果合并报告，不重复计数）。
- **修法，取能解决问题的最小一档**：
  1. 在首次使用处补一句 gloss，形式化定义留在原处。这不算 `PROSE.RESTATEMENT_DILUTION` 的复述：gloss 说的是"它是什么"，形式化定义给的是"它如何精确成立"，粒度不同，删除测试会丢信息。但 gloss 必须是一句，不能把后面的定义段整段搬过来。
  2. 把定义段移到首次使用之前（定义段本身的位置可以挪时）。
  3. 把使用段移到定义之后。
  4. 改写这次早期使用，让它不依赖这个术语（这句话其实只需要一个描述，不需要名字时）。

## Examples

### Pass

```latex
% §1 Introduction —— 首次使用处带 gloss
Our key observation is that cascade drift, the error each stage
inherits from the stage before it, grows faster than per-stage
error. \S\ref{sec:method} formalises it as ...

% §4 Method —— 形式化定义
\begin{definition}[Cascade drift]
For a pipeline $f_K \circ \dots \circ f_1$, ...
\end{definition}
```

### Fail

```latex
% §1 —— 直接使用，读者不知道 cascade drift 是什么
Our key observation is that cascade drift grows faster than
per-stage error.

% §3 —— 用章节指针代替 gloss，仍是 forward use
Cascade drift (\S\ref{sec:method}) explains the gap in Table 2.

% §4 —— 第一次告诉读者它是什么
We define cascade drift as the error each stage inherits ...
```

## Conflicts

- `PROSE.RESTATEMENT_DILUTION`：首次使用处的一句 gloss 与后文的形式化定义不构成复述（见修法 1）。
- `PROSE.FRACTAL_SUMMARY`：只提名、不依赖含义的路线图不是使用，本卡不管；但若路线图句同时对这个术语做了主张，它就是使用。
