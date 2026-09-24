---
id: PROSE.CLAUSE_CHAIN
slug: prose-clause-chain
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-background, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_style
enforcement: doc
params: {max_subordinate_units: 1}
conflicts_with: [PROSE.RHYTHM_VARIANCE]
constraint_type: guidance
autofix: none
---

## Requirement

一句话只挂一层从属结构。当一句里叠了两层以上的从属单位——关系从句（`, which ...` / `, where ...`）、插入语（`and, without X, ...`）、并列谓语（`... and sets ...`）、分词尾巴（`, reducing ...`）——就把它拆成几句，**每句只说一件事**，新起的句子用锚（`The estimates ...` / `This bound ...` / `thus` / `the same`）接回上一句。

拆链只拆**从属关系**，不改**句长分布**：拆完后不得为了补回长度或句长方差把别的句子并长（见 `PROSE.RHYTHM_VARIANCE`）。

## Rationale

从句链是 LLM 行文最稳定的指纹之一：模型倾向于把一个主张的前提、推论和限定都挂在同一个句子上，用 `which` 接推论、用插入语塞条件、用 `and` 再接第二个谓语。每一处单看都合语法，逗号数也常常只有 2–3 个，`PROSE.COMMA_OVERUSE`（≥4 触发）和 `PROSE.SENTENCE_LENGTH`（>35 词触发）都抓不到。但读者必须在工作记忆里同时持有主语、关系从句的先行词、插入的条件和两个谓语，才能拼出句意。

实测（R²D² abstract，Saber 2026-09-24 判定）：

> Four sampled paths per question suffice to estimate both properties, **which** tells in advance whether more paths will pay **and, without ground-truth answers,** sets a budget that keeps 98\% of the accuracy of 32 paths with eight paths on average.

35 词、3 个逗号，两条长度类规则都不报。作者拆成下面两句后判定"前者更清晰"：

> Four sampled paths per question suffice to estimate both properties. The estimates predict whether more paths will pay, and without ground-truth answers they set a budget that keeps 98\% of the accuracy of 32 paths with eight paths on average.

同一轮里还暴露了反方向的失效：为把句长标准差拉回 ≥10，把相邻短句并成 36–44 词的长句，作者的反馈是"改之前节奏更好，句子太长了"。所以本卡只做拆，不做并。

`which` 的先行词不是一个名词、而是前面整个小句时（"suffice to estimate both properties, which tells ..."），这条链尤其难读：`which` 指的是"四条路径就够"这件事，读者要回头重新解析。这种句式同时触发 `PROSE.ABSTRACT_AGENCY`（一个事实在"告诉"什么），拆句时给新句一个具体主语（`The estimates`）。

## Check

- **LLM 检查**（逐句）：数一句里的从属单位——关系从句、句中插入语、第二个并列谓语、句末分词/介词尾巴。≥2 个即候选。
- **优先查三种形状**：
  1. `X, which V1 ... and V2 ...`——关系从句后再接并列谓语；
  2. `..., and, <插入语>, V ...`——并列连词后紧跟插入语；
  3. `which` 的先行词是整个前句而非名词。
- **拆法**：保留原句主干为第一句；被挂上去的推论或第二谓语另起一句，**给它一个具体主语或锚**，不得以 `It` / `This` 裸指一整个小句。
- **不拆**：定义句里的限定性关系从句（`the budget that keeps ...` 这类无逗号、限定名词的 `that` 从句）不算从属链；数学条件（`whenever ...`, `where $p$ is ...` 给符号下定义）不算。
- **拆完回查**：新起的句子若 ≤10 词，必须过 `PROSE.RHYTHM_VARIANCE` 的锚定检查；拆完**不要**为了方差去并别处的句子。

## Examples

### Pass

```latex
We prove that this binary account is a lower bound on plurality accuracy
whenever the correct paths cast the same vote. We also prove that a symmetric
Dirichlet-multinomial model forces wrong-answer agreement above a lower limit
set by the correctness correlation.
```

两个 `that` 宾语从句分属两句，每句一个证明结论。`whenever ...` 是数学条件，不计入从属链。

```latex
A plurality receiver returns the most frequent answer, as self-consistency does,
and benefits when the interference is incoherent. A binary receiver sees only
whether each path is right and discards that information.
```

两种接收机各占一句，每句一层结构。

### Fail（关系从句 + 插入语 + 并列谓语）

```latex
Four sampled paths per question suffice to estimate both properties, which tells
in advance whether more paths will pay and, without ground-truth answers, sets a
budget that keeps 98\% of the accuracy of 32 paths with eight paths on average.
```

`which` 的先行词是整个前句，后面又插入条件、再接第二谓语。
修法：`... both properties. The estimates predict whether more paths will pay, and without ground-truth answers they set a budget ...`

### Fail（两个关系从句夹在一个并列句里）

```latex
A plurality receiver, which returns the most frequent answer as self-consistency
does, benefits when the interference is incoherent, and a binary receiver, which
sees only whether each path is right, discards that information.
```

两个非限定关系从句加一个并列连接，读者要同时追踪两个主语。
修法：拆成两句，每种接收机一句（见 Pass 第二例）。

### Fail（拆完又为了句长方差并长——反方向）

```latex
It determines whether the unwanted components add up or average out, and a
plurality receiver, which returns the most frequent answer as self-consistency
does, gains when they average out, whereas a binary receiver sees only whether
each path is right and discards that information.
```

44 词：原本两三句各讲一件事，为把段落句长标准差拉过 10 被并成一句，重新制造了从句链。
修法：恢复成各自独立的句子，句长方差低于阈值时接受它（`PROSE.RHYTHM_VARIANCE` 的阈值是诊断，不是验收）。

## Conflicts

与 `PROSE.RHYTHM_VARIANCE`：拆链会增加短句、可能压低句长方差。裁决：**拆链优先，方差让步**。只允许在方差卡规定的一种情形下合并（同一主张被拆在两句里、合并后仍只有一层从属），不得为方差重新制造从句链。
