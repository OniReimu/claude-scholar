---
id: SUBMIT.AVAILABILITY_COMMITMENT
slug: submit-availability-commitment
severity: warn
locked: false
layer: core
artifacts: [text]
phases: [writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: lint_script
params: {}
conflicts_with: [ANON.DOUBLE_BLIND_ANONYMIZATION]
constraint_type: guardrail
autofix: none
lint_targets: "**/*.tex"
coverage_note: "the patterns locate the conventional phrasings of a deferred or on-request release (upon acceptance, available on request, will be released, we will open-source, in the full version). A commitment phrased any other way (a maintained leaderboard, a forthcoming dataset card) is the judgment layer's to catch. A hit is a question to the author, not a defect: a promise the author has confirmed and the venue accepts stays."
lint_patterns:
  - pattern: "\\b(?:upon|after|following) (?:the paper's |paper |our )?(?:acceptance|publication)\\b"
    mode: match
  - pattern: "\\b(?:available|provided|shared|obtainable|accessible)(?: from the (?:corresponding )?authors?)? (?:upon|on) (?:reasonable )?request\\b"
    mode: match
  - pattern: "\\b(?:available|obtainable) from the (?:corresponding )?authors?\\b"
    mode: match
  - pattern: "\\b(?:[Cc]ode|[Dd]ata(?:sets?)?|[Mm]odels?|[Ww]eights|[Cc]heckpoints?|[Bb]enchmarks?|[Aa]rtifacts?|[Ii]mplementations?|[Pp]rompts|[Tt]ranscripts|[Aa]nnotations|[Mm]aterials?|[Tt]ools?)\\b[^.]{0,40}\\bwill be (?:made )?(?:publicly |freely |openly )?(?:available|released|open-?sourced|shared|published)\\b"
    mode: match
  - pattern: "\\b[Ww]e (?:will|plan to|intend to) (?:publicly )?(?:release|open-?source|share|make (?:\\w+ ){0,4}(?:available|public))\\b"
    mode: match
  - pattern: "\\b(?:in|to) (?:the|a|an) (?:camera-ready|final|extended|full|journal) version\\b"
    mode: match
---

## Requirement

论文里每一句关于**将来**或**按请求**提供代码、数据、模型、benchmark、补充结果的话，都是一条承诺。投稿前每条承诺必须落在以下三种状态之一：

1. **现在就兑现。** 提交时 artifact 已经可得，正文给出链接（双盲投稿用匿名链接，见 `ANON.DOUBLE_BLIND_ANONYMIZATION`）。写成现在时："Code and data are available at \url{…}."
2. **作者确认过的计划。** 作者明确确认了这个计划，并且写得具体：放出什么、放在哪里、在什么条件下（许可证、访问限制、伦理审查）。同时核对目标 venue **当期** CFP 和作者指南里关于代码与数据可得性的规定。各 venue 对"接收后公开"和"按请求提供"的态度不同，必须查原文，不能凭记忆。
3. **以上都不是 → 删掉，或改写成当前事实。** 没有确认过的承诺不留在稿子里。

"隐性承诺"同样算：
- "in the full / extended version" 承诺了一个尚不存在的版本，投稿稿里说附录在 full version 中，等于说审稿人看不到的证据存在。
- "a leaderboard will be maintained"、"we will continue to update the benchmark" 承诺了持续维护。

**不算承诺：** future work 里的研究方向（"Future work could test whether …"）是对问题的判断，不是对读者的交付。

## Rationale

承诺写进论文就会被当成事实来评估。审稿人和投稿 checklist 会问代码、数据在提交时是否可得，一句"将在接收后公开"可能被当成"现在不可得"来打分。论文发表后，承诺没兑现会被读者和后续工作追究，损失的是作者的信用。"available from the authors on request"还有一层额外风险：它把可得性绑在作者个人的长期响应上，作者换单位或邮箱失效后，承诺就失效了。

这类句子大多不是作者真想做出的承诺，而是写作时的惯性：模板里有一句，或者觉得"总该说点什么"。所以本卡把每条命中都变成一个问作者的问题，而不是一个替作者做的决定。

## Check

- **提取 `.tex` 正文的正确方法**：见 `policy/references/tex-prose-extraction.md`。

- **Regex 定位**：`policy/lint.sh --rule SUBMIT.AVAILABILITY_COMMITMENT`。范围包括正文、脚注、Data/Code Availability statement、checklist 答复、附录。
- **regex 覆盖不到的承诺**：措辞不走惯用说法的承诺（"a leaderboard will be maintained"、"a dataset card is forthcoming"）只能逐句读出来。惯用说法有限，可以枚举；其余的写法无法枚举，所以留给判断层。
- **逐条问作者（不可由 agent 代答）**：每条命中，以及 regex 抓不到的隐性承诺，列出原句并问：
  1. 这个 artifact 现在能不能放出来？能 → 改成状态 1，附链接。
  2. 不能，有没有确定的计划？有 → 按状态 2 写具体，并核对 venue 当期政策。
  3. 都没有 → 删掉，或改写成当前事实。
- **记录**：self-review 报告里逐条写明作者的答复。没有答复的承诺按未解决处理，**不得进入投稿**。

## Examples

### Pass

```latex
% 状态 1：现在就兑现，现在时，匿名链接
Code, prompts, and all model transcripts are available at
\url{https://anonymous.4open.science/r/xxxx}.

% 状态 2：作者确认的计划，写清内容、位置和条件
The annotated corpus contains user-generated text and is released
under a research-only licence through the project's data portal
after ethics approval (protocol ETH24-1234), expected before the
conference.
```

### Fail

```latex
% 未经确认的延期承诺
Our code will be released upon acceptance.
% 把可得性绑在作者个人响应上
The dataset is available from the corresponding author on request.
% 承诺一个审稿人看不到的版本
Full proofs are deferred to the extended version.
```

## Conflicts

- `ANON.DOUBLE_BLIND_ANONYMIZATION`：状态 1 的链接在双盲投稿里必须匿名。为了匿名而改写成"将在接收后公开"，是把一个匿名问题换成了一个承诺问题；匿名仓库可以同时解决两者。
