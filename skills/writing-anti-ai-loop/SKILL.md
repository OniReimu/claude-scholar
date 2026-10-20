---
name: writing-anti-ai-loop
description: This skill should be used when the user wants a whole LaTeX paper rewritten chapter by chapter with writing-anti-ai until its style stops improving — "逐章改写到文风分不涨", "rewrite the paper chapter by chapter until the score stops rising", "iterate the anti-AI rewrite over the whole manuscript", "把整篇论文按 anti-AI 改几轮". One subagent per chapter per round; the first round runs the full writing-anti-ai skill, later rounds only its six explaining-register principles; after every round the orchestrator reverts over-reaching edits, checks citations, labels, formulas and numbers against the original, and scores the paper with a p2g style service (P2G_STYLE_URL) when one is configured, stopping when the score within the human range stops rising. Without a service it runs two fixed rounds. Works on a copy; the original manuscript is never edited. For one paragraph or one section, use writing-anti-ai directly.
version: 1.0.0
author: OniReimu
license: MIT
tags: [Writing, AI, Anti-AI, LaTeX, Subagents]
---

# Writing Anti-AI Loop

整篇论文逐章改写，迭代到文风不再变好为止。改写规则全部来自 `writing-anti-ai` 和 `PROSE.EXPLAINING_REGISTER`（六原则），本 skill 只负责编排：分章、派子代理、每轮审改动、核保真、打分、决定停不停。

claude-scholar 根目录是本 skill 目录的上两级，下文记作 `$CS`。

## 前提

- **打分服务（可选）**：环境变量 `P2G_STYLE_URL`（p2g 文风分服务的 `/api/style` 地址），需要时加 `P2G_STYLE_TOKEN`。
  - 配了：每轮打分，按分数决定停不停。
  - 没配：固定跑两轮（第 1 轮完整 skill，第 2 轮六原则专项），然后停。6g 和 edge 两篇的涨幅大部分出在前两轮。
- **稿件会离开本机。**打分时整篇 `.tex` 会上传到这个服务。服务声明不保存文件，但这仍然是把未发表稿件交给第三方，**开跑前告诉作者并得到同意**。
- 有 `latexmk` 时每轮编译一次；没有就跳过这一步，并在最终报告里写明。

## 准备

1. **只在副本上改。**在论文目录旁边建 `<论文名>-anti-ai/`，里面放：
   - `orig/`：原稿的完整拷贝，之后不再动；
   - `work/`：当前版本，子代理只改这里；
   - `r1/`、`r2/`……：每轮结束时 `work/` 的快照；
   - `LOG.md`：每轮的分数、保真结果、回退了哪些改动。

   拷贝时只排除编译产物（`.aux` `.log` `.bbl` 和编出来的论文 PDF），图片要留着，包括 `.pdf` 格式的图，否则编译不过。下文的 `main.tex` 指论文的主文件，不叫这个名字时换成实际的文件名（例如 `main_sp.tex`）。
2. **分章**：`python3 scripts/split.py split <out>/work/main.tex <out>/chapters`。它把 `main.tex` 切成连续的几段：导言区、摘要、每个 `\section` 一段、参考文献起的尾部。`parts.json` 里标 `edit` 的才交给子代理。
   - 多文件论文：`main.tex` 里通常只剩摘要可改，各章就是 `\input` / `\include` 的那些文件，子代理直接改 `work/` 里的章节文件。
   - 单文件论文：每个 `\section` 一段，子代理改 `chapters/` 里的段落文件。
   - 每轮子代理改完，先 `python3 scripts/split.py join <out>/chapters <out>/work/main.tex` 拼回去，再做后面的检查。拼回的文件保持原稿的版式。
   - 附录照常算章。不到 150 词的节可以和相邻的节合给同一个子代理。
3. **基线**：`python3 scripts/score.py <out>/orig --main main.tex`，记进 `LOG.md`。

## 每一轮

### 1. 并行改写

每章一个全新子代理（Agent tool，`general-purpose`），**在同一条消息里一起派出去**。子代理之间不共享上下文，每轮都是新的子代理。

- **第 1 轮**：完整执行 `writing-anti-ai`，先按六原则改，再按 anti-AI 规则改，然后停手。
- **第 2 轮起**：只按六原则（`$CS/policy/rules/prose-explaining-register.md`，含它的 Check 问题清单）改上一轮的结果，不再跑 anti-AI 规则全量。整篇重跑完整 skill 不会再涨（`writing-anti-ai/references/evidence.md` E8），逐章只按原则改才会涨（E9）。

子代理提示词模板（把 `{}` 换掉）：

```
你负责改写 {file}（{chapter}），只改这一个文件{abstract_note}。
先读 {CS}/policy/rules/prose-explaining-register.md 和 {CS}/policy/rules/prose-negation-contrast.md。
{round_rule}
硬约束：
- 只用 Edit 改 {file}。不建任何文件，不改其他文件，不跑会写文件的命令。
- 不增删事实、数字、引用、公式、\label 和 \ref 的目标。figure / table 环境、caption、公式、代码、\cite 键都不动。自定义宏（例如只在某个版本出现的 \extonly{…}）的调用和包住的范围都不动。
- 不改变论断的强度和范围，不加原文没有的解释、原因和推论。原文的意思不清楚时，保留原文，在回复里提出来。
- {length_rule}
- 你看不到任何分数，也不要去找或猜检测器。按规则改，不按词频改。
回复（300 字以内）：
1. 改了大约几处；
2. 可能改变论断强度或范围的改动，逐条给出原句和新句；
3. 原文没有直说、由你推出来的内容；
4. 原文里看到的事实问题（只报告，不改）。
```

- `{round_rule}`，第 1 轮：`再读 {CS}/skills/writing-anti-ai/SKILL.md，按它的执行顺序改：先六原则，再 anti-AI 规则，然后停手。`
- `{round_rule}`，第 2 轮起：`只按六原则改，逐段过一遍卡片 Check 里的六个问题。不要跑 anti-AI 规则全量。`
- 第 2 轮起，在硬约束末尾加一条 `- 下面这些句子保持原样，不要改：` 并逐句列出此前各轮回退过的原句。不锁的话，新的子代理会把同一处再改一遍（edge 和 anti_nso 的第 2 轮都把上一轮回退的同一句又改了回来）。
- `{abstract_note}`，摘要那一章：`，并且只改 abstract 环境`；其他章为空。
- `{length_rule}`，默认：`这一章的篇幅变化控制在 ±15% 以内。`稿件卡在页数上限时：`这一章不能变长，词数不超过原文；可以缩短。`改写全文通常会变长（edge 一篇多了约 12%，多出一页），所以有页数上限时一定用后一种，并且每轮编译核对页数。

### 2. 审改动

主代理负责，不交给子代理。

1. 逐条看子代理回复里的第 2、3 项。改变了论断强度或范围的、加了原文没有的内容的，**改回原句**，并记进 `LOG.md`。
2. 逐章读本轮的 diff（`git diff --no-index --word-diff <上一轮> <out>/work`）。重点看：
   - 条件和范围：原来不限定的论断被加了条件，或者反过来；
   - 新出现的因果和推论；
   - 数字换成回指（「this rate」）以后，读者还能不能找到原值。
3. 子代理报告的原文事实问题不在这里修，收进最终报告交给作者。

### 3. 核保真

```
python3 scripts/fidelity.py <out>/orig <out>/work
```

- 退出码 1：引用键或 `\label` 出现或消失、`\ref` 指向原文没有的标签，或者出现了 `NOT IN ORIGINAL` 的数字或公式。逐项处理：新事实改回原文；只是把原文里的值说明白了（例如原文 `$W\in\{1,2,5,10\}$`，改写写出 `$W=10$`），在 `LOG.md` 里写明理由后保留。
- 丢失的数字只报告，不算失败：重复出现的值换成回指可以接受，丢掉一个结果不行。逐个回原文确认。按六原则第 5 条，正文里不再逐个报的计数，必须在论文的某张表或某张图里找得到；找不到就改回。

### 4. 编译和打分

- 有 `latexmk` 时编译 `work/` 的一份临时拷贝。未定义的引用和 ref 必须是 0，页数变化记进 `LOG.md`。
- `python3 scripts/score.py <out>/work --main main.tex --base <上一轮快照>`（第 1 轮用 `<out>/orig`），返回 `style`、`style_within_human_range`、`wording_per_1k`、`words` 和 `gain_top3_share`。
- `gain_top3_share` 是这一轮虚词变化里，变得最多的 3 个词占的比例。服务只给这个比例，不说是哪几个词。按写法改的轮次，变化分散在很多词上（六原则在 6g、edge、anti_nso 上都在 0.32–0.35）；只换几个词的轮次，变化集中在少数几个词上（anti_nso 一轮按词表换词是 0.77）。
- `cp -R <out>/work <out>/r<N>`，并在 `LOG.md` 里记一行：轮次、两个分数、保真结果、回退了几处。

### 5. 停不停

停止判据只看 `style_within_human_range`。`style` 只用来展示。

- 达到 80：停。
- 比此前最好的一轮涨不到 1 分：停。
- 已经跑满 4 轮（含第 1 轮）：停。
- 有页数上限而这一轮超了页：这一轮作废，按「没涨」处理。词数不变也可能超页，因为断行和浮动体的位置会变（anti_nso 第 2 轮词数少了 12 个，从 18 页变成 19 页）。
- `style` 涨了而 `style_within_human_range` 没涨：说明改写把某些词推到了人写论文之外（例如每句都用 `When` 开头），按「没涨」处理，并在 `LOG.md` 里记下。
- `gain_top3_share` 超过 0.5：这一轮是在换词，不是在改写法，涨的分不算，按「没涨」处理，并在 `LOG.md` 里记下。服务返回 `null`（变化太小没法判断）时不适用这一条。

停下以后，采用 `style_within_human_range` 最高的一轮，分数相同时取更早的一轮，因为改得更少。如果最好的不是最后一轮，把那一轮的快照拷回 `work/`。

## 不要做

- **不把分数、分数的变化或检测器的任何信息告诉子代理。**子代理只读规则卡。
- **不根据检测器去写规则。**哪些词被计分、各占多少，不能成为新规则的来源。规则只能从「作者在为读者做什么」出发；新规则先在没参与制定它的论文上验证，并看 `gain_top3_share`。
- 不为了涨分把正文里的数字挪进表格，也不删正文里的结果。两个时代正文的数字密度一样（`PROSE.EXPLAINING_REGISTER` Rationale）。第 5 条只允许一种情况：表或图里已经有的逐个计数，正文不再反复报。
- 不在子代理之外自己逐句改写来追分。主代理只做回退和保真修补。
- 不碰原稿，不 push 到 Git 或 Overleaf，除非作者明确批准。要推到 Overleaf，建议把改写放进 `rewrite/sections/` 再加一个 `main-rewrite.tex`，主稿不动。

## 交付

最终报告给作者：

- 最终版的位置（`<out>/work`）；
- 每轮分数表：两个分数各一列，基线也列进去；
- 保真结果，以及保留下来的每一处 `NOT IN ORIGINAL` 的理由；
- 回退了哪些改动；
- 子代理报告的原文问题，原样列出，作者自己决定改不改；
- 页数变化（如果编译过）。
