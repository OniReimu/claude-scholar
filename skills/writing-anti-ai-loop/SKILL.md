---
name: writing-anti-ai-loop
description: This skill should be used when the user wants a whole LaTeX paper rewritten chapter by chapter with writing-anti-ai until its style stops improving — "逐章改写到文风分不涨", "rewrite the paper chapter by chapter until the score stops rising", "iterate the anti-AI rewrite over the whole manuscript", "把整篇论文按 anti-AI 改几轮". One subagent per chapter per round; the first round runs the full writing-anti-ai skill, later rounds focus on its six explaining-register principles with paragraph-level records and hard-rule checks every round; after every round the orchestrator reverts over-reaching edits, checks fidelity, lock provenance and zero-tolerance candidates across the body, and scores the paper with a p2g style service (P2G_STYLE_URL) when one is configured, stopping when the score within the human range stops rising. Without a service it runs two fixed rounds. Works on a copy; the original manuscript is never edited. For one paragraph or one section, use writing-anti-ai directly.
version: 1.0.0
author: OniReimu
license: MIT
tags: [Writing, AI, Anti-AI, LaTeX, Subagents]
---

# Writing Anti-AI Loop

整篇论文逐章改写，迭代到文风不再变好为止。改写规则全部来自 `writing-anti-ai` 和 `PROSE.EXPLAINING_REGISTER`（六原则），正文段落（含脚注）的句中冒号与分号一律禁止，锁、六原则和保真都不能豁免。本 skill 只负责编排：分章、派子代理、每轮审改动、核保真、核规则、打分、决定停不停。

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
   - **章节功能与粒度**：每章先写一句读者任务。摘要交代问题、方法、规模、核心发现和贡献，只留理解发现必需的数字、条件和边界，不逐项展开原因、区间和次级计数。结论交代确立了什么、边界与下一步，不复演实验。方法和结果讲步骤、证据与关系；讨论讲行动者和适用条件。
   - 减少重复细节前，逐项记录同一事实在正文或表图中的准确位置。唯一结果、必要限定和核心发现不能删。长度和数字量只作诊断，不按范例设配额。MSR 摘要从 333 词、18 个数字缩到 167 词、6 个数字，结论从 216 缩到 108 词；分数略降，却更符合作者要求（E12）。
3. **基线**：`python3 scripts/score.py <out>/orig --main main.tex`，记进 `LOG.md`。

## 每一轮

### 1. 并行改写

每章一个全新子代理（Agent tool，`general-purpose`），**在同一条消息里一起派出去**。子代理之间不共享上下文，每轮都是新的子代理。

- **第 1 轮**：完整执行 `writing-anti-ai`，先按六原则改，再按 anti-AI 规则改，然后停手。
- **第 2 轮起**：以六原则（`$CS/policy/rules/prose-explaining-register.md`）为改写主任务，不重跑完整润色。逐段核对作者、关系、目的、条件、强度和真实对照是否清楚。关系藏在句序或名词串里时，可调句序、拆合句、重组段落；单句合格不等于整段讲清。已讲清的段落可不动。硬约束和新增改动每轮复核。已测稿件支持这个方向，不足以推断所有稿件重复完整 skill 都无益（E8、E9）。

子代理提示词模板（把 `{}` 换掉）。每条任务级硬约束都带规则 ID 或作者指令的可定位来源；工作范围约束注明本 skill 的小节。发出前核对实际 prompt 与模板的差异，新增条款缺来源就不发送。不从一次回退推导新的句式禁令或保留令，也不要求一组自然语言句子用同一种写法；术语和标签可以一致。MSR 的自加保留条款和整组统一写法分别冻结了违规对照、造成七句同框架（E12）。

```
你负责改写 {file}（{chapter}），只改这一个文件{abstract_note}。
先读 {CS}/policy/rules/prose-explaining-register.md 和 {CS}/policy/rules/prose-negation-contrast.md。
{round_rule}
硬约束：
- [来源：writing-anti-ai-loop §准备、§并行改写] 只用 Edit 改 {file}。不建任何文件，不改其他文件，不跑会写文件的命令。
- [来源：writing-anti-ai-loop §核保真] 不增删事实、数字、引用、公式、\label 和 \ref 的目标。figure / table 环境、caption、公式、代码、\cite 键都不动。自定义宏（例如只在某个版本出现的 \extonly{…}）的调用和包住的范围都不动。重复数字可减少提及，但要记录同一论断仍保留的准确位置，不能只凭数字字符串还在。caption / table 不在本次标点检查范围，只报已排除计数；其他受保护正文若违规，只报告冲突，不据此放行。
- [来源：PROSE.EXPLAINING_REGISTER、PROSE.HEDGING_DISCIPLINE] 不改变论断的强度、条件和范围。只用整篇论文已明确给出的解释、原因和关系；跨句或跨节移用要注明来源。同一关系换结构不算新增事实，推测不能当已写明的原因。原意不清时报告。
- [来源：writing-anti-ai-loop §并行改写；有作者要求时填 {length_source}] {length_rule}
- [来源：writing-anti-ai-loop §不要做] 只按规则和读者任务改，不按词频改。
- [来源：PROSE.EXPLAINING_REGISTER、PROSE.NEGATION_CONTRAST、PROSE.MIDSENTENCE_COLON、PROSE.SEMICOLON_RESTRICTION、PROSE.EM_DASH_RESTRICTION] 命题保真始终有效，锁、六原则和保真都不能豁免正文冒号与分号硬禁令。不能兼得时报告具体冲突，不删论断、不报通过。rather than / instead of 按三判据逐条给依据，不能因入锁就保留。
回复（400 字以内）：
1. 改了大约几处；
2. 没改的段落，每段一句：它按哪条原则已经合格。改了的段落不用报，看 diff；
3. 条件、强度、范围和关系有改动的，逐条给出原句和新句；
4. rather than / instead of 按三判据的判定，以及原文里看到的事实问题（只报告，不改）。
```

- `{round_rule}`，第 1 轮：`再读 {CS}/skills/writing-anti-ai/SKILL.md，按它的执行顺序改：先六原则，再 anti-AI 规则，然后停手。`
- `{round_rule}`，第 2 轮起：`以六原则为主任务，逐段过 Check 的六个问题，必要时重组段落。不重跑完整润色；硬约束和新增改动每轮复核。`
- 第 2 轮起，只锁有回退记录的原句。`locks.json` 用对象数组，每条必填 `sentence`（原句）、`orig`（相对台账或绝对路径）、`rollback_round`（正整数）、`rollback_reason`、`proposition`（含条件和强度）。`python3 scripts/guardrails.py locks <out>/locks.json` 核对原句逐字存在于 orig，输出原文哈希和位置；可填 `orig_sha256` 固定基线，变了就拒绝。旧式字符串数组、缺字段或找不到原句都拒绝。
- 机器只能验证原文来源，不能凭五个字段证明真实回退。主代理另核对该轮 diff / 回退记录确实恢复了原句，记录证据位置；未经这一步不得派发。不要给旧数组补个来源标签就算重建完成。
- 原句含零容忍候选时标「只锁命题」，不能要求逐字保留；换成合规结构。其余锁默认不动，若阻碍六原则重组，改写者给保义候选和理由，主代理只审保真。MSR 把 r4 的冒号新句误锁为原句，真正原句也可能带违规形式（E12）。
- 作者批准的事实修复单列语义基线、批准来源；后续保真对照批准版本，orig 不动，不冒充回退原句。
- `{abstract_note}`，摘要那一章：`，并且只改 abstract 环境`；其他章为空。
- `{length_rule}`，默认：`这一章的篇幅变化控制在 ±15% 以内。`稿件卡在页数上限时：`这一章不能变长，词数不超过原文；可以缩短。`改写全文通常会变长（edge 一篇多了约 12%，多出一页），所以有页数上限时一定用后一种，并且每轮编译核对页数。

### 2. 审改动

主代理负责，不交给子代理。

1. 核对逐段完成记录、风险项与 diff。MSR 每章只修 2–5 处并未说明整段已讲清，单独强调虚词也未显示额外收益（E12）。逐条看第 3、4 项。改变了论断强度或范围的、加了原文没有的内容的，**改回原句**，并记进 `LOG.md`。
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

### 4. 核规则

每轮审改动、核保真后执行，过关才打分。第 2 轮不重跑完整润色，不等于停查硬约束。

```bash
bash "$CS/policy/lint.sh" <out>/work
python3 scripts/guardrails.py <out>/work/main.tex --json <out>/rules-rN.json
```

- 完整 lint 与候选枚举不能互相代替。枚举从摘要到参考文献前的全部正文，含脚注、列表、宏参数和输入文件；caption 和 table / tabular（含输入的表格文件）排除，只报已排除计数，不计候选或违规。仅非散文豁免也计数，含数学、代码及 `\texttt` / `\verb`、`\;`、URL、时间、引用键和结构标题冒号（含段首 `\textbf{TV-1: Static extraction scope. }` / `\paragraph{...}`），不能整块跳过盒子、括号或未改内容。
- JSON 记录文件、行号、规则 ID、源哈希、正文范围与各类计数；简表输出到 stderr。退出码 0 为无候选，1 为有候选，2 为解析或覆盖失败。对照候选仍需按规则卡判档；脚本不代替语义判定，也不执行 TeX；支持静态 input/include 与源码中的宏参数。调用的自定义宏若在定义中藏了候选，报覆盖错误，展开后复核；不能据此宣称生成文本已全覆盖。
- 正文冒号和分号在明确豁免之外一律记「违规」，不再有「待判断」；列表末尾和直接引语不豁免。em-dash 照旧，档 A 对照仍可待判。非散文记明确豁免及理由，caption / table 仅报已排除计数。违规与待判断都为 0 才通过；无输出、空正文、空章节、漏摘要或解析失败一律失败。范围内的受保护正文也要报告冲突，不能静默放行。分号修法为断句或按规则卡改成逗号加连接词。
- 回退、合并、换快照后重新跑最终文件。旧报告不可复用；`--check-report <旧报告.json>` 可核对输入哈希，变化时报覆盖错误，仍会重新枚举。
- 失败时只把原句、位置和规则交回改写者，不给分数或检测器信息。每轮至多补修两次；仍失败则保留最近合格版本，汇总未解决项，不报完成。
- 批量清标点或展开解释后，组合检查新句是否套同一框架。按 `PROSE.CAUSAL_CONNECTIVE` 判断关系，不因词多换同义词；同一关系可用同一连接词，不同关系不能靠同一词掩盖。新增计数、因果、闭合列举和情态逐项回原文核对，需要重写的交回改写者。

作者 2026-10-21 定为硬规则，用来取代逐条判断；MSR 一稿因为逐条判断加上锁，违规句一路存活了 11 组。

理由：MSR H 组自报零命中，摘要的 “]; 54.2” 仍在；G 检出 14 处不证明覆盖完整。F 新增 11 个 therefore、H 七句同框架说明组合验收也不能省；这些次数不构成新禁令（E12）。

### 5. 编译和打分

- 有 `latexmk` 时编译 `work/` 的一份临时拷贝。未定义的引用和 ref 必须是 0，记录正文终页、参考文献起页和总页数，不只记总页数。有页数要求时按编译验收；临时放开要记作者来源，标「内容候选，版面未通过」，不能称可投稿完成版。
- 有 PDF 时查段末一两词的悬浮行、溢出和浮动体。先尝试不改文字的排版修复；改文字就重跑保真和规则。不删目的、情态、条件或精确限定来省一行。MSR 后补发现 12 个悬浮行，删 engineering 和 over time 又须回退（E12）。
- `python3 scripts/score.py <out>/work --main main.tex --base <上一轮快照>`（第 1 轮用 `<out>/orig`），返回 `style`、`style_within_human_range`、`wording_per_1k`、`words` 和 `gain_top3_share`。
- `gain_top3_share` 是这一轮虚词变化里，变得最多的 3 个词占的比例。服务只给这个比例，不说是哪几个词。按写法改的轮次，变化分散在很多词上（六原则在 6g、edge、anti_nso 上都在 0.32–0.35）；只换几个词的轮次，变化集中在少数几个词上（anti_nso 一轮按词表换词是 0.77）。
- `cp -R <out>/work <out>/r<N>`，并在 `LOG.md` 里记一行：轮次、两个分数、保真结果、回退了几处。

### 6. 停不停

停止判据只看 `style_within_human_range`。`style` 只用来展示。

- 达到 80：停。
- 比此前最好的一轮涨不到 1 分：停。
- 已经跑满 4 轮（含第 1 轮）：停。
- 有页数上限而这一轮超了页：这一轮作废，按「没涨」处理。词数不变也可能超页，因为断行和浮动体的位置会变（anti_nso 第 2 轮词数少了 12 个，从 18 页变成 19 页）。
- `style` 涨了而 `style_within_human_range` 没涨：说明改写把某些词推到了人写论文之外（例如每句都用 `When` 开头），按「没涨」处理，并在 `LOG.md` 里记下。
- `gain_top3_share` 超过 0.5：这一轮是在换词，不是在改写法，涨的分不算，按「没涨」处理，并在 `LOG.md` 里记下。服务返回 `null`（变化太小没法判断）时不适用这一条。例外：这一轮重点执行的是第 5 条最后一款（规则本身点名了措辞，依据是同类句子的人写用法，见 evidence E11），集中度必然高，不用 0.5 判；改为核对被点名的词是否大多落在人写范围内、没有被点名的词是否基本没动，并在 `LOG.md` 里记下。

停下以后，只在保真、规则和当前页数要求都通过的快照中，选 `style_within_human_range` 最高的一轮；同分取更早的一轮。未配置服务时取最近的合格轮。复制回 `work/` 后重新核规则；没有合格快照时交付未通过项，不称完成。未能编译时写明版面未验收，有页数要求就不能称合格。

## 不要做

- **不把分数、分数的变化或检测器的任何信息告诉子代理。**子代理只读规则卡。
- **不根据检测器去写规则。**哪些词被计分、各占多少，不能成为新规则的来源。规则只能从「作者在为读者做什么」出发；新规则先在没参与制定它的论文上验证，并看 `gain_top3_share`。
- 不为了涨分把正文里的数字挪进表格，也不删正文里的结果。两个时代正文的数字密度一样（`PROSE.EXPLAINING_REGISTER` Rationale）。第 5 条只允许一种情况：表或图里已经有的逐个计数，正文不再反复报。
- 不发送无规则或作者来源的硬约束，不给自然语言句组规定统一句式。
- 不在子代理之外自己逐句改写来追分。主代理只做回退和保真修补。
- 不碰原稿，不 push 到 Git 或 Overleaf，除非作者明确批准。要推到 Overleaf，建议把改写放进 `rewrite/sections/` 再加一个 `main-rewrite.tex`，主稿不动。

## 交付

最终报告给作者：

- 最终版的位置（`<out>/work`）；
- 全文每轮 `style_within_human_range` 表，普通 `style` 另列，含基线；子集诊断按「准备」中的口径另表；
- 保真结果，以及保留下来的每一处 `NOT IN ORIGINAL` 的理由；
- 回退了哪些改动；
- 子代理报告的原文问题，原样列出，作者自己决定改不改；
- 规则台账、源哈希、锁来源与未解决项；
- 正文终页、参考文献起页、总页数和布局验收状态（未编译则注明）。
