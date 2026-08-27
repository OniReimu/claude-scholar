---
id: PROSE.FRACTAL_SUMMARY
slug: prose-fractal-summary
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-background, writing-system-model, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: regex
enforcement: lint_script
params: {roadmap_allowance: 1}
conflicts_with: [PROSE.ANNOUNCEMENT_SENTENCE, PROSE.SEMANTIC_IDLING]
constraint_type: guardrail
autofix: assisted
lint_patterns:
  - pattern: "(?i)\\bIn this (section|subsection|chapter|part),? we\\b"
    mode: match
  - pattern: "(?i)\\bThis (section|subsection) (presents|describes|introduces|discusses|provides|covers|is organized|begins by)\\b"
    mode: match
  - pattern: "(?i)\\bAs (we have|we've) (seen|discussed|shown|noted|mentioned|established)\\b"
    mode: match
  - pattern: "(?i)\\bHaving (discussed|presented|introduced|established|described)\\b[^.]{3,60},\\s*we\\b"
    mode: match
  - pattern: "(?i)\\b(in what follows|before we (proceed|continue|move on)|to (summarize|recap),)"
    mode: match
lint_targets: "**/*.tex"
---

## Requirement

删除每一层结构上的预告句与回顾句。禁止 `In this section, we ...` / `This subsection describes ...` 开节，禁止 `As we have seen ...` / `To summarize,` 收节，禁止 `Having presented X, we now ...` 式的层间缝合句。

节的第一句直接进入内容，节的最后一句停在最后一个具体结论上。

**唯一豁免**：Introduction 结尾的一处 roadmap 段落（`roadmap_allowance`，默认 1），且仅在目标 venue 的惯例要求时保留。

**重锚不是预告。** 一节要用到几节之前定义的构件（RQ 集合、命名术语、框架阶段）时，在这里把它重新说一遍是正常的人类写法，不是本卡要抓的自相似冗余。判据两条，**都要成立**：

1. **自足**：重新提及的那段，读者不持有「标签 → 内容」的绑定也读得懂。`The study asks three questions. First, what does a reviewer's visible action contain? ...` 自足；`RQ1 establishes which public events can be interpreted. RQ2 asks whether ...` 不自足——它预设读者还记着 `RQ1` 指什么，而那正是重锚本该还清的债。**只点标签的形式在任何距离下都不合格。**
2. **距离足以让读者本来要回翻**。判回翻成本，不要数数。**不设数字阈值**：写成 `N ≥ 2` 会让判官去数节，而该问的是"读者要不要翻回去看"。至少跨过一个节边界是下限，不是闸门。**按节量，不按页**——页码随排版而变。

紧贴在它所预告的标题上方两行的段落，即使内容复述得很完整，仍然是本卡的指纹：读者刚看过，不存在回翻成本。

**分工**：本卡只管**一节之内**的位置构型（标题／预告／回顾三处重复）。**跨节**的同一命题多个 home 归 `claim-architecture-review` 的 **P2**——那里才是重锚真正会被误杀的地方，因为 ledger 只看得见"一个命题两个 home"，而它的设计目标就是把这种情况压成一个。P2 保留一个 re-anchor home 时要求写出 canonical home 的位置。

⚠️ **距离只解除本卡的构型判定，不解除 `PROSE.SEMANTIC_IDLING` 形态 A**：一句什么都没断言的回顾，隔多远都还是空转。两条分别过。

## Rationale

LLM 生成长文时会在每个层级复制同一套"预告—展开—回顾"骨架，于是一篇论文里同样的信息被讲三遍：节标题讲一遍，开头的预告句讲一遍，结尾的回顾句再讲一遍。这种自相似的冗余是 AI 长文最稳定的结构指纹之一，而且它在逐句检查中完全合格——每一句语法正确、内容真实，问题只在层级上可见。

对读者而言它同样是负担：`\section{System Model}` 后面紧跟 `In this section, we present our system model.` 是零信息句，占掉了顶会最贵的一行版面。审稿人读到的第一句应该是主张，不是目录。

学术写作里真正需要的前向引用应由交叉引用承担（`\Cref{sec:eval} reports ...`），而不是由叙述性预告承担。

## Check

- **regex 搜索**：五条 pattern 覆盖开节预告 / 收节回顾 / 层间缝合。**pattern 看不见重锚判定**——它只认 `In this section, we` 那一族措辞，对 `RQ1 establishes ...` 和 `The study asks three questions.` 都不报。自足性与距离都不是正则能判的，机械层最多标出「同一构件的相邻两次提及」作候选，本卡不为此新增 pattern
- **检查范围**：`.tex` 正文
- **豁免**：
  - Introduction 末尾的一处 roadmap（超出 `roadmap_allowance` 的第二处即违规）
  - Survey/SoK 中 taxonomy 章节的导航段（`SOK.TAXONOMY_REQUIRED` 要求的结构说明）——这一条现在是上面「重锚」判据的一个特例：taxonomy 导航段之所以合法，正是因为它自足且距上次说明隔了很远。保留它是为了让 SoK 作者不必逐条论证
  - `\Cref{}` / `\ref{}` 承载的前向引用（这是交叉引用，不是预告叙述）
  - 直接引语、reviewer comment 原文
- **删除测试**：删掉候选句后，本节信息是否零损失？零损失即违规。

## Examples

### Pass

```latex
\subsection{Threat Model}
The adversary controls $k$ of the $n$ aggregators and observes every message
on the broadcast channel, but cannot forge signatures.
```

### Fail

```latex
\subsection{Threat Model}
In this subsection, we present our threat model. This subsection describes
the adversary's capabilities and the assumptions we make.
The adversary controls $k$ of the $n$ aggregators...
As we have seen, the threat model constrains what the adversary can do.
```

## Conflicts

- `PROSE.ANNOUNCEMENT_SENTENCE` 管单句层面的承载力（短句只做预告标签、删掉零损失），本卡管**结构层面**的自相似冗余（同一信息在标题/预告/回顾三处重复）。一个句子可能同时触发两条：先按本卡删，删不掉的再按 ANNOUNCEMENT_SENTENCE 改写成承载主张的句子
- `PROSE.PARAGRAPH_TOPIC_SENTENCE` 要求段首为 topic sentence——topic sentence 陈述本段**主张**，预告句陈述本段**将要做什么**，二者不冲突
