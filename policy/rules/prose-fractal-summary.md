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

## Rationale

LLM 生成长文时会在每个层级复制同一套"预告—展开—回顾"骨架，于是一篇论文里同样的信息被讲三遍：节标题讲一遍，开头的预告句讲一遍，结尾的回顾句再讲一遍。这种自相似的冗余是 AI 长文最稳定的结构指纹之一，而且它在逐句检查中完全合格——每一句语法正确、内容真实，问题只在层级上可见。

对读者而言它同样是负担：`\section{System Model}` 后面紧跟 `In this section, we present our system model.` 是零信息句，占掉了顶会最贵的一行版面。审稿人读到的第一句应该是主张，不是目录。

学术写作里真正需要的前向引用应由交叉引用承担（`\Cref{sec:eval} reports ...`），而不是由叙述性预告承担。

## Check

- **regex 搜索**：五条 pattern 覆盖开节预告 / 收节回顾 / 层间缝合
- **检查范围**：`.tex` 正文
- **豁免**：
  - Introduction 末尾的一处 roadmap（超出 `roadmap_allowance` 的第二处即违规）
  - Survey/SoK 中 taxonomy 章节的导航段（`SOK.TAXONOMY_REQUIRED` 要求的结构说明）
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
