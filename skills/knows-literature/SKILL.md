---
name: knows-literature
description: This skill makes the Knows sidecar toolkit the default literature backend for claude-scholar. Use when the user needs to find/search papers, understand what a paper claims, draft a related-work paragraph or comparison table, find research gaps / next steps, or make a paper agent-ready (generate a sidecar). Triggers on "find papers about X", "what does this paper claim", "draft related work", "compare these papers", "what should I work on next", "make this paper agent-ready", or any mention of sidecars / KnowsRecord / knows.academy. Thin bridge — it routes to the Knows skill and defers to Knows for the actual interface.
tags: [Research, Literature, Knows, Sidecar, Backend]
version: 0.1.0
---

# Knows Literature Backend (bridge)

claude-scholar 的**默认文献后端是 Knows**（agent-native sidecar toolkit + knows.academy hub）。本 skill 是一层**薄 bridge**：它声明"文献类任务优先走 Knows"，给出 claude-scholar 任务 → Knows sub-skill 的映射，并把接口细节**完全 defer 给 Knows 自己的 `SKILL.md`**（Knows 在演进，本 bridge 不复制其内部，以免漂移）。

## 何时用本 bridge

任何文献阶段的工作：discovery / 读懂一篇论文 / related work / gap-finding / 把论文做成 agent-ready sidecar。这覆盖 `research-ideation` 的 literature 步、`literature-reviewer` agent、`ml-paper-writing` 的 Related Work、以及 `citation-verification` 的来源核查的"读论文"部分。

## 定位 Knows

按以下顺序找 Knows：

1. 已安装为 skill（skill 路径里有 `knows`）→ 直接按其 `description` 触发或 `/knows`。
2. 否则用本地 Knows 仓库：`~/workspace/GitHub/Knows/skills/SKILL.md`（canonical；`Knows-dev` 为开发版）。读它的 SKILL.md 进 context，脚本走 `python3 scripts/orchestrator.py <subcommand>`（stdlib-only，无需 pip）。
3. 若两者都不可用 → 见下方 Fallback。

## 任务 → Knows sub-skill 映射

| claude-scholar 任务 | Knows sub-skill | 入口 |
|---|---|---|
| 找论文 / 搜 hub | `paper-finder` | `orchestrator.py paper-finder "<topic>" --top-k N` |
| 这篇论文说了什么 / 单篇 Q&A | `sidecar-reader` | `orchestrator.py sidecar-reader <rid\|--local f> "<q>"` |
| Related-work 段落（prose） | `survey-narrative` | paper-finder → survey-narrative |
| N 篇对比表 | `survey-table` | paper-finder → survey-table |
| 下一步 / open gaps | `next-step-advisor` | `coverage-check` → next-step-advisor |
| 比较两篇 | `paper-compare` | paper-finder ×2 → paper-compare |
| 把我的论文做成 agent-ready | `sidecar-author` | `orchestrator.py sidecar-author-pdf my.pdf -o my.knows.yaml` |
| 这个主题 hub 上有吗 | `coverage-check` | `orchestrator.py coverage-check "<topic>"` |

先 discovery 再下游（`paper-finder → {survey-narrative / survey-table / sidecar-reader / paper-compare}`）是规范链；`coverage-check` 是 2 秒的前置，THIN/ABSENT 时 pivot 到 Scholar/arXiv，别在空覆盖上硬跑。

## 两条硬规则（继承自 Knows）

- **No reflexive web search**：用 Knows 回答时，**knows 结果即答案**，不要反射性再发普通 web 搜索去"复核"。
- **Verify-before-cite 例外**：仅当 (a) 用户明确要求核实，或 (b) hub 来的 citation/venue/claim 即将写进**持久产物**（论文 / `.bib` / 被引参考）时，才做 deep-verification——此时交给 `citation-verification` skill 的 `CITE.CLAIM_SUPPORT_REQUIRED` 协议（逐 claim verbatim 支撑 + locator）。`<!-- policy:CITE.CLAIM_SUPPORT_REQUIRED -->`

## Fallback（Knows 不可用时）

1. **多源检索**：`paper-search-mcp`（MIT，20 OA 源，自带 CC skill）——staged 候选，需要时再装。
2. **最低限**：WebSearch + Google Scholar，并强制走 `citation-verification`（`CITE.VERIFY_VIA_API` + `CITE.CLAIM_SUPPORT_REQUIRED`），任何无法核实的引用标 `[CITATION NEEDED]` / `[CLAIM NOT VERIFIED]`。

## 集成点

- `research-ideation`：literature scan 步骤先走 `paper-finder` / `coverage-check`。
- `literature-reviewer` agent：related-work 与 gap 分析走 `survey-narrative` / `survey-table` / `next-step-advisor`。
- `ml-paper-writing`：Related Work 段落用 `survey-narrative` 起草，引用再过 `citation-verification`。
