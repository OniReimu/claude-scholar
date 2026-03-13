---
id: PROSE.SUBSECTION_COMPLETENESS
slug: prose-subsection-completeness
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_style
enforcement: doc
params: {min_paragraphs: 2}
conflicts_with: []
---

## Requirement

每个 subsection 至少包含 **2 个段落**。避免单段 subsection，除非内容确实无法进一步展开。

如果某个 subsection 只有一段，考虑：(1) 合并到上级 section，(2) 扩展内容，(3) 与相邻 subsection 合并。

## Rationale

单段 subsection 通常意味着分节粒度过细或内容不充分。经典工程论文中，每个 subsection 承载一个完整的子话题，需要足够的段落来展开。

## Check

- **LLM 检查**: 统计每个 `\subsection{}` 到下一个同级或上级标题之间的段落数
- **阈值**: < 2 段则标记为违规
- **排除**: 包含大型公式、算法或表格的 subsection 可豁免

## Examples

### Pass

```latex
\subsection{Threat Model}

We consider a semi-honest adversary who controls the central server.
The adversary follows the protocol but attempts to infer private
information from the received model updates.

We assume that at most $t$ out of $n$ participants may collude with
the adversary. The remaining participants are honest and follow the
protocol faithfully.
```

### Fail

```latex
\subsection{Threat Model}

We consider a semi-honest adversary.

\subsection{Design Goals}

We aim to achieve privacy.
```
