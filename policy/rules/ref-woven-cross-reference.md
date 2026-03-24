---
id: REF.WOVEN_CROSS_REFERENCE
slug: ref-woven-cross-reference
severity: warn
locked: false
layer: core
artifacts: [text]
phases: [writing-background, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {max_words_dangling: 15}
conflicts_with: []
constraint_type: guardrail
autofix: assisted
---

## Requirement

每个 cross-reference（Fig.、Table、\S、Eq.、Algorithm）必须嵌入一个承载分析内容的句子中。核心判断标准：**把 `\ref{}` 删掉后，这句话是否仍传达有意义的论点？** 如果删掉后句子没有实质内容，就违规。

### 禁止模式

1. **Dangling tail reference** — 单独一句只为指向 float：
   - `Fig.~\ref{fig:X} illustrates the architecture.`
   - `Table~\ref{tab:Y} summarizes the results.`
   - `\S\ref{sec:Z} describes the method.`

2. **Orphan paragraph** — 单句段落只为引入 float：
   ```latex
   \subsubsection{Overview}
   Table~\ref{tab:overview} summarizes the study design.
   ```

3. **Sentence-final bare reference** — 句末追加裸引用，前文已闭合：
   ```latex
   The gap is large. See Fig.~\ref{fig:gap}.
   ```

### 可接受模式

| 模式 | 示例 |
|------|------|
| 句中括号引用 | `Diversity ranges from 0 to 6 classes (Fig.~\ref{fig:div}).` |
| 作为分析主语 | `Fig.~\ref{fig:heatmap} reveals two patterns: (1)~workflow dominance and (2)~multi-layer clustering.` |
| 从属从句 | `As Table~\ref{tab:baseline} shows, all four tools remain blind to model artifacts.` |
| 承接上文 | `...remain an identified recall risk. Fig.~\ref{fig:prop} traces how upstream changes cascade through multiple artifact layers.` |
| 带上下文的引入 | `To give a repository-level view, Table~\ref{tab:big} lists per-repository counts across the Tier~B sample.` |

## Rationale

Dangling reference 是 AI 生成文本的常见特征——模型倾向于用一整句话指向图表而不加任何分析。这种写法浪费版面空间，降低信息密度，并打断论证的连贯性。顶级论文中 cross-reference 总是嵌入在传递论点的句子里。

## Check

- **LLM 检查**: 对每个包含 `\ref{` 的句子，判断删除引用部分后是否仍有实质论点
- **Lint heuristic**:
  - 以 `Fig.~\ref` / `Table~\ref` / `\S\ref` 开头，且 ≤ 15 词，且句末无从句或枚举 → 标记
  - 句末 bare `(Fig.~\ref{...}).` 或 `(Table~\ref{...}).`，且前文已闭合 → 标记
  - 单句 `See Fig.~\ref` / `See Table~\ref` → 标记

## Examples

### Pass

```latex
Diversity ranges from 0 to 6 classes (Fig.~\ref{fig:diversity}),
with the majority of repositories containing only 1--2 artifact types.

Fig.~\ref{fig:heatmap} reveals two distinct patterns:
(1)~workflow-dominated repositories and (2)~multi-layer clustering
where documentation co-occurs with testing artifacts.

As Table~\ref{tab:baseline} shows, all four static analysis tools
remain blind to model-related artifacts, achieving 0\% recall on
the \textsc{ModelCard} and \textsc{DataSheet} categories.
```

### Fail

```latex
Fig.~\ref{fig:architecture} illustrates the overall architecture.

Table~\ref{tab:results} summarizes the results.

The performance gap is significant. See Fig.~\ref{fig:gap}.

\subsubsection{Study Design}
Table~\ref{tab:design} shows our study design.
```
