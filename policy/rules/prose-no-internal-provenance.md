---
id: PROSE.NO_INTERNAL_PROVENANCE
slug: prose-no-internal-provenance
severity: warn
locked: false
layer: core
artifacts: [text, table, figure]
phases: [writing-background, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: []
constraint_type: guardrail
autofix: assisted
---

## Requirement

论文正文（prose、caption、脚注）不得出现内部工作痕迹：

1. **内部产物名**：脚本名（`plot_fig3.py`）、内部文件路径（`figs/_preview.png`）、renderer/工具链名、DPI 自检结果、artifact bundle 备注；
2. **写作过程元话语**：`this draft`、`this manuscript aims to`、`in the camera-ready version we will`、`(placeholder)`、`[TODO]` 残留；
3. **placeholder / 验证状态标记**：`[CITATION NEEDED]`、`[CLAIM NOT VERIFIED]` 等 workflow 标记在投稿/camera-ready 前必须全部清除或落实。

这些内容属于 README、artifact spec、review 记录、audit appendix 或代码注释。

**例外（不属于内部工作痕迹）**：

- 论文本身以审计方法 / artifact evaluation 为主题、需要实名引用内部工件时；
- `EXP.FABRICATED_RESULTS_CAPTION_DISCLOSURE` / `EXP.RESULTS_STATUS_DECLARATION_REQUIRED` **要求**的科学状态披露（如 caption 声明结果为 simulated / projected / 非实跑）——那是科学 claim 的一部分，不是工作痕迹，两条 EXP 规则优先。

## Rationale

Agentic 写作的高发缺陷：生成流水线把自己的 provenance（脚本名、预览路径、自检备注）带进正文和 caption，人工通读时容易漏掉，被审稿人抓到会直接暴露"AI 生成未经通读"。工作痕迹与论文叙事分离，是投稿前的硬性卫生要求。（改编自 DELONG-L/Academic-Paper-Skills 的 paper body versus audit trail 约定，MIT。）

本条是 `PROSE.*` 中唯一 `layer: core` 的规则：其余 PROSE 规则约束的是文风偏好，归属 `domain` 层；本条约束的是投稿完整性/卫生底线（工作痕迹一旦漏进正文即是硬伤，与具体 domain、venue 无关），因此跨所有 domain 与 venue 生效，归 `core`。

## Check

- **LLM 检查**：扫描 `.tex` 正文与 caption 中的脚本文件名、相对路径、渲染工具名、DPI/自检字样、写作元话语、未清除的 placeholder 标记
- 与 `CITE.CLAIM_SUPPORT_REQUIRED` 联动：其产生的 `[CLAIM NOT VERIFIED]` 标记在 submission gate 前必须清零

## Examples

### Pass

```latex
Figure~\ref{fig:pipeline} shows the three-stage pipeline; exact
per-stage latencies are reported in Table~\ref{tab:latency}.
```

### Fail

```latex
% caption 泄漏内部 provenance + 元话语
\caption{Pipeline overview (rendered by autofigure2.py at 300 DPI,
preview checked in figs/_preview.png). In this draft we use
placeholder numbers pending the final run. [CLAIM NOT VERIFIED]}
```
