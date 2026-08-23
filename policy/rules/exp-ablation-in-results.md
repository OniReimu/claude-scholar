---
id: EXP.ABLATION_IN_RESULTS
slug: exp-ablation-in-results
severity: warn
locked: false
layer: core
artifacts: [text]
phases: [writing-experiments, self-review, revision]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: [EXP.EXPERIMENT_ROLE, PAPER.OUTCOME_LOGIC]
constraint_type: guidance
autofix: none
---

## Requirement

消融实验（Ablation Study）必须放在 Experimental Results section（通常为 Section 5）中，作为 `\subsubsection`。禁止将消融实验放在 Discussion section。

## Rationale

消融实验是方法设计的实证支撑，属于实验结果的一部分。放在 Discussion 中会与分析性讨论混淆，也不利于审稿人快速定位。

## Check

- **提取 `.tex` 正文的正确方法**：见 `policy/references/tex-prose-extraction.md`。手搓扫描器的四个典型错误（`split('%')` 在 `$95\%$` 处截断、剔数学时 `$` 奇数配对吞掉整段、逐行扫描漏掉被硬换行劈开的短语、两遍大小写策略不一致）都会产生**假的「已清零」结论**

- **LLM 检查**: "ablation" 相关内容是否出现在 Experiments section 而非 Discussion section
- **要点**: `\subsubsection{Ablation Study}` 应嵌套在 `\section{Experiments}` 或 `\section{Experimental Results}` 内部

## Examples

### Pass

```latex
\section{Experiments}
  \subsection{Main Results}
  ...
  \subsubsection{Ablation Study}
  To understand the contribution of each component, we conduct ablation experiments...
```

### Fail

```latex
\section{Discussion}
  \subsection{Ablation Study}
  We also conduct ablation experiments to analyze...
```

## Conflicts

- `EXP.EXPERIMENT_ROLE` — 本卡管消融**放在哪一节**，那条管这个消融**是否该存在**。先按那条确认它承担职责 2（说明优势从何而来），再按本卡放进 Experimental Results
- `PAPER.OUTCOME_LOGIC` — 消融是成果逻辑的组成部分；那条的流水账禁令不得用来删消融，只有「只交代作者行踪」的段落才在禁令范围内
