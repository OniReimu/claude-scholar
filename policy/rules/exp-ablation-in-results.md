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
conflicts_with: []
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
