---
id: EXP.RESULTS_SUBSECTION_STRUCTURE
slug: exp-results-subsection-structure
severity: warn
locked: false
layer: core
artifacts: [text, table, figure]
phases: [writing-experiments, self-review, revision]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: [EXP.EXPERIMENT_ROLE]
constraint_type: guidance
autofix: none
---

## Requirement

Experimental Results 中每个 `\subsubsection` 必须满足以下三个结构要求：

1. **第一句引用对应的 figure 或 table**（使用 `\ref`）
2. **包含 >=2 个实质段落**（分析和讨论实验结果）
3. **以 Takeaway box 结尾**（使用 `\fbox` 提炼核心发现）

## Rationale

结构化的结果子节便于审稿人快速理解每个实验的目的和结论。引用 figure/table 建立文图关联，Takeaway box 提炼核心发现。

## Check

- **提取 `.tex` 正文的正确方法**：见 `policy/references/tex-prose-extraction.md`。手搓扫描器的四个典型错误（`split('%')` 在 `$95\%$` 处截断、剔数学时 `$` 奇数配对吞掉整段、逐行扫描漏掉被硬换行劈开的短语、两遍大小写策略不一致）都会产生**假的「已清零」结论**

- **LLM 检查**: 每个 `\subsubsection` 是否满足三个结构要求
- **要点 1**: 第一句是否包含 `Table~\ref{...}` 或 `Figure~\ref{...}`
- **要点 2**: 是否有 >=2 个实质段落（非空行分隔的段落）
- **要点 3**: 子节末尾是否有 `\fbox{Takeaway: ...}` 或等价的 Takeaway box

## Examples

### Pass

```latex
\subsubsection{Main Comparison}
Table~\ref{tab:main} summarizes the performance of all methods on three benchmarks.

Our method achieves the best results on all three datasets. On Dataset A, we observe
a 15.3\% improvement over the strongest baseline, with particularly large gains on
the challenging subset.

The improvement is consistent across different evaluation metrics. As shown in the
rightmost columns of Table~\ref{tab:main}, both precision and recall benefit from
our proposed module.

\fbox{Takeaway: Our method consistently outperforms all baselines across three
benchmarks, with an average improvement of 12.7\%.}
```

### Fail

```latex
\subsubsection{Main Comparison}
We compare our method with baselines. Results are shown in Table 1. Our method
achieves the best results.
% 问题：(1) 未使用 \ref 引用表格, (2) 仅 1 段, (3) 无 Takeaway box
```

## Conflicts

- `EXP.EXPERIMENT_ROLE` — 本卡管**呈现格式**（引用图表、≥2 段、takeaway 收尾），那条在本卡之前生效：不承担职责的子节应先走处置阶梯（重设计 → 弱化 → 删除），补齐三项格式只是把 padding 包装得更整齐
