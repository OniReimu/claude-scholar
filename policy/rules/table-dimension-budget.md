---
id: TABLE.DIMENSION_BUDGET
slug: table-dimension-budget
severity: warn
locked: false
layer: core
artifacts: [table]
phases: [writing-experiments, self-review, revision, camera-ready]
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

表格是服务论点的紧凑对比物，不是所有可得维度的仓库。默认预算：

1. **对比表（含 Related Work 比较表）默认 3–4 个高信号维度**；第 5 维需确属必要；≥6 维必须有显式理由（写进 artifact spec 或 PR 说明）。SoK 综合大表（`SOK.BIG_TABLE_REQUIRED`）天然满足"显式理由"——taxonomy 全维覆盖是 SoK 的贡献本体，不受本预算约束，但仍应剪掉不承载 taxonomy 轴的装饰列。
2. **默认单栏 `table`**。上 `table*` 之前先做两件事：剪掉不支撑论点的列；缩短冗长表头。只有剪完仍会误导或不可读时才放宽到双栏。
3. 维度选取必须跟随论文声明的 gap，**不得只为衬托己方方法挑维度**。
4. 不加 Notes 块，除非没有它表格无法读懂。

## Rationale

初稿表格倾向保留全部可得维度，导致被迫上 `table*`、字号压缩、读者抓不到对比轴。先定维度预算再排版，比排版阶段硬塞更省页数也更有说服力。（改编自 DELONG-L/Academic-Paper-Skills 的 placement and dimension budget rule，MIT。）

## Check

- **LLM 检查**：对比表列数 >5 时，核对是否有显式理由；`table*` 环境核对是否先做过剪列/缩表头；对比维度是否与论文 gap 对应而非单方面有利于 Ours 行

## Examples

### Pass

```latex
% 单栏，3 个维度直接对应论文声明的 gap（boundary / scoped memory / leakage metric）
\begin{table}[t]
  \centering
  \caption{Comparison by boundary mechanism.}\label{tab:related}
  \resizebox{\columnwidth}{!}{%
  \begin{tabular}{lccc}
    \toprule
    Approach & Boundary & Scoped Memory & Leakage Metric \\
    \midrule
    ...
    \bottomrule
  \end{tabular}}
\end{table}
```

### Fail

```latex
% 9 维度全宽表，无一句话说明为什么需要 9 维；其中 4 列只有 Ours 是 \cmark
\begin{table*}[t]
  \begin{tabular}{lccccccccc}
    ...
  \end{tabular}
\end{table*}
```
