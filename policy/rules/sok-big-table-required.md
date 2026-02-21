---
id: SOK.BIG_TABLE_REQUIRED
slug: sok-big-table-required
severity: error
locked: false
layer: domain
artifacts: [table, text]
phases: [writing-experiments]
domains: [security, se, is]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: []
---

## Requirement

SoK 论文必须包含至少一个综合性对比大表（big comparison table），用于统一呈现代表性工作在关键维度上的横向比较。该表至少应包含：

1. 方法/论文标识
2. taxonomy 相关类别信息
3. 关键能力/假设/限制字段（按主题选择）

## Rationale

SoK 的价值不仅在分类，还在“可检索、可比较、可复用”的知识组织。综合大表是把 taxonomy 转化为可操作比较工具的关键载体。**本规则仅在 SoK profile 激活时生效**。

## Check

- **LLM 语义检查**:
  - 是否存在至少一个真正的综合对比表（非单点实验结果表）
  - 表字段是否支撑横向比较，而非简单列论文名
  - 表内容是否与 taxonomy 维度一致
- **实现建议**: 可使用 `table*` + `booktabs` + 必要的缩放控制可读性

## Examples

### Pass

```latex
\begin{table*}[t]
\centering
\caption{Systematized comparison of representative methods under our taxonomy.}
\begin{tabular}{lcccc}
\toprule
Method & Category & Threat Model & Guarantee Type & Main Limitation \\
\midrule
...
\bottomrule
\end{tabular}
\end{table*}
```

### Fail

```latex
\begin{table}[t]
\centering
\caption{Our runtime on CIFAR-10}
\begin{tabular}{lc}
\toprule
Method & Time \\
\midrule
Ours & 1.2s \\
\bottomrule
\end{tabular}
\end{table}
% 仅为单实验结果表，不是 SoK 的综合对比大表
```
