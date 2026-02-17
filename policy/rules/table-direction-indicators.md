---
id: TABLE.DIRECTION_INDICATORS
slug: table-direction-indicators
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
---

## Requirement

数值结果表格的列标题必须包含方向指示符：↑ 表示越高越好，↓ 表示越低越好。在 LaTeX 中使用 `$\uparrow$` 和 `$\downarrow$`。尤其在同一表中混合不同方向的度量（如 accuracy↑ 和 FID↓）时，方向指示符为必需项。

## Rationale

方向指示符消除度量解读歧义，审稿人无需猜测数值含义。当表格包含多种度量且方向不一致时（如 accuracy 越高越好、FID 越低越好），缺少指示符会增加误读风险，降低论文的专业性。

## Check

- **LLM 检查**: 审查含数值结果的表格，检查列标题中是否缺少 ↑/↓ 符号（或对应 LaTeX 命令 `$\uparrow$` / `$\downarrow$`）
- **要点**: 每个报告数值度量的列标题都应包含方向指示符

## Examples

### Pass

```latex
\begin{tabular}{lccc}
\toprule
Method & Acc (\%) $\uparrow$ & FID $\downarrow$ & BLEU $\uparrow$ \\
\midrule
Ours   & \textbf{92.3} & \textbf{12.4} & \textbf{38.7} \\
Baseline & 89.1 & 15.8 & 35.2 \\
\bottomrule
\end{tabular}
```

### Fail（缺少方向指示符）

```latex
\begin{tabular}{lccc}
\toprule
Method & Accuracy & FID & BLEU \\
\midrule
Ours   & \textbf{92.3} & \textbf{12.4} & \textbf{38.7} \\
Baseline & 89.1 & 15.8 & 35.2 \\
\bottomrule
\end{tabular}
% 违规：列标题 Accuracy、FID、BLEU 均缺少 ↑/↓ 指示符
% 读者无法判断 FID 12.4 < 15.8 是更好还是更差
```
