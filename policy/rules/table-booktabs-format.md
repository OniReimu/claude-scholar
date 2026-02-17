---
id: TABLE.BOOKTABS_FORMAT
slug: table-booktabs-format
severity: warn
locked: false
layer: core
artifacts: [table]
phases: [writing-experiments, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: regex
enforcement: lint_script
params: {}
conflicts_with: []
lint_patterns:
  - pattern: "\\\\begin\\{tabular\\}"
    mode: match
  - pattern: "\\\\toprule"
    mode: negative
lint_targets: "**/*.tex"
---

## Requirement

所有 LaTeX 表格必须使用 `booktabs` 包（`\toprule`、`\midrule`、`\bottomrule`）。禁止使用 `\hline`。宽表格用 `\resizebox{\columnwidth}{!}{}` 或 `\resizebox{\textwidth}{!}{}` 包裹。最佳值加粗。

## Rationale

booktabs 是学术出版的标准表格格式，提供专业的线条间距。`\hline` 产生的线条过于密集，不符合排版规范。

## Check

- **Regex 检测**: `.tex` 文件中是否有 `\begin{tabular}` 但缺少 `\toprule`
- **Hline 检测**: 是否使用了 `\hline`

## Examples

### Pass

```latex
\usepackage{booktabs}
...
\begin{table}
  \caption{Results on CIFAR-10}\label{tab:cifar}
  \resizebox{\columnwidth}{!}{
  \begin{tabular}{lcc}
    \toprule
    Method   & Accuracy       & F1 Score       \\
    \midrule
    Baseline & $81.0 \pm 0.9$ & $79.3 \pm 1.1$ \\
    Ours     & $\textbf{83.2} \pm 1.3$ & $\textbf{82.1} \pm 0.7$ \\
    \bottomrule
  \end{tabular}}
\end{table}
```

### Fail

```latex
% 使用 \hline 而非 booktabs
\begin{table}
  \begin{tabular}{|l|c|c|}
    \hline
    Method   & Accuracy & F1 Score \\
    \hline
    Baseline & 81.0     & 79.3     \\
    \hline
    Ours     & 83.2     & 82.1     \\
    \hline
  \end{tabular}
\end{table}
```
