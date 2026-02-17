---
id: LATEX.CMARK_XMARK_PMARK_MACROS
slug: latex-cmark-xmark-pmark-macros
severity: error
locked: false
layer: core
artifacts: [table, text]
phases: [writing-background, writing-methods, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: []
---

## Requirement

当论文使用 ✓ / ✗ / ◐ 这类定性符号（尤其是 Related Work 比较表）时，必须统一使用以下宏定义，并在导言区加载 `pifont` 与 `xcolor`：

```latex
\newcommand{\cmark}{\textcolor{green!80!black}{\ding{51}}}
\newcommand{\xmark}{\textcolor{red}{\ding{55}}}
\newcommand{\pmark}{\textcolor{blue!90}{\ding{109}}}
```

## Rationale

不同模板或第三方包（例如 TikZ 相关宏/符号设置）可能改变或缺失勾叉符号定义，容易导致 `Undefined control sequence` 或符号风格不一致。统一使用 `\ding{}` + `\textcolor{}` 的显式宏可避免兼容性问题，并保持全稿视觉一致性。

## Check

- **LLM 检查**: 若文中使用 `\cmark` / `\xmark` / `\pmark` 或出现 ✓ / ✗ / ◐ 风格比较表，检查导言区是否包含 `\usepackage{pifont}`、`\usepackage{xcolor}` 与三条宏定义
- **冲突处理**: 若模板已定义同名命令，允许改用 `\renewcommand`，但符号与颜色语义必须保持一致
- **常见违规**: 只写了 `\cmark` 表格内容但未定义宏；使用模板默认 `\checkmark` 导致符号风格与其他表格不一致

## Examples

### Pass

```latex
\usepackage{pifont}
\usepackage{xcolor}
\newcommand{\cmark}{\textcolor{green!80!black}{\ding{51}}}
\newcommand{\xmark}{\textcolor{red}{\ding{55}}}
\newcommand{\pmark}{\textcolor{blue!90}{\ding{109}}}

\begin{tabular}{lccc}
\toprule
Method & Robustness & Efficiency & Privacy \\
\midrule
Baseline & \xmark & \cmark & \xmark \\
Ours     & \cmark & \cmark & \pmark \\
\bottomrule
\end{tabular}
```

### Fail

```latex
% 未定义 \cmark/\xmark/\pmark，或未加载 pifont/xcolor
\begin{tabular}{lcc}
\toprule
Method & Robustness & Privacy \\
\midrule
Baseline & \xmark & \xmark \\
Ours     & \cmark & \pmark \\
\bottomrule
\end{tabular}
```
