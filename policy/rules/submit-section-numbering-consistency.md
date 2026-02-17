---
id: SUBMIT.SECTION_NUMBERING_CONSISTENCY
slug: submit-section-numbering-consistency
severity: warn
locked: false
layer: core
artifacts: [text]
phases: [self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: regex
enforcement: lint_script
params: {}
conflicts_with: []
lint_patterns:
  - pattern: "\\\\section\\*\\{(?!References|Acknowledgment|Broader Impact|Ethics|Limitations|Appendix)"
    mode: match
lint_targets: "**/*.tex"
---

## Requirement

使用一致的 LaTeX section 编号：主体 section 用 `\section{}` 编号，仅 References、Acknowledgments、Broader Impact、Ethics、Limitations、Appendix 使用 `\section*{}`（非编号）。禁止在主体内容中混用编号和非编号 section。

## Rationale

混用编号和非编号 section 显得不专业，可能导致读者在引用时混淆 section 编号。非编号 section 仅适用于约定俗成的辅助部分（如参考文献、致谢），主体内容必须统一编号以保证结构清晰。

## Check

- **regex 检查**: 检测 `\section*{}` 是否出现在允许列表（References, Acknowledgment, Broader Impact, Ethics, Limitations, Appendix）之外的标题
- **要点**: 匹配 `\section*{` 后紧跟的内容不在白名单中则违规

## Examples

### Pass

```latex
\section{Introduction}
\section{Method}
\section{Experiments}
\section{Conclusion}

\section*{References}
\section*{Acknowledgments}
% 主体全部用 \section{}，仅辅助部分用 \section*{}
```

### Fail（主体中使用非编号 section）

```latex
\section{Introduction}
\section{Related Work}
\section*{Our Approach}        % 违规：主体内容使用 \section*{}
\section{Experiments}
\section{Conclusion}
% Our Approach 不在允许列表中，应使用 \section{Our Approach}
```
