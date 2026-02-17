---
id: PAPER.SECTION_HEADINGS_MAX_6
slug: paper-section-headings-max-6
severity: error
locked: false
layer: core
artifacts: [text]
phases: [ideation, writing-methods, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: regex
enforcement: lint_script
params: {max_sections: 6}
conflicts_with: []
lint_patterns:
  - pattern: "\\\\section\\{"
    mode: count
    threshold: 6
    threshold_param: max_sections
lint_targets: "**/*.tex"
---

## Requirement

论文正文的顶级编号 section（`\section{}`）不超过 6 个（不含 Abstract、References、Acknowledgments 等非编号部分）。典型结构：1 Introduction, 2 Background, 3 System Model, 4 Methods, 5 Experiments, 6 Conclusion。

## Rationale

过多的顶级 section 导致论文结构散乱、每节内容不够深入。6 个 section 是顶会论文的常见上限，强制作者聚焦核心内容。深入的讨论和辅助材料应放在 subsection 或 appendix 中，而非增加顶级 section 数量。

不同领域对 section 数量有不同惯例：Security/AI 顶会（NeurIPS, ICML, ICLR, CCS, USENIX, NDSS, S&P）通常 6 section；Software Engineering（ICSE, FSE, ASE）和 Information Systems（MISQ, ISR）因含 Implementation、Discussion 等独立 section，通常 7-8 section。Profile 可通过 `params.max_sections` 覆盖默认阈值。

## Check

- **regex 检查**: 计数 .tex 文件中 `\section{` 的出现次数（排除 `\section*{}`），超过 `params.max_sections`（默认 6）则违规
- **要点**: `\section*{}` 不计入编号 section，不受此限制
- **lint 注意**: `lint_patterns.threshold` 使用默认值 6，Profile 覆盖的阈值由 `lint.sh` 的 `get_effective_threshold()` 在 lint 时自动应用

## Examples

### Pass

```latex
\section{Introduction}
\section{Related Work}
\section{System Model}
\section{Methods}
\section{Experiments}
\section{Conclusion}

\section*{References}
\section*{Acknowledgments}
% 6 个编号 section + 非编号部分，符合规则
```

### Fail（超过 6 个编号 section）

```latex
\section{Introduction}
\section{Related Work}
\section{Preliminary}
\section{System Model}
\section{Methods}
\section{Experiments}
\section{Discussion}
\section{Conclusion}
% 违规：8 个编号 section，超过上限 6
% 解决方式：将相关内容合并或降为 \subsection{}
```
