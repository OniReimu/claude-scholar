---
id: SE.THREATS_TO_VALIDITY_STRUCTURED
slug: se-threats-to-validity-structured
severity: error
locked: false
layer: domain
artifacts: [text]
phases: [writing-experiments, writing-conclusion, self-review]
domains: [se]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: []
constraint_type: guidance
autofix: none
---

## Requirement

实证类 SE 论文须包含**结构化 Threats to Validity** 节，按类别组织：

1. Construct validity
2. Internal validity
3. External validity
4. Conclusion validity（或 Reliability）

每条威胁 = named threat + 对应 mitigation，并锚定方法学引用（如 Wohlin et al.）。允许省略与本研究无关的类别，但须覆盖适用的全部类别。

## Rationale

结构化 ToV 是 empirical-SE 的 near-mandatory 节：它让 reviewer 确认作者已系统检视偏差来源并给出缓解。一段笼统的“可能存在局限”不可接受，因为它无法被核对、也不提供缓解信息。**面向 empirical-SE 论文，通过 `se-*` profile 激活。**

## Check

- **LLM 语义检查**：
  - 是否按 construct / internal / external / conclusion(reliability) 分类组织
  - 每条威胁是否配 named threat + mitigation
  - 是否锚定方法学引用（Wohlin 等）
- **通过标准**：reviewer 可逐条对应“威胁 → 缓解”，并定位其 validity 类别

## Examples

### Pass

```latex
\section{Threats to Validity}
\textbf{Construct validity.} Our AI/non-AI label relies on repository topics, which may
misclassify borderline repos; we mitigate with a two-annotator audit (Fleiss
$\kappa=0.67$). \textbf{External validity.} Findings cover Python/HuggingFace and may not
transfer to other ecosystems; we scope claims accordingly~\cite{wohlin2012experimentation}.
% internal / conclusion 同构
```

### Fail

```latex
\section{Limitations}
Our study has some limitations. The dataset may not be fully representative and there
could be measurement noise. We leave a more thorough analysis to future work.
% 单段笼统陈述，无 validity 分类、无 named threat + mitigation、无方法学引用
```
