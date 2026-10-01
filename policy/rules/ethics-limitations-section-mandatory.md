---
id: ETHICS.LIMITATIONS_SECTION_MANDATORY
slug: ethics-limitations-section-mandatory
severity: error
locked: false
layer: venue
artifacts: [text]
phases: [writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [neurips, icml, iclr, acl]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: [PROSE.SELF_UNDERMINING, PAPER.LIMITATION_NECESSITY]
constraint_type: guidance
autofix: none
---

## Requirement

论文必须包含独立的 Limitations section（或 Discussion 的 subsection），诚实报告方法的局限性：假设、适用范围约束、失败模式、泛化性问题。NeurIPS/ICML/ICLR/ACL 要求此 section 为必需项，且不计入页数限制。

## Rationale

主要 ML 顶会的 checklist 明确要求 Limitations section。缺失 Limitations 是审稿人的常见扣分点，也可能导致 desk rejection。

## Check

- **提取 `.tex` 正文的正确方法**：见 `policy/references/tex-prose-extraction.md`。手搓扫描器的四个典型错误（`split('%')` 在 `$95\%$` 处截断、剔数学时 `$` 奇数配对吞掉整段、逐行扫描漏掉被硬换行劈开的短语、两遍大小写策略不一致）都会产生**假的「已清零」结论**

- **LLM 检查**: 论文中是否存在 `Limitations` section 或 subsection
- **内容检查**: 内容是否实质性地讨论了局限（非敷衍一句话）。**不设条数下限**：条数由真实存在的适用边界决定，凑满配额会诱导出"为什么没做更多"和"回应假想质疑"两类条目。每一条是否该存在，由 `PAPER.LIMITATION_NECESSITY` 判
- **位置检查**: 通常位于 Conclusion 之后或 Discussion 内部

## Examples

### Pass

```latex
\section*{Limitations}

Our work has several limitations that we acknowledge:

\paragraph{Assumption of i.i.d. data.} Our theoretical analysis assumes that
training samples are independently and identically distributed, which may not
hold in sequential decision-making scenarios.

\paragraph{Computational cost.} The proposed method requires 2x more GPU memory
than the baseline due to the dual-encoder architecture, limiting applicability
to resource-constrained settings.

\paragraph{Domain specificity.} We evaluate only on English-language benchmarks.
The effectiveness on morphologically rich languages remains unexplored.
```

### Fail

```latex
% 无 Limitations section，或仅敷衍一句话：
\section*{Limitations}
Our method has some limitations.
```

## Conflicts

- `PAPER.LIMITATION_NECESSITY`：本卡要求小节存在，那张卡筛小节里的每一条。删掉的只是"作者的边界"和"对假想质疑的回应"，发现的适用边界一条不少，所以不构成削薄披露
- `PROSE.SELF_UNDERMINING` 管的是措辞，不管披露——本节要求的诚实报告优先，不得以「不递刀子」为由删减局限点或使其敷衍；那条只允许改写这些局限点的措辞（去情绪副词、回填锚点、把普遍判决收回局部）
