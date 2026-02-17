---
name: is-misq
domain: is
venue: misq
---

## Includes

- `policy/rules/fig-no-in-figure-title.md`
- `policy/rules/fig-font-ge-24pt.md`
- `policy/rules/fig-one-file-one-figure.md`
- `policy/rules/fig-colorblind-safe-palette.md`
- `policy/rules/fig-self-contained-caption.md`
- `policy/rules/fig-vector-format-required.md`
- `policy/rules/table-booktabs-format.md`
- `policy/rules/table-direction-indicators.md`
- `policy/rules/latex-eq-display-style.md`
- `policy/rules/latex-var-long-token-use-text.md`
- `policy/rules/latex-notation-consistency.md`
- `policy/rules/ref-cross-reference-style.md`
- `policy/rules/paper-section-headings-max-6.md`
- `policy/rules/paper-conclusion-single-paragraph.md`
- `policy/rules/cite-verify-via-api.md`
- `policy/rules/exp-error-bars-required.md`
- `policy/rules/exp-takeaway-box.md`
- `policy/rules/exp-ablation-in-results.md`
- `policy/rules/exp-results-subsection-structure.md`
- `policy/rules/repro-random-seed-documentation.md`
- `policy/rules/repro-compute-resources-documented.md`
- `policy/rules/submit-section-numbering-consistency.md`
- `policy/rules/prose-intensifiers-elimination.md`
- `policy/rules/prose-em-dash-restriction.md`
- `policy/rules/ethics-limitations-section-mandatory.md`
- `policy/rules/anon-double-blind-anonymization.md`
- `policy/rules/submit-page-limit-strict.md`
- `policy/rules/bibtex-consistent-citation-key-format.md`

## Overrides

| Rule ID | 字段 | 新值 | 原因 |
|---------|------|------|------|
| PAPER.SECTION_HEADINGS_MAX_6 | params.max_sections | 8 | IS 期刊论文结构灵活，常含 Theoretical Background、Research Model、Discussion 独立 section |

## Domain-Specific Rules

Information Systems 领域特有规则：

- **理论框架必需**: 须明确理论基础（theoretical lens），如 TAM, UTAUT, Institutional Theory, Resource-Based View 等
- **Research Model 图**: 须包含研究模型图，展示构念（constructs）及假设关系（hypotheses）
- **统计方法论**: 须详述统计分析方法（SEM, PLS, Regression 等）、效度检验（convergent/discriminant validity）及可靠性指标
- **Implications 双分**: Discussion 须分别讨论 Implications for Research 和 Implications for Practice

## Venue Quick Facts

| 项目 | 值 |
|------|-----|
| 期刊 | MISQ (MIS Quarterly) |
| 正文页数 | 无严格限制（一般 40-60 页手稿） |
| 格式 | 单栏，期刊模板 |
| 审稿制度 | 双盲 |
| 修订轮次 | 通常 2-3 轮 R&R |
| 引用格式 | APA 7th Edition |

## Cross-References

- `rules/experiment-reproducibility.md` — 实验可复现性要求
