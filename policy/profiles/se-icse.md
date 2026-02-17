---
name: se-icse
domain: se
venue: icse
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
| PAPER.SECTION_HEADINGS_MAX_6 | params.max_sections | 8 | SE 论文常含 Implementation、Discussion、Related Work、Threats to Validity 独立 section |

## Domain-Specific Rules

Software Engineering 领域特有规则：

- **Motivation Example**: Introduction 或单独 Motivation section 中须包含具体的软件工程问题实例
- **Artifact Availability**: 涉及工具或系统实现时，须说明公开可用性（open-source link 或 artifact evaluation）
- **Threats to Validity**: 实验部分须包含 Threats to Validity 讨论（internal, external, construct validity）
- **Baseline 选择论证**: 须解释为何选择特定 baseline，包含近两年 SOTA 方法

## Venue Quick Facts

| 项目 | 值 |
|------|-----|
| 会议 | ICSE (International Conference on Software Engineering) |
| 正文页数 | 10 页（+ 2 页参考文献），ACM 格式 |
| 格式 | 双栏，ACM SIGSOFT 模板 |
| 审稿制度 | 双盲 |
| 补充材料 | 允许（Artifact Evaluation track 单独评审） |
| 引用格式 | ACM Reference Format |

## Cross-References

- `rules/experiment-reproducibility.md` — 实验可复现性要求
- `rules/coding-style.md` — SE 项目代码规范
