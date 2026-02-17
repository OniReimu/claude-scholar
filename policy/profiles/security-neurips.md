---
name: security-neurips
domain: security
venue: neurips
---

## Includes

- `policy/rules/fig-no-in-figure-title.md`
- `policy/rules/fig-font-ge-24pt.md`
- `policy/rules/fig-one-file-one-figure.md`
- `policy/rules/latex-eq-display-style.md`
- `policy/rules/latex-cmark-xmark-pmark-macros.md`
- `policy/rules/latex-var-long-token-use-text.md`
- `policy/rules/ref-cross-reference-style.md`
- `policy/rules/paper-conclusion-single-paragraph.md`
- `policy/rules/exp-takeaway-box.md`
- `policy/rules/cite-verify-via-api.md`
- `policy/rules/exp-error-bars-required.md`
- `policy/rules/latex-notation-consistency.md`
- `policy/rules/fig-vector-format-required.md`
- `policy/rules/fig-colorblind-safe-palette.md`
- `policy/rules/fig-self-contained-caption.md`
- `policy/rules/table-booktabs-format.md`
- `policy/rules/table-direction-indicators.md`
- `policy/rules/paper-section-headings-max-6.md`
- `policy/rules/submit-section-numbering-consistency.md`
- `policy/rules/repro-random-seed-documentation.md`
- `policy/rules/repro-compute-resources-documented.md`
- `policy/rules/prose-intensifiers-elimination.md`
- `policy/rules/prose-em-dash-restriction.md`
- `policy/rules/exp-ablation-in-results.md`
- `policy/rules/exp-results-subsection-structure.md`
- `policy/rules/ethics-limitations-section-mandatory.md`
- `policy/rules/anon-double-blind-anonymization.md`
- `policy/rules/submit-page-limit-strict.md`
- `policy/rules/bibtex-consistent-citation-key-format.md`

## Overrides

| Rule ID | 字段 | 新值 | 原因 |
|---------|------|------|------|
| PAPER.CONCLUSION_SINGLE_PARAGRAPH | severity | error | NeurIPS 篇幅紧（9 页正文），Conclusion 必须紧凑 |
| FIG.FONT_GE_24PT | params.min_font_pt | 28 | Security 图表符号密集，需要更大字号确保可读性 |

> 注：FIG.FONT_GE_24PT 规则卡片声明 `params: {min_font_pt: 24}`，此处覆盖为 28。
> FIG.NO_IN_FIGURE_TITLE 为 locked=true，不可覆盖。

## Domain-Specific Rules

Security 领域特有规则（M2 补充完整卡片）：

- **Threat Model 图规范**: System model / threat model 必须包含 adversary capabilities、trust boundaries、attack vectors
- **安全证明格式**: 安全定义使用 Definition 环境，证明使用 Proof 环境
- **实验攻击场景**: 每个实验必须明确攻击场景（attack setting）和安全假设（security assumption）

## Venue Quick Facts

| 项目 | 值 |
|------|-----|
| 会议 | NeurIPS (Neural Information Processing Systems) |
| 正文页数 | 9 页（不含参考文献） |
| 格式 | 双栏，LaTeX 模板 |
| 审稿制度 | 双盲 |
| 补充材料 | 允许（不计入页数限制） |
| 引用格式 | `\citep{}` / `\citet{}` (natbib) |

## Cross-References

- `rules/experiment-reproducibility.md` — 实验可复现性要求（随机种子、配置记录）
- `rules/security.md` — 代码安全规范（密钥管理、敏感文件保护）
