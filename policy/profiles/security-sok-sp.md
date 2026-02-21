---
name: security-sok-sp
domain: security
venue: sp
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
- `policy/rules/prose-crypto-construction-template.md`
- `policy/rules/prose-intensifiers-elimination.md`
- `policy/rules/prose-em-dash-restriction.md`
- `policy/rules/exp-ablation-in-results.md`
- `policy/rules/exp-results-subsection-structure.md`
- `policy/rules/exp-fabricated-results-caption-disclosure.md`
- `policy/rules/exp-results-status-declaration-required.md`
- `policy/rules/ethics-limitations-section-mandatory.md`
- `policy/rules/anon-double-blind-anonymization.md`
- `policy/rules/submit-page-limit-strict.md`
- `policy/rules/bibtex-consistent-citation-key-format.md`
- `policy/rules/sok-taxonomy-required.md`
- `policy/rules/sok-methodology-reporting.md`
- `policy/rules/sok-big-table-required.md`
- `policy/rules/sok-research-agenda-required.md`

## Overrides

| Rule ID | 字段 | 新值 | 原因 |
|---------|------|------|------|

> v1 不继承 `security-neurips` 的 venue-specific overrides；如需 S&P 特定覆盖，后续单独添加。

## Domain-Specific Rules

Security SoK 论文建议补充：

- Threat model taxonomy 与攻击能力边界（capability boundary）的一致性说明
- 评估假设（assumption）与适用场景（deployment context）的显式映射
- 术语标准化（同义术语合并、缩写规范）

## Venue Quick Facts

| 项目 | 值 |
|------|-----|
| 会议 | IEEE Symposium on Security and Privacy (S&P) |
| 审稿制度 | 双盲 |
| 关注点 | 系统性安全洞察、威胁模型清晰性、论证严谨性 |

## Cross-References

- `rules/experiment-reproducibility.md` — 实验可复现性要求（随机种子、配置记录）
- `rules/security.md` — 代码安全规范（密钥管理、敏感文件保护）
