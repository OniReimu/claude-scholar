---
name: security-ccs
domain: security
venue: ccs
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
- `policy/rules/latex-cmark-xmark-pmark-macros.md`
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
- `policy/rules/exp-fabricated-results-caption-disclosure.md`
- `policy/rules/exp-results-status-declaration-required.md`
- `policy/rules/repro-random-seed-documentation.md`
- `policy/rules/repro-compute-resources-documented.md`
- `policy/rules/submit-section-numbering-consistency.md`
- `policy/rules/prose-crypto-construction-template.md`
- `policy/rules/prose-intensifiers-elimination.md`
- `policy/rules/prose-em-dash-restriction.md`
- `policy/rules/prose-rhetorical-self-answer.md`
- `policy/rules/ethics-limitations-section-mandatory.md`
- `policy/rules/anon-double-blind-anonymization.md`
- `policy/rules/submit-page-limit-strict.md`
- `policy/rules/bibtex-consistent-citation-key-format.md`
- `policy/rules/se-research-questions-explicit.md`
- `policy/rules/se-rq-section-binding.md`

## Overrides

| Rule ID | 字段 | 新值 | 原因 |
|---------|------|------|------|
| PAPER.SECTION_HEADINGS_MAX_6 | params.max_sections | 8 | 安全攻击/系统类论文常含 Threat Model、Design、Implementation、Evaluation、Defense/Discussion、Related Work 独立 section |

## Domain-Specific Rules

Security 领域特有规则：

- **Threat Model 图规范**: System model / threat model 必须包含 adversary capabilities、trust boundaries、attack vectors
- **威胁/防御模型分离**: Attacker model 与 Defender model 显式分开陈述，各自列 capability 与 constraint
- **实验攻击场景**: 每个实验必须明确攻击场景（attack setting）和安全假设（security assumption）
- **Responsible disclosure**: 涉及真实系统/厂商的攻击须说明 disclosure 状态与 ethics 处理

## RQ-Driven Numerical Evaluation (AI-security default)

为 **AI-security / 高数值实证类** 论文设计的 Evaluation 组织范式：借 empirical-SE 的 RQ 骨架，针对 figure/table 密度做适配。这是该类论文 Experiments 章节的默认结构。

**何时用（边界）**：评估含 **≥3 个*异质*评估维度**（如 viability / mechanism / defense / robustness / baseline / end-to-end）时设为默认。若全篇只有一个主结果（同质：同一 metric 换 setting/benchmark），不要硬拆 RQ，用 Main Results / Ablation / Analysis 即可。判据是维度的*异质性*，不是数值密度本身。

**结构** `<!-- policy:SE.RESEARCH_QUESTIONS_EXPLICIT -->` `<!-- policy:SE.RQ_SECTION_BINDING -->`：

1. Evaluation 开头一段 overview，列出 **2–6 条编号 RQ**（建议 ≤6）；每条 RQ 要么对应一个 contribution，要么对应一个 validity/positioning 检查（robustness / baseline / end-to-end）。
2. 每个 §6.x **恰好回答一个 RQ**：标题逐字复述 RQ；小节开头一句 `To answer RQx, we …`。
3. **数值适配（与 qualitative SE 的关键差异）**：每个 RQ-subsection 拥有一*簇* figures/tables，并以一个 **takeaway box** 收口；box 里拎出*那一个* load-bearing 的数字。`<!-- policy:EXP.TAKEAWAY_BOX -->`
   - 对比：measurement 论文数字在 prose、box 写 insight（见 `se-icse.md`）；数值论文数字在图表、box 反而要点出关键数。取决于数字主要落在哪。

**纪律**：

- **RQ 是概念问题，不是每张表一问**：2–4 理想，≤6 封顶。维度异质才用 RQ，单一主结果别硬拆。
- **优先 measuring 措辞**（to what extent / what governs / under what conditions），少用 yes/no 的 `Can…?/Does…?`——后者答案显然为 yes，有 rhetorical-self-answer 味道。`<!-- policy:PROSE.RHETORICAL_SELF_ANSWER -->`
- **捆绑 RQ 要拆子标题**：若一个 RQ 覆盖 N 个 desiderata（如 useful/stealthy/detectability/utility），该 subsection 内须有 N 个对应的 labeled 小标题（`\noindent\textbf{}` run-in），否则会散成一团。
- **anti-sprawl**：不要把所有图表平铺进一个 flat Results；按 RQ 分区，让每张图都有它回答的 question。

## Venue Quick Facts

| 项目 | 值 |
|------|-----|
| 会议 | ACM CCS (Conference on Computer and Communications Security) |
| 正文页数 | **以当年 CCS CFP 为准**（近年约 13 页正文，不含 references / appendices） |
| 格式 | ACM `acmart` `sigconf` 双栏 |
| 审稿制度 | 双盲 |
| 补充材料 | 允许（Artifact Evaluation 单独评审） |
| 引用格式 | ACM Reference Format |

> ⚠️ 页数 / 截止日期 / 模板版本必须在投稿前核对当年 CCS CFP，勿沿用历史值。

## Cross-References

- `rules/experiment-reproducibility.md` — 实验可复现性要求（随机种子、配置记录）
- `rules/security.md` — 代码安全规范（密钥管理、敏感文件保护）
- `policy/profiles/se-icse.md` — empirical-SE playbook（RQ 骨架的来源；measurement 论文的 box 约定）
