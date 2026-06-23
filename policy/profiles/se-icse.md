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
- `policy/rules/latex-cmark-xmark-pmark-macros.md`
- `policy/rules/latex-var-long-token-use-text.md`
- `policy/rules/latex-notation-consistency.md`
- `policy/rules/ref-cross-reference-style.md`
- `policy/rules/paper-section-headings-max-6.md`
- `policy/rules/paper-conclusion-single-paragraph.md`
- `policy/rules/cite-verify-via-api.md`
- `policy/rules/cite-claim-support-required.md`
- `policy/rules/exp-error-bars-required.md`
- `policy/rules/exp-takeaway-box.md`
- `policy/rules/exp-ablation-in-results.md`
- `policy/rules/exp-results-subsection-structure.md`
- `policy/rules/exp-fabricated-results-caption-disclosure.md`
- `policy/rules/exp-results-status-declaration-required.md`
- `policy/rules/repro-random-seed-documentation.md`
- `policy/rules/repro-compute-resources-documented.md`
- `policy/rules/submit-section-numbering-consistency.md`
- `policy/rules/prose-intensifiers-elimination.md`
- `policy/rules/prose-em-dash-restriction.md`
- `policy/rules/prose-sentence-length.md`
- `policy/rules/prose-comma-overuse.md`
- `policy/rules/prose-midsentence-colon.md`
- `policy/rules/prose-negation-contrast.md`
- `policy/rules/prose-paragraph-topic-sentence.md`
- `policy/rules/ethics-limitations-section-mandatory.md`
- `policy/rules/anon-double-blind-anonymization.md`
- `policy/rules/submit-page-limit-strict.md`
- `policy/rules/bibtex-consistent-citation-key-format.md`
- `policy/rules/se-research-questions-explicit.md`
- `policy/rules/se-rq-section-binding.md`
- `policy/rules/se-threats-to-validity-structured.md`
- `policy/rules/se-actionable-implications.md`

## Overrides

| Rule ID | 字段 | 新值 | 原因 |
|---------|------|------|------|
| PAPER.SECTION_HEADINGS_MAX_6 | params.max_sections | 8 | SE 论文常含 Implementation、Discussion、Related Work、Threats to Validity 独立 section |

## Empirical-SE Paper Playbook

> 蒸馏自 Data61 / Monash SE lineage 的 exemplar 论文（TOSEM 2024/2022、ICSE 2023）与
> ICSE/TSE/TOSEM/EMSE 通行惯例。这是一份 SE-domain 叙事/结构向导：科学内容（采样、统计、
> 可复现性）我们通常不输给 exemplar，**真正决定 empirical-SE 论文是否“可读 / 可被 reviewer
> 导航”的，是下面这些 reviewer-facing scaffolding**。与现有 `prose-*` / `exp-*` 规则重叠处
> 一律引用规则，不重述。

### A. Framing & narrative（开篇与叙事）

- **编号 Research Question（最高优先级）**：用 2–4 条显式编号 RQ 驱动全文，开放式问法
  （`What… / How… / To what extent…`），每条 RQ 紧跟一段 *justification*（“Rationale”/“Motivation”/内联说明）。
  位置随 venue：**ICSE 会议放 Introduction 末尾**；期刊（TOSEM/TSE/EMSE）放独立
  `Objective and Research Questions` 节。**不要**把论点埋成 intro 段落里的陈述句 hypothesis。
  若已 pre-register 方向性假设，可作为每条 RQ 下的“committed answer-shape”保留，但 RQ 在标题和
  intro 中是 primary。
- **Motivating vignette / running example**：用一个具体场景把问题讲“扎心”（如 SolarWinds、一次真实审计），
  之后可在全文复用为 running example。
- **Contributions 预告 *findings*，不只是 artifacts**：每条 bullet 给“我们发现了什么”，
  避免堆 deliverable；可把每条 contribution 框成“与某 named baseline 的区别”。
- **Abstract 公式**：context → gap → “we did the first / we measure X” → method+scale → headline number → deliverable/implication。
- **Article roadmap 句**（可选）：“Section 2 covers… Section 3…”，会议论文可省，期刊常见。

### B. Methodology & rigor（reviewer 信任的脊梁）

- **Study-design framing + 方法学引用**：在方法节开头一句点明研究类型并锚定一份 guideline——
  case study → Runeson & Höst；experimentation / threats → Wohlin et al.；mining → MSR norms；
  mixed-methods（interview+survey）注明。方法节标题须含 “Methodology”。
- **可复现的采样 / 数据采集协议**：queries、strata、inclusion/exclusion、dedup、样本量、**seed** 全部写明。
- **数据分析协议 + inter-rater reliability**：标注 coding scheme（grounded theory open/axial、card sorting），
  并**报出 agreement 系数的具体数值**（Cohen's κ / 多评分者 Fleiss κ）——数字必须落地，不能只说“两人标注”。
- **统计严谨**：effect size + CI + 校正后 p 值，并**同时报支持与不支持的证据**，不挑单边（参 `exp-error-bars-required`）。
- **公开 replication package**：tool versions、environment、released dataset 与 quote provenance（参 `repro-*` 规则）。
  涉及工具/系统实现时显式给 open-source link 或 Artifact Evaluation 说明。
- **Baseline 选择论证**：解释为何选择特定 baseline，覆盖近两年 SOTA，并说明可比性边界。

### C. Results presentation（最容易丢失 legibility 的地方）

- **Consolidated takeaway artifact 是不变量**：每个结果单元收口于一个 box **或** 一张 master table，
  绝不留一墙 prose。SE 惯例下 box = **conclusion-first 的 INSIGHT**（observation 句 + 加粗 implication 句），
  **不是 stats recap**——数字留在 prose / 图 / glance table，box 里不堆数字、不放 `\cite`；**一个 RQ 一个 box**
  （cross-subsection synthesis，不是机械地每段一个）。细化自 `<!-- policy:EXP.TAKEAWAY_BOX -->`。
- **RQ→section 三重绑定**：(a) 结果小节标题**逐字复述该 RQ**（`RQ1: What distinguishes…?`）；
  (b) 小节开头一句 signpost “To answer RQ1, we…”；(c) 一张 glance table 带 **RQ 列**，让索引同时充当
  RQ→finding→section 映射。
- **Results-unit evidence triad（固定节奏）**：claim/topic-sentence → quantitative anchor（number + CI）→
  evidence（table/figure ref 或 quote）→ box。节奏一致才 skimmable。配合 `<!-- policy:EXP.RESULTS_SUBSECTION_STRUCTURE -->`。
- **中心结果表 + positioning/comparison 矩阵**：一张 master grid 装原子结果，一张 ✓/✗/◐ 矩阵对比 prior work / baseline，
  可交叉引用 finding 编号。

### D. Positioning & impact（SE 论文的实用 payoff）

- **Related Work 按 line-of-work 组织**（不要 paper-by-paper），收尾给显式 positioning statement + ✓/✗/◐ feature matrix；
  期刊可把 RW 后置到 results 之后。
- **Actionable implications 三件套**：把建议格式化为 `For tool builders / For standards bodies / For practitioners`
  的 run-in 三段，各一行。SE reviewer 明确奖励这种“谁该做什么”的可执行 payoff。
- **结构化 Threats to Validity（near-mandatory）**：construct / internal / external /（conclusion 或 reliability），
  每条 = named threat + mitigation，引用 Wohlin。
- **显式 Future Work**：2–3 条具体方向，**正面框成 opportunity**（如 static call-site → runtime resolution；
  cross-sectional snapshot → longitudinal；single-registry → cross-registry），不要写成 “we have limitations” 的认输句。

### E. Prose texture（交给已 wire-in 的规则，不重述）

- 一段一个 point，约 80–150 词；2-column 下 >150 词的段是 reviewer 会跳过的灰墙——拆段。
- Topic sentence 先行（claim 在前，number + evidence 在后）：`<!-- policy:PROSE.PARAGRAPH_TOPIC_SENTENCE -->`。
- 句长 ≤ ~35 词、拆逗号长链：`<!-- policy:PROSE.SENTENCE_LENGTH -->`、`<!-- policy:PROSE.COMMA_OVERUSE -->`。
- 不用句中解释性冒号：`<!-- policy:PROSE.MIDSENTENCE_COLON -->`；不用 em-dash：`<!-- policy:PROSE.EM_DASH_RESTRICTION -->`。
- 不用 negation-contrast 框架（`X, not Y`）：`<!-- policy:PROSE.NEGATION_CONTRAST -->`。

### Scorecard（自审时按此定位 gap）

通常已达标/超过 exemplar 的是**实质**：sampling、statistics、inter-rater κ、threats、artifact、comparison table、vignette、abstract。
最常见的真实 gap 是**呈现/导航层**：① 没有显式 RQ；② 没有 RQ→section 绑定；③ 没有 per-RQ takeaway box；
④ implications 结构太薄；⑤ 没有显式 future work；⑥ 缺 study-type framing 句 + 方法学引用。
这些都比“缺实验”低风险——是 scaffolding，不是 science。

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
