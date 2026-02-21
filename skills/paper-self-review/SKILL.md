---
name: paper-self-review
description: This skill should be used when the user asks to "review paper quality", "check paper completeness", "validate paper structure", "self-review before submission", or mentions systematic paper quality checking. Provides comprehensive quality assurance checklist for academic papers.
version: 0.1.0
---

# Paper Self-Review

A systematic paper quality checking tool that helps researchers conduct comprehensive self-review before submission.

## Policy Rules

> 本 skill 执行以下论文写作规则。权威定义在 `policy/rules/`。
> 行内出现处以 HTML 注释标记引用。**冲突时以 `policy/rules/` 为准。**

| Rule ID | 摘要 |
|---------|------|
| `FIG.NO_IN_FIGURE_TITLE` | 图内不加标题 |
| `FIG.FONT_GE_24PT` | 图表字号 ≥ 24pt |
| `FIG.ONE_FILE_ONE_FIGURE` | 1 文件 = 1 图 |
| `FIG.VECTOR_FORMAT_REQUIRED` | 数据图用矢量格式 |
| `FIG.COLORBLIND_SAFE_PALETTE` | 色盲安全配色 |
| `FIG.SELF_CONTAINED_CAPTION` | Caption三要素 |
| `LATEX.EQ.DISPLAY_STYLE` | Display 公式用 equation 环境 |
| `LATEX.VAR.LONG_TOKEN_USE_TEXT` | 长变量名用 \text{} |
| `LATEX.NOTATION_CONSISTENCY` | 符号全文一致 |
| `REF.CROSS_REFERENCE_STYLE` | 交叉引用用 \ref 命令 |
| `PAPER.CONCLUSION_SINGLE_PARAGRAPH` | Conclusion 单段落 |
| `PAPER.SECTION_HEADINGS_MAX_6` | 顶级section≤6 |
| `EXP.TAKEAWAY_BOX` | 实验结果附 takeaway box |
| `EXP.ERROR_BARS_REQUIRED` | 实验需误差线 |
| `EXP.ABLATION_IN_RESULTS` | 消融实验在Results |
| `EXP.RESULTS_SUBSECTION_STRUCTURE` | 实验小节结构 |
| `EXP.FABRICATED_RESULTS_CAPTION_DISCLOSURE` | 非实跑结果 caption 强制披露 |
| `EXP.RESULTS_STATUS_DECLARATION_REQUIRED` | 非实跑结果小节状态声明 |
| `SOK.TAXONOMY_REQUIRED` | SoK 必须给出 taxonomy |
| `SOK.METHODOLOGY_REPORTING` | SoK 报告文献筛选方法 |
| `SOK.BIG_TABLE_REQUIRED` | SoK 必须有综合对比大表 |
| `SOK.RESEARCH_AGENDA_REQUIRED` | SoK 必须给出研究议程 |
| `TABLE.BOOKTABS_FORMAT` | 使用 booktabs 格式 |
| `TABLE.DIRECTION_INDICATORS` | 表头方向指示符 |
| `CITE.VERIFY_VIA_API` | 引文API验证 |
| `BIBTEX.CONSISTENT_CITATION_KEY_FORMAT` | BibTeX key格式统一 |
| `REPRO.RANDOM_SEED_DOCUMENTATION` | 随机种子文档 |
| `REPRO.COMPUTE_RESOURCES_DOCUMENTED` | 计算资源文档 |
| `PROSE.INTENSIFIERS_ELIMINATION` | 删除空洞强调词 |
| `PROSE.EM_DASH_RESTRICTION` | 限制em-dash |
| `SUBMIT.SECTION_NUMBERING_CONSISTENCY` | Section编号一致 |
| `SUBMIT.PAGE_LIMIT_STRICT` | 严格页数限制 |
| `ETHICS.LIMITATIONS_SECTION_MANDATORY` | 必须Limitations节 |
| `ANON.DOUBLE_BLIND_ANONYMIZATION` | 双盲匿名检查 |

## Core Features

### 1. Structure Review

Check whether all sections of the paper are complete and conform to academic standards:
- Does the Abstract include problem, method, results, and contributions?
- Does the Introduction clearly articulate research motivation and background?
- Is the Method detailed enough to be reproducible?
- Do the Results sufficiently support the conclusions?
- Does the Discussion address limitations and future work?

### 2. Logic Consistency Check

Verify the logical coherence of the paper:
- Do research questions match the methodology?
- Does the experimental design support the research hypotheses?
- Are result interpretations reasonable?
- Are conclusions supported by evidence?

### 3. Citation Completeness

Check the completeness and accuracy of citations:
- Are all citations present in the references? <!-- policy:CITE.VERIFY_VIA_API -->
- Is the reference format consistent? <!-- policy:BIBTEX.CONSISTENT_CITATION_KEY_FORMAT -->
- Are key related works cited?
- Do citations accurately reflect the original content?

### 4. Figure/Table Quality

Evaluate the quality and effectiveness of figures and tables:
- Do all figures/tables have clear captions + labels (no in-figure title text)? <!-- policy:FIG.NO_IN_FIGURE_TITLE -->
- Are figures in vector format (PDF/EPS)? <!-- policy:FIG.VECTOR_FORMAT_REQUIRED -->
- Are colorblind-safe palettes used? <!-- policy:FIG.COLORBLIND_SAFE_PALETTE -->
- Are captions self-contained (what, how, takeaway)? <!-- policy:FIG.SELF_CONTAINED_CAPTION -->
- Do tables use booktabs format? <!-- policy:TABLE.BOOKTABS_FORMAT -->
- Do table headers include direction indicators (↑/↓)? <!-- policy:TABLE.DIRECTION_INDICATORS -->
- Do figures/tables support the text narrative?
- Are figures/tables clear and readable?
- Do formats comply with journal/conference requirements?

### 5. Writing Clarity

Check writing clarity and readability:
- Is the language concise and clear?
- Are empty intensifiers removed? <!-- policy:PROSE.INTENSIFIERS_ELIMINATION -->
- Are em-dashes used sparingly? <!-- policy:PROSE.EM_DASH_RESTRICTION -->
- Is technical terminology used appropriately?
- Are sentence structures clear?
- Is paragraph organization logical?

### 6. LaTeX Math Conformance

Check whether math notation follows project rules:
- Are display equations written with `\begin{equation}...\end{equation}`? <!-- policy:LATEX.EQ.DISPLAY_STYLE -->
- Is raw `$$...$$` or `\[...\]` avoided for display equations? <!-- policy:LATEX.EQ.DISPLAY_STYLE -->
- Are inline equations written with `$...$` where appropriate?
- Are variable-like tokens longer than 3 letters wrapped with `\text{}` in math mode? <!-- policy:LATEX.VAR.LONG_TOKEN_USE_TEXT -->
- Are symbols consistent throughout the paper? <!-- policy:LATEX.NOTATION_CONSISTENCY -->

### 7. Experiment Structure

Check experiment section completeness:
- Do experiment results include error bars? <!-- policy:EXP.ERROR_BARS_REQUIRED -->
- Are ablation studies in the Results section (not Discussion)? <!-- policy:EXP.ABLATION_IN_RESULTS -->
- Does each experiment subsection follow the required structure? <!-- policy:EXP.RESULTS_SUBSECTION_STRUCTURE -->
- If any result is fabricated/synthetic/dummy, is it explicitly disclosed in red uppercase in caption? <!-- policy:EXP.FABRICATED_RESULTS_CAPTION_DISCLOSURE -->
- If a subsection contains fabricated results, is there a subsection-level `[FABRICATED]` status declaration comment? <!-- policy:EXP.RESULTS_STATUS_DECLARATION_REQUIRED -->
- Are random seeds documented? <!-- policy:REPRO.RANDOM_SEED_DOCUMENTATION -->
- Are compute resources documented? <!-- policy:REPRO.COMPUTE_RESOURCES_DOCUMENTED -->

### 8. Submission Compliance

Check submission requirements:
- Are top-level sections ≤ 6? <!-- policy:PAPER.SECTION_HEADINGS_MAX_6 -->
- Is section numbering consistent? <!-- policy:SUBMIT.SECTION_NUMBERING_CONSISTENCY -->
- Does the paper meet the page limit? <!-- policy:SUBMIT.PAGE_LIMIT_STRICT -->
- Is double-blind anonymization correct? <!-- policy:ANON.DOUBLE_BLIND_ANONYMIZATION -->
- Is there a Limitations section? <!-- policy:ETHICS.LIMITATIONS_SECTION_MANDATORY -->

### 9. SoK Scope Checks (When SoK profile is active)

- Is there an explicit taxonomy with clear dimensions and boundaries? <!-- policy:SOK.TAXONOMY_REQUIRED -->
- Is the survey methodology (search/screening criteria) reported? <!-- policy:SOK.METHODOLOGY_REPORTING -->
- Is there at least one taxonomy-aligned big comparison table? <!-- policy:SOK.BIG_TABLE_REQUIRED -->
- Does conclusion/discussion include a concrete research agenda? <!-- policy:SOK.RESEARCH_AGENDA_REQUIRED -->

## Quality Checklist

Use this checklist for systematic paper self-review:

```
Paper Quality Checklist:
- [ ] Abstract includes problem, method, results, contributions
- [ ] Introduction clearly states research motivation
- [ ] Method is reproducible
- [ ] Results support conclusions
- [ ] Discussion addresses limitations
- [ ] All figures/tables have captions + labels (no in-figure title text) <!-- policy:FIG.NO_IN_FIGURE_TITLE -->
- [ ] Display equations use `equation`; no `$$...$$` or `\[...\]` <!-- policy:LATEX.EQ.DISPLAY_STYLE -->
- [ ] In math mode, variable-like tokens >3 letters use `\text{}` <!-- policy:LATEX.VAR.LONG_TOKEN_USE_TEXT -->
- [ ] Citations are complete and accurate <!-- policy:CITE.VERIFY_VIA_API -->
- [ ] BibTeX key format is consistent <!-- policy:BIBTEX.CONSISTENT_CITATION_KEY_FORMAT -->
- [ ] Cross-references use correct prefix: Fig.~\ref, Table~\ref, \S\ref, \eqref, \textbf{Appendix~\ref}, Algorithm~\ref, Listing~\ref <!-- policy:REF.CROSS_REFERENCE_STYLE -->
- [ ] Conclusion is a single dense paragraph (no subsections) <!-- policy:PAPER.CONCLUSION_SINGLE_PARAGRAPH -->
- [ ] Figure source font ≥ 24pt (readable after LaTeX scaling) <!-- policy:FIG.FONT_GE_24PT -->
- [ ] Figures use vector format (PDF/EPS) <!-- policy:FIG.VECTOR_FORMAT_REQUIRED -->
- [ ] System overview/pipeline/architecture diagrams use aspect ratio ≥ 2:1 (e.g., 2.1:1, 3:1) <!-- policy:FIG.SYSTEM_OVERVIEW_ASPECT_RATIO_GE_2TO1 -->
- [ ] Colorblind-safe palettes used <!-- policy:FIG.COLORBLIND_SAFE_PALETTE -->
- [ ] Figure captions are self-contained <!-- policy:FIG.SELF_CONTAINED_CAPTION -->
- [ ] Figure 1 exists and is a conceptual system overview (not an experiment plot)
- [ ] Non-experimental figures (system/pipeline/architecture/threat-model/comparison) are generated via `paper-figure-generator` (AutoFigure-Edit) by default
- [ ] Additional non-experimental figures are added when Figure 1 cannot clearly show key mechanism/protocol details
- [ ] Each Python plot = 1 file → 1 figure (no subplots); composite via LaTeX \subfigure <!-- policy:FIG.ONE_FILE_ONE_FIGURE -->
- [ ] Tables use booktabs format <!-- policy:TABLE.BOOKTABS_FORMAT -->
- [ ] Table headers include direction indicators (↑/↓) <!-- policy:TABLE.DIRECTION_INDICATORS -->
- [ ] Symbols consistent throughout paper <!-- policy:LATEX.NOTATION_CONSISTENCY -->
- [ ] For crypto-oriented security papers, core mechanism is presented as a structured Construction (Primitives/Parameters + named procedures) <!-- policy:PROSE.CRYPTO_CONSTRUCTION_TEMPLATE -->
- [ ] Empty intensifiers removed <!-- policy:PROSE.INTENSIFIERS_ELIMINATION -->
- [ ] Em-dashes used sparingly <!-- policy:PROSE.EM_DASH_RESTRICTION -->
- [ ] Experiment results include error bars <!-- policy:EXP.ERROR_BARS_REQUIRED -->
- [ ] Experiment results subsections each end with \fbox Takeaway box <!-- policy:EXP.TAKEAWAY_BOX -->
- [ ] Ablation studies in Results section <!-- policy:EXP.ABLATION_IN_RESULTS -->
- [ ] Experiment subsections follow required structure <!-- policy:EXP.RESULTS_SUBSECTION_STRUCTURE -->
- [ ] Fabricated/synthetic/dummy results are explicitly disclosed in red uppercase caption <!-- policy:EXP.FABRICATED_RESULTS_CAPTION_DISCLOSURE -->
- [ ] Subsections containing fabricated results include a `[FABRICATED]` status declaration comment <!-- policy:EXP.RESULTS_STATUS_DECLARATION_REQUIRED -->
- [ ] SoK: taxonomy is explicit and operational <!-- policy:SOK.TAXONOMY_REQUIRED -->
- [ ] SoK: methodology reporting is complete (sources + screening criteria) <!-- policy:SOK.METHODOLOGY_REPORTING -->
- [ ] SoK: big comparison table aligned with taxonomy <!-- policy:SOK.BIG_TABLE_REQUIRED -->
- [ ] SoK: concrete research agenda in Conclusion/Discussion <!-- policy:SOK.RESEARCH_AGENDA_REQUIRED -->
- [ ] Random seeds documented <!-- policy:REPRO.RANDOM_SEED_DOCUMENTATION -->
- [ ] Compute resources documented <!-- policy:REPRO.COMPUTE_RESOURCES_DOCUMENTED -->
- [ ] Top-level sections ≤ 6 <!-- policy:PAPER.SECTION_HEADINGS_MAX_6 -->
- [ ] Section numbering consistent <!-- policy:SUBMIT.SECTION_NUMBERING_CONSISTENCY -->
- [ ] Page limit met <!-- policy:SUBMIT.PAGE_LIMIT_STRICT -->
- [ ] Double-blind anonymization correct <!-- policy:ANON.DOUBLE_BLIND_ANONYMIZATION -->
- [ ] Limitations section present <!-- policy:ETHICS.LIMITATIONS_SECTION_MANDATORY -->
```

## When to Use

Use this skill in the following scenarios:

- **Pre-submission check** - Final review before submitting to a journal or conference
- **After first draft** - Systematic review after completing the first draft
- **Before advisor review** - Self-check before requesting advisor feedback to improve quality
- **Post-revision verification** - After revising based on reviewer comments, verify all issues are addressed
- **Collaborator review** - Quality check before sending to collaborators

## Review Process

Follow these steps for systematic paper review:

### Step 1: Structure Review
Start with the overall structure, checking if all sections are complete and logically coherent.

### Step 2: Content Review
Dive into each section, checking content accuracy and completeness.

### Step 3: Citation Check
Verify the completeness and accuracy of all citations.

### Step 4: Figure/Table Review
Check the quality and captions of all figures and tables.

### Step 5: Writing Quality
Review language expression and writing clarity.

### Step 6: Math Conformance Check
Verify equation environment and variable naming style consistency.

### Step 7: Experiment Structure Check
Verify error bars, ablation placement, subsection structure, reproducibility documentation.

### Step 8: Submission Compliance Check
Verify section count, numbering, page limits, anonymization, and limitations section.

### Step 9: Final Checklist
Use the quality checklist for final verification.

## Best Practices

### Review Timing
- **Spaced review** - Wait 1-2 days after completing the draft before reviewing to maintain objectivity
- **Multiple rounds** - Conduct multiple review rounds, focusing on different aspects each time
- **Print review** - Print a hard copy for review; issues are easier to spot on paper

### Review Techniques
- **Reverse reading** - Read from conclusion backwards to check logical coherence
- **Read aloud** - Reading the paper aloud helps identify language issues
- **Reviewer perspective** - Assume you are a reviewer and read critically

### Common Issues
- Abstract too brief or too verbose
- Introduction lacks clear research question statement
- Method lacks sufficient detail for reproduction
- Results lack statistical significance tests
- Discussion doesn't address research limitations
- Figures/tables lack clear captions/labels, or contain in-figure title text <!-- policy:FIG.NO_IN_FIGURE_TITLE -->
- Display equations use `$$...$$` or `\[...\]` instead of `equation` <!-- policy:LATEX.EQ.DISPLAY_STYLE -->
- Long variable-like tokens are not wrapped with `\text{}` <!-- policy:LATEX.VAR.LONG_TOKEN_USE_TEXT -->
- Related Work comparison table uses `\cmark/\xmark/\pmark` without `pifont` + `xcolor` and unified macro definitions <!-- policy:LATEX.CMARK_XMARK_PMARK_MACROS -->
- Inconsistent citation formatting

## Summary

The Paper Self-Review skill provides a systematic paper quality checking process, helping researchers identify and resolve issues before submission, improving paper quality and acceptance rates.
