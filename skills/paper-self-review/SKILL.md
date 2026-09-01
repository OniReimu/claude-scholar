---
name: paper-self-review
description: This skill should be used when the user asks to "review paper quality", "check paper completeness", "self-review before submission", "audit this draft", "run the submission checklist", or "is this paper ready to submit". Runs the systematic pre-submission quality checklist — abstract/intro/method/experiment completeness, figure and table conformance (booktabs, direction indicators, resizebox, dimension budget), math/notation consistency, citation checks, prose-rule sweep (PROSE.*), internal-provenance leaks, and venue compliance. Report-and-propose only. Do not use for paragraph-level structural editing (claim-architecture-review, which runs BEFORE this), prose rewriting (ml-paper-writing / writing-anti-ai), or reviewer-response drafting (review-response).
version: 0.1.3
---

# Paper Self-Review

A systematic paper quality checking tool that helps researchers conduct comprehensive self-review before submission.

## ⚠️ Author Style Guide <!-- style:author-voice -->

> **MANDATORY**: Self-review 时必须对照 `policy/style-guide.md` 检查风格一致性。论文不仅要无错误，更要符合作者的个人写作指纹。

## Policy Rules

> 本 skill 执行的规则**逐条内联在 Review Process 的检查项里**，每处带 HTML 注释形式的规则标记——
> 那才是可执行的形态。此处不再镜像一份规则表：97 条规则的权威清单在 `policy/README.md` 的 Rule ID Registry，
> 单条定义在 `policy/rules/`，**冲突时以 `policy/rules/` 为准**。
> 紧凑版预防清单见 `policy/guardrail-checklist.md`。

> **为什么删表**：这张表此前有 73 行，其中 71 行的规则在本文件的检查项里已有 marker 且附带可执行的检查文本。
> 镜像表只增加了「改一条规则要同步四个文件」的成本，没有增加任何 agent 读得到的信息。

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
- Is each section organised by outcome logic (problem → design → why → evidence) rather than by the order the work happened, and does the Abstract match what the strongest evidence actually supports (re-frame the claim, never the reported set)? <!-- policy:PAPER.OUTCOME_LOGIC -->

### 3. Citation Completeness

Check the completeness and accuracy of citations:
- Are all citations present in the references? <!-- policy:CITE.VERIFY_VIA_API -->
- Does every concrete claim/quote have verifiable source support (quote → exact text + locator; paraphrase → matching span + page/§), with zero unresolved `[CLAIM NOT VERIFIED]` / `[QUOTE NOT VERIFIED]` markers? <!-- policy:CITE.CLAIM_SUPPORT_REQUIRED -->
- Is the reference format consistent? <!-- policy:BIBTEX.CONSISTENT_CITATION_KEY_FORMAT -->
- Are key related works cited?
- Do citations accurately reflect the original content?

### 4. Figure/Table Quality

Evaluate the quality and effectiveness of figures and tables:
- Do all figures/tables have clear captions + labels (no in-figure title text)? <!-- policy:FIG.NO_IN_FIGURE_TITLE -->
- Are figures consistent with `scientific-figure-making` style — colorblind-safe palette, font ≥ venue minimum, vector (PDF) export? <!-- policy:FIG.COLORBLIND_SAFE_PALETTE --> <!-- policy:FIG.FONT_GE_24PT --> <!-- policy:FIG.VECTOR_FORMAT_REQUIRED -->
- Does the system-overview / Figure 1 use a wide aspect ratio (≥ 2:1)? (owned by `paper-figure-generator`; guideline only outside it) <!-- policy:FIG.SYSTEM_OVERVIEW_ASPECT_RATIO_GE_2TO1 -->
- Do figures default to single column — is any full-width (`figure*`) figure (esp. system overview) actually dense enough to need `\textwidth`, or would a single column fit (demerit if so)? <!-- policy:FIG.COLUMN_WIDTH_JUSTIFICATION -->
- Do experiment subfigures avoid a lone subfigure on any row — 2 side-by-side (1×2) or a 4+ grid (2×2+)? <!-- policy:FIG.EXPERIMENT_SUBFIGURE_LAYOUT -->
- Do heatmaps with long row/col names abbreviate in-figure, define full names in the caption, and stay single-column? <!-- policy:FIG.HEATMAP_LABEL_ABBREVIATION -->
- If a research-gap teaser figure exists, is it single-column, placed before the system overview, and warranted (gap hard to convey in words)? Not every paper needs one. <!-- policy:FIG.RESEARCH_GAP_TEASER -->
- Are captions self-contained — non-experiment figures cover what / how / intent, while experiment figures & tables carry only "what" (finding/takeaway goes to prose per `EXP.TAKEAWAY_BOX`, never the caption)? (rule deprecated as a hard check; retained as a writing guideline owned by this skill and `ml-paper-writing` at review time) <!-- policy:FIG.SELF_CONTAINED_CAPTION -->
- Do tables use booktabs format? <!-- policy:TABLE.BOOKTABS_FORMAT -->
- Do table headers include direction indicators (↑/↓)? <!-- policy:TABLE.DIRECTION_INDICATORS -->
- Are tables wrapped in `\resizebox` to fit the column width (or do they satisfy all natural-fit exemptions)? <!-- policy:TABLE.RESIZEBOX_COLUMN_FIT -->
- Do comparison tables stay within the 3–4 dimension budget, single-column first, with an explicit reason for anything wider? <!-- policy:TABLE.DIMENSION_BUDGET -->
- For any full-width (`table*`) table, does `\resizebox` only shrink (in-table font ≤ body font) — not enlarge a sparse table? If it would enlarge, demote to single-column or add metrics/columns. <!-- policy:TABLE.FULLWIDTH_FONT_DENSITY -->
- Are prose, **captions, table cells, notation tables and appendices** free of internal provenance — script names, DPI notes, placeholder markers, draft meta-text, **result paths (`\path{experiments/results/...}`), data schema identifiers (`\texttt{empirical\_rate}`), internal fixture names ("As Golden G4"), and revision narrative ("the old bound is retracted", "superseded by", "renamed to avoid collision with")**? Eight of eleven leaks in the sweep that motivated this check sat outside body prose. Run `bash policy/lint.sh --rule PROSE.NO_INTERNAL_PROVENANCE <dir>` for P1–P5, then `bash policy/scripts/extract-undefined-identifiers.sh <dir>` and confirm every identifier is defined somewhere the reader can find it. <!-- policy:PROSE.NO_INTERNAL_PROVENANCE -->
- Do figures/tables support the text narrative?
- Are figures/tables clear and readable?
- Do formats comply with journal/conference requirements?

### 5. Writing Clarity

Check writing clarity and readability:
- Is the language concise and clear?
- Are empty intensifiers removed? <!-- policy:PROSE.INTENSIFIERS_ELIMINATION -->
- Are em-dashes fully eliminated (zero allowed — split into sentences, relative clauses, commas, or parentheses)? <!-- policy:PROSE.EM_DASH_RESTRICTION -->
- Is technical terminology used appropriately?
- Are sentence structures clear?
- Is paragraph organization logical?

### 5b. Prose Rule Coverage (anti-AI patterns + academic style)

Full policy-rule sweep (fixes live in `writing-anti-ai`; definitions in `policy/rules/`):
- No AI-tell sentence patterns — anaphora abuse, gerund-fragment litany, short punchy fragments, rhetorical self-answer, "Despite …" dismissal, superficial `-ing` analysis, copula dodge ("serves as"), "X is A, not B" copular negation contrast (zero tolerance; `rather than`/`instead of` only when the exclusion is load-bearing, and a plain negative predicate with no positive counterpart is fine), trailing afterthought, mid-sentence explanatory colon, ≥4-comma chains, cleft fronting ("that is what sets X"), hypothetical foil ("a method that only did X would …"), abstract agency ("the analogy's job", "carries decades of validation")? <!-- policy:PROSE.ANAPHORA_ABUSE --> <!-- policy:PROSE.GERUND_FRAGMENT_LITANY --> <!-- policy:PROSE.SHORT_PUNCHY_FRAGMENTS --> <!-- policy:PROSE.RHETORICAL_SELF_ANSWER --> <!-- policy:PROSE.DESPITE_DISMISSAL --> <!-- policy:PROSE.SUPERFICIAL_ING_SUFFIX --> <!-- policy:PROSE.COPULA_DODGE --> <!-- policy:PROSE.NEGATION_CONTRAST --> <!-- policy:PROSE.TRAILING_AFTERTHOUGHT --> <!-- policy:PROSE.MIDSENTENCE_COLON --> <!-- policy:PROSE.COMMA_OVERUSE --> <!-- policy:PROSE.CLEFT_CONSTRUCTION --> <!-- policy:PROSE.HYPOTHETICAL_FOIL --> <!-- policy:PROSE.ABSTRACT_AGENCY -->
- No AI-tell word/format tells — filler phrases, promotional language, formatting over-use, informal vocabulary, reflexive rule-of-three, Unicode arrows, vague attributions ("experts argue"), elegant variation (drifting terms)? <!-- policy:PROSE.FILLER_PHRASES --> <!-- policy:PROSE.PROMOTIONAL_LANGUAGE --> <!-- policy:PROSE.FORMATTING_RESTRAINT --> <!-- policy:PROSE.INFORMAL_VOCABULARY --> <!-- policy:PROSE.RULE_OF_THREE --> <!-- policy:PROSE.UNICODE_ARROWS --> <!-- policy:PROSE.VAGUE_ATTRIBUTIONS --> <!-- policy:PROSE.ELEGANT_VARIATION -->
- Sentence rhythm varies rather than sitting in one length band (stdev ≥10 words, 15–30-word band ≤55%), short sentences carry a checkable claim rather than announcing one ("The difficulty is structural."), and no setup-then-punchline split delivers a rebuttal in three words ("One might expect X. It does not.")? <!-- policy:PROSE.RHYTHM_VARIANCE --> <!-- policy:PROSE.ANNOUNCEMENT_SENTENCE --> <!-- policy:PROSE.THEATRICAL_SPLIT -->
- Academic style holds — sentence length bounded, paragraph topic-sentence first, hedging calibrated in BOTH directions (no may/could on backed results, AND no prove/demonstrate/guarantee beyond the evidence — every comparative claim carries a number/figure/citation anchor), abbreviations defined on first use, number-expression convention, no vague quantifiers, consistent tense, complete subsections, equations explained, related work organized by evolution? <!-- policy:PROSE.SENTENCE_LENGTH --> <!-- policy:PROSE.PARAGRAPH_TOPIC_SENTENCE --> <!-- policy:PROSE.HEDGING_DISCIPLINE --> <!-- policy:PROSE.ABBREVIATION_FIRST_USE --> <!-- policy:PROSE.NUMBER_EXPRESSION --> <!-- policy:PROSE.VAGUE_QUANTIFIERS --> <!-- policy:PROSE.TENSE_CONSISTENCY --> <!-- policy:PROSE.SUBSECTION_COMPLETENESS --> <!-- policy:PROSE.EQUATION_EXPLANATION --> <!-- policy:PROSE.RELATED_WORK_EVOLUTION -->
- No self-promotion scaffolding — each listed contribution names a specific result (a number, an artifact) rather than restating the abstract ("a novel method; extensive experiments; strong results"); novelty asserted via concrete difference from the closest prior work rather than "novel"/"to the best of our knowledge"/"for the first time"; no citation dumping (bracketed lists [3,7,9,12] where only one or two works matter — name them and say why)? <!-- policy:PROSE.PROMOTIONAL_LANGUAGE --> <!-- policy:PROSE.RELATED_WORK_EVOLUTION -->
- If this draft went through a compression / de-jargon / simplification pass: does every changed span hold its register? A replacement that is accurate but lower-register than what it replaced is still a defect (`route micropayments to peers` → `pay peers` loses the referent; `permanently excluding` → `excluding for good` reads as speech). Judge the **diff**, not the document, and prefer the wording the manuscript already uses elsewhere — report as `original → replacement → suggested wording → source`. **Word count and reduction percentage are reported only after this check passes.** <!-- policy:PROSE.REGISTER_PRESERVATION -->
- Register holds in **author-original** prose too, judged by the five classes rather than a wordlist — idiomatic adverbials (`at all`, `in the first place`, `ahead of time`), phrasal verbs displacing a Latinate verb (`comes with` → `entails`), judgment adjectives that are **not** already field terms (`is hard` → `is difficult`, but keep `cheap unlearning`), concrete-noun metaphors in predicate position (`the wall is a property` → the paper's own formal term), and internal-work-trace verbs (`what quarantine buys`)? Replacements come from wording the manuscript already uses. <!-- policy:PROSE.INFORMAL_VOCABULARY -->
- Does any technical phrase collide with a common English idiom, so a reviewer's first reading lands on the wrong sense (`a fair bit` meaning an unbiased bit, `on the order of` meaning a magnitude, `significant` meaning statistically significant)? Make the qualifier explicit. <!-- policy:PROSE.IDIOM_COLLISION -->
- Causal connectives chosen by type rather than defaulting to `, so` — `therefore` (entailment) / `hence` (continues the clause just established) / `thus` (by this means) / `consequently` (observed outcome), with subordination (`Because A, B`) preferred over any connective when two independent clauses are comma-spliced? Flag only three diagnosable classes — a design choice dressed as an inference, a causal claim the evidence does not support, and a proof step where `hence`/`thus` is the field convention. **Everything else stays**: blind adjudication of 42 real instances found no instance-level difference between pre-GPT papers and current drafts, so "could be more precise" is not a criterion — the signal is density (pre-GPT reference 0.18–0.28 per 1000 words against ~2.2 for the formal set), not any single sentence. `so that` / `so far` / `so large` are out of scope. <!-- policy:PROSE.CAUSAL_CONNECTIVE -->
- Are hyphenated compound modifiers either established terms of the field or used more than once? A compound coined for a single sentence (`community-shift-aware`, `brokerage-oriented`) charges the reader to decode it and never repays that cost. Measured reference: hapax compounds run 0.16 per 1000 words in 2019–2021 arXiv sources against 0.48 in 2025–2026. **Frequency is the test, not the construction** — a blockchain paper using `blockchain-based` throughout is compliant. <!-- policy:PROSE.ADHOC_COMPOUND_MODIFIER -->
- No AI lexicon fingerprint — zero tier-1 hits (delve, leverage, underscore, pivotal, seamless, intricate, tapestry, myriad, "paving the way for", "valuable insights"), tier-2 density under threshold (≤1 per sentence, ≤5 per file), and term-of-art uses (`loss landscape`, `robust`, `optimize`, `trajectory`) left untouched? <!-- policy:PROSE.AI_LEXICON -->
- No structural self-similarity — section-opening previews ("In this section, we present…"), section-closing recaps ("As we have seen…"), and inter-section stitching ("Having described X, we now…") removed, with at most one Introduction roadmap? Re-anchoring a construct defined several sections earlier is not a preview and survives, but only if it is **self-contained** (readable without holding the label→content binding — "The study asks three questions. First, what does…?" qualifies; "RQ1 establishes which…" does not, at any distance) **and** far enough that the reader would otherwise turn back. Judge the lookup cost; there is no numeric threshold, and the unit is sections, never pages. <!-- policy:PROSE.FRACTAL_SUMMARY -->
- No coined concept labels passing as established terms ("the supervision paradox", "workload creep") — each label either cites a source or is an explicit named contribution with a definition and consistent use? <!-- policy:PROSE.INVENTED_CONCEPT_LABEL -->
- No semicolons in body paragraphs — each one resolved either by a syntactic change (subordination, relative clause, or comma + a conjunction that carries real meaning) or by splitting into two sentences? A bare comma is not a resolution; a comma plus conjunction is. Prefer the syntactic change only when the second clause is under ten words, the relation is contrast/concession/complement, splitting would leave an echo fragment, and the merged sentence holds at most three commas — otherwise split. Conjunctions distributed by meaning rather than standardised on one. Inline-math conditioning notation, `\;` thin spaces, and list-item separators are exempt. <!-- policy:PROSE.SEMICOLON_RESTRICTION -->
- No proposition stated twice within a section — abstract lead-in plus evidence plus synonym-recap collapsed to the one placement closest to the evidence (deletion test: does removing it lose information)? <!-- policy:PROSE.RESTATEMENT_DILUTION -->
- Every sentence adds a falsifiable proposition — no meta-narration that names no variable, value, mechanism, or conclusion (test: could this sentence be moved verbatim into a different paper?), and no circular attribution where the `because` clause restates its own consequent? Judge by extracting the proposition, never by an embedding-similarity or filler-ratio threshold. Exempt: definitions, an abstraction cashed out by the very next sentence, and Ethics / Threats-to-Validity boilerplate. A paragraph whose sentences are *mostly* empty goes to `claim-architecture-review` P1 as a whole rather than sentence by sentence. <!-- policy:PROSE.SEMANTIC_IDLING -->
- Are cross-references woven into the prose (not bare "see Fig. 3")? <!-- policy:REF.WOVEN_CROSS_REFERENCE -->

### 6. LaTeX Math Conformance

Check whether math notation follows project rules:
- Are display equations written with `\begin{equation}...\end{equation}`? <!-- policy:LATEX.EQ.DISPLAY_STYLE -->
- Is raw `$$...$$` or `\[...\]` avoided for display equations? <!-- policy:LATEX.EQ.DISPLAY_STYLE -->
- Are inline equations written with `$...$` where appropriate?
- Are variable-like tokens longer than 3 letters wrapped with `\text{}` in math mode? <!-- policy:LATEX.VAR.LONG_TOKEN_USE_TEXT -->
- Are symbols consistent throughout the paper? <!-- policy:LATEX.NOTATION_CONSISTENCY -->
- Does pseudocode reuse the established math notation (not verbose prose) and abstract away standard operations? <!-- policy:PROSE.PSEUDOCODE_ABSTRACTION -->

### 7. Experiment Structure

Check experiment section completeness:
- Do experiment results include error bars? <!-- policy:EXP.ERROR_BARS_REQUIRED -->
- Are ablation studies in the Results section (not Discussion)? <!-- policy:EXP.ABLATION_IN_RESULTS -->
- Does each experiment subsection follow the required structure? <!-- policy:EXP.RESULTS_SUBSECTION_STRUCTURE -->
- If any result is fabricated/synthetic/dummy, is it explicitly disclosed in red uppercase in caption? <!-- policy:EXP.FABRICATED_RESULTS_CAPTION_DISCLOSURE -->
- If a subsection contains fabricated results, is there a subsection-level `[FABRICATED]` status declaration comment? <!-- policy:EXP.RESULTS_STATUS_DECLARATION_REQUIRED -->
- Does every result table/figure/subsection carry one of the four roles (prove the method works / explain where the gain comes from / show value in the target setting / rule out a competing explanation), with at least one experiment doing the fourth? Roleless ones go redesign → demote → delete, never delete-because-unfavourable. <!-- policy:EXP.EXPERIMENT_ROLE -->
- Are random seeds documented? <!-- policy:REPRO.RANDOM_SEED_DOCUMENTATION -->
- Are compute resources documented? <!-- policy:REPRO.COMPUTE_RESOURCES_DOCUMENTED -->

### 8. Submission Compliance

Check submission requirements:
- Are top-level sections ≤ 6? <!-- policy:PAPER.SECTION_HEADINGS_MAX_6 -->
- Is section numbering consistent? <!-- policy:SUBMIT.SECTION_NUMBERING_CONSISTENCY -->
- Does the paper meet the page limit? <!-- policy:SUBMIT.PAGE_LIMIT_STRICT -->
- Is double-blind anonymization correct? <!-- policy:ANON.DOUBLE_BLIND_ANONYMIZATION -->
- Is there a Limitations section? <!-- policy:ETHICS.LIMITATIONS_SECTION_MANDATORY -->
- 补 Limitations 与任何不利结果时，按「是否必须讨论 → 能否换目标解释 → 能否收缩主张到证据实际支持的范围」三步处置逐条过，并删掉情绪副词、回填数据集/指标/幅度/表号锚点；披露量由 `ETHICS.LIMITATIONS_SECTION_MANDATORY` 决定且优先，本条只约束措辞与放置，不减内容。 <!-- policy:PROSE.SELF_UNDERMINING -->

### 9. SoK Scope Checks (When SoK profile is active)

- Is there an explicit taxonomy with clear dimensions and boundaries? <!-- policy:SOK.TAXONOMY_REQUIRED -->
- Is the survey methodology (search/screening criteria) reported? <!-- policy:SOK.METHODOLOGY_REPORTING -->
- Is there at least one taxonomy-aligned big comparison table? <!-- policy:SOK.BIG_TABLE_REQUIRED -->
- Does conclusion/discussion include a concrete research agenda? <!-- policy:SOK.RESEARCH_AGENDA_REQUIRED -->

### 10. Empirical-SE / AI-security Scope Checks (When se-* or security-* profile is active)

- Are there explicit numbered RQs (2–4 typical; up to ~6 for multi-faceted security/systems evals), open-ended, each with a justification? Single-main-result / homogeneous evals may skip RQs (rule is then N/A). <!-- policy:SE.RESEARCH_QUESTIONS_EXPLICIT -->
- Is each RQ bound to its results section (heading reprints the RQ + "to answer RQx" signpost + RQ column in the glance table)? For numerical evals, does each RQ-subsection own a figure/table cluster closed by a takeaway box carrying the key number? <!-- policy:SE.RQ_SECTION_BINDING -->
- _(se-* only)_ Is Threats to Validity structured by category (construct/internal/external/conclusion), each a named threat + mitigation citing Wohlin? <!-- policy:SE.THREATS_TO_VALIDITY_STRUCTURED -->
- _(se-* only)_ Are implications actionable and stakeholder-segmented (For tool builders / standards bodies / practitioners), each tied to an RQ? <!-- policy:SE.ACTIONABLE_IMPLICATIONS -->

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
- [ ] Figures follow `scientific-figure-making` conventions (font sizing, vector export, palette, captions)
- [ ] Figure 1 exists and is a conceptual system overview (not an experiment plot)
- [ ] Non-experimental figures (system/pipeline/architecture/threat-model/comparison) are generated via `paper-figure-generator` (AutoFigure-Edit) by default
- [ ] Additional non-experimental figures are added when Figure 1 cannot clearly show key mechanism/protocol details
- [ ] Each Python plot = 1 file → 1 figure (no subplots); composite via LaTeX \subfigure <!-- policy:FIG.ONE_FILE_ONE_FIGURE -->
- [ ] Tables use booktabs format <!-- policy:TABLE.BOOKTABS_FORMAT -->
- [ ] Table headers include direction indicators (↑/↓) <!-- policy:TABLE.DIRECTION_INDICATORS -->
- [ ] Tables resizebox-fit to column width unless naturally fitting <!-- policy:TABLE.RESIZEBOX_COLUMN_FIT -->
- [ ] Comparison tables within 3–4 dimension budget, single-column first <!-- policy:TABLE.DIMENSION_BUDGET -->
- [ ] No internal provenance anywhere rendered — prose, captions, table cells, appendices (scripts, result paths, schema column names, internal fixture names, revision narrative, placeholders, meta-text); `policy/lint.sh --rule PROSE.NO_INTERNAL_PROVENANCE` is clean <!-- policy:PROSE.NO_INTERNAL_PROVENANCE -->
- [ ] Symbols consistent throughout paper <!-- policy:LATEX.NOTATION_CONSISTENCY -->
- [ ] For crypto-oriented security papers, core mechanism is presented as a structured Construction (Primitives/Parameters + named procedures) <!-- policy:PROSE.CRYPTO_CONSTRUCTION_TEMPLATE -->
- [ ] Empty intensifiers removed <!-- policy:PROSE.INTENSIFIERS_ELIMINATION -->
- [ ] Em-dashes fully eliminated (zero allowed) <!-- policy:PROSE.EM_DASH_RESTRICTION -->
- [ ] Experiment results include error bars <!-- policy:EXP.ERROR_BARS_REQUIRED -->
- [ ] Experiment results subsections each end with \fbox Takeaway box <!-- policy:EXP.TAKEAWAY_BOX -->
- [ ] Ablation studies in Results section <!-- policy:EXP.ABLATION_IN_RESULTS -->
- [ ] Experiment subsections follow required structure <!-- policy:EXP.RESULTS_SUBSECTION_STRUCTURE -->
- [ ] Fabricated/synthetic/dummy results are explicitly disclosed in red uppercase caption <!-- policy:EXP.FABRICATED_RESULTS_CAPTION_DISCLOSURE -->
- [ ] Subsections containing fabricated results include a `[FABRICATED]` status declaration comment <!-- policy:EXP.RESULTS_STATUS_DECLARATION_REQUIRED -->
- [ ] Every experiment carries one of the four roles; at least one rules out a competing explanation <!-- policy:EXP.EXPERIMENT_ROLE -->
- [ ] Sections organised by outcome logic, not by the order the work happened <!-- policy:PAPER.OUTCOME_LOGIC -->
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
- [ ] Limitations/unfavourable results went through the three-step disposition, with emotive adverbs removed and anchors kept — disclosure unchanged <!-- policy:PROSE.SELF_UNDERMINING -->
```

## Orchestrator Integration

This skill owns stage: **`self_review`**. It is the completeness / compliance review (additive) and runs AFTER `architecture_review`; it does NOT do paragraph-necessity / placement / cross-section-redundancy audit — that is `claim-architecture-review`, which runs first so completeness is judged on a de-bloated draft.

When invoked within an active research run (see `orchestrator/run-card.md`):

1. **Stage start**: Mark `self_review` → `in_progress`.
   - **Attached runs that have the `architecture_review` stage**: require `architecture_review` = `done` (or explicitly `skipped` with a re-bloat warning) AND `rewrite` = `done` or `skipped` — so the checklist runs on the post-relocation draft, not the pre-`rewrite` one. Also require `writeup` = `done`.
   - **Legacy runs** (`run.stages` has no `architecture_review` entry — created before this stage shipped): skip the above; verify only `writeup` = `done` and proceed (no architecture prerequisite, so the run is not stuck).
   - **No active run**: standalone, no precondition (see end of this section).
2. **Step A — Guardrail sweep** (auto-fix pass):
   - Run `bash policy/lint.sh --fix --profile <profile> .` to auto-fix safe guardrail violations.
   - Run `bash policy/lint.sh --constraint-type guardrail --profile <profile> .` to identify any remaining guardrail violations (assisted/none level) for manual review.
   - Record `gate_results.self_review.guardrail_clean = true`.
3. **Step B — Guidance review** (requires judgment):
   - Run `bash policy/lint.sh --constraint-type guidance --profile <profile> .` to check structural rules.
   - Execute the full quality checklist (structure, logic, citations, figures, math, experiments, submission compliance).
   - Record `gate_results.self_review.guidance_clean = true`.
4. **Stage end**: Request human approval before marking `done`.

**Expected run fields:**
- `artifacts.self_review.checklist_passed` — boolean
- `gate_results.self_review.guardrail_clean` — boolean (guardrail sweep passed)
- `gate_results.self_review.guidance_clean` — boolean (guidance review passed)

**Gate execution and persistence**:

The `self_review` stage has two sequential policy gates. After execution, persist results into run state:

```
gate_results.self_review = {
  last_run: "<ISO timestamp>",
  guardrail_clean: true|false,
  guidance_clean: true|false,
  summary: "<violation summary or 'All checks passed'>"
}
```

The stage may only be marked `done` if **both** `gate_results.self_review.guardrail_clean === true` **and** `gate_results.self_review.guidance_clean === true` **and** the user approves the checklist. `markStage()` enforces this via `validateGates()` — attempting to mark `done` without both keys set to `true` will throw an error. If either gate fails, the stage remains `in_progress`.

If no active run exists, proceed with standalone self-review (no orchestrator interaction).

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

### Step 10: Closure 判决 <!-- policy:PAPER.REVISION_CLOSURE -->
检查全部跑完后，对全稿输出一条判决，四档取其一：**STOP_REVISING**（无足以重启修订的实质根因，剩余 findings 进方向性建议或投稿准备轴）/ **ONE_BOUNDED_ROUND**（一个局部实质问题，点名根因 + 限定节段 + 限定一轮）/ **REOPEN_SUBSTANTIVE_REVISION**（claim 不成立、实验支撑缺口、结构性论证断裂）/ **UNASSESSED**（稿件缺节、含占位文本或截断，拒绝伪造整稿判决）。

依据只能是实质根因，逐条问"这条不改，稿件的主张还成立吗"。findings 数量、"还能更好"、接收概率猜测都不构成理由，也不设任何数值阈值。**防循环条款**：同一根因不得第二次触发 ONE_BOUNDED_ROUND 或 REOPEN——上一轮限定改写没解决它，就按新证据判 REOPEN 或写进已知限制。

输出为紧凑判决块（档位 / 1–2 句根因 / ≤3 条方向性建议 / 仅 ONE_BOUNDED_ROUND 附范围），不复述完整 violation report；判决写入 `self_review` 阶段的 note 或 violation report，供下一轮判决时回查。

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

<!-- policy:PROSE.OVER_DEFENSIVE -->
在完整性检查之后，按节隔离扫一遍过度防御：同一 caveat 是否有多个落点，是否有段落以免责声明收尾；再按句子形状查免责式否定谓语（作者/产物主语 + 拒绝主张：`We do not select an optimal policy`）——A 类翻成正面「我们做的是 Y」，事实性否定（定义/发现/数值结果）与三类保留（发现即否定命题、带理由的方法取舍、唯一列出被排除项处）不动。完整性检查是加法，这一条是减法，顺序不能反。
