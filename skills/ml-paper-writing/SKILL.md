---
name: ml-paper-writing
description: Write publication-ready ML/AI papers for NeurIPS, ICML, ICLR, ACL, AAAI, COLM. Use when drafting papers from research repos, conducting literature reviews, finding related work, verifying citations, or preparing camera-ready submissions. Includes LaTeX templates, citation verification workflows, and paper discovery/evaluation criteria.
version: 1.0.0
author: Orchestra Research
license: MIT
tags: [Academic Writing, NeurIPS, ICML, ICLR, ACL, AAAI, COLM, LaTeX, Paper Writing, Citations, Research]
dependencies: [semanticscholar, arxiv, habanero, requests]
---

# ML Paper Writing for Top AI Conferences

Expert-level guidance for writing publication-ready papers targeting **NeurIPS, ICML, ICLR, ACL, AAAI, and COLM**. This skill combines writing philosophy from top researchers (Nanda, Farquhar, Karpathy, Lipton, Steinhardt) with practical tools: LaTeX templates, citation verification APIs, and conference checklists.

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

## Core Philosophy: Collaborative Writing

**Paper writing is collaborative, but Claude should be proactive in delivering drafts.**

The typical workflow starts with a research repository containing code, results, and experimental artifacts. Claude's role is to:

1. **Understand the project** by exploring the repo, results, and existing documentation
2. **Deliver a complete first draft** when confident about the contribution
3. **Search literature** using web search and APIs to find relevant citations
4. **Refine through feedback cycles** when the scientist provides input
5. **Ask for clarification** only when genuinely uncertain about key decisions

**Key Principle**: Be proactive. If the repo and results are clear, deliver a full draft. Don't block waiting for feedback on every section—scientists are busy. Produce something concrete they can react to, then iterate based on their response.

---

## ⚠️ CRITICAL: Never Hallucinate Citations

**This is the most important rule in academic writing with AI assistance.**

### The Problem
AI-generated citations have a **~40% error rate**. Hallucinated references—papers that don't exist, wrong authors, incorrect years, fabricated DOIs—are a serious form of academic misconduct that can result in desk rejection or retraction.

### The Rule
**NEVER generate BibTeX entries from memory. ALWAYS fetch programmatically.** <!-- policy:CITE.VERIFY_VIA_API -->

| Action | ✅ Correct | ❌ Wrong |
|--------|-----------|----------|
| Adding a citation | Search API → verify → fetch BibTeX | Write BibTeX from memory |
| Uncertain about a paper | Mark as `[CITATION NEEDED]` | Guess the reference |
| Can't find exact paper | Note: "placeholder - verify" | Invent similar-sounding paper |

### When You Can't Verify a Citation

If you cannot programmatically verify a citation, you MUST:

```latex
% EXPLICIT PLACEHOLDER - requires human verification
\cite{PLACEHOLDER_author2024_verify_this}  % TODO: Verify this citation exists
```

**Always tell the scientist**: "I've marked [X] citations as placeholders that need verification. I could not confirm these papers exist."

### Recommended: Install Exa MCP for Paper Search

For the best paper search experience, install **Exa MCP** which provides real-time academic search:

**Claude Code:**
```bash
claude mcp add exa -- npx -y mcp-remote "https://mcp.exa.ai/mcp"
```

**Cursor / VS Code** (add to MCP settings):
```json
{
  "mcpServers": {
    "exa": {
      "type": "http",
      "url": "https://mcp.exa.ai/mcp"
    }
  }
}
```

Exa MCP enables searches like:
- "Find papers on RLHF for language models published after 2023"
- "Search for transformer architecture papers by Vaswani"
- "Get recent work on sparse autoencoders for interpretability"

Then verify results with Semantic Scholar API and fetch BibTeX via DOI.

---

## Workflow 0: Starting from a Research Repository

When beginning paper writing, start by understanding the project:

```
Project Understanding:
- [ ] Step 1: Explore the repository structure
- [ ] Step 2: Read README, existing docs, and key results
- [ ] Step 3: Identify the main contribution with the scientist
- [ ] Step 4: Find papers already cited in the codebase
- [ ] Step 5: Search for additional relevant literature
- [ ] Step 6: Outline the paper structure together
- [ ] Step 7: Draft sections iteratively with feedback
```

**Step 1: Explore the Repository**

```bash
# Understand project structure
ls -la
find . -name "*.py" | head -20
find . -name "*.md" -o -name "*.txt" | xargs grep -l -i "result\|conclusion\|finding"
```

Look for:
- `README.md` - Project overview and claims
- `results/`, `outputs/`, `experiments/` - Key findings
- `configs/` - Experimental settings
- Existing `.bib` files or citation references
- Any draft documents or notes

**Step 2: Identify Existing Citations**

Check for papers already referenced in the codebase:

```bash
# Find existing citations
grep -r "arxiv\|doi\|cite" --include="*.md" --include="*.bib" --include="*.py"
find . -name "*.bib"
```

These are high-signal starting points for Related Work—the scientist has already deemed them relevant.

**Step 3: Clarify the Contribution**

Before writing, explicitly confirm with the scientist:

> "Based on my understanding of the repo, the main contribution appears to be [X].
> The key results show [Y]. Is this the framing you want for the paper,
> or should we emphasize different aspects?"

**Never assume the narrative—always verify with the human.**

**Step 4: Search for Additional Literature**

Use web search to find relevant papers:

```
Search queries to try:
- "[main technique] + [application domain]"
- "[baseline method] comparison"
- "[problem name] state-of-the-art"
- Author names from existing citations
```

Then verify and retrieve BibTeX using the citation workflow below.

**Step 5: Deliver a First Draft**

**Be proactive—deliver a complete draft rather than asking permission for each section.**

If the repo provides clear results and the contribution is apparent:
1. Write the full first draft end-to-end
2. Present the complete draft for feedback
3. Iterate based on scientist's response

If genuinely uncertain about framing or major claims:
1. Draft what you can confidently
2. Flag specific uncertainties: "I framed X as the main contribution—let me know if you'd prefer to emphasize Y instead"
3. Continue with the draft rather than blocking

**Questions to include with the draft** (not before):
- "I emphasized X as the main contribution—adjust if needed"
- "I highlighted results A, B, C—let me know if others are more important"
- "Related work section includes [papers]—add any I missed"

---

## When to Use This Skill

Use this skill when:
- **Starting from a research repo** to write a paper
- **Drafting or revising** specific sections
- **Conducting literature reviews** and finding related work
- **Discovering recent papers** in your research area
- **Finding and verifying citations** for related work
- **Formatting** for conference submission
- **Resubmitting** to a different venue (format conversion)
- **Iterating** on drafts with scientist feedback

**Always remember**: First drafts are starting points for discussion, not final outputs.

---

## Workflow: Literature Research & Paper Discovery

When conducting literature reviews, finding related work, or discovering recent papers, use this workflow to systematically search, evaluate, and select ML papers.

### Workflow 5: Finding and Evaluating Papers

```
Literature Research Process:
- [ ] Step 1: Define search scope and keywords
- [ ] Step 2: Search arXiv and academic databases
- [ ] Step 3: Screen papers by title/abstract
- [ ] Step 4: Evaluate paper quality (5 dimensions)
- [ ] Step 5: Select top papers and extract citations
- [ ] Step 6: Verify citations programmatically
```

**Step 1: Define Search Scope**

Identify specific research areas, methods, or applications:
- **Technique-focused**: `transformer architecture`, `graph neural networks`, `self-supervised learning`
- **Application-focused**: `medical image analysis`, `reinforcement learning for robotics`, `language model alignment`
- **Problem-focused**: `out-of-distribution generalization`, `continual learning`, `fairness in ML`

**Step 2: Search arXiv**

Use arXiv search with targeted keywords:
```
URL Pattern:
https://arxiv.org/search/?searchtype=all&query=KEYWORDS&abstracts=show&order=-announced_date_first

Example Searches:
- https://arxiv.org/search/?searchtype=all&query=graph+neural+networks&abstracts=show&order=-announced_date_first
- https://arxiv.org/search/?cat:cs.LG+AND+all:transformer&abstracts=show&order=-announced_date_first
```

**Tips:**
- Combine keywords with `+` for AND
- Filter by categories: `cs.LG`, `cs.AI`, `cs.CV`, `cs.CL`
- Sort by `announced_date_first` for recent papers
- Use Chrome MCP tools when available for automation

**Step 3: Screen Papers**

Quick screening by title and abstract:
- Relevance to research topic
- Novelty of contribution
- Venue/reputation of authors
- Code availability (check for GitHub links)

**Step 4: Evaluate Quality**

Use the 5-dimension quality criteria:

| Dimension | Weight | Evaluation Focus |
|-----------|--------|------------------|
| **Innovation** | 30% | Novelty and originality |
| **Method Completeness** | 25% | Clarity and reproducibility |
| **Experimental Thoroughness** | 25% | Validation depth |
| **Writing Quality** | 10% | Presentation clarity |
| **Relevance & Impact** | 10% | Domain importance |

**Scoring**: Rate each dimension 1-5, calculate weighted total

**Step 5: Select and Extract**

- Rank papers by total score
- Select top papers for detailed review
- Extract metadata: title, authors, arXiv ID, abstract
- Note code repository links

**Step 6: Verify Citations**

For selected papers, verify citations using Semantic Scholar API:
- Fetch BibTeX programmatically via DOI
- Mark unverified citations as `[CITATION NEEDED]`
- Store in bibliography with verification status

### When to Use Literature Research

Use this workflow when:
- **Starting a new project**: Find related work and baselines
- **Writing Related Work section**: Discover recent papers in your area
- **Staying updated**: Track recent publications in your field
- **Finding baselines**: Identify state-of-the-art methods for comparison
- **Literature review**: Comprehensive survey of research area

### Quality Thresholds

- **Excellent**: 4.0+ (include definitely)
- **Good**: 3.5-3.9 (include if relevant)
- **Fair**: 3.0-3.4 (include if highly relevant)
- **Poor**: <3.0 (exclude unless essential)

### Reference Files

For detailed literature research guidance:
- **`references/literature-research/arxiv-search-guide.md`** - arXiv search strategies and URL patterns
- **`references/literature-research/paper-quality-criteria.md`** - Detailed 5-dimension evaluation rubrics

---

## Knowledge Base: Writing Patterns from ML Papers

This skill maintains a curated knowledge base of writing patterns, techniques, and requirements extracted from successful ML conference papers. The knowledge base grows as you analyze more papers.

### Knowledge Organization

The knowledge base is organized into 4 categories at `references/knowledge/`:

| Category | File | Contents |
|----------|------|----------|
| **Structure** | `structure.md` | Paper organization, IMRaD patterns, transitions, section flow |
| **Writing Techniques** | `writing-techniques.md` | Sentence patterns, transition phrases, clarity techniques |
| **Submission Guides** | `submission-guides.md` | Venue requirements (NeurIPS, ICML, ICLR, ACL, AAAI, COLM) |
| **Review Response** | `review-response.md` | Rebuttal strategies, addressing reviewer comments |

### How the Knowledge Base is Maintained

The **paper-miner agent** automatically extracts and categorizes writing knowledge from papers you provide:

```
You: "Learn writing techniques from this NeurIPS paper: path/to/paper.pdf"
↓
paper-miner analyzes the paper
↓
Extracts patterns → Categorizes into 4 types → Updates knowledge files
↓
Knowledge grows with each paper analyzed
```

**What gets extracted:**
- **Structure patterns**: How successful papers organize sections, transition between topics
- **Writing techniques**: Sentence templates, transition phrases, clarity methods
- **Venue requirements**: Page limits, required sections, formatting rules
- **Rebuttal strategies**: How to respond to specific reviewer concerns

### When to Use the Knowledge Base

**For writing patterns:**
- Stuck on how to phrase a transition? Check `writing-techniques.md`
- Need structure inspiration? Browse `structure.md`
- Writing rebuttal? Consult `review-response.md`

**For venue requirements:**
- Submitting to NeurIPS? See `submission-guides.md` for checklist
- Converting between venues? Compare page limits and requirements
- Unsure about required sections? Each venue has specific requirements

### Contributing to the Knowledge Base

Every paper you analyze makes the knowledge base richer for future use:

```bash
# Trigger paper-miner from any context
"Extract writing patterns from this paper: path/to/paper.pdf"
"Analyze structure of https://arxiv.org/abs/2301.xxxxx"
"What writing techniques does this ICLR paper use?"
```

The paper-miner agent:
1. Extracts paper content (PDF, DOCX, or arXiv link)
2. Analyzes IMRaD structure and writing patterns
3. Identifies venue-specific requirements
4. Updates appropriate knowledge files with new patterns
5. Reports what was added with source attribution

### Knowledge Base Principles

**Actionable patterns only**: Each entry provides reusable techniques with examples.

**Source attribution**: Every pattern cites the paper it came from for traceability.

**No duplicates**: Checks existing content before adding new patterns.

**Quality over quantity**: Focus on techniques that work, not comprehensive lists.

See `references/knowledge/README.md` for complete knowledge base documentation.

---

## Balancing Proactivity and Collaboration

**Default: Be proactive. Deliver drafts, then iterate.**

| Confidence Level | Action |
|-----------------|--------|
| **High** (clear repo, obvious contribution) | Write full draft, deliver, iterate on feedback |
| **Medium** (some ambiguity) | Write draft with flagged uncertainties, continue |
| **Low** (major unknowns) | Ask 1-2 targeted questions, then draft |

**Draft first, ask with the draft** (not before):

| Section | Draft Autonomously | Flag With Draft |
|---------|-------------------|-----------------|
| Abstract | Yes | "Framed contribution as X—adjust if needed" |
| Introduction | Yes | "Emphasized problem Y—correct if wrong" |
| Methods | Yes | "Included details A, B, C—add missing pieces" |
| Experiments | Yes | "Highlighted results 1, 2, 3—reorder if needed" |
| Related Work | Yes | "Cited papers X, Y, Z—add any I missed" |

**Only block for input when:**
- Target venue is unclear (affects page limits, framing)
- Multiple contradictory framings seem equally valid
- Results seem incomplete or inconsistent
- Explicit request to review before continuing

**Don't block for:**
- Word choice decisions
- Section ordering
- Which specific results to show (make a choice, flag it)
- Citation completeness (draft with what you find, note gaps)

---

## The Narrative Principle

**The single most critical insight**: Your paper is not a collection of experiments—it's a story with one clear contribution supported by evidence.

Every successful ML paper centers on what Neel Nanda calls "the narrative": a short, rigorous, evidence-based technical story with a takeaway readers care about.

**Three Pillars (must be crystal clear by end of introduction):**

| Pillar | Description | Example |
|--------|-------------|---------|
| **The What** | 1-3 specific novel claims within cohesive theme | "We prove that X achieves Y under condition Z" |
| **The Why** | Rigorous empirical evidence supporting claims | Strong baselines, experiments distinguishing hypotheses |
| **The So What** | Why readers should care | Connection to recognized community problems |

**If you cannot state your contribution in one sentence, you don't yet have a paper.**

---

## Cross-Skill Integration Map

The paper writing workflow orchestrates multiple skills at specific steps:

| Step | Skill | Purpose |
|------|-------|---------|
| Step 2 | `paper-figure-generator` | Generate editable SVG Figure 1 via AutoFigure-Edit (system overview, pipeline, architecture) |
| Step 6 | `paper-figure-generator` | System architecture diagram for System Model section (method.txt → SVG → PDF) |
| Step 8b | `rules/experiment-reproducibility.md` | Random seeds, config recording, checkpoint management |
| Step 8c | `results-analysis` | Statistical analysis, figure/table generation, visualization selection |
| Step 8c | `figures4papers` reference | Publication-ready Python plotting style |
| Step 5 | `citation-verification` | Validate references in Background & Related Work |
| Step 11 | `paper-self-review` | Multi-item quality checklist (includes figure/title and LaTeX math conformance) |
| Step 11 | `citation-verification` | Final reference validation |
| Step 11 | `writing-anti-ai` | Remove AI writing patterns if needed |

**Knowledge base** (populated by `paper-miner` agent, used throughout writing):
- `references/knowledge/structure.md` → Section organization patterns (Steps 4, 5, 6)
- `references/knowledge/writing-techniques.md` → Sentence templates, transitions (Steps 3, 4, 9)
- `references/knowledge/submission-guides.md` → Venue requirements (Step 11)

---

## Paper Structure Workflow

### Workflow 1: Writing a Complete Paper (Iterative)

Copy this checklist and track progress. **Each step involves drafting → feedback → revision:**

```
Paper Writing Progress:
- [ ] Step 1: Define the one-sentence contribution (with scientist)
- [ ] Step 2: Draft Figure 1 → get feedback → revise
- [ ] Step 3: Draft abstract → get feedback → revise
- [ ] Step 4: Draft introduction → get feedback → revise
- [ ] Step 5: Draft methods → get feedback → revise
- [ ] Step 6: Design experiment plan (contribution-aligned)
- [ ] Step 7: Execute experiments → collect results
- [ ] Step 8: Analyze results → generate figures/tables
- [ ] Step 9: Draft experiments section → get feedback → revise
- [ ] Step 10: Draft related work → get feedback → revise
- [ ] Step 11: Draft limitations → get feedback → revise
- [ ] Step 12: Complete paper checklist (required)
- [ ] Step 13: Final review cycle and submission
```

**Step 1: Define the One-Sentence Contribution**

**This step requires explicit confirmation from the scientist.**

Before writing anything, articulate and verify:
- What is the single thing your paper contributes?
- What was not obvious or present before your work?

> "I propose framing the contribution as: '[one sentence]'. Does this capture
> what you see as the main takeaway? Should we adjust the emphasis?"

**Step 2: Draft Figure 1**

Figure 1 deserves special attention—many readers skip directly to it.
- Convey core idea, approach, or most compelling result
- Use vector graphics (PDF/EPS for plots) <!-- policy:FIG.VECTOR_FORMAT_REQUIRED -->
- Write captions that stand alone without main text <!-- policy:FIG.SELF_CONTAINED_CAPTION -->
- **Accessibility**: 8% of men have color vision deficiency — use colorblind-safe palettes (Okabe-Ito or Paul Tol), verify grayscale readability, differentiate lines by style (solid/dashed/dotted) not just color <!-- policy:FIG.COLORBLIND_SAFE_PALETTE -->
- **Figure policy (default, mandatory):**
  - **Figure 1 is required** and should be a conceptual system overview generated via `paper-figure-generator` (AutoFigure-Edit, editable SVG -> PDF).
  - **All non-experimental figures default to `paper-figure-generator` + AutoFigure-Edit**, including `system-overview`, `pipeline`, `architecture`, `threat-model`, and `comparison`.
  - **Experimental figures remain separate** (metrics curves/bars/scatter, ablation plots, runtime/accuracy tradeoffs) and should be produced in Step 8 with plotting workflow.
- **MANDATORY for conceptual diagrams** (system overviews, pipelines, architectures): **activate `paper-figure-generator` skill NOW** to generate editable SVG figures via AutoFigure-Edit.
  - For `system-overview` / `pipeline` / `architecture` outputs, keep aspect ratio `width:height >= 2:1` (e.g., 2.1:1, 3:1), avoid near-square layouts <!-- policy:FIG.SYSTEM_OVERVIEW_ASPECT_RATIO_GE_2TO1 -->
  - Recommended workflow: write `figures/{slug}/brief.md` (see `paper-figure-generator/references/figure-brief.md`) → write `figures/{slug}/method.txt` → run `bash skills/paper-figure-generator/scripts/doctor.sh` → run `bash skills/paper-figure-generator/scripts/generate.sh ...` → run `bash skills/paper-figure-generator/scripts/svg-to-pdf.sh ...`
  - Keep `figures/{slug}/run.json` for reproducibility
  - Do NOT skip this step — Figure 1 is critical for reviewer first impressions.
- **When to add additional non-experimental figures (Figure 2+):**
  - Add one when Figure 1 cannot clearly express a key mechanism (e.g., synchronization, recovery, label transformation, chained verification).
  - Add one when the method includes ordered multi-role interactions (verifier/prover/server/client, etc.) that require a stepwise protocol view.
  - Add one when Figure 1 becomes overloaded (too many crossing flows or mixed abstraction levels) and must be split into overview + mechanism/process detail.
  - Keep these added figures conceptual and generate them with `paper-figure-generator` by default.

### Paper Section Structure

The final paper follows this structure. Strictly control to **6 top-level numbered sections** (+ Abstract). <!-- policy:PAPER.SECTION_HEADINGS_MAX_6 -->

| § | Section | Core Content |
|---|---------|-------------|
| — | Abstract | 5-sentence formula |
| 1 | Introduction | Problem → Contribution list → Approach overview |
| 2 | Background & Related Work | Definitions, notation, existing work (methodological grouping), qualitative comparison table (✓/✗/◐) |
| 3 | System Model | Problem formulation, system architecture, workflow overview, threat model |
| 4 | Our Approach | Technical method, algorithm/pseudocode, design decisions |
| 5 | Experiments | Settings → Results → Discussion & Analysis (including limitations) |
| 6 | Conclusion | Summary, key findings, future work |
| — | Acknowledgments | Funding, collaborators (remove during anonymous review, restore for camera-ready) |

**Section management rules:**
- **Maximum 6 numbered sections** — do not proliferate top-level headings <!-- policy:PAPER.SECTION_HEADINGS_MAX_6 -->
- Each `\subsection` must justify its existence with ≥ 2 substantive paragraphs of content
- If a subsection is thin (< 2 paragraphs), demote to inline heading: `\noindent\textbf{Heading.}` followed by text on the same line
- Same rule applies to `\subsubsection` — thin content uses `\noindent\textbf{}` instead
- Never create a sub(sub)section containing only one paragraph or a single short list

**Bibliography rules:**
- **Default to BibTeX** (not BibLaTeX) — use `\bibliographystyle{...}` + `\bibliography{refs}` with `bibtex main` compilation. Most top venue templates (NeurIPS, ICML, ICLR, ACL) use BibTeX by default.
- Store all entries in a single `.bib` file (e.g., `refs.bib` or `references.bib`)
- NEVER generate BibTeX entries from memory — always fetch programmatically via DOI/Semantic Scholar (see citation-verification skill) <!-- policy:CITE.VERIFY_VIA_API --> <!-- policy:BIBTEX.CONSISTENT_CITATION_KEY_FORMAT -->
- Compilation order: `pdflatex → bibtex → pdflatex → pdflatex`

**Punctuation rules (anti-AI):**
- **Em-dash (`---`)**: Do NOT use em-dashes for parenthetical insertions. Instead use: (1) a new sentence, (2) a relative clause (`, which...`), or (3) parenthetical commas. ≥ 2 em-dashes per paragraph is a strong AI signal. En-dash (`--`) for ranges/compounds is fine. <!-- policy:PROSE.EM_DASH_RESTRICTION -->
- **Colon (`:`)**: Do NOT use colons to introduce 3+ item inline enumerations (e.g., "X: A, B, and C"). Instead break into separate sentences or use "such as"/"including". Exception: numbered step lists ("(1)...(2)...") and formal definitions are standard.
- See `writing-anti-ai` skill, Patterns #13 and #13b for detailed examples and fix strategies.

---

**Step 3: Write Abstract (5-Sentence Formula)**

From Sebastian Farquhar (DeepMind):

```
1. What you achieved: "We introduce...", "We prove...", "We demonstrate..."
2. Why this is hard and important
3. How you do it (with specialist keywords for discoverability)
4. What evidence you have
5. Your most remarkable number/result
```

**Delete** generic openings like "Large language models have achieved remarkable success..."

**Step 4: Write Introduction (→ §1, 1-1.5 pages max)**

Must include:
- 2-4 bullet contribution list (max 1-2 lines each in two-column format)
- Clear problem statement
- Brief approach overview
- Methods should start by page 2-3 maximum

**Step 5: Write Background & Related Work (→ §2)**

This section combines foundational knowledge and positioning against existing work.

**Structure:**

```
\section{Background and Related Work}

\subsection{Preliminaries}          % Definitions, notation, formal setup
  [Only if enough content for ≥ 2 paragraphs; otherwise use \noindent\textbf{Preliminaries.}]

\subsection{[Topic Group A]}        % e.g., "Federated Learning"
  [Methodological grouping of related work]

\subsection{[Topic Group B]}        % e.g., "Privacy-Preserving ML"
  [Methodological grouping of related work]
```

**Writing principles:**
- Organize methodologically, NOT paper-by-paper
- **Good:** "One line of work uses Floogledoodle's assumption [refs] whereas we use Doobersnoddle's assumption because..."
- **Bad:** "Snap et al. introduced X while Crackle et al. introduced Y."
- Cite generously — reviewers likely authored relevant papers
- End with a clear **positioning statement**: how your work differs from and advances beyond existing approaches

**REQUIRED: Include a qualitative comparison table.** This table uses **symbols** (✓ / ✗ / ◐) or qualitative labels (High / Medium / Low) to compare features/properties — **NOT numerical values**. Numerical results belong in §5 Experiments. Use templates from `references/latex-style-guide.md`:
- **Feature Comparison Matrix** (✓/✗/◐ style) — for binary or ternary feature comparison
- **Qualitative Comparison Table** (multi-column text descriptions) — for nuanced differences
- For ✓/✗/◐ style matrices, define `\cmark/\xmark/\pmark` with `pifont` + `xcolor` in preamble (avoid template/package-default symbol commands that can conflict) <!-- policy:LATEX.CMARK_XMARK_PMARK_MACROS -->

> **Table scope separation**: §2 comparison table = qualitative properties (supports X? has property Y?). §5 experiment tables = quantitative metrics (accuracy, F1, latency). Both cover the same baselines and dimensions, but at different granularity levels.

**Step 6: Write System Model (→ §3)**

This section formally defines **what you are solving** and **under what assumptions**.

**MANDATORY: Notation Table.** This section MUST open with a notation table (`Table~\ref{tab:notation}`) that defines all symbols used throughout the paper. This table is the single source of truth — all subsequent sections (Methods, Experiments, Discussion) must use notation consistent with this table. <!-- policy:LATEX.NOTATION_CONSISTENCY -->

Best practices for the notation table:
- Use `threeparttable` + `\resizebox{\linewidth}{!}` for clean formatting
- Two columns: `Symbol | Meaning`
- Group by logical section using `\midrule` separators and optional `\cellcolor` for visual grouping (e.g., main text symbols vs. complexity analysis symbols)
- Add `\begin{tablenotes}` to explain grouping logic
- Place at `[t]` position for top-of-page placement

```latex
\begin{table}[t]
\caption{Notations}
\label{tab:notation}
\centering
\vspace{5pt}
\begin{threeparttable}
\resizebox{\linewidth}{!}{%
\begin{tabular}{c|l}
\toprule
\textbf{Symbol} & \multicolumn{1}{c}{\textbf{Meaning}} \\
\midrule
\cellcolor{pink!25} $x$ & Input data point. \\
\cellcolor{pink!25} $\theta$ & Model parameters. \\
% ... main text symbols ...
\midrule
\cellcolor{yellow!10} $\mathcal{O}(\cdot)$ & Asymptotic complexity. \\
% ... analysis-only symbols ...
\bottomrule
\end{tabular}%
}
\begin{tablenotes}
\scriptsize
\item Upper part: notations in the main text; lower part: complexity analysis.
\end{tablenotes}
\end{threeparttable}
\end{table}
```

**Components** (include all that apply to the paper context):

- **Problem Definition**: Formal mathematical formulation (input space, output space, objective function, constraints)
- **System Architecture**: High-level architecture — participants, components, communication topology
- **Workflow Overview**: End-to-end pipeline — stages, data flow between components
- **Threat Model** (security/privacy papers): Attacker capabilities, attack surface, trust assumptions, security goals
- **Assumptions**: Explicit list (e.g., "honest-but-curious server", "IID data distribution")

**Guidelines:**
- All notation in subsequent sections MUST match `Table~\ref{tab:notation}` — never introduce a symbol without defining it in the notation table first
- Use `\begin{equation}...\end{equation}` for display equations; do not use raw `$$...$$` or `\[...\]` <!-- policy:LATEX.EQ.DISPLAY_STYLE -->
- Inline equations can use `$...$`
- In math mode, if a variable-like token has more than 3 letters, write it with `\text{}` (e.g., `\text{score}`, `\text{total_loss}`), not italic math identifiers <!-- policy:LATEX.VAR.LONG_TOKEN_USE_TEXT -->
- Keep the scope tight — this section defines the problem, NOT the solution (§4)
- For non-security papers, this section may be titled "Problem Setup" or "Problem Formulation"
- If the system model is simple enough (e.g., standard supervised learning), it may be demoted to a subsection of §2 or §4

> **MANDATORY: Generate at least one non-experimental conceptual figure for this section.** Use `paper-figure-generator` skill to create the diagram via AutoFigure-Edit (brief.md → method.txt → generate SVG → convert to PDF). Select from `system-overview`, `pipeline`, `threat-model`, or `architecture` layout as appropriate.
>
> This figure is usually Figure 2 (after Figure 1). If one figure is insufficient, add Figure 3+ for mechanism/protocol details (e.g., synchronization/recovery flow, chained verification steps), and still keep them in the conceptual diagram track (AutoFigure-Edit). Do NOT proceed to Step 7 without producing the required conceptual figure(s).
>
> **MANDATORY OUTPUT for Step 6:**
> - [ ] Notation table (`Table~\ref{tab:notation}`) with all symbols used in the paper
> - [ ] LaTeX text for §3 (problem definition + architecture + threat model as applicable)
> - [ ] Architecture/workflow conceptual figure file(s) (via `paper-figure-generator`, SVG→PDF)
> - [ ] `\label{fig:system-model}` reference in the LaTeX text

**Step 7: Write Methods / Our Approach (→ §4)**

Enable reimplementation:
- Conceptual outline or pseudocode
- All hyperparameters listed
- Architectural details sufficient for reproduction
- Present final design decisions; ablations go in §5 Experiments

**Algorithm / Pseudocode** (if your method involves a non-trivial procedure):

Follow the Algorithm Presentation Pattern from `references/knowledge/structure.md`:
1. High-level overview first (1-2 paragraphs explaining intuition)
2. Mathematical formulation (objective function, constraints)
3. Algorithm pseudocode block (`algorithm` + `algorithmic` LaTeX packages)
4. Implementation details (practical choices not captured in pseudocode)

Pseudocode guidelines:
- Use `\begin{algorithm}[t]` with `\caption{}` and `\label{alg:name}`
- Number lines with `\algorithmic[1]` for easy reference in text
- Keep pseudocode at the right abstraction level — highlight what is **novel**, abstract away standard operations (e.g., "Train with SGD" not 10 lines of gradient update)
- Use consistent notation: match variable names between math formulation, pseudocode, and prose
- If the algorithm is short (< 5 lines), inline description may suffice; reserve `Algorithm` floats for procedures with ≥ 5 steps or non-obvious control flow
- For crypto-oriented security papers, present the core protocol as a structured construction (e.g., `Construction 1`) with explicit `Primitives`, `Parameters`, and named procedures (`Setup/Commit/Verify` or equivalent) <!-- policy:PROSE.CRYPTO_CONSTRUCTION_TEMPLATE -->

**Cross-reference style (mandatory in manuscript text):** <!-- policy:REF.CROSS_REFERENCE_STYLE -->
- Figure: `Fig.~\ref{...}`
- Table: `Table~\ref{...}`
- Section: `\S\ref{...}`
- Appendix: `\textbf{Appendix~\ref{...}}`
- Equation: `\eqref{...}`
- Algorithm: `Algorithm~\ref{...}`
- Listing: `Listing~\ref{...}`

**Step 8: Experiment Workflow (Plan → Execute → Analyze)**

**This is the critical step most automated workflows skip.** Before writing §5, you MUST complete all three sub-steps below and produce their mandatory outputs. Do NOT skip ahead to Step 9.

---

**8a. Design Experiment Plan (Contribution-Aligned):**

Identify contribution paradigm and match evidence standard:

| Paradigm | Evidence Standard | Typical Evaluation |
|----------|------------------|-------------------|
| **New Method/Algorithm** | Quantitative superiority on established benchmarks | Accuracy/F1/BLEU vs baselines, ablation studies, convergence analysis, complexity analysis |
| **New System** | End-to-end performance + component analysis | Throughput/latency/cost, scalability experiments, real deployment metrics |
| **New Dataset/Benchmark** | Utility demonstration + quality analysis | Baseline results, inter-annotator agreement, dataset statistics, bias analysis |
| **New Measurement/Analysis** | Reproducibility + insight validity | Statistical tests, effect sizes, multiple datasets, robustness checks |
| **New Theory/Framework** | Formal proofs + empirical validation | Synthetic experiments confirming predictions, real-world case studies |
| **User Study / HCI** | External validity + qualitative rigor | IRB approval, participant demographics, interview coding, thematic analysis, mixed methods |
| **Security/Privacy** | Threat model coverage + attack/defense evaluation | Attack success rates, defense overhead, adversarial robustness, formal guarantees |

Generate experiment plan and **present to user for explicit confirmation before proceeding**:

```
Experiment Plan:
1. Main Evaluation
   - Claim it supports: [from Step 1 contribution]
   - Datasets: [specific names, sizes, splits]
   - Baselines: [at least 3-5 strong, recent baselines from venue norms]
   - Metrics: [primary + secondary, with direction ↑/↓]
   - Statistical methodology: [seeds, runs, error bars type]

2. Ablation Study
   - Components to ablate: [each key design choice]
   - Expected insight: [what each ablation reveals]

3. Analysis Experiments
   - Sensitivity / qualitative / efficiency analysis

4. [Paradigm-specific experiments]
   - User study protocol / Attack evaluation / Scalability test / etc.
```

Validate alignment — for each experiment, verify: `Claim (Step 1) → Experiment Design → Expected Evidence → Planned Figure/Table`. If any claim lacks support, add experiments. If any experiment doesn't support a claim, move to appendix.

> **GATE: Do NOT proceed to 8b until the user confirms the experiment plan.**

---

**8b. Execute Experiments:**

**Write and run experiment code NOW.** This is not a planning step — produce actual code and actual results.

Concrete actions (execute in order):
1. **Write experiment scripts**: Python files using the project's codebase and config system (Hydra/OmegaConf), with `set_seed()` at entry point
2. **Run the scripts**: Execute via `uv run script.py` (default Python runner), capture stdout/stderr
3. **Collect raw results**: Save to CSV/JSON files in the project's results directory
4. **Log metadata**: GPU type, total hours, library versions

**Python tooling**: All experiment scripts default to `uv` for execution and dependency management:
- Run scripts: `uv run python train.py` or `uv run script.py`
- Add dependencies: `uv add torch transformers` (updates `pyproject.toml`)
- Sync environment: `uv sync`
- Record environment: `uv pip freeze > requirements.txt`

```python
import random, numpy as np, torch

def get_device() -> torch.device:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Usage: every experiment script must call both at entry point
device = get_device()
set_seed(42)
model = model.to(device)
```

Reproducibility requirements (per `rules/experiment-reproducibility.md`):
- Save Hydra config to outputs directory (auto-save enabled by default)
- Record environment: `uv pip freeze > requirements.txt`, log GPU info
- Checkpoint naming: `best_model.pt`, `checkpoint_epoch_N.pt`, `checkpoint_latest.pt`
- Record dataset hash or version tag

> **Compute constraint handling**: If experiments require GPU or extended compute that exceeds the current environment, you MUST still: (1) write complete, runnable experiment scripts, (2) write a launch script that runs all experiments end-to-end, (3) present scripts to user and **wait for user to run them and provide raw results** before proceeding to 8c. Do NOT skip to Step 9 with placeholder results.

> **GATE: Do NOT proceed to 8c until raw result files (CSV/JSON/logs) exist — either from direct execution or from user-provided results.**

---

**8c. Analyze Results and Generate Figures/Tables:**

**Produce actual figure files and/or LaTeX table code NOW.** Use `results-analysis` skill for statistical analysis.

Concrete actions:
1. **Load raw results** from 8b output files
2. **Run statistical tests** (significance, confidence intervals — see `results-analysis` skill)
3. **Generate figures** (save as PDF) and/or **generate LaTeX table code** based on data characteristics:

| Data Characteristic | Best Visualization | Tool |
|--------------------|-------------------|------|
| Trend / convergence over time | Line plot | matplotlib |
| Distribution / outliers | Box plot or violin plot | matplotlib/seaborn |
| Multi-objective tradeoff | Pareto front or scatter matrix | matplotlib |
| Ablation / component contribution | Bar chart or waterfall chart | matplotlib |
| Attention / feature importance | Heatmap | matplotlib/seaborn |
| High-dimensional embeddings | t-SNE / UMAP scatter | matplotlib |

**Figures vs tables:**
- **Figures**: Sparse data, trends/distributions, < 20 data points per comparison
- **Tables** (`booktabs` + `\resizebox`): Dense numerical results, many metrics (5+) and/or many baselines (5+), `table*` for large comparison matrices <!-- policy:TABLE.BOOKTABS_FORMAT -->

**Figure file rule: 1 file = 1 figure.** Do NOT use `plt.subplots()` to combine multiple plots into one image. Each individual plot must be saved as a separate file. Composite layouts (side-by-side comparison, multi-condition grids) are handled in LaTeX via `\subfigure` — not in Python. <!-- policy:FIG.ONE_FILE_ONE_FIGURE -->

```python
# CORRECT: one file per plot
fig, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.plot(...)
fig.savefig('imgs/runtime_batch32_ratio0p2.pdf', bbox_inches='tight')

# WRONG: do NOT combine subplots in Python
fig, axes = plt.subplots(1, 4, figsize=(16, 3))  # ← NEVER do this
```

**Shared legend**: If multiple sub-figures share the same legend, save the legend as a separate image file and include it above the subfigures in LaTeX (see subfigure template in Step 9).

**Figure quality**: Follow [figures4papers](https://github.com/ChenLiu-1996/figures4papers) — consistent style, **font size ≥ 24pt in source** (critical: smaller fonts become unreadable after scaling to column width), colorblind-safe palettes (Okabe-Ito or Paul Tol), line width ≥ 2.5pt, PDF vector format. See `results-analysis` skill for recommended `plt.rcParams` template. <!-- policy:FIG.FONT_GE_24PT --> <!-- policy:FIG.VECTOR_FORMAT_REQUIRED --> <!-- policy:FIG.COLORBLIND_SAFE_PALETTE -->

> **MANDATORY OUTPUT for Step 8 (all three required before proceeding to Step 9):**
> - [ ] Experiment plan document (confirmed by user in 8a)
> - [ ] Raw result files — CSV/JSON/logs (produced in 8b, or provided by user)
> - [ ] Figure PDF files AND/OR LaTeX table code (produced in 8c)
>
> **If any output is missing, STOP and address it. Do NOT write §5 Experiments without actual data.**

**Step 9: Write Experiments Section**

The Experiments section number depends on the paper (e.g., §5 or §6). The structure below is fixed regardless of section number.

**Section opening** (1-2 short paragraphs):
Summarize what experiments were conducted and why, before diving into details. This gives readers a roadmap of the experimental evaluation.

---

**\subsection{Experimental Settings}**

Use `\smallskip\noindent\textbf{}` fourth-level headings to organize (NOT `\subsubsection`):

```latex
\subsection{Experimental Settings}

\smallskip\noindent\textbf{Datasets.} We evaluate on three benchmarks: ...

\smallskip\noindent\textbf{Baselines.} We compare against five methods: ...

\smallskip\noindent\textbf{Metrics.} We report accuracy, F1 score, and ...

\smallskip\noindent\textbf{Implementation Details.} All experiments use PyTorch ...
We set the learning rate to ... Hyperparameter search ranges are in Appendix~\ref{app:hyperparams}.
All experiments run on [GPU type] for [total hours]. Seeds: ... <!-- policy:REPRO.COMPUTE_RESOURCES_DOCUMENTED --> <!-- policy:REPRO.RANDOM_SEED_DOCUMENTATION -->
```

---

**\subsection{Experimental Results}** <!-- policy:EXP.RESULTS_SUBSECTION_STRUCTURE -->

Use `\subsubsection` for each experiment group. **Every** `\subsubsection` MUST:
1. `\ref` to its corresponding figure or table (e.g., "as shown in Table~\ref{tab:main}" or "Fig.~\ref{fig:convergence} shows...")
2. End with a **Takeaway box** summarizing the key finding: <!-- policy:EXP.TAKEAWAY_BOX -->

```latex
\begin{center}
\fbox{%
  \begin{minipage}{0.9\linewidth}
  \textbf{Takeaway ([concise title]).} [1-2 sentence summary of the key finding
  from this experiment, written as a self-contained statement.]
  \end{minipage}
}
\end{center}
```

Example structure:

```latex
\subsection{Experimental Results}

\subsubsection{Main Performance Comparison}
Table~\ref{tab:main} presents ... [analysis of results] ...
\begin{center}
\fbox{\begin{minipage}{0.9\linewidth}
\textbf{Takeaway (state-of-the-art on all benchmarks).} Our method
outperforms all baselines on three datasets, with ...
\end{minipage}}
\end{center}

\subsubsection{Ablation Study}
To validate each component's contribution, we ... Table~\ref{tab:ablation} ...
\begin{center}
\fbox{\begin{minipage}{0.9\linewidth}
\textbf{Takeaway (both modules are necessary).} Removing either module
degrades performance by ...
\end{minipage}}
\end{center}

\subsubsection{Convergence Analysis}
Fig.~\ref{fig:convergence} shows ... [analysis] ...
\begin{center}
\fbox{\begin{minipage}{0.9\linewidth}
\textbf{Takeaway (fast convergence).} Our method converges in ...
\end{minipage}}
\end{center}
```

**Subfigure template** — when a `\subsubsection` needs multiple related plots (e.g., varying a parameter), use `\subfigure` with separate image files (1 file = 1 plot, never Python subplots):

```latex
\begin{figure*}[!t]
    \centering
    % Shared legend (if applicable) — saved as a separate image file
    \begin{minipage}[c]{1\textwidth}
      \centering
      \includegraphics[width=7in]{imgs/legend.pdf}
    \end{minipage}
    \vspace{-15pt}
    \\
    \subfigure[\shortstack{\small condition A}]{
    \begin{minipage}[t]{0.235\textwidth}
    \centering
    \includegraphics[width=1.8in]{imgs/result_condA.pdf}
    \end{minipage}
    \label{fig:result_condA}
    }
    \subfigure[\shortstack{\small condition B}]{
    \begin{minipage}[t]{0.235\textwidth}
    \centering
    \includegraphics[width=1.8in]{imgs/result_condB.pdf}
    \end{minipage}
    \label{fig:result_condB}
    }
    % Add more \subfigure blocks as needed
    \caption{Description of what the figure shows across conditions.
    The legend applies to all subgraphs.}
    \label{fig:result_comparison}
\end{figure*}
```

**Subfigure guidelines:**
- Use `figure*` (double-column) when ≥ 3 subfigures, `figure` (single-column) for 1-2
- Each subfigure has its own `\label` for individual `\ref` in text
- Shared legend as a separate image file at the top, with `\vspace{-15pt}` to reduce gap
- `\shortstack` in subfigure captions for multi-line condition labels

**Rules:**
- Quantitative tables use error bars (mean ± std, n runs) and complement the qualitative comparison table in Background & Related Work (same baselines, concrete numbers) <!-- policy:EXP.ERROR_BARS_REQUIRED -->
- **Ablation studies belong HERE** (as a `\subsubsection` of Results), NOT in Discussion <!-- policy:EXP.ABLATION_IN_RESULTS -->
- Each `\subsubsection` must have enough content (≥ 2 paragraphs + takeaway box) <!-- policy:EXP.TAKEAWAY_BOX -->

---

**\subsection{Discussion and Analysis}**

Opens with 1-2 paragraphs introducing what aspects need deeper analysis and why, then uses `\subsubsection` for each topic.

**Mathematical rigor requirement:** Discussion and Analysis must provide **reliable, theoretically grounded analysis** wherever possible. Do not make vague claims — back them up with mathematical expressions, formal arguments, or references to established theory. All notation MUST match the notation table (`Table~\ref{tab:notation}`).

```latex
\subsection{Discussion and Analysis}

[1-2 paragraphs: overview of key discussion points and motivation]

\subsubsection{Security Analysis}
% Formal security arguments: reduction proofs, game-based definitions,
% or structured threat coverage analysis with references to §3 threat model.
...

\subsubsection{Complexity Analysis}
% MUST provide Big-O complexity bounds for time and space.
% Use notation from Table~\ref{tab:notation}.
% Example:
% "The per-query cost is $\mathcal{O}(|\mathcal{P}| \cdot C_{\max}^2 \cdot T)$
%  where $|\mathcal{P}|$ is the number of eligible edges,
%  $C_{\max} = \max_\ell C_\ell$ is the largest channel width,
%  and $T$ is the probe set size."
% Compare with baseline complexity when applicable.
...

\subsubsection{Limitations}
Honestly report what does NOT work and why.
Explain why limitations do not undermine core claims.
...

\subsubsection{Real-World Applicability}
...
```

**Complexity Analysis guidelines:**
- Provide Big-O bounds for both **time** and **space** complexity
- Break down by algorithm phase if multi-stage (e.g., "Stage 1: $\mathcal{O}(n \log n)$, Stage 2: $\mathcal{O}(n^2 d)$")
- Compare against baseline methods' complexity in a mini table or inline comparison
- Use exact variable names from `Table~\ref{tab:notation}` — never introduce ad-hoc symbols
- If full analysis is lengthy, provide summary bounds in main text and defer proofs to Appendix with `\ref`

**What belongs here:** Security analysis, complexity analysis, limitations, real-world applicability, failure mode analysis, scalability discussion.

**What does NOT belong here:**
- Ablation studies → go in Experimental Results <!-- policy:EXP.ABLATION_IN_RESULTS -->
- Ethics considerations, open science, GenAI tools usage → go AFTER Conclusion (see Step 10)

---

**Page limit notes** (critical — some venues treat limitations specially): <!-- policy:SUBMIT.PAGE_LIMIT_STRICT -->
- **NeurIPS**: Limitations section does NOT count toward page limit (can be extracted as standalone after Conclusion) <!-- policy:ETHICS.LIMITATIONS_SECTION_MANDATORY -->
- **ICML**: Broader Impact Statement required after conclusion, outside page limit
- **ICLR**: Mandatory Limitations section (can stay as subsection or standalone) <!-- policy:ETHICS.LIMITATIONS_SECTION_MANDATORY -->
- **ACL**: Mandatory Limitations section, does NOT count toward page limit <!-- policy:ETHICS.LIMITATIONS_SECTION_MANDATORY -->

> If a venue **requires** a standalone Limitations section, extract it from Discussion and place after Conclusion. This does not add to the section count since it's typically unnumbered or outside the page limit.

**Step 10: Write Conclusion and Post-Conclusion Declarations**

**Conclusion (§6 or last numbered section):**

Structure:
- **Single paragraph only**: Include summary + key findings + future work in one cohesive paragraph <!-- policy:PAPER.CONCLUSION_SINGLE_PARAGRAPH -->

Guidelines:
- Do NOT introduce new information or results
- Do NOT repeat the abstract verbatim — rephrase at a higher level
- Keep concise: typically 0.5-0.75 column in two-column format
- End on a forward-looking note

**Post-Conclusion Declarations** (unnumbered, after Conclusion, before References):

Place the following as needed (check venue requirements):

```latex
\section*{Ethics Considerations}
[If applicable: IRB approval, data privacy, potential misuse, dual-use concerns]

\section*{Open Science Statement}
[Code/data availability, reproducibility artifacts, anonymized repo link]

\section*{Use of Generative AI Tools}
[Disclosure of LLM usage in writing/coding, per venue policy. E.g., ICLR requires this.]

\section*{Acknowledgments}
[Funding, collaborators — remove during anonymous review, restore for camera-ready]
```

These are NOT counted in the 6 numbered sections and typically do NOT count toward page limits.

**Step 11: Paper Checklist & Final Review**

NeurIPS, ICML, and ICLR all require paper checklists. See `references/checklists.md` for complete venue-specific checklists.

**Venue-specific critical items:**
- **NeurIPS**: 16-item mandatory checklist, Broader Impact section
- **ICML**: Broader Impact Statement (after conclusion, outside page limit)
- **ICLR**: LLM disclosure required if LLMs used in research process
- **ACL**: Responsible NLP Research checklist, mandatory Limitations section

**Final pass before submission:**
- Use `paper-self-review` skill for the latest multi-item quality checklist (including figure/title and LaTeX math conformance)
- Use `citation-verification` skill for reference validation
- Use `writing-anti-ai` skill if needed for natural voice
- Verify claim-evidence-figure alignment from Step 8a
- Check page limits, anonymization, supplementary materials <!-- policy:SUBMIT.PAGE_LIMIT_STRICT --> <!-- policy:ANON.DOUBLE_BLIND_ANONYMIZATION --> <!-- policy:SUBMIT.SECTION_NUMBERING_CONSISTENCY -->

---

## Writing Philosophy for Top ML Conferences

**This section distills the most important writing principles from leading ML researchers.** These aren't optional style suggestions—they're what separates accepted papers from rejected ones.

> "A paper is a short, rigorous, evidence-based technical story with a takeaway readers care about." — Neel Nanda

### The Sources Behind This Guidance

This skill synthesizes writing philosophy from researchers who have published extensively at top venues:

| Source | Key Contribution | Link |
|--------|-----------------|------|
| **Neel Nanda** (Google DeepMind) | The Narrative Principle, What/Why/So What framework | [How to Write ML Papers](https://www.alignmentforum.org/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml-papers) |
| **Sebastian Farquhar** (DeepMind) | 5-sentence abstract formula | [How to Write ML Papers](https://sebastianfarquhar.com/on-research/2024/11/04/how_to_write_ml_papers/) |
| **Gopen & Swan** | 7 principles of reader expectations | [Science of Scientific Writing](https://cseweb.ucsd.edu/~swanson/papers/science-of-writing.pdf) |
| **Zachary Lipton** | Word choice, eliminating hedging | [Heuristics for Scientific Writing](https://www.approximatelycorrect.com/2018/01/29/heuristics-technical-scientific-writing-machine-learning-perspective/) |
| **Jacob Steinhardt** (UC Berkeley) | Precision, consistent terminology | [Writing Tips](https://bounded-regret.ghost.io/) |
| **Ethan Perez** (Anthropic) | Micro-level clarity tips | [Easy Paper Writing Tips](https://ethanperez.net/easy-paper-writing-tips/) |
| **Andrej Karpathy** | Single contribution focus | Various lectures |

**For deeper dives into any of these, see:**
- [references/writing-guide.md](references/writing-guide.md) - Full explanations with examples
- [references/sources.md](references/sources.md) - Complete bibliography

### Time Allocation (From Neel Nanda)

Spend approximately **equal time** on each of:
1. The abstract
2. The introduction
3. The figures
4. Everything else combined

**Why?** Most reviewers form judgments before reaching your methods. Readers encounter your paper as: **title → abstract → introduction → figures → maybe the rest.**

### Writing Style Guidelines

#### Sentence-Level Clarity (Gopen & Swan's 7 Principles)

These principles are based on how readers actually process prose. Violating them forces readers to spend cognitive effort on structure rather than content.

| Principle | Rule | Example |
|-----------|------|---------|
| **Subject-verb proximity** | Keep subject and verb close | ❌ "The model, which was trained on..., achieves" → ✅ "The model achieves... after training on..." |
| **Stress position** | Place emphasis at sentence ends | ❌ "Accuracy improves by 15% when using attention" → ✅ "When using attention, accuracy improves by **15%**" |
| **Topic position** | Put context first, new info after | ✅ "Given these constraints, we propose..." |
| **Old before new** | Familiar info → unfamiliar info | Link backward, then introduce new |
| **One unit, one function** | Each paragraph makes one point | Split multi-point paragraphs |
| **Action in verb** | Use verbs, not nominalizations | ❌ "We performed an analysis" → ✅ "We analyzed" |
| **Context before new** | Set stage before presenting | Explain before showing equation |

**Full 7 principles with detailed examples:** See [references/writing-guide.md](references/writing-guide.md#the-7-principles-of-reader-expectations)

#### Micro-Level Tips (Ethan Perez)

These small changes accumulate into significantly clearer prose:

- **Minimize pronouns**: ❌ "This shows..." → ✅ "This result shows..."
- **Verbs early**: Position verbs near sentence start
- **Unfold apostrophes**: ❌ "X's Y" → ✅ "The Y of X" (when awkward)
- **Delete filler words**: "actually," "a bit," "very," "really," "basically," "quite," "essentially"

**Full micro-tips with examples:** See [references/writing-guide.md](references/writing-guide.md#micro-level-writing-tips)

#### Word Choice (Zachary Lipton)

- **Be specific**: ❌ "performance" → ✅ "accuracy" or "latency" (say what you mean)
- **Eliminate hedging**: Drop "may" and "can" unless genuinely uncertain
- **Avoid incremental vocabulary**: ❌ "combine," "modify," "expand" → ✅ "develop," "propose," "introduce"
- **Delete intensifiers**: ❌ "provides *very* tight approximation" → ✅ "provides tight approximation" <!-- policy:PROSE.INTENSIFIERS_ELIMINATION -->

#### Precision Over Brevity (Jacob Steinhardt)

- **Consistent terminology**: Different terms for same concept creates confusion. Pick one and stick with it.
- **State assumptions formally**: Before theorems, list all assumptions explicitly
- **Intuition + rigor**: Provide intuitive explanations alongside formal proofs

### What Reviewers Actually Read

Understanding reviewer behavior helps prioritize your effort:

| Paper Section | % Reviewers Who Read | Implication |
|---------------|---------------------|-------------|
| Abstract | 100% | Must be perfect |
| Introduction | 90%+ (skimmed) | Front-load contribution |
| Figures | Examined before methods | Figure 1 is critical |
| Methods | Only if interested | Don't bury the lede |
| Appendix | Rarely | Put only supplementary details |

**Bottom line**: If your abstract and intro don't hook reviewers, they may never read your brilliant methods section.

---

## Conference Requirements Quick Reference

| Conference | Page Limit | Extra for Camera-Ready | Key Requirement |
|------------|------------|------------------------|-----------------|
| **NeurIPS 2025** | 9 pages | +0 | Mandatory checklist, lay summary for accepted |
| **ICML 2026** | 8 pages | +1 | Broader Impact Statement required |
| **ICLR 2026** | 9 pages | +1 | LLM disclosure required, reciprocal reviewing |
| **ACL 2025** | 8 pages (long) | varies | Limitations section mandatory |
| **AAAI 2026** | 7 pages | +1 | Strict style file adherence |
| **COLM 2025** | 9 pages | +1 | Focus on language models |

**Universal Requirements:**
- Double-blind review (anonymize submissions) <!-- policy:ANON.DOUBLE_BLIND_ANONYMIZATION -->
- References don't count toward page limit <!-- policy:SUBMIT.PAGE_LIMIT_STRICT -->
- Appendices unlimited but reviewers not required to read
- LaTeX required for all venues

**LaTeX Templates:** See [templates/](templates/) directory for all conference templates.

---

## Using LaTeX Templates Properly

### Workflow 4: Starting a New Paper from Template

**Always copy the entire template directory first, then write within it.**

```
Template Setup Checklist:
- [ ] Step 1: Copy entire template directory to new project
- [ ] Step 2: Verify template compiles as-is (before any changes)
- [ ] Step 3: Read the template's example content to understand structure
- [ ] Step 4: Replace example content section by section
- [ ] Step 5: Keep template comments/examples as reference until done
- [ ] Step 6: Clean up template artifacts only at the end
```

**Step 1: Copy the Full Template**

```bash
# Create your paper directory with the complete template
cp -r templates/neurips2025/ ~/papers/my-new-paper/
cd ~/papers/my-new-paper/

# Verify structure is complete
ls -la
# Should see: main.tex, neurips.sty, Makefile, etc.
```

**⚠️ IMPORTANT**: Copy the ENTIRE directory, not just `main.tex`. Templates include:
- Style files (`.sty`) - required for compilation
- Bibliography styles (`.bst`) - required for references
- Example content - useful as reference
- Makefiles - for easy compilation

**Step 2: Verify Template Compiles First**

Before making ANY changes, compile the template as-is:

```bash
# Using latexmk (recommended)
latexmk -pdf main.tex

# Or manual compilation
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

If the unmodified template doesn't compile, fix that first. Common issues:
- Missing TeX packages → install via `tlmgr install <package>`
- Wrong TeX distribution → use TeX Live (recommended)

**Step 3: Keep Template Content as Reference**

Don't immediately delete all example content. Instead:

```latex
% KEEP template examples commented out as you write
% This shows you the expected format

% Template example (keep for reference):
% \begin{figure}[t]
%   \centering
%   \includegraphics[width=0.8\linewidth]{example-image}
%   \caption{Template shows caption style}
% \end{figure}

% Your actual figure:
\begin{figure}[t]
  \centering
  \includegraphics[width=0.8\linewidth]{your-figure.pdf}
  \caption{Your caption following the same style.}
\end{figure}
```

**Step 4: Replace Content Section by Section**

Work through the paper systematically:

```
Replacement Order:
1. Title and authors (anonymize for submission)
2. Abstract
3. Introduction
4. Methods
5. Experiments
6. Related Work
7. Conclusion
8. References (your .bib file)
9. Appendix
```

For each section:
1. Read the template's example content
2. Note any special formatting or macros used
3. Replace with your content following the same patterns
4. Compile frequently to catch errors early

**Step 5: Use Template Macros**

Templates often define useful macros. Check the preamble for:

```latex
% Common template macros to use:
\newcommand{\method}{YourMethodName}  % Consistent method naming
\newcommand{\eg}{e.g.,\xspace}        % Proper abbreviations
\newcommand{\ie}{i.e.,\xspace}
\newcommand{\etal}{\textit{et al.}\xspace}
```

**Step 6: Clean Up Only at the End**

Only remove template artifacts when paper is nearly complete:

```latex
% BEFORE SUBMISSION - remove these:
% - Commented-out template examples
% - Unused packages
% - Template's example figures/tables
% - Lorem ipsum or placeholder text

% KEEP these:
% - All style files (.sty)
% - Bibliography style (.bst)
% - Required packages from template
% - Any custom macros you're using
```

### Template Pitfalls to Avoid

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Copying only `main.tex` | Missing `.sty`, won't compile | Copy entire directory |
| Modifying `.sty` files | Breaks conference formatting | Never edit style files |
| Adding random packages | Conflicts, breaks template | Only add if necessary |
| Deleting template content too early | Lose formatting reference | Keep as comments until done |
| Not compiling frequently | Errors accumulate | Compile after each section |

### Quick Template Reference

| Conference | Main File | Key Style File | Notes |
|------------|-----------|----------------|-------|
| NeurIPS 2025 | `main.tex` | `neurips.sty` | Has Makefile |
| ICML 2026 | `example_paper.tex` | `icml2026.sty` | Includes algorithm packages |
| ICLR 2026 | `iclr2026_conference.tex` | `iclr2026_conference.sty` | Has math_commands.tex |
| ACL | `acl_latex.tex` | `acl.sty` | Strict formatting |
| AAAI 2026 | `aaai2026-unified-template.tex` | `aaai2026.sty` | Very strict compliance |
| COLM 2025 | `colm2025_conference.tex` | `colm2025_conference.sty` | Similar to ICLR |

---

## Conference Resubmission & Format Conversion

When a paper is rejected or withdrawn from one venue and resubmitted to another, format conversion is required. This is a common workflow in ML research.

### Workflow 3: Converting Between Conference Formats

```
Format Conversion Checklist:
- [ ] Step 1: Identify source and target template differences
- [ ] Step 2: Create new project with target template
- [ ] Step 3: Copy content sections (not preamble)
- [ ] Step 4: Adjust page limits and content
- [ ] Step 5: Update conference-specific requirements
- [ ] Step 6: Verify compilation and formatting
```

**Step 1: Key Template Differences**

| From → To | Page Change | Key Adjustments |
|-----------|-------------|-----------------|
| NeurIPS → ICML | 9 → 8 pages | Cut 1 page, add Broader Impact if missing |
| ICML → ICLR | 8 → 9 pages | Can expand experiments, add LLM disclosure |
| NeurIPS → ACL | 9 → 8 pages | Restructure for NLP conventions, add Limitations |
| ICLR → AAAI | 9 → 7 pages | Significant cuts needed, strict style adherence |
| Any → COLM | varies → 9 | Reframe for language model focus |

**Step 2: Content Migration (NOT Template Merge)**

**Never copy LaTeX preambles between templates.** Instead:

```bash
# 1. Start fresh with target template
cp -r templates/icml2026/ new_submission/

# 2. Copy ONLY content sections from old paper
# - Abstract text
# - Section content (between \section{} commands)
# - Figures and tables
# - Bibliography entries

# 3. Paste into target template structure
```

**Step 3: Adjusting for Page Limits**

When cutting pages (e.g., NeurIPS 9 → AAAI 7):
- Move detailed proofs to appendix
- Condense related work (cite surveys instead of individual papers)
- Combine similar experiments into unified tables
- Use `\subfigure` with smaller individual plots (1 file = 1 plot, composed in LaTeX)
- Tighten writing: eliminate redundancy, use active voice

When expanding (e.g., ICML 8 → ICLR 9):
- Add ablation studies reviewers requested
- Expand limitations discussion
- Include additional baselines
- Add qualitative examples

**Step 4: Conference-Specific Adjustments**

| Target Venue | Required Additions |
|--------------|-------------------|
| **ICML** | Broader Impact Statement (after conclusion) |
| **ICLR** | LLM usage disclosure, reciprocal reviewing agreement |
| **ACL/EMNLP** | Limitations section (mandatory), Ethics Statement |
| **AAAI** | Strict adherence to style file (no modifications) |
| **NeurIPS** | Paper checklist (appendix), lay summary if accepted |

**Step 5: Update References**

```latex
% Remove self-citations that reveal identity (for blind review)
% Update any "under review" citations to published versions
% Add new relevant work published since last submission
```

**Step 6: Addressing Previous Reviews**

When resubmitting after rejection:
- **Do** address reviewer concerns in the new version
- **Do** add experiments/clarifications reviewers requested
- **Don't** include a "changes from previous submission" section (blind review)
- **Don't** reference the previous submission or reviews

**Common Conversion Pitfalls:**
- ❌ Copying `\usepackage` commands (causes conflicts)
- ❌ Keeping old conference header/footer commands
- ❌ Forgetting to update `\bibliography{}` path
- ❌ Missing conference-specific required sections
- ❌ Exceeding page limit after format change

---

## Citation Workflow (Hallucination Prevention)

**⚠️ CRITICAL**: AI-generated citations have ~40% error rate. **Never write BibTeX from memory.** <!-- policy:CITE.VERIFY_VIA_API -->

### The Golden Rule

```
IF you cannot verify a citation through web search:
    → Mark it as [CITATION NEEDED] or [PLACEHOLDER - VERIFY]
    → Tell the scientist explicitly
    → NEVER invent a plausible-sounding reference
```

**MANDATORY**: Use WebSearch tool to verify EVERY citation before adding to bibliography.

### Workflow 2: Adding Citations

```
Citation Verification (MANDATORY for every citation):
- [ ] Step 1: Use WebSearch to find the paper
- [ ] Step 2: Verify paper exists on Google Scholar
- [ ] Step 3: Confirm paper details (title, authors, year, venue)
- [ ] Step 4: Retrieve BibTeX from Google Scholar or DOI
- [ ] Step 5: Verify the claim you're citing actually appears in the paper
- [ ] Step 6: Add verified BibTeX to bibliography
- [ ] Step 7: If ANY step fails → mark as placeholder, inform scientist
```

**Step 1: Use WebSearch to Find the Paper**

When you need to cite a paper, ALWAYS start with web search:

```
WebSearch query examples:
- "Attention is All You Need Vaswani 2017"
- "RLHF language model alignment 2023"
- "sparse autoencoders interpretability Anthropic"
- "transformer architecture NeurIPS"
```

**What to look for in search results:**
- Paper title matches your intended citation
- Authors are correct
- Publication year is correct
- Venue (conference/journal) is identified

**Step 2: Verify on Google Scholar**

After finding the paper, verify it exists on Google Scholar:

```
WebSearch query: "site:scholar.google.com [paper title] [first author]"

Example: "site:scholar.google.com Attention is All You Need Vaswani"
```

**Verification checklist:**
- ✅ Paper appears in Google Scholar results
- ✅ Title matches exactly (or very close)
- ✅ Authors match
- ✅ Year matches
- ✅ Venue is listed (conference/journal)
- ✅ Citation count is reasonable (not 0 for old papers)

**If paper NOT found on Google Scholar:**
- ❌ STOP - Do not cite
- Mark as `[CITATION NEEDED - not found on Google Scholar]`
- Inform scientist explicitly

**Step 3: Confirm Paper Details**

Before retrieving BibTeX, double-check all details:

```
Verification checklist:
- Title: [exact title from Google Scholar]
- Authors: [all authors, in order]
- Year: [publication year]
- Venue: [conference/journal name]
- DOI: [if available]
```

**Step 4: Retrieve BibTeX**

**Option 1: From Google Scholar (Recommended)**

1. Find the paper on Google Scholar
2. Click "Cite" button below the paper
3. Select "BibTeX" format
4. Copy the BibTeX entry

**Option 2: From DOI (if available)**

1. Use WebSearch to find: `"doi.org/[DOI]"`
2. Look for BibTeX export option on the publisher's page
3. Copy the BibTeX entry

**Option 3: From arXiv (for preprints)**

1. Find paper on arXiv
2. Click "Export BibTeX Citation" on the right sidebar
3. Copy the BibTeX entry

**CRITICAL**: Never write BibTeX from memory. Always copy from verified source.

**Step 5: Verify the Claim**

Before citing for a specific claim, verify the claim actually appears in the paper:

```
Verification process:
1. Use WebSearch to access the paper (PDF or HTML)
2. Search for keywords related to your claim
3. Confirm the claim is explicitly stated or clearly implied
4. Note the section/page where claim appears
```

**If you cannot access the paper:**
- ❌ Do not cite for specific claims
- Only cite for general contributions (if verified on Google Scholar)
- Mark as `[CLAIM NOT VERIFIED - no access to paper]`

**Step 6: Add Verified BibTeX to Bibliography**

Only after completing all verification steps:

```latex
% Add to your .bib file
@inproceedings{vaswani2017attention,
  title={Attention is All You Need},
  author={Vaswani, Ashish and Shazeer, Noam and ...},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
}

% Use in your paper
\cite{vaswani2017attention}
```

**Step 7: Handle Failures Explicitly**

If you cannot verify a citation at ANY step:

```latex
% Option 1: Explicit placeholder
\cite{PLACEHOLDER_smith2023_verify}  % TODO: Could not verify - scientist must confirm

% Option 2: Note in text
... as shown in prior work [CITATION NEEDED - could not verify Smith et al. 2023].
```

**Always inform the scientist:**
> "I could not verify the following citations and have marked them as placeholders:
> - Smith et al. 2023 on reward hacking - not found on Google Scholar
> - Jones 2022 on scaling laws - found similar paper but different authors
> Please verify these before submission."

### Summary: Citation Rules

| Situation | Action |
|-----------|--------|
| Found on Google Scholar, verified details, got BibTeX | ✅ Use the citation |
| Found paper, verified on Google Scholar, no BibTeX | ✅ Create BibTeX from Google Scholar info |
| Paper exists but details don't match | ⚠️ Mark placeholder, inform scientist |
| Not found on Google Scholar | ❌ Mark `[CITATION NEEDED]`, inform scientist |
| "I think there's a paper about X" | ❌ **NEVER cite** - search first or mark placeholder |

**🚨 NEVER generate BibTeX from memory—always verify through WebSearch and Google Scholar. 🚨**

### Complete Citation Workflow Example

**Scenario**: You need to cite the Transformer paper.

```
Step 1: WebSearch
Query: "Attention is All You Need Vaswani 2017"
Result: Found paper on multiple sources

Step 2: Google Scholar Verification
Query: "site:scholar.google.com Attention is All You Need Vaswani"
Result: ✅ Paper found, 50,000+ citations, NeurIPS 2017

Step 3: Confirm Details
- Title: "Attention is All You Need"
- Authors: Vaswani, Ashish; Shazeer, Noam; Parmar, Niki; ...
- Year: 2017
- Venue: NeurIPS (NIPS)
- DOI: Available

Step 4: Retrieve BibTeX
- Click "Cite" on Google Scholar
- Select BibTeX format
- Copy entry

Step 5: Verify Claim
- Access paper via WebSearch
- Confirm claim appears in paper
- Note section/page

Step 6: Add to Bibliography
- Paste BibTeX to .bib file
- Use \cite{vaswani2017attention} in paper

Step 7: Success
- Citation verified and added
- No placeholder needed
```

---

## Common Issues and Solutions

**Issue: Abstract too generic**

Delete first sentence if it could be prepended to any ML paper. Start with your specific contribution.

**Issue: Introduction exceeds 1.5 pages**

Split background into Related Work. Front-load contribution bullets. Methods should start by page 2-3.

**Issue: Experiments lack explicit claims**

Add sentence before each experiment: "This experiment tests whether [specific claim]..."

**Issue: Reviewers find paper hard to follow**

- Add explicit signposting: "In this section, we show X"
- Use consistent terminology throughout
- Include figure captions that stand alone

**Issue: Missing statistical significance**

Always include:
- Error bars (specify: std dev or std error) <!-- policy:EXP.ERROR_BARS_REQUIRED -->
- Number of runs
- Statistical tests if comparing methods

---

## Reviewer Evaluation Criteria

Reviewers assess papers on four dimensions:

| Criterion | What Reviewers Look For |
|-----------|------------------------|
| **Quality** | Technical soundness, well-supported claims |
| **Clarity** | Clear writing, reproducible by experts |
| **Significance** | Community impact, advances understanding |
| **Originality** | New insights (doesn't require new method) |

**Scoring (NeurIPS 6-point scale):**
- 6: Strong Accept - Groundbreaking, flawless
- 5: Accept - Technically solid, high impact
- 4: Borderline Accept - Solid, limited evaluation
- 3: Borderline Reject - Solid but weaknesses outweigh
- 2: Reject - Technical flaws
- 1: Strong Reject - Known results or ethics issues

See [references/reviewer-guidelines.md](references/reviewer-guidelines.md) for detailed reviewer instructions.

---

## Tables and Figures

### Tables

Use `booktabs` LaTeX package for professional tables. Always wrap tables in `\resizebox` to fit column/page width: <!-- policy:TABLE.BOOKTABS_FORMAT -->

```latex
\usepackage{booktabs}
\usepackage{graphicx} % for \resizebox

% Single-column table
\begin{table}[t]
\centering
\caption{Main results on benchmark X.}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcc}
\toprule
Method & Accuracy ↑ & Latency ↓ \\
\midrule
Baseline & 85.2 & 45ms \\
\textbf{Ours} & \textbf{92.1} & 38ms \\
\bottomrule
\end{tabular}%
}
\end{table}

% Double-column table (many metrics or baselines)
\begin{table*}[t]
\centering
\caption{Comprehensive comparison across all benchmarks.}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l*{8}{c}}
\toprule
...
\bottomrule
\end{tabular}%
}
\end{table*}
```

**Rules:**
- Bold best value per metric
- Include direction symbols (↑ higher is better, ↓ lower is better) <!-- policy:TABLE.DIRECTION_INDICATORS -->
- Right-align numerical columns
- Consistent decimal precision
- Always use `\resizebox{\columnwidth}{!}` (or `\textwidth` for `table*`)

**When to use tables vs figures:**
- **Tables**: Many metrics AND/OR many baselines, data is dense enough to justify structured grid layout, double-column (`table*`) when needed
- **Figures (Python plots)**: Fewer data points, showing trends/distributions/relationships, data needs spatial/visual encoding to convey meaning

### Figures

- **Vector graphics** (PDF, EPS) for all plots and diagrams <!-- policy:FIG.VECTOR_FORMAT_REQUIRED -->
- **Raster** (PNG 600 DPI) only for photographs
- Use **colorblind-safe palettes** (Okabe-Ito or Paul Tol) <!-- policy:FIG.COLORBLIND_SAFE_PALETTE -->
- Verify **grayscale readability** (8% of men have color vision deficiency)
- **No title inside figure**—the caption serves this function <!-- policy:FIG.NO_IN_FIGURE_TITLE -->
- **Self-contained captions**—reader should understand without main text <!-- policy:FIG.SELF_CONTAINED_CAPTION -->

**Figure type distinction:**
- **Data-driven plots** (bar charts, line plots, heatmaps, scatter plots): use `results-analysis` skill with matplotlib/seaborn
- **Conceptual diagrams** (system overviews, pipelines, architectures, threat models, comparisons): use `paper-figure-generator` skill with AutoFigure-Edit (outputs editable SVG)

---

## References & Resources

### Reference Documents (Deep Dives)

| Document | Contents |
|----------|----------|
| [writing-guide.md](references/writing-guide.md) | Gopen & Swan 7 principles, Ethan Perez micro-tips, word choice |
| [citation-workflow.md](references/citation-workflow.md) | Citation APIs, Python code, BibTeX management |
| [checklists.md](references/checklists.md) | NeurIPS 16-item, ICML, ICLR, ACL requirements |
| [reviewer-guidelines.md](references/reviewer-guidelines.md) | Evaluation criteria, scoring, rebuttals |
| [sources.md](references/sources.md) | Complete bibliography of all sources |
| [latex-style-guide.md](references/latex-style-guide.md) | LaTeX 格式规范：四级标题、packeditemize、hyperref、对比表格模板 |
| **Literature Research:** |
| [arxiv-search-guide.md](references/literature-research/arxiv-search-guide.md) | arXiv search strategies, URL patterns, Chrome MCP automation |
| [paper-quality-criteria.md](references/literature-research/paper-quality-criteria.md) | 5-dimension paper evaluation rubrics (innovation, method, experiments, writing, impact) |

### LaTeX Templates

Templates in `templates/` directory: **ICML 2026**, **ICLR 2026**, **NeurIPS 2025**, **ACL/EMNLP**, **AAAI 2026**, **COLM 2025**.

**Compiling to PDF:**
- **VS Code/Cursor**: Install LaTeX Workshop extension + TeX Live → Save to auto-compile
- **Command line**: `latexmk -pdf main.tex` or `pdflatex` + `bibtex` workflow
- **Online**: Upload to [Overleaf](https://overleaf.com)

See [templates/README.md](templates/README.md) for detailed setup instructions.

### Key External Sources

**Writing Philosophy:**
- [Neel Nanda: How to Write ML Papers](https://www.alignmentforum.org/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml-papers) - Narrative, "What/Why/So What"
- [Farquhar: How to Write ML Papers](https://sebastianfarquhar.com/on-research/2024/11/04/how_to_write_ml_papers/) - 5-sentence abstract
- [Gopen & Swan: Science of Scientific Writing](https://cseweb.ucsd.edu/~swanson/papers/science-of-writing.pdf) - 7 reader expectation principles
- [Lipton: Heuristics for Scientific Writing](https://www.approximatelycorrect.com/2018/01/29/heuristics-technical-scientific-writing-machine-learning-perspective/) - Word choice
- [Perez: Easy Paper Writing Tips](https://ethanperez.net/easy-paper-writing-tips/) - Micro-level clarity

**APIs:** [Semantic Scholar](https://api.semanticscholar.org/api-docs/) | [CrossRef](https://www.crossref.org/documentation/retrieve-metadata/rest-api/) | [arXiv](https://info.arxiv.org/help/api/basics.html)

**Venues:** [NeurIPS](https://neurips.cc/Conferences/2025/PaperInformation/StyleFiles) | [ICML](https://icml.cc/Conferences/2025/AuthorInstructions) | [ICLR](https://iclr.cc/Conferences/2026/AuthorGuide) | [ACL](https://github.com/acl-org/acl-style-files)
