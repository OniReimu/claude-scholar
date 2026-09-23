# Claude Scholar

<div align="center">
  <img src="LOGO.jpeg" alt="Claude Scholar Logo" width="100%"/>
</div>

**Language**: [English](README.md) | [中文](README.zh-CN.md)

Personal Claude Code configuration repository, optimized for academic research and software development - a complete working environment.

## News

- **2026-09-24 (v1.33.1)**: The orphan-sentence regex added in v1.32.3 no longer flags short sentences anchored by a demonstrative ("assume that layout") or opened by a defined label ("RQ2 asks …"). The policy CI is green again.
- **2026-09-24 (v1.33.0)**: New rule `PAPER.DEFINE_BEFORE_USE`: in reading order, a term must be defined or glossed at or before its first use. A definition three sections later does not count, and a section pointer is not a gloss. `claim-architecture-review` now catches this in its P1 sweep, because line edits see one paragraph and cannot notice a definition that sits sections away.
- **2026-09-21 (v1.32.3)**: `PROSE.RHYTHM_VARIANCE` treats variance as a constraint, not a target. The cheapest way to reach sd ≥ 10, dropping in a 6–8 word sentence, is now the card's most common violation: a short sentence must be a checkable claim or an anchored pivot and must pass the deletion test, and the distribution may be widened only by re-splitting existing information.
- **2026-09-21 (v1.32.2)**: `PROSE.FRACTAL_SUMMARY` now measures the turn-back distance from a reviewer who enters a section cold, not from someone who just finished the previous one, so a top-level section's only roadmap stays when it says what the parts do. `PROSE.RESTATEMENT_DILUTION` stops treating an expansion as a restatement.
- **2026-09-19 (v1.32.1)**: Skill validation now runs in CI, which it never had — `validate-skills.sh` existed but no workflow called it. Its first run found a `SKILL.md` whose frontmatter was not valid YAML and a contributor's home directory committed across three files.
- **2026-09-19 (v1.32.0)**: `research-workspace-layout` — adopting the [research-workspace contract](https://github.com/DELONG-L/Research-Workflow-Skills) in a repository, and every way attaching Overleaf as a submodule goes wrong. It treats the topology as a **closed set**: eleven parallel cache directories appear not because anyone chose eleven, but because nothing said where the twelfth would have to go.
- **2026-09-14 (v1.31.0)**: `peer-review` — a routing skill for the one review Claude Scholar had no skill for: reviewing someone else's submission for a venue. It carries no review logic, handing off to the separately installed [More Than Peer Review](https://github.com/DELONG-L/More-Than-Peer-Review-Skill) **before the manuscript is opened**, so no criticism formed on this side crosses over.
- **2026-09-11 (v1.30.0)**: `PROSE.CAUSAL_CONNECTIVE` gets the instrument its own argument requires. The card had proved the effect lives entirely in **density** (0.18–0.28 per 1000 words in pre-GPT corpora against 3.76 in a measured draft), yet lint counted five formal connectives and never counted `, so` at all — so lint now prints both as per-1000-word rates against the baseline, and the scan order is explicit: read the density first, then judge instance by instance.
- **2026-09-01 (v1.28.0)**: The polish loop gets an exit that is a verdict instead of a count. `PAPER.REVISION_CLOSURE` ends self-review with one of four calls on the whole manuscript (**STOP_REVISING**, **ONE_BOUNDED_ROUND**, **REOPEN_SUBSTANTIVE_REVISION**, **UNASSESSED**), because *stop at zero findings* prices at zero the one cost this repository has measured: every repair round settles new statistical signature into the draft.
- **2026-08-28 (v1.27.0)**: Connective monoculture gets a structural fix after recurring in two projects. The three punctuation-clearing rules push the relation into the lexical layer where `CAUSAL_CONNECTIVE` hands over its safest item, one or two per pass, each locally defensible — 22 of 26 formal connectives ending up `therefore` with every per-instance check green — so lint now prints a report-only distribution table and the remedy ladder puts *delete* first.
- **2026-08-28 (v1.26.0)**: `PROSE.OVER_DEFENSIVE` gains a **sentence layer** for disclaimer-style negative predicates, which its five position-and-count clauses all passed while sixteen of them added up to a manuscript filing disclaimers at 1.7 per 1000 words. The remedy is one move rather than deletion — flip *we do not do X* into *what we do is Y* — and wiring it exposed an engine defect: lint ran patterns only for `check_kind: regex`, so a judgment card carrying a locator validated clean and never executed.
- **2026-08-28 (v1.25.1)**: The reversed copular order — `is not A, but B` — sat in v1.25's tier-A table and was covered by no pattern, because the original rule anchored on `, not ` which only exists in positive-first order. A sixth pattern now covers it, and writing the corpus case found that `not only … but also` is uncovered *deliberately* (it is frequency-managed, so one instance is not a violation), which is now a `coverage_note` rather than an accident.
- **2026-08-27 (v1.25.0)**: `PROSE.NEGATION_CONTRAST` splits into zero-tolerance and advisory tiers, after a manuscript reported **15** contrast constructions where **38** existed. Probing every `lint_script` rule against the forms its own Requirement enumerates found the same narrowing in three more cards — all of which had *declared* it in a Check section no executor opens — so a `coverage_note` field now states the gap where lint prints it, enforced in both directions by `validate.sh`.
- **2026-08-27 (v1.24.0)**: `PROSE.SEMICOLON_RESTRICTION` gains a second remedy, because the first manufactured violations of a rule whose own fix undid it — across 39 measurable semicolons, "split into two sentences" left a second clause under ten words in 19 of them, which is exactly the shape `PROSE.THEATRICAL_SPLIT` bans and tells you to merge back. The remedy is now two-tier, syntactic change before split, with tier ① gated on four conditions at once so it stays the minority case (32 of 39 still split).
- **2026-08-27 (v1.23.0)**: Re-anchoring gets a distance dimension: restating a construct defined several sections earlier is ordinary writing, not the self-similar redundancy `PROSE.FRACTAL_SUMMARY` exists to catch, yet the card had no notion of distance and judged a paragraph two lines above its heading identically to one three sections downstream. The fix landed in `claim-architecture-review` P2 first, where the ledger would otherwise collapse every cross-section re-anchor, and ships **no numeric threshold** — the question is whether the reader would have to turn back, and every threshold-on-judgment here has ended up measuring the instrument.
- **2026-08-27 (v1.22.0)**: Three fixes from reading the pipeline's own "clean" output critically. `PROSE.SENTENCE_LENGTH` was silently under-firing because any dot-bearing token acted as a sentence boundary and decimals live in exactly the sentences it exists to catch; new rule `PROSE.SEMICOLON_RESTRICTION` ships as a builtin since a plain pattern fires on inline math; and `PROSE.EM_DASH_RESTRICTION`'s remedy was wrong, so both cards now share one criterion — swapping punctuation is not a fix, the structure must change.
- **2026-08-27 (v1.21.0)**: New suite `policy/test-pipeline.sh` — the first test of what the rules do *together*, scoring a planted-defect fixture with three assertion verbs plus a check that every number in the output already existed in the input. It also asserts that the **unedited draft fails**, and that negative control immediately earned its place: two assertions labelled as catching an ordering trap turned out to pass on a line-only edit, because labels drift toward what you meant rather than what they test.
- **2026-08-27 (v1.20.0)**: `PROSE.SEMANTIC_IDLING` gains a **Rewrite contract**, after blind-judging a supplied specification's own ten gold rewrites: **6 of 10 still violate**, and two are the segments it had diagnosed correctly as circular then rewritten circularly. So a rewrite is emitted only when a proposition survives — compression purifies a paragraph with content and merely shortens one without, and short emptiness reads like a conclusion — and the 75–85% ratio is recorded as an observation, never a target.
- **2026-08-27 (v1.19.1)**: `PROSE.SEMANTIC_IDLING` acceptance eval — ten supplied wheel-spinning paragraphs blind-mixed with ten unseen pre-GPT published ones so the judge could not infer the base rate: **10/10 flagged, 0/10 false positives**. Two corrections followed: Future Work boilerplate is explicitly not exempt, and form B is recorded as over-firing on standard mechanisms, so the test is now a question — does the explanans introduce an independently measurable quantity?
- **2026-08-27 (v1.19.0)**: New rule `PROSE.SEMANTIC_IDLING` — every sentence must add a falsifiable proposition. "Wheel-spinning" turned out to be three phenomena, two of which already had owners, so this card takes only the two that fell through every other: the long meta-narrative sentence that names nothing, and circular attribution whose `because` clause restates its own consequent. It **forbids metric proxies** on this repository's own evidence that a judgment behind a threshold measures the instrument.
- **2026-08-27 (v1.18.0)**: `PROSE.ADHOC_COMPOUND_MODIFIER` becomes a three-way verdict — **flag** / **hint** / **clear** — because a binary one was wrong in both directions at once on the same paragraph set. Cleared items are now reported too, since reporting only violations leaves the author unable to tell whether a term was checked and passed or never seen; paragraph length was investigated and **rejected** as a tell.
- **2026-08-27 (v1.17.0)**: Sharpened `PROSE.ADHOC_COMPOUND_MODIFIER`, where one addition mattered more than the rest: **`-based` is a construction, not a coinage**, and including it dilutes the era separation from 16x to 4x, so it moves outside the default suffix set. The card also states plainly that **the 16x is not quotable as an effect size**, and clearing a compound as established now requires naming its prior source — a claim a reader can check.
- **2026-08-27 (v1.16.1)**: Disclosed the limitation inside `PROSE.ADHOC_COMPOUND_MODIFIER` rather than leaving it implied: the mechanical half returns the same number whoever runs it, but deciding whether a compound is an established field term is an LLM judgement, bounded by a knowledge cutoff and weaker in niche subfields. The card carries a self-caught misjudgement as evidence and sets the direction conservatively — when unsure, do not flag, because a false positive on a real field term costs the whole rule.
- **2026-08-27 (v1.16.0)**: New rule `PROSE.ADHOC_COMPOUND_MODIFIER` — the only signal from a day of measurement that survived scrutiny. **Frequency, not construction, is the test**: compounds used exactly once run 0.16 per 1000 words in 2019–2021 against 0.48 in 2025–2026, while a pre-GPT blockchain manuscript carries them at 7x baseline through repeated use, so scoring by volume would condemn an entire field. What makes it credible is that it needs no judgment — the two candidates measured the same day did, and one collapsed from 16x to 1.22x under independent blind adjudication.
- **2026-08-27 (v1.15.1)**: Closed the five `doc`-enforced rules that `writing-anti-ai` declared in a table and never executed, and made that class of gap machine-detected — a `lint_script` rule has a regex backstop whatever the skill body says, a `doc` rule has none. Writing the check surfaced something worse: `set -eo pipefail` let one legitimately-empty `grep` end the run mid-section, and a truncated run prints **fewer** `FAIL:` lines than a complete one, so CI had been reading the early exit as an improvement.
- **2026-08-27 (v1.15.0)**: New rule `PROSE.CAUSAL_CONNECTIVE`, and an evaluation that changed what the rule says. 42 flagged sentences adjudicated blind showed the locator perfect (42/42) but **instance-level discrimination nil and then inverted** — pre-GPT instances judged worth changing at 64% against 29% for current drafts — so only the density differs, and the rule ships as a causal-precision rule rather than an AI tell, flagging three diagnosable classes and keeping everything else.
- **2026-08-27 (v1.14.1)**: CI had been red since 2026-08-17 on three FAILs that only ever appear on the runner: `deprecated_by` resolvability tested the successor with `-e`, which follows symlinks, and three skills are symlinks into vendor submodules. The tree was missing content, not the reference — `-L` now answers the same either way, and the workflow checks out submodules.
- **2026-08-26 (v1.14.0)**: Two suites that measure what the existing ones never asked — `policy/test-corpus.sh` runs 88 annotated fixtures and reports recall misses and false positives per case, and `policy/test-referrals.sh` machine-checks the referral graph that had been hand-verified until now. Writing them surfaced six defects, including byte-wise character classes where `[→←↔]` matched every em dash on the perl path macOS resolves to, a YAML comment silently truncating a `lint_patterns` block, and commented-out prose being linted.
- **2026-08-26 (v1.13.4)**: Made the referral path work for the workflow people actually use — on an existing manuscript the entry point is `writing-anti-ai`, and everything structural reaches the author only if that skill refers successfully, which it did not. Both sides now state the minimum operation rather than a bare skill name, and `claim-architecture-review` went from 1 marker to 6 on the principle that **a skill carries a rule's marker when it executes that rule**, not when it is merely related.
- **2026-08-26 (v1.13.3)**: Closed the cross-section seam v1.13.2 left open — the rule scoped itself to paragraph and section while the information ledger was keyed on propositions and mentioned enumeration nowhere, so both sides assumed the other had it. Separately, `validate.sh` was forking `grep` ~1400 times per run and an occasional failed fork was indistinguishable from a missing field: **a validator that intermittently fails on correct input teaches its users to re-run until green, which is worse than not checking.**
- **2026-08-24 (v1.13.2)**: `PROSE.RULE_OF_THREE` widened from triads to **enumeration density and repetition**, after a real methods paragraph where only one of three lists was a triad and the sole mechanical hit reported commas rather than enumeration. `writing-anti-ai` had also been instructing *"prefer two or four items"* — inherited from general anti-AI advice, but in academic prose that **produces** longer enumeration walls; the reverse guard now leads the card: the fix is name-then-refer, never dropping items.
- **2026-08-23 (v1.13.1)**: Routed three line-editable pieces of the narrative-agency principles into `writing-anti-ai`, where a draft scan can actually fix them — `PAPER.OUTCOME_LOGIC` had been wired at write time and review time, so chronology leaks were *found* with no step to *fix* them. The boundary that matters most ships with it: only implementation detours are deleted, since an experiment whose result bounds the claim is evidence; principles needing whole-paper context were deliberately **not** routed, because a line editor judging them is the documented over-execution failure mode.
- **2026-08-23 (v1.13.0)**: Narrative-agency layer, from a peer's diagnosis that an agent fearing criticism pre-emptively attacks the paper on behalf of an imagined reviewer. Three new rules (`PROSE.SELF_UNDERMINING`, `EXP.EXPERIMENT_ROLE`, `PAPER.OUTCOME_LOGIC`), each shipping an **integrity boundary** so none reads as licence to hide evidence — wording is governed but disclosure never reduced, an experiment is dropped for serving no claim and never for an unfavourable number. The audit also found `policy/style-guide.md` contradicting the rules it is declared co-equal with, so `validate.sh` now lints the style guide's own prose.
- **2026-08-17 (v1.12.0)**: Register coverage for **author-original** prose, from an NDSS submission that had already passed a `writing-anti-ai` run and still carried ~30 register defects — feeding 26 of them to `PROSE.INFORMAL_VOCABULARY`'s nine patterns matched **zero**, because 29 of 30 were multi-word constructions against a card shipping single-word regexes. It is rebuilt as a five-class taxonomy with LLM judgement and allowlists, joined by `PROSE.IDIOM_COLLISION` and a reference on the four measured ways a hand-rolled `.tex` scanner produces a false "clean" verdict.
- **2026-08-15 (v1.11.0)**: Two hardenings from real submission sweeps. `PROSE.NO_INTERNAL_PROVENANCE` existed and still let eleven development artifacts reach a compiled ACM ASIA CCS submission — three holes, not a missing rule: absent from the checklist agents actually read, zero lint coverage, and `severity: warn` contradicting its own rationale. And new `PROSE.REGISTER_PRESERVATION`, which enforces on the **diff** because register is a property of the edit rather than the word: one compression pass introduced nine violations that the existing patterns caught zero of.
- **2026-08-15 (v1.10.0)**: Anti-AI writing overhaul, a policy conflict audit, and the first evidence-backed eval run (39/39 against 37/39 for the pre-overhaul snapshot, both deltas landing exactly on the new capabilities). Four new rules ship alongside fixes for a self-contradictory ban, an autofix repair loop and a `SENTENCE_LENGTH` pattern that counted sentences per file rather than words per sentence, and `validate.sh` now machine-checks that **no autofix output may trigger another rule**. Borrowed with review from [zksecurity/zk-skills](https://github.com/zksecurity/zk-skills) (`circom-auditor`) and [AIScientists-Dev/academic-humanizer](https://github.com/AIScientists-Dev/academic-humanizer).
- **2026-07-09 (v1.7.1)**: Post-release review fixes for v1.7.0 — venue-split figure sizing so hard-spec venues override the default `FigureStyle` font size instead of silently conflicting with it, stale `FIG.FONT_GE_24PT` references cleaned up, `TABLE.DIMENSION_BUDGET`'s example de-domained and its off-by-one reconciled, and reverse pointers into the `knows-literature` bridge so it fires on direct invocation too.
- **2026-07-09 (v1.7.0)**: Typesetting-constraint pack, skill-routing upgrade and a Knows bridge refresh, borrowed with review from [DELONG-L/Academic-Paper-Skills](https://github.com/DELONG-L/Academic-Paper-Skills) — three new table and provenance rules, a figure visual-QA closed loop that renders and actually *reads* the PNG before any vector export, and rewritten descriptions for five high-frequency writing skills. Two things were deliberately **not** borrowed: a header-arrow ban that conflicts with `TABLE.DIRECTION_INDICATORS`, and a local-`.bib`-only citation policy weaker than `CITE.VERIFY_VIA_API`.
- **2026-07-08 (v1.6.2)**: New rule `EXP.MULTIRUN_AGGREGATE_CONSISTENCY` — multi-run tables and figures must source from a machine-generated aggregate carrying per-run validity, a cross-run consistency verdict and provenance, with hand-transcribing numbers forbidden. It complements `CITE.CLAIM_SUPPORT_REQUIRED`'s transcription fidelity with **source trustworthiness**: that the runs are real, mutually comparable and traceable.
- **2026-07-02 (v1.6.1)**: `paper-figure-generator` now forces **non-italic sans-serif** fonts in generated SVGs, injecting an `!important` `<style>` at both write points to fix the default italic Times New Roman output.
- **2026-06-25 (v1.6.0)**: Added the `architecture_review` stage and the `claim-architecture-review` skill — a post-draft **structural edit** (paragraph necessity and placement, cross-section redundancy, claim spine and story closure) that runs before self-review and anti-AI polish. It is propose-only, with the `rewrite` stage applying approved moves; the pipeline is now 12 stages.
- **2026-03-02 (v1.4.1)**: Added the Workflow Orchestrator — a stateful, resumable run-coordination layer with persistent state, SHA256 artifact fingerprinting, auto-stale detection, rollback with downstream cascade, and stage gates. Zero new commands: it activates transparently through existing skills, agents and hooks.
- **2026-02-21**: Added the first SoK policy pack — four semantic `SOK.*` rule cards, the `security-sok-sp` profile, and entry-skill marker wiring. SoK remains profile-activated scope in v1.
- **2026-02-19 (v1.3.0)**: Introduced the paper policy engine (`policy/`): rule-card based design in `policy/rules/` (single source of truth), layered scope (`core/domain/venue`), profile overlays in `policy/profiles/`, and executable validation/lint workflows via `policy/validate.sh` and `policy/lint.sh`. Synced Figure workflow policy (Figure 1 required; non-experimental figures default to AutoFigure-Edit).
- **2026-02-16 (v1.2.1)**: Added a global figure rule: no in-image titles for any generated visuals (AutoFigure-Edit conceptual diagrams, legacy image APIs, or Python experimental plots). Use captions in paper text/LaTeX instead.
- **2026-02-16**: Enforced `paper-figure-generator` execution priority: default `AutoFigure-Edit + OpenRouter` first, fallback to legacy Gemini/OpenAI flow only after failure; added troubleshooting note for outdated plugin cache prompts (`GOOGLE_API_KEY` / `OPENAI_API_KEY`).
- **2026-02-15**: Migrated `paper-figure-generator` to AutoFigure-Edit — generates editable SVG vector figures from method text descriptions; replaces Gemini/OpenAI raster generation; supports style transfer via reference images; uses OpenRouter + Roboflow (free SAM3 API)
- **2026-02-13**: Added `paper-figure-generator` skill; packaged project as Claude Code plugin (`.claude-plugin/plugin.json`); added `.env.example`; deep workflow integration across ml-paper-writing, results-analysis, post-acceptance, and using-claude-scholar; 34 skills total
- **2026-02-11**: Major update — added 10 new skills (research-ideation, results-analysis, citation-verification, review-response, paper-self-review, post-acceptance, daily-coding, frontend-design, ui-ux-pro-max, web-design-reviewer), 7 new agents, 8 research workflow commands, 2 new rules (security, experiment-reproducibility); restructured CLAUDE.md; 89 files changed
- **2026-01-26**: Rewrote all Hooks to cross-platform Node.js; completely rewrote README; expanded ML paper writing knowledge base; merged PR #1 (cross-platform support)

## Introduction

Claude Scholar is a personal configuration system for Claude Code CLI, providing rich skills, commands, agents, and hooks optimized for:
- **Academic Research** - Complete research lifecycle: idea generation → experimentation → results analysis → paper writing → review response → conference preparation
- **Software Development** - Git workflows, code review, test-driven development, ML project architecture
- **Plugin Development** - Skill, Command, Agent, Hook development guides with quality assessment
- **Project Management** - Planning documents, code standards, automated workflows with cross-platform hooks

## Quick Navigation

| Topic | Description |
|-------|-------------|
| 🚀 [Quick Start](#quick-start) | Get up and running in minutes |
| 📚 [Core Workflows](#core-workflows) | Paper writing, code organization, skill evolution |
| 🛠️ [What's Included](#whats-included) | Skills, commands, agents overview |
| 📖 [Installation Guide](#installation-options) | Full, minimal, or selective setup |
| 🔧 [Project Rules](#project-rules) | Coding rules + paper policy engine |

## Core Workflows

### Primary Workflows

Complete academic research lifecycle - 7 stages from idea to publication.

#### 1. Research Ideation

Systematic research startup with idea generation and literature review:

**Tools**: `research-ideation` skill + `literature-reviewer` agent

**Process**:
- **5W1H Brainstorming**: What, Why, Who, When, Where, How → structured thinking framework
- **Literature Review**: arXiv + Semantic Scholar integration → automated paper search and classification
- **Gap Analysis**: 5 types (Literature, Methodological, Application, Interdisciplinary, Temporal) → identify research opportunities
- **Research Question**: SMART principles → formulate specific, measurable questions

**Command**: `/research-init "topic"` → launches complete research startup workflow

#### 2. ML Project Development

Maintainable ML project structure for experiment code:

**Tools**: `architecture-design` skill + `code-reviewer` agent + `git-workflow` skill

**Process**:
- **Structure**: Factory & Registry patterns → config-driven models (only `cfg` parameter) → enforced by `rules/coding-style.md`
- **Code Style**: 200-400 line files → type hints required → `@dataclass(frozen=True)` for configs → max 3-level nesting
- **Debug** (`bug-detective`): Error pattern matching for Python/Bash/JS → stack trace analysis → anti-pattern identification
- **Git**: Conventional Commits (`feat/scope: message`) → branch strategy (master/develop/feature) → merge with `--no-ff`

**Commands**: `/plan`, `/commit`, `/code-review`, `/tdd`

#### 3. Experiment Analysis

Statistical analysis and visualization of experimental results:

**Tools**: `results-analysis` skill + `data-analyst` agent

**Process**:
- **Data Processing**: Automated cleaning and preprocessing of experiment logs
- **Statistical Testing**: t-test, ANOVA, Wilcoxon signed-rank → validate significance
- **Visualization**: matplotlib/seaborn integration → publication-ready figures (line plots, bar charts, heatmaps)
- **Ablation Studies**: Systematic component analysis → understand contribution of each part

**Command**: `/analyze-results <experiment_dir>` → generates analysis report with figures and statistics

#### 4. Paper Writing

Systematic paper writing from template to final draft:

**Tools**: `ml-paper-writing` skill + `paper-miner` agent + `latex-conference-template-organizer` skill

**Process**:
- **Template Preparation**: Download conference .zip → extract main files → remove sample content → clean Overleaf-ready structure
- **Citation Verification** (`citation-verification`): Multi-layer validation (Format → API → Information → Content) → prevents hallucinations
- **Systematic Writing**: Narrative framing → 5-sentence abstract formula → section-by-section drafting with feedback cycles
- **Anti-AI Processing** (`writing-anti-ai`): Remove inflated symbolism, promotional language, vague attributions → add human voice and rhythm → bilingual support (EN/CN)

**Venues**: NeurIPS, ICML, ICLR, ACL, AAAI, COLM, Nature, Science, Cell, PNAS

#### 5. Paper Self-Review

Quality assurance before submission:

**Tools**: `paper-self-review` skill

**Process**:
- **Structure Check**: Logical flow, section balance, narrative coherence
- **Logic Validation**: Argument soundness, claim-evidence alignment, assumption clarity
- **Citation Audit**: Reference accuracy, proper attribution, citation completeness
- **Figure Quality**: Visual clarity, caption completeness, color accessibility
- **Writing Polish**: Grammar, clarity, conciseness, academic tone
- **Compliance**: Page limits, formatting requirements, ethical disclosures

**Multi-item checklist** → systematic quality assessment (including figure/title and LaTeX math conformance)

#### 6. Submission & Rebuttal

Paper submission and review response:

**Tools**: `review-response` skill + `rebuttal-writer` agent

**Submission Process**:
- **Pre-submission**: Conference-specific checklists (NeurIPS 16-item, ICML Broader Impact, ICLR LLM disclosure)
- **Format Check**: Page limits, anonymization, supplementary materials
- **Final Review**: Proofread, check references, verify figures

**Rebuttal Process**:
- **Review Analysis**: Parse and classify comments (Major/Minor/Typo/Misunderstanding)
- **Response Strategy**: Accept/Defend/Clarify/Experiment → tailored approach per comment type
- **Rebuttal Writing**: Structured response with evidence and reasoning
- **Tone Management**: Professional, respectful, evidence-based language

**Command**: `/rebuttal <review_file>` → generates complete rebuttal document with experiment plan

#### 7. Post-Acceptance Processing

Conference preparation and research promotion:

**Tools**: `post-acceptance` skill

**Process**:
- **Presentation**: Slide creation guidance (15/20/30 min formats) → visual design principles → storytelling structure
- **Poster**: Academic poster templates (A0/A1 sizes) → layout optimization → visual hierarchy
- **Promotion**: Social media content (Twitter/X, LinkedIn) → blog posts → press releases → research summaries

**Commands**: `/presentation`, `/poster`, `/promote` → automated content generation

**Coverage**: 90% of academic research lifecycle (from idea to publication)

### Workflow Orchestrator

Claude Scholar includes a stateful **Workflow Orchestrator** that tracks progress across the research lifecycle as a single, resumable run. No new commands are needed -- the orchestrator activates transparently when relevant skills and agents are invoked.

**Key features:**
- **Single mode, resumable runs**: State persists in `.claude/orchestrator/` across sessions. Resume from where you left off.
- **12-stage pipeline**: intake -> literature -> proposal -> development -> experiments -> analysis -> writeup -> architecture_review -> self_review -> rewrite -> rebuttal -> post_acceptance
- **Stage gates**: Human approval and policy lint checks at stage boundaries prevent premature progression.
- **Artifact fingerprinting**: SHA256 hashes detect file changes and mark affected stages as `stale`.
- **Contract-backed fingerprinting**: Stage file artifacts are fingerprinted deterministically, and `writeup` expands local LaTeX dependencies from `main_tex`.
- **Experiments boundary**: The `experiments` stage enters `blocked` until the user provides a `data_path` with actual results. Rollback is always possible ("roll back to stage X").

**How it works:**
- Session start hook displays active run ID, current stage, and next action.
- Skills and agents automatically read/write run state per the [Run Card contract](orchestrator/run-card.md).
- Stage registry defined in `orchestrator/stages.json`; runtime library at `scripts/lib/orchestrator.js`.

See [docs/orchestrator.md](docs/orchestrator.md) for full documentation.

### Supporting Workflows

These workflows run in the background to enhance the primary workflows.

#### Automated Enforcement Workflow

Cross-platform hooks (Node.js) automate workflow enforcement:

```
Session Start → Skill Evaluation → Session End → Session Stop
```

- **skill-forced-eval** (`skill-forced-eval.js`): Before EVERY user prompt → dynamically scans all available skills (local + plugins) → forces evaluation of each skill → requires activation before implementation → ensures no relevant skill is missed
- **session-start** (`session-start.js`): Session begins → displays Git status, pending todos, available commands, package manager → shows project context at a glance
- **session-summary** (`session-summary.js`): Session ends → generates comprehensive work log → summarizes all changes made → includes orchestrator status and recent run events
- **stop-summary** (`stop-summary.js`): Session stops → quick status check → detects temporary files → shows actionable cleanup suggestions

**Cross-platform**: All hooks use Node.js (not shell scripts) ensuring Windows/macOS/Linux compatibility.

#### Knowledge Extraction Workflow

Two specialized mining agents continuously extract knowledge to improve skills:

- **paper-miner** (agent): Analyze research papers (PDF/DOCX/arXiv links) → extracts writing patterns, structure insights, venue requirements, rebuttal strategies → updates `ml-paper-writing/references/knowledge/` with categorized entries (structure.md, writing-techniques.md, submission-guides.md, review-response.md)
- **kaggle-miner** (agent): Study winning Kaggle competition solutions → extract competition briefs, front-runner detailed technical analysis, code templates, best practices → update the `kaggle-learner` skill's knowledge base (`references/knowledge/[domain]/` directories, categorized by NLP/CV/Time Series/Tabular/Multimodal)

**Knowledge feedback loop**: Each paper or solution analyzed enriches the knowledge base, creating a self-improving system that evolves with your research.

#### Skill Evolution System

3-step continuous improvement cycle for maintaining and improving skills:

```
skill-development → skill-quality-reviewer → skill-improver
```

1. **Develop** (`skill-development`): Create skills with proper YAML frontmatter → clear descriptions with trigger phrases → progressive disclosure (lean SKILL.md, details in `references/`)
2. **Review** (`skill-quality-reviewer`): 4-dimension quality assessment → Description Quality (25%), Content Organization (30%), Writing Style (20%), Structural Integrity (25%) → generates improvement plan with prioritized fixes
3. **Improve** (`skill-improver`): Merges suggested changes → updates documentation → iterates on feedback → reads improvement plans and applies changes automatically

## File Structure

```
claude-scholar/
├── AGENTS.md            # Codex behavioral reference (kept in repo; no longer copied)
├── .codex/              # Codex-specific files
│   └── INSTALL.md               # Codex installation guide
│
├── hooks/               # Cross-platform JavaScript hooks (Claude Code only)
│   ├── session-start.js         # Session begin - shows Git status, todos, commands
│   ├── skill-forced-eval.js     # Force skill evaluation before each prompt
│   ├── session-summary.js       # Session end - generates work log with recommendations
│   ├── stop-summary.js          # Session stop - quick status check, temp file detection
│   └── security-guard.js        # Security validation for file operations
│
├── skills/              # 35 specialized skills (domain knowledge + workflows)
│   ├── ml-paper-writing/        # Full paper writing: NeurIPS, ICML, ICLR, ACL, AAAI, COLM
│   │   └── references/
│   │       └── knowledge/        # Extracted patterns from successful papers
│   │       ├── structure.md           # Paper organization patterns
│   │       ├── writing-techniques.md  # Sentence templates, transitions
│   │       ├── submission-guides.md   # Venue requirements (page limits, etc.)
│   │       └── review-response.md     # Rebuttal strategies
│   │
│   ├── research-ideation/        # Research startup: 5W1H, literature review, gap analysis
│   │   └── references/
│   │       ├── 5w1h-framework.md           # Systematic thinking tool
│   │       ├── gap-analysis-guide.md       # 5 types of research gaps
│   │       ├── literature-search-strategies.md
│   │       ├── research-question-formulation.md
│   │       ├── method-selection-guide.md
│   │       └── research-planning.md
│   │
│   ├── results-analysis/         # Experiment analysis: statistics, visualization, ablation
│   │   └── references/
│   │       ├── statistical-methods.md      # t-test, ANOVA, Wilcoxon
│   │       ├── visualization-best-practices.md  # matplotlib/seaborn
│   │       ├── results-writing-guide.md    # Writing results sections
│   │       └── common-pitfalls.md          # Common analysis mistakes
│   │
│   ├── review-response/          # Systematic rebuttal writing
│   │   └── references/
│   │       ├── review-classification.md    # Major/Minor/Typo/Misunderstanding
│   │       ├── response-strategies.md      # Accept/Defend/Clarify/Experiment
│   │       ├── rebuttal-templates.md       # Structured response templates
│   │       └── tone-guidelines.md          # Professional language
│   │
│   ├── paper-self-review/        # multi-item quality checklist
│   ├── post-acceptance/          # Conference preparation
│   │   └── references/
│   │       ├── presentation-templates/     # Slide creation (15/20/30 min)
│   │       ├── poster-templates/           # Academic poster design
│   │       ├── promotion-examples/         # Social media content
│   │       └── design-guidelines.md        # Visual design principles
│   │
│   ├── citation-verification/    # Multi-layer citation validation
│   ├── writing-anti-ai/         # Remove AI patterns: symbolism, promotional language
│   │   └── references/
│   │       ├── patterns-english.md    # English AI patterns to remove
│   │       └── patterns-chinese.md     # Chinese AI patterns to remove
│   │
│   ├── architecture-design/     # ML project patterns: Factory, Registry, Config-driven
│   ├── git-workflow/            # Git discipline: Conventional Commits, branching
│   ├── bug-detective/           # Debugging: Python, Bash, JS/TS error patterns
│   ├── code-review-excellence/  # Code review: security, performance, maintainability
│   ├── skill-development/       # Skill creation: YAML, progressive disclosure
│   ├── skill-quality-reviewer/  # Skill assessment: 4-dimension scoring
│   ├── skill-improver/          # Skill evolution: merge improvements
│   ├── kaggle-learner/          # Learn from Kaggle winning solutions
│   ├── doc-coauthoring/         # Document collaboration workflow
│   ├── latex-conference-template-organizer  # Template cleanup for Overleaf
│   └── ... (10+ more skills)
│
├── commands/            # 50+ slash commands (quick workflow execution)
│   ├── research-init.md         # Launch research startup workflow
│   ├── analyze-results.md       # Analyze experiment results
│   ├── rebuttal.md              # Generate systematic rebuttal document
│   ├── presentation.md          # Create conference presentation outline
│   ├── poster.md                # Generate academic poster design plan
│   ├── promote.md               # Generate promotion content
│   ├── plan.md                  # Implementation planning with agent delegation
│   ├── commit.md                # Conventional Commits: feat/fix/docs/refactor
│   ├── code-review.md           # Quality and security review workflow
│   ├── tdd.md                   # Test-driven development: Red-Green-Refactor
│   ├── build-fix.md             # Fix build errors automatically
│   ├── verify.md                # Run verification loops
│   ├── checkpoint.md            # Save verification state
│   ├── refactor-clean.md        # Remove dead code
│   ├── learn.md                 # Extract patterns from code
│   └── sc/                      # SuperClaude command suite (20+ commands)
│       ├── sc-agent.md           # Agent management
│       ├── sc-estimate.md       # Development time estimation
│       ├── sc-improve.md         # Code improvement
│       └── ...
│
├── agents/              # 14 specialized agents (focused task delegation)
│   ├── literature-reviewer.md   # Literature search and trend analysis
│   ├── data-analyst.md          # Automated data analysis and visualization
│   ├── rebuttal-writer.md       # Systematic rebuttal writing
│   ├── paper-miner.md           # Extract paper knowledge: structure, techniques
│   ├── architect.md             # System design: architecture decisions
│   ├── code-reviewer.md         # Review code: quality, security, best practices
│   ├── tdd-guide.md             # Guide TDD: test-first development
│   ├── kaggle-miner.md          # Extract engineering practices from Kaggle
│   ├── build-error-resolver.md  # Fix build errors: analyze and resolve
│   ├── refactor-cleaner.md      # Remove dead code: detect and cleanup
│   ├── bug-analyzer.md          # Deep code execution flow analysis and root cause investigation
│   ├── dev-planner.md           # Implementation planning and task breakdown
│   ├── ui-sketcher.md           # UI blueprint design and interaction specs
│   └── story-generator.md       # User story and requirement generation
│
├── rules/               # Global guidelines (always-follow constraints)
│   ├── coding-style.md          # ML project standards: file size, immutability, types
│   ├── agents.md                # Agent orchestration: when to delegate, parallel execution
│   ├── security.md              # Secrets management, sensitive file protection
│   └── experiment-reproducibility.md  # Random seeds, config recording, checkpoints
│
├── orchestrator/        # Workflow Orchestrator (stage registry + run card)
│   ├── stages.json              # Stage definitions (10 stages, artifacts, gates)
│   └── run-card.md              # Skills/agents integration contract
│
├── policy/              # Paper policy engine (rule cards + validation + lint)
│   ├── rules/                    # Canonical paper-writing rule cards (single source of truth)
│   ├── profiles/                 # Domain/venue overlays (severity/params tuning)
│   ├── validate.sh               # Rule-card integrity validation
│   ├── lint.sh                   # Machine-enforceable lint checks
│   └── README.md                 # Policy engine design and conventions
│
├── scripts/
│   ├── install-codex.sh         # Codex installer (macOS/Linux, symlink-based)
│   ├── install-codex-windows.ps1 # Codex installer (Windows, junction-based)
│   └── lib/                     # Shared script utilities
│
├── CLAUDE.md            # Global configuration: project overview, preferences, rules
│
└── README.md            # This file - overview, installation, features
```

## Feature Highlights

### Skills (49 total)

**Writing & Academic:**
- `ml-paper-writing` - Full paper writing guidance for top conferences/journals
- `writing-anti-ai` - Remove AI writing patterns (bilingual support)
- `doc-coauthoring` - Structured document collaboration workflow
- `latex-conference-template-organizer` - LaTeX template management
- `daily-paper-generator` - Automated daily paper generation for research tracking

**Research Workflow:**
- `research-ideation` - Research startup: 5W1H brainstorming, literature review, gap analysis
- `results-analysis` - Experiment analysis: statistical testing, visualization, ablation studies
- `review-response` - Systematic rebuttal writing with tone management
- `paper-self-review` - multi-item quality checklist for paper self-assessment (figures + LaTeX math conformance)
- `post-acceptance` - Conference preparation: presentations, posters, promotion
- `citation-verification` - Multi-layer citation validation to prevent hallucinations
- `paper-figure-generator` - Generate editable SVG academic figures (system overviews, pipelines, architectures) via AutoFigure-Edit

**Development:**
- `daily-coding` - Daily coding checklist (minimal, auto-triggered)
- `git-workflow` - Git best practices (Conventional Commits, branching)
- `code-review-excellence` - Code review guidelines
- `bug-detective` - Debugging for Python, Bash, JS/TS
- `architecture-design` - ML project design patterns
- `verification-loop` - Testing and validation

**Security Audit:**
- `circom-auditor` - Circom / ZK circuit audit: soundness, completeness, privacy, constraint bugs (17-agent delegated workflow, vendored from [zk-skills](https://github.com/zksecurity/zk-skills))

**Plugin Development:**
- `skill-development` - Skill creation guide
- `skill-improver` - Skill improvement tools
- `skill-quality-reviewer` - Quality assessment
- `command-development` - Slash command creation
- `agent-identifier` - Agent configuration
- `hook-development` - Hook development guide
- `mcp-integration` - MCP server integration

**Utilities:**
- `uv-package-manager` - Modern Python package management
- `planning-with-files` - Markdown-based planning
- `lineage` - Opt-in experiment-lineage page for a research project: which lines are running, which stalled, which results the paper never reads
- `kaggle-learner` - Learn from Kaggle solutions

### Commands (50+)

**Research Commands:**
| Command | Purpose |
|---------|---------|
| `/research-init` | Launch research startup workflow (5W1H, literature review, gap analysis) |
| `/analyze-results` | Analyze experiment results (statistics, visualization, ablation) |
| `/rebuttal` | Generate systematic rebuttal document from review comments |
| `/presentation` | Create conference presentation outline |
| `/poster` | Generate academic poster design plan |
| `/promote` | Generate promotion content (Twitter, LinkedIn, blog) |

**Development Commands:**
| Command | Purpose |
|---------|---------|
| `/plan` | Create implementation plans |
| `/commit` | Commit with Conventional Commits |
| `/code-review` | Perform code review |
| `/tdd` | Test-driven development workflow |
| `/build-fix` | Fix build errors |
| `/verify` | Verify changes |
| `/checkpoint` | Create checkpoints |
| `/refactor-clean` | Refactor and cleanup |
| `/learn` | Extract reusable patterns |
| `/sc` | SuperClaude command suite (20+ commands) |

### Agents (14 specialized)

**Research Agents:**
- **literature-reviewer** - Literature search, classification, and trend analysis
- **data-analyst** - Automated data analysis and visualization
- **rebuttal-writer** - Systematic rebuttal writing with tone optimization
- **paper-miner** - Extract paper writing knowledge from successful publications

**Development Agents:**
- **architect** - System architecture design
- **build-error-resolver** - Fix build errors
- **code-reviewer** - Review code quality
- **refactor-cleaner** - Remove dead code
- **tdd-guide** - Guide TDD workflow
- **kaggle-miner** - Extract Kaggle engineering practices
- **bug-analyzer** - Deep code execution flow analysis and root cause investigation
- **dev-planner** - Implementation planning and task breakdown

**Design & Content Agents:**
- **ui-sketcher** - UI blueprint design and interaction specs
- **story-generator** - User story and requirement generation

## Quick Start

### Multi-Runtime Support

Claude Scholar supports two runtimes:

| | Claude Code | Codex |
|---|------------|-------|
| **Skills** | 35 (full) | 27 universal + 6 reference |
| **Hooks** | 5 automated | N/A (using-claude-scholar skill replaces) |
| **Commands** | 50+ slash commands | N/A (use skills directly) |
| **Agents** | 14 specialized | 14 (via `spawn_agent`) |
| **Install** | Clone / Plugin | Symlink only (native skill discovery) |

### Installation Options

#### Claude Code Installation

Choose the installation method that fits your needs:

##### Option 1: Plugin Installation (Recommended)

Install via Claude Code plugin manager:

```bash
# Step 1: Add marketplace
claude plugin marketplace add OniReimu/claude-scholar

# Step 2: Install plugin
claude plugin install claude-scholar@claude-scholar
```

**Benefits**: Automatic component discovery, version tracking, easy updates via `claude plugin update`.

**Includes**: All 35 skills, 50+ commands, 14 agents, 5 hooks, and project rules.

##### Option 2: Full Installation (Git Clone)

Complete setup by cloning directly to `~/.claude`:

```bash
# Clone the repository (--recursive pulls vendored skills:
# scientific-figure-making, fireworks-tech-graph, circom-auditor)
git clone --recursive https://github.com/OniReimu/claude-scholar.git ~/.claude

# Already cloned without submodules?
git -C ~/.claude submodule update --init --recursive

# Restart Claude Code CLI
```

**Includes**: All 35 skills, 50+ commands, 14 agents, 5 hooks, and project rules.

##### Option 3: Minimal Installation

Core hooks and essential skills only (faster load, less complexity):

```bash
# Clone repository
git clone https://github.com/OniReimu/claude-scholar.git /tmp/claude-scholar

# Copy only hooks and core skills
mkdir -p ~/.claude/hooks ~/.claude/skills
cp /tmp/claude-scholar/hooks/*.js ~/.claude/hooks/
cp -r /tmp/claude-scholar/skills/ml-paper-writing ~/.claude/skills/
cp -r /tmp/claude-scholar/skills/research-ideation ~/.claude/skills/
cp -r /tmp/claude-scholar/skills/results-analysis ~/.claude/skills/
cp -r /tmp/claude-scholar/skills/review-response ~/.claude/skills/
cp -r /tmp/claude-scholar/skills/writing-anti-ai ~/.claude/skills/
cp -r /tmp/claude-scholar/skills/git-workflow ~/.claude/skills/
cp -r /tmp/claude-scholar/skills/bug-detective ~/.claude/skills/

# Cleanup
rm -rf /tmp/claude-scholar
```

**Includes**: 5 hooks, 7 core skills (complete research workflow + essential development).

##### Option 4: Selective Installation

Pick and choose specific components:

```bash
# Clone repository
git clone https://github.com/OniReimu/claude-scholar.git /tmp/claude-scholar
cd /tmp/claude-scholar

# Copy what you need, for example:
# - Hooks only
cp hooks/*.js ~/.claude/hooks/

# - Specific skills
cp -r skills/latex-conference-template-organizer ~/.claude/skills/
cp -r skills/architecture-design ~/.claude/skills/

# - Specific agents
cp agents/paper-miner.md ~/.claude/agents/

# - Project rules
cp rules/coding-style.md ~/.claude/rules/
cp rules/agents.md ~/.claude/rules/
```

**Recommended for**: Advanced users who want custom configurations.

#### Codex Installation

```bash
# Clone the repository
git clone https://github.com/OniReimu/claude-scholar.git ~/claude-scholar

# Run the install script (creates symlinks, migrates legacy AGENTS.md)
chmod +x ~/claude-scholar/scripts/install-codex.sh
~/claude-scholar/scripts/install-codex.sh
```

**Windows (PowerShell):**
```powershell
git clone https://github.com/OniReimu/claude-scholar.git $HOME\claude-scholar
& "$HOME\claude-scholar\scripts\install-codex-windows.ps1"
```

**What it does:**
- Creates symlinks: `~/.agents/skills/claude-scholar` and `~/.codex/skills/claude-scholar` → `skills/`
- Detects and migrates legacy `~/.codex/AGENTS.md`
- Updates via `git pull` — no re-install needed

See [.codex/INSTALL.md](.codex/INSTALL.md) for detailed Codex installation guide.

### Requirements

- Claude Code CLI or Codex CLI (v0.91+)
- Git
- (Optional) Node.js (for hooks)
- (Optional) uv, Python (for Python development)

### First Run

After installation, the hooks provide automated workflow assistance:

1. **Every prompt** triggers `skill-forced-eval` → ensures applicable skills are considered
2. **Session starts** with `session-start` → displays project context
3. **Sessions end** with `session-summary` → generates work log with recommendations plus orchestrator state/event summary
4. **Session stops** with `stop-summary` → provides status check

## Project Rules

### Paper Policy Engine

Defined in `policy/`:
- `policy/rules/` is the single source of truth for paper-writing constraints (figures, LaTeX, citations, experiments, submission).
- Rule-card design uses frontmatter metadata (`id`, `layer`, `artifacts`, `phases`, `check_kind`, `enforcement`) plus required sections (`Requirement`, `Rationale`, `Check`, `Examples`).
- Layering model: `core` (always on), `domain` (field-specific), `venue` (conference/journal specific); profile overlays live in `policy/profiles/*.md`.
- SoK in v1 is activated by profile (for example `policy/profiles/security-sok-sp.md`), currently with semantic `SOK.*` rules (`SOK.TAXONOMY_REQUIRED`, `SOK.METHODOLOGY_REPORTING`, `SOK.BIG_TABLE_REQUIRED`, `SOK.RESEARCH_AGENDA_REQUIRED`).
- Current limitation: `policy/lint.sh --profile` loads a single flat profile file (no inheritance/composition yet).
- Validation and enforcement workflow:
  - `bash policy/validate.sh` for structure/integration checks
  - `bash policy/lint.sh` for machine-enforceable checks
- Skills/commands reference rules via `<!-- policy:RULE_ID -->` markers.

### Coding Style

Enforced by `rules/coding-style.md`:
- **File Size**: 200-400 lines maximum
- **Immutability**: Use `@dataclass(frozen=True)` for configs
- **Type Hints**: Required for all functions
- **Patterns**: Factory & Registry for all modules
- **Config-Driven**: Models accept only `cfg` parameter

### Agent Orchestration

Defined in `rules/agents.md`:
- Available agent types and purposes
- Parallel task execution
- Multi-perspective analysis

### Security

Defined in `rules/security.md`:
- Secrets management (environment variables, `.env` files)
- Sensitive file protection (never commit tokens, keys, credentials)
- Pre-commit security checks via hooks

### Experiment Reproducibility

Defined in `rules/experiment-reproducibility.md`:
- Random seed management for reproducibility
- Configuration recording (Hydra auto-save)
- Environment recording and checkpoint management

## Contributing

This is a personal configuration, but you're welcome to:
- Fork and adapt for your own research
- Submit issues for bugs
- Suggest improvements via issues

## License

MIT License

## Acknowledgments

Built with Claude Code CLI and enhanced by the open-source community.

### References

This project is inspired by and builds upon excellent work from the community:

- **[everything-claude-code](https://github.com/anthropics/everything-claude-code)** - Comprehensive resource for Claude Code CLI
- **[AI-research-SKILLs](https://github.com/zechenzhangAGI/AI-research-SKILLs)** - Research-focused skills and configurations
- **[zk-skills](https://github.com/zksecurity/zk-skills)** (MIT, zkSecurity) - ZK circuit security skills; `circom-auditor` is vendored via the `vendor/zk-skills` submodule

These projects provided valuable insights and foundations for the research-oriented features in Claude Scholar.

---

**For data science, AI research, and academic writing.**

Repository: [https://github.com/OniReimu/claude-scholar](https://github.com/OniReimu/claude-scholar)
