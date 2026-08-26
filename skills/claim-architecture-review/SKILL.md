---
name: claim-architecture-review
description: This skill performs a post-draft STRUCTURAL edit of a paper — judging whether each paragraph should EXIST, where it belongs, and whether information is duplicated across sections — before any line-level polish. Use when the user asks to "review the paper structure", "check the story/logic is closed", "is this section bloated", "should this paragraph be here / moved / merged / cut", "find redundancy across Method/Results/Appendix", or "audit the claim architecture". It is SUBTRACTIVE and PROPOSE-ONLY — it emits a structural-edit plan, it does not rewrite the manuscript. Run it BEFORE writing-anti-ai (architecture before line edit) and BEFORE paper-self-review. Do not use for sentence-level polish (writing-anti-ai), checklist QA (paper-self-review), or drafting replacement text (ml-paper-writing).
tags: [Writing, Structure, Claim, Architecture, Review]
version: 0.1.2
---

# Claim & Paragraph Architecture Review

The editorial hierarchy is **structural edit → line edit**. `writing-anti-ai` is the line edit (sentence AI-tells, prose). This skill is the missing **structural edit**: it audits the claim spine, paragraph necessity, paragraph placement, and cross-section redundancy. It does NOT polish sentences and does NOT mutate the manuscript — it produces a plan the `rewrite` stage (or the user) applies.

> Style authority still applies: `<!-- style:author-voice -->` (`policy/style-guide.md`). Claim↔source support is a separate layer handled by `<!-- policy:CITE.CLAIM_SUPPORT_REQUIRED -->` (citation-verification); this skill is about claim↔**paragraph placement**, not citation support.

## Core principle — advance vs soothe

For every paragraph ask: **is it advancing a claim, or soothing anxiety?** A paragraph earns its place only if it does at least one of: define the problem, support an experiment, interpret a result, set a required boundary (threat model / scope / overclaim defense), or carry necessary navigation. A paragraph that does none of these is usually making the paper *look* safer while making it weaker — the paragraph-level analog of the "we do not claim… / is…not…" sentence tic. Default bias is **subtractive**, but see Safety: required boundary-setting is load-bearing and stays.

## File-backed working space (handles long papers)

NEVER try to hold the whole paper plus all analysis in context. This skill is **multi-pass and file-backed**: at any moment context holds only `{small spine + small ledger + ONE current section}`. All state lives under a **static, repo-relative** directory **`architecture-review/`** (not "next to main.tex" — fixed path so orchestrator fingerprint/stale stay stable):

| File | Role |
|---|---|
| `architecture-review/spine.md` | **Working state.** 1–3 paper-level claims (each tagged with the kind of lead it carries) + one obligation per core section. Small, kept resident across passes. The insight anchor. |
| `architecture-review/information-ledger.md` | **Working state.** Append-only redundancy index. One row per *information unit*: `info-key | canonical-proposition-gloss | first-home | other-homes | unique?`. Redundancy is caught by **lookup in this ledger**, not by recalling the whole paper. |
| `architecture-review/progress.md` | **Working state.** Sections audited so far → resume after interruption. |
| `architecture-review/paragraph-audit.md` | **Final artifact.** Per-paragraph audit table (built incrementally). |
| `architecture-review/relocation-map.md` | **Final artifact.** Cross-section redundancy clusters → canonical home + collapse plan. |

Do NOT pre-create empty files; write each as its pass produces content. Field schemas: `references/audit-schema.md`.

## Protocol (4 passes)

- **P0 — Spine + lead.** Read only abstract + intro + section headings + each paragraph's topic sentence. Extract 1–3 paper-level claims + one obligation per core section. For **each** claim also record `lead` — what kind of defensible advantage it carries — from the closed set `{capability, mechanism, cost, scale, none}` (新能力 = a capability that did not exist before / 新机制 = an explanation or mechanism others lack / 更低成本 / 更好扩展性 / `none` = this claim is true but leads on nothing). `none` is an explicit, legal value and a **finding**, not a blank: a spine where every claim is `lead: none` usually means the paper is on the wrong battlefield, and the fix is repositioning (see `ml-paper-writing` → *选对战场 — Choosing the Battlefield*), not cutting paragraphs. Write `spine.md`.
- **P1 — Per-section sweep.** For each section, read ONLY that section. Audit each paragraph against the resident `spine.md`; append a row to `paragraph-audit.md`. Decompose a compound paragraph into multiple information units. **An enumerated set counts as one unit, keyed by the set** — `{CNN channels, MLP hidden units, attention coords}` is one info-key, not four; that is how a second enumeration of the same set in another section gets caught (`PROSE.RULE_OF_THREE` owns the within-section case, this pass owns cross-section). For each unit: **lookup-before-create** in `information-ledger.md` (does this proposition or set already have a home?) — if yes, record the duplicate home; if no, create a row with a canonical-proposition gloss. Also record `backs_lead` — does this paragraph serve a claim whose `lead ≠ none`? **没有形成优势的内容不进主线**: `backs_lead=false` makes the paragraph a candidate for `tighten` or `move:appendix` (see Verdicts & safety for the limits on this). Update `progress.md`.
- **P2 — Redundancy / relocation.** Read the ledger only. Any proposition with >1 home → a cluster. Choose ONE canonical home (Method = protocol; Results = read the table; Appendix = detail, not re-explanation; Limitation = mark the boundary, not soothe) and write the collapse plan to `relocation-map.md`.
- **P3 — Narrative closure.** Read `spine.md` + section topic sentences only. Is the spine a closed loop (each claim set up and paid off, in order) or scattered? Record the gap list at the top of `relocation-map.md`.

Linear in paper length; never quadratic.

## Verdicts & safety (conservative by default)

Per-paragraph verdict ∈ `{keep, tighten, merge, move:<section>, move:appendix, split, delete, escalate}`.

- **`delete` is legal ONLY when `unique_info=false` AND `required_caveat=false`.** Every `move`/`merge`/`delete` must name the **surviving home** for the information.
- **Required caveats stay by default**: threat-model boundaries, scope conditions, overclaim defenses, and venue-mandated Limitations are load-bearing, not deletable.
- **`backs_lead=false` is NOT a deletion licence.** "Supports no advantage" downgrades a paragraph's claim on main-line space — it can justify `tighten` or `move:appendix`, and nothing more. `delete` still requires `unique_info=false` AND `required_caveat=false`, judged independently of `lead`. Required boundary-setting (threat model, scope conditions, overclaim defense, venue-mandated Limitations) stays regardless of whether it backs a lead — a paper does not become safer to overclaim just because its caveats win no comparison.
- **Low confidence → `escalate`, never `delete`.** Most safe wins are `tighten` / `merge` / replace-with-forward-reference, not hard deletion.
- **Propose-only**: this skill writes the two artifacts; it does not edit the manuscript. The `rewrite` stage (or the user) applies approved moves.

## Orchestrator Integration

This skill owns stage: **`architecture_review`** (between `writeup` and `self_review`).

**Attach only when manuscript-aware and compatible** — otherwise run standalone:
1. Attach iff: an active run exists AND `writeup` is `done` AND the target `main.tex` matches `run.artifacts.writeup.main_tex` AND the run's `stages` map contains `architecture_review` (i.e. a run created on/after this stage shipped).
2. If attached: mark `architecture_review` → `in_progress`; run P0–P3; then `fingerprintStageArtifacts({ cwd, run, stageId: 'architecture_review', extraPaths: ['architecture-review/spine.md', 'architecture-review/information-ledger.md', 'architecture-review/progress.md'] })` (the two final artifacts come from the stage's `kind:file` contract; the working-state files are tracked via `extraPaths` — do NOT assume whole-directory fingerprinting). Request human approval of the plan before marking `done`; `next_stage` is `rewrite` (which applies approved moves, or is marked `skipped` if the plan is a no-op).
3. **No active run, or any condition above unmet → standalone**: run P0–P3, emit the two artifacts, no orchestrator interaction. This (manual invocation) is the primary mode. Do NOT call `initRun()`; do NOT attempt to `markStage` a stage the run lacks (it throws `Unknown stageId`). Runs created before this stage existed are **new-runs-only** in v1 — for them, run standalone (migration deferred to v1.1).

Bypass (checklist-only review requested): set `run.inputs.skip_architecture_review = true` as an audit trace, then mark `architecture_review` → `skipped` with a note warning that additive completeness checks (paper-self-review) may re-bloat the draft.

## Boundaries (no overlap)

- `ml-paper-writing` — how to structure WHILE drafting (write-time narrative principle; also owns *选对战场 — Choosing the Battlefield*, the repositioning move to reach for when the spine's leads come back `none`). Hands off here after the draft exists.
- **`claim-architecture-review` (this) — post-draft architecture audit + relocation plan.**
- `paper-self-review` — completeness / compliance checklist (additive); runs AFTER this skill so completeness is judged on a de-bloated draft.
- `writing-anti-ai` — line/copy polish, LAST. It owns within-paragraph and within-section repeated enumeration (`PROSE.RULE_OF_THREE`); a set this pass names once is what its "refer to the name" fix references.

## When to use

After a draft exists and before polish; when a section feels bloated; when the same thing seems explained in several places; when the story feels scattered rather than a closed loop. Deferred to v1.1: `/claim-arch` command alias, write-time silent claim ledger, fuzzy semantic dedup, old-run migration.
