---
name: claim-architecture-review
description: This skill performs a post-draft STRUCTURAL edit of a paper — judging whether each paragraph should EXIST, where it belongs, and whether information is duplicated across sections — before any line-level polish. Use when the user asks to "review the paper structure", "check the story/logic is closed", "is this section bloated", "should this paragraph be here / moved / merged / cut", "find redundancy across Method/Results/Appendix", or "audit the claim architecture". It is SUBTRACTIVE and PROPOSE-ONLY — it emits a structural-edit plan, it does not rewrite the manuscript. Run it BEFORE writing-anti-ai (architecture before line edit) and BEFORE paper-self-review. Do not use for sentence-level polish (writing-anti-ai), checklist QA (paper-self-review), or drafting replacement text (ml-paper-writing).
tags: [Writing, Structure, Claim, Architecture, Review]
version: 0.2.0
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
| `architecture-review/information-ledger.md` | **Working state.** Append-only redundancy index. One row per *information unit*: `info-key | canonical-proposition-gloss | first-home | other-homes | unique?`. An entry in `other-homes` may be tagged `(re-anchor)` — a second home P2 keeps on purpose, with the canonical home recorded beside it. Redundancy is caught by **lookup in this ledger**, not by recalling the whole paper. |
| `architecture-review/progress.md` | **Working state.** Sections audited so far → resume after interruption. |
| `architecture-review/paragraph-audit.md` | **Final artifact.** Per-paragraph audit table (built incrementally). |
| `architecture-review/relocation-map.md` | **Final artifact.** Cross-section redundancy clusters → canonical home + collapse plan. |

Do NOT pre-create empty files; write each as its pass produces content. Field schemas: `references/audit-schema.md`.

## Protocol (4 passes)

- **P0 — Spine + lead.** Read only abstract + intro + section headings + each paragraph's topic sentence. Extract 1–3 paper-level claims + one obligation per core section. For **each** claim also record `lead` — what kind of defensible advantage it carries — from the closed set `{capability, mechanism, cost, scale, none}` (新能力 = a capability that did not exist before / 新机制 = an explanation or mechanism others lack / 更低成本 / 更好扩展性 / `none` = this claim is true but leads on nothing). `none` is an explicit, legal value and a **finding**, not a blank: a spine where every claim is `lead: none` usually means the paper is on the wrong battlefield, and the fix is repositioning (see `ml-paper-writing` → *选对战场 — Choosing the Battlefield*), not cutting paragraphs. Write `spine.md`.
- **P1 — Per-section sweep.** For each section, read ONLY that section. Audit each paragraph against the resident `spine.md`; append a row to `paragraph-audit.md`. Decompose a compound paragraph into multiple information units. **An enumerated set counts as one unit, keyed by the set** — `{CNN channels, MLP hidden units, attention coords}` is one info-key, not four; that is how a second enumeration of the same set in another section gets caught (`PROSE.RULE_OF_THREE` owns the within-section case, this pass owns cross-section). For each unit: **lookup-before-create** in `information-ledger.md` (does this proposition or set already have a home?) — if yes, record the duplicate home; if no, create a row with a canonical-proposition gloss. A repeated enumerated set caught this way is `PROSE.RULE_OF_THREE`'s cross-section case — the line edit owns it within a section, this pass owns it across sections. <!-- policy:PROSE.RULE_OF_THREE -->
  A paragraph referred here by `writing-anti-ai` because **most of its sentences carry no proposition** is decided at this pass, not sentence by sentence: extract the paragraph's information units as usual — if none survives (no variable, value, mechanism, or conclusion is asserted anywhere in it), it has `unique_info=false` and takes `merge` into the paragraph that does carry the claim, or `delete`. Do **not** send it back for line-level rewriting; "make each empty sentence more specific" produces better-sounding filler. If a unit does survive, the paragraph is `tighten` and the surviving unit is what it keeps. <!-- policy:PROSE.SEMANTIC_IDLING -->
  For a **Results / Evaluation** section, also judge each experiment against the spine: which of the four roles does it carry (establish the method / explain where the advantage comes from / show value in the target scenario / rule out the most plausible competing explanation)? An experiment carrying none takes the redesign → demote → delete ladder, mapped onto this skill's verdicts as `tighten` → `move:appendix` → `delete`, and `delete` still needs `unique_info=false AND required_caveat=false`. **An unfavourable result is never roleless** — it is evidence, and dropping it is misconduct, not editing. <!-- policy:EXP.EXPERIMENT_ROLE -->
  Also record `backs_lead` — does this paragraph serve a claim whose `lead ≠ none`? **没有形成优势的内容不进主线**: `backs_lead=false` makes the paragraph a candidate for `tighten` or `move:appendix` (see Verdicts & safety for the limits on this). Update `progress.md`.
- **P2 — Redundancy / relocation.** Read the ledger only. Any proposition with >1 home → a cluster. Choose ONE canonical home (Method = protocol; Results = read the table; Appendix = detail, not re-explanation; Limitation = mark the boundary, not soothe) and write the collapse plan to `relocation-map.md`. This pass owns the cross-section half of `PROSE.RESTATEMENT_DILUTION` (the same proposition in several sections) and of `PROSE.OVER_DEFENSIVE` (the same caveat with several homes — the canonical home is the design description or Limitations, and **a caveat that is a reviewer comment's only visible answer is moved, never deleted**). <!-- policy:PROSE.RESTATEMENT_DILUTION --> <!-- policy:PROSE.OVER_DEFENSIVE -->

  **Re-anchor homes survive the collapse.** Not every second home is redundancy. When a reader reaches a section that turns on a construct defined several sections earlier — an RQ set, a named term, a framework stage — restating it there is ordinary human writing, and collapsing it to the first home makes the reader turn back. This pass is where that damage happens, because the ledger sees only "one proposition, two homes" and is built to compress exactly that. A second home is a **re-anchor**, not a duplicate, when **both** hold:

  1. **Self-contained.** The re-mention reads without holding the label→content binding. *"The study asks three questions. First, what does a reviewer's visible action contain? ..."* stands alone; *"RQ1 establishes which public events can be interpreted. RQ2 asks whether ..."* does not — it presupposes the reader is still carrying what `RQ1` names, which is the debt the re-anchor was supposed to discharge. **A label ping never qualifies, at any distance.**
  2. **Far enough that the reader would otherwise turn back.** Judge the lookup cost, do not count. There is deliberately **no numeric threshold**: a number turns this into "count the intervening sections" when the answer is "would the reader have to go back and look". At least one intervening section boundary is a floor, not a gate. Measure in sections, never in pages — page positions move with layout.

  Both conditions, not either. A full self-contained restatement placed two lines above the heading it previews is still the fingerprint; distance without self-containment is still a label ping.

  **Keeping a re-anchor is an auditable act**: tag the entry in `other-homes` as `(re-anchor)` and record where the canonical home is. An unexplained survivor reads the same as a missed duplicate.

  ⚠️ **This defeats the redundancy collapse only.** It does not exempt anything from `PROSE.SEMANTIC_IDLING` form A — a recap that asserts nothing is still empty however far it sits from what it recaps. The two tests run separately and a re-anchor must pass both. <!-- policy:PROSE.SEMANTIC_IDLING --> <!-- policy:PROSE.FRACTAL_SUMMARY -->
- **P3 — Narrative closure.** Read `spine.md` + section topic sentences only. Two questions, not one. **(a) Closure**: is the spine a closed loop (each claim set up and paid off, in order) or scattered? **(b) Fit**: is this spine the proposition the strongest evidence points at, or the one the original outline planned? A paper can close perfectly around a claim its own data does not support. When (b) fails, stop planning relocations on the wrong spine — `PAPER.OUTCOME_LOGIC` authorises redefining the problem and reordering the contributions, and that reframing is a *ml-paper-writing* move, not a relocation. Sections ordered by implementation history rather than by what they establish are also this pass's finding. <!-- policy:PAPER.OUTCOME_LOGIC -->
  Record the gap list at the top of `relocation-map.md`.

Linear in paper length; never quadratic.

**Targeted entry (the common case).** A referral from `writing-anti-ai` while polishing one section does **not** require the full P0–P3. `spine.md` present → run **P1 on that section only**. Absent → **P0** first (abstract + intro + headings + topic sentences only — it does not read body prose) then P1 on that section. Run P2/P3 only when the finding needs a global relocation plan or the spine itself is in question. `progress.md` makes this resumable, so a targeted run is a down payment on the full audit, not throwaway work.

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
