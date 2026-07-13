---
name: grant-application-writing
description: This skill turns a blank funding form (Word / PDF / web-portal) plus a directory of an academic's materials into a filled, submission-ready application. Covers competitive research funding — grants, fellowships, awards, nominations (写本子) — across three modes (narrative track-record award, prospective project grant, retroactive impact) and many schemes (ARC, NHMRC, ERC, NSF, CRC-P, AEA, EF, industry, internal). Use for eligibility analysis, parsing a scheme into a form-schema, building a reusable evidence store, drafting each field to its rubric criterion + weight, claim/verb defensibility, anti-double-counting, budget-math and compliance checks, adversarial review, and rendering to the form's native modality (paste-ready, .docx, AcroForm PDF). Do not use for journal/conference papers (ml-paper-writing), citation-only checks (citation-verification), rebuttal drafting (review-response), general docs/proposals not tied to a funding form (doc-coauthoring), or figure rendering (paper-figure-generator).
version: 0.1.0
author: Orchestra Research
license: MIT
tags: [Grant Writing, Fellowship, Award, ARC, NHMRC, ERC, NSF, CRC-P, AEA, Research Funding, Applications, 本子]
dependencies: [python-docx, pypdf, pyyaml]
---

# Grant / Fellowship / Award Application Writing (写本子)

Turn **(a blank funding form + a messy directory of the applicant's materials)** into
**the same form, filled, in its native modality**. This skill is the funding-application
sibling of `ml-paper-writing`: papers persuade on *contribution novelty*; applications
persuade on *the applicant's defensibility and fit against a rubric*, are **form-shaped**
(fill boxes, not free prose), and fail on **auto-ineligibility** or a single indefensible
number — a different genre with its own machinery.

## ⚠️ Core discipline

1. **Never invent evidence.** Every claim traces to a source file in the evidence store
   with a provenance pointer and a defensibility status. Patent "granted" ≠ "filed";
   a paper's date must fall inside the eligibility window; a role is first / corresponding
   / co-first / CI / supervising — never upgraded. Under-claim before you over-claim.
   Two corollaries (`method-passes.md` §1.8): an invented technical spec/number the input
   didn't supply is tagged **`[TO SET]`**, never shown as real; and an **unverified/stale
   context** (employment, load, role) is stated honestly, **never smoothed into a favourable
   framing** ("teaching-heavy", not laundered to "research-focused").
2. **Eligibility gates run BEFORE drafting.** If a hard gate fails, say so and stop —
   do not draft an application the applicant cannot submit.
3. **The verb is the evidentiary commitment.** Pick the strongest verb the backing
   evidence survives under reviewer probe, and no stronger (see `method-passes.md`
   verb-tiering).
4. **Honesty about modality.** A form modality without a working renderer is declared
   unsupported and downgraded to paste-ready — never faked as an official fill.
5. **Zero real applicant data in this skill.** The skill ships the type model, protocol,
   passes, templates, and renderers; worked examples use a fictional applicant. Each real
   application is an IR instance in its own project folder — never hardcode a scheme's
   boxes or a real person's evidence here.
6. **Narrative fields obey the repo author-voice policy — and `writing-anti-ai` is a
   MANDATORY pre-render gate, not an optional polish.** `narrative`/`criterion-scored`
   prose is a writing task: it follows `policy/style-guide.md` + relevant `PROSE.*` rules.
   Every drafted narrative/criterion-scored box MUST pass `writing-anti-ai` at line-edit
   **before Stage D render** — a box that has not been through it is not render-ready.
   Grant prose is a dense generator of AI tells (em-dashes, `X, not Y` negation-contrast,
   rule-of-three, promotional adjectives, comma-overload); leaving them in reads as
   machine-written to an assessor. Assessor-facing text under a rubric is exactly where the
   tells cost score, so the gate is enforced, not advisory. This is distinct from — and runs
   after — the grant-specific evidentiary rules above (never let a de-AI edit weaken a verb's
   evidentiary commitment or launder an honest `[TO SET]`).

## Output & drafting conventions (non-negotiable)

These sit alongside Core discipline and govern HOW the output is packaged and drafted.

1. **Deliverables are Markdown; IR is YAML and kept separate; the final export is .docx/.pdf.**
   The human-facing outputs (paste-ready, budget table, blockers→`HUMAN-ACTIONS.md`, letters) are
   **`.md`, never `.txt`**. The machine IR (`scheme`/`entity`/`project-plan`/`budget`/`evidence`/
   `*-plan`/`manifest`/`milestones`) is YAML and lives in an **`_ir/` subfolder**, NOT interleaved
   with the `.md` deliverables. The final artifact the applicant submits is a rendered **`.docx`
   (or `.pdf`)** — the filled official template (Stage D `render_docx.py`), not a loose text dump.
2. **⚠️ TEMPLATE FIDELITY IS ABSOLUTE — the final .docx MUST be the official template, filled
   in place (100% fidelity). This is non-negotiable.** You MUST **dissect the official template's
   real structure first** — enumerate its paragraphs, headings, and tables (`Document(...).paragraphs`
   / `.tables`; see `scripts/render_docx.py`) — and then fill answers INTO that structure:
   - **NEVER build a self-styled document as a substitute.** A clean doc you author yourself, even
     with the right headings, is WRONG. The funder's exact template, labels, instructions, and
     tables must all survive verbatim.
   - Most official forms are **tag-less** (plain Normal-styled label paragraphs, no content controls,
     no Heading styles). For these you locate **each field's label paragraph and write the answer
     into the empty answer slot immediately after it** — INSERTING extra paragraphs, never
     overwriting any label or instruction. The **budget goes INTO the template's own line-item
     table** (never in a separate file). `render_docx.py` does exactly this: strategy-3 `under-label`
     (via a `--label-map` of {field → the label's text in THIS template}) + `--tables` for the budget.
   - The locator map is per-template **instance data** (`_ir/render-map.yaml`, `_ir/tables.yaml`),
     produced by reading the actual template — not guessed.
   - The filled template is handed back for **human cross-validation** against the live form before
     submission; the skill states this explicitly and never implies the fill is final.
3. **Context-provenance: briefing ≠ assertable content (don't leak the brief).** Material given to
   BRIEF the agent (background, prior work, "for your understanding") must NOT be auto-promoted into
   the application text as confirmed claims. Distinguish, per claim: `corpus-confirmed` (a fact the
   applicant stated) · `briefing-background` (context for the agent) · `web-verified` (with a
   source) · `inferred`. Public-facing fields (plain-language summaries) stay generic and do not
   lean on internal project scaffolding; technical fields may cite real prior work but every
   background-derived claim is **labelled so the human can strike it** (a companion provenance note,
   or an inline tag). When unsure whether background is assertable, ASK — do not assert it.
4. **Fill to 90–95% of every limit.** Under-use forfeits score. `charcount.py` flags OVER; the
   drafting target is **≥90% of each field's char/word limit** (not merely "under"). A field left at
   ~60% is a flagged gap to expand, exactly as an over-limit field is a gap to cut.
5. **Single source of truth — paired machine/human artifacts MUST NOT drift.** Several outputs exist
   as a machine copy + a human copy of the SAME facts: `blockers.yaml` ↔ `blockers.md` ↔
   `HUMAN-ACTIONS.md`; `entity-store` contributions ↔ the budget contributions table; `budget.yaml`
   ↔ the Section-5 table. The **YAML/IR is canonical**; the human-readable is either (a) **rendered
   from it** (preferred — `budget-section5` renders from `entity-store`, the docx/md from
   `values.yaml`), or (b) if hand-maintained, **updated in the SAME edit** as its machine twin. Never
   change one copy and leave the other stale — editing `blockers.yaml` without `blockers.md`, or
   fixing an `entity-store` figure without the table that shows it, silently ships a contradiction to
   the assessor. A structured fact belongs in the DATA first (§2.8 "budget authored as data"), never
   as free prose that duplicates — and then contradicts — the store.

## Intake enrichment (Stage A0/B — automatic intelligence)

At intake, don't rely solely on the supplied corpus. Actively enrich the entity/evidence stores from
public sources (WebSearch/WebFetch): the partner org + its lab/centre, each named investigator's
public profile + real expertise, and the pre-existing relationship between the applicant's and the
partner's institutions (MoUs, joint centres, prior grants). This routinely (a) STRENGTHENS the case
(e.g. a decade-old joint research centre corrects an understated "new relationship" framing) and
(b) SURFACES honesty risks (e.g. a named lead whose public expertise does not match the role the
draft assigns them). Every enriched fact carries a `provenance` URL + `as_of`; a formal form field
(legal name, lead of record) still needs the partner's own confirmation, so enrichment INFORMS but
does not replace `[VERIFY]`.

## IO contract

```
INPUT
 ├─ blank form        (.docx | fillable .pdf | flat .pdf | web-portal fields)
 ├─ guidelines        (rubric / weights / eligibility)      ← usually a separate doc
 └─ material corpus/  (CV, prior apps, pub lists, funding records, screenshots… messy)

OUTPUT  the SAME form, filled, in its native modality
 ├─ web-portal → PASTE-READY.txt   (one block per field, char-count header)
 ├─ .docx      → filled copy of the official template
 ├─ fillable .pdf → AcroForm field fill
 └─ flat/scanned .pdf → paste-ready + companion .docx, marked non-official (honest degrade)

 + ALWAYS, alongside the filled form (Stage F++):
 ├─ HUMAN-ACTIONS.md          plain-language, jargon-free handoff of every remaining human
 │                            action/decision (the readable twin of the technical blockers.md)
 └─ letters/ (as required)    a DRAFT of every required proforma letter/attachment the applicant
                              can't auto-fill (partner + lead-org letters of support), [TO SET]-marked
```

## Dispatch is TWO axes

Route every scheme on **two orthogonal axes** before drafting: `mode` (what you're judged ON)
selects the register + method-pass group; `process` (how the judging is STRUCTURED) selects the
process-archetype overlays. A DECRA and a Google research award can share a `mode` yet need
completely different pipelines because their *process* differs — and an ARC DECRA (narrative-award)
and an ARC Linkage (prospective-project) share ARC process conventions across different modes. Set
`mode` first (below), then `process` (next section).

## Funding-mode dispatch (axis 1 — do this FIRST, after intake)

Determine `mode` — it selects which field-clusters dominate and which method passes run.
The three modes barely overlap in required fields; a single flat approach cannot serve all.

| mode | when | examples | drafting spine |
|------|------|----------|----------------|
| `narrative-award` | judged on track record; char-limited prose + ranked outputs | ECR / early-career awards, ARC DECRA track-record, NHMRC Investigator, ERC track-record | past achievements, defensibility, criterion-weighted prose |
| `prospective-project` | judged on a future project; budget + team + compliance | ARC DP/LP/FT, CRC-P, AEA, NHMRC Ideas, NSF, ERC B2, EF-ESP, big-tech | aims→methods→milestones→budget coherence; feasibility; impact pathway |
| `retroactive-impact` | judged on past impact delivered; no future plan | Optimism RetroPGF, Gitcoin | impact-evidence marshalling → attestation → Sybil/netting → freshness → past-impact scoring (`method-passes.md` Group 3; built, lightly-validated — not yet run on a live retro round) |

**`mode` also picks the drafting register** (`author-voice.md`): `prospective-project` /
commercialisation → §1–7 (humble team, unquantified market, partner-as-anchor); `narrative-award`
/ fellowship → §8 (person-as-product, quantified-but-sourced eminence, ROPE, future-leadership).
Using the wrong register — a market TAM in a fellowship, or bare self-eminence in a project grant —
reads as off-genre to an assessor. Pick the register from `mode` before drafting.

## Assessment-process dispatch (axis 2 — do this SECOND, after mode)

`mode` fixes what you're judged on; `process` fixes **how the competition is structured** — and
that changes which stages and passes run, and how heavy they should be. `process` is a **set** (a
scheme can be several at once) drawn from a closed archetype vocabulary. It selects the
**process-archetype overlays** (`method-passes.md` Group 5), which run *on top of* the Group the
`mode` picked.

| `process` tag | shape | exemplars | what the overlay does |
|---------------|-------|-----------|-----------------------|
| `single-stage-review` | one full submission, expert-panel rubric-scored | ARC DP/DECRA/LP/FT | default weight; **+rejoinder-prep** if a right-of-reply window exists (`rejoinder.enabled`) |
| `staged` | a gating EOI / pre-proposal precedes the full | NHMRC, some ERC, foundations | **+EOI sub-pipeline** drafted to *its own* rubric+limits; the full (phase 2) must not contradict the EOI |
| `interview-gated` | written shortlist → live interview/pitch decides | ERC Step-2, some fellowships, corporate finals | **+defense-prep** artifact (anticipated Qs from the written weak points); the written *sets up* the interview |
| `panel-routed` | classification codes route the app to the scoring panel | ARC FoR, NHMRC, NSF directorate | **taxonomy/classification fields become gate-critical** — a wrong code → wrong assessors → silent loss |
| `curated` | program-officer discretion, light-touch form, no scored rejoinder | industry gift awards, internal seed, philanthropy | **down-scales the machinery** — the failure mode is *over-engineering* |
| `rolling` | no synchronized deadline; timing is strategic | some foundations, non-retro crypto grants | Stage F timing strategic; freshness matters even outside retro |

`rejoinder` (a within-round right-of-reply) is a **capability**, not an archetype: model it as
`rejoinder: {enabled, window?, char_limit?}` in the IR; `single-stage-review` may carry it.

**Compose the two axes.** ARC DECRA = `narrative-award` × `{single-stage-review, panel-routed}` +
rejoinder. NHMRC Investigator = `narrative-award` × `staged`. ERC StG = `prospective-project` ×
`{staged, interview-gated}`. Google research award = `prospective-project` × `curated`.

**Scale the machinery to the process — don't over-build.** The most common batch-3 error is running
the full heavyweight pipeline on a `curated` scheme: there is no scored budget to validate, no panel
rubric to saturate, no rejoinder to reserve for. On `curated`, deliberately *skip* the passes the
scheme does not score and lead with one fundable hook + program fit (repo simplicity-first
discipline). Distinct from `mode`, funder-family (ARC/NHMRC/NSF/ERC/industry) only tunes the
`reviewer_model` register (`method-passes.md` §4.1) — it is a soft framing lens, not a structural axis.

## Pipeline

```
A0  scheme intake → compliance_matrix.yaml   rulebook BEFORE drafting: eligibility, mandatory docs,
                                             limits, fonts, certs, internal deadlines, submission owner
A   form + guidelines → scheme.yaml (IR)     parse ANY scheme into the normalized type model
B   corpus → evidence-store.yaml             reusable across every application (see evidence-store.md)
B2  people/orgs/partners → entity-store      project mode: CI/PI/partner/subaward/commitments/approvals
B4  costing → budget.yaml                     project mode: `scripts/build_budget.py` COSTS + itemises the budget from
                                             human inputs (personnel = person×FTE×rate×years + on-costs; other_costs per
                                             year) → validate_budget's rows[] schema + an itemised Section-5 table. It
                                             COMPUTES from supplied rates, never invents one (a missing rate = `[TO SET]`).
B4s scheduling → milestones.yaml             `scripts/build_timeline.py` builds the schedule from the spine
                                             (tasks[].years[]+depends_on → Gantt + dependency-ordered milestones). Budget +
                                             timeline are two of a THIRD script class — BUILDERS (compute an artifact from
                                             inputs), distinct from validators/renderers. Six builders exist (build_budget/build_timeline/build_rope_time/
                                             build_effort_allocation/build_track_record_metrics/build_cocontribution; see CLAUDE.md "BUILDERS").
B3  project substance → project-plan.yaml    project mode: aims/design, benefits, additionality/VfM, risk-triggers +
                                             the traceability SPINE (stable ids: aim→objective→task→subtask→output→benefit,
                                             crossed by person→year→budget) — the §2.14–§2.18 substance passes render from it
                                             (mechanized by validate_ir --plan; the spine turns cross-field consistency into a deterministic check)
C   fill: per field → select evidence + mode-aware method passes
C+  anti-AI line-edit (MANDATORY gate)        every narrative/criterion-scored box passes `writing-anti-ai`
                                             (policy/style-guide.md + PROSE.* rules) BEFORE render — strips
                                             em-dashes / negation-contrast / promotional / comma-overload;
                                             must NOT weaken an evidentiary verb or launder a `[TO SET]`
D   render to native modality                paste-ready / docx write-back / AcroForm / honest degrade
E   review: checklist-driven contract        eligibility·compliance·evidence·consistency·budget-math·
                                             attachments·panel-fit·risk·portal dry-run — the mechanical
                                             items run as ONE gate: `scripts/validate_ir.py` (single
                                             pre-submit dry-run; composes charcount + validate_budget,
                                             enforces cross-field couplings). Judgement items stay adversarial.
F   submission_plan.yaml                      owners, due dates, internal cutoffs, approvals, dependency graph
F+  build-manifest.yaml                       run-audit: input hashes + artifacts + the validate_ir verdict +
                                             per-criterion readiness + open blockers → ONE fail-closed `ready_to_submit`
                                             boolean. `scripts/build_manifest.py` composes validate_ir; reproducible, not hand-authored.
F++ human handoff (MANDATORY)                 the run ALWAYS ends by writing, into the application's own folder, (1) a
                                             PLAIN-LANGUAGE handoff `HUMAN-ACTIONS.md` — every remaining human
                                             action/decision, jargon-free, in the applicant's language (default the
                                             application's language, usually English), ranked by severity, with owner +
                                             by-when; the readable twin of `blockers.md` (which stays technical); AND (2) a
                                             DRAFT of every required proforma letter/attachment the applicant cannot
                                             auto-fill (partner + lead-org letters of support, hosting/commitment letters),
                                             each with `[TO SET]` placeholders that reconcile with entity/budget (never an
                                             invented figure) and each through the C+ anti-AI gate. See `submission-management.md`.
```

Each stage is specified in a `references/` file. Read the ones the task needs:

- **`references/type-model.md`** — the two-axis field model (widget × semantic-role + attributes). Load-bearing; read first.
- **`references/form-schema-ir.md`** — the `scheme.yaml` IR (Stage A) and the A0 compliance matrix. `scripts/extract_form.py` bootstraps the field skeleton from a blank .docx/.pdf.
- **`references/evidence-store.md`** — Stage B: building/hardening the reusable evidence store; the entity store (B2).
- **`references/method-passes.md`** — Stage C/E: the mode-aware drafting and review passes.
- **`references/author-voice.md`** — Stage C: the register funded grant prose is written in (composition, sentence patterns, lexicon, strategic moves). Read when drafting `criterion-scored` narrative.
- **`references/modality-renderers.md`** — Stage D: the modality registry and each renderer's contract.
- **`references/submission-management.md`** — Stage A0 + F: compliance matrix and submission plan.

## When to use

- A blank grant / fellowship / award / nomination form to fill.
- Eligibility analysis ("can I even apply / is this my last year?").
- Parsing an unfamiliar scheme's guidelines into a workable structure.
- Building or refreshing a reusable research evidence base for applications.
- Adversarial pre-submission review of a drafted application.

## Cross-skill integration map (reuse, don't reinvent)

- **`citation-verification`** — verify every publication claim in the evidence store.
- **`claim-architecture-review`** — structural audit of narrative fields before line polish.
- **`writing-anti-ai`** — **MANDATORY Stage-C+ gate** (not optional): every drafted narrative/
  criterion-scored box passes it before render (see Core discipline #6 + pipeline C+).
- **`paper-self-review`** — the checklist idiom; Stage E adapts it for applications.
- **`review-response`** — resubmissions: response-to-reviewers on a re-applied scheme.
- **`knows-literature`** — pull publication metadata / citations for the evidence store.
- **`fireworks-tech-graph`** / **`paper-figure-generator`** — render the project description's structural artefacts (architecture figure, pipeline diagram) as real figures, not ASCII (`modality-renderers.md` §0).
- **`research-profile-evidence-base`** *(planned graduation of Stage B)* — the evidence store is reusable for CVs, biosketches, promotion cases, nominations; built as a module here, graduates to its own skill.

## Generality boundary

Per project policy: abstract rules live in this skill; scheme-specific instances live in the
application's own project folder (e.g. `.../Grants/<scheme>/`). An ECR scheme is the
first worked instance and the `narrative-award` regression fixture — its full drafts and the
applicant's real evidence never enter this skill (worked examples reference the case only
illustratively, with a fictional applicant).
