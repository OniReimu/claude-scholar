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
6. **Narrative fields obey the repo author-voice policy.** `narrative`/`criterion-scored`
   prose is a writing task: it follows `policy/style-guide.md` + relevant `PROSE.*` rules,
   applied at line-edit via `writing-anti-ai`. This is distinct from the grant-specific
   evidentiary rules above.

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
B3  project substance → project-plan.yaml    project mode: aims/design, benefits, additionality/VfM, risk-triggers —
                                             the registers the §2.14–§2.17 substance passes render from (mechanized by validate_ir --plan)
C   fill: per field → select evidence + mode-aware method passes
D   render to native modality                paste-ready / docx write-back / AcroForm / honest degrade
E   review: checklist-driven contract        eligibility·compliance·evidence·consistency·budget-math·
                                             attachments·panel-fit·risk·portal dry-run — the mechanical
                                             items run as ONE gate: `scripts/validate_ir.py` (single
                                             pre-submit dry-run; composes charcount + validate_budget,
                                             enforces cross-field couplings). Judgement items stay adversarial.
F   submission_plan.yaml                      owners, due dates, internal cutoffs, approvals, dependency graph
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
- **`writing-anti-ai`** — strip AI-pattern prose from drafted boxes.
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
