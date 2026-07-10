---
name: grant-application-writing
description: This skill turns a blank funding form (Word / PDF / web-portal) plus a messy directory of an academic's materials into a filled, submission-ready application. Covers competitive research funding — grants, fellowships, awards, nominations (写本子) — across three modes (narrative track-record award, prospective project grant, retroactive impact funding) and many schemes (ARC, NHMRC, ERC, NSF, CRC-P, AEA, EF, big-tech, university internal). Use for eligibility-gate analysis, parsing a scheme into a normalized form-schema, building a reusable evidence store, drafting each field against its rubric criterion and weight, claim/verb defensibility, anti-double-counting, budget-math and compliance completeness, adversarial review, and rendering into the form's native modality (paste-ready, filled .docx, AcroForm PDF). Do not use for journal/conference papers (ml-paper-writing), citation-only checks (citation-verification), rebuttal drafting (review-response), general docs/specs/proposals not tied to a funding form (doc-coauthoring), or figure rendering (paper-figure-generator).
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
2. **Eligibility gates run BEFORE drafting.** If a hard gate fails, say so and stop —
   do not draft an application the applicant cannot submit.
3. **The verb is the evidentiary commitment.** Pick the strongest verb the backing
   evidence survives under reviewer probe, and no stronger (see `method-passes.md`
   verb-tiering).
4. **Honesty about modality.** A form modality without a working renderer is declared
   unsupported and downgraded to paste-ready — never faked as an official fill.
5. **Zero scheme instances in this skill.** The skill ships the type model, protocol,
   passes, templates, and renderers. Each application is an IR instance in its own
   project folder. Never hardcode a specific scheme's boxes here.

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

## Funding-mode dispatch (do this FIRST, after intake)

Determine `mode` — it selects which field-clusters dominate and which method passes run.
The three modes barely overlap in required fields; a single flat approach cannot serve all.

| mode | when | examples | drafting spine |
|------|------|----------|----------------|
| `narrative-award` | judged on track record; char-limited prose + ranked outputs | UTS ECR, ARC DECRA track-record, NHMRC Investigator, ERC track-record | past achievements, defensibility, criterion-weighted prose |
| `prospective-project` | judged on a future project; budget + team + compliance | ARC DP/LP/FT, CRC-P, AEA, NHMRC Ideas, NSF, ERC B2, EF-ESP, big-tech | aims→methods→milestones→budget coherence; feasibility; impact pathway |
| `retroactive-impact` | judged on past impact delivered; no future plan | Optimism RetroPGF, Gitcoin | live-artifact links + attestations as scored evidence |

## Pipeline

```
A0  scheme intake → compliance_matrix.yaml   rulebook BEFORE drafting: eligibility, mandatory docs,
                                             limits, fonts, certs, internal deadlines, submission owner
A   form + guidelines → scheme.yaml (IR)     parse ANY scheme into the normalized type model
B   corpus → evidence-store.yaml             reusable across every application (see evidence-store.md)
B2  people/orgs/partners → entity-store      project mode: CI/PI/partner/subaward/commitments/approvals
C   fill: per field → select evidence + mode-aware method passes
D   render to native modality                paste-ready / docx write-back / AcroForm / honest degrade
E   review: checklist-driven contract        eligibility·compliance·evidence·consistency·budget-math·
                                             attachments·panel-fit·risk·portal dry-run
F   submission_plan.yaml                      owners, due dates, internal cutoffs, approvals, dependency graph
```

Each stage is specified in a `references/` file. Read the ones the task needs:

- **`references/type-model.md`** — the two-axis field model (widget × semantic-role + attributes). Load-bearing; read first.
- **`references/form-schema-ir.md`** — the `scheme.yaml` IR (Stage A) and the A0 compliance matrix.
- **`references/evidence-store.md`** — Stage B: building/hardening the reusable evidence store; the entity store (B2).
- **`references/method-passes.md`** — Stage C/E: the mode-aware drafting and review passes.
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
- **`research-profile-evidence-base`** *(planned graduation of Stage B)* — the evidence store is reusable for CVs, biosketches, promotion cases, nominations; built as a module here, graduates to its own skill.

## Generality boundary

Per project policy: abstract rules live in this skill; scheme-specific instances live in the
application's own project folder (e.g. `.../Grants/uts-ecr/`). The UTS ECR application is the
first worked instance and the `narrative-award` regression fixture — its content never enters
this skill.
