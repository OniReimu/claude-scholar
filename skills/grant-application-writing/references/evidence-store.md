# Evidence Store (Stage B) + Entity Store (Stage B2)

> Stage B turns the **messy material corpus** (CVs, prior applications, publication
> lists, funding letters, PDFs, screenshots) into a **normalized, queryable, reusable
> evidence base**, built **once per applicant** and amortized over every application.
> It carries exactly the metadata the Stage-C method passes consume: role, status,
> confidence, provenance, defensibility, eligibility-window fit.
>
> **This is a shared module.** It is designed to graduate into its own skill
> `research-profile-evidence-base` — see the graduation note at the end. Build it here,
> but keep it self-contained so a CV / biosketch / promotion case / nomination can reuse
> the same store without importing grant machinery.

## Why a store, not per-field lookup

Method passes (verb-tiering, number-defensibility, anti-double-counting, role/credit
discipline — see `method-passes.md`) all ask the same questions of every claim: *is it
true, is it current, is it mine, what role, does it fall inside the eligibility window,
can it survive reviewer probe?* Answering per-field re-derives the same facts and drifts
between fields. The store answers them **once**, records the provenance, and every field
draws from the same defended base — so cross-field consistency is structural, not hoped for.

## Core discipline (why v1 failed review)

v1 tagged each item with a single `defensibility` flag. A Codex architecture review
graded this a **FAIL**: it conflated three independent things — a patent's **legal
status** (filed vs granted), the **evidentiary certainty** of the number, and the
**authority of the source** that supplied it. A "filed" patent from a screenshot and a
"granted" patent from an official register cannot share one tag.

**Rule: never store a single `defensibility` field.** Split trust into orthogonal
dimensions, and derive `defensibility` from them.

| dimension | question it answers | example values |
|-----------|--------------------|----------------|
| `status` | legal / lifecycle state | `granted` `filed` · `published` `in-press` `in-review` · `awarded` `submitted` `not-funded` |
| `confidence` | how certain is the *value* | `high` `medium` `low` |
| `source_authority` | who vouches for it | `official-record` > `orcid` `scopus` `scholar` > `screenshot` > `self-reported` |
| `as_of` | when was it last observed | ISO date |
| `validity_window` | when does it stop being current | `{from, to}` or `perpetual` |
| `sensitivity` | disclosure constraint | `public` `internal` `embargoed` `confidential` |
| `use_permission` | may it appear in an application | `public` `internal` `embargoed` |
| `defensibility` | **DERIVED, not stored** | computed from the above (see below) |

`defensibility` = a function of the row: high `status` + high `confidence` +
`source_authority ∈ {official-record, orcid, scopus}` + inside `validity_window` +
`use_permission ≠ do-not-use` → **defensible**; degrade otherwise. Recomputed on read, so
a stale `as_of` or a downgraded source automatically lowers it. Never hand-write it.

## `evidence-store.yaml` schema

```yaml
# ── evidence-store.yaml — one per applicant, reused across all applications ──
applicant: "Jane Q. Researcher"
identity:                               # disambiguation anchors (see below)
  orcid: "0000-0002-1825-0097"
  scopus_author_id: "57200000000"
  known_name_variants: ["J. Q. Researcher", "J.Q. Researcher", "Q. Jane"]
  affiliation_history:
    - {org: "UTS", from: 2022-03, to: null}
    - {org: "CSIRO Data61", from: 2019-01, to: 2022-02}

metrics:
  h_index:        {value: 18, source_authority: scholar, as_of: 2026-07-01}
  i10_index:      {value: 27, source_authority: scholar, as_of: 2026-07-01}
  total_citations:{value: 2140, source_authority: scholar, as_of: 2026-07-01}
  # metrics are source-stamped: Scholar h-index ≠ Scopus h-index; never merge silently.

dates:
  phd_conferral: {value: 2019-04-12, source_authority: official-record,
                  provenance: "corpus/testamur.pdf", confidence: high}
  # feeds the eligibility-window `computed` gate in scheme.yaml (e.g. DECRA 5-yr rule).

publications:
  - id: pub-2024-unlearn
    title: "Certified Machine Unlearning under Distribution Shift"
    venue: "NeurIPS"
    year: 2024
    role: first                         # first | co-first | corresponding | supervising | co-author
    metrics: {citations: {value: 41, source_authority: scholar, as_of: 2026-07-01}}
    status: published                   # published | in-press | under-review
    confidence: high
    source_authority: orcid
    as_of: 2026-07-01
    validity_window: {from: 2024-12, to: perpetual}
    sensitivity: public
    use_permission: free
    provenance: "corpus/cv.pdf#p3 ; orcid:0000-0002-1825-0097"

funding:
  - id: grant-arc-dp23
    title: "Scalable Privacy-Preserving Data Systems"
    amount: {value: 480000, currency: AUD}
    role: CI                            # CI | PI | co-I  (NOT scholarship/stipend — see screenshot note)
    dates: {from: 2023-01, to: 2025-12}
    project_code: "DP230100000"
    status: awarded
    confidence: high
    source_authority: official-record
    as_of: 2026-06-15
    provenance: "corpus/arc-award-letter.pdf ; corpus/funding-screenshot.png (OCR)"

patents:
  - id: pat-unlearn-2025
    title: "Method for Verifiable Data Deletion in ML Pipelines"
    status: filed                       # granted | filed  — THIS DISTINCTION MATTERS UNDER PROBE
    number: "AU2025900000"
    role: co-inventor
    confidence: high
    source_authority: official-record
    as_of: 2026-05-01
    provenance: "corpus/ip-filing-receipt.pdf"
    # "filed" ≠ "granted": a reviewer will check. Verb-tiering must not upgrade this.

service:
  - {id: svc-neurips-ac-25, role: "Area Chair", org: "NeurIPS", year: 2025,
     status: confirmed, source_authority: official-record, provenance: "corpus/invite-email.eml"}

awards:
  - {id: awd-best-paper-24, title: "Best Paper Award", org: "KDD", year: 2024,
     status: awarded, confidence: high, source_authority: official-record,
     provenance: "corpus/kdd-cert.pdf"}
```

Every leaf item carries the six hardening fields (`status`, `confidence`,
`source_authority`, `as_of`, `validity_window`, `sensitivity`/`use_permission`) plus
`provenance`. Missing a field is a load-bearing signal, not a default — see
negative-evidence handling.

## Governance blocks (store-level, alongside the item lists)

```yaml
refresh_policy:
  metrics:      {ttl_days: 30,  warn_after_days: 45}    # citation counts drift fast
  publications: {ttl_days: 180}
  funding:      {ttl_days: 365}
  # on read past ttl → freshness warning; past warn_after → block use until re-verified.

source_precedence:                       # conflict-resolution ordering (high → low)
  default: [official-record, orcid, scopus, scholar, screenshot, self-reported]
  metrics: [scopus, scholar, orcid]      # per-field overrides allowed
  # when sources disagree, the higher-precedence value wins and the loser is recorded
  # in contradiction_records — never silently dropped.

contradiction_records:
  - field: "publications[pub-2024-unlearn].citations"
    values: [{v: 41, src: scholar}, {v: 33, src: scopus}]
    resolved_to: {v: 41, src: scholar, reason: "metrics precedence = scopus>scholar? NO — see note"}
    status: open                         # open | resolved | wont-fix
    note: "Scholar over-counts preprints; use Scopus 33 for conservative claims."

audit_log:
  - {ts: 2026-07-01T09:00Z, action: refresh, target: metrics.h_index,
     old: 17, new: 18, source_authority: scholar, actor: "ingest-run-2026-07-01"}
  - {ts: 2026-06-15T00:00Z, action: ingest, target: funding[grant-arc-dp23],
     source: "corpus/funding-screenshot.png", method: vision-ocr}

negative_evidence:                       # absence recorded explicitly, not left blank
  - {claim: "corresponding-author on pub-2023-x", status: unverified,
     reason: "CV lists it; ORCID does not expose author role; do NOT claim without confirmation"}
```

## Identity disambiguation (name ambiguity)

Author-name collisions silently poison metrics and publication lists (two "J. Zhang"
merged; a namesake's h-index imported). The `identity` block is the disambiguation anchor:
resolve every ingested item against `orcid` / `scopus_author_id` first, fall back to
`known_name_variants` **scoped by `affiliation_history` dates**. An item that matches on
name but not on ORCID/affiliation is quarantined with `confidence: low` and a
`negative_evidence` note — never auto-merged into the applicant's record.

## Conflict resolution (CV vs ORCID vs Scholar vs Scopus)

The four sources routinely disagree (CV inflates, Scholar over-counts preprints, Scopus
lags, ORCID is incomplete). Resolution is deterministic:

1. Apply `source_precedence` (per-field override, else `default`).
2. Higher-precedence value wins; the losing value is written to `contradiction_records`
   with `status: open` — **the discrepancy is preserved, not erased**.
3. If precedence can't decide (same tier, or the conservative choice is lower), prefer the
   **lower / more defensible** number and flag it for the number-defensibility pass.
4. Unresolved contradictions on a `criterion-scored` field surface as a review warning
   before drafting proceeds.

## Multi-modal ingest

The corpus is not text-only. Each modality has an ingest path that stamps
`source_authority` and writes an `audit_log` entry:

| modality | example | ingest path | authority stamp |
|----------|---------|-------------|-----------------|
| structured text PDF | CV, publication list | PDF text extract → parse | as-cited (usually `self-reported`) |
| **prior-application prose** | last year's ECR / DECRA narrative | extract → **reusable text spans**, tagged by criterion | `self-reported`, but pre-defended |
| official document | award letter, testamur, filing receipt | PDF extract | `official-record` |
| **screenshot / image** | **funding-record screenshot proving a project is CI, not a scholarship** | **vision / OCR** → transcribe → human-confirm | `screenshot` (until confirmed against an official-record) |

**Prior-application prose is first-class reusable material** — spans that already survived
a real submission are gold for the next application; tag them by rubric criterion so
Stage C can retrieve "the best defended sentence I've written about impact." **Screenshots
need vision/OCR**: a common, high-value case is a funding portal screenshot that
establishes an item's `role: CI` rather than a student scholarship — exactly the kind of
role distinction the role/credit-discipline pass must not get wrong. OCR'd values enter at
`source_authority: screenshot` and `confidence: medium` until confirmed.

## Entity Store (B2) — prospective-project mode

For `prospective-project` applications the store extends with **people and organizations**:
CIs, PIs, partners, subawardees — their roles, commitments, and institutional approvals.
This feeds the `team-partner` role and the `contribution-matrix` / `budget-matrix` widgets.

```yaml
# ── entity-store.yaml — people & orgs for a project application ──
people:
  - id: person-ci-jane
    name: "Jane Q. Researcher"
    role: CI                            # CI | PI | co-I | AI | PhD | RA
    fte_commitment: {value: 0.2, basis: "0.2 FTE across 3 yrs"}
    org: person-org-uts
    approvals: [{type: "Head-of-School sign-off", status: obtained, as_of: 2026-06-20}]
    profile_ref: "evidence-store.yaml#applicant"   # links back to the evidence store

organizations:
  - id: person-org-uts
    name: "University of Technology Sydney"
    role: lead                          # lead | partner | subaward
    approvals: [{type: "DVCR authorization", status: pending}]

partners:
  - id: partner-acme
    name: "ACME Analytics Pty Ltd"
    role: partner
    contributions:
      cash:   [{fy: 2026, value: 100000, currency: AUD, status: committed,
                provenance: "corpus/loi-acme.pdf", confidence: high}]
      in_kind:[{fy: 2026, value: 60000, currency: AUD, description: "engineer time",
                status: committed, source_authority: official-record}]
    approvals: [{type: "signed Letter of Support", status: obtained, as_of: 2026-06-25}]
    # cash + in-kind feed the contribution-matrix and the matched-funding `computed` gate.
```

Same hardening applies: partner contributions carry `status` (committed vs indicative),
`confidence`, `source_authority`, and `provenance` — a "committed" cash figure backed by a
signed LOI is defensible; an email promise is not.

## Graduation note

This module is deliberately grant-agnostic. It is scheduled to graduate into a standalone
skill, **`research-profile-evidence-base`**, reusable for CVs, biosketches (NSF SciENcv,
NHMRC), promotion cases, award nominations, and annual reviews — anywhere a defended,
provenance-tracked research profile is needed. For now it ships inside
`grant-application-writing` as a self-contained reference so it can be lifted out without
untangling grant-specific logic. Keep it that way: nothing in this file should assume a
grant is the consumer.
