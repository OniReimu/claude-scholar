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
`use_permission ∈ {public, internal}` (≠ `embargoed`) → **defensible**; degrade otherwise. Recomputed on read, so
a stale `as_of` or a downgraded source automatically lowers it. Never hand-write it. One
extra input for track-record items: a claim written as a **superlative** ("first / only /
leading / largest") with no `attributor` derives to a **lower tier** — statable, but not as a
superlative (see ROPE + sourced-eminence fields below).

## `evidence-store.yaml` schema

```yaml
# ── evidence-store.yaml — one per applicant, reused across all applications ──
owner:                                   # top-level owner + disambiguation anchors (see below)
  name: "Jane Q. Researcher"
  orcid: "0000-0002-1825-0097"
  scopus_author_id: "57200000000"
  known_name_variants: ["J. Q. Researcher", "J.Q. Researcher", "Q. Jane"]
  affiliation_history:
    - {org: "Northbridge University", from: 2022-03, to: null}
    - {org: "Meridian Research Institute", from: 2019-01, to: 2022-02}
  as_of: 2026-07-01                      # store-wide last full refresh

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
    window: since-PhD                   # ROPE: career-relative bound a count is stated against — since-PhD | last-5y | since-2019
    attributor: null                    # external validator for a superlative (ranking body / flagship venue / evidence id); null ⇒ statable but NOT as a superlative
    metrics: {citations: {value: 41, source_authority: scholar, as_of: 2026-07-01}}
    status: published                   # published | in-press | in-review
    confidence: high
    source_authority: orcid
    as_of: 2026-07-01
    validity_window: {from: 2024-12, to: perpetual}
    sensitivity: public
    use_permission: public              # public | internal | embargoed
    provenance: "corpus/cv.pdf#p3 ; orcid:0000-0002-1825-0097"

funding:
  - id: grant-arc-dp23
    title: "Scalable Privacy-Preserving Data Systems"
    amount: {value: 480000, currency: AUD}
    role: CI                            # CI | PI | co-I  (NOT scholarship/stipend — see screenshot note)
    window: since-PhD                   # ROPE bound for grant tallies (e.g. "grants won since PhD")
    attributor: null                    # scheme / award name or evidence id — required to state a superlative ("largest…")
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
  - {id: svc-flagship-ac-25, role: chair,          # role: chair | editor | committee-member
     window: last-5y, attributor: null,            # attributor: named body/size — required for "largest committee of…"
     org: "NeurIPS", year: 2025, status: confirmed,
     source_authority: official-record, provenance: "corpus/invite-email.eml"}

supervision:
  - id: sup-phd-alpha
    student: "<anonymised student id>"
    level: PhD                          # PhD | MPhil | Masters | Honours | postdoc
    role: principal                     # principal | co — the supervisory role the count is claimed under
    window: since-PhD                   # ROPE bound for "students supervised to completion" tallies
    attributor: null                    # graduation record / evidence id — required to state a superlative
    status: completed                   # ongoing | completed | withdrawn   ("graduated" ≠ "enrolled")
    confidence: high
    source_authority: official-record
    as_of: 2026-07-01
    validity_window: {from: 2023-06, to: perpetual}
    sensitivity: internal
    use_permission: internal
    provenance: "corpus/graduation-record.pdf"

awards:
  - {id: awd-best-paper-24, title: "Best Paper Award", org: "KDD", year: 2024,
     role: sole, window: since-PhD,                # role: sole | co-recipient
     attributor: "KDD Best Paper committee",       # the conferring body IS the attributor that sources the superlative
     status: awarded, confidence: high, source_authority: official-record,
     provenance: "corpus/kdd-cert.pdf"}

impact:
  - id: imp-toolkit-adoption
    claim: "open-source toolkit adopted by external deployments"
    kind: adoption                      # adoption | policy | deployment | commercial | community
    role: lead                          # lead | contributor
    window: last-5y                     # ROPE bound for reach tallies
    attributor: "12,000 installs (package registry)"  # units/adoption figure or evidence id that sources the eminence claim; null ⇒ no superlative
    metric: {value: 12000, unit: installs}            # optional quantified reach
    status: realised                    # realised | in-progress
    confidence: medium
    source_authority: self-reported
    as_of: 2026-07-01
    validity_window: {from: 2022-01, to: perpetual}
    sensitivity: public
    use_permission: public
    provenance: "corpus/registry-stats.png (OCR)"

# ── Outputs context — ROPE FIELD-CALIBRATION (narrative-award mode) — teach the assessor the field's esteem terms ──
# Rendered by the drafting layer, never improvised. Read by method-passes.md §1.10 + §1.5; validated by validate_ir outputs-context-completeness.
outputs_context:
  field_norms:
    venue_tiers:                           # every venue → tier + PLAIN-LANGUAGE rank (gloss for a generalist assessor)
      - {venue: "NeurIPS",         tier: "CORE A*", plain_rank: "top-3 venue in the field"}
      - {venue: "<Field Journal>", tier: "JCR Q1",  plain_rank: "leading journal in the subfield"}
    authorship_convention:                 # DECODE what author position MEANS in THIS subfield (bounded credit, NOT a role upgrade) — defuses "why not first author?"
      - pattern: "last/second author on a co-supervised student paper"
        meaning: "candidate contributed the main idea, design, and writing; first author ran the experiments"
        applies_to: [pub-2024-unlearn]
    ranking_attributor:                    # sourced-eminence via an external ranking SERVICE (the service IS the attributor, not a self-assertion)
      {claim: "ranked Nth in the field in <country>", service: "<a ranking service>", as_of: 2026-06-01}
  clusters:                                # group outputs into ~3-5 NAMED research threads
    - {thread: "certified machine unlearning", outputs: [pub-2024-unlearn], venues: ["NeurIPS"],
       primacy: {claim: "first certified-unlearning method under distribution shift", attributor: pub-2024-unlearn}}
    - {thread: "privacy-preserving data systems", outputs: [pub-2023-x], venues: ["<Field Journal>"],
       primacy: {claim: "milestone in a tightly-scoped area", attributor: null}}   # null ⇒ statable, but NOT written as a superlative
  career_best:
    label_scheme: {best: "[*]", journal: "[J*]", conference: "[C*]"}   # stable ids linking the listing ↔ the context narrative
    ids: [pub-2024-unlearn]                # every id here MUST appear in >=1 clusters[].outputs (completeness discipline)
  # denominator renders the denominator rule (§1.10): high-tier out of ALL outputs, never a bare count.
  contribution_summary: {significant_conceptual: "4 of 6 papers", basis: "lead conceptual contributor: idea + design + writing", denominator: {high_tier: 4, all: 6}}   # bounded credit, not "all mine"

# ── Multi-CI investigator model — one row per named investigator, person-indexed ──
# Single-applicant schemes: the `owner` block above IS the lead-CI shorthand (owner == investigators[0]);
# investigators[] may be omitted. Multi-CI schemes populate one row per CI/PI/partner. ROPE
# window/role/attributor apply PER investigator, each relative to their OWN opportunity —
# never pool a senior + an ECR into one tally. Read by the multi-CI ROPE pass (method-passes.md §2.7).
investigators:
  - id: inv-lead
    name: "Jane Q. Researcher"             # owner == first investigator for single-applicant schemes
    role: lead-CI                          # lead-CI | CI | PI | partner-investigator
    rope_context:                          # this person's opportunity envelope — read ROPE relative to THIS, not the team
      years_since_phd: 7
      career_stage: mid-career             # e.g. ECR | mid-career | senior
      interruptions: [{from: 2021-03, to: 2021-11, fte_fraction: 0.4, reason: "parental leave"}]
    track_record_ref:                      # THIS person's own items; the referenced rows' window/role/attributor are read relative to THIS investigator
      publications: [pub-2024-unlearn]
      funding: [grant-arc-dp23]
      awards: [awd-best-paper-24]
    task_ownership: [aim-2, wp-3]          # aim / WP ids this investigator leads
    distinctive_contribution: "certified-unlearning method design; leads WP3 evaluation"
    fte: {value: 0.2, basis: "0.2 FTE across 3 yrs"}
    current_commitments:                   # concurrent awards drawing on this person's time (availability check)
      - {award: grant-arc-dp23, fte: 0.15}

# ── SOTA / significance evidence classes — render significance FROM evidence, not assertion ──
comparators:                               # external state-of-the-art / prior-art the project is positioned against
  - ref: "Doe et al. 2025, 'Baseline Unlearning', NeurIPS"   # the comparator work / product / standard
    kind: scholarly                        # scholarly | commercial | standard | own-work — own-work is NOT an external comparator
    provenance: "corpus/related-work.pdf ; doi:10.1234/xxxxx"

context_evidence:                          # source-backed problem-significance (a dated stat, not "an important problem")
  - claim: "data-deletion requests are a growing compliance burden"
    stat: "1.2M erasure requests/yr across surveyed operators"
    source: "Regulator Annual Report 2025, table 4"
    as_of: 2025-11
```

Every leaf item carries the six hardening fields (`status`, `confidence`,
`source_authority`, `as_of`, `validity_window`, `sensitivity`/`use_permission`) plus
`provenance`. Track-record items additionally carry the three ROPE + sourced-eminence
fields (`window`, `role`, `attributor`; see next section). Missing a field is a
load-bearing signal, not a default — see negative-evidence handling.

## ROPE + sourced-eminence fields (track-record items)

Beyond the trust profile, every track-record item (publication, funding, award, service,
supervision, impact) may carry three fields that turn a raw count into a *defensible* one —
the two moves a fellowship application lives on:

| field | what it fixes | example values |
|-------|---------------|----------------|
| `window` | the career-relative bound a count is stated against — makes every count **relative-to-opportunity (ROPE)** | `since-PhD` · `last-5y` · `since-2019` |
| `role` | the role a count is claimed under (never upgraded) | pubs `first`/`corresponding` · funding `CI`/`PI`/`co-I` · supervision `principal`/`co` · service `chair`/`editor`/`committee-member` · awards `sole`/`co-recipient` · impact `lead`/`contributor` |
| `attributor` | the external validator that makes a superlative **sourced** — a ranking body, award name, named flagship venue, a units-sold/adoption figure, or an evidence-store id | `"[ranking body]"` · `"12,000 installs"` · `pub-2024-unlearn` |

The drafting layer **renders** "sourced eminence" and "relative-to-opportunity" prose *from*
these fields — it never improvises either. A count is only written with its `window` and
`role` attached (*"[N] papers in flagship venues since the PhD, as corresponding author"*),
and a superlative is only written when an `attributor` is present (*"the field's leading
early-career researcher, as named by [ranking body]"*). **A superlative with no `attributor`
derives to a lower `defensibility` tier** — the item may be stated, but not as a superlative.

These fields feed **prong 2 (the fellowship prong) of the `number-defensibility` pass** in
`method-passes.md` and the "sourced eminence" / ROPE / defensible-primacy moves of
`author-voice.md` §8. They are inert in `prospective-project` mode, where the defensible move
on a market number is *omit*, not *scope + source* (prong 1).

## Outputs-context — field-calibration layer (`outputs_context`, narrative-award mode)

A narrative-award panel mixes generalists with specialists, so a raw outputs listing
under-reads: a generalist cannot see that a venue is top-3, cannot decode what a last-author
slot means in this subfield, and cannot weight a "first" without a source. The
`outputs_context` block is the **field-calibration layer** — it teaches the assessor the
field's own esteem terms so the drafting layer can *render* them, never improvise. It is gated
on `mode == narrative-award` and inert otherwise.

Five registers:

- `field_norms.venue_tiers` — every venue the listing leans on gets a `tier` **and** a
  `plain_rank` gloss (*"CORE A*, top-3 venue in the field"*), so a generalist reads the same
  signal a specialist does.
- `field_norms.authorship_convention` — DECODE what an author position means in THIS subfield
  (*last/second author on a co-supervised student paper = the candidate supplied idea, design,
  and writing*), pre-empting the "why not first author?" reflex. **A decode is bounded credit,
  not a role upgrade** — it states what the position already means, evidence-backed, and never
  promotes the item's `role`. It ties to the role/credit-discipline pass (`method-passes.md`
  §1.5): where the field's convention differs from "first author = most credit", decode it for
  the assessor from `authorship_convention` instead of silently capping — a decode is not an
  upgrade.
- `field_norms.ranking_attributor` — a sourced-eminence claim via an external ranking
  **service** (*"ranked Nth in the field, per a ranking service, as of <date>"*). The service
  IS the `attributor`; same discipline as any superlative — no service, no ranking claim.
- `clusters` — group the outputs into ~3-5 NAMED research threads, each with a scoped
  `primacy: {claim, attributor}`. **A cluster `primacy` with no `attributor` (`null`) is
  statable but NOT as a superlative** — identical discipline to the item-level `attributor`
  field above (see ROPE + sourced-eminence). Only a sourced primacy is written as a "first".
- `career_best` — a bounded subset flagged with a stable `label_scheme` (`[*]`/`[J*]`/`[C*]`)
  linking the listing to the context narrative, plus the `ids`. **Completeness discipline:
  every `career_best.ids` entry MUST appear in ≥1 `clusters[].outputs`** (a career-best output
  with no thread and no tier is an unplaced claim). `contribution_summary` states bounded
  credit across the corpus (*"significant conceptual contribution on M of N papers"* + how
  counted), never "all of it is mine". `contribution_summary.denominator` (`{high_tier: M, all: N}`)
  renders the **denominator rule** (§1.10): a strength is stated as *high-tier out of ALL outputs*
  (`M of N`), never a bare count — the denominator is what makes the fraction defensible.

Consumed by the **outputs-context / field-calibration pass (`method-passes.md` §1.10)** and
rendered by `author-voice.md` §10. `validate_ir.py` `outputs-context-completeness` enforces the
two invariants: every career-best id is clustered, and every cluster `primacy.claim` carries an
`attributor` (else the superlative is unsourced) — FAIL in `--mode submission` / WARN in draft.

## Multi-CI investigator model (`investigators[]`)

The single `owner` block models one applicant. Multi-CI schemes (team, linkage, program
grants) score **each named investigator** against the tasks they own, so the store carries a
person-indexed `investigators[]`. The `owner` block stays the **lead-CI shorthand**: for a
single-applicant scheme `owner == investigators[0]` and `investigators[]` may be omitted; a
multi-CI scheme populates one row per `lead-CI | CI | PI | partner-investigator`.

Each investigator carries their **own** `rope_context` (years since PhD, career stage,
interruptions) and their **own** `track_record_ref` into the shared item lists. The
load-bearing rule: **ROPE `window` / `role` / `attributor` apply per investigator, each
relative to their OWN opportunity.** A senior CI's long record and an ECR's short one are
never pooled into a single team tally — each is read against that person's `rope_context`.
`task_ownership` maps each investigator to the aims / WPs they lead (so no aim is unstaffed
and no two CIs redundantly duplicate one); `fte` + `current_commitments` feed the
availability check (declared FTE vs concurrent awards). The drafting layer's **multi-CI ROPE
pass (`method-passes.md` §2.7)** reads this block to assess the team as a *composition*, not
a bag of CVs.

## SOTA & significance evidence classes (`comparators[]`, `context_evidence[]`)

Significance and state-of-the-art are **rendered from evidence, not asserted.** Two classes
back them:

- `comparators[]` — the external work the project is positioned against, each tagged
  `kind: scholarly | commercial | standard | own-work`. **`own-work` is explicitly NOT an
  external comparator**: the applicant's own prior work grounds a *primacy* claim but cannot
  stand in for independent state-of-the-art. If every comparator is `own-work`, the SOTA pass
  has only self-reference and flags it.
- `context_evidence[]` — source-backed problem-significance: a `{claim, stat, source,
  as_of}` quad giving a real dated figure for why the problem matters
  (*"1.2M erasure requests/yr, Regulator Annual Report 2025"*), never "an important problem."

These feed the **SOTA / significance pass (`method-passes.md` §2.11)**, the
**defensible-primacy move (§1.9)**, and **author-voice §5.1 costed-stake** — so the drafting
layer renders significance from a dated source, and writes a "first" / "only" only against a
real external comparator.

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
merged; a namesake's h-index imported). The top-level `owner` block is the disambiguation
anchor: resolve every ingested item against `owner.orcid` / `owner.scopus_author_id` first,
fall back to `owner.known_name_variants` **scoped by `owner.affiliation_history` dates**. An item that matches on
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
    org: person-org-lead
    approvals: [{type: "Head-of-School sign-off", status: obtained, as_of: 2026-06-20}]
    profile_ref: "evidence-store.yaml#owner"       # links back to the evidence store

organizations:
  - id: person-org-lead
    name: "Northbridge University"
    role: lead                          # lead | partner | subaward
    approvals: [{type: "DVCR authorization", status: pending}]
    institutional_support:              # the HOST-INSTITUTION STATEMENT's committed co-investment (3rd-party attestation; analog to partner contributions)
      items:                            # each committed line — hardened like partner contributions (status + provenance)
        - {kind: establishment-grant, value: 150000, currency: AUD, status: committed, provenance: "corpus/host-statement.pdf"}
        - {kind: stipend-topup,       value: 60000,  currency: AUD, status: committed, provenance: "corpus/host-statement.pdf"}
        - {kind: salary-shortfall,    value: 90000,  currency: AUD, status: committed, provenance: "corpus/host-statement.pdf"}
        - {kind: teaching-relief,     value: null, basis: "reduced teaching load funded from salary savings", status: committed, provenance: "corpus/host-statement.pdf"}
      total: {value: 300000, currency: AUD}   # the statement's STATED total — kept SEPARATE from sum(items) so a mismatch is VISIBLE (mirror partner letter_commitment vs contributions)
      protected_capacity:                 # capacity-conversion: relief/coverage → protected research time + execution capacity, not just $
        teaching_relief_to_research_days: 60   # teaching-relief converted to ~60 protected research days/yr; null if unquantified
        continuing_position: true              # a continuing (not fixed-term) position on success → sustained execution capacity
        note: "salary-shortfall cover + teaching relief free ~60 days/yr for project execution"
      strategic_fit: "host-strategy ↔ candidate ↔ project: the org's data-systems priority ↔ the candidate recruited to accelerate it ↔ the project delivers its ingestion-latency milestone; project finds a home at the host research centre"
      continuing_offer: "continuing position on success"   # or null
      statement_provenance: "corpus/host-statement.pdf"

partners:
  - id: partner-acme
    name: "ACME Analytics Pty Ltd"
    role: partner
    legal_entity:                       # which entity SIGNS vs OPERATES — a partnership is a legal relationship, not a name
      signing_entity:   "ACME Group Holdings Ltd"   # who signs the letter / agreement (has the authority to commit)
      operating_entity: "ACME Analytics Pty Ltd"    # who actually does the work / hosts the platform or data access
      relationship: parent-subsidiary   # parent-subsidiary | jv | division | branch | same | consortium-member
      jurisdiction: "AU"                # ISO country of the COMMITTING entity — feeds the eligibility gate (offshore-partner / national-interest)
      registration: {type: ABN, id: "00 000 000 000", source_authority: official-record}
      capacity_evidence:                # does the signer have authority + resources to commit?
        - {claim: "signing authority", provenance: "corpus/letter-signatory-title.pdf", confidence: high}
      flow_note: "offshore parent commits cash; disbursed via onshore operating_entity"  # or null
    contributions:                      # the APPLICATION's figures (what the budget/narrative claims)
      cash:   [{fy: 2026, value: 100000, currency: AUD, status: committed,
                provenance: "corpus/loi-acme.pdf", confidence: high}]
      in_kind:[{fy: 2026, value: 60000, currency: AUD, description: "engineer time",
                status: committed, source_authority: official-record}]
    letter_commitment:                  # what the SUPPORT LETTER literally states — kept SEPARATE from contributions so #8 can see a mismatch
      cash:    {value: 100000, currency: AUD, conditional: false}
      in_kind: {value: 60000,  currency: AUD, conditional: false, description: "engineer time"}
      role_stated: "co-investigator"    # the role the letter asserts — reconciled vs the claimed role
      personnel: ["<named contact>"]    # people the letter names as committed — must appear in the team table
      provenance: "corpus/loi-acme.pdf"
      as_of: 2026-06-25
    approvals: [{type: "signed Letter of Support", status: obtained, as_of: 2026-06-25}]
    # contributions feed the contribution-matrix + matched-funding `computed` gate;
    # letter_commitment is reconciled against them by §2.13 / validate_ir partner-commitment-reconciliation.

collaborators:                          # specialist reach WITHOUT displacing applicant ownership (§10.2 independence-plus-network)
  - id: collab-1
    name: "Dana Q. Specialist"
    organization: "ACME University"
    prior_relationship: "co-authored one workshop paper two years ago"
    task_expertise: "formal-verification techniques for the backpressure controller in out-2"
    engagement_mode: method-consult      # advisory | data-access | co-supervision | method-consult | letter-only
    applicant_independence: "applicant designs and leads the ingestion architecture and all WP2 evaluation independently"
    provenance: "corpus/collab-letter.pdf"

mentors:                                # differentiated by COMPETENCY GAP, not eminence (§10.3 mentor-by-competency-gap)
  - id: mentor-1
    name: "Robin Q. Mentor"
    organization: "Northbridge University"
    competency: "industry-translation and impact pathways — a gap in the applicant's academic-only record"
    function: "quarterly review of the benefits-realisation plan and partner-facing translation coaching"
    prior_relationship: null
    provenance: "corpus/mentor-statement.pdf"
```

Same hardening applies: partner contributions carry `status` (committed vs indicative),
`confidence`, `source_authority`, and `provenance` — a "committed" cash figure backed by a
signed LOI is defensible; an email promise is not.

## Partner legal-entity model (`partners[].legal_entity`)

A claimed partnership is a **legal relationship between entities**, not a group name. In
project / linkage mode the partner is the credibility anchor and a reviewer probes it
hardest — so each `partners[]` row makes the legal entity explicit rather than leaving
"ACME" to mean whatever the reader assumes.

- **Signing vs operating entity.** `signing_entity` is who has the authority to *commit*
  (signs the letter / agreement); `operating_entity` is who actually *does the work* or
  hosts the platform / data / site access. These are often different corporate persons —
  a parent signs, a subsidiary operates. Recording both stops a letter signed by
  "ACME Group Holdings Ltd" from being silently read as a commitment by the working entity
  "ACME Analytics Pty Ltd", and lets `capacity_evidence` test that the signer actually held
  the authority it exercised (signatory title from an official record, not an assumption).
- **Relationship types.** `relationship ∈ {parent-subsidiary | jv | division | branch |
  same | consortium-member}` names how signing and operating entities relate. `same` is the
  simple case (one entity signs and operates); the others flag that a commitment may cross a
  corporate boundary and needs a `flow_note` (e.g. an offshore parent commits cash that must
  be disbursed via an onshore operating subsidiary — the money and the eligibility do not sit
  on the same entity).
- **`jurisdiction` feeds the eligibility gate.** `jurisdiction` is the ISO country of the
  **committing** entity (the one whose resources are pledged), not merely where the work
  happens. Many schemes require an eligible domestic legal entity, or trigger
  offshore-partner / national-interest checks when the committing party is foreign. That gate
  reads `legal_entity.jurisdiction`, so an offshore parent committing cash surfaces as an
  eligibility question even when the operating subsidiary is onshore — exactly the case a
  `flow_note` documents.

### Why `letter_commitment` is kept separate from `contributions`

`contributions.cash|in_kind` are the **application's** figures (what the budget and narrative
claim); `letter_commitment` is the **letter's** figures (what the support letter literally
states, plus the role and personnel it names, and whether each figure is `conditional`).
They are deliberately **not** unified — same discipline as `source_precedence` /
`contradiction_records`: two sources that can disagree are stored side by side so a mismatch
is **visible**, not silently averaged into one number. If the letter says AUD 100k cash but
the budget line claims 120k, or the letter makes the cash `conditional: true` ("subject to
board approval") while the application renders it as unconditional `committed`, that gap is a
finding — not something the store papers over.

Consumers of these two blocks:

- **`method-passes.md` §2.12 partnership-authenticity** reads `partners[]` (legal entity,
  contributions, provenance) to distinguish a genuine co-design partner from
  fee-for-service or letterhead-only support.
- **`method-passes.md` §2.13 partner-commitment reconciliation** reconciles
  `letter_commitment` against `contributions`, the contribution-matrix line, and the
  narrative — figure, role, personnel, and conditionality must all agree.
- **`validate_ir.py` `partner-commitment-reconciliation` check** does the mechanical
  cross-check: any numeric mismatch, or a `letter_commitment.conditional: true` rendered as
  a `committed` contribution, FAILs in submission mode / WARNs in draft. A partner with a
  cash line but no `letter_commitment` and no `provenance` is UNVERIFIED and fails closed.

Same hardening as the rest of the store: a `letter_commitment` figure is only trustworthy
with a signed-LOI `provenance` (`corpus/loi-acme.pdf`); an email promise does not qualify a
figure as `committed`, and a signature by an entity with no `capacity_evidence` for signing
authority degrades the commitment rather than upgrading it.

## Host-institution statement (`organizations[].institutional_support`)

The host-institution statement is a **third-party attestation** — the administering
organisation's own committed co-investment in the fellowship — carried on the lead
`organizations[]` row and hardened **exactly like partner contributions**: every item carries
`status` and `provenance`, and a "committed" figure is only defensible with the statement
backing it (`corpus/host-statement.pdf`), never an assumed intention.

- `items[]` — each `{kind, value, status, provenance}` line of committed support
  (`establishment-grant`, `stipend-topup`, `salary-shortfall`, `teaching-relief`, …). A
  non-cash line (e.g. `teaching-relief`) may carry `value: null` with a `basis` instead of a
  figure — the relief is real but not a dollar the budget can double-count.
- `total` — the statement's **STATED** total, kept **deliberately SEPARATE from `sum(items)`**
  — the same discipline as partner `letter_commitment` vs `contributions` (batch 2): two
  numbers that can disagree are stored side by side so a mismatch is **VISIBLE**, not silently
  reconciled into one. If the statement's headline total says AUD 300k but the itemised lines
  sum to 260k, that gap is a finding — not something the store papers over.
- `protected_capacity` — **capacity-conversion**: the reconciliation must not stop at the dollar.
  `teaching_relief_to_research_days` (int, or `null` if committed-but-unquantified) and
  `continuing_position` (bool) convert teaching-relief / salary-cover / a continuing offer into
  **protected research time + execution capacity** — the thing that actually lets the project run,
  not just a co-investment figure. Read by the institutional-reconciliation pass
  (`method-passes.md` §4.5).
- `strategic_fit` — states the **host-strategy ↔ candidate ↔ project triangle** (all three legs:
  the org's strategy, why it recruited the candidate, and where this project advances it / finds a
  home), not a one-sided "good fit." `continuing_offer` is the institution's continuing-position
  commitment on success (or `null`).
- `statement_provenance` — the source document for the statement as a whole.

Consumed by the **institutional-statement reconciliation pass (`method-passes.md` §4.5**; analog
to §2.13 partner reconciliation): the stated `total` must reconcile with `sum(items)` AND — if a
budget is present — with the budget's non-ARC / institutional-contribution lines; every
`committed` item must have `provenance`. Mechanized by `validate_ir.py`
`institutional-support-reconciliation`: a `total` ≠ `sum(items)` mismatch (>1%), or a
`committed` item with no `provenance`, FAILs in `--mode submission` / WARNs in draft — exactly
the fail-closed behaviour of `partner-commitment-reconciliation`.

## Collaborators & mentors (`collaborators[]`, `mentors[]`)

Collaborators and mentors are **people**, so they live in the entity store (B2) alongside
`people[]` / `partners[]` — but they are fielded to **defuse name-dropping**, not to list eminent
names. Both feed the collaborator/mentor moves of `author-voice.md` (§10.2 / §10.3).

- `collaborators[]` — **independence-plus-network**: a collaborator extends *specialist reach*
  while the applicant keeps *intellectual ownership*. Each row carries `prior_relationship` (how
  the applicant actually knows them), `task_expertise` (the specific capability they add, tied to a
  spine `task`/`output` id), `engagement_mode` (`advisory | data-access | co-supervision |
  method-consult | letter-only`), and — load-bearing — `applicant_independence` (what the applicant
  leads independently *of* this collaborator). A collaborator with no `task_expertise` +
  `applicant_independence` is decoration, and the pass flags it.
- `mentors[]` — **mentor-by-competency-gap**: each mentor addresses a **distinct capability gap**
  with a **differentiated `function`**, not an interchangeable eminent name. `competency` names the
  specific gap (never "general guidance") and `function` names the differentiated role (e.g.
  quarterly method review of `task-2`; industry-translation coaching). Two mentors with the same
  `competency` and `function` are redundant name-dropping.

## Project-plan store (B3) — prospective-project mode

The evidence store (Stage B) and the entity store (B2) are **reusable applicant assets** —
built once, amortized across every application. The **project-plan store** is the opposite:
it is **PROJECT-SPECIFIC**, one `project-plan.yaml` per application, living in that
application's own folder. It does not hold track record or people/orgs — it holds **the
project itself**, so that a demanding `prospective-project` panel scores substance *rendered
from structure*, not asserted in prose. Template: `templates/project-plan.template.yaml`;
validated by `validate_ir.py --plan`; consumed ONLY in `prospective-project` mode.

Five top-level registers carry the four things such a panel actually stress-tests:

- `aims[]` + `design[]` — each aim (`{id, statement, success_criterion}`) is answered by a
  design row (`{aim, methods, controls, validity{sample_size, power, threats}, answers_aim}`)
  that can **produce** the claimed knowledge — aim↔method coverage, controls/comparators,
  empirical validity where the aim is empirical, and an explicit `answers_aim` justification.
- `benefits[]` — each `{id, benefit, type, beneficiary, owner, timing, metric, preconditions}`
  is realisable, measurable, and **owned**; `type` distinguishes output / outcome / impact.
- `additionality{}` — `{counterfactual, not_business_as_usual, leverage{grant,
  co_contribution, currency}, cost_per_outcome{value, basis}}`; leverage = co-contribution /
  grant is the value-for-money ratio, and the partner co-investment from B2 IS this leverage
  story (recomputed to cross-check the budget totals).
- `risks[]` — each `{id, risk, likelihood, impact, trigger, monitoring, contingency, owner}`
  is a live **trigger→contingency** register: "if X by month M → do Y", not "there is a risk;
  it is mitigated."

Consumers — the four `prospective-project`-mode Group-2 passes, each mechanized by a
fail-closed `validate_ir.py` check:

- **§2.14 research-design adequacy** (deepens §2.3 methods-feasibility) — reads `aims[]` +
  `design[]`. Mechanized by the **`research-design-adequacy`** check: every aim id appears in
  ≥1 `design[].aim`, every aim has a non-empty `success_criterion`, every covered design has a
  non-empty `answers_aim`.
- **§2.15 benefits-realisation** (deepens §2.5 impact-pathway) — reads `benefits[]`. Mechanized
  by the **`benefits-realisation`** check: every benefit has a non-empty `owner` AND `metric`
  AND `timing`.
- **§2.16 additionality / value-for-money** (ties to partnership §2.12 + budget §2.8) — reads
  `additionality{}`. Mechanized by the **`additionality-vfm`** check: non-empty `counterfactual`,
  `leverage.grant` / `leverage.co_contribution` present (reports the ratio, cross-checks a
  supplied `--budget`).
- **§2.17 trigger-driven risk** (deepens §2.4 risk-mitigation) — reads `risks[]`. Mechanized by
  the **`risk-triggers`** check: every high-impact risk has a non-empty `trigger` AND
  `contingency` AND `owner`.

Same hardening spirit as the rest of the store: **a design/benefit/risk row is only as strong
as its `answers_aim` / `owner`+`metric` / `trigger`+`contingency`.** An empty register field is
a **load-bearing gap, not a default** — a benefit with no owner is aspirational, a high-impact
risk with no trigger is a wish, an aim with no `success_criterion` cannot be scored. In
`--mode submission` these fail closed (BLOCK); in `--mode draft` they WARN. The store never
green-washes a present-but-empty field into a pass.

### Traceability spine (`objectives[]` / `tasks[]` / `outputs[]` / `validations[]`)

The five registers above answer *is each aim/benefit/risk sound in isolation*. They do **not**
guarantee the project hangs together — that every activity has an owner, a year, a budget line,
and a purpose it serves. The **traceability spine** adds that: four more registers that **EXTEND**
(never replace) the five above, giving every activity a **stable id** and forcing it to trace
**UP** to a purpose (`aim → objective → task → subtask → output → benefit`) and **ACROSS** to a
resource (`person → year → budget → evidence`). Encoding the spine turns a family of "stylistic"
consistency checks into a **deterministic, fail-closed validator** (`validate_ir.py`
`traceability-spine`, gated on `mode == prospective-project`).

- `objectives[]` — `{id, aim, statement}`: each numbered objective is the condition an aim is met
  by. `aim` resolves to an `aims[].id` (no objective without an aim).
- `tasks[]` — `{id, objective, statement, foundational, depends_on[], subtasks[], person[],
  years[], budget_lines[], validation}`. **One-to-one objective↔task** where possible; `objective`
  resolves to an `objectives[].id`. **Foundation-first**: exactly the task(s) that supply shared
  theory / data / tools / measures carry `foundational: true`, and later tasks name them in
  `depends_on` (the dependency architecture is explicit, not implied). Each `subtasks[]` entry is
  written in the **gap→consequence→move→mechanism** grammar (*current method lacks X → causes
  failure Y → introduce Z → Z fixes Y*; see method-passes.md §2.19) and names the `output` it
  produces. **Cross-axis**: `person` resolves to entity-store `investigators[].id` (capability
  coverage — no unstaffed task), `years` to the timetable, `budget_lines` to budget rows (the
  **four-way crosswalk** task↔person↔year↔budget), `validation` to a `validations[].id`.
- `outputs[]` — `{id, task, kind, benefit}`: each output is produced by a `task` and (where it
  ladders to impact) traces to a `benefits[].id`. `kind ∈ theory | method | tool | demonstrator |
  publication`.
- `validations[]` — `{id, task, baseline, stress, mechanism_check, metric, comparator_class}`: a
  **colocated** validation block per task (not a late generic "evaluation" section). It names
  competitive `baseline`s, a `stress` scenario **designed to expose the targeted failure**, a
  component-level `mechanism_check` (does the mechanism behave as intended), a `metric` aligned to
  that failure (not a generic success metric), and a `comparator_class ∈ scholarly | commercial |
  standard | own-work` (reusing the batch-1 comparators discipline; `own-work` is not external
  SOTA).

Worked example (**referentially consistent** — every id resolves: `objectives[].aim` → an `aims[]`
row, `tasks[].objective` → an `objectives[]` row, `subtasks[].output` → an `outputs[]` row,
`tasks[].person` → an entity-store `investigators[].id`, `outputs[].benefit` → a `benefits[]` row):

```yaml
# aims[] carries aim-1, aim-2; benefits[] carries ben-1, ben-2 (the five registers above)
objectives:
  - {id: obj-2, aim: aim-2, statement: "Deliver an ingestion path that sustains p99 < 800 ms at 10x throughput"}
tasks:
  - id: task-1                       # foundational — supplies the shared measures task-2 consumes
    objective: obj-1
    foundational: true
    depends_on: []
    subtasks: [{id: st-1.1, statement: "current method lacks a latency-attribution model → root causes are guessed → introduce a per-stage cost model → localises the bottleneck", output: out-1}]
    person: [inv-lead]               # → evidence-store investigators[].id
    years: [1]
    budget_lines: [bl-1]
    validation: val-1
  - id: task-2
    objective: obj-2                 # → objectives[].id
    foundational: false
    depends_on: [task-1]             # dependency architecture: consumes task-1's cost model
    subtasks: [{id: st-2.1, statement: "batch ingest lacks backpressure → tail latency spikes under burst → introduce adaptive batching → bounds p99 at 10x", output: out-2}]
    person: [inv-lead, inv-ci-2]
    years: [2, 3]
    budget_lines: [bl-2]
    validation: val-2
outputs:
  - {id: out-1, task: task-1, kind: theory,       benefit: ben-1}
  - {id: out-2, task: task-2, kind: demonstrator, benefit: ben-2}   # → benefits[].id (open-source ingestion toolkit)
validations:
  - {id: val-2, task: task-2, baseline: "single-node batch-ingest; prior-year pipeline",
     stress: "10x burst replay on peak-hour traces, sized to force tail-latency blowup",
     mechanism_check: "does adaptive batching engage backpressure at the queue-depth threshold",
     metric: "p99 write latency < 800 ms sustained at 10x baseline throughput", comparator_class: standard}
```

`validate_ir.py` `traceability-spine` enforces the referential integrity + crosswalk: a dangling
or duplicate id, a task with no `person` (unstaffed) or no `years`, a resource-dependent task with
no `budget_lines` (or a non-institutional budget row referenced by no task — the reverse leg), or a
`benefit`/`output`/`objective` link resolving to nothing → **FAIL in `--mode submission` / WARN in
`--mode draft`**, naming the specific broken edge. The full template is
`templates/project-plan.template.yaml`.

## Graduation note

This module is deliberately grant-agnostic. It is scheduled to graduate into a standalone
skill, **`research-profile-evidence-base`**, reusable for CVs, biosketches (NSF SciENcv,
NHMRC), promotion cases, award nominations, and annual reviews — anywhere a defended,
provenance-tracked research profile is needed. For now it ships inside
`grant-application-writing` as a self-contained reference so it can be lifted out without
untangling grant-specific logic. Keep it that way: nothing in this file should assume a
grant is the consumer.
