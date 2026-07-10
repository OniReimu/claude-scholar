# The Two-Axis Field Type Model

> Load-bearing abstraction. Read before any other reference. A naive flat set of field
> types (prose / list / budget-table / checkbox / attachment / link) was stress-tested
> against ARC, CRC-P, AEA, NHMRC, ERC, NSF, Ethereum Foundation, and Google/Microsoft/
> Amazon forms and FAILED — every scheme broke it. The working abstraction is:
>
> **a field = one WIDGET (structural shape) × one SEMANTIC ROLE (meaning) + ATTRIBUTES.**
>
> The provenance column names the scheme that forces each type into existence, so the set
> is falsifiable: if a real field won't fit, fix the set — never special-case the scheme.

## Axis 1 — Widget primitives (structural shape)

### Leaf widgets

| widget | shape | forced by |
|--------|-------|-----------|
| `narrative` | prose box; `limit = {value, unit: chars\|words\|pages, min?, nested_sublimits[]?}` | all — NHMRC uses chars, NSF/ERC use pages, EF has min+max (500–2000), ARC FT B10 nests 200w+300w sub-budgets in one box |
| `scalar` | single typed value: text / number / date / bool | all (turnover, headcount, FTE, conferral date) |
| `money` | `{value, currency}` — a single requested amount | EF `budgetRequest`+`currency`, Amazon cash gift |
| `single-choice` | enum pick; options **round-scoped & version-bound**; `Other → conditional field` | EF `domain`/`output`, Google/Amazon rotating tracks, AEA focus areas |
| `multi-choice` | pick-N from enum | AEA priority areas, keywords |
| `taxonomy-code` | controlled-vocab classification; hierarchical/searchable; often **% allocation summing to 100**; carries scheme version | FoR/SEO/ANZSIC (ARC, AEA, NHMRC), NSF/ERC panels |
| `boolean-gate` | Yes/No; some are **hard submit-blockers**; often conditional-trigger | CRC-P §B eligibility self-assessment, EF `paymentAcknowledgement`, AEA/ARC disclosures |
| `declaration` | attestation / consent / COI / ethics agreement | NSF, ARC, all |
| `link` | URL / external id; **attr `role: metadata \| evidence`** | EF repo link (metadata) vs RetroPGF contribution links (scored evidence) |
| `payout-target` | discriminated union: `crypto{wallet, ens_resolved}` \| `fiat{beneficiary, IBAN, SWIFT}` | EF GranteeFinance (post-award) |
| `credit-request` | cloud/compute ask distinct from cash: `{amount, justification, products[], datasets[], counts_toward_cash_total: bool}` | AFMR (credits-only), Amazon ARA (parallel to cash), Microsoft (in budget table but excluded from total) |

### Container / computed widgets

| widget | shape | forced by |
|--------|-------|-----------|
| `repeating-group` | list-of-objects; `{min, max}`; per-item nested fields **with their own limits** | NHMRC Top-10-in-10 (10 × {citation ≤500 + explanation ≤1000}), AEA objectives (≤4 × 750ch), milestones, personnel |
| `conditional-group` / `decision-tree` | branching questionnaire; answers **conditionally require** annexes/prose | ERC Ethics & Security self-assessment, NHMRC Sex/Gender statement, foreign-interference blocks |
| `budget-matrix` | categories × year × org × cash/in-kind; **per-row caps** + **cross-field validation** + **live totals** | CRC-P (audit ≤1%, overseas ≤10%), ARC, AEA phased, NHMRC |
| `contribution-matrix` | partner × contribution-type × financial-year (CRC-P's 3-way cash / cash-staff / in-kind), auto-subtotalled | CRC-P |
| `relational-table` | multiple linked tables encoding a graph | NSF Collaborators & Other Affiliations (5 relational tables = conflict graph) |
| `milestone-table` / `stage-gate` | rows `{milestone, deliverable, measure-of-success (picked from list), date/month, status}`; **two representations: proposed (in application) vs contracted (at award)** | AEA B2, CRC-P G9, EF post-award milestones, MS timeline |
| `risk-register` | bounded rows `{risk, likelihood/impact, pre-mitigation, mitigation, residual}`; `{min, max}` | AEA B3/B4 (min1/max3), CRC-P |
| `computed` | value **derived** from other fields; may **gate eligibility**; not stored, must be recomputed + validated | ARC career-interruption-total → eligibility window; AEA/CRC-P co-contribution ratio → threshold (10% Ignite / 1:1 Innovate) |
| `linked-profile` | field backed by an external **authority record**; referenced, not free-typed; includes **rank-ordered-pick-from-profile + per-item annotation** | ARC 10 Career-Best (RMS Person Profile), NSF SciENcv biosketch/Current&Pending, ERC structured CV, AEA/all ORCID |
| `structured-upload` | a PDF/doc that is really a schema; sub-kinds below | ubiquitous |

`structured-upload` sub-kinds — **an attachment is never "just a blob":**

- `free` — genuinely open upload (rare).
- `proforma` — fixed wording that **must not be altered** + embedded structured data + signature block (ARC Linkage G2 certification letter, AEA partner declaration).
- `composite` — bundled single PDF with **conditional page-limits** (Google RSP: proposal + PI CV + optional co-PI CV, cap shifts 5→7pp with co-PI; strict filename pattern).
- `heading-sequenced` — mandated internal headings in fixed order, **scheme-specific** (ARC Project Description headings differ DP vs DECRA vs FT; NHMRC template; each maps to a rubric criterion).
- `system-generated` — rendered from a profile authority, not typed (NSF SciENcv, ERC prescribed CV).

## Axis 2 — Semantic role (orthogonal to widget)

The same widget means different things and is evidenced/scored differently by its role:

| role | what it is | drives |
|------|-----------|--------|
| `eligibility-gate` | hard pass/fail (may be `computed`/derived) | run first; block drafting on failure |
| `criterion-scored` | maps to a rubric item + weight — the persuasive content | effort ∝ weight; verb-tiering; anti-double-count |
| `admin-metadata` | contact, IDs, dates | validation only |
| `classification` | FoR / panel / theme | reviewer routing / panel fit |
| `compliance` | ethics / security / foreign-interference / COI / DMP | completeness pass; often assessor-invisible |
| `budget-resource` | cash / in-kind / credits | budget-math validation |
| `team-partner` | people / orgs / roles / contributions | entity store (B2) |
| `evidence` | track-record item / impact link / attestation | provenance + defensibility |
| `logistics` | phase / signature / declaration | submission plan (F) |

## Field attributes (cross-cutting; apply to any widget × role)

- **`limit`** — `{value, unit: chars|words|pages, min?, nested_sublimits[]?}`. Never assume max-only or a single unit.
- **`visibility`** — subset of `[assessor, institution, funder, minister-public]`. **≥3 audiences, orthogonal to type.** ARC: NIT, COI, foreign-interference declarations never render in the assessor PDF but go to the institution / minister. Determines what a reviewer actually reads.
- **`stage_lock`** — `{authored_at, editable_at[], locked_from}`. Two-stage lock-forward: ARC DP26 fields authored in the EOI render **read-only** in the Full Application.
- **`submission_phase`** — `minimum-data | EOI | full | post-award`. NHMRC Sapphire minimum-data (synopsis + ≥5 keywords) is due before the full app to drive reviewer matching; ARC/AEA EOI; EF 3-stage (inquiry → application → finance).
- **`per_person_multiplicity`** — replicated per CI/investigator (NSF biosketch + Current&Pending per senior person; MS per-person effort).
- **`scoring_adjustment`** — `relative-to-opportunity | career-disruption`. A **normalization lens that modifies how OTHER fields are scored** (NHMRC track-record 70% is all RtO; ERC; ARC ROPE), backed by a dated-FTE disruption declaration. The same interruption is often entered twice (eligibility extension + ROPE narrative).
- **`depends_on` / referential-integrity** — cross-field / cross-part coupling. ARC teaching-relief requested in Part B must be actioned into the Part D budget. **ARC Linkage Partner Organisation is one logical entity spanning four form locations (A3 pick + D1 sub-budget cross-validated against the main table + G3/G4 classification + G2 proforma letter) with RMS mismatch-blocks — the hardest single thing to model.**
- **`award_vehicle`** — `unrestricted_gift | grant | credits_only`. Google/Amazon cash = unrestricted gift (no overhead); Microsoft = grant with budget; AFMR = credits only. Changes obligations and budget shape.
- **`conditional_obligation`** — bindingness depends on `award_vehicle`. Google data-policy: a statement of intent under a gift, a **mandatory award condition** under a constrained topic.
- **`counts_toward_total`** — cash-total inclusion flag for `credit-request` (Microsoft Azure credits sit in the budget table but are excluded from the cash total).

## How the axes compose (worked reads)

- ARC "10 Career-Best Outputs" = `linked-profile` (rank-ordered pick + per-item ≤150ch annotation) × role `evidence` + attr `per_person_multiplicity` (per CI).
- NHMRC "Top 10 in 10" = `repeating-group` (10 × {cite ≤500, expl ≤1000}) × role `criterion-scored` (weight 35%) + attr `scoring_adjustment: relative-to-opportunity`.
- CRC-P partner contributions = `contribution-matrix` × role `team-partner`/`budget-resource` + attr `depends_on` (cross-validates the budget matrix + drives the matched-funding `computed` gate).
- Microsoft Azure line = `credit-request` × role `budget-resource` + attr `counts_toward_total: false`.
- ARC career-interruption total = `computed` × role `eligibility-gate` (derived boolean: PhD date + interruption total vs cutoff).
- A "strategic-priorities" box (fictional: "tick ≤2 of six priorities + describe fit in ≤N chars") = a `section`/`fieldset` of **two** sub-fields — a `multi-choice` (pick-N) × role `classification` **and** a `narrative @N` × role `criterion-scored` — **not** one `narrative` sized to the whole visible box. Same shape as AEA "TRL: pick 3–5 + justify" and any "focus area + rationale". A composite field is always decomposed; collapsing it invents a wrong `limit` and produces a false `char-fit` PASS (see `form-schema-ir.md` → *Composite fields & the no-silent-fallback rule*). Read the sub-field's real limit from the scheme's own form — never from an example here.

## Project-grant machinery notes (from CRC-P validation)

Container/computed widgets carry sharper rules than the table row implies — logged when
`prospective-project` mode was validated end-to-end on CRC-P R19:

- **`budget-matrix` caps carry a denominator.** A `max_pct` cap is meaningless without saying
  *percent of what*. Schemes mix denominators in one form: CRC-P computes audit/overseas/travel
  caps against **total cash** (excludes the in-kind row) but computes the ≤50% grant rule against
  **total incl. in-kind**. Encode `of: total-cash | total | requested` per cap — never assume a
  single total. Wrong denominator = a silent false-pass on the exact cap the scheme cares about.
- **`budget-matrix` cross-validation includes cumulative cash-flow, not just row caps.** A
  per-financial-year liquidity check (cumulative spend ≤ cumulative cash-in) can fail a budget
  whose spend is front-loaded and cash back-loaded, even when every row cap passes.
- **`contribution-matrix` partner-axis ownership.** When a `contribution-matrix` is nested inside a
  per-partner `repeating-group` (each partner item holds its own type×FY sub-table), the
  `repeating-group` owns the partner axis — do not double-count it inside the matrix widget.
- **External calculator artifact.** A mandatory offline workbook (e.g. CRC-P's `.xlsx` financial
  workbook) that back-stops in-form matrices is not a field, an upload, or a `computed` — model it
  as `link` × attr `role: external-calc`; its computed cells must be **re-derived and validated
  in-form**, never trusted blind.
- **`rubric[].weight` may nest sub-indicators.** A single `criterion-scored` box can carry
  internal point weights (CRC-P C1 = 10/8/7; C4 = 6/6/4/9). Capture them as `rubric[].sub_weights`
  so effort ∝ sub-weight within the box, distinct from `narrative.nested_sublimits` (char budgets).
- **`allocation_sums_to` is a general attribute,** not taxonomy-code-only — it also applies to a
  `repeating-group` whose rows carry percentages (e.g. per-site % of project value summing to 100).

## Falsifiability

This set is a hypothesis. When ingesting a new scheme (Stage A), if a field maps to no
widget above, **do not force it** — log it, extend the widget table with a provenance note,
and update this file. A skill that silently coerces a novel field into the wrong type
produces a broken application. The set grew from 6 to the above precisely by this rule.
