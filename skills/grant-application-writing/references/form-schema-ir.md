# Form-Schema IR — `scheme.yaml` (Stage A)

> Layer-1 abstraction. Stage A ingests **a blank form + its guidelines** and emits one
> normalized `scheme.yaml`. Every field in that file is expressed in the **widget × role +
> attributes** model — this file does **not** re-list the widget/role/attribute sets; that is
> `type-model.md`, which is load-bearing and must be read first. Here we specify only the
> *container* schema that holds those fields, and the protocol for producing it.
>
> The IR is the contract between parsing (Stage A) and drafting (Stage C): a value the
> drafter needs — a limit, a weight, a gate, a phase lock — must be captured here or it does
> not exist downstream. If it is not in `scheme.yaml`, the drafter cannot see it.

## Why an IR at all

A scheme is never one document. The **form** carries the fields, their order, and their
limits; the **guidelines** carry the rubric, the weights, the eligibility rules, and the
compliance obligations. Neither alone is sufficient. `scheme.yaml` is the join: it fuses
both (and any addenda — CFP page, portal help text, funding rules PDF) into a single
machine-checkable object so the rest of the pipeline reads structure, not prose.

The A0 `compliance_matrix.yaml` (see `submission-management.md`) is authored *alongside*
`scheme.yaml` from the same sources — it captures the rulebook (fonts, certs, internal
deadlines, submission owner); `scheme.yaml` captures the fillable structure. They cross-
reference by `scheme` + `scheme_version`.

**The scheme carries the FORM; the project-plan carries the PROJECT.** In
`prospective-project` mode the project's *substance* — aims/design, benefits, additionality/VfM,
and a trigger-driven risk register — is NOT the form structure; it lives in a separate
`project-plan.yaml` sidecar (B3, see `evidence-store.md`), one per application, and is validated by
`validate_ir.py --plan` against the §2.14–§2.18 substance passes. Keep it out of `scheme.yaml`:
the IR describes boxes and limits, the plan describes the research.

The project-plan also carries the **traceability spine** (stable ids: aim→objective→task→subtask→
output→benefit, crossed by person→year→budget). Those ids are the **join** between otherwise
disconnected fields: a value referenced in a project figure, a `milestone-table`, or a `budget-matrix`
line must resolve to a spine id, so cross-field consistency (does this funded line map to a real task?
does this figure box name a real output?) becomes a deterministic `validate_ir.py` `traceability-spine`
check rather than a manual read.

## The `scheme.yaml` schema

```yaml
scheme: "Example ECR Grant"    # human name (an ECR scheme)
classification:                                # AXIS 0 (Stage-A0 dispatch) — set FIRST, from the FORM's structure not its name
  instrument: award                            #   award | grant — an award funds no project; a grant does
  register: academic                           #   industrial | academic — industry-partnered (ARC LP, CRC-P, DFAT) vs pure-academic (ARC DP/DECRA/FT); ORTHOGONAL to funder_family
  funder_family: ARC                           #   ARC | NHMRC | CRC-P | DFAT | NSF | ERC | internal | industry — scheme-specific conventions
  requires: []                                 #   deliverables to build ⊆ {budget, work_plan, in_kind, stipend, co_contribution}; award ⇒ [] (skip B3/B4/B4s + validate_ir 13–16,19); a grant lists what it demands
mode: narrative-award                          # AXIS 1 — narrative-award | prospective-project | retroactive-impact
process: [single-stage-review, panel-routed]   # AXIS 2 — SET from the closed archetype vocab (see below); selects Group-5 overlays
rejoinder: {enabled: true, window: "10 working days", char_limit: 2000}  # within-round right-of-reply CAPABILITY (single-stage-review may carry it); omit if none
scheme_version: "2026-R1"                       # round/year tag; taxonomy + option sets are bound to this
portal: "web-form"                              # RMS | Elements | Submittable | Sapphire | SmartSimple | SmartyGrants | web-form | docx | ...
source_docs:                                    # every doc that fed this IR — provenance for every value below
  - {id: form,  path: "ecr-2026-form.docx",       role: fields+limits}
  - {id: guide, path: "ecr-2026-guidelines.pdf",  role: rubric+eligibility+weights}

eligibility_gates:                              # role=eligibility-gate, hoisted to the top; run BEFORE drafting
  - id: within-window
    rule: "≤5 years since PhD conferral at close of round"
    binding: hard            # hard = submit-blocker | soft = disadvantage only
    derived: "(round_close - phd_conferral_date) - career_interruption_total <= 5y"   # a `computed` gate: expr over other fields, recomputed not stored
    check: "phd_conferral_date, career_interruption_total"                            # fields the expr depends on
  - id: employed-at-inst
    rule: "Continuing/fixed-term appointment at the institution at time of application"
    binding: hard
    check: "appointment_status"                 # plain gate: no derived expr, just a field to validate

submission:
  modality: web-form        # web-form | docx | pdf-acroform | pdf-xfa | pdf-flat | pdf-scanned | latex | xlsx | sf424-xml
  phases: [full]            # ordered subset of [minimum-data, EOI, full, post-award]; gates/locks are per-phase
  deadline: "2026-08-14T17:00+10:00"
  timeout: "portal session 30min"               # portal idle-logout, if any
  submission_authority: "Research Office (not applicant)" # matters for submission_plan (F)
  hard_fail_rules:                              # scheme-level auto-reject conditions, distinct from per-field gates
    - "over char limit on any narrative field"
    - "missing mandatory attachment"

sections:                                       # CONTAINERS, not a flat field list — preserve form order
  - id: track-record
    title: "Research Track Record"
    repeating: null                             # or {min, max} for list-of-section (e.g. per-CI blocks)
    fields:                                     # each field = the type-model.md object
      - id: significance
        label: "Significance & impact of research"
        widget: narrative
        role: criterion-scored
        limit: {value: 4000, unit: chars}
        visibility: [assessor, institution]
        stage_lock: null                        # or {authored_at, editable_at[], locked_from}
        submission_phase: full
        # ...any further attrs from type-model.md §attributes apply here

rubric:                                         # role=criterion-scored fields link here by criterion
  - criterion: "Significance & impact"
    weight: 0.40
    evidence_types: [publication, citation-metric, grant, invited-talk]
    reviewer_panel: null                        # panel/college that scores it, if scheme routes by classification
    minimum_evidence: [publication, external-comparator]   # evidence CLASSES required before this criterion can score; absent → readiness `unsupported`
    readiness_rule: ">=1 backed claim per sub-indicator + no [TO SET]/[COMPARATOR NEEDED] in its fields"   # what must hold to reach `substantiated`

attachments:                                    # every upload; kind is NEVER "just a blob" (type-model.md structured-upload)
  - name: "CV"
    kind: system-generated                      # free | proforma | composite | heading-sequenced | system-generated
    source: "ORCID export"
    constraints: {pages: 2, headings: null, filename: "surname_cv.pdf"}

requirements:                                   # the CFP's NORMATIVE obligations — the must/should/desirable logic, parsed from guidelines
  - id: req-a1                                  #   a narrative field alone can't stop a proposal answering a `desirable` and skipping a `mandatory`
    text: "protocol relies on weaker-than-previous security assumptions"
    strength: mandatory                         # mandatory | expected | desirable | optional  (obligation strength — governs whether a gap BLOCKS)
    applies_if: "classification.workstream == A" # predicate over a `classification` field; omit/null = always applies
    quantifier: at_least_one                    # all | at_least_one — only for a group carrying `alternatives`
    alternatives: [req-a1, req-a1b]             # ids that jointly satisfy the group under `quantifier` (and/or logic); omit for a lone req
    criterion: "Scientific excellence and contribution beyond the SOTA"   # rubric criterion it rolls up to
    # applicant's project-plan objectives/tasks/outputs carry `addresses: [req-ids]`; validate_ir `requirement-coverage` joins them
```

### Field notes

| key | rule |
|-----|------|
| `classification` | AXIS 0 (Stage-A0, set FIRST). Three ORTHOGONAL facets: `instrument` (`award`\|`grant`), `register` (`industrial`\|`academic`), `funder_family`; plus `requires ⊆ {budget, work_plan, in_kind, stipend, co_contribution}`. **Classify by the FORM's real structure, not its name** — a DECRA is *named* a fellowship/award but has a budget → `instrument: grant` (even though its `mode` is `narrative-award`). `instrument: award` ⇒ `requires: []` ⇒ the pipeline skips B3/B4/B4s + `build_budget`/`build_timeline`/in-kind/stipend + `validate_ir` checks 13–16 & 19 (don't over-engineer a prize). `register` is ORTHOGONAL to `funder_family` (ARC LP is ARC yet industrial). `validate_ir.py` `classification` check FAILs an unknown value or an award that requires a budget; a legacy scheme with no block falls back to the `mode` heuristic. This is the axis that decouples the budget/plan machinery from `mode`. |
| `mode` | AXIS 1. Set from the funding-mode dispatch (SKILL.md). Selects which passes run; a mismatch here mis-routes the whole pipeline. Distinct from `classification.instrument`: `mode` is what you're JUDGED ON, `instrument` is what you must BUILD (a DECRA is `mode: narrative-award` + `instrument: grant`). |
| `process` | AXIS 2 (SKILL.md assessment-process dispatch). **Mandatory, closed vocab**: a non-empty subset of `{single-stage-review, staged, interview-gated, panel-routed, curated, rolling}`. Selects the Group-5 process overlays. Fail-closed: an unknown tag, or a missing `process` on a rubric-bearing scheme, is a parse gap — `validate_ir.py` (`process-dispatch` check) FAILs it, never defaults it. Wires to existing machinery rather than duplicating it: `staged` → `submission.phases` must carry an EOI/pre-proposal/minimum-data phase; `panel-routed` → the `taxonomy-code`/`classification` fields become gate-critical. |
| `rejoinder` | A within-round right-of-reply CAPABILITY (distinct from cross-round resubmission, §4.2). `{enabled, window?, char_limit?}`; only `single-stage-review` schemes carry it. Consumed by the §5.1 overlay. Omit when the scheme has no reply window. |
| `scheme_version` | Mandatory. Option enums, taxonomy code sets, weights, and limits all drift between rounds; a value with no version is unciteable. |
| `source_docs[].role` | Records *which document* a value came from, so a later conflict (form says 4000ch, guide says 500 words) is traceable, not silently resolved. |
| `eligibility_gates[].derived` | Present only for `computed` gates. The expr is stored as text and **recomputed at check time** — never cache a stale boolean. Omit for plain gates. |
| `sections[].repeating` | `null` for a normal section; `{min,max}` when the whole section repeats (per-CI, per-partner). Per-*item* repetition inside a field stays a `repeating-group` widget on the field, not here. |
| partner section `depends_on` | A partner/collaborator section's fields may carry a `depends_on` to a legal-entity **jurisdiction** gate: the jurisdiction of the *committing* entity (evidence-store `partners[].legal_entity.jurisdiction`) can trip an offshore-partner / national-interest eligibility rule. Model that coupling as `depends_on` so the gate re-checks when the committing entity changes, rather than treating jurisdiction as inert prose. |
| `fields[]` | Each entry is exactly a `type-model.md` field object. This file adds no new field keys — it only nests them under sections. |
| `rubric[]` | Derived from the **guidelines**, not the form. Every `criterion-scored` field must resolve to one `criterion` here; a scored box with no rubric weight is a parse gap, not a zero. |
| `requirements[]` | The CFP's normative obligations, parsed from the guidelines: each carries a `strength` (`mandatory` / `expected` / `desirable` / `optional`), an optional `applies_if` predicate (over a `classification` field — e.g. a chosen workstream), and `quantifier`/`alternatives` for and/or groups. A `mandatory` obligation collapsed into a generic "theme-fit" gate is a parse defect — it lets `criterion-readiness` false-pass a proposal that answers a `desirable` and skips a `must`. The applicant's `project-plan` nodes carry `addresses: [req-ids]`; `validate_ir.py` `requirement-coverage` joins the two and BLOCKS an unaddressed `mandatory` in submission mode. |
| budget `funding_status` / `funding_window` | A call may fund only year-1 with later years indicative/continuation-conditional. Budget rows carry `funding_status: requested | indicative | conditional` (default `requested`) and the scheme a `funding_window: {funded: [years]}`; `validate_budget` reports `requested_total` distinct from `indicative_total` and FAILs a `requested` row spending outside the funded window — so a multi-year plan cannot read as fully funded. |
| `attachments[].kind` | One of the five `structured-upload` sub-kinds. `free` is rare and must be justified — most "uploads" are `proforma`/`heading-sequenced`/`system-generated`. |
| ROPE `outputs-context` field | In `narrative-award` schemes (DECRA/FT/Investigator), a `narrative` field with its OWN char limit whose job is **field-calibration** — teach the assessor the discipline's esteem norms (venue tiers, authorship conventions, a ranking-service standing), NOT to repeat the outputs listing. Renders from the evidence-store `outputs_context` block; validated by `validate_ir.py` `outputs-context-completeness` (every career-best output tiered + clustered; a cluster superlative needs an attributor). See `author-voice.md` §10 + `method-passes.md` §1.10. |
| host-institution statement | The administering-organisation statement is a `structured-upload` (`proforma`/`heading-sequenced`) authored in the **institution's third-party voice**, committing concrete support (establishment grant, stipend top-up, salary-shortfall cover, teaching relief) with a stated total. Renders from entity-store `organizations[].institutional_support`; its total is reconciled against sum(items) + the budget's non-ARC lines by `validate_ir.py` `institutional-support-reconciliation` (`method-passes.md` §4.5) — the fellowship analog of the batch-2 partner-letter reconciliation. |
| career-best subset | A `repeating-group` listing subset (e.g. "10 career-best outputs") with a **stable label scheme** (`[*]`/`[J*]`/`[C*]`) so the outputs-context narrative can cross-reference specific outputs compactly. Backed by evidence-store `outputs_context.career_best`. |

## Extraction protocol (form + guidelines → `scheme.yaml`)

Both documents are inputs; run the two passes and then reconcile.

1. **Form pass (structure + limits).** Walk the blank form in order. For each control emit a
   `field` under its `section`: assign `widget` (type-model.md Axis 1), read the `limit`
   verbatim (never assume max-only or a single unit — check for min, page vs char vs word,
   nested sub-limits), capture `stage_lock`/`submission_phase` if the portal shows them.
   Preserve document order — reviewers read in it, and `depends_on` couplings often follow it.
2. **Guidelines pass (meaning + scoring + gates).** Read the guidelines for: the **rubric**
   (criteria + weights + accepted evidence types) → `rubric[]`; **eligibility rules** →
   `eligibility_gates[]` (mark `binding`, write `derived` for computed ones); **compliance
   obligations** (ethics/COI/foreign-interference) → fields with `role: compliance`;
   **hard-fail conditions** → `submission.hard_fail_rules`.
3. **Reconcile + assign roles.** Join the two: attach each `criterion-scored` field to its
   `rubric` criterion; set `visibility` (guidelines/portal reveal who reads what — often ≥3
   audiences); resolve any form-vs-guide conflict explicitly in `source_docs` provenance
   rather than picking one silently.
4. **Multi-doc schemes.** List every source in `source_docs` (funding-rules PDF, CFP page,
   portal help, addenda). When two disagree, the more authoritative/binding document wins and
   the loser is recorded — do not average or guess. Round-specific addenda override the base
   template for that `scheme_version`.

## Worked mini-example

Three real fields, expressed in the IR (abridged to the load-bearing keys):

```yaml
# 1. A scored narrative box: limit from the FORM, weight from the GUIDELINES
- id: significance
  label: "Significance and innovation of the proposed research"
  widget: narrative
  role: criterion-scored
  limit: {value: 4000, unit: chars}
  visibility: [assessor, institution]
# ...and its rubric row (from the guidelines):
rubric:
  - criterion: "Significance and innovation"
    weight: 0.40
    evidence_types: [publication, prior-grant]

# 2. A taxonomy field with %-allocation summing to 100, version-bound
- id: for-codes
  label: "Fields of Research (FoR)"
  widget: taxonomy-code
  role: classification
  taxonomy: {scheme: "ANZSRC-FoR", version: "2020", levels: 4, max_codes: 3, allocation_sums_to: 100}
  visibility: [assessor, funder]   # drives panel routing

# 3. A derived eligibility gate (computed over other fields, recomputed not stored)
- id: within-window          # lives under eligibility_gates[], not a section field
  rule: "≤5 years since PhD conferral, net of career interruptions, at round close"
  binding: hard
  derived: "(round_close - phd_conferral_date) - career_interruption_total <= 5y"
  check: "phd_conferral_date, career_interruption_total"
```

## Falsifiability

If a form control maps to **no** widget in `type-model.md`, do not coerce it into the
nearest fit — a wrong widget produces a silently broken application. Instead: log the
unmatched control (scheme, field, why it doesn't fit), extend the `type-model.md` widget
table with a provenance note, and only then encode it in `scheme.yaml`. The widget set grew
from 6 to its current size precisely by this rule; `scheme.yaml` inherits the same
discipline. A parse gap is a visible TODO, never a guessed value.

### Composite fields & the no-silent-fallback rule

A control that *partly* matches a widget is more dangerous than one that matches none —
the falsifiability log fires, but Stage A can still emit a plausible-looking default that
buries the flag. Two hard rules:

1. **Composite fields are two (or more) sub-fields, never one.** A pick-N paired with a
   bounded justification (a "strategic-priorities: tick ≤2 + describe in ≤N chars"
   box; an "AEA TRL: pick 3–5 + justify" box; a "focus area + rationale" box) is a
   `section`/`fieldset` holding a `multi-choice` (or `single-choice`) **and** a separate
   `narrative` **with its own limit** — not a single `narrative` sized to the visible box.
   Collapsing it invents a wrong `limit`.
2. **A flagged widget/limit ambiguity must be RESOLVED before Stage A finalizes the
   field — never defaulted.** If the true widget or its `limit` is uncertain, do not fall
   back to `narrative @ <whatever-max-is-visible>`; go back to the form/guidelines (or the
   live portal) and read the real sub-structure and per-sub-field limits. A guessed limit
   silently defeats the `char-fit` pass downstream: it validates the draft against the
   wrong number and reports a **false PASS** on an over-limit field. If the real limit
   genuinely cannot be found, encode it as `limit: {value: null, note: "UNVERIFIED — do
   not char-fit"}` so the pass fails closed instead of green-washing.
