# Submission Management — Stage A0 (compliance matrix) & Stage F (submission plan)

> Real applications fail on **logistics, not prose**: a missing signature, a blown
> internal deadline, an un-authorised submitter, a font that violates the format rule,
> an attachment that never got signed. Both artifacts here were Codex FAIL findings on
> v1 — the skill drafted beautiful boxes for forms that could never be submitted.
>
> - **A0 `compliance_matrix.yaml`** is built at intake, **before any drafting** — the rulebook.
> - **F `submission_plan.yaml`** is built before submit — the execution tracker.
> - **F+ `build-manifest.yaml`** is generated at the submit gate — the run-audit record + the one
>   `ready_to_submit` verdict (`scripts/build_manifest.py`, see the Stage F+ section).
>
> `scheme.yaml` (Stage A) is **derived with** the compliance matrix, not before it: the matrix's
> `limits`, `format_rules`, `phases`, and `submission_authority` populate `scheme.yaml`'s `submission:` head
> and each field's `limit`/`stage_lock`/`submission_phase` attributes. Build A0 first; A0 → A → … → F.

## Stage A0 — `compliance_matrix.yaml`

The inventory of every rule that can block or invalidate a submission, extracted from the
guidelines / funding rules / portal help **before a single field is drafted**. If a hard
eligibility rule fails here, stop and say so — do not draft (see SKILL.md core discipline #2).

```yaml
scheme: "Example ECR Award (2026 round)"
scheme_version: "2026"
source_docs: [funding-rules.pdf, portal-help.html, faculty-internal-process.md]
mode: narrative-award                       # dispatches downstream passes (see SKILL.md)

eligibility:                                 # hard pass/fail — resolve BEFORE drafting
  - id: elig.phd_window
    rule: "PhD conferred ≤ 5 years before 1 Mar 2026 (career-disruption extensions apply)"
    binding: hard                            # hard = submit-blocker | soft = weakens
    derived: "phd_date + interruptions vs cutoff"   # → scheme.yaml computed eligibility-gate
    check: unresolved                        # unresolved | pass | fail | n/a
  - id: elig.employment
    rule: "Continuing/fixed-term staff at the administering org, ≥0.5 FTE at nomination"
    binding: hard
    check: unresolved

mandatory_docs:                              # every artifact that MUST accompany the form
  - id: doc.nominee_statement
    name: "Nominee statement (main narrative)"
    kind: heading-sequenced                  # structured-upload sub-kind (see type-model.md)
    required: true
  - id: doc.nominator_statement
    name: "Nominator statement"
    kind: proforma                           # fixed wording + signature block
    required: true
  - id: doc.faculty_endorsement
    name: "Faculty endorsement"
    kind: proforma
    required: true

limits:                                      # ALL of them — never assume max-only / single unit
  - field: nominee_statement
    unit: words                              # chars | words | pages
    max: 1000
    min: null
    nested_sublimits: []                     # e.g. FT B10 200w + 300w inside one box
  - field: career_best_outputs
    unit: items
    max: 5

format_rules:                                # format IS a compliance rule, not cosmetics
  font: "Arial ≥11pt (Calibri ≥11pt accepted)"
  line_spacing: "single"
  margins: "≥2 cm all sides"
  page_size: A4
  filename_pattern: "{surname}_{scheme}_{doc}.pdf"   # portals reject on mismatch
  reference_style: null

certifications:                              # who must sign what, in wet/e-sign
  - id: sig.nominator
    name: "Nominator statement signature"
    on: doc.nominator_statement
    signatory: nominator                     # applicant | nominator | associate-dean | partner | DVC-R
    kind: eligibility-attestation
    method: e-signature
    status: pending
  - id: sig.faculty
    name: "Faculty endorsement signature"
    on: doc.faculty_endorsement
    signatory: associate-dean
    kind: eligibility-attestation
    method: e-signature
    status: pending

deadlines:
  funder_deadline: 2026-03-01T17:00+11:00    # the real, external cutoff
  internal_cutoffs:                          # institution gates that fall BEFORE the funder's
    - id: cut.research_office
      what: "Research Office pre-award sign-off"
      due: 2026-02-23T17:00+11:00
    - id: cut.faculty_endorsement
      what: "Associate Dean endorsement returned"
      due: 2026-02-20T17:00+11:00

submission_authority:                        # authorised submitter — often NOT the nominee
  authorised_role: research-office           # nominee drafts; RO/Associate Dean submits
  note: "Nominee cannot self-submit; submission is via the nominating Faculty"

policy_constraints:                          # scheme-/institution-wide obligations
  foreign_interference: "UFIT declaration if any foreign affiliation/funding"
  open_access: "n/a for award"
  data_management: "n/a for award (applies to project modes)"
  coi: "declare relationship to nominator/assessors"
  ethics: "n/a"
```

Notes:
- `binding: hard` items are the gate list. Any `check: fail` on a hard item → **stop**.
- `internal_cutoffs` are the deadlines that actually govern the calendar — the funder deadline
  is the last one, not the first. Work backwards from `funder_deadline` through every cutoff.
- `format_rules` feed `scheme.yaml` field validation and the Stage E format check.
- For project modes, expand `policy_constraints` (DMP mandate, budget-cap rules, matched-funding
  thresholds) and `mandatory_docs` (budget, partner letters, ethics annexes).

## Stage F — `submission_plan.yaml`

The execution tracker built after drafting, before submit. Turns the compliance matrix into an
owned, dated, dependency-ordered plan and tracks approval + attachment state to the moment of submit.

```yaml
scheme: "Example ECR Award (2026 round)"
funder_deadline: 2026-03-01T17:00+11:00
internal_deadline: 2026-02-23T17:00+11:00
portal: "Symplectic Elements"
portal_notes: "session timeout ~20 min; save often; no autosave on narrative boxes"
submission_authority: {authorised_role: research-office, person: "TBC"}

artifacts:                                   # one row per fillable field OR attachment
  - id: art.nominee_statement
    type: attachment                         # attachment | form-field
    owner: nominee                           # nominee | nominator | associate-dean | partner | RO
    submission_phase: full                   # minimum-data | EOI | full | post-award
    due: 2026-02-16
    internal_cutoff: 2026-02-18              # must be done by, ahead of external
    approval_state: drafting                 # drafting | in-review | approved | signed | locked
    attachment_status: not-attached          # n/a | not-attached | attached | verified-in-portal
  - id: art.nominator_statement
    type: attachment
    owner: nominator
    submission_phase: full
    due: 2026-02-18
    internal_cutoff: 2026-02-20
    approval_state: drafting
    attachment_status: not-attached
  - id: art.faculty_endorsement
    type: attachment
    owner: associate-dean
    submission_phase: full
    due: 2026-02-20
    internal_cutoff: 2026-02-23
    approval_state: not-started
    attachment_status: not-attached
  - id: art.discovery_profile
    type: form-field
    owner: nominee
    submission_phase: full
    due: 2026-02-16
    approval_state: drafting
    attachment_status: n/a

dependency_graph:                            # X blocks Y — do not schedule Y before X clears
  - blocks: art.nominator_statement_signed   # signed nominator PDF …
    depends_on: [art.nominator_statement]    # … needs the drafted statement first
  - blocks: art.faculty_endorsement
    depends_on: [art.nominee_statement]      # AD endorses the drafted narrative
  # project-mode example: budget depends on partner contributions
  # - blocks: art.budget_matrix
  #   depends_on: [art.partner_contributions]

phase_locks:                                 # what freezes after each phase clears
  - phase: EOI
    locks: []                                # narrative-award ECR is single-phase; none
  # two-stage / multi-phase schemes populate this — see below

version_history:
  - {ts: 2026-02-10, event: "matrix built; eligibility elig.phd_window = pass"}
  - {ts: 2026-02-14, event: "nominee statement v2 → in-review"}

pre_submit_checklist:                        # final gate, run in portal, in order
  - "[ ] all hard eligibility gates = pass"
  - "[ ] every mandatory_document attached AND verified-in-portal"
  - "[ ] all required signatures collected (nominator, associate-dean)"
  - "[ ] format_rules satisfied on every PDF (font/margin/filename)"
  - "[ ] char/word/page limits pass on every field"
  - "[ ] internal cutoffs all cleared; RO has the file"
  - "[ ] authorised submitter confirmed and available before deadline"
  - "[ ] portal dry-run: no hidden-required field, no validation error"
```

## Multi-phase awareness

For schemes with more than one submission round, every artifact and field carries a
`submission_phase` (from `type-model.md`), and `phase_locks` records what becomes **read-only**
after each round closes (the `stage_lock` attribute: `{authored_at, editable_at[], locked_from}`).

| scheme | phases | what locks after phase 1 |
|--------|--------|--------------------------|
| ARC DP26 | EOI → Full | EOI fields render read-only in the Full Application (`stage_lock`) |
| AEA / Innovate | EOI → Full application | eligibility + focus-area selection lock; only shortlisted advance |
| NHMRC (Sapphire) | minimum-data → full | synopsis + ≥5 keywords lock; they drive reviewer matching |
| Ethereum Foundation ESP | inquiry → application → finance | scope locks at application; `payout-target` opens only at finance |

The plan tracks, per artifact: which `submission_phase` it belongs to, whether that phase is
open/closed, and — once closed — that its `approval_state: locked` blocks any further edit.
Schedule phase-2 work only against fields whose phase is open; never plan an edit to a locked field.

## Worked example — an ECR scheme (narrative-award, single-phase)

A minimal real instance. Nominator is the nominee's line manager; endorsement comes from the
Associate Dean; the Discovery Profile link must be refreshed; portal is Symplectic Elements
with a ~20-min session timeout and a hard deadline.

```yaml
scheme: "Example ECR Award (2026 round)"
funder_deadline: 2026-03-01T17:00+11:00
internal_deadline: 2026-02-23T17:00+11:00
portal: "Symplectic Elements"
portal_notes: "~20 min timeout; draft narrative offline, paste last"
submission_authority: {authorised_role: research-office, person: "Faculty RO"}

artifacts:
  - {id: art.nominee_statement, owner: nominee, due: 2026-02-16,
     internal_cutoff: 2026-02-18, approval_state: in-review, attachment_status: not-attached}
  - {id: art.nominator_statement, owner: nominator,        # = line manager
     due: 2026-02-18, internal_cutoff: 2026-02-20,
     approval_state: drafting, attachment_status: not-attached}
  - {id: art.faculty_endorsement, owner: associate-dean,
     due: 2026-02-20, internal_cutoff: 2026-02-23,
     approval_state: not-started, attachment_status: not-attached}
  - {id: art.discovery_profile, owner: nominee, due: 2026-02-16,
     approval_state: drafting, attachment_status: n/a}      # refresh link, verify live

dependency_graph:
  - {blocks: art.nominator_statement_signed, depends_on: [art.nominator_statement]}
  - {blocks: art.faculty_endorsement,        depends_on: [art.nominee_statement]}

pre_submit_checklist:
  - "[ ] PhD-window eligibility = pass"
  - "[ ] nominator statement drafted → signed PDF → approval_state: locked"
  - "[ ] Associate Dean endorsement attached + verified-in-portal"
  - "[ ] Discovery Profile link refreshed and resolving"
  - "[ ] narrative ≤1000 words; Arial ≥11pt; A4; filename pattern OK"
  - "[ ] RO confirmed as submitter, available before 1 Mar 17:00 AEDT"
```

The load-bearing moves: the **nominator (line manager) statement** goes
`drafting → signed PDF → locked` and nothing downstream touches it once signed; the **Associate
Dean endorsement** cannot start until the nominee narrative is drafted (dependency edge); the
**Discovery Profile** is a `form-field` whose only job is a live, refreshed link; and the whole
plan is paced by **internal cutoffs** that all fall days before the funder's hard deadline.
