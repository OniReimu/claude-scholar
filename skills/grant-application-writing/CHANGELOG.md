# Changelog — grant-application-writing

## 0.6.0

The "scheme-KB + register-driven literature review" release. Closes meeting-audit items A3 and A4.
Both are reference/discipline additions (no fragile new validators): the value is the accumulating
knowledge base + the register-calibrated drafting rule.

### Added — A3: per-scheme accumulating knowledge base
- `templates/scheme-kb.template.yaml` — a standing, year-over-year KB for ONE scheme: the marketing /
  selection criteria (with interpretation) + past reviewer/panel comments ("pings") + failure modes +
  exemplar pointers. INSTANCE DATA that accumulates round over round (like `rate-table.yaml`), not
  hardcoded. Placeholders only. (Meeting #16/35/82.)
- method-passes §4.1 (reviewer/panel tailoring): the `reviewer_model`'s scoring-emphasis + red-flags are
  now built FROM the `scheme-kb` when present (what the panel actually praised/punished last round),
  each scored field aligned to the matching criterion's interpretation. Honesty guards: a reviewer quote
  is verbatim (never invented); "does the draft address a past concern" is a review-pass judgement, not
  a mechanical check.

### Added — A4: register-driven literature-review shape
- method-passes §2.11(i): how deep/formal the literature treatment goes is driven by
  `classification.register` — **academic** → a genuine literature review (position against prior art,
  cite, name the gap); **industrial** → story completeness (problem→solution→benefit), where a thin
  lit-review is EXPECTED, not a defect. The "literature-thin narrative" flag is now raised only for
  `register: academic`; an industrial application is not flagged for lacking a scholarly review it was
  never meant to carry. Dispatch note added to SKILL.md's register bullet. (Meeting #2.)

## 0.5.0

The "profiles + CV + meeting-nuance" release. Closes the largest concrete gaps from the working-meeting
audit: the reusable profiles were a written rule with no artifact, and length-adaptive CV / two stressed
nuances weren't baked.

### Added — reusable profile artifacts (were prose-only; now real templates)
- `templates/institution-profile.template.yaml` — the administering-org KB (legal name, ABN, address,
  org type, HESA eligibility, authorised signatory, standard institutional-support boilerplate, rate-table
  pointer, standing partnerships). Admin fields that are KNOWN institutional facts fill FROM here, not
  `[TO SET]`. Placeholders only (zero real data in the skill).
- `templates/applicant-profile.template.yaml` — the applicant bio/admin/eligibility scaffold + a
  `cv_config`. POINTS to the evidence store for the research record (no duplication); carries the
  eligibility facts (citizenship/PR, appointment basis, PhD date, interruptions) the eligibility gate needs.

### Added — length-adaptive CV builder
- `scripts/build_cv.py` — reads the applicant profile's `cv_config` (per-page section CAPS = instance data)
  + the evidence store, SELECTS the top-N per section (publications ranked by tier→year via `--tiers`,
  funding by amount, rest by recency), and assembles a Markdown CV to a `2pp`/`6pp` page budget. Pure
  select/order/format of REAL data (never authors prose); missing name/title = `[TO SET]`; every trim is
  REPORTED (no silent cap). `--self-test` covered. (Meeting #64/65.)

### Added — disciplines baked from meeting nuances
- **Never fabricate to hit a word count** (Output-convention #4): the ≥90% fill target is for real
  substance; under limit-pressure a model invents ("撑到 200 字→硬编"). Under-limit with no genuine
  substance → leave it short + flag, never pad. (Meeting #26.)
- **Concrete ARC density** (plainness dial): an ARC technical description uses ~1–4 formulas, describes
  the model briefly, and includes NO experiment figures (no results exist yet — a schematic is fine, a
  benchmark plot is off-genre); align content to the scheme's marketing/selection criteria + past
  reviewer comments where available. (Meeting #33/34/35.)

## 0.4.0

The "Stage-A0 classification" release. Adds an up-front instrument/register/deliverable
classification (the funding-application analog of a paper skill deciding paper-type first) and
fixes a real gating bug it exposed.

### Added — AXIS 0 (Stage-A0 dispatch classification)
- A `classification` block in `scheme.yaml` set at intake, BEFORE mode/process: three ORTHOGONAL
  facets — `instrument` (award|grant), `register` (industrial|academic), `funder_family` — plus a
  `requires` list of the deliverables to build. Documented in SKILL.md ("Stage-A0 classification")
  + `form-schema-ir.md` (schema line + field note).
- **Classify by the FORM's structure, not its name**: a DECRA is *named* a fellowship/award but has
  a budget → `instrument: grant`. `instrument: award` ⇒ `requires: []` ⇒ the pipeline skips
  B3/B4/B4s + build_budget/build_timeline/in-kind/stipend (the meeting's "don't over-engineer a
  prize"). A grant lists the deliverables it demands.
- **`register` is orthogonal to `funder_family`**: an ARC Linkage (LP) is ARC yet *industrial*
  (industry partner + co-contribution → plain language); ARC DP/DECRA/FT are *academic*. ARC spans
  both — the register comes from "is there an industry partner", not "is it ARC".
- `validate_ir.py` check 21 `classification`: fail-closed validation of the block (unknown
  instrument/register, unknown `requires` deliverable, or an award that requires a budget → FAIL;
  a grant that builds nothing → WARN; no block → SKIP, legacy fallback to `mode`).

### Fixed — the DECRA gating bug the classification exposed
- The project-substance passes (validate_ir checks 13–16, 19) and budget/plan machinery were gated
  on `mode == prospective-project`. A DECRA is `mode: narrative-award` yet HAS a budget + work-plan,
  so its budget/plan was WRONGLY skipped. Now gated on `classification.requires` (via
  `_scheme_requires`, with a legacy `mode` fallback) — `instrument`/`requires` decouples "what you
  BUILD" from "what you're JUDGED ON". Regression tests cover the DECRA case + the award-skips case.

## 0.3.0

The "builders + fidelity + disciplines" release. Driven by a real AVSTICI (Australia-Vietnam) worked
instance and a working-meeting review. Root theme: the skill had rich validators + type-models but few
BUILDERS, and several honesty/scope disciplines were implicit.

### Added — BUILDERS (the third script class: compute an artifact from inputs, `[TO SET]` for gaps, fail-closed)
- `build_budget.py` — cost + itemise a budget (personnel `person × FTE × rate × years + on-costs`;
  `rate_ref` lookup from an institution rate table with annual step progression; **HDR stipend =
  base + top-up**, itemised, no on-cost; other-costs per year). Emits validate_budget's `rows[]` schema.
- `build_timeline.py` — schedule from the spine (`tasks[].years[]` + `depends_on`) → Gantt + milestones.
- `build_rope_time.py` — ROPE / eligibility date arithmetic (PhD + interruptions → effective
  years-since-PhD + within-window verdict, `borderline` band).
- `build_effort_allocation.py` — per-CI FTE vs `current_commitments` → over-subscription flags.
- `build_track_record_metrics.py` — narrative-award scored substance (pubs/yr, venue-tier distribution,
  M-of-N denominator, assistive career-best ranking; tiers via `--tiers` instance table).
- `build_cocontribution.py` — matched-funding split (`grant × ratio → required` per partner × FY).
- `import_uts_rates.py` — parse an institution salary calculator xlsx → a `rate-table.yaml` (regenerate
  yearly; a plausibility floor guards a column-shift). Ships the SCHEMA only, never a year's numbers.
- `extract_values.py` + `render_md.py` — first-class render pipeline (PASTE-READY.md → values.yaml →
  docx + md mirror), replacing scratchpad scripts; single source of truth.

### Added — render fidelity
- `render_docx.py` strategy-3 **under-label** + `--tables`: fill a TAG-LESS official template in place
  (locate each label, fill the answer slot; fill the budget table), 100% fidelity — never a lookalike.

### Added — disciplines (baked into SKILL.md / method-passes / submission-management)
- writing-anti-ai is a MANDATORY pre-render gate (§4.7).
- Template fidelity is absolute (dissect the real template, fill in place).
- Context-provenance (briefing ≠ assertable) + **corpus-first** (use the applicant's own design/prior
  applications directly; author only genuine gaps, marked `[DOMAIN-EXPERT TO VERIFY]`).
- 90–95% fill **with project substance, not background padding** (background economy).
- Single source of truth — machine/human twins (blockers.yaml ↔ .md ↔ HUMAN-ACTIONS) must not drift.
- In-kind must be STRUCTURED (entity-store items), never omitted or zeroed; equipment-ineligible schemes → in-kind is equipment's home.
- Risk register = PROJECT-DELIVERY risks, not submission/eligibility risks (those → blockers).
- No hand-waving — a scored technical field states the concrete mechanism, not just the activity.
- Funder-family plainness dial (industry/CRC-P plain · ARC mid · fellowship academic).
- Reusable institutional + applicant profiles cut `[TO SET]` noise (length-adaptive CV).
- Stage F++ human handoff (HUMAN-ACTIONS.md + drafted proforma letters) is mandatory.

### Notes
- Builder audit CLOSED — all 6 identified construction gaps built.
- Deferred: run the narrative-award builders on a live DECRA; retroactive-impact (Group 3) live test.

## 0.1.0
Initial: two-axis dispatch (mode × process), type-model, method-passes, evidence/entity/project stores,
validate_ir (20 checks) + validate_budget + charcount + build_manifest, ECR narrative-award fixture.
