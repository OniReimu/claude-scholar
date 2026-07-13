# Changelog — grant-application-writing

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
