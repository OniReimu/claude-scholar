# Method Passes — Stage C (fill) & Stage E (review)

> Mode-aware operations on the `scheme.yaml` IR. Every pass reads and writes the IR
> (or the drafted field values attached to it); none is scheme-specific. Which passes
> run is decided by `mode` (§ Funding-mode dispatch in SKILL.md). Narrative-award passes
> always run; project passes are *added* for `prospective-project`; cross-cutting passes
> run in both. Stage E re-checks their outputs as a contract before submission.
>
> Worked example throughout: **UTS ECR** (`narrative-award`) — regenerated from its corpus.

---

## Group 1 — narrative-award passes (Stage C)

### 1.1 gate-check — RUNS FIRST, blocks on fail
- **Does:** evaluate every `eligibility_gates[]` entry (incl. `computed`/`derived` gates) before any drafting. A failed hard gate (`binding: true`) stops the pipeline — do not draft an application the applicant cannot submit.
- **In:** `eligibility_gates`, evidence-store dates/facts. **Out:** pass/block verdict per gate + one-line reason; on block, a stop report, not a draft.
- **Example (ECR):** "years-since-PhD ≤ 5 at the round census date" is a `computed` gate: PhD-conferral date − career-interruption total vs cutoff. Compute it from the record, don't trust a remembered number. If it fails, report the exact margin and stop.

### 1.2 verb-tiering / claim defensibility
- **Does:** for each `criterion-scored` narrative claim, pick the **strongest verb the evidence survives under reviewer probe, and no stronger**. The verb *is* the evidentiary commitment. Under-claim before over-claim.
- **In:** claim + backing evidence (`status`, `source_authority`, `confidence`). **Out:** verb assigned per claim + the tier that licensed it, so review can re-audit.
- **Tier framing (the evidentiary ladder):**
  - **Tier A — "led / built / delivered / secured":** applicant is the direct, sole or clearly-primary cause; evidence is first-authored, PI-held, or an official record naming them.
  - **Tier B — "contributed to / co-developed / drove":** genuine causal role shared with others; evidence is co-authorship, CI status, or a named-but-shared contribution.
  - **Tier C — "engaged with / participated in / informed by":** real but diffuse involvement; evidence is membership, attendance, or downstream association.
- **Worked example (ECR, DFCRC/RBA line):** the applicant worked on a project connected to the DFCRC / RBA CBDC pilot. Evidence supported a *named contributing role on a workstream*, not project leadership. Verb-tiering drops "led the RBA CBDC pilot" (Tier A — indefensible, invites "you were one of many") to **Tier B "contributed to the DFCRC/RBA CBDC pilot"** — the strongest verb that survives a reviewer who knows the project. Choosing the honest tier pre-empts the probe instead of inviting it.

### 1.3 anti-double-counting
- **Does:** assign each funding item / achievement to **one primary field**. The same grant or output must not inflate two boxes as if it were two things; genuine overlaps are reframed as **distinct contributions on one timeline**.
- **In:** all drafted `evidence`-role fields. **Out:** each item tagged with its single home field; overlaps rewritten as different facets (funding vs the paper it produced vs the talk it seeded), not repeats.
- **Example (ECR):** a grant that also yielded a flagship paper appears once as *funding secured* and once as *a distinct research output* — same event, two honestly-different contributions — never counted twice under "total funding."

### 1.4 number-defensibility
- **Does:** every quantitative claim must be (a) **true**, (b) **internally consistent across all fields** (totals, counts, dates reconcile), (c) **within the eligibility timeline**, and (d) **not so abnormal it taints its neighbours** (one implausible figure makes a reviewer distrust the surrounding true ones).
- **In:** all numeric claims across the IR. **Out:** each number verified, cross-field-reconciled, in-window, and flagged if it reads as an outlier needing context.
- **Example (ECR):** "total competitive funding $X" must equal the sum of the itemised grants, every grant dated inside the ECR window, and citation/h-index figures must match the profile authority — an inflated total next to a modest publication count invites disbelief in both.

### 1.5 role/credit discipline
- **Does:** frame authorship / project role honestly — first / corresponding / co-first / CI / supervising — and **pre-empt the "not sole first" objection** by stating the shared credit before a reviewer infers it.
- **In:** `evidence` items with role metadata. **Out:** each output annotated with its true role; co-first / shared-CI stated, not implied.
- **Example (ECR):** a co-first-authored paper is written "co-first author (equal contribution)" rather than presented as sole-first; a supervised student's output is framed as supervision, not personal authorship.

### 1.6 char-fit
- **Does:** pack each `narrative` field under its `limit` (respecting `nested_sublimits`), preserving the highest-weight content. **Report the character/word count in the output header** so the fit is auditable.
- **In:** drafted prose + `limit`. **Out:** trimmed prose with a `[chars: 1487/1500]`-style header per field.
- **Fail closed on an unverified limit.** char-fit is only as trustworthy as the `limit` Stage A encoded. If a field's `limit` is `null`/`UNVERIFIED` (e.g. a composite field whose sub-limit was never resolved), do **not** report PASS against a guessed number — surface it as a blocker and send it back to Stage A. A green char-fit on the wrong limit is the failure mode this pass exists to prevent (a composite "pick-N + 600-char describe" box validated against a 4000-char default reports a false PASS on an over-limit draft).
- **Example (ECR):** an achievement box capped at N chars keeps the Tier-A claims and cuts hedging; header shows the final count for portal-paste confidence.

### 1.7 sensitive-content editorial
- **Does:** exclude content that is risky or unverifiable — internal-review-sensitive collaborations, unrefereed preprints where a published version is expected, and any entry resting on a **misspelled name + guessed title** (an unverifiable citation is worse than an omitted one).
- **In:** candidate evidence + `sensitivity` / `use_permission` / `confidence`. **Out:** a filtered evidence set + a short "excluded, and why" note for the applicant.
- **Example (ECR):** a collaboration under confidential institutional review is dropped from the public narrative; a half-remembered paper title is verified against the profile authority or cut.

---

## Group 2 — prospective-project passes (ADDED for `prospective-project`)

Run these *in addition* to Group 1 when `mode = prospective-project`.

### 2.1 project-coherence
- **Does:** verify **aims ↔ methods ↔ milestones ↔ budget** form one consistent chain — every aim has a method, every method a milestone, every milestone a funded line; no orphan aim, no unfunded activity, no budget line without a purpose.
- **In:** aims, methods narrative, `milestone-table`, `budget-matrix`. **Out:** a coherence map + list of orphans/gaps.

### 2.2 budget-aims alignment
- **Does:** confirm each budget category traces to a stated aim/method and the spend shape matches the ambition (no aim starved, no line unexplained).
- **In:** `budget-matrix` × aims. **Out:** per-line justification links; flags for unmotivated or under-resourced lines.

### 2.3 methods-feasibility
- **Does:** check the proposed methods are achievable in the timeframe with the requested resources and the team's demonstrated capability.
- **In:** methods narrative, timeline, team record. **Out:** feasibility verdict per aim + over-reach flags.

### 2.4 risk-mitigation completeness
- **Does:** ensure the `risk-register` covers the material risks, each with a mitigation and a residual, within min/max row bounds; no headline risk left unmitigated.
- **In:** `risk-register`, methods. **Out:** coverage check + missing-risk flags.

### 2.5 impact-pathway
- **Does:** verify a traceable line from outputs → outcomes → impact appropriate to the scheme (academic, translational, commercial, public).
- **In:** impact narrative, aims. **Out:** pathway completeness + broken-link flags.

### 2.6 facilities/resources
- **Does:** confirm required facilities, infrastructure, and data access are stated and credibly available.
- **In:** facilities field, entity-store (B2). **Out:** availability check.

### 2.7 team-capability
- **Does:** confirm the team collectively covers the skills each aim needs; gaps named and staffed or justified.
- **In:** entity-store, aims. **Out:** capability-coverage matrix + gap flags.

### 2.8 budget-math validation
- **Does:** mechanically validate the `budget-matrix` / `contribution-matrix` arithmetic and every scheme rule: **per-row caps** (e.g. audit ≤ 1%, overseas ≤ 10%), **matched-funding ratio ≥ threshold**, **phased-budget gating** (per-phase totals), and **credit-vs-cash separation** (credit-request lines respect `counts_toward_total`).
- **In:** budget/contribution matrices, `computed` ratio gates. **Out:** pass/fail per rule with the offending cell + amount; blocks submission on a hard cap breach.
- **Example (CRC-P family):** overseas spend computed as % of total must be ≤ 10%; matched cash/in-kind must meet the co-contribution ratio the `computed` gate enforces — recompute, never trust the portal's cached total.

### 2.9 compliance completeness
- **Does:** confirm every `compliance`-role field is present and mutually consistent — ethics, security, COI, DMP, foreign-interference — and that `conditional-group` triggers fired their required annexes.
- **In:** compliance fields, decision-tree answers. **Out:** presence + consistency check; missing-annex flags. (Often assessor-invisible but hard submit-blockers.)

---

## Group 3 — cross-cutting passes (both modes)

### 3.1 reviewer/panel tailoring (`reviewer_model`)
- **Does:** build a `reviewer_model` for the target panel — **expertise level, jargon tolerance, scoring emphasis (which criteria carry weight), and red-flag claims that panel punishes** — then reframe the *same project* to fit it.
- **In:** scheme rubric + panel description. **Out:** a `reviewer_model` object + per-field framing notes.
- **Reframing the one project across panels:** **ARC** — significance + national benefit, ROPE-aware, restrained tone; **NHMRC** — health translation, RtO normalisation, structured track-record; **NSF** — intellectual merit + broader impacts as co-equal, US-centric; **ERC** — high-risk/high-gain frontier, PI-centric, ambition rewarded; **industry (Google/MS/Amazon)** — product relevance and open-source/data intent, concise; **internal (UTS ECR)** — early-career trajectory and institutional fit over raw scale.

### 3.2 prior-submission / review-response
- **Does:** for resubmissions, thread prior reviewer feedback into the draft and (where the scheme has a response field) produce a point-by-point response; align claims with what changed.
- **In:** prior reviews, current draft. **Out:** response-to-reviewers + a change-log the narrative reflects. (Integrates the `review-response` skill.)

### 3.3 COI / reviewer-management
- **Does:** populate reviewer-management fields — NSF COA relational tables, **excluded reviewers**, **suggested reviewers** — consistently with the entity-store's collaboration graph.
- **In:** entity-store relationships, COA rules. **Out:** filled conflict tables + exclusion/suggestion lists, cross-checked against co-authorship within the scheme's lookback window.

---

## Stage E — review as a checklist-driven CONTRACT

Stage E is **not** "run tool X." It is a contract of checkable items; each must be
green before submission. The **operational method** is an **adversarial, multi-round**
review — read as a hostile reviewer looking for the one disqualifying flaw — reinforced
by a **cross-model pass (Codex)** so a second model re-derives the checks independently.
But the deliverable is the checklist verdict, item by item:

- **eligibility** — every hard gate re-evaluated green (gate-check, 1.1) at the census date; no drafted content assumes an unmet gate.
- **compliance** — all `compliance`-role fields present and consistent; conditional annexes triggered (2.9).
- **evidence-provenance** — every claim traces to an evidence-store item with `status` / `source_authority` / date; no orphan claim; verbs match their tier (1.2).
- **cross-field consistency** — numbers, totals, roles, and dates reconcile across all fields; no double-count (1.3, 1.4).
- **budget-math** — all caps, ratios, phase totals, and credit/cash separation validated (2.8); zero hard breaches.
- **attachments-complete** — every `structured-upload` present in its correct sub-kind, within page/heading limits, proforma wording unaltered, filenames matching the required pattern.
- **panel-fit** — framing matches the `reviewer_model` (3.1); no red-flag claim for this panel; effort ∝ criterion weight.
- **risk-coverage** — material risks each mitigated with a residual (2.4), within row bounds.
- **portal validation dry-run** — modality-appropriate final check (char counts under limit, required web-form fields non-empty, AcroForm fields filled, hidden-required satisfied) simulating the portal's own submit-time validation before the human pastes/uploads.

A failing item returns to its Stage C pass, not to ad-hoc editing. Ship only when the
whole contract is green under both the adversarial and the cross-model reading.
