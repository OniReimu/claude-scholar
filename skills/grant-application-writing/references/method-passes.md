# Method Passes — Stage C (fill) & Stage E (review)

> Mode-aware operations on the `scheme.yaml` IR. Every pass reads and writes the IR
> (or the drafted field values attached to it); none is scheme-specific. Which passes
> run is decided by `mode` (§ Funding-mode dispatch in SKILL.md). Narrative-award passes
> always run; project passes are *added* for `prospective-project`, retroactive-impact
> passes for `retroactive-impact`; cross-cutting passes run in all. Stage E re-checks their
> outputs as a contract before submission.
>
> Worked example throughout: an **internal ECR scheme** (`narrative-award`) — regenerated from its corpus.

---

## Group 1 — narrative-award passes (Stage C)

> These passes govern *what you may claim*. The *register* the drafted prose is written in —
> composition, sentence patterns, lexicon, the funded-vs-first-draft moves — is `author-voice.md`.
> Draft in that voice; these passes keep it defensible.
>
> **National-benefit ladder (both exemplars).** Every scored claim ladders its so-what up to
> national / societal benefit — the scheme scores it — so a claim that stops at the direct user
> is under-pitched. Escalate direct user → sector → national economy / sovereign capability
> (`author-voice.md` §1 altitude ladder, §5.3): "cuts [task] cost for [users]" becomes "…,
> strengthening [sector] productivity and the nation's [strategic capability]."

### 1.1 gate-check — RUNS FIRST, blocks on fail
- **Does:** evaluate every `eligibility_gates[]` entry (incl. `computed`/`derived` gates) before any drafting. A failed hard gate (`binding: hard`) stops the pipeline — do not draft an application the applicant cannot submit (a `binding: soft` gate is a disadvantage, not a blocker).
- **In:** `eligibility_gates`, evidence-store dates/facts. **Out:** pass/block verdict per gate + one-line reason; on block, a stop report, not a draft.
- **Example (ECR):** "years-since-PhD ≤ 5 at the round census date" is a `computed` gate: PhD-conferral date − career-interruption total vs cutoff. Compute it from the record, don't trust a remembered number. If it fails, report the exact margin and stop.

### 1.2 verb-tiering / claim defensibility
- **Does:** for each `criterion-scored` narrative claim, pick the **strongest verb the evidence survives under reviewer probe, and no stronger**. The verb *is* the evidentiary commitment. Under-claim before over-claim.
- **In:** claim + backing evidence (`status`, `source_authority`, `confidence`). **Out:** verb assigned per claim + the tier that licensed it, so review can re-audit.
- **Tier framing (the evidentiary ladder):**
  - **Tier A — "led / built / delivered / secured":** applicant is the direct, sole or clearly-primary cause; evidence is first-authored, PI-held, or an official record naming them.
  - **Tier B — "contributed to / co-developed / drove":** genuine causal role shared with others; evidence is co-authorship, CI status, or a named-but-shared contribution.
  - **Tier C — "engaged with / participated in / informed by":** real but diffuse involvement; evidence is membership, attendance, or downstream association.
- **Worked example (fictional):** an applicant worked on a high-profile national infrastructure pilot run by a government agency. The evidence supports a *named contributing role on one workstream*, not project leadership. Verb-tiering drops "led the national pilot" (Tier A — indefensible, invites "you were one of many") to **Tier B "contributed to the pilot's [specific workstream]"** — the strongest verb that survives a reviewer who knows the project. Choosing the honest tier pre-empts the probe instead of inviting it. (When applying this to a real application, name the concrete project only in the applicant's own IR instance, never here.)
- **Defensible-primacy sub-move (novelty claims).** "first / only / record / world-leading" are Tier-A verbs on the *novelty* axis, so a **bare, unscoped superlative is a Tier-A over-claim** — "first X" invites "no, [prior work] already did X." SCOPE it with qualifiers until it survives a specialist probe: the strongest *defensible* form is the audaciously-scoped claim, not the bare one. **Example (fictional):** not "first [technique]" but "first *robust* [technique] *above [threshold]* in *any* [substrate]" — metric ∧ regime ∧ breadth qualifiers let an audacious "first" survive the reviewer who knows the prior art, where the bare superlative would not. (Novelty verb-tiering; see `author-voice.md` §8, "defensible primacy via tight scoping.")

### 1.3 anti-double-counting
- **Does:** assign each funding item / achievement to **one primary field**. The same grant or output must not inflate two boxes as if it were two things; genuine overlaps are reframed as **distinct contributions on one timeline**.
- **In:** all drafted `evidence`-role fields. **Out:** each item tagged with its single home field; overlaps rewritten as different facets (funding vs the paper it produced vs the talk it seeded), not repeats.
- **Example (ECR):** a grant that also yielded a flagship paper appears once as *funding secured* and once as *a distinct research output* — same event, two honestly-different contributions — never counted twice under "total funding."

### 1.4 number-defensibility — two strategies, one discipline
- **Does:** enforce the unifying rule — **never a number you can't defend** — which resolves mode-aware into one of two moves, *omit* or *scope+source*:
  - **Forward / market claims** (esp. `prospective-project`): prefer confident **qualitative** momentum ("rapidly expanding demand", "clear market pull"); do **not** put an unsourced TAM/ROI figure on the page. If a number is genuinely required, it must be sourced or explicitly caveated. The defensible move is **omit** — leave no fragile figure to attack (`author-voice.md` §5.4).
  - **Track-record / person claims** (esp. fellowship / `narrative-award`): quantify **freely**, but every number = **value + scope + attributor**, where scope = metric ∧ time-window ∧ role and attributor = an external validator or an evidence-store pointer. An unscoped superlative is indefensible; a scoped, sourced one reads as fact. The defensible move is **scope+source** (`author-voice.md` §8; rendered from the evidence-store `window` / `role` / `attributor` fields, never improvised).
  - **Baseline for every number, both modes:** (a) **true**, (b) **internally consistent across all fields** (totals, counts, dates reconcile), (c) **within the eligibility timeline**, (d) **not so abnormal it taints its neighbours** (one implausible figure makes a reviewer distrust the surrounding true ones).
- **In:** all numeric claims across the IR + the funding `mode`. **Out:** market figures omitted for qualitative momentum (or sourced/caveated where required); track-record figures emitted as value+scope+attributor; all cross-field-reconciled, in-window, outliers flagged.
- **Example — market (fictional):** a serviceable-market size is **not** stated as "$X M/year"; it becomes "rapidly expanding as [regulation] tightens; demand is clear" — no figure to source, none to defend.
- **Example — track record (ECR):** "total competitive funding $X" must equal the sum of the itemised grants, every grant dated inside the ECR window and tagged with the `role` it was held under (PI/CI), and citation/h-index figures must match — and cite — the profile authority (`attributor`); an inflated total next to a modest publication count invites disbelief in both.

### 1.5 role/credit discipline
- **Does:** frame authorship / project role honestly — first / corresponding / co-first / CI / supervising — and **pre-empt the "not sole first" objection** by stating the shared credit before a reviewer infers it.
- **In:** `evidence` items with role metadata. **Out:** each output annotated with its true role; co-first / shared-CI stated, not implied.
- **Example (ECR):** a co-first-authored paper is written "co-first author (equal contribution)" rather than presented as sole-first; a supervised student's output is framed as supervision, not personal authorship.
- **Authorship-convention-decoding sub-move (narrative-award).** Where the field's convention differs
  from "first-author = most credit" — a subfield that orders authors alphabetically, or where
  last/second author on a co-supervised paper signals the senior idea-and-design contributor — do
  **not silently CAP** the claim to what a first-author-centric reader assumes. DECODE the convention
  for the assessor from `outputs_context.authorship_convention` (bounded, evidence-backed: what the
  position means in THIS subfield + who did what), pre-empting "why not first author". **A decode is
  not an upgrade:** it states the field's actual credit assignment, still bounded by
  `contribution_summary` (§1.4) and never role-upgraded past what the evidence supports — the honest
  *floor* (the §1.9 specificity-floor mirror of the honest ceiling), not a promotion. Rendered in
  §1.10 (outputs-context field-calibration) FROM the store block, never improvised. (Cross-ref §1.10
  authorship-convention decoding, `outputs_context.authorship_convention` Agent B.)

### 1.6 char-fit
- **Does:** pack each `narrative` field under its `limit` (respecting `nested_sublimits`), preserving the highest-weight content. **Report the character/word count in the output header** so the fit is auditable.
- **In:** drafted prose + `limit`. **Out:** trimmed prose with a `[chars: 1487/1500]`-style header per field.
- **Fail closed on an unverified limit.** char-fit is only as trustworthy as the `limit` Stage A encoded. If a field's `limit` is `null`/`UNVERIFIED` (e.g. a composite field whose sub-limit was never resolved), do **not** report PASS against a guessed number — surface it as a blocker and send it back to Stage A. A green char-fit on the wrong limit is the failure mode this pass exists to prevent (a composite "pick-N + 600-char describe" box validated against a 4000-char default reports a false PASS on an over-limit draft).
- **Example (ECR):** an achievement box capped at N chars keeps the Tier-A claims and cuts hedging; header shows the final count for portal-paste confidence.

### 1.7 sensitive-content editorial
- **Does:** exclude content that is risky or unverifiable — internal-review-sensitive collaborations, unrefereed preprints where a published version is expected, and any entry resting on a **misspelled name + guessed title** (an unverifiable citation is worse than an omitted one).
- **In:** candidate evidence + `sensitivity` / `use_permission` / `confidence`. **Out:** a filtered evidence set + a short "excluded, and why" note for the applicant.
- **Example (ECR):** a collaboration under confidential institutional review is dropped from the public narrative; a half-remembered paper title is verified against the profile authority or cut.

### 1.8 invented-specifics marking & context-freshness
Two failure modes that "never invent evidence" must catch even when the prose *looks* fine — both surfaced by real DECRA runs.
- **Mark invented specifics `[TO SET]`.** When drafting needs a concrete technical figure the input did not supply — a threshold (ε, accuracy, overhead %), a metric, a dataset, a budget unit-rate, a spec — the writer may propose a *plausible placeholder* to keep the prose concrete, but MUST tag it `[APPLICANT/DOMAIN-EXPERT TO SET]` (or `[VERIFY]`). Never present a skill-generated number as if it were real: a reviewer probing "why ε = 0.02?" must not be met with a figure the applicant never chose. This extends `number-defensibility`: an invented spec is a number you cannot defend until the applicant owns it.
- **Never smooth an unverified context into a favourable framing.** Employment, load, role, and status facts (e.g. "research-focused position", "principal supervisor", "since 2021") must be verified as *current* against the evidence store, with a `source_authority`/`as_of`. If the corpus may be stale or a context fact is unconfirmed, state the honest or unknown version — **never the flattering one**. A teaching-heavy Lectureship is drafted as teaching-heavy (then reframed as productivity-relative-to-opportunity, `author-voice.md` §8), not laundered into "research-focused" — which is both weaker and, on stale input, factually wrong. Flag a possibly-stale corpus rather than trusting its most favourable reading.
- **Markers are two-mode — never ship raw markers in the submission.** Every `[TO SET]` /
  `[VERIFY]` / `[EXTERNAL COMPARATOR NEEDED]` / `[STAT — SOURCE]` (from §1.8 and §1.9) is a
  **draft-mode annotation for the applicant**, not final prose. **An assessor scores the submitted
  case, not the editorial workflow** — a visible `[TO SET]` reads as an *unfinished application*
  and is penalised. So: in **draft/internal mode** surface the markers inline (correct for a
  working draft + review); at **submission/final render** every marker is either **resolved** to a
  committed, defensible value, or **lifted out** into a separate `blockers.md` "resolve before
  submit" list while the field is explicitly held pending the applicant — the submission prose
  **never carries a raw marker**. Fail closed: an unresolved marker at final render is a blocker,
  not shipped (mirrors `charcount` fail-closed and `render_*` no-partial-official).
  - **Purge the workflow shadow too — not just the brackets.** Removing `[TO SET]` is necessary
    but not sufficient: the submission prose must ALSO carry no *reference* to the drafting
    workflow — no "see blockers.md", no "to be confirmed during Y1 planning", no "drafting angle",
    no "figure inserted at submission", no `[Student]`-style role placeholders. An assessor
    re-detects the same unfinished signal by a different name. Submission prose reads as a
    **committed final case**; the entire to-do apparatus lives only in `blockers.md`, invisible to
    the assessor. Grep the submission text for blocker-pointers/hedge-to-later phrasing before render.

### 1.9 claim–evidence proportionality (over-claim guard)
The claim's *strength* must not exceed what its backing actually delivers — distinct from
verb-tiering (verb ↔ role) and number-defensibility (numbers true/scoped). This is the single
most common way a competent draft loses a panel: guarantees, legal effects and firsts that
over-reach. Three checks, each with a downgrade-or-mark action:
- **Capability over-reach.** A claim to "prove / guarantee / certify X" must be backed by a method
  that actually delivers X. If the method delivers a weaker thing — e.g. "proves the aggregation
  was computed correctly" ≠ "proves *which data* a model learned from"; "output deviation ≤ ε" ≠
  "legal erasure" — narrow the claim to what the method gives, or state the exact gap as a caveat.
  **Never let the guarantee exceed the mechanism.**
- **Legal / regulatory over-reach.** A claim of legal effect ("satisfies GDPR Art. 17",
  "certified right to erasure", "meets the Privacy Act") must be backed by the clause's *actual*
  effect, verified — not asserted from a one-line summary. Unverified → mark `[LEGAL EFFECT —
  VERIFY]` and state the defensible weaker form ("supports compliance with", not "certifies").
- **Primacy over-reach.** "first / only / no prior system does X" must cite **≥1 INDEPENDENT
  external comparator** — never the applicant's own survey or a self-made comparison table as the
  sole support. Absent one → scope the claim (defensible-primacy, §1.2) or mark `[EXTERNAL
  COMPARATOR NEEDED]`.
- **The dual — don't UNDER-use what the source DOES support (specificity floor).** Over-reach's
  mirror failure is over-caution: generalising a **source-backed concrete specific** into a vague
  noun forfeits credibility at **zero fabrication risk**. If the input names the actual technology,
  system, artefact, standard, figure or partner (e.g. "Hyperledger Fabric", "ServiceNow", a named
  API, a deployed N-node pilot), **name it** as a feasibility/impact anchor — do not launder it to
  "a permissioned ledger" / "an enterprise system". Honesty guards the ceiling of a claim; the
  specificity floor guards its bottom — say exactly what the evidence supports, no less as well as
  no more.
- **In:** every `criterion-scored` claim + its backing (method spec, cited clause, comparators,
  named source specifics).
  **Out:** each strong claim either backed, narrowed, or marked; every source-supported concrete
  specific named (not generalised away); the "guarantees exceed methods" panel concern pre-empted.
  (Cross-ref `author-voice.md` §5.4/§8; audited in Stage E.)

### 1.10 outputs-context / field-calibration (narrative-award; renders the `outputs_context` store block)
> Runs only when `mode = narrative-award` AND an `outputs_context` block is present (else SKIP,
> labelled). This is the ROPE **field-calibration** layer: it teaches a mixed generalist+specialist
> panel to read the CV in the field's OWN esteem terms. It RENDERS from the store, never improvises —
> `author-voice.md` §10 writes the register, this pass keeps it defensible.
- **Does:** render the field-calibration moves from `outputs_context`, each check-gated:
  - **(a) venue-tier glossing.** Every career-best venue carries a `field_norms.venue_tiers[]` entry
    — tier (e.g. "CORE A*", "JCR Q1") **plus** a PLAIN-LANGUAGE rank ("top-3 in `<field>`") so a
    generalist reads the eminence without knowing the field. A career-best output with no tier is
    **untiered → flagged**.
  - **(b) authorship-convention decoding.** Where author position does not mean "most credit" in this
    subfield, DECODE the convention from `field_norms.authorship_convention[]` (what last/second
    author means here + who did what), pre-empting "why not first author". Bounded, evidence-backed —
    a decode is **NOT a role upgrade** (this is the §1.5 authorship-convention-decoding sub-move,
    rendered here).
  - **(c) ranking attributor.** An eminence claim ("ranked Nth in `<field>`") is sourced to
    `field_norms.ranking_attributor` (an external ranking SERVICE + `as_of` date), never self-asserted.
  - **(d) output-clustering.** Group outputs into ~3–5 NAMED research threads (`clusters[]`); **every
    `career_best.ids` entry must appear in ≥1 cluster**. Each cluster's `primacy.claim` is statable,
    but with `primacy.attributor: null` it is **NOT written as a superlative** — reuse the
    defensible-primacy discipline (§1.2) / primacy over-reach guard (§1.9); the attributor is what
    licenses "first / milestone".
  - **(e) bounded credit.** Aggregate contribution is stated from `contribution_summary` ("M of N
    papers", with `basis`) — **bounded, never "all of it mine"**, and never role-upgraded (§1.5,
    §1.4 number-defensibility value+scope+attributor).
- **Submission mode:** a career-best output **unclustered or untiered** is a **WARN + a `blockers.md`
  entry** (structural — the calibration is incomplete, not fatal). A **cluster primacy claim with no
  attributor written as a superlative** is a **BLOCK** (an unsourced "first / milestone" is an
  over-claim, same discipline as §1.9). **Draft mode → WARN** throughout, consistent with
  markers-two-mode §1.8.
- **Mechanized by `validate_ir.py` `outputs-context-completeness`** (Agent D): gated on
  `mode == narrative-award` + an `outputs_context` block; every `career_best.ids` entry appears in ≥1
  `clusters[].outputs`, and every cluster carrying a `primacy.claim` has a non-empty
  `primacy.attributor` — FAIL (submission) / WARN (draft), per output / per cluster.
- **In:** `outputs_context` (`field_norms.venue_tiers|authorship_convention|ranking_attributor`,
  `clusters[]`, `career_best`, `contribution_summary`). **Out:** a field-calibrated outputs narrative
  — every career-best tiered + plain-ranked + clustered, author conventions decoded, eminence sourced
  to a ranking service, primacy claims either attributor-backed superlatives or non-superlative thread
  statements, aggregate credit bounded; incomplete calibration lifted to `blockers.md`, an unsourced
  superlative blocked. (Cross-ref `author-voice.md` §10 presentation register, §1.5 role/credit,
  §1.2/§1.9 primacy, §1.4 number-defensibility; the `outputs_context` store block Agent B.)
- **Example (fictional):** a candidate lists a career-best paper `[*]` at `<Flagship Conf>`. This pass
  renders its tier ("CORE A*, top-3 in `<field>`") from `venue_tiers`, decodes its last-author slot
  ("candidate contributed the main idea, design and writing; the student first author ran the
  experiments") from `authorship_convention`, and places it in the "`<named direction>`" cluster. The
  cluster's `primacy.claim` "milestone in `<tightly-scoped area>`" has `attributor: null` → it ships
  as a **thread statement, not** "the first ever" — writing it as a superlative with no attributor is
  a **BLOCK**. A separate "ranked Nth in `<field>`" line is sourced to "a ranking service" (`as_of`
  date); the aggregate reads "significant conceptual contribution on M of N papers" (per
  `contribution_summary`), not "all mine". A second career-best id present in `career_best.ids` but in
  no cluster → **WARN + `blockers.md`** (untiered/unclustered structural gap), caught mechanically by
  `outputs-context-completeness`.

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

### 2.7 team-capability (assess the team as a *composition*, not a skills bag)
- **Does:** confirm the team is a defensible composition for *this* project — not merely that each needed skill exists *somewhere* in the roster. A per-CI-scored Investigator criterion demands a person-indexed read, so this pass runs four labelled sub-checks over the evidence-store `investigators[]` (each person-indexed: `role`, `rope_context`, `track_record_ref`, `task_ownership`, `fte`, `current_commitments`; single-applicant schemes use the lead-CI shorthand where `owner` == first investigator).
  - **(a) individual capability.** For each CI, match their *own* `track_record_ref` against the `task_ownership` (aim/WP ids) they lead — the person who owns an aim must demonstrably be able to deliver it. A CI leading an aim their record does not support is a gap, even if another CI could.
  - **(b) multi-CI ROPE.** Read each CI's record **relative to their OWN career stage** (`rope_context`: years-since-PhD, stage, interruptions). **Never pool a senior + an ECR into one tally** — a shared team total launders a thin ECR record under a senior's output. ROPE window/role/attributor apply *per investigator*, each relative to their own opportunity (mirrors `number-defensibility` §1.4 value+scope+attributor, per person).
  - **(c) availability.** Check each CI's `fte` against their `current_commitments` (concurrent awards + committed FTE) — a fully-committed CI cannot also lead two aims at the declared FTE. Over-subscription is a feasibility flag, not a capability gap.
  - **(d) complementarity / synergy.** Confirm the composition *covers every aim* (no aim unstaffed) with *no redundant duplication* (two senior CIs on one narrow aim while another is uncovered). The team's shape must map onto the aim set.
- **In:** evidence-store `investigators[]` (person-indexed), aims/WP ids, entity-store. **Out:** a per-CI capability × ROPE × availability matrix + a team-composition verdict (every aim staffed by a CI whose own record and FTE support it; no unstaffed aim, no redundant pooling); gaps named and staffed or justified. (Cross-ref `number-defensibility` §1.4, `role/credit discipline` §1.5.)

### 2.12 partnership-authenticity (RUNS BEFORE §2.8 — feeds the budget)
> Numbered 2.12 to avoid renumbering, but it runs *before* §2.8 budget-math and §2.13:
> establish whether the partnership is REAL before its cash/in-kind figures enter the
> `contribution-matrix`. `author-voice.md` §5.2 WRITES the partner as "cash + in-kind, co-design,
> not fee-for-service"; this pass CHECKS that claim survives evidence — a rhetorical framing is
> not a verified relationship.
- **Does:** distinguish a **genuine co-design partnership** from **fee-for-service** or
  **letterhead-only** support, reading the evidence-store `partners[]` fields (`legal_entity`,
  `letter_commitment`, `contributions.cash|in_kind` — Agent B). Authenticity is graded from what
  the partner actually commits, not from how warmly the letter is worded.
- **Authenticity signals (evidence-backed, each raises authenticity):** partner commits **both
  cash and in-kind**; **named personnel with FTE**; platform / data / customer / site access; IP
  co-ownership or explicit background-IP terms; a prior working relationship; a **partner-specific
  (not boilerplate) letter**; and each contribution **tied to a specific aim/WP** rather than to
  the project in general.
- **Red flags (each lowers authenticity → surfaced):** a generic support letter (interchangeable
  across applicants); **in-kind ONLY** with no cash and no named resource; vague "will provide
  feedback / guidance" with no deliverable; no named contact; a contribution **not mapped to any
  aim/WP**; a commitment **conditional on a future decision** ("subject to board approval") stated
  as if already committed (reconciled hard in §2.13 against `letter_commitment.conditional`).
- **Fee-for-service tell.** The partner **pays the team to do work FOR the partner**
  (one-directional, deliverable-for-fee) rather than **co-investing in a shared research risk**.
  This is a legitimate relationship — but dressing it in co-design language is a
  **misrepresentation risk**: flag it, and either rewrite the claim to what it is or evidence the
  genuine co-investment.
- **Submission mode:** a partner asserted as *co-investing / cash partner* with **no evidenced
  cash or in-kind commitment** — `contributions` status ≠ `committed`, or no `provenance`, or no
  `letter_commitment` at all — is a **BLOCK**. **Draft mode → WARN + a `blockers.md` entry**
  ("evidence the partner's commitment before submit"), consistent with markers-two-mode §1.8.
- **In:** evidence-store `partners[]` (`legal_entity`, `letter_commitment`,
  `contributions.cash|in_kind`), aims/WP ids. **Out:** a per-partner authenticity grade
  (co-design / fee-for-service / letterhead-only) with the signals and red flags that set it;
  fee-for-service-dressed-as-co-design flagged as a misrepresentation risk; an unevidenced cash
  partner blocked (submission) or lifted to `blockers.md` (draft). (Cross-ref `author-voice.md`
  §5.2; the figures this grades are reconciled in §2.13.)
- **Example (fictional):** *ACME Analytics Pty Ltd* is written as a "co-design cash partner". This
  pass reads `partners[]`: `contributions.cash` is empty and the only in-kind line is "will
  provide feedback", the letter is boilerplate, no personnel are named, and the contribution maps
  to no aim — three red flags, no cash, no named resource. Graded **letterhead-only**; in
  submission mode the co-investing claim is a **BLOCK**, and in draft it becomes a `blockers.md`
  entry rather than shipped prose. A second partner, *BorealGrid Pty Ltd*, commits $80k cash + a
  named 0.2-FTE engineer against WP2 with a partner-specific letter → graded **co-design**, its
  figures passed to §2.8 and §2.13.

### 2.13 partner-commitment reconciliation (semantic layer; mechanized by validate_ir.py)
> Runs after §2.12 has graded the partnership real. §2.12 asks *is the partnership authentic*;
> this pass asks *does the LETTER match the APPLICATION BODY* — a distinct, self-consistency check.
> Keep them separate: a genuine partner (passes §2.12) can still have a letter whose figure,
> role, or personnel contradict the budget and narrative (fails §2.13).
- **Does:** for each `partners[]` entry, reconcile the **support letter** (`letter_commitment` —
  Agent B, the source of truth for what the letter literally states) against the **application
  body** — the `contribution-matrix` / `budget-matrix` line, the narrative figure, and the claimed
  role in the team/entity model. Four couplings, each a finding on mismatch:
  - **(a) FIGURE.** `letter_commitment.cash` / `.in_kind` must equal the `contribution-matrix`
    line **and** the figure stated in the narrative. A letter that pledges $100k against a budget
    line of $120k is a contradiction a reviewer reading both documents catches.
  - **(b) ROLE.** `letter_commitment.role_stated` (advisory vs co-investigator vs data-provider)
    must equal the role claimed for that partner in the team / entity model. A letter offering
    "advisory input" cannot back a "co-investigator" claim in the body.
  - **(c) PERSONNEL.** every person named in `letter_commitment.personnel` as committed must
    appear in the team table; a person the letter commits but the application never lists (or vice
    versa) is a gap.
  - **(d) CONDITIONALITY.** a commitment the letter makes **conditional**
    (`letter_commitment.conditional: true`, "subject to…") must **NOT** be rendered as
    unconditional `committed` in the body — the application may only claim what the letter
    unconditionally gives.
- **Submission mode:** any **amount or role mismatch on a scored budget** — (a) or (b) against a
  `contribution-matrix` line the panel scores — is a **BLOCK**; a conditional-rendered-as-committed
  (d) is likewise a **BLOCK**. **Draft mode → WARN + a `blockers.md` entry** per mismatch.
  Fail-closed: a partner carrying a cash line but **no `letter_commitment` and no `provenance`** is
  **UNVERIFIED → BLOCK in submission** (there is no letter to reconcile against).
- **Mechanized by `validate_ir.py`.** This pass is the semantic layer; its arithmetic counterpart
  is the `partner-commitment-reconciliation` check in `scripts/validate_ir.py` (Agent C, added to
  `--self-test`), which recomputes (a)/(d) mechanically across `contributions.cash|in_kind`, the
  matching `budget-matrix` line, and `letter_commitment` — **FAIL in submission, WARN in draft**,
  non-zero exit on any hard mismatch. The mechanical check owns the numbers; this pass owns the
  role/personnel/framing judgement the numbers cannot self-check. (Cross-ref §2.12
  partnership-authenticity, §2.8 budget-math, `criterion-readiness` §4.4.)
- **In:** evidence-store `partners[]` (`letter_commitment`, `contributions.cash|in_kind`, claimed
  role), `contribution-matrix` / `budget-matrix`, team table, narrative figures. **Out:** a
  per-partner reconciliation verdict (figure ∧ role ∧ personnel ∧ conditionality) with every
  mismatch named to its source; scored-budget amount/role mismatches blocked (submission) or lifted
  to `blockers.md` (draft); an unletter'd cash partner blocked as UNVERIFIED.
- **Example (fictional):** *BorealGrid Pty Ltd*'s `letter_commitment.cash` is $80k, conditional
  `false`, `role_stated: "data-provider"`, `personnel: ["<named contact>"]`. The application body,
  however, lists $95k on the `contribution-matrix` line, claims BorealGrid as a **co-investigator**,
  and its narrative names a second engineer not in the letter. This pass returns **three findings**
  — (a) $80k ≠ $95k on a scored line, (b) data-provider ≠ co-investigator, (c) the extra engineer
  is uncommitted — each a **BLOCK in submission mode**; `validate_ir.py`'s
  `partner-commitment-reconciliation` catches (a) mechanically (exit 1), and this pass owns (b)/(c).
  In draft mode the same three become `blockers.md` entries, not shipped prose.

### 2.8 budget-math validation
- **Does:** mechanically validate the `budget-matrix` / `contribution-matrix` arithmetic and every scheme rule: **per-row caps with an explicit denominator** (`of: total | total-cash | requested` — e.g. audit/overseas ≤ 10% of total *cash*, excludes in-kind), **matched-funding ratio ≥ threshold**, **phased-budget gating** (per-phase totals), **credit-vs-cash separation** (credit-request lines respect `counts_toward_total`), and **opt-in cumulative cash-flow liquidity** (`cash_flow_check`: per-FY cumulative spend ≤ cumulative cash-in).
- **In:** budget/contribution matrices, `computed` ratio gates. **Out:** pass/fail per rule with the offending cell + amount; blocks submission on a hard cap breach.
- **Example (CRC-P family):** overseas spend computed as % of total must be ≤ 10%; matched cash/in-kind must meet the co-contribution ratio the `computed` gate enforces — recompute, never trust the portal's cached total.
- **Process rule — budget is authored as data, never as prose.** Feasibility/budget content is
  written as a structured `budget.yaml` FIRST, run through `scripts/validate_budget.py`, and only
  THEN rendered to prose; the prose totals must equal the validated file. **Never hand-write a
  budget figure into prose unvalidated** — that is exactly how sub-line arithmetic slips through
  (two $2,500 items summed as "$4,000/yr"; two $7,000 trips budgeted at $7,000 total). If a number
  is in the narrative, it came from a `validate_budget.py`-passing `budget.yaml`.
- **Named-entity type consistency.** A travel/attendance line's named venue must actually be that
  kind of venue — a *conference* destination must be a conference, not a journal (e.g. "TIFS" is a
  journal, not a meeting). Cross-check named venues/orgs against their type in the evidence-store;
  a type mismatch is a flag, not a cost line.

### 2.9 compliance completeness
- **Does:** confirm every `compliance`-role field is present and mutually consistent — ethics, security, COI, DMP, foreign-interference — and that `conditional-group` triggers fired their required annexes.
- **In:** compliance fields, decision-tree answers. **Out:** presence + consistency check; missing-annex flags. (Often assessor-invisible but hard submit-blockers.)

### 2.10 novelty-boundary (new research vs scale-up)
- **Does:** draw a sharp line between (a) what is **already done/published** — including the
  applicant's *own* prior work — and (b) the **specific unresolved problem this project newly
  solves**. The project's claimed contribution must sit in (b). This pre-empts the panel's
  "does the grant fund new research, or scale-up/integration of published methods?" — the fatal
  read when aims say "extends / builds on / integrates" prior papers.
- **Red flag:** if the aims read as extending, scaling, or integrating the applicant's published
  work, EITHER reframe them around the new hard problem (name the unsolved sub-questions), OR
  argue explicitly why the scale-up/integration is *itself* the novel research contribution (with
  the specific unsolved problems it raises). Never let a project read as "more of my published work".
- **Self-overlap check (do this literally).** Cross-check each "new" sub-problem against the
  **titles and scope of the applicant's OWN outputs** in the evidence-store — not just an
  assertion of novelty. If a prior output already names it (e.g. claiming "streaming-drift
  unlearning" as new while the applicant has a published *"…Rollback for Streaming Unlearning"*),
  the claim is self-contradicted: either reconcile (the prior paper is the *starting point*; the
  new problem must be a genuinely harder object beyond it — name what is beyond it) or drop the
  "new" framing. A "new problem" the applicant has already published is the fastest way a
  specialist assessor kills the innovation score.
- **In:** aims/innovation × the applicant's own outputs (evidence-store, titles + scope). **Out:** a
  done-vs-new boundary statement, each "new" claim reconciled against the applicant's own output
  titles; aims that only extend prior work flagged for reframing. (Cross-ref defensible-primacy
  §1.2, `author-voice.md` §5 scope-frame / §8.)

### 2.11 state-of-the-art & significance (RUNS BEFORE §2.10 in reading order)
> Numbered 2.11 to avoid renumbering, but it runs *before* §2.10 novelty-boundary: field-ground the
> problem and the prior art FIRST, then §2.10 draws the done-vs-new line against that grounding.
> §2.10 guards self-overlap and unsupported "first"; it does **not** ensure the project is
> field-grounded or the problem important — that is this pass.
- **Does:** confirm significance and innovation are *field-grounded from evidence*, not asserted. Four checks:
  - **(i) independent current literature / prior-art coverage.** The state of the art is set by **external comparators**, not the applicant's own survey. Absent an independent current source, reuse §1.9 primacy's `[EXTERNAL COMPARATOR NEEDED]` marker (two-mode per §1.8 — resolved or lifted to `blockers.md` at submission, never shipped raw).
  - **(ii) an explicit unresolved gap.** The project must target a *named* open problem the current SOTA leaves open — not a vague "more work is needed". No stated gap → flag; the significance has nowhere to land.
  - **(iii) source-backed problem-significance.** Why the problem matters is carried by a **real, dated statistic** (`context_evidence[]`: `{claim, stat, source, as_of}`), per `author-voice.md` §5.1 costed-stake — a costed, sourced stake, **not** "an important problem" / "a critical challenge". An unsourced significance claim is a `[STAT — SOURCE]` marker, not prose.
  - **(iv) distinguish comparator classes.** Tag each comparator by `kind` — **scholarly** work / **commercial** alternative / **standard** / the applicant's **own prior work**. **Own-work is NOT an external comparator** (that collapses into §2.10 self-overlap); a commercial product or a standard is not a scholarly baseline. Mixing the classes lets a self-citation masquerade as independent SOTA.
- **In:** aims/significance/innovation × evidence-store `comparators[]` (`{ref, kind: scholarly|commercial|standard|own-work, provenance}`) and `context_evidence[]` (`{claim, stat, source, as_of}`). **Out:** significance + innovation both field-grounded (independent comparators cited by class, one named unresolved gap, a dated sourced significance stat); a literature-thin-but-coherent narrative flagged; unbacked significance/comparator claims marked `[EXTERNAL COMPARATOR NEEDED]` / `[STAT — SOURCE]`. (Cross-ref §1.9 primacy, §2.10 novelty-boundary, `author-voice.md` §5.1.)
- **Example (fictional):** a proposal opens "reducing energy waste in edge inference is an important problem." This pass rejects (iii) — no costed stake — and requires a `context_evidence[]` entry: "edge-inference workloads consumed ~[X] TWh in [year] ([named report], as-of [date])". For (i)/(iv) it demands an *external* baseline: the applicant's own workshop paper is tagged `own-work` and does **not** discharge the comparator; a named commercial runtime and a peer-reviewed method are cited as `commercial` / `scholarly`. For (ii) it names the open problem the SOTA leaves — "no method holds accuracy above [threshold] under [regime]" — which §2.10 then tests the "new" claim against. Absent the external comparator, the SOTA claim ships an `[EXTERNAL COMPARATOR NEEDED]` marker to `blockers.md`, not raw prose.

---

### Project-substance passes §2.14–§2.17 (RUN ONLY in `prospective-project` mode)

> Numbered 2.14–2.17 to avoid renumbering, but read as one cluster appended after §2.13.
> Where §2.1–§2.11 mostly read the `scheme.yaml` IR and the reusable evidence-store, these four
> render project SUBSTANCE from a **structured project-plan register** — the `project-plan.yaml`
> sidecar (B3, `evidence-store.md`), one per application, read by `validate_ir.py --plan`. Each
> DEEPENS an earlier pass (its coverage counterpart) and is mechanized by a named `validate_ir.py`
> check, fail-closed on the same BLOCK-vs-WARN discipline as §2.12/§2.13/§4.4. They run **only when
> `mode = prospective-project`**; outside project mode (or with no `--plan` supplied) they SKIP,
> labelled. The register is the source of truth: substance is *rendered from structure, not asserted
> in prose*, and a present-but-empty register field is a load-bearing gap, never a silent default.

### 2.14 research-design adequacy (deepens §2.3 methods-feasibility; register-driven)
> §2.3 asks *can the methods be DONE in the time with the resources*; this pass asks the harder
> question *does the design ANSWER the aim* — i.e. will the methodology actually PRODUCE the claimed
> knowledge. Feasible ≠ adequate: a design can be perfectly deliverable and still not establish what
> the aim claims.
- **Does:** for every `aims[]` entry, confirm the `design[]` register answers it — pre-empting the
  panel's "the methodology won't produce the claimed knowledge." Reads each aim's
  `success_criterion` (the measurable definition-of-done) against its design rows on four couplings:
  - **(a) COVERAGE.** every `aims[]` id appears in ≥1 `design[].aim` — no aim without a method. An
    orphan aim (stated but unanswered by any design) is the fatal gap this pass exists to catch.
  - **(b) CONTROLS.** each covered design carries `controls` — a baseline / ablation / comparator
    that *isolates the effect*. A method with no control cannot attribute its result to the aim.
  - **(c) VALIDITY.** where the aim is empirical, `validity` states `sample_size` / `power` and
    named `threats` (each with its mitigation); non-empirical aims set these `null` **with a note**,
    not silently. An empirical aim with no power/threat account is under-designed.
  - **(d) ANSWERS-AIM.** each covered design has a non-empty `answers_aim` — the explicit
    justification that this design produces the aim's claimed knowledge, not merely activity near it.
- **Submission mode:** an aim with **no design coverage** (a) **or** no `success_criterion` is a
  **BLOCK** — an unanswerable or undefined-done aim cannot be scored. A covered aim missing
  `answers_aim`, controls, or (for an empirical aim) validity is a **WARN + `blockers.md`** entry.
  **Draft mode → WARN** throughout, consistent with markers-two-mode §1.8.
- **Mechanized by `validate_ir.py` `research-design-adequacy`** (Agent C): every `aims[]` id covered
  by ≥1 `design[].aim`, every aim's `success_criterion` non-empty, every covered design's
  `answers_aim` non-empty — FAIL (submission) / WARN (draft), per aim.
- **In:** `project-plan.yaml` `aims[]` (`statement`, `success_criterion`) × `design[]` (`aim`,
  `methods`, `controls`, `validity`, `answers_aim`). **Out:** a per-aim adequacy verdict
  (covered ∧ controlled ∧ (empirically) valid ∧ answers_aim) with each uncovered/undefined aim
  named; the "won't produce the claimed knowledge" concern pre-empted. (Cross-ref §2.3
  methods-feasibility, §2.1 project-coherence.)
- **Example (fictional):** *aim-2* — "establish that [method] holds accuracy above [threshold] under
  [regime]" — has a `success_criterion` but no `design[]` row names `aim: aim-2`. **Uncovered → BLOCK
  in submission.** *aim-1* is covered but its design lists no `controls` (no baseline to isolate the
  effect) and `answers_aim` is empty → **WARN + `blockers.md`**. Once a design row with a baseline
  comparator, a power'd sample, and an `answers_aim` justification is added for each, both pass.

### 2.15 benefits-realisation (deepens §2.5 impact-pathway; register-driven)
> §2.5 verifies the *pathway* outputs → outcomes → impact has no broken link. This pass verifies each
> benefit on that pathway is actually REALISABLE — measurable, time-bound, and OWNED — because a
> traceable pathway to a benefit **no one is accountable for realising** is still aspirational.
- **Does:** for each `benefits[]` entry, confirm it is realisable rather than hoped-for, reading
  `{benefit, type, beneficiary, owner, timing, metric, preconditions}`:
  - **(a) TYPED.** `type` is one of output / outcome / impact — and the three are distinguished, with
    impact laddering from outputs (mirrors the national-benefit ladder, Group 1 intro / §2.5). A
    benefit that conflates an output with an impact over-claims its realisation.
  - **(b) OWNED.** `owner` names who realises/captures it — **a benefit with no owner is aspirational**
    by construction, no matter how large. `beneficiary` (who gains) is distinct from `owner` (who acts).
  - **(c) MEASURABLE.** `metric` gives an indicator + target — how realisation is *measured*; a
    benefit with no metric cannot be shown to have been realised.
  - **(d) TIMED & CONDITIONED.** `timing` says when (e.g. by Y3 / +2yr post-award) and
    `preconditions` name what must be true for realisation — an unconditioned benefit hides its risks.
- **Submission mode:** on a scheme that **scores a realisation / benefits plan**, a benefit with no
  `owner` **or** no `metric` **or** no `timing` is a **BLOCK** (it is asserted impact, unrealisable
  as written). **Draft mode → WARN + `blockers.md`** per benefit, consistent with §1.8.
- **Mechanized by `validate_ir.py` `benefits-realisation`** (Agent C): every `benefits[]` row has
  non-empty `owner` AND `metric` AND `timing` — FAIL (submission) / WARN (draft), per benefit.
- **In:** `project-plan.yaml` `benefits[]` (`benefit`, `type`, `beneficiary`, `owner`, `timing`,
  `metric`, `preconditions`). **Out:** a per-benefit realisation verdict (typed ∧ owned ∧ measured ∧
  timed) with each ownerless/metricless/untimed benefit flagged as aspirational; a benefits list that
  reads as a *realisation plan*, not a wish list. (Cross-ref §2.5 impact-pathway, §2.14 success_criterion.)
- **Example (fictional):** *ben-1* — "[sector] adopts the open toolkit" — is typed `outcome`,
  `beneficiary: "[sector] SMEs"`, but `owner` is empty and `metric` is "wide uptake" (no target),
  `timing` blank. Ownerless + no measurable target + untimed → **BLOCK in submission**. Reframed with
  `owner: "the partner's product team"`, `metric: "≥[N] deployments by +2yr post-award"`,
  `timing: "+2yr"`, and a `preconditions` line ("toolkit released under [licence]") → realisable, passes.

### 2.16 additionality / value-for-money (new; ties partnership §2.12 + budget §2.8; register-driven)
> New pass, no earlier coverage counterpart — it answers the two questions a demanding public-funding
> panel asks that no other Group-2 pass owns: **"why public money?"** (additionality) and **"is it
> worth it?"** (value-for-money). It reads the `additionality{}` register and cross-checks the
> §2.8 budget.
- **Does:** argue **additionality** and **VfM** from the `additionality{}` register, pre-empting
  "why fund this / why not the partner alone":
  - **(a) COUNTERFACTUAL.** `counterfactual` states what would NOT happen without THIS grant, and
    *why* — explicitly **not business-as-usual**, **not already funded**, and **not what the partner
    (or industry) would do alone** (`not_business_as_usual: true`). A project that would proceed
    anyway is not additional; public money buys nothing it scores.
  - **(b) LEVERAGE / VfM.** `leverage: {grant, co_contribution}` yields the VfM ratio
    `co_contribution / grant`. **The batch-2 partner co-investment IS the leverage story** — the
    cash + in-kind §2.12 graded as *genuine co-design* (not letterhead-only, not fee-for-service) is
    exactly what makes the public spend leveraged rather than a straight subsidy. A "co-investment"
    §2.12 could not evidence must **not** be counted as leverage here.
  - **(c) COST-PER-OUTCOME (optional).** where the scheme rewards it, `cost_per_outcome:
    {value, basis}` gives a costed denominator (e.g. grant ÷ trained researchers / ÷ SMEs reached),
    each figure held to `number-defensibility` §1.4 (value + scope + source, never improvised).
- **Submission mode:** on a scheme that **scores additionality**, a **missing/empty `counterfactual`**
  is a **BLOCK** — the additionality criterion has nothing to score. If a `--budget` is supplied, the
  `leverage.grant` / `leverage.co_contribution` figures must **reconcile with the budget totals**
  (mismatch > 1% → **BLOCK submission / WARN draft**, the same tolerance as §2.13
  partner-commitment). **Draft mode → WARN + `blockers.md`** throughout.
- **Mechanized by `validate_ir.py` `additionality-vfm`** (Agent C): `additionality.counterfactual`
  non-empty; `leverage.grant` & `leverage.co_contribution` present → reports the ratio and, when a
  `--budget` is supplied, cross-checks the two figures against budget totals — FAIL (submission) /
  WARN (draft) on a missing counterfactual or a >1% mismatch.
- **In:** `project-plan.yaml` `additionality{}` (`counterfactual`, `not_business_as_usual`,
  `leverage{grant, co_contribution, currency}`, `cost_per_outcome{value, basis}`), the §2.12-graded
  `partners[]` co-investment, and (if supplied) the `budget-matrix` / `contribution-matrix` totals.
  **Out:** an additionality statement (a genuine counterfactual, not BAU / not already funded / not
  partner-alone), a VfM ratio recomputed from co-contribution/grant and reconciled against the
  budget, an optional costed cost-per-outcome; the "why public money" concern pre-empted. (Cross-ref
  §2.12 partnership-authenticity — the co-investment IS the leverage; §2.8 budget-math — the totals
  this reconciles against; `number-defensibility` §1.4.)
- **Example (fictional):** a proposal claims strong VfM but its `counterfactual` reads "this work is
  important and timely" — that is not a counterfactual (it does not say what fails to happen without
  the grant) → **BLOCK in submission**. Reframed: "without this grant the [risky, pre-commercial]
  method stays unbuilt — the partner funds only near-market work and will not carry this research
  risk alone." Its `leverage` is `{grant: 500000, co_contribution: 150000}` → **VfM 0.30**, and the
  $150k is exactly *BorealGrid Pty Ltd*'s §2.12-verified cash + in-kind co-investment; with `--budget`
  supplied the $150k reconciles to the `contribution-matrix` total (within 1%). Additionality + VfM
  both land.

### 2.17 trigger-driven risk (deepens §2.4 risk-mitigation; register-driven)
> §2.4 checks *coverage* — that each material risk has a mitigation and a residual within row bounds.
> This pass upgrades a static register to a **live trigger→contingency register**: not "there is a
> risk; it is mitigated" but "**if X by month M → do Y, checked at Z, owned by W**". A panel scores a
> risk plan it can see FIRE, not a reassurance.
- **Does:** for each `risks[]` entry, confirm it is a monitored, actionable contingency, reading
  `{risk, likelihood, impact, trigger, monitoring, contingency, owner}`:
  - **(a) GRADED.** `likelihood` and `impact` each low / medium / high — the grade sets how hard the
    trigger/contingency requirement bites (a high-impact risk cannot be left static).
  - **(b) TRIGGER.** `trigger` is an **observable threshold** — "if milestone M slips past month 9",
    "if recruitment < N by month 6" — not a vague "if things go wrong". A risk with no observable
    trigger cannot be acted on in time.
  - **(c) MONITORING.** `monitoring` says where/when the trigger is checked (e.g. quarterly review),
    so the trigger is actually watched rather than notional.
  - **(d) CONTINGENCY + OWNER.** `contingency` is the **pre-committed action** if the trigger fires,
    and `owner` names who acts — a contingency no one owns does not execute.
- **Submission mode:** a **high-impact risk** (`impact == high`, or `likelihood == high` AND
  `impact == high`) with **no `trigger`** **or** **no `contingency`** **or** **no `owner`** is a
  **BLOCK** — the risk the panel most fears is left without a live response. Lower-graded risks
  missing a trigger/contingency are a **WARN + `blockers.md`**. **Draft mode → WARN** throughout,
  consistent with §1.8.
- **Mechanized by `validate_ir.py` `risk-triggers`** (Agent C): every risk with `impact == high`
  (or `likelihood == high` AND `impact == high`) has non-empty `trigger` AND `contingency` AND
  `owner` — FAIL (submission) / WARN (draft), per risk.
- **In:** `project-plan.yaml` `risks[]` (`risk`, `likelihood`, `impact`, `trigger`, `monitoring`,
  `contingency`, `owner`). **Out:** a per-risk verdict (graded ∧ triggered ∧ monitored ∧ contingent
  ∧ owned) with every high-impact risk lacking a trigger/contingency/owner blocked; a risk register
  that reads as "if X → do Y", not "there is a risk; it is mitigated". (Cross-ref §2.4
  risk-mitigation completeness; the coverage/row-bounds check stays there, this pass owns the
  trigger→contingency liveness.)
- **Example (fictional):** *risk-1* — "key dataset access is withdrawn" — is graded
  `likelihood: medium`, `impact: high`, mitigation "we will find an alternative". High-impact but no
  observable `trigger`, no pre-committed `contingency`, no `owner` → **BLOCK in submission**.
  Reframed: `trigger: "if the data-sharing agreement is unsigned by month 4"`,
  `monitoring: "checked at each quarterly review"`, `contingency: "switch to the [named public
  corpus] fallback pipeline already scoped in WP1"`, `owner: "the lead CI"` → a live contingency,
  passes.

---

### 2.18 traceability spine & cross-field reconciliation (capstone; ties §2.1 coherence + §2.7 capability + §2.8 budget; register-driven)
> The capstone of the project-plan register. §2.1 project-coherence checks the aims↔methods↔milestones↔budget
> chain informally; this pass makes it a **stable-id spine** `validate_ir.py` can verify deterministically —
> every activity carries an id and traces **UP** (aim→objective→task→subtask→output→benefit) and **ACROSS**
> (task→person→year→budget→evidence). Where §2.14–§2.17 each render one register block, this pass reconciles
> them into one referentially-consistent whole. Runs only in `prospective-project` with a `--plan` spine
> present (else SKIP, labelled). The register is the source of truth: the spine is *encoded, not asserted* —
> a family of "does everything line up" stylistic checks turned into a deterministic fail-closed validator.
- **Does:** render and reconcile the traceability spine from `project-plan.yaml` (`objectives[]`, `tasks[]`,
  `subtasks[]`, `outputs[]`, `validations[]`), each check-gated:
  - **(a) STABLE IDS.** every `objectives`/`tasks`/`subtasks`/`outputs`/`validations` entry carries a
    **unique stable id**, and every cross-reference resolves — `tasks[].objective`→`objectives[].id`,
    `subtasks[].output`→`outputs[].id`, `outputs[].task`→a task id, `validations[].task`→a task id. A
    **dangling** reference or a **duplicate** id is a broken spine, not a cosmetic slip.
  - **(b) ONE-TO-ONE objective↔task.** each `objectives[]` id is achieved by a `tasks[]` entry
    (`task.objective` resolves), **one-to-one where the design allows** — an objective no task delivers, or
    a task under no objective, is an orphan a reviewer reading the two lists side-by-side catches.
  - **(c) FOUNDATION-FIRST dependency architecture.** `depends_on` wires the task DAG; **≥1
    `foundational: true` task** supplies the shared theory / data / tools / measures later tasks consume,
    and the order is **acyclic** — a task must not depend on work the plan schedules after it. State which
    task is the foundation, not just that a dependency exists.
  - **(d) SUBTASK GRAMMAR.** each `subtasks[]` statement is written in the **gap→consequence→move→mechanism**
    form ("current method lacks X → causes failure Y → introduce Z → Z fixes Y") and names the `output` it
    produces — an activity described as its causal mechanism, never as a bare to-do.
  - **(e) COLOCATED VALIDATION.** each task ends with **its own** `validations[]` block — `baseline`
    (competitive comparators), `stress` (a scenario designed to expose the *targeted* failure),
    `mechanism_check` (a component-level diagnostic that the mechanism behaves as intended), `metric`
    (aligned to the failure, **not** a generic success metric), and `comparator_class`
    (`scholarly|commercial|standard|own-work`, the batch-1 comparators discipline, §2.11(iv)) — **not** a
    single late generic "evaluation" section bolted on at the end.
  - **(f) TRACES UP + ACROSS.** every task traces **UP** — task→objective→`aim` (batch-4 `aims[].id`),
    output→`benefit` (batch-4 `benefits[].id`) — **and ACROSS** — task→`person` (entity `investigators[].id`,
    the §2.7 capability owner), `years` (timetable), `budget_lines` (budget rows): the **four-way crosswalk
    task↔person↔year↔budget** must close in **both** directions (each task funded/staffed; each
    non-institutional budget row owned by a task).
- **Submission mode:** a **dangling or duplicate id**, an **unstaffed** task (no `person`) or **unfunded**
  task (no `years`), a **resource-dependent task with no budget line** (or a **budget line mapping to no
  task**) → **BLOCK**. **Draft mode → WARN + a `blockers.md` entry** per broken edge, each naming the
  specific ids, consistent with markers-two-mode §1.8.
- **Mechanized by `validate_ir.py` `traceability-spine`** (Agent C): gated on `mode == prospective-project`
  + a spine present in `--plan`; checks referential integrity (every cross-axis link resolves), no duplicate
  ids across the spine, every task with ≥1 `person` AND ≥1 `years`, and — with `--entity`/`--budget`
  supplied — each `person`→`investigators[].id` and the four-way crosswalk closing (each `budget_lines`
  entry → a budget row AND, in reverse, every non-institutional budget row referenced by ≥1 task) — FAIL
  (submission) / WARN (draft) per broken edge, with the offending ids named.
- **In:** `project-plan.yaml` `objectives[]`/`tasks[]`/`subtasks[]`/`outputs[]`/`validations[]` + the
  cross-axis links (`aim`→batch-4 `aims[].id`, `person`→entity `investigators[].id`, `budget_lines`→budget
  rows, `benefit`→batch-4 `benefits[].id`); (if supplied) `--entity`, `--budget`. **Out:** a reconciled
  spine — every activity id-stable, each objective delivered one-to-one by a task, a foundation-first
  acyclic DAG, subtasks in gap→consequence→move→mechanism form, a colocated validation block per task, and
  every task traced up (→aim/benefit) and across (→person/year/budget); every
  dangling/duplicate/unstaffed/unfunded edge blocked (submission) or lifted to `blockers.md` (draft).
  (Cross-ref §2.1 project-coherence, §2.7 team-capability — the capability owner this reuses,
  §2.8 budget-math — the rows the crosswalk closes against, §2.14 `success_criterion`; the
  `project-plan.yaml` spine block Agent B, `traceability-spine` Agent C.)
- **Example (fictional):** a plan states *obj-1* but no `tasks[]` entry sets `objective: obj-1` → orphan
  objective, **BLOCK**. *task-1* (`foundational: true`) supplies the shared measurement harness *task-2* and
  *task-3* consume via `depends_on: [task-1]`; each subtask reads "current [method] lacks [property] →
  causes [failure] → introduce [move] → [move] closes [failure]" and names an `output`. *task-2* carries
  `person: [inv-lead]`, `years: [1,2]`, `budget_lines: [bl-3]`, and a `validations[]` block
  (`baseline`, `stress`, `mechanism_check`, `metric`, `comparator_class: scholarly`). A fourth task lists no
  `budget_lines` while requesting a compute-heavy method → **unfunded, BLOCK**; a budget row `bl-7`
  referenced by no task → **orphan line, BLOCK**. In draft mode each becomes a `blockers.md` entry naming
  the id; `validate_ir.py`'s `traceability-spine` catches the referential and crosswalk breaks mechanically
  (exit 1), while this pass owns the grammar / foundation-first / colocation judgement the ids cannot
  self-check.

---

## Group 3 — retroactive-impact passes (ADDED for `retroactive-impact`)

Run these *in addition* to Group 1's evidentiary discipline (verb-tiering, anti-double-counting,
number-defensibility, char-fit still apply) when `mode = retroactive-impact`. There is **no future
project**, so **Group 2 does not run** — no budget-matrix, no milestones, no feasibility, no
requested amount. The application reports **impact already delivered**; evidence is live artifacts
and third-party attestations, scored first-class.

### 3.1 impact-evidence marshalling
- **Does:** resolve each claimed contribution to its **live artifact** — repo / deployment / dashboard / on-chain address — as a `link` × role `evidence` (**scored, not decorative `metadata`**), each carrying provenance (what the URL points to, who controls it) + an **as-of date** it was checked live. A claim with no reachable artifact is downgraded to unverifiable, not asserted.
- **In:** contribution list, evidence-store `link`/`evidence` items. **Out:** per-contribution artifact set, each `{url, provenance, as_of}`; contributions with no live artifact flagged.
- **Example (RetroPGF-shaped, fictional):** "shipped a public indexer used by N downstream apps" resolves to the repo + the deployed endpoint + a usage dashboard, each dated as-of the round census — the dashboard number *is* the scored claim, not the prose around it.

### 3.2 attestation / third-party verification
- **Does:** back each impact claim with **third-party verification** — an on-chain attestation, an independent audit, or a named external confirmation — ranked above self-report. Self-asserted impact with no external corroboration is **downgraded**, never presented as verified.
- **In:** attestation records (`declaration`/`link` × `evidence`), external confirmations. **Out:** each claim tagged verified (with the attestation pointer) or self-reported (downgraded); a verification-gap list.
- **Example:** an on-chain attestation of "audited contract, no criticals" outranks the applicant's own "we audited it"; a claim backed by neither is marked self-reported and de-emphasised.

### 3.3 duplication / Sybil / netting
- **Does:** ensure one contribution is **not double-claimed across collaborating entities** (one contribution, one primary claimant — anti-double-counting applied across *applicants*, not just fields), and **subtract prior funding already received** for that work (funding-history netting) so the round rewards *unrewarded* impact only.
- **In:** contribution↔entity map (entity-store B2), prior-funding records. **Out:** each contribution assigned one claimant + a net-of-prior-funding figure; overlap / Sybil flags where two entities claim one artifact.
- **Example:** two team members each listing the same repo as their sole contribution collapse to one primary claim with the other as contributor; a contribution that already drew a prior grant reports impact **net** of that funding.

### 3.4 freshness
- **Does:** confirm every impact metric and attestation is **dated inside the round's measurement window**. Stale numbers (a dashboard snapshot predating the window, an attestation from a prior round) are out of scope and excluded.
- **In:** each metric/attestation's date, round measurement window. **Out:** in-window / stale verdict per metric; stale items re-fetched live or dropped.
- **Example:** a "10k users" figure as-of eight months before the window opened is not evidence for *this* round; re-pull it as-of a date inside the window or drop it.

### 3.5 retroactive scoring
- **Does:** map contributions to a **past-impact rubric** — delivered impact, reach, and verification strength — **not** a future plan. There is **no requested amount, no milestones, no budget-math, no feasibility**; scoring rewards what shipped and can be verified, weighted by the round's own criteria.
- **In:** verified contribution set, round rubric. **Out:** per-contribution impact score + the evidence tier that licensed it (parallels verb-tiering 1.2), framed for the round's badgeholders/panel (see 4.1).
- **Example:** two contributions of equal prose ambition score differently purely on verification — the one with a live dashboard + an on-chain attestation outscores the one resting on self-report.

---

## Group 4 — cross-cutting passes (all modes)

### 4.1 reviewer/panel tailoring (`reviewer_model`)
- **Does:** build a `reviewer_model` for the target panel — **expertise level, jargon tolerance, scoring emphasis (which criteria carry weight), and red-flag claims that panel punishes** — then reframe the *same project* to fit it.
- **In:** scheme rubric + panel description. **Out:** a `reviewer_model` object + per-field framing notes.
- **Reframing the one project across panels:** **ARC** — significance + national benefit, ROPE-aware, restrained tone; **NHMRC** — health translation, RtO normalisation, structured track-record; **NSF** — intellectual merit + broader impacts as co-equal, US-centric; **ERC** — high-risk/high-gain frontier, PI-centric, ambition rewarded; **industry (Google/MS/Amazon)** — product relevance and open-source/data intent, concise; **internal (an ECR scheme)** — early-career trajectory and institutional fit over raw scale.
- **Two axes, don't conflate them.** The funder-family reframing above drives the `reviewer_model`'s **register / emphasis — a SOFT tuning** (tone, jargon, which criteria to lean into). It is *not* the same axis as the scheme's **assessment process** (`scheme.process[]`), which drives **HARD pipeline structure — which stages/passes actually run** (Group 5). They *compose*: an ARC-registered, panel-routed, staged scheme takes the ARC framing here **and** the §5.2 + §5.4 structural overlays. Never encode a structural stage (an EOI gate, a routing code) as a soft framing note, nor a register choice as a pipeline stage. (Cross-ref Group 5 — process-archetype overlays.)

### 4.2 prior-submission / review-response
- **Does:** for resubmissions, thread prior reviewer feedback into the draft and (where the scheme has a response field) produce a point-by-point response; align claims with what changed.
- **In:** prior reviews, current draft. **Out:** response-to-reviewers + a change-log the narrative reflects. (Integrates the `review-response` skill.)

### 4.3 COI / reviewer-management
- **Does:** populate reviewer-management fields — NSF COA relational tables, **excluded reviewers**, **suggested reviewers** — consistently with the entity-store's collaboration graph.
- **In:** entity-store relationships, COA rules. **Out:** filled conflict tables + exclusion/suggestion lists, cross-checked against co-authorship within the scheme's lookback window.

### 4.4 criterion-readiness (mechanise the blockers discipline)
- **Does:** map **every scored `criterion`** to a readiness state from the evidence actually present, so a high-weight criterion cannot pass unevidenced while "all supplied artefacts are valid". Readiness enum (canonical): **`unsupported | partial | substantiated | submission-ready`**. For each criterion, read the `rubric[].minimum_evidence` (the evidence classes required to score it) + `readiness_rule` (what must hold to reach `substantiated`) that Stage A recorded in `form-schema-ir.md`, compute the state from the IR/evidence, and report it **per criterion**.
  - **`unsupported`** — no backing evidence for a scored criterion. In **submission mode** this is a **BLOCK, not a silent skip** — a scored criterion with zero evidence fails the gate.
  - **`partial`** — some but not all `minimum_evidence` present. Surfaces as a **warning + a `blockers.md` entry** ("resolve before submit"), consistent with markers-two-mode §1.8.
  - **`substantiated` / `submission-ready`** — `minimum_evidence` met and the `readiness_rule` holds; `submission-ready` additionally carries no unresolved §1.8/§1.9 markers on that criterion's fields.
- **Mode behaviour.** **Submission/final:** a hard-required scored criterion at `unsupported` is a hard block; `partial` → warning + `blockers.md`. **Draft/internal:** may proceed with the state annotated inline (the applicant sees which criteria are thin) — parallels the two-mode marker handling in §1.8.
- **Machine-run.** This pass is executed mechanically by `scripts/validate_ir.py` (a `criterion-readiness` check, added to `--self-test`): for each rubric criterion it computes readiness from the IR/evidence and returns a **FAIL in submission mode** for a scored criterion with no backing evidence — **not a SKIP**. SKIP is reserved for genuinely optional sidecars; a SKIP that would hide a scored-criterion gap is reclassified as a FAIL. (Cross-ref markers-two-mode / `blockers.md` §1.8; the mechanized-gate paragraph below.)
- **In:** `rubric[]` (`weight`, `binding`, `minimum_evidence`, `readiness_rule`), the drafted IR + evidence-store. **Out:** a per-criterion readiness table (`criterion → state`); in submission mode, hard blocks for `unsupported` hard-required criteria + `blockers.md` entries for every `partial`; in draft mode, the same states annotated, non-blocking.

### 4.5 institutional-statement reconciliation (semantic layer; mechanized by validate_ir.py)
> Analog to §2.13 partner-commitment reconciliation, but for the **host-institution statement** — a
> third-party attestation of committed support (establishment grant, stipend top-up, salary
> shortfall, teaching relief). Runs whenever an `organizations[].institutional_support` block is
> present (else SKIP, labelled). §2.13 reconciles a *partner's* letter against the body; this pass
> reconciles the *host institution's* statement. `author-voice.md` §10 WRITES the statement in the
> institution's third-party voice; this pass CHECKS its stated total matches the parts and the budget.
- **Does:** for the `organizations[].institutional_support` block, reconcile the statement's STATED
  `total` against the parts and the budget, prove every committed item, and convert the support into
  research capacity. Four couplings:
  - **(a) TOTAL ↔ SUM(ITEMS).** `institutional_support.total.value` must equal `sum(items[].value)`
    within 1% — the stated total is held **separate** from the item sum precisely so a mismatch is
    VISIBLE (same discipline as partner `letter_commitment` vs `contributions`, §2.13).
  - **(b) TOTAL ↔ BUDGET.** if a `--budget` is present, `.total` must reconcile with the budget's
    **non-ARC / institutional-contribution** lines (the co-investment the budget declares) within 1% —
    a host statement pledging one figure while the budget's contribution column shows another is a
    contradiction a panel reading both catches.
  - **(c) PROVENANCE.** every `status: committed` item carries a non-empty `provenance` pointing to
    the host statement (or its source) — a committed figure with no attestation is **unproven**.
  - **(d) CAPACITY-CONVERSION.** reconciling the dollars is not enough — convert each support item into
    the **research capacity it buys**: `teaching-relief` / `salary-cover` / a `continuing-position`
    become **protected research time + execution capacity** (from `protected_capacity`, Agent B), not a
    bare line-item value. And check the **host-strategy↔candidate↔project triangle** is stated — the
    host's strategic reason to back *this* candidate on *this* project — so the support reads as a
    deliberate investment, not a generic top-up. This is the §4.5 semantic (framing) half; the $
    couplings (a)/(b) stay the mechanized BLOCK conditions.
- **Submission mode:** a **total mismatch** (a) or (b) > 1%, **or** a **`committed` item with no
  `provenance`**, is a **BLOCK**. **Draft mode → WARN + a `blockers.md` entry** per finding,
  consistent with markers-two-mode §1.8.
- **Mechanized by `validate_ir.py` `institutional-support-reconciliation`** (Agent D): gated on an
  `institutional_support` block present; recomputes `sum(items)` vs `.total` (>1% FAIL), cross-checks
  `.total` against the budget non-ARC/institutional total when `--budget` is supplied, and asserts
  every committed item has provenance — FAIL (submission) / WARN (draft), mirroring
  `partner-commitment-reconciliation` §2.13.
- **In:** entity-store `organizations[].institutional_support` (`items[]` value/status/provenance,
  `total`, `statement_provenance`), and (if supplied) the `budget-matrix` non-ARC/institutional
  contribution lines. **Out:** a reconciliation verdict (total ↔ sum(items) ↔ budget, every committed
  item provenanced) with each mismatch named to its source; a mismatched total or an unproven
  committed item blocked (submission) or lifted to `blockers.md` (draft). (Cross-ref §2.13
  partner-commitment reconciliation — same discipline; the `institutional_support` store block
  Agent B; `author-voice.md` §10 institutional-statement register.)
- **Example (fictional):** *ACME University*'s host statement pledges a `total` of AUD 300,000, but
  its `items[]` sum to 280,000 (establishment grant + stipend top-up + salary shortfall) — a 20,000
  gap → **BLOCK in submission** (total ≠ sum). A `teaching-relief` item is marked `committed` but
  carries no `provenance` → a second **BLOCK** (unproven). With `--budget` supplied, the statement's
  300,000 must also match the budget's institutional-contribution column; a 300,000-vs-260,000 split
  is a third mismatch. In draft mode all three become `blockers.md` entries, not shipped prose;
  `validate_ir.py`'s `institutional-support-reconciliation` catches (a) and (b) mechanically (exit 1),
  this pass owns the provenance judgement.

---

## Group 5 — process-archetype overlays (all modes)

> **The second dispatch axis.** Group 1/2/3 dispatch on `mode` (what you're judged *on*
> → register + which passes run). Group 5 dispatches on `scheme.process[]` (how the judging
> is *structured*) and runs **ON TOP of whatever `mode` already selected** — it does not
> replace a mode's passes, it adds/removes/reweights stages around them. `process` is a **set**:
> a scheme may be `staged` AND `panel-routed` AND rejoinder-enabled at once, so more than one
> overlay can fire. The closed archetype vocabulary is
> `{single-stage-review, staged, interview-gated, panel-routed, curated, rolling}`; a scheme
> declares its members in the `scheme.process[]` IR field, and `scripts/validate_ir.py`'s
> **`process-dispatch`** check (Agent C) fail-closes on an empty/unknown tag. `rejoinder` is a
> **capability** (an IR block `rejoinder: {enabled, window?, char_limit?}`), consumed by §5.1 —
> **not** a seventh archetype. Each overlay below states what it ADDS / REMOVES / REWEIGHTS
> relative to the mode-selected baseline; submission-vs-draft semantics are the standard
> fail-closed BLOCK-vs-WARN used throughout.

### 5.1 single-stage-review — default weight (+ rejoinder capability)
- **Does:** the baseline shape — one full submission, expert-panel rubric-scored. **No up/down-scale by itself**; the mode-selected passes run at their normal weight.
- **Adds (only if `rejoinder.enabled`):** a **rejoinder-prep** note — a within-round *right-of-reply* artifact. Reserve the strongest rebuttal-ready evidence rather than spending it in the first draft, and pre-identify the **2–3 claims a panel will most probe** (the over-reach candidates §1.9 already surfaced) so a reply can be drafted fast inside the `rejoinder.window` / `rejoinder.char_limit`.
- **Not §4.2.** §4.2 is *cross-round* resubmission (a new round, prior reviews threaded in); this is *within-round* reply to this round's panel. Keep them distinct — different artifact, different timeline.
- **Removes / Reweights:** nothing.
- **Out:** the mode-selected pipeline unchanged; plus, when `rejoinder.enabled`, a reserved-evidence list + a probe-anticipation note keyed to the 2–3 most-attackable claims. (Cross-ref `rejoinder{}` IR block; consistency-checked by `validate_ir.py` `process-dispatch` — `rejoinder.enabled` ⇒ `single-stage-review` ∈ `process`.)

### 5.2 staged — EOI / pre-proposal gate before the full
- **Does:** insert a **gating first phase** (EOI / pre-proposal / minimum-data) that must pass before the full is invited.
- **Adds:** an **EOI sub-pipeline** drafted to **its OWN rubric + limits** — not a compressed copy of the full. The EOI is a **triage gate**, so it **leads with the single most fundable hook** and lets the reviewer say *yes to phase 2*, not an exhaustive rubric sweep. Wire it to `submission.phases` (which must contain one of `EOI | pre-proposal | minimum-data`).
- **Reweights:** a **consistency lock** across phases — the full (phase 2) **must not contradict the EOI**: same core fields, same figures, no walked-back or upgraded claim a phase-1 reviewer would catch on re-read.
- **Removes:** nothing structurally, but heavyweight full-application passes (full budget-math, full risk-register) are **deferred to phase 2** — do not build them for the EOI unless the EOI's own rubric scores them.
- **Submission mode:** `staged` ∈ `process` with **no `EOI | pre-proposal | minimum-data` entry in `submission.phases`** is a **BLOCK** (submission) / **WARN** (draft) — the gate stage is undeclared. (Mechanized by `validate_ir.py` `process-dispatch`.)
- **Out:** an EOI drafted to its own rubric+limits leading with the top hook; `submission.phases` populated; a phase-1↔phase-2 consistency verdict (no contradiction). (Cross-ref `submission.phases` IR field, `criterion-readiness` §4.4 run per phase.)

### 5.3 interview-gated — written shortlist → live interview decides
- **Does:** treat the written submission as a **shortlisting instrument**, not the closing argument — a live interview / pitch makes the final call.
- **Adds:** a **defense-prep artifact** — anticipated questions generated **from the written case's weakest points** (the panel interviews to *probe*, so pre-empt the probe). A **Stage-F deliverable**, produced alongside the submission plan.
- **Reweights:** the written submission's job **shifts to *setting up* the interview** — open the strongest doors, surface (not bury) the ambitious claims you can defend live — rather than exhaustively closing every point on paper.
- **Removes:** nothing.
- **Submission mode:** absence of a defense-prep deliverable on an `interview-gated` scheme is a **WARN** (soft — the interview is downstream of submit), surfaced to `blockers.md`.
- **Out:** an anticipated-Q defense-prep artifact keyed to the written case's weak points + a note reframing the written submission as interview-setup. (Cross-ref §1.9 over-claim guard — its flagged claims are the likeliest interview probes; `validate_ir.py` `process-dispatch` soft check.)

### 5.4 panel-routed — classification codes route to the scoring panel
- **Does:** recognise that **routing is itself a gate** — a taxonomy/classification code sends the application to a specific assessing panel before any scoring happens.
- **Reweights:** **ELEVATE the taxonomy/classification fields** (`taxonomy-code`, FoR / RtO / directorate) to **gate-critical**. A wrong code routes to the **wrong assessors → a silent loss** (no rejection reason, just a low score from a mismatched panel) — a §-level failure, not a cosmetic metadata slip.
- **Adds:** the §4.1 `reviewer_model` must be built for the **ROUTED panel specifically**, not a generic reviewer — the whole point of routing is that a *particular* panel scores this.
- **Removes:** nothing.
- **Submission mode:** an **unset or low-confidence routing code** on a `panel-routed` scheme is a **BLOCK** (submission) — you cannot submit into an unknown panel. Draft mode → **WARN** + a `blockers.md` "confirm routing code" entry. (Structural-reminder check in `validate_ir.py` `process-dispatch`.)
- **Out:** taxonomy/classification fields elevated to gate-critical with a confidence flag; a routed-panel-specific `reviewer_model`; an unset/low-confidence code blocked (submission) or lifted to `blockers.md` (draft). (Cross-ref §4.1 `reviewer_model`, `taxonomy-code` fields.)

### 5.5 curated (light-touch) — program-officer discretion, no scored rubric
- **Does:** recognise a **discretionary, light-touch** scheme (short form, program-officer judgement, no scored rejoinder) and **DOWN-scale the machinery to match**. The failure mode here is **OVER-engineering**, not under-evidencing.
- **Removes (this is a genuine down-scale, not a warning — explicitly PERMITTED to skip):**
  - **no heavy budget-math** where there is **no scored budget** — do not build a `budget.yaml` + run `validate_budget.py` for a scheme that does not score a budget line;
  - **don't over-build the evidence store** — resolve only the evidence a short discretionary form actually rewards, not an exhaustive `minimum_evidence` sweep for criteria that don't exist;
  - **skip heavyweight passes the scheme does not score** — full criterion-readiness (§4.4), risk-register completeness (§2.4), partner-reconciliation (§2.13) run only insofar as the light-touch form asks for them.
- **Reweights:** what a curated form *does* reward is **one fundable hook + program/officer fit** over rubric coverage — the §1 discipline (never invent, honest tiering) still holds, but the *volume* of machinery drops.
- **Ties to `simplicity-first`.** This is the honest anti-over-engineering overlay: matching effort to what is actually scored is the same discipline as *"no features beyond what was asked; no abstraction for single-use"* — building a heavyweight pipeline a curated scheme ignores is the grant-writing form of gold-plating.
- **Out:** a deliberately reduced pass set — the mode's *evidentiary* discipline kept, the *heavyweight scored-artifact* passes skipped with a one-line "not scored by this scheme" note so the skip is auditable, not silent.

### 5.6 rolling — no synchronized deadline
- **Does:** recognise there is **no fixed round date** — submission timing is itself a strategic lever.
- **Reweights:** **Stage F timing becomes strategic** — submit **when the evidence is freshest** (a milestone just landed, a metric just cleared a threshold), not against a calendar date. Apply **§3.4-style freshness even outside `retro-impact` mode**: a stale figure weakens a rolling application the same way it fails a retro round.
- **Removes:** any **synchronized-round assumption** from the submission plan — no "round census date", no fixed-deadline gate logic; the plan is trigger-driven, not date-driven.
- **Adds:** nothing structural beyond the timing lever.
- **Out:** a trigger-driven (not date-driven) submission plan; freshness applied to every dated claim regardless of mode; no fixed-round assumptions. (Cross-ref §3.4 freshness, Stage F submission plan.)

---

Stage E is **not** "run tool X." It is a contract of checkable items; each must be
green before submission. The **operational method** is an **adversarial, multi-round**
review — read as a hostile reviewer looking for the one disqualifying flaw — reinforced
by a **cross-model pass (Codex)** so a second model re-derives the checks independently.
But the deliverable is the checklist verdict, item by item:

- **eligibility** — every hard gate re-evaluated green (gate-check, 1.1) at the census date; no drafted content assumes an unmet gate.
- **compliance** — all `compliance`-role fields present and consistent; conditional annexes triggered (2.9).
- **evidence-provenance** — every claim traces to an evidence-store item with `status` / `source_authority` / date; no orphan claim; verbs match their tier (1.2).
- **criterion-readiness** — every scored criterion mapped to `unsupported | partial | substantiated | submission-ready` from its `rubric[].minimum_evidence` (4.4); in submission mode no hard-required scored criterion at `unsupported` (hard block) and every `partial` lifted to `blockers.md`. Machine-run by `validate_ir.py`.
- **claim–evidence proportionality** — no guarantee exceeds its method, no legal effect exceeds the cited clause, no "first/only" rests on the applicant's own survey alone (1.9); over-reaches narrowed or marked `[VERIFY]`/`[EXTERNAL COMPARATOR NEEDED]`.
- **novelty-boundary** — the claimed contribution is the new unresolved problem, not an extension of already-published work (2.10); "scale-up vs new research" pre-empted.
- **cross-field consistency** — numbers, totals, roles, and dates reconcile across all fields; no double-count (1.3, 1.4).
- **budget-math** — all caps, ratios, phase totals, and credit/cash separation validated (2.8); zero hard breaches.
- **attachments-complete** — every `structured-upload` present in its correct sub-kind, within page/heading limits, proforma wording unaltered, filenames matching the required pattern.
- **panel-fit** — framing matches the `reviewer_model` (4.1); no red-flag claim for this panel; effort ∝ criterion weight.
- **risk-coverage** — material risks each mitigated with a residual (2.4), within row bounds.
- **portal validation dry-run** — modality-appropriate final check (char counts under limit, required web-form fields non-empty, AcroForm fields filled, hidden-required satisfied) simulating the portal's own submit-time validation before the human pastes/uploads.

**Mechanized gate.** The checkable, non-judgement items above are run mechanically by
`scripts/validate_ir.py` — the IR-level integrity gate and single pre-submit dry-run. It
**composes** the existing validators rather than duplicating their math: `charcount.py` for the
char roll-up and `validate_budget.py` for budget-math, and adds the cross-field couplings prose
cannot self-check: `allocation_sums_to` (taxonomy-code / repeating-group % rows sum to 100 ±tol),
contribution↔budget-matrix integrity (the F.2↔H.1 reconciliation; no double-count), `computed`
eligibility / co-contribution / matched-funding gates recomputed from the actual values,
conditional-annex triggers (a fired `decision-tree`/`conditional-group` answer → its required
attachment present), `stage_lock`/`submission_phase` ordering (no field edited where `locked_from`),
attachment rules (correct `structured-upload` sub-kind, filename pattern, page limit), and
**criterion-readiness** (4.4 — each scored criterion's state computed from `rubric[].minimum_evidence`;
a scored criterion with no backing evidence is a FAIL in submission mode, not a SKIP). Non-zero
exit on any hard fail. The human-judgement items — panel-fit, verb-tier audit, sensitive-content —
stay in the adversarial + cross-model reading; the gate does not adjudicate them.

A failing item returns to its Stage C pass, not to ad-hoc editing. Ship only when the
whole contract is green under both the adversarial and the cross-model reading.

At the submit gate (Stage F+), `scripts/build_manifest.py` **composes** this whole mechanical gate
into a reproducible run-audit: it runs `validate_ir.py`, hashes the inputs, records the built
artifacts and open blockers, and reduces the lot to one **fail-closed `ready_to_submit`** boolean
(true only when there is no hard FAIL, no open hard blocker, and every scored criterion is
`substantiated`/`submission-ready`). See `submission-management.md` Stage F+.

### Stage E — retroactive-impact contract

For `mode = retroactive-impact` the contract **replaces** budget-math, feasibility, and
risk-coverage with impact-provenance, Sybil, and freshness checks. Green means:

- **impact-provenance** — every claimed contribution resolves to a live artifact (`link` × `evidence`) with provenance + as-of date (3.1); unreachable artifacts excluded, not asserted.
- **attestation** — verified claims carry a third-party attestation pointer; self-reported impact is marked and downgraded (3.2).
- **no-double-claim / netting** — one contribution, one primary claimant across entities; figures reported net of prior funding (3.3); Sybil overlaps resolved.
- **freshness** — every impact metric / attestation dated inside the round's measurement window (3.4).
- **past-impact scoring** — mapped to the round's delivered-impact rubric (3.5); no requested amount, milestones, budget-math, or feasibility asserted.
- **still applies** — Group 1 char-fit, any round eligibility gate (1.1), and badgeholder/panel framing (4.1).

The budget-math, risk-coverage, and compliance-annex items **do not** run in this mode; scoring is
judgement, not arithmetic, so `validate_ir.py` here covers only char roll-up and any attachment rules.

> **Maturity note.** The `retroactive-impact` mode is **built but not yet validated on a live retro
> round** — the passes above and this contract are modelled on Optimism RetroPGF / Gitcoin mechanics
> (past-impact reporting, live-artifact + on-chain-attestation evidence, funding-history netting,
> Sybil resistance), not yet regression-tested against a real submission. Treat its output as
> draft-grade until a live round exercises it — honest, in the same spirit as an unproven modality is
> downgraded rather than faked.
