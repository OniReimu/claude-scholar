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
