# Author Voice — the register funded grant prose is written in

> The Stage-C drafting style for `narrative` / `criterion-scored` fields. `method-passes.md`
> governs *what you may claim* (verb-tiering, number-defensibility, anti-double-count); this
> file governs *how the sentence sounds*. A defensible claim in a first-draft register still
> reads as a first draft; the gap between "strong draft" and "funded" is this voice.
>
> Distilled from funded competitive-grant prose. Every example here is FICTIONAL — never
> paste a real applicant's wording; apply the pattern to the applicant's own evidence.

## 0. How to use this

1. **Pick the register from `mode`** (SKILL.md dispatch): `prospective-project` / commercialisation
   → the commercialisation lean of §1–7; `narrative-award` / fellowship → the inversion in §8.
   Shared craft (§1–4, §6–7) applies to both.
2. **Draft in that register**, applying the composition (§1), sentence patterns (§3), lexicon (§4),
   and strategic moves (§5 / §8).
3. **The method passes keep it defensible** — voice is subordinate (§6). Run the smell test (§7)
   and the worked rewrites (§9) as a target.

## 1. Composition (macro)

- **Headings ARE the scored criteria.** For a `heading-sequenced` upload, each internal
  heading is a rubric criterion; the assessor scores by heading. Restate the criterion in
  the scheme's own words.
- **Open each section with a topic sentence that names the section's job** in the scheme's
  language — no throat-clearing.
- **Paragraph three-beat:** `claim / topic → mechanism (how) → so-what (who benefits)`.
  4–7 tight paragraphs per section, one sub-point each.
- **Altitude ladder:** escalate the "so-what" — direct user → the sector → national economy —
  because the scheme scores national benefit. The same problem→layer→validate→pathway logic
  repeats at paragraph, section, and whole-document scale.
- **Push structured data to tables** (team/FTE, Gantt, budget, IP); prose *refers* to them
  ("as summarised in the team table below"), never re-types them.

## 2. Register (tone)

- **Impersonal third person.** "The project will…", "The team brings…". Almost no I/we in
  the persuasive body.
- **Confidence lives in declarative future-tense verbs**, not adjectives: *will deliver,
  will advance, is expected to reach TRL 4–5, is essential*. State outcomes; don't hedge
  every clause.
- **Evaluatives must be the defensible-strong kind**, not hype: `critical, essential,
  significant, substantial, major barrier, non-negotiable, rapidly expanding`. No
  exclamation, no rhetorical questions, no adjective stacking.
- **Insider, not explainer.** The business case speaks the assessor's vernacular fluently.
  (The *plain-language public summary* is the one place to switch to lay register — keep the
  two registers distinct.)

## 3. Sentence patterns

1. **Participial benefit-tail — the rhythm engine.** Main clause makes the claim; a trailing
   `..., V-ing X and V-ing Y` stacks the consequences.
   > *"The modular design lowers integration cost, **shortening** time-to-market and
   > **widening** the base of adopters."*
2. **Rule-of-three.** Nouns/adjectives cluster in threes: *efficient, auditable, and compliant*.
3. **Enumerated barriers `(i)(ii)(iii)`** in parallel structure — signals rigour.
4. **"not X, but Y" / "rather than X"** to fix positioning: *"not a fee-for-service build,
   **but** a co-developed translation effort"*.
5. **"Unlike existing X…, the proposed technology…"** — differentiation made explicit.
6. **Sentence-initial framing adverbials** organise the sweep: *Without the grant, … / At a
   national level, … / From an industry perspective, … / Beyond direct outcomes, …*.
7. Sentences run long (~30–40 words) and subordinated, but **controlled** — one complete
   argument each, never rambling.

## 4. Lexicon strategy

- **Mirror the scheme's own rubric words back** at it: *translation, proof-of-concept, TRL
  N→M, time-to-market, serviceable market, deployability, follow-on competitive funding,
  targeted licensing, contract research, cash and in-kind, public-good benefits*.
- **Hook the scheme's stated priorities verbatim** (national-benefit, sovereignty,
  productivity, the named strategic capability) — these are scored.
- **Keep a tight core-term set (~6–8) and repeat it relentlessly** so the reviewer leaves
  holding the thesis. A diffuse vocabulary reads as an unfocused project.
- Reach for `by design` / `embedded directly into` when the architecture *is* the differentiator.
- Use the IP/compliance boilerplate register where the criterion calls for it (national IP
  principles, background vs project IP, good-faith licensing) — it signals you know the rules.

## 5. Strategic moves (what separates funded from competent)

1. **Pick one defensible scope frame and repeat it at every altitude.** A frame that reads as
   cheaper / deployable / industry-real / hard-to-attack (e.g. "we build the surrounding
   *layer*, not the expensive core") pre-empts the obvious objection and gives the whole
   document a spine. Weld it into every section.
2. **The industry partner is the credibility anchor** (project/prospective mode). Name them
   repeatedly, always tied to `cash and in-kind`, `co-design`, `not fee-for-service`,
   platform/customer access. A named, co-investing partner is worth more than any adjective.
   *Name the legal entity that actually commits* (per evidence-store `partners[].legal_entity` —
   the signing/operating entity, not a loose group label); the authenticity of that commitment is
   an evidence-backed check (`method-passes.md` §2.12 partnership-authenticity + §2.13
   partner-commitment reconciliation), never a rhetorical flourish — the register makes the claim,
   the passes keep it honest.
3. **Ladder every claim to national / societal benefit** and drop a `public-good benefits`
   line — the scheme scores it.
4. **Market claims: assert momentum qualitatively, don't quantify what you can't source.**
   This is the highest-leverage move and it reconciles with `number-defensibility`: a
   confident *"rapidly expanding market, clear demand, high likelihood of commercial success"*
   leaves **no TAM figure to defend**. Prefer this to a quantified market size. If a number is
   genuinely required, it must be sourced or explicitly caveated — but the funded default is
   to not put a defensible-only-with-a-footnote number on the page at all.
5. **Embed risk + mitigation inline in the staged plan** (`Risks at this stage relate to…;
   these are mitigated through…`) rather than as a bolted-on section — it satisfies the
   "capacity + risk-aware" criterion in flow.

## 6. Interplay with the method passes

Voice is subordinate to defensibility, never overrides it:
- `verb-tiering` still caps the verb (a confident register does not license "led" for a
  contributing role).
- `number-defensibility` — §5.4 is its natural ally: the funded register mostly *avoids* the
  fragile number rather than caveating it.
- `anti-double-counting` — humans do sometimes restate the commercialisation pathway at two
  section-closes for emphasis; allow deliberate rhetorical repetition, but never re-count the
  same funding/output as two distinct achievements.
- `writing-anti-ai` (repo `policy/style-guide.md` + `PROSE.*`) runs at line-edit — this voice
  guide sets the target register; the anti-AI pass removes the tells.

### Register moves vs the anti-AI pass — resolve the tension
The signature moves here (rule-of-three, participial benefit-tails, "not X but Y", long
subordinated sentences) can **collide** with the repo's `PROSE.*` anti-AI rules (which flag
rule-of-three *repetition*, superficial `-ing` tails, negative parallelism, run-on length). They
are **not** adversarial — they are **sequential**: this guide sets the register; `writing-anti-ai`
strips the *tells*. The resolution:
- These are register tools used with **restraint** — **at most one signature move per paragraph**,
  and vary sentence length deliberately. A device becomes an AI *tell* only when **mechanically
  repeated** (a rule-of-three in every sentence, a participial tail on every claim).
- The anti-AI pass strips **overuse**, not principled sparing use. When a move and a `PROSE.*` rule
  genuinely conflict, **clarity + defensibility win** — drop the device, don't force it.
- So: draft in the register (sparingly), then run `writing-anti-ai` to de-tic. If the line-edit
  keeps removing the same device, that device was overused — thin it at the source, here.

## 7. Smell test — first-draft vs funded

| reads as first-draft | reads as funded |
|---|---|
| explains the field to the reader | speaks the assessor's vernacular |
| quantifies the market, then caveats it | asserts market momentum, no fragile number |
| benefits stated flatly, one per sentence | benefits stacked on a participial tail |
| partner mentioned once | partner is the recurring credibility anchor |
| so-what stops at the user | so-what ladders to national benefit |
| risk in a separate box | risk embedded in the staged plan |
| vocabulary diffuse | 6–8 core terms hammered |

## 8. Register by mode — the fellowship / track-record inversion

§1–7 lean toward `prospective-project` / commercialisation prose. A **fellowship** (mid-career
FT-style, or any `narrative-award` judged on the person) inverts several of those rules,
because **the person is the product** — the CV is not backing, it is the argument. Distilled
from funded fellowship prose; all examples fictional.

- **Front-load the claimant.** Establish the applicant as credible *before* the intellectual
  case — an Investigator/Capability + Leadership + National-Standing block comes early and
  runs long (often ~1/5 of the document). The reader must trust the claimant before weighing
  the claim.
- **Quantification is INVERTED.** Where a business case avoids a fragile market number (§5.4),
  a fellowship *quantifies the person and the field position* and does so freely: papers in
  flagship venues, students supervised to completion, grants won, cash from partners, awards,
  keynotes, editorial/committee roles. Both are defensible by the same discipline — every
  number carries a **scope** (metric + window + role) and, for the boldest, an **external
  attributor or evidence pointer**.
  > *"…the field's leading early-career researcher on five-year citation counts, as named by
  > [ranking body]"* — bounded by metric, window, and attributor, so it reads as fact, not boast.
- **Sourced eminence — the anti-arrogance move.** Never "I am excellent." Launder every
  superlative through an external authority ("named by X", "elected to the largest committee
  of Y", "published in the field's leading journal") and park the boldest personal claims
  behind a cross-reference to the evidence section. This is the fellowship's `number-defensibility`.
- **Future-leadership framing (mid-career signal).** Assert *emergence, not arrival*:
  "established the institution's first group in [area]", "a fast-growing, independent
  trajectory", mentoring and pipeline-building framed as a *funded outcome*, not a byproduct.
- **Defensible primacy via tight scoping.** Not "first [technique]" (false) but "first
  *wideband* [technique] *above [threshold]* in *any* [material]" — the qualifiers make an
  audacious "first" survive a specialist's probe. (This is `verb-tiering` applied to novelty.)
- **National-interest register replaces the value proposition.** Instead of TAM/ROI: sovereign
  capability, supply-chain resilience, verbatim mapping to the government's named priorities,
  STEM-ecosystem capacity. **Pre-empt the obvious National-Interest hole and rebut it in place**
  (e.g. "although [step] occurs offshore, the project still contributes to [priority] by…").
- **Significance over commercial.** Be comfortable that the *science* justifies the spend;
  pre-commercial ("enabling applications not yet commercially viable") is a feature, not a gap.
- **Rule-of-four scaffold + underlined thesis lines.** Objectives, significance points,
  innovation points, and tasks run as parallel numbered fours, each opening with an
  <u>underlined thesis sentence</u> that does the assessor's highlighting for them.
- **Altitude gradient.** High-altitude national framing brackets low-altitude method, so the
  generalist and specialist assessor are each served by the same document.
- **Genre-tag section openers** ("Motivation —", "Significance —", "Innovation —") make the
  surface skimmable: every paragraph advertises its function to a fast reader.

Reconciliation: commercialisation prose is **humble about the team, loud (but unquantified)
about the market**; fellowship prose inverts it — **loud (but sourced and quantified) about the
person, content to let the science, not a market, justify the spend.** Pick the register from
`mode`.

§8 sets the *drafting* register for the narrative-award body. The **presentation-layer register**
in §10 covers how the surrounding, evidence-heavy fields — the outputs-context statement, the
collaborator environment, the host-institution statement, the budget justification — are surfaced
*from the store blocks*, so read §10 alongside §8 for any `narrative-award` run.

## 9. Worked rewrites (weak → funded) — fictional

**Commercialisation register.** A market claim, first-draft then funded:
> ✗ *"We estimate the addressable market at roughly $40M/year for our tool."*
> ✓ *"The serviceable market — organisations deploying the technology under compliance
> obligations — is rapidly expanding as regulation tightens; demand is clear and the barrier
> we remove is the one blocking adoption today."*
Why: the figure invited "$40M from where?"; the rewrite asserts momentum with no fragile number
to defend (§5.4), stacks the benefit, and speaks the assessor's vernacular.

**Fellowship register.** A track-record claim, first-draft then funded:
> ✗ *"I am a leading researcher in the field and have published many strong papers."*
> ✓ *"Named the field's leading early-career researcher by [ranking body] on five-year
> citation counts, the applicant has published [N] papers in its flagship venues since the PhD,
> and — as principal supervisor — has graduated [M] doctoral students (evidence: §track-record)."*
Why: "I am leading" is bare self-praise; the rewrite launders eminence through an external
attributor, scopes every count by window + role, and points to the evidence — loud but
defensible (§8, and `number-defensibility` prong 2).

## 10. ROPE / track-record presentation register (narrative-award)

§8 sets how the fellowship *body* sounds; this section sets how the evidence-heavy fields *around*
it are surfaced. In a `narrative-award` scheme the assessor is a mixed panel — a few specialists in
the applicant's subfield, the rest generalists reading across the whole discipline. The
presentation register's job is to let the generalist read the CV in the field's own esteem terms
*without* the applicant sounding boastful, and to render every claim FROM a store block so it stays
reconcilable. Four moves, each drawn from a named block; the register makes the claim, the passes
keep it honest. All examples fictional.

### 10.1 Outputs-context = field-calibration

The outputs listing states *what* was published; the **Research Outputs Context** field teaches the
assessor *how to read it* in the field's terms. It is a distinct `narrative` field with its own char
limit — not a re-typed listing — rendered wholesale from the `outputs_context` store block
(evidence-store §B, and `method-passes.md` §1.10 which performs the render). Four sub-moves, each
one used sparingly (§6: at most one signature move per paragraph):

- **Venue-tier glossing.** Every venue named carries a tier *and* a plain-language rank, so a
  generalist need not know the subfield's pecking order: *"published at `<Flagship Conf>` (CORE A*,
  a top-three venue in `<field>`) and in `<Journal>` (JCR Q1, the leading journal in `<subfield>`)."*
  The tier is the insider signal; the plain rank does the generalist's translation. Render from
  `field_norms.venue_tiers[]` — never gloss a venue that has no store entry.
- **Authorship-convention decoding.** Where the subfield's convention differs from "first author =
  most credit", *state what the position means here* rather than silently accepting a capped verb —
  this pre-empts the "why isn't the applicant first author?" reflex a cross-field assessor brings:
  *"On co-supervised student papers the applicant appears as last/senior author per this field's
  convention, having contributed the central idea, design, and writing while the student ran the
  experiments."* This is a **decode, not an upgrade** — it is bounded, evidence-backed credit drawn
  from `field_norms.authorship_convention[]`, and it ties to the role/credit discipline
  (`method-passes.md` §1.5, authorship-convention-decoding sub-move); `verb-tiering` still caps the
  verb, so the decode explains the position, it does not promote it to "led".
- **Ranking-service attributor.** A standing-claim is laundered through an external service, not
  self-asserted: *"ranked Nth in `<field>` nationally by a ranking service (as of `<date>`)."*
  Render from `field_norms.ranking_attributor` — the service, the metric, and the as-of date are all
  the store's, so the line reads as a sourced fact (§8 sourced-eminence move), never a boast.
- **Output-clustering into named threads.** Rather than a flat list, group the outputs into ~3–5
  **named research threads**, each optionally tagged with a *tightly-scoped* primacy claim:
  *"Thread — `<named direction>`: `<N>` outputs at `<Flagship Conf>`, establishing the first
  `<tightly-scoped milestone>` in the area."* A cluster whose `primacy.attributor` is `null` is
  *statable but never written as a superlative* — say "an early contribution to" not "the first",
  exactly the defensible-primacy discipline of §8 and `method-passes.md` §1.9/§1.10. Every
  career-best id (`career_best.ids`) must land in some cluster, and the credit stays bounded by
  `contribution_summary` ("significant conceptual contribution on M of N papers") — never "all
  mine". `validate_ir`'s **`outputs-context-completeness`** check enforces both: an unclustered
  career-best id, or a cluster primacy with no attributor written as a superlative, FAILs under
  `--mode submission` (WARNs in draft).

*Worked rewrite (outputs-context, §9 style — fictional):*
> ✗ *"I have published widely at the best venues and my papers are highly cited; I pioneered several
> directions in my field and am one of the top researchers nationally."*
> ✓ *"The applicant's outputs cluster into three threads. `<Thread A>` — `<N>` papers at
> `<Flagship Conf>` (CORE A*, top-three in `<field>`), including an early contribution to
> `<tightly-scoped area>`. `<Thread B>` — work in `<Journal>` (JCR Q1) where, per this field's
> convention, the applicant is senior author on co-supervised student papers, having contributed the
> core idea and design. Across the corpus the applicant made a significant conceptual contribution to
> M of N papers, and a ranking service places them Nth in `<field>` nationally (as of `<date>`)."*
> Why: the weak version stacks four unsourced superlatives ("best", "highly cited", "pioneered",
> "top"); the rewrite glosses each venue for a generalist, decodes the authorship position instead of
> leaving it to be misread, bounds the credit to M of N, laddered a scoped-not-superlative primacy,
> and launders the ranking through a service — every clause traces to an `outputs_context` field.

### 10.2 Collaborator / mentor eminence-borrowing — the environment as ROPE opportunity

A narrative-award scores the applicant *relative to opportunity* (ROPE). A rich collaborator
environment is a legitimate part of that opportunity — but it is **their** eminence, not the
applicant's, and the register must keep that line honest. Name collaborators and mentors, each
tagged with their *own* sourced eminence (Fellow of `<Academy>`, `<Award>` laureate, "a pioneer of
`<method>`"), and ladder them by reach: **international → domestic → interdisciplinary → industry**.
The move is *"the applicant's work is valued by, and is conducted alongside, `<a world-leading
collaborator>` in `<area>`"* — the eminence is borrowed as *context* for what the applicant can
achieve, framed as the opportunity the fellowship amplifies, not as the applicant's own output. Keep
one eminence-tag per sentence (§6): a paragraph that stacks five laureates reads as name-dropping,
not environment. The honest form asserts *"places the applicant in an environment of world-leading
depth in `<area>`"* (opportunity, claimable) rather than *"the applicant is world-leading"* (their
eminence silently annexed). Where these collaborators are also entity-store people/orgs, the
eminence tag itself is evidence-backed the same way a partner commitment is — the register borrows
it, the store sources it.

### 10.3 Institutional-statement register — the host's third-party voice

The **host-institution statement** is authored in the *institution's* voice, not the applicant's —
a third-party attestation, register-distinct from the candidate's first-person ROPE. It is a
`structured-upload` (proforma / heading-sequenced) that reads as the university speaking: an
independent value gloss (*"the University regards the applicant as a strategic recruit in
`<priority>`"*), a concrete co-investment total (establishment grant + stipend top-up + salary
shortfall + teaching relief), a strategic-fit line (*"the project finds a home at `<Centre>`"*), and
a continuing-offer (*"a continuing position on success"*). It renders from the entity-store
`organizations[].institutional_support` block. Because it is third-party, keep the applicant's own
first person *out* of it entirely. The block's `total` is stored separately from `sum(items)` so any
mismatch is **visible** — hardened exactly like a partner's `letter_commitment` vs `contributions`
(the batch-2 discipline). `method-passes.md` §4.5 (institutional-statement reconciliation) and
`validate_ir`'s **`institutional-support-reconciliation`** check enforce it: the stated `total` must
reconcile with `sum(items)` and (when a budget is present) with the budget's non-ARC / institutional
contribution lines, and every committed item must carry provenance — a mismatch or a provenance-less
committed item BLOCKs submission (WARNs in draft).

### 10.4 Per-line budget-justification register

The budget justification is prose, one beat per line, and every line does four things: ties the cost
to a **specific Task**, itemizes the breakdown, defends the necessity against the obvious "why not
cheaper?" objection, and names an expected **output**. The signature defense for a travel/visit line
is the *"cannot be done by email"* rebuttal:
> *"Travel to `<a world-leading collaborator>` at ACME University supports Task `<n>`: `<airfare>` +
> `<registration>` + `<N>` days × `<per-diem>`. The exchange requires sustained in-person hours over
> co-designed experiments that cannot be conducted by correspondence, and is expected to yield one
> co-authored `<Flagship Conf>` submission per year."*
Necessity is argued, not asserted; the itemization signals the number is real; the per-year output
makes the spend accountable. The **voice** is this section's business — the **math** (row caps,
matched-funding ratio, totals) is `validate_budget`'s (SKILL.md Stage E), and the institutional
co-investment lines that appear here must reconcile with §10.3's `institutional_support.total` under
`institutional-support-reconciliation`. Keep one such defense per line (§6); a justification that
stacks the "cannot be done by email" rhetoric on every row reads as padding.
