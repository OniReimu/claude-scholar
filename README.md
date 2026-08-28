# Claude Scholar

<div align="center">
  <img src="LOGO.jpeg" alt="Claude Scholar Logo" width="100%"/>
</div>

**Language**: [English](README.md) | [中文](README.zh-CN.md)

Personal Claude Code configuration repository, optimized for academic research and software development - a complete working environment.

## News

- **2026-08-28 (v1.25.1)**: The reversed copular order — `is not A, but B` — was in v1.25's tier-A table and covered by no pattern on either card. The original rule was anchored on the byte sequence `, not `, which only exists in positive-first order; in the reversed form the comma sits before `but`, and `PROSE.NEGATIVE_PARALLELISM`'s patterns require a doubled pronoun or `not just`. Surfaced by the user asking why pre-v1.25 manuscripts kept their reversed contrasts. A sixth pattern now covers it, with a lookahead handing `not just/only/merely/simply` to the parallelism card; a concessive but-clause (*"is not complete, but the gap is small"*) locates and is cleared by judgment, same contract as tier B. Writing the corpus case found the next one: `not only … but also` was claimed by this card's exclusion note as NEGATIVE_PARALLELISM's, whose patterns also do not carry it — deliberately, it turns out, because that form is frequency-managed (twice per paper) and a single instance is not a violation. That deliberateness is now a `coverage_note` on the parallelism card, per 9d, and a corpus case pins the silence.
- **2026-08-27 (v1.25.0)**: `PROSE.NEGATION_CONTRAST` splits into two tiers, and the coverage gap that let it under-report becomes visible everywhere it exists. The card named three constructions but carried one pattern, so a full manuscript reported **15** contrast constructions where **38** existed — and the largest missing class, `rather than` at 17 instances, was the very form the card warned in prose against reflexively rewriting `, not Y` into. The card had *declared* the omission ("legitimate uses are too common for a hard regex"), which was a defensible call in isolation and the wrong trade once tiering was available: **tier A** (copular contrast, `X is A, not B` / `neither … nor`, a relative of `PROSE.NEGATIVE_PARALLELISM`'s `It is not X, it is Y`) is zero tolerance, **tier B** (`rather than` / `instead of` on a verb) is advisory, because there the exclusion often is the claim — *"we mark X as unavailable rather than guessing its timing"* is a methodological choice, not a flourish. A tier-B hit being advisory dissolves the precision objection, so all five forms are now located mechanically. Reverse guardrail, written into the card: **a plain negative predicate with no positive counterpart is not in scope** — *"that difference is not an effect estimate"* — 5 of 16 hits on the measured manuscript were this shape, and a rule that eats them gets switched off. Behind the specific fix, a general one. Probing every `lint_script` rule against the forms its own Requirement enumerates found the same narrowing in `PROSE.SUPERFICIAL_ING_SUFFIX`, `PROSE.ABSTRACT_AGENCY` and `PROSE.TRAILING_AFTERTHOUGHT` — all three **had declared it**, in a Check section an executor working from lint output never opens. Their patterns were left alone (their reasoning holds and no measurement contradicts it); instead a `coverage_note` field now states the gap where it matters, lint prints it under the rule header, and `validate.sh` **9d** enforces the correspondence in both directions: declare a gap in prose without the note, or carry a note the card never explains, and validation fails.
- **2026-08-27 (v1.24.0)**: `PROSE.SEMICOLON_RESTRICTION` gains a second remedy, because the first one manufactured violations of a rule whose own fix undid it. Applied across a full manuscript (47 semicolons, 39 measurable), "split into two sentences" left a second clause with a median of **10 words**, under ten in **19/39** cases and under eight in **8/39**; the worst instance, *"The point-estimate rule passes. Interval equivalence does not."* — four words plus four — is precisely the shape `PROSE.THEATRICAL_SPLIT` names as a violation, **and that card's prescribed fix is to merge back with `but`/`yet`/`, and`**. The two cards were pointing at each other: this one forced the split, the other demanded the join, and the wording here ("do not swap the semicolon for a comma") had closed the exit. The wording conflated two different moves — `;` to a **bare** comma is a comma splice and stays banned; `;` to a comma **plus a conjunction** introduces a word that carries meaning and is a syntactic change, not a punctuation swap. The remedy is now two-tier: **① syntactic change** (subordination, relative clause, or comma + conjunction — aligned with `PROSE.CAUSAL_CONNECTIVE`'s existing ladder for the same phenomenon under different punctuation) and **② split**. Tier ① requires four conditions at once: second clause under ten words, a contrast/concession/complement relation where the conjunction is not glue, a split that would leave an echo fragment, and at most three commas in the merged sentence — that last one measured, after a sentence carrying two three-item lists merged to five commas and traded a semicolon violation for a `PROSE.COMMA_OVERUSE` one. Parallel specification lists stay split; 32 of the 39 instances still take tier ②, so ① is the minority case and not a new default. Reverse guardrail: conjunctions are distributed by meaning, never standardised — a manuscript uniformly on `while` has swapped a semicolon fingerprint for a `while` fingerprint and flattened its sentence-length variance. No `fix_patterns`, for the same reason `PROSE.CAUSAL_CONNECTIVE` carries `autofix: none`: the choice of conjunction depends on the semantic relation, and a mechanical substitution changes the meaning, which is worse than leaving the semicolon.
- **2026-08-27 (v1.23.0)**: Re-anchoring gets a distance dimension. A section that turns on a construct defined several sections earlier — an RQ set, a named term, a framework stage — may restate it, and that is ordinary human writing rather than the self-similar redundancy `PROSE.FRACTAL_SUMMARY` exists to catch. The card had no notion of distance at all, so a paragraph two lines above the heading it previews and one three sections downstream were judged identically. The fix landed in **`claim-architecture-review` P2 first**, because that is where the damage is systematic: the ledger sees only "one proposition, two homes" and is built to compress exactly that, so it would collapse every cross-section re-anchor, while a line edit only meets one by chance. P2 now keeps **re-anchor homes**, tagged `(re-anchor)` in `other-homes` with the canonical home recorded beside them — an unexplained survivor reads the same as a missed duplicate. Two conditions, both required: the re-mention must be **self-contained** (readable without holding the label→content binding — *"The study asks three questions. First, what does a reviewer's visible action contain?"* qualifies; *"RQ1 establishes which public events can be interpreted"* does not, because it presupposes the very binding the re-anchor was meant to discharge, and a label ping never qualifies at any distance) and **far enough that the reader would otherwise turn back**. Deliberately **no numeric threshold**: a number turns the question into "count the intervening sections" when it is really "would the reader have to go back and look", and every threshold-on-judgment this repository has tried has ended up measuring the instrument. The unit is sections, never pages — page positions move with layout. Distance defeats the redundancy collapse only; it exempts nothing from `PROSE.SEMANTIC_IDLING` form A, since a recap that asserts nothing is empty however far it sits. The card's existing Survey/SoK taxonomy-navigation exemption turns out to be a special case of exactly this rule, hard-coded to one genre. No pattern was added — self-containment and distance are not regex-decidable — and the corpus gains the 2x2 as four precision cases that lock the five existing patterns to silence on all of them.
- **2026-08-27 (v1.22.0)**: Three fixes that came out of reading the pipeline's own "clean" output critically. **`PROSE.SENTENCE_LENGTH` was silently under-firing**: it counted words with `[^.!?\s]+`, so any dot-bearing token acted as a sentence boundary and `$\beta=0.9$` split a 55-word sentence into runs of 30 and 24, both under threshold. Decimals live in exactly the sentences this rule exists to catch, so the under-count was systematic. The word class now admits dots that are not followed by whitespace, wrapped in an atomic group because the same nested-quantifier shape already caused catastrophic backtracking in `PROSE.COMMA_OVERUSE`. Abbreviations stay invisible and are recorded as an `@XFAIL` case: `i.e.` is byte-for-byte a full stop and no regex can tell them apart. **New rule `PROSE.SEMICOLON_RESTRICTION`** bans semicolons in body paragraphs — two independent clauses are split into two sentences, not re-punctuated. Implemented as a builtin because a plain pattern fires on `p(y \mid x; \theta)`, and 7–16% of semicolons in the reference corpora sit inside inline math; list items, `\;` thin spaces and verbatim environments are stripped too, and the cleared counts are reported. The card states plainly that this is an **author-chosen style constraint, not a validated tell**: the density gap (an author's own 2.62 → 5.60 per 1000 words; arXiv 1.32 → 1.80) is recorded but the corpora are not comparable and no blind adjudication was run. **`PROSE.EM_DASH_RESTRICTION`'s remedy was wrong** — it listed "comma parenthetical" as an approved substitute, so `--- a 12.5x reduction` became `, a 12.5x reduction` and the trailing tail the rule exists to remove survived intact. Both cards now share one criterion: swapping punctuation is not a fix, the structure must change. The new rule immediately fired on four cases in this repository's own corpus, including the sentence written as `PROSE.CAUSAL_CONNECTIVE`'s model *correct* rewrite; those and two Pass examples were corrected. The pipeline fixture's trap paragraph carried an incidental semicolon and a 55-word sentence, which put it in contradiction with the new rules — it is now clean on every axis except the one under test, and the reference run was re-recorded.
- **2026-08-27 (v1.21.0)**: New suite `policy/test-pipeline.sh` — the first test of what the rules do *together*. The three existing suites each check one rule or one mechanism; none of them can tell whether `claim-architecture-review` → `writing-anti-ai` removes what should go, keeps what should stay, and runs in that order. A fixture is a draft with planted defects, planted false-positive traps, and a sealed key written before the run; the agent half is manual (`PROMPT.md`), the scoring half is deterministic — three assertion verbs (`GONE` / `KEPT` / `VERBATIM`) plus a check that every number in the output already existed in the input, so a fabricated figure fails regardless of the assertions. The seed fixture is a 378-word section carrying five structural defects, five line-level ones, and three traps that must survive untouched; it scores 22/22, with the trap paragraph — whose `because` clause introduces an independently measurable quantity — surviving byte-identical. CI cannot run the agent, so it scores the recorded reference output, which keeps the assertions from rotting as rules change, and it also asserts that the **unedited draft fails**: a scorer that passes everything reports nothing. That negative control immediately earned its place. Two assertions were labelled as catching the ordering trap (a line-level hit inside a paragraph the structural pass should have removed first); feeding in a line-only edit showed they pass in exactly that case, and the real guard is the paragraph's own topic phrase. Labels drift toward what you meant rather than what they test.
- **2026-08-27 (v1.20.0)**: `PROSE.SEMANTIC_IDLING` gains a **Rewrite contract**, from testing a supplied specification's own gold rewrites against the rule. Ten gold rewrites for ten wheel-spinning paragraphs were blind-judged alongside ten real sentences: **6 of the 10 still violate**. Four remain form A (`Our framework consistently outperforms baseline methods on standard benchmarks` still names no baseline, benchmark, or margin) and two remain form B (`Minimizing intermediate computation time reduces overall end-to-end inference latency` — the two quantities are one quantity) — and those two are precisely the segments the specification had **diagnosed correctly** as circular, then rewritten circularly. The failures cluster where no proposition survived; the passes cluster where one did. So: emit a rewrite only when a proposition survives, preserve it 100%, and when nothing survives emit deletion or the referral instead, because compression purifies a paragraph that has content and merely shortens one that does not — and short emptiness reads like a conclusion. The **compression ratio is rejected as a target**: 75–85% is what is observed after filler is removed, not a number to aim at, and aiming at it creates pressure to cut content to hit the quota — the same reason the Check refuses thresholds. Also corrected: the verdict is **per paragraph, not per sentence**. Extraction is sentence-by-sentence, but isolated single sentences over-flag systematically — 3/10 false positives on real published prose at sentence granularity against 0/30 at paragraph granularity, the casualties being topic sentences cashed out by the next line. The specification's ten diagnosis labels were **not adopted**: they collapse to four mechanisms, two of which belong to `PROSE.RESTATEMENT_DILUTION` and `PROSE.ELEGANT_VARIATION`, so importing them would be ten names for four things spread across three cards.
- **2026-08-27 (v1.19.1)**: `PROSE.SEMANTIC_IDLING` acceptance eval on ten supplied wheel-spinning paragraphs, blind-mixed with ten unseen paragraphs from a real author's pre-GPT published papers so the judge could not infer the base rate: **10/10 flagged, 0/10 false positives**. The split is graded rather than blanket — seven `escalate` (no sentence survives, so the whole paragraph goes to `claim-architecture-review` P1) against three `flag-B`, and the three are exactly the paragraphs whose opening sentence does assert something, with the circularity downstream. Two corrections came out of it. Future Work / Conclusion boilerplate is now **explicitly not exempt**: the judge had to derive that from first principles, and "paving the way for progressive improvements in subsequent research endeavors" is interchangeable with any paper's future-work section, which is form A by definition. And a limitation is recorded: form B **over-fires on standard mechanisms**, calling `because the representations remain invariant across distribution shifts` circular when invariance → OOD reliability is a genuine causal claim. The test is now stated as a question — does the explanans introduce an independently measurable quantity? An unmeasured mechanism claim is vagueness, not idling.
- **2026-08-27 (v1.19.0)**: New rule `PROSE.SEMANTIC_IDLING` — every sentence must add a falsifiable proposition. The request was for "wheel-spinning" prose, the most-perceived AI tell; the survey found it is not one phenomenon but three, and the largest one already has an owner. Synonymous restatement is `PROSE.RESTATEMENT_DILUTION` (proposition layer) plus `PROSE.ELEGANT_VARIATION` (term layer) plus `claim-architecture-review` P2 (cross-section). What fell through every card is a **long** meta-narrative sentence that previews nothing and names nothing — `PROSE.FRACTAL_SUMMARY` only catches section-position previews, `PROSE.ANNOUNCEMENT_SENTENCE` only **short** label sentences, `PROSE.FILLER_PHRASES` only listed idioms — and **circular attribution**, where the `because` clause restates its own consequent. Both share one test (what does this sentence assert that could be false), so they are one card with two failure shapes, A and B. Restatement says a proposition twice; idling says none at all, which is why the deletion test does not transfer: an empty sentence also costs nothing to delete, but the fix is to supply a fact, not to delete a duplicate. The card **forbids metric proxies** — no embedding cosine, no filler-token ratio, no proposition-density threshold — on this repository's own evidence that a judgment behind a threshold measures the instrument: the structural-repetition signal read 16x self-judged and 1.22x under fresh blind adjudication. Anti-AI and `claim-architecture-review` are genuinely coupled here: a paragraph whose sentences are *mostly* empty is escalated whole to P1, which now carries the marker and decides `merge`/`delete`, because making each empty sentence more specific only produces better-sounding filler. Blind evaluation, 26 unlabelled paragraphs judged by a fresh model: **20/20 clear** on a real author's pre-GPT published papers, **4/4 caught** on constructed violations, **2/2 clear** on exemption traps (a cashed-out topic sentence and an ethics statement). The informative part is not the recall but the three deferrals — the judge routed a short label sentence to `ANNOUNCEMENT_SENTENCE`, a section preview to `FRACTAL_SUMMARY`, and an overlapping pair to `RESTATEMENT_DILUTION` rather than claiming them.
- **2026-08-27 (v1.18.0)**: `PROSE.ADHOC_COMPOUND_MODIFIER` becomes a three-way verdict, because a binary one demonstrably fails in the middle. Judged against a set of target paragraphs, the binary rule over-flagged `gradient-norm-dependent` (clear but clumsy) and missed `pre-training/fine-tuning` (standard but improvable) — wrong in both directions at once. The verdicts are now **flag** (a coinage the reader must stop to decode; two concrete rewrites of different kinds), **hint** (grammatical and clear, merely clumsy; one line, the author may ignore it) and **clear** (an established term, **which must be reported with its prior source**). Cleared items are now reported too: reporting only violations leaves the author unable to tell whether `read/write` was checked and passed or never seen — silence is not cleanliness, a lesson this repository has now learned three times. Slash junctions (`load-balancing/routing module` — one module doing two things, or one of two modules?) fold into the same rule rather than becoming their own, because the measurement does not support a separate one: arXiv sources go 0.52 → 1.04 per 1000 words across the eras, and a local author's pre-GPT manuscripts run *higher* than their current drafts. Standard pairs clear; only genuinely ambiguous junctions flag. Paragraph length was investigated and **rejected**: median length is 57 words in both eras, means 74 against 71, CV 0.86 against 0.80. What the request was actually about — a two-sentence paragraph that belongs in its neighbour — is `claim-architecture-review`'s existing `merge`/`split` verdict, so `writing-anti-ai` now routes it there and states plainly that paragraph length itself is not a tell.
- **2026-08-27 (v1.17.0)**: Sharpened `PROSE.ADHOC_COMPOUND_MODIFIER` against a requirements spec, and one of the additions turned out to matter more than the rest. **`-based` is a construction, not a coinage**: `X-based` is the compressed form of "based on X", it composes freely, and both eras use it heavily — 22 of the 25 hapax compounds in the 2019–2021 sources end in it, against 43 of 83 in 2025–2026. Including it dilutes the separation from 16x to 4x, so it now sits outside the default suffix set, recoverable with `LINT_ADHOC_INCLUDE_BASED=1` since `concatenation-based` shows it can still be coined awkwardly. **The 16x is not quotable as an effect size** — the pre-GPT side has three non-`-based` hapax compounds in total, and the card says so. Three further mechanical refinements: findings are raised only in **attributive position** (`a model-agnostic estimator`, not `the estimator is model-agnostic`), which drops 8% of hits with no loss of separation; a compound followed by an acronym definition or written with every segment capitalised is cleared as an explicit naming act; and a multi-part left element (`out-of-distribution-driven`) is tagged as the highest-risk shape rather than filtered on, since it covers only a tenth of hits. On the judgement side, clearing a compound as an established term now **requires naming its prior source** — a claim a reader can check, rather than an assertion they must accept — and the rewrite guidance routes by intent (compression shortcut → verbalise; concept being introduced → name it explicitly or use a conventional collocation) instead of a list of priorities to try. Widening the frequency window to two occurrences was measured rather than assumed: precision is unchanged (~50% either way) and recall rises a third, so it is recorded as adoptable but ranked below the suffix work, which is what actually addresses the false positives.
- **2026-08-27 (v1.16.1)**: Disclosed the limitation inside `PROSE.ADHOC_COMPOUND_MODIFIER` rather than leaving it implied. The rule's mechanical half — counting compounds that appear exactly once — returns the same number whoever runs it. Its other half, deciding whether a flagged compound is an established term of the field, is an LLM judgement from training knowledge: not reproducible across models or versions, bounded by a knowledge cutoff, and weaker in niche subfields. The card now says so, and carries a self-caught misjudgement as evidence: `brokerage-oriented` was listed as a coinage although *brokerage* is established in network science. The judgement direction is therefore conservative — when unsure, do not flag, because a false positive on a real field term costs the whole rule. Two mechanical alternatives are recorded with their status: a vocabulary built from the pre-GPT corpus was tested and is insufficient (56 papers yield 89 types and miss `agent-based`, `model-agnostic`, `sharpness-aware`); checking the paper's own bibliography is promising but was not successfully measured and is claimed as nothing.
- **2026-08-27 (v1.16.0)**: New rule `PROSE.ADHOC_COMPOUND_MODIFIER`, and the only signal from a day of measurement that survived scrutiny. Hyphenated compound modifiers — `X-based`, `X-aware`, `X-driven` — are ordinary technical English; what is not ordinary is coining one and using it once, which charges the reader to decode `community-shift-aware` and never repays the cost. Measured across 40 arXiv sources, compounds appearing exactly once run 0.16 per 1000 words in 2019–2021 against 0.48 in 2025–2026, a 3.1x gap. **Frequency, not construction, is the test**: a local pre-GPT blockchain manuscript carries these at 7x the arXiv baseline, almost entirely from `blockchain-based`, `sharding-based` and `PBFT-based` used repeatedly — scoring by volume would condemn an entire field, scoring by one-off use does not. That counter-example is where the rule's criterion came from. Because the discriminator is a per-document frequency, no per-line regex can express it, so this ships as a `lint.sh` builtin rather than as `lint_patterns`. **What makes this one credible is that it needs no judgment.** Two other candidates measured the same day did: `, so` connective density and structural repetition (restated propositions, re-enumerated sets) separated the two eras by 1.00x and 1.22x respectively — and the structural result had first appeared to separate them 16x when the author of the rules also did the judging, collapsing to 1.22x under an independent blind adjudication. A hapax count is mechanical; it returns the same number whoever runs it. It is still not an AI detector: the distributions overlap heavily and no single paper can be classified. Known false-positive class, kept visible in the corpus rather than suppressed: an established field term that happens to appear once (`sharpness-aware minimisation`) is surfaced by the builtin and cleared at the judgment layer.
- **2026-08-27 (v1.15.1)**: Closed the five doc-enforced rules that `writing-anti-ai` declared in its table and never executed — `ELEGANT_VARIATION`, `FORMATTING_RESTRAINT`, `ANAPHORA_ABUSE`, `GERUND_FRAGMENT_LITANY`, `SHORT_PUNCHY_FRAGMENTS` — and, more usefully, made that class of gap machine-detected. A rule with `enforcement: lint_script` has a regex backstop whatever the skill body says; a `doc` rule has none, so a skill that names it only in a table claims coverage it does not have. `validate.sh` section 9b now checks this, opt-in per skill via `<!-- policy-table:checklist -->`, because a Policy Rules table is sometimes a checklist and sometimes a catalogue — `using-claude-scholar` indexes all 101 rules for Codex discovery and executes none. Writing the check surfaced something worse: `set -eo pipefail` means one `grep` that legitimately matches nothing ends the run mid-section, and a truncated run prints **fewer** `FAIL:` lines than a complete one, so the CI gate read the early exit as an improvement. validate.sh now counts the sections it ran against the number defined in the file, and CI fails when the summary is missing rather than counting whatever printed before the crash. `ELEGANT_VARIATION` was the one worth writing out: it is the constraint two other fixes depend on — `RULE_OF_THREE` names an enumerated set and requires that name be used consistently, and `INFORMAL_VOCABULARY` replaces wording using terms the manuscript already has.
- **2026-08-27 (v1.15.0)**: New rule `PROSE.CAUSAL_CONNECTIVE`, and an evaluation that changed what the rule says. The observation: `X, so Y` is the causal connective of speech, and using it discards the distinction the formal set carries — `therefore` (entailment), `hence` (continues the clause just established), `thus` (by this means), `consequently` (observed outcome). Measured against a pre-ChatGPT baseline built from arXiv sources: `, so` at 0.18 per 1000 words against 2.16 for the formal set, a ratio near 12:1, and that baseline replicates on a held-out set of different categories and months (0.28 against 2.18). Local drafts run 15–25 times the baseline density. **Then the evaluation contradicted the framing.** Held-out categories (cs.SE, stat.ML, cs.DB) show only a 1.3x rise by 2026 against 3.4x in cs.LG/CR/CL, and two month-windows in the same categories differ by nearly 2x — so the drift is venue-specific and single-window figures are not point estimates. More decisively: 42 flagged sentences drawn equally from pre-GPT papers, 2026 papers and local drafts were adjudicated blind, and the locator was perfect (42/42, no non-causal `so`) while **instance-level discrimination was nil and then inverted** — under a tightened criterion, pre-GPT instances were judged worth changing at 64% against 29% for drafts. Any single `, so` is about equally improvable whatever wrote it; only the density differs. The rule is therefore a causal-precision rule, not an AI tell, and it flags only three diagnosable classes — a design choice dressed as an inference, a causal claim the evidence does not support, and a proof step where `hence`/`thus` is the field convention. Everything else stays, because "could be more precise" would edit pre-GPT-quality prose just as eagerly. Applying it to a real section moved that rule 5 → 0 with **no change to any other rule**, four fewer commas, and the formal connectives spread across three words rather than collapsed onto one.
- **2026-08-27 (v1.14.1)**: CI had been red since 2026-08-17, on three FAILs that only ever appear on the runner. `deprecated_by` resolvability tested the successor with `-e`, which follows symlinks — and three skills are symlinks into vendor submodules, so a checkout without them leaves the links dangling and the successors unresolvable. The tree was missing content, not the reference; `-L` now answers the same in either tree, and the workflow checks out submodules so the rest of the run sees the real one. Reproduced in a submodule-less clone before and after.
- **2026-08-26 (v1.14.0)**: Two test suites that measure what the existing ones never asked. `policy/test-lint.sh` tests lint *mechanics* — flags, fix emission, exit codes — and never asks whether a rule fires on the right sentence; `validate.sh` checks registry invariants and never reads prose. **`policy/test-corpus.sh`** runs 88 annotated fixtures (single sentences, term-of-art near-misses, and whole paragraphs) and reports recall misses and false positives per case. Precision cases carry the weight: over-firing is this engine's documented failure mode, and a rule that flags every threat-model sentence gets switched off by the author. **`policy/test-referrals.sh`** machine-checks the referral graph that was hand-verified until now — destination exists (R1), destination executes the rule it is handed (R2, with recorded waivers for word-level rules and for vendored submodule skills that cannot carry markers), and the referral names the minimum operation to run rather than only a skill name (R3, scoped to destinations where a bare name costs a whole-document run). Writing them surfaced six defects. **Byte-wise character classes**: on the perl path — the one macOS resolves to — `[→←↔]` matched every em dash, curly quote and ellipsis, because the pattern arrived from the environment as bytes. CI runs GNU grep and never saw it, so the corpus now runs under both engines and `LINT_ENGINE` forces either. **Commented-out prose was linted**, so an author's parked alternative phrasings produced findings no edit could clear. **A YAML comment inside a `lint_patterns` block silently truncated it**, disabling every pattern after it with no diagnostic; `validate.sh` gained section 4d to make that an error. **Sentence-initial leaks**: case-sensitive patterns missed `A lot of`, `Significantly`, `Things`, `Dramatically` and `A number of` at exactly the position where informality shows up. **`harness`** was flagged as AI lexicon even as the noun every evaluation harness in the field is called; it is now verb-only. **`PROSE.NEGATIVE_PARALLELISM` required the contraction** (`It's not X, it's Y`) and so never fired on the uncontracted form that academic prose actually uses. Two more bare-name referrals were found and fixed, the same class as v1.13.4's.
- **2026-08-26 (v1.13.4)**: Made the referral path work for the workflow people actually use. Drafting (`ml-paper-writing`) is a zero-to-one activity; on an existing manuscript the entry point is `writing-anti-ai`, invoked section by section, and everything structural reaches the author only if that skill refers successfully. It did not, in two ways. **Coverage**: `PROSE.RULE_OF_THREE`'s cross-section case carried no referral at all (the boundary had been added to the rule card the day before and never synced to the skill) and `PROSE.SELF_UNDERMINING`'s was indirect. Both are now explicit. **Reachability**: a referral that names a skill and stops reads as "go run a four-pass whole-paper audit" when the author asked to clean one section. Both sides now state the minimum — `claim-architecture-review` documents a **targeted entry** (spine present → P1 on that section only; absent → the cheap P0, which reads abstract, intro, headings and topic sentences but no body prose, then P1; P2/P3 only for a global plan), and `progress.md` makes a targeted run a down payment rather than throwaway work. **Ownership**: arch-review was the declared destination for five rules' structural halves while carrying markers for none of them — it went from 1 marker to 6, each attached to the pass that actually performs the check, on the principle that *a skill carries a rule's marker when it executes that rule*, not when it is merely related. `EXP.EXPERIMENT_ROLE` now runs in P1 against the spine for Results sections, with its redesign → demote → delete ladder mapped onto the skill's existing verdicts — previously it fired only at draft time and in self-review, so on a revision path it never ran at all. P3 gains a second question: not just whether the spine closes, but whether it is the proposition the strongest evidence points at, since a paper can close perfectly around a claim its own data does not support.
- **2026-08-26 (v1.13.3)**: Closed the seam v1.13.2 left open and hardened the validator. `PROSE.RULE_OF_THREE`'s new "the same set may not be enumerated twice" criterion had no cross-section owner: the rule scoped itself to paragraph and section but never said what happens when a set is enumerated in Method and again in Discussion, while `claim-architecture-review`'s information ledger is keyed on *propositions* and mentioned enumeration nowhere. Both sides assumed the other had it. The rule now states the boundary in the same words `PROSE.RESTATEMENT_DILUTION` uses (within-section is the line edit's, cross-section is the architecture pass's), and the ledger now treats **an enumerated set as one information unit keyed by the set** — `{CNN channels, MLP hidden units, attention coords}` is one info-key, not four — so a second enumeration elsewhere collides on `lookup-before-create`. Separately, `validate.sh` Section 1 was forking `grep` fourteen times per card, about 1400 forks over the rule set; under heavy system load an occasional failed fork was indistinguishable from a missing field, so the validator reported content errors on correct files. Field presence is now tested with bash pattern matching (zero forks), an unreadable frontmatter is reported as its own distinct failure, and a run costs roughly half the time. **A validator that intermittently fails on correct input teaches its users to re-run until green, which is worse than not checking.**
- **2026-08-24 (v1.13.2)**: `PROSE.RULE_OF_THREE` widened from triads to **enumeration density and repetition**, after a real methods paragraph exposed two holes. Only one of its three lists was a triad; the other two (four items each) escaped the rule by definition, and the sole mechanical hit was `PROSE.COMMA_OVERUSE` — a side-effect catch that reports commas, not enumeration, so an author following it edits the wrong thing. Worse, `writing-anti-ai` instructed *"Rule of three: prefer two or four items"* — inherited from general anti-AI advice where a four-item list defeats the triad signature, but in academic prose it **produces** longer enumeration walls, and it contradicted the card's own "more than three → enumerate". The rule now carries four criteria: ≤1 triad per paragraph (unchanged); **the same set may not be enumerated twice** — name it at first mention and refer to the name afterwards; inline caps split by item length (≤4 short items, ≤2 multi-word noun phrases), so `expand, duplicate, reorder, or rescale` stays inline while four long noun phrases do not; and ≤2 list-carrying sentences per paragraph. The reverse guard leads the card: **enumeration is legitimate in technical prose, and the fix is name-then-refer, never dropping items** — `PROSE.COMMA_OVERUSE` explicitly must not be satisfied by deleting a list item. **Behaviour change** on the second and fourth criteria. The ID is retained for reference stability with a note that its scope now exceeds its name.
- **2026-08-23 (v1.13.1)**: Routed three line-editable pieces of the narrative-agency principles into `writing-anti-ai`, where a draft scan can actually fix them. `PAPER.OUTCOME_LOGIC` had zero markers there — it was wired at write time and at review time, so chronology leaks were *found* in review with no line-edit step to *fix* them. New §8b covers the three sentence-level forms (`we first tried X, which did not work…`, `initially we used A but later switched to B`, `in an earlier version of this work`) with a deletion test, an explicit no-regex instruction (`first`/`then`/`initially` are legitimate far more often than not), and the boundary that matters most: only implementation detours are deleted — an experiment that ran and whose result bounds the claim is evidence, and ablations and boundary-setting negative results are outcome logic, not chronology. Section ordering stays with `claim-architecture-review`. `PROSE.AI_LEXICON`'s formulaic-opener and connective-density classes were being enforced by the linter but were **invisible in the skill that executes it** — an executor reading the skill had no reason to check the opening sentence; both are now surfaced, with the positive requirement that the first sentence of the Abstract/Introduction carries a verifiable structural fact rather than a trend statement. `PROSE.OVER_DEFENSIVE` gains a fifth, **document-scope** placement error: in the Abstract and Introduction, no caveat before the contribution lands. **Behaviour change** — drafts that passed before may now flag on that fifth class; its disposal is move-only, never delete, and only after confirming Limitations already carries the boundary in full. Principles ① (organise around the lead), ② (choose the battlefield) and ④ (experiment roles) were deliberately **not** routed here: each needs whole-paper claim or evidence context, and a line editor that judges them is the documented over-execution failure mode.
- **2026-08-23 (v1.13.0)**: Narrative-agency layer, from a peer's field diagnosis of agentic paper writing. The diagnosis: an agent that fears being criticised for insufficient thoroughness pre-emptively attacks the paper on behalf of an imagined reviewer — flagging every conceivable weakness everywhere, preferring mediocrity to risk. Auditing the repo against the peer's six principles found four covered, two absent, and **two live internal contradictions**. (1) **Three new rules.** `PROSE.SELF_UNDERMINING` — the lexical layer of "do not hand the reviewer a knife", which `PROSE.OVER_DEFENSIVE` explicitly disclaims (it owns placement, not wording); 15 high-precision patterns with term-of-art lookaheads, so `falls short of the information-theoretic limit` and `lags behind by two time steps` pass while `Unfortunately, our method does not outperform…` fails, and a **neutral quantified statement of an unfavourable result passes** — that split is the rule's whole point. `EXP.EXPERIMENT_ROLE` — every experiment carries one of four argumentative roles (establish the method / explain where the advantage comes from / demonstrate value in the target scenario / rule out the most plausible competing explanation), with a redesign → demote → delete ladder; all seven existing `EXP.*` rules were about format and integrity, none about an experiment's purpose. `PAPER.OUTCOME_LOGIC` — write the logic that finally holds rather than the order the work happened, carrying the **authorisation** no other rule grants: when the evidence does not support the original narrative, redefining the problem and reordering the contributions is the correct move, not a concession. (2) **Every one of the three ships an integrity boundary**, because each could otherwise read as licence to hide evidence: wording is governed, disclosure is never reduced; an experiment is dropped for serving no claim, never for an unfavourable number; reordering changes the sequence and framing, never the reported set. Preregistered and reviewer-requested experiments are exempt from every ladder. (3) **`policy/style-guide.md` was contradicting the rules it is declared co-equal with** — its canonical paragraph exemplar opened `With the rapid development of X, Y has attracted significant attention` and closed on `significantly improves`, both banned by `PROSE.AI_LEXICON` and `PROSE.INTENSIFIERS_ELIMINATION`. An agent following the mandatory-read style guide was flagged by the same repo's linter one step later. Exemplars fixed (a second violating template in §3.1 was found in the sweep), the author's five-part shape and voice left untouched, and **`validate.sh` Section 11b** now lints the style guide's own prose blocks so a co-equal authority cannot drift from the rules again. (4) **`validate.sh` Section 4c** validates `phases:` against the Phase 词汇表, which nothing had ever checked — `writing-intro` had sat in a card for 30+ commits as an undeclared value; Introduction is a real phase, so it joined the vocabulary and the Step→Phase map. Also added: advantage-typed claim spine in `claim-architecture-review` (with an explicit guard that "backs no advantage" is never a deletion licence) and a **选对战场** positioning procedure in `ml-paper-writing` — deliberately not a rule, since no sentence-level check distinguishes a well-chosen frame from a self-serving one, and its hard limit is that the comparison the field expects still gets reported.
- **2026-08-17 (v1.12.0)**: Register coverage for **author-original** prose, from a field report on an NDSS submission (20 pages) that had already passed a `writing-anti-ai` run. A section-by-section cleanup found ~30 register defects; feeding 26 representative strings to `PROSE.INFORMAL_VOCABULARY`'s nine lint patterns matched **zero**. The diagnosis was a scope/tool mismatch rather than an execution failure: the card *claimed* responsibility for "the wordlist layer, plus text this pass did not change", but shipped only nine single-word regexes, while 29 of the 30 defects were multi-word constructions. (1) **`PROSE.INFORMAL_VOCABULARY` rebuilt as a five-class taxonomy** — idiomatic adverbials (regex-judged), phrasal verbs displacing a Latinate verb, judgment adjectives, concrete-noun metaphors in predicate position, and internal-work-trace verbs; the last four are LLM-judged with per-class criteria and `params` allowlists, because over-execution is the documented failure mode (`from scratch`, `rules out`, `falls back` and `cheap unlearning` are all terms of art that a longer blacklist would have destroyed). Classes 2–4 are deliberately **not** autofixable. (2) **New `PROSE.IDIOM_COLLISION`** — a technical phrase that is also a common English idiom gets read as the idiom first (`a fair bit` meaning an unbiased bit, `on the order of`, `significant`). Neither a register nor an accuracy defect, so no existing rule covered it. (3) **New `policy/references/tex-prose-extraction.md`**, referenced from the Check section of all 59 `llm_*` rule cards: four measured ways a hand-rolled `.tex` scanner produces a **false "clean" verdict** — `split('%')` truncating at `$95\%$`, odd-`$` math stripping swallowing whole paragraphs (one run lost 42% of a section), line-wise scanning missing phrases split by hard wrapping, and inconsistent case policy between passes. (4) `PROSE.OVER_DEFENSIVE` gains a multi-home caveat ordering step, and `PROSE.REGISTER_PRESERVATION` now names anti-AI passes as a trigger — one `at all` was introduced by an agent making a sentence "easier to read", since **lowering the reading barrier and lowering the register get systematically conflated**. `REGISTER_PRESERVATION`'s diff-only scope is deliberately left untouched.
- **2026-08-15 (v1.11.0)**: Two evidence-driven policy hardenings, both from real submission sweeps, plus a CI fix. (1) **`PROSE.NO_INTERNAL_PROVENANCE` hardened** — the rule already existed and still let eleven development artifacts reach a compiled ACM ASIA CCS submission (one was a local filesystem path printed in the evaluation body). The failure was three holes, not a missing rule: it was absent from `guardrail-checklist.md` (the compact list agents actually read while drafting), `lint.sh` had zero coverage of it, and `severity: warn` contradicted its own Rationale calling a leak a hard defect. Now: registered in the checklist, a **builtin P1–P5 detector** in `lint.sh` that strips `\includegraphics`/`\input`/reference keys/the artifact `\url`/EXP-mandated disclosures *before* matching (an exclusion list is the difference between a guardrail people keep and one they switch off), `severity: error` with `params.drafting_severity` documenting the profile downgrade, taxonomy 3 → 7 classes (data-source paths, schema identifiers, internal fixture names, revision narrative), a Requirement that leads with *where provenance belongs* rather than the ban — the root cause is a norm collision, since same-turn compute-then-write makes citing the path feel like compliance with a virtue the project rewards — and `policy/scripts/extract-undefined-identifiers.sh` for the two-stage undefined-identifier sub-check. (2) **New `PROSE.REGISTER_PRESERVATION`** — register is a property of the *edit*, not of the word, so the enforcement point is a diff. One compression pass (968 → 728 words) introduced nine register violations that `PROSE.INFORMAL_VOCABULARY`'s five patterns caught **zero** of, and two of the nine were not informal at all, only less precise. A wider regex over expert-accepted text measured **precision 0.00 / recall 0.00** across 35 hits, so the rule ships `check_kind: llm_style` with **no** `lint_patterns` by design. It leads with the repair rule (reuse the wording the manuscript already uses elsewhere — seven of nine fixes were recoverable that way), carries a five-class drift taxonomy and ten measured exclusions, and adds a workflow gate: **no word count or reduction percentage until the register check passes**. `writing-anti-ai` v1.2.0's "Do NOT Over-Correct" section now leads with *substitution* drift; its six previous items were all deletion-direction and none of the nine violations deleted anything. (3) **CI fix**: `grep -c` exits 1 on a zero count under `bash -eo pipefail`, so the policy workflow failed precisely because `validate.sh` had become clean.
- **2026-08-15 (v1.10.0)**: Anti-AI writing overhaul + policy conflict audit + first evidence-backed eval run, plus a vendored ZK security skill. (1) **4 new policy rules**: `PROSE.AI_LEXICON` (tier-1 zero-tolerance AI vocabulary + tier-2 density threshold + formulaic openers + sentence-initial connective budget, with term-of-art exemptions so `loss landscape` / `robust` / `optimize` / `trajectory` survive), `PROSE.FRACTAL_SUMMARY` (no per-level preview/recap scaffolding), `PROSE.INVENTED_CONCEPT_LABEL` (coined labels need a citation or an explicit naming contribution), `PROSE.RESTATEMENT_DILUTION` (one proposition, one placement per section). (2) **`PROSE.HEDGING_DISCIPLINE` made bidirectional** — over-claiming verbs and unanchored comparatives now fail the same rule as over-hedging, with a calibration red line so fixing one direction cannot manufacture the other. (3) **Policy conflict/redundancy audit**: fixed a self-contradictory ban (`smaller`), an autofix repair loop (`a lot of`→`many` triggering `VAGUE_QUANTIFIERS`), bare-word false positives that hit mathematical existential quantifiers (`for some ε > 0`), a semantically wrong `SENTENCE_LENGTH` lint pattern (it counted sentences per file, not words per sentence), and empty `conflicts_with` across the four-card short-sentence cluster that let agents oscillate; added a lexicon ownership table to `policy/README.md`. (4) **`validate.sh` Section 5c (fix-emission safety)** makes "no autofix output may trigger another rule" a machine-checked invariant. (5) **`writing-anti-ai` v1.1.0**: reader-tells vs statistical-detectors separation (no detector-evasion promises), interleave protocol, register-split voice guidance, a Do-NOT-Over-Correct guard, `references/evidence.md` (every claim carries source/date/instrument/sample size/expiry), and `evals/evals.json`. First eval run: 39/39 vs 37/39 for the pre-overhaul snapshot, with both deltas landing exactly on the new capabilities. Borrowed with review from [zksecurity/zk-skills](https://github.com/zksecurity/zk-skills) (`circom-auditor`, vendored submodule) and [AIScientists-Dev/academic-humanizer](https://github.com/AIScientists-Dev/academic-humanizer) (claim–evidence calibration, novelty padding, over-correction guardrails; its proposal mode deliberately not imported since `grant-application-writing` covers it). Also fixed `skill-forced-eval.js`, which silently skipped every symlinked skill directory — `fireworks-tech-graph` and `scientific-figure-making` had never appeared in the forced-evaluation list.
- **2026-07-09 (v1.7.1)**: Post-release review fixes for v1.7.0 — venue-split figure sizing discipline (`results-analysis`: ML-conference venues keep the default `FigureStyle` pipeline, hard-spec venues now explicitly override its default font size with the physical-size spec instead of silently conflicting with it), cleaned up stale `FIG.FONT_GE_24PT` references (deprecated in favor of `scientific-figure-making`'s `FigureStyle`) across `figure-visual-qa.md`, `journal-figure-specs.md`, and the policy registry, `TABLE.DIMENSION_BUDGET` example de-domained (generic accuracy/latency/memory dimensions instead of a maintainer's in-progress paper's vocabulary) plus its dimension-vs-column-count off-by-one reconciled, and `review-response` / `paper-self-review` given reverse pointers into the `knows-literature` bridge so it fires on direct invocation too.
- **2026-07-09 (v1.7.0)**: Typesetting-constraint pack + skill-routing upgrade + Knows bridge refresh (borrowed with review from [DELONG-L/Academic-Paper-Skills](https://github.com/DELONG-L/Academic-Paper-Skills), MIT). (1) **3 new policy rules**: `TABLE.RESIZEBOX_COLUMN_FIT` (tables default to `\resizebox` column fit, explicit natural-fit exemptions), `TABLE.DIMENSION_BUDGET` (comparison tables: 3–4 high-signal dimensions, single-column first, prune before `table*`), `PROSE.NO_INTERNAL_PROVENANCE` (no script names / paths / DPI notes / placeholder markers / draft meta-text in paper body or captions — a high-frequency agentic-writing defect); `PROSE.FORMATTING_RESTRAINT` extended with `\texttt{}` discipline. (2) **Figure visual-QA closed loop** (`results-analysis/references/figure-visual-qa.md`): render 150-dpi PNG → actually Read the image → 8-item perceptual checklist (clipping, occlusion, panel alignment, grayscale) → fix at source → re-render, ≤3 rounds, before any vector export; plus `journal-figure-specs.md` venue hard-spec quick table. (3) **Skill descriptions rewritten** for 5 high-frequency writing skills with dense task-noun coverage + explicit negative boundaries — raises auto-invocation hit rate in both Claude Code and Codex runtimes. (4) **knows-literature bridge v0.2.0**: mapping extended to `rebuttal-builder` / `review-sidecar` / `cite-key`, and reverse pointers wired into `ml-paper-writing` + `citation-verification` (previously one-directional). Deliberately NOT borrowed: header-arrow ban (conflicts with `TABLE.DIRECTION_INDICATORS`), local-`.bib`-only citation policy (weaker than `CITE.VERIFY_VIA_API` + Knows).
- **2026-07-08 (v1.6.2)**: Added policy rule `EXP.MULTIRUN_AGGREGATE_CONSISTENCY` (core, error) — multi-run result tables/figures must source from a machine-generated aggregate artifact carrying per-run validity, a cross-run must-match consistency verdict, and provenance (run ids + upstream build ids); `INCONSISTENT` verdicts require explicit caption/body disclosure, and hand-transcribing numbers is forbidden. Complements `CITE.CLAIM_SUPPORT_REQUIRED` (transcription fidelity) with **source trustworthiness** (runs are real, mutually comparable, traceable). Producer-agnostic: reference implementation is the `exp aggregate` CLI (bridges an experiment-ops workflow to paper writing via a plain-file contract, runs local or on a cluster), but any script emitting the three field groups qualifies. Wired into `ml-paper-writing`, `results-analysis`, and `publication-tables`.
- **2026-07-02 (v1.6.1)**: `paper-figure-generator` now forces **non-italic sans-serif** fonts in generated SVGs — `normalize_svg_fonts()` injects an `!important` `<style>` at both write points (`template.svg` + `final.svg`), fixing the default italic Times New Roman output.
- **2026-06-25 (v1.6.0)**: Added the `architecture_review` orchestrator stage + `claim-architecture-review` skill — a post-draft **structural edit** (paragraph necessity / placement, cross-section redundancy, claim spine + story closure) that runs before self-review and anti-AI polish. File-backed multi-pass design scales to long papers; propose-only (the `rewrite` stage applies approved moves). Pipeline is now 12 stages.
- **2026-03-02 (v1.4.1)**: Added Workflow Orchestrator — stateful, resumable research run coordination layer. 10-stage pipeline with persistent run state (`.claude/orchestrator/`), artifact fingerprinting (SHA256), auto-stale detection at session start, rollback with downstream cascade, stage gates (human approval + policy lint). Zero new commands — activates transparently via existing skills/agents/hooks.
- **2026-02-21**: Added first SoK policy pack: 4 semantic `SOK.*` rule cards, `security-sok-sp` profile, and entry-skill marker wiring. SoK remains profile-activated scope in v1 (no schema migration yet).
- **2026-02-19 (v1.3.0)**: Introduced the paper policy engine (`policy/`): rule-card based design in `policy/rules/` (single source of truth), layered scope (`core/domain/venue`), profile overlays in `policy/profiles/`, and executable validation/lint workflows via `policy/validate.sh` and `policy/lint.sh`. Synced Figure workflow policy (Figure 1 required; non-experimental figures default to AutoFigure-Edit).
- **2026-02-16 (v1.2.1)**: Added a global figure rule: no in-image titles for any generated visuals (AutoFigure-Edit conceptual diagrams, legacy image APIs, or Python experimental plots). Use captions in paper text/LaTeX instead.
- **2026-02-16**: Enforced `paper-figure-generator` execution priority: default `AutoFigure-Edit + OpenRouter` first, fallback to legacy Gemini/OpenAI flow only after failure; added troubleshooting note for outdated plugin cache prompts (`GOOGLE_API_KEY` / `OPENAI_API_KEY`).
- **2026-02-15**: Migrated `paper-figure-generator` to AutoFigure-Edit — generates editable SVG vector figures from method text descriptions; replaces Gemini/OpenAI raster generation; supports style transfer via reference images; uses OpenRouter + Roboflow (free SAM3 API)
- **2026-02-13**: Added `paper-figure-generator` skill; packaged project as Claude Code plugin (`.claude-plugin/plugin.json`); added `.env.example`; deep workflow integration across ml-paper-writing, results-analysis, post-acceptance, and using-claude-scholar; 34 skills total
- **2026-02-11**: Major update — added 10 new skills (research-ideation, results-analysis, citation-verification, review-response, paper-self-review, post-acceptance, daily-coding, frontend-design, ui-ux-pro-max, web-design-reviewer), 7 new agents, 8 research workflow commands, 2 new rules (security, experiment-reproducibility); restructured CLAUDE.md; 89 files changed
- **2026-01-26**: Rewrote all Hooks to cross-platform Node.js; completely rewrote README; expanded ML paper writing knowledge base; merged PR #1 (cross-platform support)

## Introduction

Claude Scholar is a personal configuration system for Claude Code CLI, providing rich skills, commands, agents, and hooks optimized for:
- **Academic Research** - Complete research lifecycle: idea generation → experimentation → results analysis → paper writing → review response → conference preparation
- **Software Development** - Git workflows, code review, test-driven development, ML project architecture
- **Plugin Development** - Skill, Command, Agent, Hook development guides with quality assessment
- **Project Management** - Planning documents, code standards, automated workflows with cross-platform hooks

## Quick Navigation

| Topic | Description |
|-------|-------------|
| 🚀 [Quick Start](#quick-start) | Get up and running in minutes |
| 📚 [Core Workflows](#core-workflows) | Paper writing, code organization, skill evolution |
| 🛠️ [What's Included](#whats-included) | Skills, commands, agents overview |
| 📖 [Installation Guide](#installation-options) | Full, minimal, or selective setup |
| 🔧 [Project Rules](#project-rules) | Coding rules + paper policy engine |

## Core Workflows

### Primary Workflows

Complete academic research lifecycle - 7 stages from idea to publication.

#### 1. Research Ideation

Systematic research startup with idea generation and literature review:

**Tools**: `research-ideation` skill + `literature-reviewer` agent

**Process**:
- **5W1H Brainstorming**: What, Why, Who, When, Where, How → structured thinking framework
- **Literature Review**: arXiv + Semantic Scholar integration → automated paper search and classification
- **Gap Analysis**: 5 types (Literature, Methodological, Application, Interdisciplinary, Temporal) → identify research opportunities
- **Research Question**: SMART principles → formulate specific, measurable questions

**Command**: `/research-init "topic"` → launches complete research startup workflow

#### 2. ML Project Development

Maintainable ML project structure for experiment code:

**Tools**: `architecture-design` skill + `code-reviewer` agent + `git-workflow` skill

**Process**:
- **Structure**: Factory & Registry patterns → config-driven models (only `cfg` parameter) → enforced by `rules/coding-style.md`
- **Code Style**: 200-400 line files → type hints required → `@dataclass(frozen=True)` for configs → max 3-level nesting
- **Debug** (`bug-detective`): Error pattern matching for Python/Bash/JS → stack trace analysis → anti-pattern identification
- **Git**: Conventional Commits (`feat/scope: message`) → branch strategy (master/develop/feature) → merge with `--no-ff`

**Commands**: `/plan`, `/commit`, `/code-review`, `/tdd`

#### 3. Experiment Analysis

Statistical analysis and visualization of experimental results:

**Tools**: `results-analysis` skill + `data-analyst` agent

**Process**:
- **Data Processing**: Automated cleaning and preprocessing of experiment logs
- **Statistical Testing**: t-test, ANOVA, Wilcoxon signed-rank → validate significance
- **Visualization**: matplotlib/seaborn integration → publication-ready figures (line plots, bar charts, heatmaps)
- **Ablation Studies**: Systematic component analysis → understand contribution of each part

**Command**: `/analyze-results <experiment_dir>` → generates analysis report with figures and statistics

#### 4. Paper Writing

Systematic paper writing from template to final draft:

**Tools**: `ml-paper-writing` skill + `paper-miner` agent + `latex-conference-template-organizer` skill

**Process**:
- **Template Preparation**: Download conference .zip → extract main files → remove sample content → clean Overleaf-ready structure
- **Citation Verification** (`citation-verification`): Multi-layer validation (Format → API → Information → Content) → prevents hallucinations
- **Systematic Writing**: Narrative framing → 5-sentence abstract formula → section-by-section drafting with feedback cycles
- **Anti-AI Processing** (`writing-anti-ai`): Remove inflated symbolism, promotional language, vague attributions → add human voice and rhythm → bilingual support (EN/CN)

**Venues**: NeurIPS, ICML, ICLR, ACL, AAAI, COLM, Nature, Science, Cell, PNAS

#### 5. Paper Self-Review

Quality assurance before submission:

**Tools**: `paper-self-review` skill

**Process**:
- **Structure Check**: Logical flow, section balance, narrative coherence
- **Logic Validation**: Argument soundness, claim-evidence alignment, assumption clarity
- **Citation Audit**: Reference accuracy, proper attribution, citation completeness
- **Figure Quality**: Visual clarity, caption completeness, color accessibility
- **Writing Polish**: Grammar, clarity, conciseness, academic tone
- **Compliance**: Page limits, formatting requirements, ethical disclosures

**Multi-item checklist** → systematic quality assessment (including figure/title and LaTeX math conformance)

#### 6. Submission & Rebuttal

Paper submission and review response:

**Tools**: `review-response` skill + `rebuttal-writer` agent

**Submission Process**:
- **Pre-submission**: Conference-specific checklists (NeurIPS 16-item, ICML Broader Impact, ICLR LLM disclosure)
- **Format Check**: Page limits, anonymization, supplementary materials
- **Final Review**: Proofread, check references, verify figures

**Rebuttal Process**:
- **Review Analysis**: Parse and classify comments (Major/Minor/Typo/Misunderstanding)
- **Response Strategy**: Accept/Defend/Clarify/Experiment → tailored approach per comment type
- **Rebuttal Writing**: Structured response with evidence and reasoning
- **Tone Management**: Professional, respectful, evidence-based language

**Command**: `/rebuttal <review_file>` → generates complete rebuttal document with experiment plan

#### 7. Post-Acceptance Processing

Conference preparation and research promotion:

**Tools**: `post-acceptance` skill

**Process**:
- **Presentation**: Slide creation guidance (15/20/30 min formats) → visual design principles → storytelling structure
- **Poster**: Academic poster templates (A0/A1 sizes) → layout optimization → visual hierarchy
- **Promotion**: Social media content (Twitter/X, LinkedIn) → blog posts → press releases → research summaries

**Commands**: `/presentation`, `/poster`, `/promote` → automated content generation

**Coverage**: 90% of academic research lifecycle (from idea to publication)

### Workflow Orchestrator

Claude Scholar includes a stateful **Workflow Orchestrator** that tracks progress across the research lifecycle as a single, resumable run. No new commands are needed -- the orchestrator activates transparently when relevant skills and agents are invoked.

**Key features:**
- **Single mode, resumable runs**: State persists in `.claude/orchestrator/` across sessions. Resume from where you left off.
- **12-stage pipeline**: intake -> literature -> proposal -> development -> experiments -> analysis -> writeup -> architecture_review -> self_review -> rewrite -> rebuttal -> post_acceptance
- **Stage gates**: Human approval and policy lint checks at stage boundaries prevent premature progression.
- **Artifact fingerprinting**: SHA256 hashes detect file changes and mark affected stages as `stale`.
- **Contract-backed fingerprinting**: Stage file artifacts are fingerprinted deterministically, and `writeup` expands local LaTeX dependencies from `main_tex`.
- **Experiments boundary**: The `experiments` stage enters `blocked` until the user provides a `data_path` with actual results. Rollback is always possible ("roll back to stage X").

**How it works:**
- Session start hook displays active run ID, current stage, and next action.
- Skills and agents automatically read/write run state per the [Run Card contract](orchestrator/run-card.md).
- Stage registry defined in `orchestrator/stages.json`; runtime library at `scripts/lib/orchestrator.js`.

See [docs/orchestrator.md](docs/orchestrator.md) for full documentation.

### Supporting Workflows

These workflows run in the background to enhance the primary workflows.

#### Automated Enforcement Workflow

Cross-platform hooks (Node.js) automate workflow enforcement:

```
Session Start → Skill Evaluation → Session End → Session Stop
```

- **skill-forced-eval** (`skill-forced-eval.js`): Before EVERY user prompt → dynamically scans all available skills (local + plugins) → forces evaluation of each skill → requires activation before implementation → ensures no relevant skill is missed
- **session-start** (`session-start.js`): Session begins → displays Git status, pending todos, available commands, package manager → shows project context at a glance
- **session-summary** (`session-summary.js`): Session ends → generates comprehensive work log → summarizes all changes made → includes orchestrator status and recent run events
- **stop-summary** (`stop-summary.js`): Session stops → quick status check → detects temporary files → shows actionable cleanup suggestions

**Cross-platform**: All hooks use Node.js (not shell scripts) ensuring Windows/macOS/Linux compatibility.

#### Knowledge Extraction Workflow

Two specialized mining agents continuously extract knowledge to improve skills:

- **paper-miner** (agent): Analyze research papers (PDF/DOCX/arXiv links) → extracts writing patterns, structure insights, venue requirements, rebuttal strategies → updates `ml-paper-writing/references/knowledge/` with categorized entries (structure.md, writing-techniques.md, submission-guides.md, review-response.md)
- **kaggle-miner** (agent): Study winning Kaggle competition solutions → extract competition briefs, front-runner detailed technical analysis, code templates, best practices → update the `kaggle-learner` skill's knowledge base (`references/knowledge/[domain]/` directories, categorized by NLP/CV/Time Series/Tabular/Multimodal)

**Knowledge feedback loop**: Each paper or solution analyzed enriches the knowledge base, creating a self-improving system that evolves with your research.

#### Skill Evolution System

3-step continuous improvement cycle for maintaining and improving skills:

```
skill-development → skill-quality-reviewer → skill-improver
```

1. **Develop** (`skill-development`): Create skills with proper YAML frontmatter → clear descriptions with trigger phrases → progressive disclosure (lean SKILL.md, details in `references/`)
2. **Review** (`skill-quality-reviewer`): 4-dimension quality assessment → Description Quality (25%), Content Organization (30%), Writing Style (20%), Structural Integrity (25%) → generates improvement plan with prioritized fixes
3. **Improve** (`skill-improver`): Merges suggested changes → updates documentation → iterates on feedback → reads improvement plans and applies changes automatically

## File Structure

```
claude-scholar/
├── AGENTS.md            # Codex behavioral reference (kept in repo; no longer copied)
├── .codex/              # Codex-specific files
│   └── INSTALL.md               # Codex installation guide
│
├── hooks/               # Cross-platform JavaScript hooks (Claude Code only)
│   ├── session-start.js         # Session begin - shows Git status, todos, commands
│   ├── skill-forced-eval.js     # Force skill evaluation before each prompt
│   ├── session-summary.js       # Session end - generates work log with recommendations
│   ├── stop-summary.js          # Session stop - quick status check, temp file detection
│   └── security-guard.js        # Security validation for file operations
│
├── skills/              # 35 specialized skills (domain knowledge + workflows)
│   ├── ml-paper-writing/        # Full paper writing: NeurIPS, ICML, ICLR, ACL, AAAI, COLM
│   │   └── references/
│   │       └── knowledge/        # Extracted patterns from successful papers
│   │       ├── structure.md           # Paper organization patterns
│   │       ├── writing-techniques.md  # Sentence templates, transitions
│   │       ├── submission-guides.md   # Venue requirements (page limits, etc.)
│   │       └── review-response.md     # Rebuttal strategies
│   │
│   ├── research-ideation/        # Research startup: 5W1H, literature review, gap analysis
│   │   └── references/
│   │       ├── 5w1h-framework.md           # Systematic thinking tool
│   │       ├── gap-analysis-guide.md       # 5 types of research gaps
│   │       ├── literature-search-strategies.md
│   │       ├── research-question-formulation.md
│   │       ├── method-selection-guide.md
│   │       └── research-planning.md
│   │
│   ├── results-analysis/         # Experiment analysis: statistics, visualization, ablation
│   │   └── references/
│   │       ├── statistical-methods.md      # t-test, ANOVA, Wilcoxon
│   │       ├── visualization-best-practices.md  # matplotlib/seaborn
│   │       ├── results-writing-guide.md    # Writing results sections
│   │       └── common-pitfalls.md          # Common analysis mistakes
│   │
│   ├── review-response/          # Systematic rebuttal writing
│   │   └── references/
│   │       ├── review-classification.md    # Major/Minor/Typo/Misunderstanding
│   │       ├── response-strategies.md      # Accept/Defend/Clarify/Experiment
│   │       ├── rebuttal-templates.md       # Structured response templates
│   │       └── tone-guidelines.md          # Professional language
│   │
│   ├── paper-self-review/        # multi-item quality checklist
│   ├── post-acceptance/          # Conference preparation
│   │   └── references/
│   │       ├── presentation-templates/     # Slide creation (15/20/30 min)
│   │       ├── poster-templates/           # Academic poster design
│   │       ├── promotion-examples/         # Social media content
│   │       └── design-guidelines.md        # Visual design principles
│   │
│   ├── citation-verification/    # Multi-layer citation validation
│   ├── writing-anti-ai/         # Remove AI patterns: symbolism, promotional language
│   │   └── references/
│   │       ├── patterns-english.md    # English AI patterns to remove
│   │       └── patterns-chinese.md     # Chinese AI patterns to remove
│   │
│   ├── architecture-design/     # ML project patterns: Factory, Registry, Config-driven
│   ├── git-workflow/            # Git discipline: Conventional Commits, branching
│   ├── bug-detective/           # Debugging: Python, Bash, JS/TS error patterns
│   ├── code-review-excellence/  # Code review: security, performance, maintainability
│   ├── skill-development/       # Skill creation: YAML, progressive disclosure
│   ├── skill-quality-reviewer/  # Skill assessment: 4-dimension scoring
│   ├── skill-improver/          # Skill evolution: merge improvements
│   ├── kaggle-learner/          # Learn from Kaggle winning solutions
│   ├── doc-coauthoring/         # Document collaboration workflow
│   ├── latex-conference-template-organizer  # Template cleanup for Overleaf
│   └── ... (10+ more skills)
│
├── commands/            # 50+ slash commands (quick workflow execution)
│   ├── research-init.md         # Launch research startup workflow
│   ├── analyze-results.md       # Analyze experiment results
│   ├── rebuttal.md              # Generate systematic rebuttal document
│   ├── presentation.md          # Create conference presentation outline
│   ├── poster.md                # Generate academic poster design plan
│   ├── promote.md               # Generate promotion content
│   ├── plan.md                  # Implementation planning with agent delegation
│   ├── commit.md                # Conventional Commits: feat/fix/docs/refactor
│   ├── code-review.md           # Quality and security review workflow
│   ├── tdd.md                   # Test-driven development: Red-Green-Refactor
│   ├── build-fix.md             # Fix build errors automatically
│   ├── verify.md                # Run verification loops
│   ├── checkpoint.md            # Save verification state
│   ├── refactor-clean.md        # Remove dead code
│   ├── learn.md                 # Extract patterns from code
│   └── sc/                      # SuperClaude command suite (20+ commands)
│       ├── sc-agent.md           # Agent management
│       ├── sc-estimate.md       # Development time estimation
│       ├── sc-improve.md         # Code improvement
│       └── ...
│
├── agents/              # 14 specialized agents (focused task delegation)
│   ├── literature-reviewer.md   # Literature search and trend analysis
│   ├── data-analyst.md          # Automated data analysis and visualization
│   ├── rebuttal-writer.md       # Systematic rebuttal writing
│   ├── paper-miner.md           # Extract paper knowledge: structure, techniques
│   ├── architect.md             # System design: architecture decisions
│   ├── code-reviewer.md         # Review code: quality, security, best practices
│   ├── tdd-guide.md             # Guide TDD: test-first development
│   ├── kaggle-miner.md          # Extract engineering practices from Kaggle
│   ├── build-error-resolver.md  # Fix build errors: analyze and resolve
│   ├── refactor-cleaner.md      # Remove dead code: detect and cleanup
│   ├── bug-analyzer.md          # Deep code execution flow analysis and root cause investigation
│   ├── dev-planner.md           # Implementation planning and task breakdown
│   ├── ui-sketcher.md           # UI blueprint design and interaction specs
│   └── story-generator.md       # User story and requirement generation
│
├── rules/               # Global guidelines (always-follow constraints)
│   ├── coding-style.md          # ML project standards: file size, immutability, types
│   ├── agents.md                # Agent orchestration: when to delegate, parallel execution
│   ├── security.md              # Secrets management, sensitive file protection
│   └── experiment-reproducibility.md  # Random seeds, config recording, checkpoints
│
├── orchestrator/        # Workflow Orchestrator (stage registry + run card)
│   ├── stages.json              # Stage definitions (10 stages, artifacts, gates)
│   └── run-card.md              # Skills/agents integration contract
│
├── policy/              # Paper policy engine (rule cards + validation + lint)
│   ├── rules/                    # Canonical paper-writing rule cards (single source of truth)
│   ├── profiles/                 # Domain/venue overlays (severity/params tuning)
│   ├── validate.sh               # Rule-card integrity validation
│   ├── lint.sh                   # Machine-enforceable lint checks
│   └── README.md                 # Policy engine design and conventions
│
├── scripts/
│   ├── install-codex.sh         # Codex installer (macOS/Linux, symlink-based)
│   ├── install-codex-windows.ps1 # Codex installer (Windows, junction-based)
│   └── lib/                     # Shared script utilities
│
├── CLAUDE.md            # Global configuration: project overview, preferences, rules
│
└── README.md            # This file - overview, installation, features
```

## Feature Highlights

### Skills (29 total)

**Writing & Academic:**
- `ml-paper-writing` - Full paper writing guidance for top conferences/journals
- `writing-anti-ai` - Remove AI writing patterns (bilingual support)
- `doc-coauthoring` - Structured document collaboration workflow
- `latex-conference-template-organizer` - LaTeX template management
- `daily-paper-generator` - Automated daily paper generation for research tracking

**Research Workflow:**
- `research-ideation` - Research startup: 5W1H brainstorming, literature review, gap analysis
- `results-analysis` - Experiment analysis: statistical testing, visualization, ablation studies
- `review-response` - Systematic rebuttal writing with tone management
- `paper-self-review` - multi-item quality checklist for paper self-assessment (figures + LaTeX math conformance)
- `post-acceptance` - Conference preparation: presentations, posters, promotion
- `citation-verification` - Multi-layer citation validation to prevent hallucinations
- `paper-figure-generator` - Generate editable SVG academic figures (system overviews, pipelines, architectures) via AutoFigure-Edit

**Development:**
- `daily-coding` - Daily coding checklist (minimal, auto-triggered)
- `git-workflow` - Git best practices (Conventional Commits, branching)
- `code-review-excellence` - Code review guidelines
- `bug-detective` - Debugging for Python, Bash, JS/TS
- `architecture-design` - ML project design patterns
- `verification-loop` - Testing and validation

**Security Audit:**
- `circom-auditor` - Circom / ZK circuit audit: soundness, completeness, privacy, constraint bugs (17-agent delegated workflow, vendored from [zk-skills](https://github.com/zksecurity/zk-skills))

**Plugin Development:**
- `skill-development` - Skill creation guide
- `skill-improver` - Skill improvement tools
- `skill-quality-reviewer` - Quality assessment
- `command-development` - Slash command creation
- `agent-identifier` - Agent configuration
- `hook-development` - Hook development guide
- `mcp-integration` - MCP server integration

**Utilities:**
- `uv-package-manager` - Modern Python package management
- `planning-with-files` - Markdown-based planning
- `kaggle-learner` - Learn from Kaggle solutions

### Commands (50+)

**Research Commands:**
| Command | Purpose |
|---------|---------|
| `/research-init` | Launch research startup workflow (5W1H, literature review, gap analysis) |
| `/analyze-results` | Analyze experiment results (statistics, visualization, ablation) |
| `/rebuttal` | Generate systematic rebuttal document from review comments |
| `/presentation` | Create conference presentation outline |
| `/poster` | Generate academic poster design plan |
| `/promote` | Generate promotion content (Twitter, LinkedIn, blog) |

**Development Commands:**
| Command | Purpose |
|---------|---------|
| `/plan` | Create implementation plans |
| `/commit` | Commit with Conventional Commits |
| `/code-review` | Perform code review |
| `/tdd` | Test-driven development workflow |
| `/build-fix` | Fix build errors |
| `/verify` | Verify changes |
| `/checkpoint` | Create checkpoints |
| `/refactor-clean` | Refactor and cleanup |
| `/learn` | Extract reusable patterns |
| `/sc` | SuperClaude command suite (20+ commands) |

### Agents (14 specialized)

**Research Agents:**
- **literature-reviewer** - Literature search, classification, and trend analysis
- **data-analyst** - Automated data analysis and visualization
- **rebuttal-writer** - Systematic rebuttal writing with tone optimization
- **paper-miner** - Extract paper writing knowledge from successful publications

**Development Agents:**
- **architect** - System architecture design
- **build-error-resolver** - Fix build errors
- **code-reviewer** - Review code quality
- **refactor-cleaner** - Remove dead code
- **tdd-guide** - Guide TDD workflow
- **kaggle-miner** - Extract Kaggle engineering practices
- **bug-analyzer** - Deep code execution flow analysis and root cause investigation
- **dev-planner** - Implementation planning and task breakdown

**Design & Content Agents:**
- **ui-sketcher** - UI blueprint design and interaction specs
- **story-generator** - User story and requirement generation

## Quick Start

### Multi-Runtime Support

Claude Scholar supports two runtimes:

| | Claude Code | Codex |
|---|------------|-------|
| **Skills** | 35 (full) | 27 universal + 6 reference |
| **Hooks** | 5 automated | N/A (using-claude-scholar skill replaces) |
| **Commands** | 50+ slash commands | N/A (use skills directly) |
| **Agents** | 14 specialized | 14 (via `spawn_agent`) |
| **Install** | Clone / Plugin | Symlink only (native skill discovery) |

### Installation Options

#### Claude Code Installation

Choose the installation method that fits your needs:

##### Option 1: Plugin Installation (Recommended)

Install via Claude Code plugin manager:

```bash
# Step 1: Add marketplace
claude plugin marketplace add OniReimu/claude-scholar

# Step 2: Install plugin
claude plugin install claude-scholar@claude-scholar
```

**Benefits**: Automatic component discovery, version tracking, easy updates via `claude plugin update`.

**Includes**: All 35 skills, 50+ commands, 14 agents, 5 hooks, and project rules.

##### Option 2: Full Installation (Git Clone)

Complete setup by cloning directly to `~/.claude`:

```bash
# Clone the repository (--recursive pulls vendored skills:
# scientific-figure-making, fireworks-tech-graph, circom-auditor)
git clone --recursive https://github.com/OniReimu/claude-scholar.git ~/.claude

# Already cloned without submodules?
git -C ~/.claude submodule update --init --recursive

# Restart Claude Code CLI
```

**Includes**: All 35 skills, 50+ commands, 14 agents, 5 hooks, and project rules.

##### Option 3: Minimal Installation

Core hooks and essential skills only (faster load, less complexity):

```bash
# Clone repository
git clone https://github.com/OniReimu/claude-scholar.git /tmp/claude-scholar

# Copy only hooks and core skills
mkdir -p ~/.claude/hooks ~/.claude/skills
cp /tmp/claude-scholar/hooks/*.js ~/.claude/hooks/
cp -r /tmp/claude-scholar/skills/ml-paper-writing ~/.claude/skills/
cp -r /tmp/claude-scholar/skills/research-ideation ~/.claude/skills/
cp -r /tmp/claude-scholar/skills/results-analysis ~/.claude/skills/
cp -r /tmp/claude-scholar/skills/review-response ~/.claude/skills/
cp -r /tmp/claude-scholar/skills/writing-anti-ai ~/.claude/skills/
cp -r /tmp/claude-scholar/skills/git-workflow ~/.claude/skills/
cp -r /tmp/claude-scholar/skills/bug-detective ~/.claude/skills/

# Cleanup
rm -rf /tmp/claude-scholar
```

**Includes**: 5 hooks, 7 core skills (complete research workflow + essential development).

##### Option 4: Selective Installation

Pick and choose specific components:

```bash
# Clone repository
git clone https://github.com/OniReimu/claude-scholar.git /tmp/claude-scholar
cd /tmp/claude-scholar

# Copy what you need, for example:
# - Hooks only
cp hooks/*.js ~/.claude/hooks/

# - Specific skills
cp -r skills/latex-conference-template-organizer ~/.claude/skills/
cp -r skills/architecture-design ~/.claude/skills/

# - Specific agents
cp agents/paper-miner.md ~/.claude/agents/

# - Project rules
cp rules/coding-style.md ~/.claude/rules/
cp rules/agents.md ~/.claude/rules/
```

**Recommended for**: Advanced users who want custom configurations.

#### Codex Installation

```bash
# Clone the repository
git clone https://github.com/OniReimu/claude-scholar.git ~/claude-scholar

# Run the install script (creates symlinks, migrates legacy AGENTS.md)
chmod +x ~/claude-scholar/scripts/install-codex.sh
~/claude-scholar/scripts/install-codex.sh
```

**Windows (PowerShell):**
```powershell
git clone https://github.com/OniReimu/claude-scholar.git $HOME\claude-scholar
& "$HOME\claude-scholar\scripts\install-codex-windows.ps1"
```

**What it does:**
- Creates symlink: `~/.agents/skills/claude-scholar` → `skills/`
- Detects and migrates legacy `~/.codex/AGENTS.md`
- Updates via `git pull` — no re-install needed

See [.codex/INSTALL.md](.codex/INSTALL.md) for detailed Codex installation guide.

### Requirements

- Claude Code CLI or Codex CLI (v0.91+)
- Git
- (Optional) Node.js (for hooks)
- (Optional) uv, Python (for Python development)

### First Run

After installation, the hooks provide automated workflow assistance:

1. **Every prompt** triggers `skill-forced-eval` → ensures applicable skills are considered
2. **Session starts** with `session-start` → displays project context
3. **Sessions end** with `session-summary` → generates work log with recommendations plus orchestrator state/event summary
4. **Session stops** with `stop-summary` → provides status check

## Project Rules

### Paper Policy Engine

Defined in `policy/`:
- `policy/rules/` is the single source of truth for paper-writing constraints (figures, LaTeX, citations, experiments, submission).
- Rule-card design uses frontmatter metadata (`id`, `layer`, `artifacts`, `phases`, `check_kind`, `enforcement`) plus required sections (`Requirement`, `Rationale`, `Check`, `Examples`).
- Layering model: `core` (always on), `domain` (field-specific), `venue` (conference/journal specific); profile overlays live in `policy/profiles/*.md`.
- SoK in v1 is activated by profile (for example `policy/profiles/security-sok-sp.md`), currently with semantic `SOK.*` rules (`SOK.TAXONOMY_REQUIRED`, `SOK.METHODOLOGY_REPORTING`, `SOK.BIG_TABLE_REQUIRED`, `SOK.RESEARCH_AGENDA_REQUIRED`).
- Current limitation: `policy/lint.sh --profile` loads a single flat profile file (no inheritance/composition yet).
- Validation and enforcement workflow:
  - `bash policy/validate.sh` for structure/integration checks
  - `bash policy/lint.sh` for machine-enforceable checks
- Skills/commands reference rules via `<!-- policy:RULE_ID -->` markers.

### Coding Style

Enforced by `rules/coding-style.md`:
- **File Size**: 200-400 lines maximum
- **Immutability**: Use `@dataclass(frozen=True)` for configs
- **Type Hints**: Required for all functions
- **Patterns**: Factory & Registry for all modules
- **Config-Driven**: Models accept only `cfg` parameter

### Agent Orchestration

Defined in `rules/agents.md`:
- Available agent types and purposes
- Parallel task execution
- Multi-perspective analysis

### Security

Defined in `rules/security.md`:
- Secrets management (environment variables, `.env` files)
- Sensitive file protection (never commit tokens, keys, credentials)
- Pre-commit security checks via hooks

### Experiment Reproducibility

Defined in `rules/experiment-reproducibility.md`:
- Random seed management for reproducibility
- Configuration recording (Hydra auto-save)
- Environment recording and checkpoint management

## Contributing

This is a personal configuration, but you're welcome to:
- Fork and adapt for your own research
- Submit issues for bugs
- Suggest improvements via issues

## License

MIT License

## Acknowledgments

Built with Claude Code CLI and enhanced by the open-source community.

### References

This project is inspired by and builds upon excellent work from the community:

- **[everything-claude-code](https://github.com/anthropics/everything-claude-code)** - Comprehensive resource for Claude Code CLI
- **[AI-research-SKILLs](https://github.com/zechenzhangAGI/AI-research-SKILLs)** - Research-focused skills and configurations
- **[zk-skills](https://github.com/zksecurity/zk-skills)** (MIT, zkSecurity) - ZK circuit security skills; `circom-auditor` is vendored via the `vendor/zk-skills` submodule

These projects provided valuable insights and foundations for the research-oriented features in Claude Scholar.

---

**For data science, AI research, and academic writing.**

Repository: [https://github.com/OniReimu/claude-scholar](https://github.com/OniReimu/claude-scholar)
