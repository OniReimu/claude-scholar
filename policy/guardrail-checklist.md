# Guardrail Checklist (Compact)

> Hand-maintained digest of the `constraint_type: guardrail` rules, meant to be embedded in
> writing prompts — so entries stay at trigger + fix. Rationale, evidence, thresholds and
> allowlists live in the rule card; if an entry here needs a paragraph, it belongs there instead.
> Full cards in `policy/rules/`; violations are caught post-hoc by
> `policy/lint.sh --constraint-type guardrail`.

## Prohibited Patterns (do NOT generate these)

- **NO filler phrases**: "In order to" → "to"; "It is important to note that" → delete; "plays a crucial role in" → "is critical for"
- **NO copula dodges**: "serves as" → "is"; "stands as" → "is"; "marks a" → "is a"
- **NO intensifiers without data**: very, extremely, highly, significantly, remarkably, substantially (except "statistically significant")
- **NO em-dashes** (---/—): use commas, semicolons, "which" clauses, or new sentences
- **NO promotional language**: groundbreaking, game-changing, pioneering, revolutionary
- **NO informal register** — five classes, a wordlist only reaches the first: `LEXIS` "at all", "in the first place", "ahead of time" → delete or formalise; `PHRASAL-VERB` "comes with" → "entails"; `JUDGMENT-ADJ` "is hard" → "is difficult"; `PREDICATE-METAPHOR` "the wall is a property" → the paper's own term; `WORK-TRACE` "what quarantine buys". **Check the card's allowlists first** — "from scratch", "rules out", "cheap unlearning" are terms of art. Word level: "a lot of" → "many"; "kind of" → "somewhat"
- **NO idiom collision**: a technical phrase that is also a common idiom is read as the idiom first — "a fair bit", "on the order of", "significant". Make the qualifier explicit ("an unbiased random bit")
- **NO register drop when simplifying or shortening** — judged on the diff: a replacement that is accurate but lower-register than what it replaced is still wrong ("route micropayments to peers" → "pay peers"). Replace using **wording the manuscript already uses elsewhere**. Report word count only *after* this check passes
- **NO vague attributions**: "Experts argue" / "Studies show" → cite specific source
- **NO vague quantifiers**: "some"/"many"/"several" → cite or quantify
- **NO Unicode arrows**: → ← ↔ ⇒ → use `$\rightarrow$` etc.
- **NO cleft constructions**: "That is what sets X" / "which is what makes X" / "What X is is Y" → plain subject-verb-object "X sets Y"
- **NO hypothetical foil**: "A method that only did X would stop there. Ours does Y." / "Once you view it as X" → state the result directly
- **NO abstract agency**: "the analogy's job", "the estimator carries decades of validation", "built to catch" → literal verbs ("has been validated", "detects")
- **NO rhetorical self-answers**: "The result? A new framework." → state directly
- **NO negative parallelism**: "It's not X — it's Y" / "not just X, but Y"
- **NO unnecessary contrast**: "X, not Y" / "X rather than Y" / "X instead of Y" → default to plain positive "X is A"; keep the contrast ONLY when ruling out Y carries information (don't just swap "not Y"→"rather than Y")
- **NO colon-numbered lists**: "we: (1)...(2)...(3)..." → use `enumerate` or prose
- **NO mid-sentence colons**: "key observation: the model fails" → full sentence or split (heading colons `\textbf{X:}` exempt)
- **NO trailing afterthoughts**: "..., as editable." comma + short tag → fold into main clause
- **Comma overuse**: max 3 commas per sentence (≥4 → split or use semicolons)
- **NO internal provenance** in text, captions or tables — result paths, data files, column identifiers, internal fixture names, or a superseded earlier version (`\path{experiments/...}`, `\texttt{empirical\_rate}`, "As Golden G4", "the old bound is retracted"). Provenance lives in the ledger and the released artifact; the manuscript cites the artifact
- **NO AI lexicon (tier-1)**: delve, leverage, underscore, harness, foster, showcase, seamless, intricate, nuanced, pivotal, tapestry, realm, myriad, "paving the way for", "valuable insights" → plain word or concrete noun. Tier-2 (comprehensive, essential, ensure, explore, insights, paradigm) ≤1 per sentence. Terms of art exempt (loss landscape, robust, optimize)
- **NO section previews/recaps**: "In this section, we present…" / "As we have seen…" / "Having described X, we now…" → start on content, end on the last concrete result; forward refs via `\Cref{}`
- **NO coined concept labels**: "the supervision paradox" / "workload creep" → cite a source, or declare it as your named contribution with a definition, or just describe the phenomenon
- **NO restating a proposition twice in one section**: abstract lead-in + evidence + synonym recap → keep only the placement next to the evidence
- **NO despite-dismissal**: "Despite challenges, X continues to thrive" → analyze the challenge
- **NO superficial -ing suffixes**: trailing ", highlighting/underscoring/emphasizing/showcasing/fostering…" → be specific. Open set (", enabling/ensuring/providing…") is LLM-judged, not regex: delete the tail and ask whether checkable information was lost
- **NO dangling cross-references**: "Fig.~\ref{} illustrates X." → weave ref into analytical sentence; delete `\ref{}` and check if claim remains
- **Sentence length**: max 35 words per sentence
- **LaTeX**: `\begin{equation}` not `$$`; `\toprule/\midrule/\bottomrule` not `\hline`; BibTeX keys: `lastname_year_word` format
