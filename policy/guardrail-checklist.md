# Guardrail Checklist (Compact)

> Auto-generated from `constraint_type: guardrail` rules. Embed in writing prompts (~200 tokens).
> Full rule cards in `policy/rules/`. This checklist is for prevention during writing;
> violations are caught post-hoc by `policy/lint.sh --constraint-type guardrail`.

## Prohibited Patterns (do NOT generate these)

- **NO filler phrases**: "In order to" → "to"; "It is important to note that" → delete; "plays a crucial role in" → "is critical for"
- **NO copula dodges**: "serves as" → "is"; "stands as" → "is"; "marks a" → "is a"
- **NO intensifiers without data**: very, extremely, highly, significantly, remarkably, substantially (except "statistically significant")
- **NO em-dashes** (---/—): use commas, semicolons, "which" clauses, or new sentences
- **NO promotional language**: groundbreaking, game-changing, pioneering, revolutionary
- **NO informal vocabulary**: "a lot of" → "many"; "kind of"/"sort of" → "somewhat"; "bigger" → "larger"
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
- **NO internal provenance in the manuscript**: text, captions and tables never name a result path, data file, column identifier, internal fixture, or a superseded earlier version of the work (`\path{experiments/results/...}`, `\texttt{empirical\_rate}`, "As Golden G4", "the old bound is retracted"). Provenance lives in the ledger and the released artifact; the manuscript cites the artifact. Exempt: `\includegraphics`/`\input` paths, the artifact `\url{}`, model names, seeds, and EXP-mandated status disclosures
- **NO AI lexicon (tier-1)**: delve, leverage, underscore, harness, foster, showcase, streamline, seamless, intricate, meticulous, nuanced, multifaceted, pivotal, tapestry, realm, myriad, plethora, "paving the way for", "valuable insights", "at its core" → plain word or a concrete noun. Tier-2 (comprehensive, essential, vital, ensure, explore, enhance, insights, paradigm, interplay) ≤1 per sentence. Term-of-art uses (loss landscape, robust, optimize, trajectory) are exempt
- **NO section previews/recaps**: "In this section, we present…" / "This subsection describes…" / "As we have seen…" / "Having described X, we now…" → start on content, end on the last concrete result; forward references via `\Cref{}`
- **NO coined concept labels**: "the supervision paradox" / "workload creep" → cite a source, or declare it as your named contribution with a definition, or just describe the phenomenon
- **NO restating a proposition twice in one section**: abstract lead-in + evidence + synonym recap → keep only the placement next to the evidence
- **NO despite-dismissal**: "Despite challenges, X continues to thrive" → analyze the challenge
- **NO superficial -ing suffixes**: trailing ", enabling/ensuring/providing..." → be specific
- **NO dangling cross-references**: "Fig.~\ref{} illustrates X." → weave ref into analytical sentence; delete `\ref{}` and check if claim remains
- **Sentence length**: max 35 words per sentence
- **LaTeX**: `\begin{equation}` not `$$`; `\toprule/\midrule/\bottomrule` not `\hline`; BibTeX keys: `lastname_year_word` format
