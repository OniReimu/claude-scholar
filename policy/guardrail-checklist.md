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
- **NO rhetorical self-answers**: "The result? A new framework." → state directly
- **NO negative parallelism**: "It's not X — it's Y" / "not just X, but Y"
- **NO colon-numbered lists**: "we: (1)...(2)...(3)..." → use `enumerate` or prose
- **NO despite-dismissal**: "Despite challenges, X continues to thrive" → analyze the challenge
- **NO superficial -ing suffixes**: trailing ", enabling/ensuring/providing..." → be specific
- **Sentence length**: max 35 words per sentence
- **LaTeX**: `\begin{equation}` not `$$`; `\toprule/\midrule/\bottomrule` not `\hline`; BibTeX keys: `lastname_year_word` format
