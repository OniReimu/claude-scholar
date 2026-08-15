# Acceptance cases — `PROSE.REGISTER_PRESERVATION`

These are **not** lint fixtures. The rule is `check_kind: llm_style` and scoped to a
diff, so there is nothing for `policy/lint.sh` to run; shipping regex for this class
was measured at precision 0.00 / recall 0.00 (see the rule card's Rationale).

The harness is a judgement pass: give the model the pre-edit and post-edit text of
one editing pass, apply the rule card's two questions, and compare against the
expected verdict below. A runnable version of case 1 lives in
`skills/writing-anti-ai/evals/evals.json` (case 7).

All fixtures are real: the must-flag rows are the nine violations a single
compression pass introduced into an ACM ASIA CCS §1 (968 → 728 words, 2026-08-15);
the must-not-flag rows are measured false positives from a 35-hit scan over §3–§6
of the same manuscript, which had passed eight external audit rounds.

## Must flag

The comparison basis is the pre-edit wording. Each row is a violation even though
the replacement is accurate and readable.

| # | Pre-edit (original) | Post-edit (violation) | Drift class |
|---|---|---|---|
| 1 | route micropayments to peers | pay peers | 3 — precision loss |
| 2 | a sanction set too high deters | too heavy a one drives off | 3 + 2 — pronoun for noun, phrasal verb |
| 3 | permanently excluding a verifier | excluding a verifier for good | 1 — informal lexis |
| 4 | on a positive draw it shadow-sends | on heads it also sends | 1 — colloquial referent |
| 5 | Institutional instruments impose a cost | Institutions hold levers | 4 — metaphor in predicate |
| 6 | identifies the right failure | has the right instinct | 4 — anthropomorphism (also `PROSE.ABSTRACT_AGENCY`) |
| 7 | requires raising $q^*$ | calls for raising $q^*$ | 2 — phrasal verb for Latinate verb |

A finding is only complete when it names the repair **and its source**:

```
original → replacement → suggested wording → source of the suggestion
```

For rows 1–7 the suggested wording is the pre-edit column, and the source is the
pre-edit draft. In the live sweep, seven of nine repairs were recoverable this way
without inventing any phrasing.

## Must not flag

| # | Text | Why it must pass |
|---|---|---|
| 8 | `good-state attempts have expected count at most …` | Defined technical term (state slice $G_t$), not an evaluative adjective |
| 9 | `C1 rules out unilateral one-shot deviations.` | Standard game-theory idiom |
| 10 | `so the costly effort lands where the cheap signal is uncertain` | The paper's own defined contrast, introduced in the model section |
| 11 | `the composition lemma can plug in a measurable quantity` | Standard composition/modularity prose |
| 12 | `The relaxation loses information but enables a clean union bound.` | Established mathematical usage |
| 13 | A calibrated hedge left in place (`suggests`, `is consistent with`) | `PROSE.HEDGING_DISCIPLINE` owns these and outranks this rule |
| 14 | An unchanged span, however informal, that this pass did not touch | Author decision; `PROSE.INFORMAL_VOCABULARY` owns the word level |

Case 14 is the load-bearing one. A model that flags it has reverted to
document-scope judging, which is the failure mode this rule exists to avoid.

## Also excluded (from the same measured scan)

- `\big|`, `\Big[`, `\bigl(` — LaTeX sizing macros; a naive `\bbig\b` matches all of them
- Line-initial `and` / `so` / `also` — LaTeX source wrapping, not sentence-initial
  connectives. Rejoin wrapped lines before judging
- `several`, `various`, `a number of` — `PROSE.VAGUE_QUANTIFIERS` owns these; do not double-report
- Verbatim quotations and cited titles — never the author's register
