# Sealed key — adaptive-compression fixture

Read this only when scoring or when an assertion fails. The pipeline runner
must not see it; `PROMPT.md` puts it out of bounds.

Input: 378 words, 7 paragraphs. Two layers of planted defects plus three
false-positive traps.

## Structure layer — `claim-architecture-review` should own these

| # | paragraph | expected verdict |
|---|-----------|------------------|
| S1 | P1 `In this section, we present...` | `delete` — roadmap outside the Introduction |
| S2 | P2 `Understanding the interplay...` | `delete` — every sentence form A; `PROSE.SEMANTIC_IDLING` escalates the whole paragraph to P1, which finds `unique_info=false` |
| S3 | P4 `The codebooks are refreshed...` | `merge` into P3 — two-sentence orphan, codebook-maintenance detail |
| S4 | P5 `The proposed framework achieves faster convergence...` | `delete` — form B, one quantity in five wordings |
| S5 | P7 | `delete` — future-work boilerplate (explicitly not exempt) plus a `To summarize` recap |

## Line layer — `writing-anti-ai` should own these

| # | instance | expected action |
|---|----------|-----------------|
| L1 | `---` in P3 | fix — zero-tolerance em dash, and **structurally**: rewriting to `, a 12.5x reduction` keeps the trailing appositive and does not count |
| L1b | `; the inner codebook` in P3 | fix — `PROSE.SEMICOLON_RESTRICTION`, split into two sentences |
| L2 | `convergence-aware scheduler` | `flag` or `hint`; either is defensible. The card's conservative direction (§Rationale: do not report when uncertain) makes `hint` the better answer if the judge cannot cite a prior source |
| L3 | `, so the per-step communication volume drops` | **keep** — the consequence is empirically supported (512→41 MB) and falls outside the three convertible subclasses. Converting it to `therefore` is a failure |
| L4 | `Our approach` / `The proposed framework` / `the presented method` | one term survives; the other two go with their paragraphs |
| L5 | intensifier run in P5 | gone with the paragraph |

## False-positive traps — nothing may fire

| # | target | why it must survive |
|---|--------|---------------------|
| F1 | P3 body | named structure, dimensions, bit widths, measured numbers |
| F2 | P6 whole paragraph | the `because` clause introduces `1/(1-β)`, a quantity independent of the conclusion — a real mechanism, not circular attribution. Byte-identical survival is the assertion. Kept deliberately clean on every other axis (no semicolon, longest sentence 30 words) so that anything firing on it is the thing under test |
| F3 | `therefore` in P6 | correctly typed entailment from a stated premise |

## Ordering trap

`plays a crucial role` and `valuable insights` sit inside the P2 idling
paragraph. Both are textbook `PROSE.FILLER_PHRASES` / `PROSE.AI_LEXICON` hits.
They must disappear **because the paragraph did**. A run that edits the phrases
and keeps the paragraph has inverted the pipeline order, and the visible result
is a cleaner-sounding paragraph that still says nothing — which is the whole
reason the line edit runs last.

## Compression

Expect roughly 155–200 words out. The number is an observation, not a target:
every surviving word is a fact from the input, and the reduction comes from
paragraph deletion rather than from shortening sentences. `reference-final.tex`
records the 157-word run this fixture was built from.
