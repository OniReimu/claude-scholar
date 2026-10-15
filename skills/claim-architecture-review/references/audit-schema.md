# Audit artifact schemas

Field contracts for the five files under `architecture-review/`. Working-state files (`spine`,
`information-ledger`, `progress`) drive the passes; the two final artifacts are the deliverable plan.

## `spine.md` (working state — insight anchor)

```markdown
# Spine

## Paper-level claims (1–3)
- C1: <one sentence — the central claim> — `lead: capability`
- C2: <...> — `lead: none`

## Section obligations (one per core section)
- §Intro: motivate C1; state the gap.
- §Method: specify the protocol that makes C1 testable.
- §Results: present the evidence that supports/refutes C1–C2.
- §<...>: <the single job this section owes the spine>
```

Keep it small; it stays resident across all passes.

`lead` is the **kind of defensible advantage** the claim carries, from the closed set:

| `lead` | Meaning |
|---|---|
| `capability` | 新能力 — the paper does something that could not be done before |
| `mechanism` | 新机制 — it explains something competing work cannot explain |
| `cost` | 更低成本 — same outcome, materially cheaper (compute / data / annotation / latency) |
| `scale` | 更好扩展性 — it holds where others break (size, distribution shift, deployment scope) |
| `none` | true but leads on nothing — legal and explicit; a spine of all-`none` is a positioning problem, see `ml-paper-writing` → *选对战场* |

## `information-ledger.md` (working state — redundancy index, append-only)

One row per **information unit** (decompose compound paragraphs into multiple units). Redundancy is
detected by **lookup-before-create**: before adding a unit, scan existing rows for the same
proposition; if found, append the new location to `other-homes` instead of creating a row.

| Field | Meaning |
|---|---|
| `info-key` | short stable slug, e.g. `baseline-adaptation-protocol`. An **enumerated set** gets one key for the whole set (e.g. `feature-axis-instances`), never one key per member — a second enumeration of that set elsewhere must collide with this row |
| `canonical-proposition-gloss` | one plain sentence of the actual proposition (guards against paraphrase/key drift — two rows with the same gloss are duplicates even if worded differently) |
| `first-home` | section + paragraph locator where it first appears |
| `other-homes` | additional locators carrying the same proposition (the redundancy) |
| `unique?` | `true` if first-home is the only home |
| `first-use` | `def:<term>` rows only — the first locator, in reading order, that depends on the term's meaning. When it precedes `first-home`, the row is a forward use; `first-home: PENDING` at the end of P1 means undefined |

## `paragraph-audit.md` (final artifact — per-paragraph table)

One row per paragraph, appended section-by-section.

| Field | Values |
|---|---|
| `loc` | section + paragraph locator plus file and line (e.g. `§4 ¶3 · sections/design.tex:88`); paragraph counts drift under run-in headings and `\looseness`, the line does not |
| `evidence` | a short verbatim quote that grounds the verdict (≤ 20 words) |
| `role` | `claim` \| `evidence` \| `interpretation` \| `method-setup` \| `positioning` \| `scope-limitation` \| `navigation` |
| `supports` | which `paper_claim` (C1..) or `section_obligation` it serves |
| `unique_info` | `true` \| `false` (false = its information has another home; cross-check the ledger) |
| `backs_lead` | `true` \| `false` (true = the claim it `supports` has `lead ≠ none`; `false` ⇒ candidate for `tighten` / `move:appendix`, never a licence to `delete`) |
| `required_caveat` | `true` \| `false` (true = threat-model boundary / scope condition / overclaim defense / venue-mandated Limitation — load-bearing, not deletable) |
| `method_role` | Method only (one or more sections): `problem-setup` \| `novel-mechanism` \| `standard-component` \| `instantiation` \| `evaluation-leak` \| `running-example` \| `formal-result` \| `navigation`. `instantiation` and `evaluation-leak` take a `move` with a named destination; a `novel-mechanism` row names the claim it backs and notes which of *what it does / why this design / where it sits in the data flow* is missing |
| `embedded` | Method only, optional: a sentence of another role inside the paragraph, e.g. `instantiation: "0.2 × mean column norm" → §6.1` |
| `lim_class` | Limitations items only: `A` (bounds a named finding) \| `B` (explains the authors' effort) \| `C` (answers a critique nobody raised). Only `A` sets `required_caveat=true` |
| `canonical_home` | where this information should live (may differ from `loc`) |
| `verdict` | `keep` \| `tighten` \| `merge` \| `move:<section>` \| `move:appendix` \| `split` \| `delete` \| `escalate`; additive, proposed only: `gloss` \| `add-reason` \| `add-value:<section>` |
| `confidence` | `high` \| `medium` \| `low` (low ⇒ verdict must be `escalate`, never `delete`) |

Rules: `delete` legal only when `unique_info=false` AND `required_caveat=false`; every `move`/`merge`/`delete` names the surviving `canonical_home`. `backs_lead=false` alone never authorises `delete` — it caps the paragraph at `tighten` / `move:appendix`, and `required_caveat=true` overrides it entirely.

## `relocation-map.md` (final artifact — cross-section consolidation)

```markdown
# Relocation map

## Narrative-closure gaps (from P3)
- <gap: e.g. "C2 is introduced in §5 but never set up in §1">

## Redundancy clusters (from P2)
### Cluster: <info-key> — "<canonical-proposition-gloss>"
- Homes: §4 ¶3 (full), §5 ¶1 (re-explained), Appendix B (re-explained)
- Canonical home: §4 ¶3 (protocol)
- Collapse plan: §5 ¶1 → one forward reference ("per the protocol in §4"); Appendix B → keep only the extra numeric detail, drop the re-explanation.

## Method allocation (from the M-pass)
- Word share by role: novel-mechanism 18% · standard-component 41% · instantiation 33% · problem-setup 8% (exemplar range for novel-mechanism: 35–60%)
- §4 ¶5 (novel-mechanism): states what the gate does; missing why a mean aggregator fails here → add one design-reason sentence.
- §4 ¶2–¶3 (standard-component): FedAvg local training in three paragraphs → one sentence + citation.
- §4 ¶6 (instantiation): λ = 0.1, ResNet-18, SGD settings → move to §5.1 Experimental Settings (Implementation Details paragraph).
- Unvalued parameters (check 5): τ, T_min, γ_drop have no value in §5.1, the appendix or any table → add-value:§5.1, values `<TODO>`.

## Forward uses and undefined terms (from P1)
- `cascade-drift`: first use §1 ¶3, defined §4 ¶2 → one-clause gloss at §1 ¶3 ("the error each stage inherits from the stage before it"); definition stays in §4.
- `C6`: used §3–§6 (9×), never defined → define at first use in §3 ¶1, or replace the label with the claim it names.
```

## `exemplars.md` (working state, optional — from P0b)

```markdown
# Exemplars — <problem class>, <venue set>
Relaxations: <none | window → 5 years | class → "<parent class>" | venue → next tier>
| # | Paper (verified) | Venue, year | ID | Source |
|---|---|---|---|---|
| E1 | <title> | ICLR 2025 | arXiv:2xxx.xxxxx / DOI | LaTeX · venue via OpenReview decision |

Dropped: <title> — <reason: venue unverifiable / out of scope / pre-acceptance source>

## Script profile (method_profile.py; same flags apart from section selection)
| | method sections | words | share | words/para | display_eq | ladder | bare_trans | numeric/100w | where | cites | unvalued |
| E1 | … |
| Draft | … |
| Range (E1–En) | min–max per column |

## Judgment columns
| | novel-mechanism share | values in Method? | symbol style | design reason stated? |
| E1 | 45% | none (all in §5.1) | data-flow role | yes |
```

Fewer than 3 verified exemplars → write that line and no range; the M-pass falls back to the rule-card defaults.

## `progress.md` (working state — resumability)

```markdown
# Progress
- [x] P0 spine
- [x] P1 §1 Introduction
- [ ] P1 §4 Method   ← resume here
- [ ] P2 redundancy
- [ ] P3 closure
```
