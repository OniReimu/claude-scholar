# Audit artifact schemas

Field contracts for the five files under `architecture-review/`. Working-state files (`spine`,
`information-ledger`, `progress`) drive the passes; the two final artifacts are the deliverable plan.

## `spine.md` (working state — insight anchor)

```markdown
# Spine

## Paper-level claims (1–3)
- C1: <one sentence — the central claim>
- C2: <...>

## Section obligations (one per core section)
- §Intro: motivate C1; state the gap.
- §Method: specify the protocol that makes C1 testable.
- §Results: present the evidence that supports/refutes C1–C2.
- §<...>: <the single job this section owes the spine>
```

Keep it small; it stays resident across all passes.

## `information-ledger.md` (working state — redundancy index, append-only)

One row per **information unit** (decompose compound paragraphs into multiple units). Redundancy is
detected by **lookup-before-create**: before adding a unit, scan existing rows for the same
proposition; if found, append the new location to `other-homes` instead of creating a row.

| Field | Meaning |
|---|---|
| `info-key` | short stable slug, e.g. `baseline-adaptation-protocol` |
| `canonical-proposition-gloss` | one plain sentence of the actual proposition (guards against paraphrase/key drift — two rows with the same gloss are duplicates even if worded differently) |
| `first-home` | section + paragraph locator where it first appears |
| `other-homes` | additional locators carrying the same proposition (the redundancy) |
| `unique?` | `true` if first-home is the only home |

## `paragraph-audit.md` (final artifact — per-paragraph table)

One row per paragraph, appended section-by-section.

| Field | Values |
|---|---|
| `loc` | section + paragraph locator (e.g. `§4 ¶3`) |
| `role` | `claim` \| `evidence` \| `interpretation` \| `method-setup` \| `positioning` \| `scope-limitation` \| `navigation` |
| `supports` | which `paper_claim` (C1..) or `section_obligation` it serves |
| `unique_info` | `true` \| `false` (false = its information has another home; cross-check the ledger) |
| `required_caveat` | `true` \| `false` (true = threat-model boundary / scope condition / overclaim defense / venue-mandated Limitation — load-bearing, not deletable) |
| `canonical_home` | where this information should live (may differ from `loc`) |
| `verdict` | `keep` \| `tighten` \| `merge` \| `move:<section>` \| `move:appendix` \| `split` \| `delete` \| `escalate` |
| `confidence` | `high` \| `medium` \| `low` (low ⇒ verdict must be `escalate`, never `delete`) |

Rules: `delete` legal only when `unique_info=false` AND `required_caveat=false`; every `move`/`merge`/`delete` names the surviving `canonical_home`.

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
```

## `progress.md` (working state — resumability)

```markdown
# Progress
- [x] P0 spine
- [x] P1 §1 Introduction
- [ ] P1 §4 Method   ← resume here
- [ ] P2 redundancy
- [ ] P3 closure
```
