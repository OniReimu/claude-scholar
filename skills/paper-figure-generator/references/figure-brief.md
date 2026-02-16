# Figure Brief Template (Conceptual Diagram)

Use this brief to make figure generation repeatable and reviewable. Save it as:

`figures/{topic-slug}/brief.md`

---

## Goal

- Paper section: (e.g., Figure 1 / System Model / Method Overview)
- Reader takeaway (one sentence):

## Layout

Choose one:
- `system-overview`
- `pipeline`
- `threat-model`
- `comparison`
- `architecture`

## Components (3-8 recommended)

List the blocks/entities you want in the diagram.

- Name:
  - Role (one line):
  - Key annotations (optional): (e.g., dimensions, formulas)

## Relationships / Data Flow

List directed edges in a machine-readable way:

- `A -> B`: label (optional)
- `C -> D`: label (optional)

## Grouping (optional)

- Group name: [component1, component2, ...]

## Inputs / Outputs

- Inputs:
- Outputs:

## Style

- Reference image path (optional): (used for style transfer)
- Notes: (e.g., "clean minimal", "NeurIPS 2025 gradient", "colorblind-safe")

## Constraints

- Must include: (e.g., "Top-k retrieval", "privacy boundary", "two-stage training")
- Must avoid: (e.g., "too many tiny blocks", "dense text")
- Must avoid: in-figure title/top heading text (use caption outside the figure instead)

## Caption Draft (optional)

One paragraph caption that can stand alone.
