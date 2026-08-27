# Runner prompt — adaptive-compression fixture

Hand this to a **fresh** agent that has not seen `expect.md`, `assert.txt`, or
`reference-final.tex`. Substitute `<RUNDIR>` for a scratch directory holding a
copy of `input.tex`.

---

You are executing the claude-scholar paper-polishing pipeline on one LaTeX
section. Work entirely inside `<RUNDIR>`. The input is `input.tex`. Do not read
any file named `expect.md`, `assert.txt`, or `reference-final.tex` — opening one
invalidates the run.

## Stage 1 — claim-architecture-review (structure first)

Read `skills/claim-architecture-review/SKILL.md` and run its targeted entry on
this single section: P0 (spine from headings + topic sentences), then P1
(per-paragraph audit). Number the paragraphs in document order. Give each a
verdict from `{keep, tighten, merge, move, split, delete, escalate}` with a
one-line reason, and write the audit to `stage1-audit.md`. The stage is
propose-only; for this run, treat your own verdicts as approved and apply them
to produce `stage1-out.tex`.

## Stage 2 — writing-anti-ai (line edit, LAST)

Read `skills/writing-anti-ai/SKILL.md` and `policy/style-guide.md`, plus any
card under `policy/rules/` a judgment call needs. Apply the skill to
`stage1-out.tex`. Honor the skill's reporting requirement: report flagged items
**and** deliberately kept or cleared ones with reasons. Write `final.tex` and
`stage2-report.md`.

## Constraints

- Do not invent experimental numbers. Facts already present are real and must
  survive if their sentence survives.
- A paragraph with no surviving proposition is deleted or merged, never
  paraphrased shorter.
- Keep the LaTeX valid.

Return: the Stage-1 verdict table, the Stage-2 report (flagged and kept), the
full `final.tex`, and input vs final word count.
