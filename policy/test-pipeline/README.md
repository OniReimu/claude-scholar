# Pipeline acceptance fixtures

`policy/test-corpus.sh` scores **one rule at a time** against labelled snippets.
This suite scores the **two-stage pipeline end to end**: does
`claim-architecture-review` → `writing-anti-ai` remove what should go, keep what
should stay, and do it in that order.

A fixture is a directory under `policy/test-pipeline/`:

| file | role |
|------|------|
| `input.tex` | the draft, with planted defects and planted false-positive traps |
| `PROMPT.md` | the exact instruction given to a fresh agent |
| `assert.txt` | machine-checkable checkpoints (`GONE` / `KEPT` / `VERBATIM`) |
| `expect.md` | the sealed narrative key — what each checkpoint tests and why |
| `reference-final.tex` | a recorded passing run, so CI can verify the assertions still hold without spawning an agent |

## Running it

The deterministic half is a script; the pipeline half is an agent, so the two
are separate steps on purpose.

```bash
# 1. give PROMPT.md to a fresh agent with a copy of input.tex in <RUNDIR>
# 2. score whatever it produced
./test-pipeline.sh adaptive-compression <RUNDIR>/final.tex

# self-test: score the recorded reference run (this is what CI does)
./test-pipeline.sh adaptive-compression
```

The scorer also checks that **every number in the output appears in the input** —
a fabricated figure fails the run regardless of the assertions.

## What this suite cannot check

Whether the agent gave the *right reason*. `assert.txt` sees only the final text,
so a run that deletes the right paragraph for a wrong reason still passes. The
reasons live in the agent's Stage-1 table and Stage-2 report, and a human reads
them against `expect.md`. Do not add assertions that try to pattern-match
reasoning — that is how a harness starts measuring its own vocabulary.

## Adding a fixture

Plant three things, not one: defects the pipeline should catch, **traps it must
not fire on**, and at least one **ordering trap** — a line-level hit sitting
inside a paragraph that the structural pass should have removed first. Without
the last one, a pipeline that runs the stages backwards still scores full marks.

Then **run the negative controls before trusting the fixture**. A harness that
has only ever passed carries no information:

```bash
./test-pipeline.sh <case> <case>/input.tex          # unedited draft must FAIL loudly
```

The first fixture was written with two assertions whose labels claimed to catch
the ordering trap; feeding in a line-only edit showed they pass in exactly that
case, and the real guard is the paragraph's own topic phrase. Labels drift
toward what you meant rather than what they test — the control is what tells
you which one you wrote.
