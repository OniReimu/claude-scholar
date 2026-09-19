---
name: lineage
description: "Opt-in experiment-lineage view for a research project — which lines of work are running, which quietly stalled, which finished and what they concluded, and which results were produced but never read by the paper. Use when the user asks to turn it on ('open lineage', 'track this project', '开 lineage'), and whenever they describe the symptom it exists for: 'what am I running right now', 'did that job finish or die', 'what did I finish this week', 'I keep losing track of which experiments are still open', 'is anything blocking submission', '哪条线还在跑', '我做到哪了'. Do NOT start it unprompted on a project that has no outline yet — offer it once, in one line, and let the user decide."
---

# Lineage

A project's experiment lines live in an outline (`docs/lineage.md`); their state is collected
(`lineage.py` → `docs/lineage.json`) and rendered into a page a person opens.

**You maintain the structure. The collector supplies the state. Never write state into the outline.**

## Reading state — do this before answering anything about progress

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/skills/lineage/scripts/lineage.py collect          # refresh, then read docs/lineage.json
```

`lineage.json` is the authority on what is running and what landed. Do **not** quote job ids,
exit codes or scores from `task_plan.md`, `handover.md` or any prose file — those are snapshots
somebody wrote on an earlier day and are routinely stale by the time you read them.

Useful fields: `counts` (open leaves by state), `nodes[]` (each line: `signal`, `jobs`,
`artifacts`, `consumed_by`, `last_activity`), `evidence` (landed vs quoted in generator code),
`buckets` (scripted prefixes with nothing to show), `coverage` (activity belonging to no line),
`new_events` (what changed since last collection).

## Refreshing the page — after anything changes

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/skills/lineage/scripts/lineage.py --emit docs/site
```

Writes `docs/site/index.html` (self-contained) and `docs/site/lineage.json`. Tell the person the
path once; they keep it bookmarked and see whatever your last run collected.

Run this after: submitting or cancelling a job · syncing results · editing the outline ·
finishing a manuscript pass · anything the person would want reflected.

## Writing structure — when work changes shape

Edit `docs/lineage.md`. Indentation is derivation — a child forked from its parent.

```markdown
- **Cross-model table 6** `[v107xm]` → its own table
  > does the K-score ordering survive across Mistral and Qwen at seed 0
  - Nothing reads the 44 cells yet `[v107xm]`
    > artifacts are synced and no generator names the prefix
  - Where v107xm lands in the paper `[—]` ✓ its own table; Table 6 stays
- **Six columns have no cross-model number** `[v115bench]` → tab:benchmark_compare
  > !blocks the table cannot ship with three of nine columns empty
```

- `[...]` — the match pattern: a prefix (`v107xm`), a range (`v40-v50`), a path (`paper/**`),
  a script (`run_x.pbs`), or `—` when the line leaves no trace. Backticks are the house style;
  bare `[v107xm]` is read too. A range takes any stem — `v40-v50`, `exp1-exp3`, `exp1-3`. **Writing a prefix before the
  first file lands is how a line becomes PLANNED** rather than "no experiment yet".
- `>` — one sentence: what this line would establish or overturn.
- `> !blocks <reason>` — on its own `>` line under the node: this line stands between the paper
  and submission. It goes to the top of the frontier in its own group, with a tile at the top of
  the page. Written inline in the node's title instead, it is just words in the title and no
  blocker is recorded. Closing the line clears it.
- `✓` / `✗` on closing. `✗` is falsified — that is paper material, not debt. A child may close
  while its parent stays open.
- Open a new node the moment a new line of work starts. A line nobody wrote down is invisible,
  and the page's "unattributed activity" section is where that failure shows up.

## When the tool is wrong

You will sometimes know something the collector cannot see — you read the log and know *why*
a job died, you read the generator and know a prefix really is consumed, you know results were
synced under a different name.

**Do not write that conclusion into `lineage.json`, and do not write state into `lineage.md`.**
Either one turns the page back into a hand-written document that nobody can date or verify: a
reader seeing `consumed` could no longer tell whether a probe confirmed it or an agent asserted
it two weeks ago, and your assertion will not expire when the code changes underneath it.

Fix the input, or fix the probe:

| what you found | what to do |
|---|---|
| a generator consumes a prefix through a variable | name it literally, or add `# lineage: consumes <prefix>` beside it — the scanner reads that comment |
| a line's results landed under another name | correct the `[...]` pattern in the outline |
| a job died for a knowable reason | that reason is a new line — write the node |
| the probe simply cannot see this class of thing | say so to the owner; a missing probe is a tool change, not an outline edit |

If you cannot fix it now, **write the mismatch itself as a node**. "The tool reads X as stalled,
but the results are on the cluster under Y" is a real piece of work with a real fix, and as a node
it survives until someone does it. Silently correcting the page loses that.

What you *should* write freely: structure, intent, verdicts, `!blocks` markers. Those are
judgements no probe can reach, and they are why the outline exists.

A quoted prefix in a generator file is **a mention, not proof the paper reads it** — a label, an
output path or dead code satisfies the same match. The page says "named in generator code" for
exactly that reason; do not upgrade it to "verified" when reporting to the user.

## Never

- Put job ids, exit codes, scores, progress or paths in the outline. Those rot; the collector
  supplies them fresh every run.
- Report "nothing is running" from a failed probe. The collector marks a probe that could not
  look, and holds its previous reading — read `collectors[].ok` before drawing conclusions.

## Using this on another project

`python3 ${CLAUDE_PLUGIN_ROOT}/skills/lineage/scripts/lineage.py --init` probes the repository, guesses the prefix shape from the filenames it
finds, and writes a starter outline. Everything project-specific then lives in one optional block
at the top of that outline:

```
<!-- lineage
results: results                 # directory holding artifacts
scripts: run_*.pbs               # job scripts
generators: paper/scripts/*.py   # code that reads artifacts into the paper
prefix: v\d+[a-z]*               # what a version prefix looks like here
scheduler: pbs                   # pbs | slurm | none
clusters: host:username, ...     # ssh targets
remote_results: host:/scratch/*/out   # where a cluster writes, as a glob
training: saves/|--do_train      # marks a script whose output is not a result file
-->
```

**Every key is optional, and the tool degrades in layers rather than failing:**

| what the project has | what works |
|---|---|
| an outline only | the tree, the frontier, closing lines, `!blocks`, `[decision]` — no configuration at all |
| \+ git | last activity per line, live vs stalled, uncommitted work, unattributed activity |
| \+ a results directory | artifact counts, which lines have produced anything |
| \+ generator globs | whether any generator file quotes those prefixes |
| \+ a scheduler | running jobs, jobs that died far too soon |
| \+ a remote path | results produced on a cluster and never copied back |

A probe with nothing configured says so; it never reports an empty result as a clean one.

**What is not portable.** The scheduler layer speaks PBS (`qstat -x -u`) and Slurm
(`sacct -X -P`). Anything else — LSF, a bare queue, a laptop — should set `scheduler: none` and
lose only the job columns. The Slurm path has been unit-tested against real `sacct` output format
but not against a live Slurm cluster; if the first collection returns rows it cannot parse it will
say `N rows unparsed` rather than pretend the queue is empty.

**What generalises perfectly**, because it is not about clusters at all: the outline, derivation
by indentation, the frontier grouped by stage, `!blocks`, `[decision]`, closing a line with a
verdict, and the record of what closed recently. That is the part worth handing to someone else.

## Turning it on for a project that has none

This skill is off by default. When the user asks for it, or when they describe the symptom:

```bash
cd <the project>
python3 ${CLAUDE_PLUGIN_ROOT}/skills/lineage/scripts/lineage.py --init
```

That probes the repository, guesses the prefix shape from real filenames, and writes a starter
outline at `docs/lineage.md`. Then fill the outline in **with the user**, not for them: the nodes
are their lines of work and their judgement of what each one would establish. A generated outline
nobody agreed with is worse than none.

The script locates the project by walking up from the working directory to the nearest outline,
so it runs correctly from anywhere inside the project; pass `--project DIR` to be explicit.

If the project should be self-contained for people who do not have this plugin, copy the script in
once — `cp ${CLAUDE_PLUGIN_ROOT}/skills/lineage/scripts/lineage.py .` — and run it from there.

## New project

`python3 ${CLAUDE_PLUGIN_ROOT}/skills/lineage/scripts/lineage.py --init` probes the repository, guesses the prefix shape and writes a starter
outline. With no config block it degrades to git-only and still renders the tree.
