---
name: research-workspace-layout
description: This skill should be used when the user asks to "start a new paper project", "set up a repo for this paper", "scaffold a research project", "organise this project", "the directory is a mess", "apply the workspace contract", "adopt the research file structure", "where should this cache directory go", "add PROJECT-WORKSPACE.md", "link Overleaf as a submodule", "why does git say the submodule has untracked changes", or when a research repository has accumulated parallel copies of the same thing (cache_v2, cache_v2_now, cache_v2_now_rx) and needs a canonical layout. Covers both a project started from scratch and an existing one retrofitted. Do not use for LaTeX template organisation (latex-conference-template-organizer), experiment provenance narrative (lineage), or general git hygiene (git-workflow).
version: 0.1.0
tags: [Research, Workspace, Git, Overleaf, Reproducibility]
---

# Research Workspace Layout

Two jobs that turn out to be one: give a research project a directory layout that
cannot silently grow a twelfth cache directory, and attach its manuscript repository
without the two git repositories corrupting each other's view of the world.

The layout comes from the **project workspace contract** in
[Research-Workflow-Skills](https://github.com/DELONG-L/Research-Workflow-Skills)
(`research-workspace-governance`). That repository is the authoritative text for the
contract's fields. This skill is about **using** it: laying a new project out
correctly on day one, and retrofitting one that was not.

## Which of the two

| Situation | Go to |
|---|---|
| A project that does not exist yet, or exists as an empty repository | [Starting a new project](#starting-a-new-project) below — minutes, no risk |
| A project with existing code, data and history | [references/migration-playbook.md](references/migration-playbook.md) — order matters, and the first step is not the one people reach for |

Read the rest of this file either way. The closed set, the two collisions and the
relocation table apply to both; the difference is only whether anything has to move.

Starting new is not a smaller version of retrofitting — it is the case where the
expensive problems never form. `experiments/` never becomes a code tree, so the
collision below never arises; the ignore policy is right before any generator is
written, so `code_revision` is available from the first figure; and the first
experiment record is opened at design time rather than reconstructed from directory
timestamps afterwards. Every one of those is cheap now and costly later.

## The one idea

The topology is a **closed set**, not a description.

```text
<project>/
  PROJECT-WORKSPACE.md  task_plan.md  notes.md
  experiments/EXP-YYYY-NNN/     records only — no code, no run output
  src/  configs/  scripts/
  data/                          manifests and pointers
  artifacts/paper/               promotable lightweight exports
  paper/
```

Nothing else at the project root. This is stricter than the upstream contract, which
names the topology without forbidding additions, and the strictness is the entire
value: **eleven `cache_*` directories do not appear because anyone decided to have
eleven. They appear because nothing said where the twelfth would have to go.** A
layout that merely describes what exists cannot prevent the next one.

Adoption is opt-in per project, by placing `PROJECT-WORKSPACE.md` at its root. Never
infer managed status from a directory being called `paper`, `experiment` or
`research` — the contract forbids it explicitly.

## Two collisions and a trap, whichever case you are in

**`paper/` means two different things.** In the contract it is an *independent paper
Git repository*. In most monorepos it is an ordinary directory holding `main.tex`.
Configure the `paper` extension only for a real separate repository; leave it `null`
otherwise, and keep figure provenance in the experiment record instead — the
contract permits the manifest to live in "the manifest, experiment record, or
project notes".

**`experiments/` means two different things.** In the contract it is the root for
`EXP-*` *records*. In most research repos it is the implementation code tree. In a
new project, put code in `src/` from the first commit and the collision never forms.
In an existing one, move it there — and do **not** instead nest the records deeper
(`experiments/records/EXP-*`), which invents a directory the contract does not have,
the very habit being cured.

**The trap: the ignore policy decides whether any of this is possible.** A repository
that ignores `*/scripts/` or `*/experiments/` keeps its generators out of version
control, and then two contract fields are unimplementable: `local.source_roots` is
*defined* as version-controlled roots, and every promoted figure must cite the
`code_revision` of its generator. Check the ignore rules first. If they block by
directory name, that is the first thing to change — see
[references/migration-playbook.md](references/migration-playbook.md), which also
covers why the change must be dry-run before it is made.

## Starting a new project

Copy the templates in `assets/`, do not retype them.

```sh
SKILL=<path to this skill>
mkdir -p <project>/{experiments,src,configs,scripts,data,artifacts/paper}
cd <project>
cp "$SKILL/assets/PROJECT-WORKSPACE.md" .
touch task_plan.md notes.md
cat "$SKILL/assets/gitignore.starter" >> .gitignore     # or the repository root's
```

Then, in order:

1. **Fill the contract.** `project_id` must match the directory name — a mismatch is
   an audit finding, and the id is what records and the optional extensions key on.
   Declare only source roots that exist *and* are version-controlled; `assets/gitignore.starter`
   already blocks by kind rather than by directory name, which is what keeps that
   true.
2. **Attach the manuscript repository, if it is a separate one.** Create the Overleaf
   project first, then `git submodule add <url> paper`. Copy
   `assets/paper-workspace-link.yml` to `paper/.research-workspace.yml` and
   `assets/figure-manifest.yml` to `paper/figure-manifest.yml`, then switch the
   contract's `paper: null` to the mapping shown in the template. If the manuscript
   is an ordinary directory inside this repository, leave `paper: null` and keep
   figure provenance in the experiment record instead.
3. **Open the first record before running anything.** `cp -r assets/EXP-YYYY-NNN
   experiments/EXP-2026-001` and fill `design.md`, `protocol.md` and
   `analysis-plan.md` *first*. The value of the seven files is almost entirely in
   the three written before the run: they are what makes `deviations.md` meaningful
   later, and a plan reconstructed after the fact cannot be distinguished from the
   result it was written to fit.
4. **Append to `execution-log.md` as runs happen.** This is the one that gets
   skipped, and the cost is paid in full later: which run directory belongs to which
   planned phase is recorded nowhere else, and timestamps cannot recover it.
5. **Run the validator** once the first record exists, and keep it in whatever check
   the project already runs.

## Where an existing directory goes

Relocation should be a lookup, not a fresh judgement each time.

| Pattern | Destination |
|---|---|
| `cache*`, `results*`, `runs*`, `analysis-output`, `*_artifacts` | the `experiments/EXP-*/` of the experiment that produced it; heavy payloads stay on disk outside git, named in that record's `data-manifest.yml` |
| `pbs*`, `*_scripts`, job submission trees | `scripts/` |
| the project's own package, `*_eval`, `*-eval` | `src/` |
| `cfgs`, `*_configs`, `generated_configs` | `configs/` |
| `figures`, `imgs`, `img` | `paper/figures/` when in the paper, `artifacts/paper/` when exports |
| `anonymous_release`, `*_release`, `arxiv_submission` | `artifacts/paper/` |
| `*review*`, loose `*_review_comments*.md` | `paper/` |
| `docs`, `notes`, `handover`, loose planning `.md` | `notes.md`, or the `experiments/EXP-*/` it describes |
| `_archive_*`, `ARCHIVE_*`, `*_retired_*`, `.prev_*` | deleted or moved out of the repository — **not relocated inside it** |
| `__pycache__`, `tmp`, `log_files`, `scratch_*` | gitignored and deleted |
| vendored upstreams, `external/` | third-party; not part of the project tree |

Two directories recur that the upstream topology has no slot for. Neither becomes a
new root-level slot:

- **`tests/` → `src/tests/`.** Not `experiments/`, which is records only; not
  `scripts/`, where a test failure reads as a job failure. Tests assert things about
  `src/` and belong beside it, and `src/` is already a declared source root, so
  nothing new appears at the root.
- **`docs/` is refused.** It looks like a missing category but is an overflow pile:
  experiment plans are `EXP-*/design.md` and `analysis-plan.md`, status notes are
  `results.md`, data catalogues are `data-manifest.yml`, `*_appendix.tex` is paper
  material, discussion logs are `execution-log.md`. Adding the slot preserves the
  pile and leaves the records empty, which is the opposite of the point.

## Multiple manuscript versions

`paper/` is a container for the manuscript family, one directory per version:

```text
paper/
  iclr27/      the live submission
  neurips26/   the superseded one, with its reviews
  arxiv/       the preprint package
```

Only the directory that is an actual Overleaf repository is a submodule. See
[references/overleaf-submodule.md](references/overleaf-submodule.md).

## Doing the migration

The full procedure, including the ignore-policy change and its dry-run loop, is in
[references/migration-playbook.md](references/migration-playbook.md). Two rules from
it are load-bearing enough to state here:

**Leave nothing at the old path, and add no symlinks.** With eleven similarly-named
cache directories in play, a missed reference that *silently resolves to a different
one* can change a number in a paper. A missed reference that raises
`FileNotFoundError` cannot. Loud failure is the safety property being bought.

**Find the module that computes paths from its own location before moving it.** A
constants module with `PROJECT_ROOT = Path(__file__).resolve().parent.parent` lands
silently wrong the moment the package descends a level, taking every derived path
with it and raising nothing. Grep for `__file__` and `parent` before the move, not
after.

## Verifying

`scripts/validate_workspace.py` checks a repository against the contract and against
the closed set. Run it before and after any restructuring:

```sh
uv run --with pyyaml <skill>/scripts/validate_workspace.py            # whole repo
uv run --with pyyaml <skill>/scripts/validate_workspace.py <project>  # one project
```

It reports: contract front-matter validity; declared source roots that do not exist
or are gitignored; `EXP-*` records outside the declared record root; paper link files
whose `parent_contract` resolves to nothing; manifest entries missing provenance;
a manifest `project_id` matching neither its project nor the parent; a paper-only
remote on the primary repository; registered submodules committed as ordinary blobs;
and every root entry outside the closed set.

Four manifest forms the checker must accept, because the contract allows them and a
naive checker reports all four as broken: a `generator` that *describes a manual
process* instead of naming a file; `source_data` as a semicolon-separated list or a
YAML sequence; a path addressing a member inside an archive as
`archive.zip::inner/path`; and `code_revision` declared once for the whole manifest
rather than per artifact, with `data_snapshot` standing in for `experiment_id` when
no executable run produced the artifact.

## What not to decide alone

The contract is explicit, and it is right: changes touching a **project identifier**,
a **repository boundary**, a **remote location**, or a **data-retention policy** are
reported, not repaired. Rewriting a parent repository's index, pushing a deletion to
a manuscript remote, and moving a submodule are all in that class.
