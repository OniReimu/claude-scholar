# Migrating an existing repository into the contract

For a repository with years of accumulation, not a fresh project. Order matters: each
step below is a precondition for the next.

## 0. Survey before proposing anything

Count what is outside the closed set, per project. The number is the argument.

```sh
CANON="experiments src configs scripts data artifacts paper"
for d in */; do d=${d%/}
  n=$(ls -1 "$d" 2>/dev/null | while read e; do
        [ -d "$d/$e" ] && ! echo "$CANON" | grep -qw "$e" && echo "$e"; done | wc -l)
  echo "$n $d"
done | sort -rn
```

Also record, for each candidate directory: size, newest mtime, tracked-file count,
and how many places reference it. "Referenced in N files" is what decides whether a
move is cheap or delicate.

## 1. Fix the ignore policy first

Everything else depends on this. A policy that blocks **by directory name** —
`*/experiments/`, `*/scripts/`, `*/pbs/` — keeps generators out of version control,
and then `local.source_roots` cannot be satisfied and figure `code_revision` cannot
be produced. Change it to block **by kind**: artifacts and heavy data at any depth,
code everywhere.

**Dry-run before changing the file.** Two directions must both be checked, and the
second is the one that gets forgotten:

```python
# what would newly become visible
import pathspec, subprocess
from pathlib import Path
spec = pathspec.PathSpec.from_lines('gitwildmatch',
    [l for l in Path('proposed.gitignore').read_text().splitlines()
     if l.strip() and not l.startswith('#')])
ignored = subprocess.run(['git','ls-files','--others','--ignored','--exclude-standard'],
                         capture_output=True, text=True).stdout.splitlines()
new = [f for f in ignored if not spec.match_file(f)]
print(len(new), sum(Path(f).stat().st_size for f in new if Path(f).is_file())/1e6, "MB")
```

```sh
# what is already TRACKED that the new rules would ignore  ← the forgotten direction
git ls-files | git check-ignore --stdin --no-index --verbose | wc -l
```

A sweeping rewrite typically fails on the second check: a rule like `**/results/`
can match thousands of files that are already committed. Git keeps tracking them, so
nothing breaks loudly, but the rules and the repository now disagree permanently.
Measure the baseline first — the existing file usually has drift of its own — and
keep the new number at or below it.

Expect several rounds. Things that surface only in a dry run:

- A negation re-including a directory (`!**/data/**/`) pulls entire virtualenvs back in.
- A handful of files carry most of the bytes. Check for anything over 50 MB before
  pushing; a host will reject files over 100 MB and the rejection comes at push time,
  after the commit.
- Image and dataset trees under `data/` and `exps/` dwarf everything else; the
  contract already says `data/` holds manifests and pointers, so ignore its contents
  and re-include only the manifests.

**Prefer the minimal surgical change.** Removing the four rules that block code beats
rewriting the whole policy, and the result is far easier to review.

## 2. Move directories into the canon

Three rules, in order of how much damage they prevent.

**Leave nothing at the old path. Add no symlinks.** With several similarly-named
directories (`cache_v2`, `cache_v2_now`, `cache_v2_now_rx`), a missed reference that
silently resolves to a *different existing* directory can change a published number.
A missed reference that raises `FileNotFoundError` cannot. Loud failure is the whole
safety argument for the move.

**Find self-locating path code before moving it.** A constants module holding

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
```

lands silently wrong the moment the package descends into `src/`: `PROJECT_ROOT`
becomes `src/`, and every derived path follows it, raising nothing. Grep for
`__file__` together with `parent` before the move. Afterwards, that module should be
the *only* place that knows where data lives — convert cwd-relative literals in
scripts (`pd.read_csv("results_now/x.csv")`) to constants at the same time.

**Separate local paths from remote ones.** Job scripts carry cluster-side paths
(`#PBS -o /shared/homes/.../log_files/`) that describe a filesystem the migration did
not touch. Rewriting those silently breaks jobs. Filter them out before any bulk
edit:

```sh
grep -rIn 'old_path/' scripts | grep -v '/shared/' | grep -v '\.md:'
```

Grouping caches and result directories into experiment records needs a mapping from
directory to experiment. Look for an inventory the project already wrote before
inventing one; a stale inventory beats a guess, and where it is silent, say so in the
record rather than assigning an experiment id that nobody verified.

## 3. Write the contract, then the records

`PROJECT-WORKSPACE.md` per project. Declare only roots that exist and are
version-controlled — a declared root that is gitignored is a false claim the
validator will catch.

The first `EXP-*` record is worth writing retrospectively, and is worth writing
honestly:

- Prefer a project with a real pre-registration or design document; a record for an
  engineering milestone with no hypotheses turns the seven files into fill-in-the-blanks.
- Mark a reconstructed record as reconstructed. If no execution log was kept, the
  `execution-log.md` is an assembly from timestamps and committed planning documents,
  and should say so.
- `deviations.md` is where the migration pays off. Facts like "the plan file has said
  *Phase 4 next* since May while runs continue through August", or "these result
  directories correspond to arms the pre-registration does not describe", are
  discovered exactly once — while writing the record — and are worth more than the
  layout.
- Report a headline result with the caveats attached, not appended. A statistic that
  passes in every cell says nothing about the test's sensitivity when no negative
  control exists, and that belongs in the same paragraph as the number.

## 4. Verify, then commit in reviewable pieces

Run the validator. Then verify the things a validator cannot:

- **Build the paper from a scratch copy**, not in place — building in place
  overwrites artefacts someone else may be working on. `latexmk` returning 0 is not
  enough; check the **final** pass for unresolved references, since early passes
  legitimately report them. A byte-identical PDF is the strongest evidence a
  restructuring changed nothing.
- **Run one analysis script end to end**, the one that reads the moved data and
  writes a table the paper uses.
- **Scan for credentials** before pushing anything that was previously ignored.
  Expect false positives from test fixtures — a corpus of deliberately fake keys in
  an adversarial test file is not a leak, and documented example values such as
  AWS's `AKIAIOSFODNN7EXAMPLE` are not either. Read the context before reporting.

Commit the policy change, the bulk file addition, and each structural move
separately, so any one of them can be reverted alone. Say in the message which
verification was run and what its output was.

## 5. What stays open

Some findings are reports, not repairs. Provenance that points at generator scripts
which exist nowhere in the repository cannot be fixed by editing the manifest — the
files were never migrated, and the honest output is a record naming each missing one.
Changes to a project identifier, a repository boundary, a remote location or a
data-retention policy go to the user by the contract's own instruction.
