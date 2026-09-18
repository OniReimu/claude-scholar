# Attaching Overleaf as a submodule

Overleaf gives every project a Git remote. Treating it as a submodule of the research
repository works, and the contract permits it "after an explicit team decision" —
but the two repositories have different jobs, and most of the failures below come
from letting that distinction blur.

## The shape

**Only `paper/` is the submodule.** Overleaf carries the manuscript; everything else
belongs to the research repository.

```text
<project>/
  PROJECT-WORKSPACE.md
  src/  scripts/  experiments/  artifacts/paper/
  paper/                      ← the Overleaf repository
    .research-workspace.yml   ← parent_contract: "../PROJECT-WORKSPACE.md"
    main.tex  sections/  figures/
    figure-manifest.yml
```

With several manuscript versions, `paper/` becomes a container and only the Overleaf
one is a submodule:

```text
paper/
  aaai27/      ordinary directory
  arxiv/       ordinary directory
  preprint/    ← the Overleaf repository; parent_contract is "../../PROJECT-WORKSPACE.md"
```

**Do not mount the Overleaf repository as the whole subproject.** It is the most
natural-looking mistake and it breaks the link file: `.research-workspace.yml` says
`parent_contract: "../PROJECT-WORKSPACE.md"`, and when the submodule sits directly
under the monorepo root, `..` is the monorepo, which has no contract. The fix is not
to edit the link file — it was always right — but to add the project directory layer
above it. Moving the submodule down one level makes the existing line resolve.

Configure the contract's paper extension accordingly:

```yaml
paper:
  repository: "paper"                     # or "paper/preprint"
  link_file: ".research-workspace.yml"
  figure_manifest: "figure-manifest.yml"
```

## What may live in the Overleaf repository

The manuscript, its sections, figures, tables, bibliography, the link file and the
figure manifest. That is all.

Analysis code, raw data, logs and scratch files stay in the research repository. When
a project is migrated into the shape above, this usually means moving a `scripts/`
tree *out* of Overleaf into `<project>/scripts/`. Check first that no `.tex` file and
no manifest `generator` entry references it; if a manifest does cite it, the manifest
is what needs rethinking, because a generator is not manuscript content.

## Failure modes, in the order they bite

### Detached HEAD hides how far behind you are

A submodule checked out on a detached HEAD reports one sha and nothing else. The
parent can be pinned many commits behind the manuscript's real state — including
submission-relevant commits, author-list changes, section moves — and neither `git
status` nor `git submodule status` will say so.

```sh
git -C <sub> status -sb                       # "## HEAD (no branch)" is the tell
git -C <sub> branch -vv                       # where is the local branch?
git -C <sub> rev-list --left-right --count origin/main...HEAD
```

Before fast-forwarding, prove nothing unique is lost: both the detached HEAD and the
local branch must be **ancestors** of `origin/main`, i.e. zero commits ahead of it.
Then `git checkout main && git merge --ff-only origin/main`, and commit the gitlink in
the parent.

### Double-tracking: registered as a submodule, committed as blobs

The path is in `.gitmodules` and `.git/config`, the directory has its own `.git`, and
yet the parent index holds its files as ordinary `100644` blobs. The symptom is that
`git submodule status` **does not list it at all** — the parent sees a pile of files,
not a submodule, so the manuscript's real history is invisible from the research
repository.

```sh
git ls-files -s <path> | awk '{print $1}' | sort -u   # 160000 expected; 100644 means double-tracked
```

Repair rewrites the parent index, so check four things first:

1. The submodule itself is healthy — on a branch, clean, in sync with its remote.
2. `git diff --name-only -- <path>` is empty: the parent holds no content the
   submodule lacks.
3. Compare `git ls-files <path>` against `git -C <path> ls-files`. Paths tracked only
   by the parent are usually LaTeX build products the submodule deliberately ignores;
   they stop being tracked by either repository, which is correct. Anything else on
   that list needs a decision before it disappears from tracking.
4. The path is not gitignored — an ignored path cannot hold a gitlink.

Then:

```sh
git rm -r --cached <path>
git add <path>          # warns "adding embedded git repository"; the result is still correct
```

### `git add -A` turns every nested repository into a broken gitlink

Any directory with a `.git` inside — a vendored upstream, an independent subproject,
a stray clone — is added as a gitlink by a blanket `git add`, producing exactly the
double-tracked state above, several at a time. Before any blanket add:

```sh
git status --porcelain | awk '{print $2}' | while read f; do
  [ -e "$f/.git" ] && echo "EMBEDDED: $f"
done
```

Unstage them and add them to `.gitignore`, or register them properly as submodules.

### The submodule's `.git` is a file, not a directory

Local-only ignores do not go where reflex puts them:

```sh
GITDIR=$(git -C <sub> rev-parse --git-dir)   # e.g. ../.git/modules/<name>
printf 'path/to/file\n' >> "$GITDIR/info/exclude"
```

Writing to `<sub>/.git/info/exclude` fails with "not a directory".

### Moving the submodule

`git mv` handles it and updates `.gitmodules` automatically. A directory cannot move
into itself, so it takes two steps:

```sh
git mv <sub> _tmp && mkdir <sub> && git mv _tmp <sub>/paper
```

The `.gitmodules` *section name* keeps its old value while `path` updates. That is
cosmetic; renaming the section is riskier than leaving it.

### The access token lives in `.git/config`

Overleaf remotes are usually `https://<token>@git.overleaf.com/<id>`. `.git/config`
is not version-controlled, so the token is not in the repository — but it will be if
it is ever copied into `.gitmodules` or any committed file. Redact it when printing
remotes:

```sh
git -C <sub> remote -v | sed 's#//[^@]*@#//***@#'
```

## Hygiene inside the manuscript repository

Figure-generation pipelines write run records — `run.json` and similar — containing
`project_root`, `output_dir` and input paths **as absolute paths on the authoring
machine**. These reach the manuscript repository easily and are worth a scan:

```sh
git -C <sub> ls-files | while read f; do
  [ -f "$f" ] && grep -l -E '/Users/|/home/' "$f"
done
```

Three things to get right when cleaning up:

- **Check whether it is an anonymous submission before calling it an anonymity
  incident.** A stray `aaai2027.bst` in the file list proves nothing; read
  `\documentclass` and the author block. A public preprint carrying the author list
  and deliberate project links is a hygiene question, not a leak.
- **Preserve the provenance while deleting the path.** A run record's provider,
  model, arguments and reference-image flags are real provenance; move them into the
  figure manifest's `generator` field before deleting the file, and the deletion
  costs nothing.
- **Deleting from the tip does not purge history.** Say so rather than implying the
  file is gone. An anonymous version needs a fresh repository, not a delete commit.

## Checks worth keeping

```sh
git submodule status                         # every registered submodule listed, each on a branch
git -C <sub> status -sb                      # clean, and tracking a branch
git -C <sub> rev-list --left-right --count origin/main...HEAD   # 0 0
```

Plus, from the research repository: no paper-only remote on the primary repository
(the contract forbids it), and every `.research-workspace.yml` naming a
`parent_contract` that resolves to a file that exists.
