---
name: Co-author Excerpt
description: This skill should be used when the user asks to "compile just this section separately", "make a standalone single-column PDF of the evaluation section", "I only want to see this part, not the whole paper", "put all the files this fragment needs in one folder", "make the prose fragmented / bullet points so colleagues stop saying it reads like AI", or complains that a separately compiled section "is full of ?? references". Produces a standalone excerpt PDF of one paper section for co-author review, sharing sources with the main document.
version: 0.1.0
tags: [latex, review, collaboration, writing]
---

# Co-author Excerpt

Produce a standalone, single-column PDF of **one section** of a paper so
co-authors can review it without opening the full manuscript, and without the
review turning into a debate about draft prose.

Two properties define the deliverable and neither is negotiable:

1. **Zero `??` in the compiled PDF.** An excerpt that prints unresolved
   references reads as broken and the reviewer stops trusting it.
2. **The excerpt shares source files with the main document.** Never copy a
   section into the excerpt directory. A copy drifts, and the two builds start
   disagreeing silently.

## Workflow

### 1. Establish the directory

Put every file the excerpt needs in one subdirectory of the paper, including the
driver, the section sources, the tables, the figures, and a `ref.bib` symlink.
Users ask for this explicitly and it also keeps the freeze step honest.

```
paper/
├── main.tex
└── excerpt/
    ├── excerpt.tex            <- driver, from assets/excerpt-driver.tex
    ├── section-body.tex       <- shared with main.tex
    ├── tables/
    ├── figures/
    ├── ref.bib -> ../ref.bib
    └── xrefs-frozen.tex       <- generated
```

### 2. Wire the shared sources

Define a path macro that resolves differently in each build root, then use it at
every `\input`:

```latex
% main.tex, BEFORE the first \input that uses it
\newcommand{\evalbase}{excerpt/}
\input{\evalbase section-body.tex}

% excerpt/excerpt.tex
\newcommand{\evalbase}{}
```

Move section files into the excerpt directory and repoint `main.tex` at the new
location. Do not duplicate.

### 3. Install the driver

Copy `assets/excerpt-driver.tex` and fill in the title, `\sysname`, and the
`\input` lines. It carries the disclaimer box, the preamble the shared sources
expect, and `\importfrozenlabels`, which is the part that is easy to get wrong.

### 4. Freeze the cross-references

Build the main document first so its `.aux` is current, then:

```bash
python3 scripts/freeze-xrefs.py \
  --aux paper/main.aux \
  --out paper/excerpt/xrefs-frozen.tex \
  --scan paper/excerpt --scan paper/excerpt/tables
```

Re-run it whenever the main document renumbers or a file is added to the excerpt.
Wire it as a `make xrefs` target so it is not forgotten.

### 5. Convert the prose to fragment style

Run-in bold headers and plain `itemize`, telegraphic entries, one visible
disclaimer at the top. Consult `references/fragment-prose.md` for the conventions
and, more importantly, for what fragment style does **not** license.

### 6. Verify

`exit 0` is not evidence. Check the artifact:

```bash
pdftotext excerpt.pdf - | grep -c '??'        # must be 0
grep -c '^!' build.log                         # must be 0
grep 'Overfull \hbox' excerpt.log | grep -c 'in paragraph'
```

Build the main document too, and confirm its page count and `??` count are
unchanged. The excerpt work touches shared files, so it can break the main build.

## Failure modes worth knowing before they happen

Consult `references/standalone-crossrefs.md` for the mechanics. In brief:

| Symptom | Cause |
|---|---|
| "Can be used only in preamble", once per label | `\newlabel` cannot be `\input` in the body |
| Every `\cref` is `??` but every `\ref` resolves | The `@cref` companion entries were not extracted |
| Excerpt prints the main document's float numbers | Frozen labels overwrote the excerpt's own; add the `\@ifundefined` guard |
| A `??` appears long after a file was added | The freeze script has a hardcoded file list |
| Tables shrink to unreadable type | `\resizebox`; use `p{}` columns and `\footnotesize` instead |

## Discipline

**Never wrap a table in `\resizebox`.** It scales each table by a different
factor, so the document ends up with several table font sizes, none matching the
body. Fix width with column types and font size: prose cells become
`>{\raggedright\arraybackslash}p{0.3\columnwidth}` (requires `array`), dense
numeric tables take `\footnotesize` plus `\setlength{\tabcolsep}{2pt}`.

**Grep with a positive control.** Before trusting a "no stale numbers found"
result, search for a value known to be present. A quoting mistake, or zsh's lack
of word splitting on unquoted variables, turns a multi-file grep into a search
for one nonexistent filename and returns a false all-clear.

**Sweep repo-wide when a number changes.** The same figure is usually restated in
the section body, in a limitation paragraph, and in the appendix. Fixing one site
and rebuilding cleanly proves nothing about the other two.

**Keep terminology single-valued.** Column headers and prose must use one word
for one object. Introducing "reimplementation" in a header while the prose says
"reconstruction" is elegant variation and reads as two different artifacts.

## Additional Resources

- **`assets/excerpt-driver.tex`** — the standalone driver, with the label reader
  and the disclaimer box.
- **`scripts/freeze-xrefs.py`** — discovers the excerpt's sources, extracts both
  the plain and `@cref` entries, and refuses to freeze locally defined labels.
- **`references/standalone-crossrefs.md`** — why each failure mode happens and
  the exact fix.
- **`references/fragment-prose.md`** — the writing conventions, the noun-pile
  trap, and the rules fragment style does not suspend.
