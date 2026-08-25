# Making an excerpt compile standalone with zero `??`

Four failure modes account for essentially every `??` in an excerpt build. Each
one produced a broken PDF in practice before it was understood.

## 1. `\newlabel` cannot be `\input` in the body

The obvious approach is to copy the `\newlabel` lines out of `main.aux` into a
file and `\input` it. That fails with **"Can be used only in preamble"** on every
entry, because `\newlabel` is only valid while LaTeX is reading a `.aux` file.

Moving the `\input` into the preamble does not help either, since the labels must
survive into the body.

**Fix.** Read the file with `\newlabel` locally redefined to install the reference
directly into the macro `\ref` and `\cref` actually look up:

```latex
\makeatletter
\newcommand{\importfrozenlabels}[1]{%
  \begingroup
    \def\newlabel##1##2{%
      \@ifundefined{r@##1}{\global\@namedef{r@##1}{##2}}{}}%
    \input{#1}%
  \endgroup}
\makeatother
```

Plain LaTeX, no package. Call it after `\maketitle`.

## 2. cleveref needs the `@cref` companion

`\cref` does not read `r@<label>`. It reads `r@<label>@cref`, which carries the
reference type and number separately so that cleveref can print "Table 4" rather
than "4".

Extracting only the plain `\newlabel{L}` entries leaves every `\cref` as `??`
while every `\ref` resolves, which is a confusing symptom because the labels
visibly *are* present.

**Fix.** For each external label, extract both `L` and `L@cref` from the `.aux`.

## 3. Frozen labels overwrite the excerpt's own

Once the excerpt defines floats of its own, it has local labels. LaTeX reads the
excerpt's `.aux` at `\begin{document}`, so those labels exist by the time
`\importfrozenlabels` runs in the body. Without a guard, the frozen copy
overwrites them with the **main document's** numbering, and the excerpt prints
"Table VIII" for what is its own Table IV.

**Fix.** The `\@ifundefined{r@##1}` guard above. Local definitions win; the frozen
copy fills only genuinely external references.

**Corollary for the freeze script.** Never freeze a label the excerpt defines
itself. Compute `external = used - local` by scanning the excerpt's own sources
for both `\ref`-family calls and `\label` definitions.

## 4. A hardcoded file list goes stale

If the freeze script enumerates the excerpt's sources by hand, adding a file to
the excerpt silently drops its external references out of the frozen set. The
symptom appears much later as a `??` with no obvious cause.

**Fix.** Discover the sources by globbing the excerpt directory, skipping the
driver and the generated `xrefs-frozen.tex`.

## Sharing sources with the main document

The excerpt must build the **same files** the main document builds. A copy drifts.

Use a path macro defined differently in each root:

```latex
% main.tex
\newcommand{\evalbase}{excerpt/}
\input{\evalbase section-body.tex}

% excerpt/excerpt.tex
\newcommand{\evalbase}{}
\input{\evalbase section-body.tex}
```

Define `\evalbase` **before** the first `\input` that uses it in the main
document. Use the macro form consistently: a bare `\input{section-body.tex}` in
the excerpt happens to work because latexmk runs with that directory as cwd, but
it breaks the moment anything moves.

## Verification

`exit 0` from latexmk is not evidence. Check the compiled PDF:

```bash
pdftotext excerpt.pdf - | grep -c '??'          # must be 0
grep -c '^!' build.log                           # must be 0
grep 'Overfull \hbox' excerpt.log | grep -c 'in paragraph'   # tables
```

`in paragraph at lines X--Y` is a table or paragraph overflowing. `detected at
line N` is a display-math box and is usually pre-existing; do not conflate the
two counts.

When hunting stale numbers with grep, **run a positive control first**: search
for a value known to be present. A shell quoting mistake, or zsh's lack of word
splitting on unquoted variables, silently turns a multi-file grep into a search
for one nonexistent filename and returns a false all-clear.
