# Fragment-style prose for co-author review

## What it is for

An excerpt circulated for review invites two different kinds of comment. One is
about the content and the numbers. The other is about the writing, and when the
draft prose reads as machine-generated, that second kind crowds out the first:
co-authors spend the review objecting to the register instead of checking whether
the measurements are right.

Fragment style removes the target. Nobody objects to the phrasing of a bullet
list, because a bullet list does not claim to be finished prose.

This is a **review artifact**, not a draft of the section. Do not let it become
one.

## The conventions

**Run-in bold headers, then a plain list.** No flowing paragraphs.

```latex
\smallskip\noindent\textbf{Checkpoint runtime.}
\begin{itemize}
  \item Four architectures, same diagnostic workload (\cref{fig:runtime}).
\end{itemize}
```

**Telegraphic entries.** Label, colon, contents. A leading label colon is a
heading, and is exempt from the prohibition on mid-sentence explanatory colons.

```latex
\item Load: scheduled leaf position and metadata, plus the tree convention.
```

**A visible disclaimer at the top of the PDF**, so no reader has to guess:

> Reference material for co-authors. The prose here is deliberately fragmentary
> and is not manuscript text. Read it for content and for numbers. Do not paste
> it into the paper.

**Every number carries its provenance in a comment**, since the excerpt is where
co-authors will challenge the figures:

```latex
% Provenance for 94.9 and 49.5. reports/attribution.md, job 65668 iteration 3.
```

## What fragment style does not license

Fragment style suspends the requirement to write connected prose. It suspends
nothing else.

- **Register stays academic.** Terse is not casual. `entails`, not `comes with`.
  Compression is the most common way register drifts downward, because the
  cheapest words to cut are the precise multi-syllable ones.
- **Claims still need evidence.** A bullet asserting a comparison still needs its
  number and its reference in the same bullet.
- **Table and figure rules still apply.** Every cell is a measured quantity.
  Never put a growth rate in a cell whose row label names a quantity; if a cell
  needs the caption to be understood, restructure the table instead.
- **Numbers must be current.** The excerpt is the document co-authors read most
  closely, so a superseded figure here is more damaging than one buried in the
  main text.

## Bullet-level readability

The failure mode specific to this style is the noun pile: one `\item` carrying a
dozen comma-separated items. It is technically a list, and it is unreadable.

Split by kind, at most about four short entries inline:

```latex
% before: one item, 40 words, 14 commas
\item Every native invocation, local equation, producer-consumer edge, arity
      incidence bit, range and comparison width, bound proof, counter limb ...

% after
\item Every native invocation with its local equation, and its equation
      descriptor.
\item Every producer-consumer edge, arity incidence bit and transcript slot.
\item Every range and comparison width, bound proof, counter limb and carry
      constraint, and tree-shape check.
```

A wall of bare `\eqref` calls has the same problem. Name what is being referenced
so the reader learns something from the line:

```latex
% before
\item Recompute \eqref{eq:a}, \eqref{eq:b}, \eqref{eq:c}, \eqref{eq:d}, ...

% after
\item Recompute each canonical table with its commitment. SiLU is \eqref{eq:a}
      and \eqref{eq:b}, Softmax \eqref{eq:c} and \eqref{eq:d}.
```

## Promotion to manuscript prose

When a section graduates from excerpt to submission text, the bullets are raw
material, not a draft. Rewrite into connected prose and then run
`claim-architecture-review` before `writing-anti-ai`; line editing a structure
that is still in bullet order produces polished fragments in the wrong sequence.
