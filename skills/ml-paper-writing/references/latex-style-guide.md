# LaTeX Style Guide

Preferred LaTeX formatting conventions based on CCS/TIFS/S&P and top AI conference paper styles.

---

## Section Heading Hierarchy

| Level | LaTeX Command | Format | Example |
|-------|--------------|--------|---------|
| 1st | `\section{}` | Numbered + ALL CAPS | `1 INTRODUCTION` |
| 2nd | `\subsection{}` | Numbered + Title Case | `2.1 Proof-of-Anything` |
| 3rd | `\subsubsection{}` | Numbered + Sentence case + period | `4.2.1 Generating proof.` |
| 4th | `\smallskip\noindent\textbf{}` | Bold inline heading + period | `Chained watermark generation.` |

### 4th-Level Heading Format (`\smallskip\noindent\textbf{}`)

The 4th-level heading is the most commonly used paragraph-level structure for further subdividing content under `\subsubsection`.

**Formatting Rules:**
- `\smallskip` adds a compact visual separator before short paragraph-level headings
- `\noindent` removes paragraph indentation
- `\textbf{}` bolds the heading text
- Heading ends with a **period** (not a colon, not a period followed by a blank line)
- Body text follows **immediately after** the heading, within the same paragraph
- Use Sentence case (only capitalize the first word and proper nouns)

**Correct Usage:**
```latex
\smallskip\noindent\textbf{Chained watermark generation.} To ensure cryptographic
linkage between successive training phases, PoLO derives the watermark
$\Lambda_x$ for shard $s_x$ from the final weights $W_{x-1}$.

\smallskip\noindent\textbf{Watermark embedding in model training.} During the
training of $s_x$, the prover $\mathcal{P}$ selects specific layers
in the model to embed a unique watermark $\Lambda_x$.

\smallskip\noindent\textbf{Obfuscation-based privacy protection and shard formation.}
PoLO applies Gaussian noise by randomly selecting a subset of weights.
```

**Incorrect Usage:**
```latex
% Wrong: using colon instead of period
\smallskip\noindent\textbf{Watermark embedding:} During the training...

% Wrong: line break after heading, creating a separate paragraph
\smallskip\noindent\textbf{Watermark embedding.}

During the training...

% Wrong: missing \smallskip and \noindent
\textbf{Watermark embedding.} During the training...
```

---

## List Environments

### Compact List (packeditemize)

Use the custom `packeditemize` environment instead of the default `itemize` to reduce vertical spacing:

```latex
\newenvironment{packeditemize}{
    \begin{list}{$\bullet$}{
            \setlength{\labelwidth}{4pt}
            \setlength{\itemsep}{0pt}
            \setlength{\leftmargin}{\labelwidth}
            \addtolength{\leftmargin}{\labelsep}
            \setlength{\parindent}{0pt}
            \setlength{\listparindent}{\parindent}
            \setlength{\parsep}{0pt}
            \setlength{\topsep}{1pt}}}{\end{list}}
```

**Usage:**
```latex
\begin{packeditemize}
\item First item with no extra spacing.
\item Second item follows tightly.
\item Third item maintains compact layout.
\end{packeditemize}
```

### List Items with Bold Lead-in Words

List items begin with a bold phrase, followed by a colon and body text:

```latex
\begin{packeditemize}
\item \textbf{Heightened vulnerability:} As first framed by Jia et al.
(S\&P'21~\cite{xxx}), existing PoLs require exposing intermediate
training states.
\item \textbf{High verification costs:} In the Jia et al. formulation,
verifiers must replay or retrain from recorded snapshots.
\end{packeditemize}
```

### Nested Lists

- First-level list: `$\bullet$` (filled circle)
- Second-level list: `$\circ$` (open circle) or use nested `\begin{itemize}`

---

## Citation Style (hyperref)

```latex
\usepackage{hyperref}
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=teal,
    urlcolor=teal,
    citecolor=magenta,
    pdftitle={An Example},
    pdfpagemode=FullScreen,
}
```

- `\cite{}` citations display in **magenta**
- Internal cross-references (`\ref{}`) display in **blue**
- URLs display in **teal**

---

## Cross-Reference Format

Formatting for referencing figures, tables, and sections in body text:

```latex
% Figure
Fig.~\ref{fig:overview}

% Table
Table~\ref{tab:main}

% Section
\S\ref{sec:method}

% Appendix (bold style required by this project)
\textbf{Appendix~\ref{app:proofs}}

% Equation
\eqref{eq:main_bound}

% Algorithm
Algorithm~\ref{alg:training}

% Listing
Listing~\ref{lst:config}
```

Project convention:
- Use exactly `Fig.~\ref{}` for figures (not `Figure~\ref{}` in running text)
- Use exactly `Table~\ref{}` for tables
- Use `\S\ref{}` for section references
- Use `\textbf{Appendix~\ref{}}` for appendix references
- Use `\eqref{}` for equations
- Use `Algorithm~\ref{}` and `Listing~\ref{}` for algorithm/code listing references

---

## Math Notation Hygiene

Rules for variable names in math mode:
- Use `\[...\]` for display equations (avoid raw `$$...$$` blocks).
- Do not write long words as italic math identifiers in display equations.
- If a variable-like token has more than 3 letters, wrap it in `\text{}`.
- Keep symbolic variables short (e.g., `x`, `w_t`, `\Lambda_x`) and define all symbols in the notation table.

```latex
% Recommended
\[
\text{score} = \frac{\text{correct}}{\text{total}}
\]

% Avoid (long words in math italics)
\[
score = \frac{correct}{total}
\]
```

## Definition/Theorem Environments

```latex
\textbf{Definition 1} (Proof-of-anything, PoX). \textit{A prover
$\mathcal{P}$ sends a proof $P$ to a verifier $\mathcal{V}$, where
the resources required for verifying...}
```

Formatting notes:
- Number is **bold**
- Term name is placed in **parentheses**
- Definition body uses **italics**
- Ends with a period

---

## Related Work Comparison Table Templates

Related Work must include a comparison table with existing work. Below are two common templates:

### Template 1: Feature Comparison Matrix (checkmark/cross style)

Suitable for comparing security properties, feature coverage, and other binary attributes.

```latex
\begin{table}[!]
\centering
\caption{Comparison with existing PoL.}
\label{tab_comp}
\renewcommand{\arraystretch}{1}
\resizebox{\linewidth}{!}{
\begin{threeparttable}
\begin{tabular}{c|c|ccc|cc|cc}

\midrule

 \multicolumn{1}{c}{}
 & \multicolumn{1}{c}{ \textbf{\makecell{Ownership}}}
 & \multicolumn{3}{c}{\textbf{Security}}
 & \multicolumn{2}{c}{\textbf{Privacy}}
 & \multicolumn{2}{c}{\textbf{Low Ov.}}  \\

 \multicolumn{1}{c}{} &  \cellcolor{yellow!15}\ding{172} & \cellcolor{yellow!15}\ding{173} & \cellcolor{yellow!15}\ding{174} & \cellcolor{yellow!15}\ding{175} & \cellcolor{yellow!15}\ding{176} &  \cellcolor{yellow!15}\ding{177} & \cellcolor{yellow!15}\ding{178} & \cellcolor{yellow!15}\ding{179}  \\

\midrule

\makecell{PoL (GD)\\~\cite{ref1,ref2,ref3,ref4}}
& \xmark & \cmark & \xmark & \xmark & \xmark & \xmark  & \xmark & \xmark  \\

  PoL (hash)~\cite{ref5}
& \xmark  & \cmark & \cmark  &  \cmark & \xmark & \xmark  & \xmark & \xmark \\

 PoL (zkp)~\cite{ref6}
& \xmark & \cmark  & \cmark & \cmark & \xmark & \cmark & \multicolumn{1}{c}{\xmark} & \cmark  \\

\midrule

\multicolumn{1}{c}{\textbf{Ours}}
&\cellcolor{blue!10} \cmark & \cellcolor{blue!10} \cmark &\cellcolor{blue!10} \cmark & \cellcolor{blue!10} \cmark & \cellcolor{blue!10} \cmark & \cellcolor{blue!10} \cmark & \cellcolor{blue!10} \cmark & \cellcolor{blue!10} \cmark  \\
\cmidrule{5-9}
\end{tabular}
\begin{tablenotes}
        \small
        \item  \textbf{Notation:} \cmark~ for attack-resistance/property-held; \xmark~ vice versa; \textbf{Ov.} for overhead. Prevent:
        \item  \ding{172} Property A? \ding{173} Property B?
        \item \ding{174} Property C? \ding{175} Property D?
        \item \ding{176} Property E? \ding{177} Property F?
        \item \ding{178} Property G?  \ding{179} Property H?
\end{tablenotes}
\end{threeparttable}
}
\vspace{-0.4cm}
\end{table}
```

**Formatting notes:**
- Highlight own method row with `\cellcolor{blue!10}`
- Column headers use `\ding{172}`–`\ding{179}` numbering, explained in footnotes
- `threeparttable` + `tablenotes` for in-table footnotes
- `\resizebox{\linewidth}{!}{}` for adaptive column width

### Template 2: Qualitative Comparison Table (multi-column text descriptions)

Suitable for comparing frameworks, system designs, and multi-dimensional attributes.

```latex
\begin{table*}[!ht]
    \centering
    \caption{Qualitative comparisons between the proposed \textsc{Method} and existing frameworks}
    \label{tab:comparison}

    \begin{threeparttable}
    \resizebox{\textwidth}{!}{
    \begin{tabular}{>{\columncolor{DARKGRAY}}lllllll}
    \toprule
     \rowcolor{DARKGRAY}
     \multicolumn{1}{l}{\textbf{Framework}}   &
    \multicolumn{1}{l}{\textbf{Dimension A}}   &
    \multicolumn{1}{l}{\textbf{Dimension B}} &
    \multicolumn{1}{l}{\textbf{Dimension C}} &
    \multicolumn{1}{l}{\textbf{Dimension D}} &
    \multicolumn{1}{l}{\textbf{Dimension E}}  &
    \multicolumn{1}{l}{\textbf{Dimension F}}
    \\
    \midrule

    Method A~\cite{ref1} & Value & Value & Value & Value & \Circle & \Circle \\
    Method B~\cite{ref2} & Value & Value & Value & Value & \Circle & \Circle \\
    Method C~\cite{ref3} & Value & Value & Value & Value & Value & Value \\

    \midrule

    \textsc{Ours} & Value & Value & Value & Value & Value & \textbf{All} \\

    \bottomrule
    \end{tabular}
    }
    \begin{tablenotes}
        \scriptsize
        \item[\Circle] Lack of corresponding designs.
    \end{tablenotes}
    \end{threeparttable}
\end{table*}
```

**Formatting notes:**
- `table*` spans both columns in two-column layout
- `\rowcolor{DARKGRAY}` for header row background color
- `>{\columncolor{DARKGRAY}}l` for first column background color
- `\textsc{}` for method names in small caps
- `\Circle` indicates lack of corresponding design

---

## Common LaTeX Packages

```latex
% Required
\usepackage{amsmath, amssymb}     % Math symbols
\usepackage{graphicx}              % Images
\usepackage{booktabs}              % Professional table rules (\toprule, \midrule, \bottomrule)
\usepackage{threeparttable}        % In-table footnotes
\usepackage{makecell}              % Line breaks within cells
\usepackage{pifont}                % \ding symbols (checkmarks, crosses, etc.)
\usepackage{colortbl}              % Table cell colors
\usepackage{xcolor}                % Color definitions
\usepackage{hyperref}              % Hyperlinks

% Common symbol definitions
\newcommand{\cmark}{\textcolor{green!80!black}{\ding{51}}} % ✓
\newcommand{\xmark}{\textcolor{red}{\ding{55}}}            % ✗
\newcommand{\pmark}{\textcolor{blue!90}{\ding{109}}}       % ○ (partial)
```
