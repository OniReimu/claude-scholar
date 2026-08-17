# Extracting prose from `.tex` — the canonical recipe

**Who this is for**: every rule with `check_kind: llm_semantic` or `llm_style`. Those
rules ship no `lint_patterns`, so an agent executing them writes its own scanner —
and four specific mistakes in that scanner all produce the same outcome: a **false
"clean" verdict**. The text was never scanned; the report says it was.

`policy/lint.sh` already does this correctly (see `lint_prose_no_internal_provenance`).
Hand-rolled scanners are where it goes wrong. Copy the function below instead of
writing a new one.

---

## The four failure modes (all measured, 2026-08-17)

| Mistake | What it costs | Correct form |
|---|---|---|
| `line.split('%')[0]` to strip comments | Truncates at `$95\%$`, so **the rest of the line is never scanned**. Evaluation sections and every table are full of `\%` | `re.split(r'(?<!\\)%', line)[0]` |
| `re.sub(r'\$[^$]*\$', ' ', text)` to strip math | When `$` count is odd, pairing shifts and **whole paragraphs are swallowed**. One measured run lost 42% of a section and 45% of an appendix | For word-level checks, **do not strip math at all** — words like `buys` or `significantly` essentially never occur inside formulas |
| Scanning line by line | LaTeX hard wrapping splits phrases across lines, so multi-word matches silently miss (`no state discovery:\nit includes`) | Also scan a whitespace-collapsed copy of the whole text |
| Inconsistent `re.I` between two passes | A hit after a colon or at a sentence start is reported in one pass and missed in the other | Fix one case policy per checklist and apply it everywhere |

The second one is the dangerous one: it fails **silently and proportionally to how
much math the section contains**, which means the sections with the densest technical
prose are the ones scanned least.

## Why not just `rg` over the raw `.tex`

`rg` has the opposite failure: it matches **commented-out历史草稿**. Manuscripts
carry large `%`-commented previous versions, and hits inside them are pure false
positives that make a real finding list untrustworthy. Strip comments first, then
match.

---

## Canonical extraction

```python
import re

def extract_prose(tex: str, strip_math: bool = False) -> tuple[list[tuple[int, str]], str]:
    """Return (numbered_lines, joined_text) of renderable prose.

    numbered_lines : [(1-based line number, text)] — for file:line reporting
    joined_text    : whitespace-collapsed single string — for multi-word phrases
                     that LaTeX hard wrapping split across lines

    strip_math=False is the default and the right choice for word- and
    phrase-level checks. Set it True only for checks that must not see symbols,
    and accept that odd-$ files will lose text.
    """
    out = []
    for i, line in enumerate(tex.splitlines(), start=1):
        # Comments: only an UNESCAPED % starts one. `$95\%$` must survive.
        line = re.split(r'(?<!\\)%', line)[0]

        # Constructs that never render into the PDF.
        line = re.sub(r'\\(includegraphics|input|include|bibliography|usepackage|documentclass)(\[[^]]*\])?\{[^}]*\}', ' ', line)
        line = re.sub(r'\\(label|ref|Cref|cref|eqref|autoref|cite[a-z]*)\{[^}]*\}', ' ', line)

        if strip_math:
            # Pair-safe: only strip when the line has an even number of
            # unescaped $, otherwise leave the line intact rather than
            # swallowing the remainder of the paragraph.
            if len(re.findall(r'(?<!\\)\$', line)) % 2 == 0:
                line = re.sub(r'(?<!\\)\$[^$]*(?<!\\)\$', ' ', line)

        if line.strip():
            out.append((i, line))

    joined = re.sub(r'\s+', ' ', ' '.join(t for _, t in out))
    return out, joined
```

### Excluding display environments

Equation, figure, table and algorithm bodies are usually not prose. Strip them from
the *joined* copy only — dropping them line-wise would break the line numbering that
makes findings actionable:

```python
DISPLAY = re.compile(
    r'\\begin\{(equation|align|gather|figure|table|algorithm|lstlisting|verbatim)\*?\}'
    r'.*?\\end\{\1\*?\}', re.S)

joined_prose = DISPLAY.sub(' ', joined)
```

Captions are **not** display content and must stay in scope — in one measured sweep,
8 of 11 defects sat in captions, notation tables and appendices.

## Reporting

Report `file:line` from `numbered_lines`, never an offset into `joined`. A finding a
reader cannot navigate to is a finding they will not act on.

When a hit is found only in the joined copy (a phrase split by hard wrapping), report
the line number of its **first** token and say the phrase spans a line break.

## Case policy

Pick one per checklist and state it in the rule card:

- **Case-insensitive** for idioms and phrasal verbs (`At all` at a sentence start, `On heads` after a colon)
- **Case-sensitive** for defined terms and identifiers, where capitalisation carries meaning (`Golden-G4`, `L1`)

Two passes with different case policies over the same checklist produce findings that
disagree with each other, which reads as flakiness and gets the whole check discarded.

## Self-check before reporting "clean"

A clean verdict is a claim. Before making it, confirm:

1. The scanned character count is within ~10% of the source's non-comment character count. A large shortfall means the stripper ate prose
2. A known-present control string (pick any distinctive phrase from the section by eye) is found by the scanner
3. Both the line-wise and joined copies were scanned
