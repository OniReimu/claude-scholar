#!/usr/bin/env python3
"""What a rewrite changed in a paper's citations, cross-references, formulas and numbers.

    python3 fidelity.py ORIGINAL_DIR REWRITE_DIR

Reads every .tex file under each directory, comments removed. For each kind it prints what was lost and what
was added (as multisets). An added number or formula is checked against the original text: "in original" means
the value was restated, "NOT IN ORIGINAL" means a new fact, which must be removed or justified.
Exit 1 when a citation key or a label appears or disappears, a reference points to no label of the original, or
something NOT IN ORIGINAL was added; 0 otherwise. A key cited or referenced more or fewer times is only reported.
Lost numbers are reported but do not fail: a repeat replaced by "this rate" is fine, a lost result is not,
and only reading the place tells which.
"""
import re
import sys
from collections import Counter
from pathlib import Path

KEYS = r"\\(?:ref|label|eqref|autoref|cref|Cref|cite[pt]?)\{[^}]*\}"


def grab(d: Path) -> tuple[str, dict[str, Counter]]:
    body = "\n".join(re.sub(r"(?<!\\)%.*", "", f.read_text(encoding="utf-8", errors="ignore"))
                     for f in sorted(d.rglob("*.tex")))
    math = re.findall(r"\$[^$]+\$|\\begin\{(?:equation|align)\*?\}.*?\\end\{(?:equation|align)\*?\}", body, re.S)
    prose = re.sub(KEYS, " ", re.sub(r"\$[^$]+\$", " ", body))
    return body, {
        "citation": Counter(k.strip() for c in re.findall(r"\\cite[pt]?\{([^}]*)\}", body) for k in c.split(",")),
        "label": Counter(re.findall(r"\\label\{([^}]*)\}", body)),
        "reference": Counter(k.strip() for c in re.findall(r"\\(?:ref|eqref|autoref|cref|Cref)\{([^}]*)\}", body)
                             for k in c.split(",")),
        "formula": Counter(" ".join(m.split()) for m in math),
        "number": Counter(re.findall(r"(?<![\w.])\d+(?:[.,]\d+)*%?", prose)),
    }


def flat(s: str) -> str:
    return re.sub(r"[{}~\s]|\\,", "", s)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    ob, o = grab(Path(sys.argv[1]))
    _, w = grab(Path(sys.argv[2]))
    bad = False
    for kind in o:
        lost, added = o[kind] - w[kind], w[kind] - o[kind]
        if not lost and not added:
            print(f"{kind}: unchanged")
            continue
        print(f"{kind}: lost {sum(lost.values())}, added {sum(added.values())}")
        for x, n in lost.items():
            gone = kind in ("citation", "label") and x not in w[kind]
            bad |= gone
            print(f"  -{n} {x[:70]}" + ("  GONE" if gone else ""))
        for x, n in added.items():
            if kind in ("citation", "label"):
                note = "in original" if x in o[kind] else "NEW KEY"
            elif kind == "reference":
                note = "to a label of the original" if x in o["label"] else "TO NO LABEL OF THE ORIGINAL"
            elif kind == "number":
                note = "in original" if re.search(rf"(?<![\w.]){re.escape(x)}(?![\d])", ob) else "NOT IN ORIGINAL"
            elif x in ob or flat(x.strip("$")) in flat(ob):
                note = "in original"
            else:
                note = "NOT IN ORIGINAL"
            bad |= note.isupper()
            print(f"  +{n} {x[:70]}  {note}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
