#!/usr/bin/env python3
"""Cut a one-file paper into chapters for the subagents, and put it back together.

    python3 split.py split MAIN.tex PARTS_DIR   -> PARTS_DIR/NN-name.tex and PARTS_DIR/parts.json
    python3 split.py join  PARTS_DIR MAIN.tex   -> MAIN.tex rebuilt from the parts, in order

The parts are consecutive slices of the file: the preamble up to the abstract, the abstract environment, one
part per \\section, and the tail from the bibliography or \\end{document} on. Joining them gives the file back
byte for byte, so the rewrite keeps the paper's own layout. Only parts marked "edit" in parts.json go to a
subagent; the preamble, the text between the abstract and the first section, and the tail stay as they are.
"""
import json
import re
import sys
from pathlib import Path


def cuts(s: str) -> list[tuple[int, str, bool]]:
    out = [(0, "preamble", False)]
    a = re.search(r"\\begin\{abstract\}", s)
    if a:
        out.append((a.start(), "abstract", True))
        out.append((s.index("\\end{abstract}", a.end()) + len("\\end{abstract}"), "front", False))
    secs = [m for m in re.finditer(r"^[ \t]*\\section\*?\{([^}]*)\}", s, re.M) if not a or m.start() > a.start()]
    for m in secs:
        out.append((m.start(), re.sub(r"[^a-z0-9]+", "-", m.group(1).lower()).strip("-")[:30] or "section", True))
    start = secs[-1].end() if secs else out[-1][0]
    t = re.compile(r"\\bibliography\{|\\begin\{thebibliography\}|\\printbibliography|\\end\{document\}").search(s, start)
    if t:
        out.append((t.start(), "tail", False))
    return out


def split(main: Path, d: Path) -> None:
    s = main.read_text(encoding="utf-8")
    c = cuts(s)
    d.mkdir(parents=True, exist_ok=True)
    parts = []
    for i, (pos, name, edit) in enumerate(c):
        end = c[i + 1][0] if i + 1 < len(c) else len(s)
        f = f"{i:02d}-{name}.tex"
        (d / f).write_text(s[pos:end], encoding="utf-8")
        parts.append({"file": f, "edit": edit})
    (d / "parts.json").write_text(json.dumps(parts, indent=1))
    assert join_text(d) == s
    for p in parts:
        print(("edit  " if p["edit"] else "keep  ") + p["file"])


def join_text(d: Path) -> str:
    return "".join((d / p["file"]).read_text(encoding="utf-8") for p in json.loads((d / "parts.json").read_text()))


def main() -> int:
    if len(sys.argv) != 4 or sys.argv[1] not in ("split", "join"):
        print(__doc__)
        return 2
    if sys.argv[1] == "split":
        split(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        Path(sys.argv[3]).write_text(join_text(Path(sys.argv[2])), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
