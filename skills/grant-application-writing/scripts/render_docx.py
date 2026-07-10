# /// script
# requires-python = ">=3.10"
# dependencies = ["python-docx", "pyyaml"]
# ///
"""Stage-D docx renderer: write a filled IR back into the OFFICIAL .docx template.

Fills values into the *same* template so the funder's formatting, headers, and mandated
structure survive. Two write strategies, tried in order per field:
  1. content-control  — match a Word content control (SDT) by tag/alias, replace its text.
  2. under-heading    — match a mandated heading, insert the value as a paragraph beneath it.
A field whose target cannot be located is REPORTED (never silently dropped) so the caller
can fix the mapping rather than ship a form with a missing box.

    uv run render_docx.py filled.yaml template.docx -o application.docx

IR format (any of):
    field_id: "text"                       # flat mapping
    {fields: {field_id: "text", ...}}      # nested
    [{id: field_id, text: "..."}]          # list of objects
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def load_ir(path: Path) -> dict[str, str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "fields" in raw and isinstance(raw["fields"], dict):
        raw = raw["fields"]
    out: dict[str, str] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            out[str(k)] = "" if v is None else str(v)
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and "id" in item:
                out[str(item["id"])] = str(item.get("text", item.get("value", "")))
    else:
        sys.exit("error: IR must be a mapping or a list of {id,text} objects")
    return out


def fill_content_controls(doc, ir: dict[str, str], filled: set[str]) -> None:
    """Replace text inside SDTs whose tag/alias matches an IR key. python-docx has no
    SDT API, so operate on raw XML: clear existing w:t runs, write the value into the first."""
    from docx.oxml.ns import qn

    keys = {norm(k): k for k in ir}
    for sdt in list(doc.element.body.iter(qn("w:sdt"))):
        sdtpr = sdt.find(qn("w:sdtPr"))
        if sdtpr is None:
            continue
        ident = None
        for tagname in ("w:tag", "w:alias"):
            el = sdtpr.find(qn(tagname))
            if el is not None and el.get(qn("w:val")):
                ident = el.get(qn("w:val"))
                break
        key = keys.get(norm(ident)) if ident else None
        if not key or key in filled:
            continue
        content = sdt.find(qn("w:sdtContent"))
        if content is None:
            continue
        texts = list(content.iter(qn("w:t")))
        if not texts:
            continue
        texts[0].text = ir[key]
        texts[0].set(qn("xml:space"), "preserve")
        for extra in texts[1:]:
            extra.text = ""
        filled.add(key)


def fill_under_headings(doc, ir: dict[str, str], filled: set[str], report: list) -> None:
    """For each unfilled IR key, find a heading whose text matches and insert the value
    as a new paragraph right after it, inheriting the body ('Normal') style."""
    from docx.text.paragraph import Paragraph

    remaining = {norm(k): k for k in ir if k not in filled}
    if not remaining:
        return
    paras = doc.paragraphs
    for idx, para in enumerate(paras):
        style = (para.style.name or "") if para.style else ""
        if not style.startswith("Heading"):
            continue
        key = remaining.get(norm(para.text))
        if not key:
            continue
        new_p = para.insert_paragraph_before(ir[key])  # placeholder, then move after heading
        # insert_paragraph_before puts it above; move the new element to sit after `para`
        para._p.addnext(new_p._p)
        try:
            new_p.style = doc.styles["Normal"]
        except KeyError:
            pass
        filled.add(key)
        del remaining[norm(para.text)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ir", type=Path, help="filled IR YAML (field-id -> text)")
    ap.add_argument("template", type=Path, help="official .docx template to write into")
    ap.add_argument("-o", "--out", type=Path, required=True, help="output .docx path")
    args = ap.parse_args()

    for p in (args.ir, args.template):
        if not p.exists():
            sys.exit(f"error: file not found: {p}")
    if args.template.suffix.lower() != ".docx":
        sys.exit("error: template must be a .docx")

    try:
        from docx import Document
    except ImportError:
        sys.exit("error: python-docx not available. Run via `uv run` so PEP 723 deps install.")

    ir = load_ir(args.ir)
    if not ir:
        sys.exit("error: IR contained no fields")

    doc = Document(str(args.template))
    filled: set[str] = set()
    report: list = []

    fill_content_controls(doc, ir, filled)
    n_cc = len(filled)
    fill_under_headings(doc, ir, filled, report)
    n_head = len(filled) - n_cc

    unresolved = [k for k in ir if k not in filled]
    doc.save(str(args.out))

    print(f"wrote {args.out}")
    print(f"  filled: {len(filled)}/{len(ir)}  (content-control: {n_cc}, under-heading: {n_head})")
    if unresolved:
        print(f"  UNRESOLVED ({len(unresolved)}) — no content-control tag or matching heading found:")
        for k in unresolved:
            print(f"    - {k}")
        print("  These were NOT written. Fix the field-id to match a control tag/heading, or place them by hand.")
        sys.exit(2)


if __name__ == "__main__":
    main()
