# /// script
# requires-python = ">=3.9"
# dependencies = ["pyyaml"]
# ///
"""values.yaml (+ scheme + tables) → a clean Markdown MIRROR of the filled application (Stage D).

The `.docx` (render_docx.py) is the official-template deliverable; this is its readable twin — the
same filled content laid out as Markdown, in form order, with section headings, per-field limits, and
any budget/contribution table. Same source (values.yaml) as the docx, so the two never drift
(SKILL.md Output-convention #5). Deliverable is `.md`, per Output-convention #1.

    uv run render_md.py values.yaml --scheme scheme.yaml [--tables tables.yaml] -o application.md
    uv run render_md.py --self-test
"""
import argparse
import re
import sys


def load(path):
    import yaml
    return yaml.safe_load(open(path, encoding="utf-8")) or {}


def field_order(scheme):
    """→ ([field-id...], {id: (label, limit, unit)}, {id: section-title}) in form order."""
    meta, order, sec_of = {}, [], {}

    def walk(node, sect=None):
        if isinstance(node, dict):
            if node.get("title") and node.get("fields") is not None:
                sect = node["title"]
            if "id" in node and "widget" in node:
                fid, lim = node["id"], (node.get("limit") or {})
                meta[fid] = (node.get("label", fid), lim.get("value"), lim.get("unit"))
                sec_of[fid] = sect
                if fid not in order:
                    order.append(fid)
            for v in node.values():
                walk(v, sect)
        elif isinstance(node, list):
            for v in node:
                walk(v, sect)

    walk(scheme.get("sections"))
    return order, meta, sec_of


def unwrap(body):
    """Blank-line = paragraph; unwrap hard line breaks within a paragraph."""
    return "\n\n".join(re.sub(r"\s+", " ", b.replace("\n", " ")).strip()
                       for b in str(body).split("\n\n") if b.strip())


def render(scheme, values, tables, title="Application (DRAFT)"):
    order, meta, sec_of = field_order(scheme)
    L = [f"# {scheme.get('scheme', title)} — DRAFT", ""]
    L += ["> Markdown mirror of the filled official template. DRAFT for human cross-validation before "
          "the portal. Markers `[TO SET]` / `[VERIFY]` / `[DOMAIN-EXPERT TO VERIFY]` are unresolved.", ""]
    cur_sec, table_fields = None, set(tables or {})
    done_tables = set()
    for fid in order:
        if fid in table_fields and fid not in done_tables:
            spec = tables[fid]
            L.append(f"## {meta.get(fid, (fid,))[0]}")
            L.append("")
            for row in spec.get("rows") or []:
                cells = [str(c).replace("|", "/") for c in row]
                L.append("| " + " | ".join(cells) + " |")
                if len(L) and row is (spec.get("rows") or [None])[0]:
                    L.append("|" + "---|" * len(cells))
            L.append("")
            done_tables.add(fid)
            continue
        if fid not in values:
            continue
        label, lim, unit = meta.get(fid, (fid, None, None))
        sect = sec_of.get(fid)
        if sect and sect != cur_sec:
            L += [f"# {sect}", ""]
            cur_sec = sect
        L.append(f"## {label}" + (f"  _(limit: {lim} {unit})_" if lim else ""))
        L += ["", unwrap(values[fid]), ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("values", nargs="?", help="values.yaml")
    ap.add_argument("--scheme", required=False, help="scheme.yaml (field order + labels + limits)")
    ap.add_argument("--tables", help="tables.yaml {field: {rows: [[...]]}}")
    ap.add_argument("-o", "--out")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.values or not args.scheme:
        ap.error("need values.yaml + --scheme (or --self-test)")
    md = render(load(args.scheme), load(args.values), load(args.tables) if args.tables else {})
    if args.out:
        open(args.out, "w", encoding="utf-8").write(md)
        print(f"wrote {args.out} ({md.count(chr(10).join([''])+'## '):d}+ sections)")
    else:
        print(md)
    return 0


def self_test():
    scheme = {"scheme": "Test Scheme", "sections": [
        {"title": "Narrative", "fields": [
            {"id": "summary", "widget": "narrative", "label": "Project Summary", "limit": {"value": 100, "unit": "words"}},
            {"id": "budget", "widget": "budget-matrix", "label": "Budget"}]}]}
    values = {"summary": "line one\nline two"}
    tables = {"budget": {"rows": [["Category", "Amount"], ["Labour", "100"]]}}
    md = render(scheme, values, tables)
    assert "# Test Scheme — DRAFT" in md
    assert "## Project Summary  _(limit: 100 words)_" in md
    assert "line one line two" in md          # unwrapped
    assert "| Category | Amount |" in md and "| Labour | 100 |" in md
    assert "|---|---|" in md
    print("self-test OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
