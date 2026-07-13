# /// script
# requires-python = ">=3.9"
# dependencies = ["pyyaml"]
# ///
"""PASTE-READY.md → values.yaml (Stage D bridge) — the drafted content, keyed by field id.

The paste-ready block file is the CANONICAL drafted content (one block per field, in form order).
Renderers (`render_docx.py`, `render_md.py`) consume a flat `values.yaml` ({field-id: body}). This
script bridges the two: it parses the block grammar and emits values.yaml, so the paste-ready stays
the single source of truth and the renderers never drift from it (SKILL.md Output-convention #5).

Block grammar (must match PASTE-READY.template.txt / charcount.py):
    === <field-id> | <LABEL> | limit: <N|null> <unit> ===
    <body, may span lines>
    === /<field-id> ===

Non-narrative field values (a budget total, an in-kind field, a scalar the paste-ready doesn't carry)
are NOT in the block file — supply them with `--extra key=value` (repeatable) or merge afterwards.
Lines outside any block (headers, `#` comments) are ignored.

    uv run extract_values.py PASTE-READY.md -o values.yaml [--extra requested_total="AUD 285,000"]
    uv run extract_values.py --self-test
"""
import argparse
import re
import sys

HEADER = re.compile(r"^===\s+(\S+)\s+\|.*===\s*$")
TERM = re.compile(r"^===\s+/(\S+)\s+===\s*$")


def parse(text):
    """→ {field-id: body}. A block's body is everything strictly between its header and terminator."""
    vals, cur, buf = {}, None, []
    for line in text.splitlines():
        mt = TERM.match(line)
        mh = HEADER.match(line)
        if mt and cur == mt.group(1):
            vals[cur] = "\n".join(buf).strip()
            cur, buf = None, []
        elif mh and not mt:
            cur, buf = mh.group(1), []
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        raise ValueError(f"unterminated block: {cur!r} (missing === /{cur} ===)")
    return vals


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paste_ready", nargs="?", help="PASTE-READY.md")
    ap.add_argument("-o", "--out", help="values.yaml path (else prints field ids)")
    ap.add_argument("--extra", action="append", default=[], help="key=value for a non-block field (repeatable)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.paste_ready:
        ap.error("need PASTE-READY.md (or --self-test)")
    try:
        vals = parse(open(args.paste_ready, encoding="utf-8").read())
    except ValueError as exc:
        print(f"PARSE ERROR (fail-closed): {exc}", file=sys.stderr)
        return 2
    for kv in args.extra:
        if "=" not in kv:
            ap.error(f"--extra needs key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        vals[k.strip()] = v.strip()
    print(f"extracted {len(vals)} field(s): {', '.join(vals)}")
    if args.out:
        import yaml
        yaml.safe_dump(vals, open(args.out, "w", encoding="utf-8"),
                       sort_keys=False, allow_unicode=True, default_flow_style=False, width=100)
        print(f"wrote {args.out}")
    return 0


def self_test():
    txt = ("╔ header box ignored ╗\n"
           "=== f1 | Label One | limit: 100 words ===\n"
           "para one line one\npara one line two\n"
           "=== /f1 ===\n"
           "# a comment between blocks, ignored\n"
           "=== f2 | Label Two | limit: null words ===\n"
           "body two\n"
           "=== /f2 ===\n")
    v = parse(txt)
    assert set(v) == {"f1", "f2"}, v
    assert v["f1"] == "para one line one\npara one line two", repr(v["f1"])
    assert v["f2"] == "body two", v
    # unterminated block → error
    try:
        parse("=== f | L | limit: 1 words ===\nbody\n")
    except ValueError:
        pass
    else:
        raise AssertionError("unterminated block must raise")
    # terminator for a different id does not close the block
    v2 = parse("=== a | L | limit: 1 words ===\nx\n=== /b ===\n=== /a ===\n")
    assert v2["a"] == "x\n=== /b ===", repr(v2["a"])
    print("self-test OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
