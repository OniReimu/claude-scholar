# /// script
# requires-python = ">=3.9"
# dependencies = ["pyyaml"]
# ///
"""字段字数/字符数核对 (Stage E: char-fit pass) — FAIL-CLOSED.

给定一份 PASTE-READY 文本 (C1 canonical grammar) 或字段 YAML，逐字段报告
count vs limit，标出违规字段，支持 chars / words / pages 三种单位。

FAIL-CLOSED 规则 (C2):
  * limit == null / 缺失 / UNVERIFIED  → status BLOCK, 非零退出 (除非 --allow-unverified)。
    "对着没有 limit 的字段判绿" 是被禁止的。
  * unit == pages                       → status NEEDS-RENDER, 非零退出 (不可测量 ≠ 通过)，
    除非用 --pages <field=count> 显式提供渲染后的页数。
  * nested_sublimits[]                  → 逐子块对各自 limit 校验；任一缺失/超限即 FAIL。
  * min                                 → 强制最小长度；不足即 FAIL。
  * over-limit                          → 非零退出 (原有行为)。

C1 PASTE-READY grammar (一字段一块，块头 + 显式终止符，末块不会吞掉后续文本):

    === <field-id> | <LABEL> | limit: <N|null> <unit> ===
    <body text, may span multiple lines>
    === /<field-id> ===

  <unit> ∈ chars | words | pages。limit 可以是字面量 null (UNVERIFIED)。
  正文严格取块头行与匹配的 `=== /<field-id> ===` 终止符之间的内容。
  缺失终止符 = 硬错误 (末块吞并保护)。

YAML 格式 (可表达 min / nested_sublimits):
    fields:
      synopsis:   {text: "...", limit: {value: 2000, unit: chars}}
      career_best:{text: "...", limit: "150 words"}          # 简写: "<n> <unit>"
      bio:        {text: "...", limit: {value: 500, unit: chars, min: 100}}
      ft_b10:                                                 # nested 子限额
        text: "..."
        limit: {value: null}
        nested_sublimits:
          - {id: past, text: "...", limit: {value: 200, unit: words}}
          - {id: plan, text: "...", limit: {value: 300, unit: words}}

用法:
    uv run charcount.py --text ECR-PASTE-READY.txt
    uv run charcount.py --yaml fields.yaml
    uv run charcount.py --text form.txt --pages B1_business_case=4 --allow-unverified
    uv run charcount.py --self-test

退出码: 任一字段 BLOCK / NEEDS-RENDER / OVER / UNDER-MIN / nested-fail → 1，否则 0。
"""
import argparse
import re
import sys

# C1 canonical grammar
HEADER_RE = re.compile(
    r"^===\s*(?P<id>[^|]+?)\s*\|\s*(?P<label>.*?)\s*\|\s*"
    r"limit:\s*(?P<limit>null|\d+)\s+(?P<unit>chars|words|pages)\s*===\s*$",
    re.I,
)
TERM_RE = re.compile(r"^===\s*/\s*(?P<id>.+?)\s*===\s*$")

_CHAR_UNITS = ("char", "chars", "character", "characters")
_WORD_UNITS = ("word", "words")
_PAGE_UNITS = ("page", "pages", "pp")


class ParseError(ValueError):
    """PASTE-READY 结构错误 (如缺失终止符)。"""


def render_block(fid, label, limit, unit, body):
    """C1 producer: 把一个字段渲染成 canonical PASTE-READY 块 (含终止符)。"""
    lim = "null" if limit is None else str(int(limit))
    return f"=== {fid} | {label} | limit: {lim} {unit} ===\n{body}\n=== /{fid} ==="


def normalize_limit(raw):
    """把 None / int / '2000 chars' / {value,unit,min} 归一成 (value|None, unit, min|None)。"""
    if raw is None:
        return None, "chars", None
    if isinstance(raw, dict):
        val = raw.get("value")
        val = None if val is None else int(val)
        unit = str(raw.get("unit", "chars")).lower()
        minv = raw.get("min")
        minv = None if minv is None else int(minv)
        return val, unit, minv
    if isinstance(raw, int):
        return raw, "chars", None
    s = str(raw).strip()
    if s.lower() in ("null", "unverified", ""):
        return None, "chars", None
    m = re.match(r"^\s*(\d+)\s*([a-zA-Z]+)?\s*$", s)
    if not m:
        raise ValueError(f"无法解析 limit: {raw!r}")
    unit = (m.group(2) or "chars").lower()
    return int(m.group(1)), unit, None


def measure(text, unit):
    """返回 (count, measurable)。pages 不可从纯文本测量。"""
    body = text.strip()
    if unit in _CHAR_UNITS:
        return len(body), True
    if unit in _WORD_UNITS:
        return len(body.split()), True
    if unit in _PAGE_UNITS:
        return None, False
    raise ValueError(f"未知单位: {unit}")


def parse_text(raw):
    """C1 PASTE-READY 文本 → [field dict]。缺失/错配终止符 → ParseError。"""
    fields = []
    lines = raw.splitlines()
    i, n = 0, len(lines)
    while i < n:
        m = HEADER_RE.match(lines[i])
        if not m:
            i += 1
            continue
        fid = m.group("id").strip()
        label = m.group("label").strip()
        limit_raw = m.group("limit")
        unit = m.group("unit").lower()
        body, i, term = [], i + 1, None
        while i < n:
            t = TERM_RE.match(lines[i])
            if t:
                term = t.group("id").strip()
                break
            if HEADER_RE.match(lines[i]):
                raise ParseError(
                    f"字段 {fid!r} 缺少终止符 '=== /{fid} ==='（在其终止前出现了下一个块头）"
                )
            body.append(lines[i])
            i += 1
        if term is None:
            raise ParseError(f"字段 {fid!r} 缺少终止符 '=== /{fid} ==='（到文件结尾未找到）")
        if term != fid:
            raise ParseError(f"终止符不匹配：期望 '=== /{fid} ===' 却得到 '=== /{term} ==='")
        limit = None if limit_raw.lower() == "null" else int(limit_raw)
        fields.append({"id": fid, "label": label, "text": "\n".join(body),
                       "limit": limit, "unit": unit, "min": None, "nested": None})
        i += 1
    return fields


def parse_yaml(raw):
    import yaml
    data = yaml.safe_load(raw) or {}
    out = []
    for fid, spec in (data.get("fields") or {}).items():
        spec = spec or {}
        value, unit, minv = normalize_limit(spec.get("limit"))
        nested = None
        if spec.get("nested_sublimits"):
            nested = []
            for sub in spec["nested_sublimits"]:
                sv, su, sm = normalize_limit(sub.get("limit"))
                nested.append({"id": sub.get("id", "sub"), "text": sub.get("text", ""),
                               "limit": sv, "unit": su, "min": sm})
        out.append({"id": fid, "label": spec.get("label", fid),
                    "text": spec.get("text", ""), "limit": value, "unit": unit,
                    "min": minv, "nested": nested})
    return out


def _judge(fid, text, value, unit, minv, allow_unverified, pages_map):
    """单个 (子)字段判定 → (count, value, unit, status, ok)。"""
    if unit in _PAGE_UNITS:
        if fid in pages_map:
            count = pages_map[fid]
        else:
            return None, value, unit, "NEEDS-RENDER (pages unmeasurable; supply --pages)", False
    else:
        count, _ = measure(text, unit)
    if value is None:
        if allow_unverified:
            return count, None, unit, "UNVERIFIED (allowed)", True
        return count, None, unit, "BLOCK (no verified limit)", False
    if minv is not None and count < minv:
        return count, value, unit, f"UNDER-MIN (need >={minv}, have {count})", False
    if count > value:
        return count, value, unit, f"OVER +{count - value}", False
    left = value - count
    return count, value, unit, f"ok ({left} left)", True


def check(fields, allow_unverified=False, pages_map=None):
    """返回 (report_rows, any_fail)。row = (fid, count, value, unit, status, ok)。"""
    pages_map = pages_map or {}
    rows, any_fail = [], False
    for f in fields:
        fid, unit = f["id"], f["unit"]
        nested = f.get("nested")
        if nested:
            for sub in nested:
                c, v, u, st, ok = _judge(f"{fid}/{sub['id']}", sub["text"], sub["limit"],
                                         sub["unit"], sub.get("min"), allow_unverified, pages_map)
                any_fail = any_fail or not ok
                rows.append((f"{fid}/{sub['id']}", c, v, u, st, ok))
            # 父字段仅在自带 limit 时校验整段文本
            if f["limit"] is not None:
                c, v, u, st, ok = _judge(fid, f["text"], f["limit"], unit, f.get("min"),
                                         allow_unverified, pages_map)
                any_fail = any_fail or not ok
                rows.append((fid, c, v, u, st, ok))
            continue
        c, v, u, st, ok = _judge(fid, f["text"], f["limit"], unit, f.get("min"),
                                 allow_unverified, pages_map)
        any_fail = any_fail or not ok
        rows.append((fid, c, v, u, st, ok))
    return rows, any_fail


def render(rows):
    w = max((len(r[0]) for r in rows), default=5)
    print(f"{'field'.ljust(w)}  {'count':>7}  {'limit':>7}  unit    status")
    print("-" * (w + 40))
    for fid, count, value, unit, status, _ok in rows:
        c = "-" if count is None else str(count)
        l = "-" if value is None else str(value)
        print(f"{fid.ljust(w)}  {c:>7}  {l:>7}  {unit:<6}  {status}")


def self_test():
    # ── normalize / measure ──
    assert normalize_limit("150 words") == (150, "words", None)
    assert normalize_limit(2000) == (2000, "chars", None)
    assert normalize_limit({"value": 5, "unit": "pages"}) == (5, "pages", None)
    assert normalize_limit({"value": 500, "unit": "chars", "min": 100}) == (500, "chars", 100)
    assert normalize_limit({"value": None}) == (None, "chars", None)
    assert measure("a b c", "words") == (3, True)
    assert measure("hello", "chars") == (5, True)
    assert measure("x", "pages") == (None, False)

    # ── C1 round-trip: render 2 blocks → parse → counts match, terminator respected ──
    b1 = render_block("f1", "First", 3, "chars", "abcd")          # 4 > 3 → OVER
    b2 = render_block("f2", "Second", 2, "words", "one two")      # 2 words ok
    doc = b1 + "\n" + b2 + "\n"
    parsed = parse_text(doc)
    assert [p["id"] for p in parsed] == ["f1", "f2"], parsed
    assert parsed[0]["text"] == "abcd", parsed[0]
    assert parsed[1]["text"] == "one two", parsed[1]
    rows, any_fail = check(parsed)
    assert any_fail is True, rows
    d = {r[0]: r for r in rows}
    assert d["f1"][3] == "chars" and d["f1"][4].startswith("OVER"), d["f1"]
    assert d["f2"][4].startswith("ok"), d["f2"]

    # trailing text after last terminator must NOT be absorbed into f2
    assert d["f2"][1] == 2, "terminator must stop the body count"

    # ── missing terminator = hard ParseError ──
    try:
        parse_text("=== x | X | limit: 10 chars ===\nbody with no terminator\n")
    except ParseError:
        pass
    else:
        raise AssertionError("missing terminator must raise ParseError")

    # ── C2: null limit → BLOCK + fail ──
    nul = [{"id": "a", "label": "A", "text": "hi", "limit": None, "unit": "chars",
            "min": None, "nested": None}]
    rows, fail = check(nul)
    assert fail is True and rows[0][4].startswith("BLOCK"), rows
    rows, fail = check(nul, allow_unverified=True)
    assert fail is False and rows[0][4].startswith("UNVERIFIED"), rows

    # ── C2: pages → NEEDS-RENDER + fail; --pages override → measurable ──
    pg = [{"id": "bc", "label": "BC", "text": "x" * 10, "limit": 5, "unit": "pages",
           "min": None, "nested": None}]
    rows, fail = check(pg)
    assert fail is True and rows[0][4].startswith("NEEDS-RENDER"), rows
    rows, fail = check(pg, pages_map={"bc": 4})
    assert fail is False and rows[0][4].startswith("ok"), rows
    rows, fail = check(pg, pages_map={"bc": 6})       # 6 > 5 pages → OVER
    assert fail is True and rows[0][4].startswith("OVER"), rows

    # ── C2: min → under-min fails ──
    mn = [{"id": "bio", "label": "Bio", "text": "short", "limit": 500, "unit": "chars",
           "min": 100, "nested": None}]
    rows, fail = check(mn)
    assert fail is True and rows[0][4].startswith("UNDER-MIN"), rows

    # ── C2: nested_sublimits → each sub validated; one over → fail ──
    nested = [{"id": "b10", "label": "B10", "text": "", "limit": None, "unit": "chars", "min": None,
               "nested": [{"id": "past", "text": "one two three", "limit": 2, "unit": "words", "min": None},
                          {"id": "plan", "text": "a b", "limit": 5, "unit": "words", "min": None}]}]
    rows, fail = check(nested)
    dn = {r[0]: r for r in rows}
    assert fail is True, rows
    assert dn["b10/past"][4].startswith("OVER"), dn["b10/past"]   # 3 > 2
    assert dn["b10/plan"][4].startswith("ok"), dn["b10/plan"]     # 2 <= 5

    # ── YAML path parses min + nested ──
    yfields = parse_yaml(
        "fields:\n"
        "  s: {text: 'ab', limit: {value: 3, unit: chars}}\n"
        "  n:\n"
        "    text: ''\n"
        "    limit: {value: null}\n"
        "    nested_sublimits:\n"
        "      - {id: p, text: 'w w w', limit: {value: 5, unit: words}}\n"
    )
    assert {f["id"] for f in yfields} == {"s", "n"}, yfields
    rows, fail = check(yfields)
    assert fail is False, rows

    print("self-test OK")
    return 0


def _parse_pages(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"error: --pages 需要 field=count, 得到 {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = int(v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", help="PASTE-READY.txt 路径 (C1 grammar)")
    g.add_argument("--yaml", help="字段 YAML 路径")
    g.add_argument("--self-test", action="store_true", help="运行内建自检")
    ap.add_argument("--allow-unverified", action="store_true",
                    help="放行 limit==null 字段 (默认 fail-closed 拦截)")
    ap.add_argument("--pages", action="append", metavar="FIELD=COUNT",
                    help="为 pages 单位字段提供渲染后的页数 (可重复)")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    pages_map = _parse_pages(args.pages)
    raw = open(args.text or args.yaml, encoding="utf-8").read()
    try:
        fields = parse_text(raw) if args.text else parse_yaml(raw)
    except ParseError as exc:
        print(f"PARSE ERROR: {exc}", file=sys.stderr)
        return 2
    if not fields:
        print("未解析到任何字段", file=sys.stderr)
        return 2
    rows, any_fail = check(fields, allow_unverified=args.allow_unverified, pages_map=pages_map)
    render(rows)
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
