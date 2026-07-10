# /// script
# requires-python = ">=3.9"
# dependencies = ["pyyaml"]
# ///
"""字段字数/字符数核对 (Stage E: char-fit pass)

给定一份 PASTE-READY 文本或字段→(text, limit) 的 YAML，逐字段报告
count vs limit，标出超限字段，支持 chars / words / pages 三种单位。
pages 无法从纯文本测量，只回显 limit 并标记 needs-render。

用法:
    uv run charcount.py --text ECR-PASTE-READY.txt
    uv run charcount.py --yaml fields.yaml
    uv run charcount.py --self-test

PASTE-READY 文本格式 (一字段一块，块头声明 limit):
    === synopsis [2000 chars] ===
    正文……
    === career_best [150 words] ===
    正文……

YAML 格式:
    fields:
      synopsis:   {text: "...", limit: {value: 2000, unit: chars}}
      career_best:{text: "...", limit: "150 words"}   # 简写: "<n> <unit>"

退出码: 任一字段超限 → 1，否则 0 (可直接接 CI / lint)。
"""
import argparse
import re
import sys

HEADER_RE = re.compile(r"^===\s*(?P<id>.+?)\s*\[\s*(?P<limit>[^\]]+?)\s*\]\s*===\s*$")


def normalize_limit(raw):
    """把 int / '2000 chars' / {value,unit} 归一成 (value:int, unit:str)。"""
    if raw is None:
        return None, "chars"
    if isinstance(raw, dict):
        return int(raw["value"]), str(raw.get("unit", "chars")).lower()
    if isinstance(raw, int):
        return raw, "chars"
    m = re.match(r"^\s*(\d+)\s*([a-zA-Z]+)?\s*$", str(raw))
    if not m:
        raise ValueError(f"无法解析 limit: {raw!r}")
    unit = (m.group(2) or "chars").lower()
    return int(m.group(1)), unit


def measure(text, unit):
    """返回 (count, measurable)。pages 不可从文本测量。"""
    body = text.strip()
    if unit in ("char", "chars", "character", "characters"):
        return len(body), True
    if unit in ("word", "words"):
        return len(body.split()), True
    if unit in ("page", "pages"):
        return None, False
    raise ValueError(f"未知单位: {unit}")


def parse_text(raw):
    """PASTE-READY 文本 → [(id, text, limit_raw)]。"""
    fields, cur_id, cur_limit, buf = [], None, None, []
    for line in raw.splitlines():
        m = HEADER_RE.match(line)
        if m:
            if cur_id is not None:
                fields.append((cur_id, "\n".join(buf), cur_limit))
            cur_id, cur_limit, buf = m.group("id"), m.group("limit"), []
        elif cur_id is not None:
            buf.append(line)
    if cur_id is not None:
        fields.append((cur_id, "\n".join(buf), cur_limit))
    return fields


def parse_yaml(raw):
    import yaml
    data = yaml.safe_load(raw) or {}
    out = []
    for fid, spec in (data.get("fields") or {}).items():
        out.append((fid, spec.get("text", ""), spec.get("limit")))
    return out


def check(fields):
    """返回 (report_rows, any_over)。"""
    rows, any_over = [], False
    for fid, text, limit_raw in fields:
        value, unit = normalize_limit(limit_raw)
        count, measurable = measure(text, unit)
        if not measurable:
            rows.append((fid, count, value, unit, "NEEDS-RENDER"))
            continue
        if value is None:
            rows.append((fid, count, None, unit, "no-limit"))
            continue
        over = count > value
        any_over = any_over or over
        status = f"OVER +{count - value}" if over else f"ok ({value - count} left)"
        rows.append((fid, count, value, unit, status))
    return rows, any_over


def render(rows):
    w = max((len(r[0]) for r in rows), default=5)
    print(f"{'field'.ljust(w)}  {'count':>7}  {'limit':>7}  unit    status")
    print("-" * (w + 34))
    for fid, count, value, unit, status in rows:
        c = "-" if count is None else str(count)
        l = "-" if value is None else str(value)
        print(f"{fid.ljust(w)}  {c:>7}  {l:>7}  {unit:<6}  {status}")


def self_test():
    v, u = normalize_limit("150 words")
    assert (v, u) == (150, "words"), (v, u)
    assert normalize_limit(2000) == (2000, "chars")
    assert normalize_limit({"value": 5, "unit": "pages"}) == (5, "pages")
    assert measure("a b c", "words") == (3, True)
    assert measure("hello", "chars") == (5, True)
    assert measure("x", "pages") == (None, False)
    sample = "=== f1 [3 chars] ===\nabcd\n=== f2 [2 words] ===\none two\n"
    parsed = parse_text(sample)
    assert [p[0] for p in parsed] == ["f1", "f2"], parsed
    rows, any_over = check(parsed)
    assert any_over is True, rows          # f1 has 4>3
    assert rows[0][4].startswith("OVER"), rows[0]
    assert rows[1][4].startswith("ok"), rows[1]
    print("self-test OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", help="PASTE-READY.txt 路径")
    g.add_argument("--yaml", help="字段 YAML 路径")
    g.add_argument("--self-test", action="store_true", help="运行内建自检")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    raw = open(args.text or args.yaml, encoding="utf-8").read()
    fields = parse_text(raw) if args.text else parse_yaml(raw)
    if not fields:
        print("未解析到任何字段", file=sys.stderr)
        return 2
    rows, any_over = check(fields)
    render(rows)
    return 1 if any_over else 0


if __name__ == "__main__":
    sys.exit(main())
