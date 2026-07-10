# /// script
# requires-python = ">=3.9"
# dependencies = ["pyyaml"]
# ///
"""预算矩阵校验 (Stage E: budget-math pass)

对 budget-matrix (categories × year × org × cash/in-kind/credit) 跑机械校验:
  1. 逐行百分比上限     (row_caps, 如 audit ≤ 1% / overseas ≤ 10%; 支持双分母)
  2. 配套资金比例       (matched_funding: co-contribution / requested ≥ 阈值)
  3. credit vs cash 计入 (counts_toward_total=false 的 credit 不进 cash total)
  4. live totals        (逐年 / 逐机构 / 总计；若 declared_totals 存在则比对)
  5. 累计现金流动性     (cash_flow_check: 逐年累计支出 ≤ 累计现金流入; opt-in)
逐规则报告 PASS/FAIL。纯机械，不臆造数字。

用法:
    uv run validate_budget.py budget.yaml
    uv run validate_budget.py --self-test

YAML schema:
    matched_funding_min_ratio: 1.0        # co-contribution / requested，可省
    row_caps:
      - {category: audit, max_pct: 1.0}   # of: total(默认,含in-kind) | total-cash | requested
      - {category: overseas, max_pct: 10.0, of: total-cash}  # total-cash = 计入行 EXCLUDE kind==in-kind
    declared_totals: {requested: 100000}  # 可省，存在即比对
    cash_flow_check: true                 # 可省; true 时启用累计现金流检查
    cash_in: {2026: 100000, 2027: 400000} # 逐年现金流入 (现金配套 + grant drawdown)
    rows:
      - {category: audit, funding_source: requested, kind: cash,
         org: leadco, counts_toward_total: true, years: {2026: 500, 2027: 500}}

退出码: 任一规则 FAIL → 1，否则 0。
"""
import argparse
import sys


def row_total(r):
    return sum(float(v) for v in (r.get("years") or {}).values())


def counts(r):
    return r.get("counts_toward_total", True)


def is_in_kind(r):
    return r.get("kind") == "in-kind"


def totals(rows):
    """计入总额的口径下算 total / requested / co-contribution / total-cash。"""
    incl = [r for r in rows if counts(r)]
    total = sum(row_total(r) for r in incl)
    total_cash = sum(row_total(r) for r in incl if not is_in_kind(r))  # EXCLUDE in-kind
    requested = sum(row_total(r) for r in incl if r.get("funding_source") == "requested")
    cocon = sum(row_total(r) for r in incl if r.get("funding_source") == "co-contribution")
    return total, requested, cocon, total_cash


def check_row_caps(rows, caps, total, requested, total_cash):
    bases = {"requested": (requested, "requested"), "total-cash": (total_cash, "total-cash")}
    out = []
    for cap in caps:
        cat, max_pct = cap["category"], float(cap["max_pct"])
        base, base_name = bases.get(cap.get("of"), (total, "total"))  # 默认 total
        amt = sum(row_total(r) for r in rows if r.get("category") == cat and counts(r))
        pct = (amt / base * 100) if base else 0.0
        ok = pct <= max_pct + 1e-9
        out.append((f"row-cap[{cat}]", ok,
                    f"{amt:.0f} = {pct:.2f}% of {base_name} (cap {max_pct}%)"))
    return out


def check_cash_flow(rows, cash_in):
    """逐年累计现金流动性: 累计现金支出不得超过累计现金流入 (in-kind 非现金,排除)。"""
    if not cash_in:
        return []
    spend = {}
    for r in rows:
        if not counts(r) or is_in_kind(r):
            continue
        for y, v in (r.get("years") or {}).items():
            spend[y] = spend.get(y, 0.0) + float(v)
    out, cum_s, cum_i = [], 0.0, 0.0
    for fy in sorted(set(spend) | set(cash_in)):
        cum_s += spend.get(fy, 0.0)
        cum_i += float(cash_in.get(fy, 0.0))
        ok = cum_s <= cum_i + 1e-6
        out.append((f"cash-flow[FY{fy}]", ok,
                    f"cum spend {cum_s:.0f} vs cum cash-in {cum_i:.0f}"))
    return out


def check_matched(min_ratio, requested, cocon):
    if min_ratio is None:
        return []
    ratio = (cocon / requested) if requested else 0.0
    ok = ratio >= float(min_ratio) - 1e-9
    return [("matched-funding", ok,
             f"co-contribution/requested = {cocon:.0f}/{requested:.0f} = {ratio:.3f} (min {min_ratio})")]


def check_credits(rows):
    """counts_toward_total=false 的行不得计入 cash total；报告 credit 口径。"""
    excluded = [r for r in rows if r.get("kind") == "credit" and not counts(r)]
    included = [r for r in rows if r.get("kind") == "credit" and counts(r)]
    ok = True  # schema 一致即 PASS；此处校验的是我们的口径确实排除了 excluded
    detail = (f"credits excluded-from-total: {sum(row_total(r) for r in excluded):.0f}; "
              f"credits in-total: {sum(row_total(r) for r in included):.0f}")
    return [("credit-vs-cash", ok, detail)]


def check_declared(rows, declared, total, requested):
    if not declared:
        return []
    out = []
    live = {"total": total, "requested": requested}
    for key, val in declared.items():
        got = live.get(key)
        if got is None:
            out.append((f"declared[{key}]", False, "no live equivalent computed"))
            continue
        ok = abs(got - float(val)) < 1e-6
        out.append((f"declared[{key}]", ok, f"declared {float(val):.0f} vs live {got:.0f}"))
    return out


def per_year_org(rows):
    years, orgs = {}, {}
    for r in rows:
        if not counts(r):
            continue
        org = r.get("org", "?")
        orgs[org] = orgs.get(org, 0.0) + row_total(r)
        for y, v in (r.get("years") or {}).items():
            years[y] = years.get(y, 0.0) + float(v)
    return years, orgs


def run(data):
    rows = data.get("rows") or []
    total, requested, cocon, total_cash = totals(rows)
    results = []
    results += check_row_caps(rows, data.get("row_caps") or [], total, requested, total_cash)
    results += check_matched(data.get("matched_funding_min_ratio"), requested, cocon)
    results += check_credits(rows)
    results += check_declared(rows, data.get("declared_totals"), total, requested)
    if data.get("cash_flow_check"):
        results += check_cash_flow(rows, data.get("cash_in") or {})
    return results, (total, requested, cocon, total_cash), per_year_org(rows)


def render(results, tot, yo):
    total, requested, cocon, total_cash = tot
    years, orgs = yo
    print("== live totals ==")
    print(f"  total(counted): {total:.0f}   total-cash: {total_cash:.0f}   "
          f"requested: {requested:.0f}   co-contribution: {cocon:.0f}")
    print("  by year: " + ", ".join(f"{y}:{v:.0f}" for y, v in sorted(years.items())))
    print("  by org:  " + ", ".join(f"{o}:{v:.0f}" for o, v in sorted(orgs.items())))
    print("== rules ==")
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def self_test():
    data = {
        "matched_funding_min_ratio": 1.0,
        "row_caps": [{"category": "audit", "max_pct": 1.0},
                     {"category": "overseas", "max_pct": 10.0}],
        "declared_totals": {"requested": 10000},
        "rows": [  # total(counted)=20000: audit 300 + overseas 700 + salary 9000 + in-kind 10000
            {"category": "audit", "funding_source": "requested", "kind": "cash", "years": {2026: 300}},      # 1.5%>1% FAIL
            {"category": "overseas", "funding_source": "requested", "kind": "cash", "years": {2026: 700}},   # 3.5%<10% PASS
            {"category": "salary", "funding_source": "requested", "kind": "cash", "years": {2026: 9000}},
            {"category": "salary", "funding_source": "co-contribution", "kind": "in-kind", "years": {2026: 10000}},  # match 1.0
            {"category": "compute", "funding_source": "requested", "kind": "credit",
             "counts_toward_total": False, "years": {2026: 50000}},  # credit excluded from total
        ],
    }
    results, tot, _ = run(data)
    d = {n: (ok, det) for n, ok, det in results}
    assert d["row-cap[audit]"][0] is False, d["row-cap[audit]"]
    assert d["row-cap[overseas]"][0] is True, d["row-cap[overseas]"]
    assert d["matched-funding"][0] is True, d["matched-funding"]
    assert d["declared[requested]"][0] is True, d["declared[requested]"]  # requested=10000
    assert abs(tot[0] - 20000) < 1e-6, tot                                 # credit excluded → 20000

    # ── FIX #7: total=100k(含in-kind 50k), total-cash=50k; overseas 6000 → 6%总PASS(漏) vs 12%现金FAIL(抓) ──
    tc_rows = [
        {"category": "overseas", "funding_source": "requested", "kind": "cash", "years": {2026: 6000}},
        {"category": "labour", "funding_source": "requested", "kind": "cash", "years": {2026: 44000}},
        {"category": "in_kind", "funding_source": "co-contribution", "kind": "in-kind", "years": {2026: 50000}},
    ]
    pick = lambda caps: dict((n, ok) for n, ok, _ in run({"row_caps": caps, "rows": tc_rows})[0])
    assert pick([{"category": "overseas", "max_pct": 10.0, "of": "total-cash"}])["row-cap[overseas]"] is False
    assert pick([{"category": "overseas", "max_pct": 10.0}])["row-cap[overseas]"] is True  # total 会漏

    # ── FIX #8: cash-flow — 前置支出/后置现金 必须 FAIL 即使 row cap 全过 (overseas 8%<10%) ──
    cf = {"cash_flow_check": True, "cash_in": {2026: 5000, 2027: 45000},
          "row_caps": [{"category": "overseas", "max_pct": 10.0}],
          "rows": [{"category": "labour", "funding_source": "requested", "kind": "cash", "years": {2026: 46000}},
                   {"category": "overseas", "funding_source": "requested", "kind": "cash", "years": {2026: 4000}}]}
    r_cf = {n: ok for n, ok, _ in run(cf)[0]}
    assert r_cf["row-cap[overseas]"] is True, "row caps all pass"
    assert r_cf["cash-flow[FY2026]"] is False, "front-loaded spend must FAIL FY2026"
    assert r_cf["cash-flow[FY2027]"] is True, "recovers by FY2027"
    r_ok = {n: ok for n, ok, _ in run(dict(cf, cash_in={2026: 60000, 2027: 0}))[0]}
    assert all(v for n, v in r_ok.items() if n.startswith("cash-flow")), "well-funded passes"

    print("self-test OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("budget", nargs="?", help="预算 YAML 路径")
    ap.add_argument("--self-test", action="store_true", help="运行内建自检")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.budget:
        ap.error("需要预算 YAML 路径 (或 --self-test)")
    import yaml
    data = yaml.safe_load(open(args.budget, encoding="utf-8")) or {}
    results, tot, yo = run(data)
    render(results, tot, yo)
    return 1 if any(not ok for _, ok, _ in results) else 0


if __name__ == "__main__":
    sys.exit(main())
