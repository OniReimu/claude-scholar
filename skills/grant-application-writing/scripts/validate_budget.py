# /// script
# requires-python = ">=3.9"
# dependencies = ["pyyaml"]
# ///
"""预算矩阵校验 (Stage E: budget-math pass) — FAIL-CLOSED (C3).

对 budget-matrix (categories × year/phase × org × cash/in-kind/credit) 跑机械校验:
  1. 逐行百分比上限 (row_caps)   —— `of` 分母 REQUIRED ∈ {total, total-cash, requested}
  2. 配套资金比例 (matched_funding: co-contribution / requested ≥ 阈值)
  3. credit vs cash 计入        —— total-cash EXCLUDES kind∈{credit,in-kind}；credit 必须显式声明
  4. live totals                —— 逐年/逐机构/总计；declared_totals 存在即比对
  5. 累计现金流动性 (cash_flow_check) —— 缺 cash_in 即 FAIL，非静默 no-op
  6. 分期预算 (phased_if_min)    —— requested ≥ 阈值 时要求 ≥2 个有成本的 phase
  7. 双录平衡 (balance_check)    —— enabled 时 income==expenditure（含 in-kind 双边），|delta| ≤ tolerance

FAIL-CLOSED 硬错误 (BudgetError → 非零退出，非 rule-FAIL):
  * row-cap 缺 `of` 或 `of` 非法          → error
  * 负值 / 非有限值 (除 kind: refund 行)  → error

FAIL-CLOSED 规则 (rule FAIL → 非零退出):
  * 零分母 + 非零分子                      → FAIL invalid-denominator (两者皆零 → N/A)
  * cash_flow_check:true 而某相关 FY 缺 cash_in → FAIL
  * credit 行缺显式 counts_toward_total    → FAIL
  * requested ≥ phased_if_min 但 <2 个有成本 phase → FAIL
  * balance_check.enabled 而某计入行缺 side       → FAIL (无法核平衡，绝不默认 side)
  * balance_check.enabled 而 |income-expenditure| > tolerance → FAIL

用法:
    uv run validate_budget.py budget.yaml
    uv run validate_budget.py --self-test

YAML schema:
    matched_funding_min_ratio: 1.0
    phased_if_min: 200000                 # 可省; requested ≥ 此值 → 需 ≥2 个 phase
    row_caps:
      - {category: audit,    max_pct: 1.0,  of: total}         # of REQUIRED
      - {category: overseas, max_pct: 10.0, of: total-cash}    # total-cash → 分子默认排除 in-kind/credit
      - {category: travel,   max_pct: 10.0, of: total, kind: cash, funding_source: requested, org: leadco}  # 可选分子过滤
    declared_totals: {requested: 100000}
    cash_flow_check: true
    cash_in: {2026: 100000, 2027: 400000}
    balance_check: {enabled: true, tolerance: 0}   # 可省; enabled 时 income==expenditure，行需带 side
    rows:
      - {category: audit, funding_source: requested, kind: cash, org: leadco,
         counts_toward_total: true, phase: p1, years: {2026: 500, 2027: 500}}
      # 双录时行带 side: income | expenditure（in-kind 双边各一行）

退出码: 任一规则 FAIL 或硬错误 → 1，否则 0。
"""
import argparse
import math
import sys

_OF_BASES = ("total", "total-cash", "requested")
_EPS = 1e-9


class BudgetError(ValueError):
    """输入结构性错误 (缺 of / 非法数值)，fail-closed 硬错误。"""


def row_total(r):
    return sum(float(v) for v in (r.get("years") or {}).values())


def counts(r):
    return r.get("counts_toward_total", True)


def is_in_kind(r):
    return r.get("kind") == "in-kind"


def is_credit(r):
    return r.get("kind") == "credit"


def validate_values(rows):
    """负值 / 非有限值 → BudgetError (kind: refund 行允许负值)。"""
    for r in rows:
        refund = r.get("kind") == "refund"
        for y, v in (r.get("years") or {}).items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                raise BudgetError(f"非数值预算项 {r.get('category')!r} FY{y}: {v!r}")
            if not math.isfinite(fv):
                raise BudgetError(f"非有限预算值 {r.get('category')!r} FY{y}: {v!r}")
            if fv < 0 and not refund:
                raise BudgetError(
                    f"负数预算值 {r.get('category')!r} FY{y}: {v} (如为退款用 kind: refund 显式标注)")


def totals(rows):
    """计入总额口径下的 total / requested / co-contribution / total-cash。"""
    incl = [r for r in rows if counts(r)]
    total = sum(row_total(r) for r in incl)
    # total-cash EXCLUDES in-kind AND credit
    total_cash = sum(row_total(r) for r in incl if not is_in_kind(r) and not is_credit(r))
    requested = sum(row_total(r) for r in incl if r.get("funding_source") == "requested")
    cocon = sum(row_total(r) for r in incl if r.get("funding_source") == "co-contribution")
    return total, requested, cocon, total_cash


def cap_numerator(rows, cap, denom_name):
    """按 category + 可选 kind/funding_source/org 过滤求分子；total-cash 分母默认排除 in-kind/credit。"""
    cat = cap["category"]
    sel = [r for r in rows if r.get("category") == cat and counts(r)]
    for attr in ("kind", "funding_source", "org"):
        if attr in cap:
            sel = [r for r in sel if r.get(attr) == cap[attr]]
    if denom_name == "total-cash" and "kind" not in cap:
        sel = [r for r in sel if not is_in_kind(r) and not is_credit(r)]
    return sum(row_total(r) for r in sel)


def check_row_caps(rows, caps, total, requested, total_cash):
    bases = {"total": total, "total-cash": total_cash, "requested": requested}
    out = []
    for cap in caps:
        of = cap.get("of")
        if of not in _OF_BASES:
            raise BudgetError(
                f"row-cap[{cap.get('category')}] 需要 of ∈ {_OF_BASES}，得到 {of!r}（禁止静默默认 total）")
        cat, max_pct = cap["category"], float(cap["max_pct"])
        base = bases[of]
        amt = cap_numerator(rows, cap, of)
        if base == 0:
            if amt == 0:
                out.append((f"row-cap[{cat}]", True, f"N/A: 分子与分母 {of} 皆为 0"))
            else:
                out.append((f"row-cap[{cat}]", False,
                            f"invalid-denominator: {amt:.0f} 计入但 {of} 分母为 0"))
            continue
        pct = amt / base * 100
        ok = pct <= max_pct + _EPS
        out.append((f"row-cap[{cat}]", ok, f"{amt:.0f} = {pct:.2f}% of {of} (cap {max_pct}%)"))
    return out


def check_cash_flow(rows, cash_in):
    """逐年累计现金流动性；cash_flow_check 已开启。缺 cash_in / 某相关 FY 无入账 → FAIL。"""
    spend = {}
    for r in rows:
        if not counts(r) or is_in_kind(r):
            continue
        for y, v in (r.get("years") or {}).items():
            spend[str(y)] = spend.get(str(y), 0.0) + float(v)
    cash_in = {str(k): float(v) for k, v in (cash_in or {}).items()}
    if not cash_in:
        return [("cash-flow", False, "cash_flow_check:true 但未提供 cash_in（fail-closed）")]
    out, cum_s, cum_i = [], 0.0, 0.0
    for fy in sorted(set(spend) | set(cash_in)):
        if fy in spend and fy not in cash_in:
            cum_s += spend.get(fy, 0.0)
            out.append((f"cash-flow[FY{fy}]", False,
                        f"FY{fy} 有支出 {spend[fy]:.0f} 但未声明 cash_in（fail-closed）"))
            continue
        cum_s += spend.get(fy, 0.0)
        cum_i += cash_in.get(fy, 0.0)
        ok = cum_s <= cum_i + 1e-6
        out.append((f"cash-flow[FY{fy}]", ok, f"cum spend {cum_s:.0f} vs cum cash-in {cum_i:.0f}"))
    return out


def check_matched(min_ratio, requested, cocon):
    if min_ratio is None:
        return []
    ratio = (cocon / requested) if requested else 0.0
    ok = ratio >= float(min_ratio) - _EPS
    return [("matched-funding", ok,
             f"co-contribution/requested = {cocon:.0f}/{requested:.0f} = {ratio:.3f} (min {min_ratio})")]


def check_credits(rows):
    """credit 行必须显式声明 counts_toward_total (fail-closed)；total-cash 恒排除 credit。"""
    creds = [r for r in rows if is_credit(r)]
    if not creds:
        return []
    for r in creds:
        if "counts_toward_total" not in r:
            return [("credit-vs-cash", False,
                     f"credit 行 {r.get('category')!r} 缺显式 counts_toward_total（fail-closed）")]
    excluded = sum(row_total(r) for r in creds if not counts(r))
    included = sum(row_total(r) for r in creds if counts(r))
    detail = (f"credits excluded-from-total: {excluded:.0f}; credits in-total: {included:.0f}; "
              f"total-cash excludes all credit")
    return [("credit-vs-cash", True, detail)]


def check_phase(rows, phased_if_min, requested):
    """requested ≥ phased_if_min → 需 ≥2 个有成本 phase (first-class phase 轴)。"""
    if phased_if_min is None:
        return []
    thresh = float(phased_if_min)
    if requested < thresh:
        return [("phased-budget", True,
                 f"requested {requested:.0f} < {thresh:.0f}: 无需分期")]
    phases = {}
    for r in rows:
        if r.get("funding_source") != "requested" or not counts(r):
            continue
        ph = r.get("phase")
        if ph is None:
            continue
        if row_total(r) > 0:
            phases[str(ph)] = phases.get(str(ph), 0.0) + row_total(r)
    n = len([p for p, v in phases.items() if v > 0])
    ok = n >= 2
    detail = f"requested {requested:.0f} >= {thresh:.0f}: {n} costed phase(s) {sorted(phases)}"
    if not ok:
        detail += " — 需 >=2 个有成本 phase (rows[].phase)"
    return [("phased-budget", ok, detail)]


def check_balance(rows, cfg):
    """双录平衡: enabled 时 income==expenditure（|delta| ≤ tolerance）。
    fail-closed —— 任一计入行缺 side∈{income,expenditure} 即 FAIL，绝不默认 side。
    in-kind 天然出现在双边（income 行 + expenditure 行各一），按 side 求和即自洽，不特判。"""
    if not cfg or not cfg.get("enabled"):
        return []
    tol = float(cfg.get("tolerance", 0) or 0)
    counted = [r for r in rows if counts(r)]
    missing = [r for r in counted if r.get("side") not in ("income", "expenditure")]
    if missing:
        cats = ", ".join(str(r.get("category")) for r in missing)
        return [("balance", False,
                 f"balance_check.enabled 但 {len(missing)} 个计入行缺 side∈{{income,expenditure}}: "
                 f"{cats}（fail-closed，无法核平衡）")]
    income = sum(row_total(r) for r in counted if r.get("side") == "income")
    expenditure = sum(row_total(r) for r in counted if r.get("side") == "expenditure")
    delta = income - expenditure
    ok = abs(delta) <= tol + _EPS
    return [("balance", ok,
             f"income {income:.0f} vs expenditure {expenditure:.0f} "
             f"(delta {delta:+.0f}, tol {tol:.0f})")]


def check_declared(rows, declared, total, requested, total_cash):
    if not declared:
        return []
    out = []
    live = {"total": total, "requested": requested, "total-cash": total_cash}
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
            years[str(y)] = years.get(str(y), 0.0) + float(v)
    return years, orgs


def run(data):
    rows = data.get("rows") or []
    validate_values(rows)                       # fail-closed hard error
    total, requested, cocon, total_cash = totals(rows)
    results = []
    results += check_row_caps(rows, data.get("row_caps") or [], total, requested, total_cash)
    results += check_matched(data.get("matched_funding_min_ratio"), requested, cocon)
    results += check_credits(rows)
    results += check_phase(rows, data.get("phased_if_min"), requested)
    results += check_balance(rows, data.get("balance_check"))
    results += check_declared(rows, data.get("declared_totals"), total, requested, total_cash)
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


def _fails(data):
    """辅助：跑 run()，返回 {name: ok}。"""
    return {n: ok for n, ok, _ in run(data)[0]}


def self_test():
    # ── valid base case: of REQUIRED on every cap ──
    data = {
        "matched_funding_min_ratio": 1.0,
        "row_caps": [{"category": "audit", "max_pct": 1.0, "of": "total"},
                     {"category": "overseas", "max_pct": 10.0, "of": "total"}],
        "declared_totals": {"requested": 10000},
        "rows": [  # total(counted)=20000: audit 300 + overseas 700 + salary 9000 + in-kind 10000
            {"category": "audit", "funding_source": "requested", "kind": "cash", "years": {2026: 300}},      # 1.5%>1% FAIL
            {"category": "overseas", "funding_source": "requested", "kind": "cash", "years": {2026: 700}},   # 3.5%<10% PASS
            {"category": "salary", "funding_source": "requested", "kind": "cash", "years": {2026: 9000}},
            {"category": "salary", "funding_source": "co-contribution", "kind": "in-kind", "years": {2026: 10000}},
            {"category": "compute", "funding_source": "requested", "kind": "credit",
             "counts_toward_total": False, "years": {2026: 50000}},
        ],
    }
    results, tot, _ = run(data)
    d = {n: (ok, det) for n, ok, det in results}
    assert d["row-cap[audit]"][0] is False, d["row-cap[audit]"]
    assert d["row-cap[overseas]"][0] is True, d["row-cap[overseas]"]
    assert d["matched-funding"][0] is True, d["matched-funding"]
    assert d["declared[requested]"][0] is True, d["declared[requested]"]
    assert d["credit-vs-cash"][0] is True, d["credit-vs-cash"]         # credit excluded, flag explicit
    assert abs(tot[0] - 20000) < 1e-6, tot                             # credit excluded → 20000
    assert abs(tot[3] - 10000) < 1e-6, ("total-cash excludes in-kind+credit", tot)

    # ── C3: `of` REQUIRED — missing/unknown → BudgetError ──
    for bad in (None, "total_cash", "cash"):
        try:
            run({"row_caps": [{"category": "x", "max_pct": 5.0, **({"of": bad} if bad else {})}],
                 "rows": [{"category": "x", "funding_source": "requested", "kind": "cash", "years": {2026: 1}}]})
        except BudgetError:
            pass
        else:
            raise AssertionError(f"missing/unknown of ({bad!r}) must raise BudgetError")

    # ── C3: dual denominator — total PASS(漏) vs total-cash FAIL(抓) ──
    tc_rows = [
        {"category": "overseas", "funding_source": "requested", "kind": "cash", "years": {2026: 6000}},
        {"category": "labour", "funding_source": "requested", "kind": "cash", "years": {2026: 44000}},
        {"category": "in_kind", "funding_source": "co-contribution", "kind": "in-kind", "years": {2026: 50000}},
    ]
    assert _fails({"row_caps": [{"category": "overseas", "max_pct": 10.0, "of": "total-cash"}],
                   "rows": tc_rows})["row-cap[overseas]"] is False
    assert _fails({"row_caps": [{"category": "overseas", "max_pct": 10.0, "of": "total"}],
                   "rows": tc_rows})["row-cap[overseas]"] is True

    # ── C3: zero denominator + nonzero numerator → invalid-denominator FAIL ──
    zden = _fails({"row_caps": [{"category": "overseas", "max_pct": 10.0, "of": "requested"}],
                   "rows": [{"category": "overseas", "funding_source": "co-contribution",
                             "kind": "cash", "years": {2026: 5000}}]})
    assert zden["row-cap[overseas]"] is False, "nonzero cap against zero requested → FAIL"
    # both zero → N/A PASS
    zna = _fails({"row_caps": [{"category": "overseas", "max_pct": 10.0, "of": "requested"}],
                  "rows": [{"category": "labour", "funding_source": "co-contribution",
                            "kind": "cash", "years": {2026: 5000}}]})
    assert zna["row-cap[overseas]"] is True, "both zero → N/A"

    # ── C3: negative value → BudgetError; kind: refund allowed ──
    try:
        run({"rows": [{"category": "x", "funding_source": "requested", "kind": "cash", "years": {2026: -5}}]})
    except BudgetError:
        pass
    else:
        raise AssertionError("negative value must raise BudgetError")
    run({"rows": [{"category": "x", "funding_source": "requested", "kind": "refund", "years": {2026: -5}}]})  # ok

    # ── C3: credit missing explicit counts_toward_total → FAIL ──
    cm = _fails({"rows": [{"category": "compute", "funding_source": "requested",
                           "kind": "credit", "years": {2026: 1000}}]})
    assert cm["credit-vs-cash"] is False, "credit without explicit flag must FAIL"

    # ── C3: cash-flow front-loaded FAIL even when row caps pass ──
    cf = {"cash_flow_check": True, "cash_in": {2026: 5000, 2027: 45000},
          "row_caps": [{"category": "overseas", "max_pct": 10.0, "of": "total"}],
          "rows": [{"category": "labour", "funding_source": "requested", "kind": "cash", "years": {2026: 46000}},
                   {"category": "overseas", "funding_source": "requested", "kind": "cash", "years": {2026: 4000}}]}
    r_cf = _fails(cf)
    assert r_cf["row-cap[overseas]"] is True, "row caps all pass"
    assert r_cf["cash-flow[FY2026]"] is False, "front-loaded spend must FAIL FY2026"
    assert r_cf["cash-flow[FY2027]"] is True, "recovers by FY2027"
    r_ok = _fails(dict(cf, cash_in={2026: 60000, 2027: 0}))
    assert all(v for n, v in r_ok.items() if n.startswith("cash-flow")), "well-funded passes"

    # ── C3: cash_flow_check true but cash_in missing → FAIL ──
    r_missing = _fails({"cash_flow_check": True,
                        "rows": [{"category": "x", "funding_source": "requested", "kind": "cash", "years": {2026: 100}}]})
    assert r_missing["cash-flow"] is False, "cash_flow_check with no cash_in must FAIL"
    # cash_in present but a spending FY absent → that FY FAILs
    r_gap = _fails({"cash_flow_check": True, "cash_in": {2026: 100},
                    "rows": [{"category": "x", "funding_source": "requested", "kind": "cash", "years": {2027: 50}}]})
    assert r_gap["cash-flow[FY2027]"] is False, "spend FY with no cash_in must FAIL"

    # ── C3: FY key normalize — mixed int/str keys sort without error ──
    per_year_org([{"category": "x", "years": {2026: 1}}, {"category": "y", "years": {"p1": 2}}])
    check_cash_flow([{"category": "x", "funding_source": "requested", "kind": "cash", "years": {"p1": 10}}],
                    {"p1": 20})

    # ── C3: phased_if_min — requested >= threshold needs >=2 costed phases ──
    phased_ok = _fails({"phased_if_min": 200000,
                        "rows": [{"category": "a", "funding_source": "requested", "kind": "cash", "phase": "p1", "years": {2026: 150000}},
                                 {"category": "b", "funding_source": "requested", "kind": "cash", "phase": "p2", "years": {2026: 100000}}]})
    assert phased_ok["phased-budget"] is True, "2 costed phases passes"
    phased_bad = _fails({"phased_if_min": 200000,
                         "rows": [{"category": "a", "funding_source": "requested", "kind": "cash", "phase": "p1", "years": {2026: 250000}}]})
    assert phased_bad["phased-budget"] is False, ">=200k with 1 phase must FAIL"
    phased_na = _fails({"phased_if_min": 200000,
                        "rows": [{"category": "a", "funding_source": "requested", "kind": "cash", "years": {2026: 100000}}]})
    assert phased_na["phased-budget"] is True, "under threshold → phasing not required"

    # ── D21: double-entry balance — income==expenditure PASS ──
    bal_ok = _fails({"balance_check": {"enabled": True},
                     "rows": [{"category": "grant-income", "side": "income", "kind": "cash", "years": {2026: 60000, 2027: 40000}},
                              {"category": "salaries", "side": "expenditure", "kind": "cash", "years": {2026: 60000, 2027: 40000}}]})
    assert bal_ok["balance"] is True, "balanced double-entry must PASS"
    # in-kind on BOTH sides balances naturally (income in-kind + expenditure in-kind) — no special-case
    bal_ik = _fails({"balance_check": {"enabled": True},
                     "rows": [{"category": "grant-income", "side": "income", "kind": "cash", "years": {2026: 100000}},
                              {"category": "partner-inkind", "side": "income", "kind": "in-kind", "years": {2026: 20000}},
                              {"category": "salaries", "side": "expenditure", "kind": "cash", "years": {2026: 100000}},
                              {"category": "inkind-effort", "side": "expenditure", "kind": "in-kind", "years": {2026: 20000}}]})
    assert bal_ik["balance"] is True, "in-kind on both sides still balances"

    # ── D21: unbalanced (income != expenditure) → FAIL ──
    bal_bad = _fails({"balance_check": {"enabled": True},
                      "rows": [{"category": "grant-income", "side": "income", "kind": "cash", "years": {2026: 100000}},
                               {"category": "salaries", "side": "expenditure", "kind": "cash", "years": {2026: 120000}}]})
    assert bal_bad["balance"] is False, "income != expenditure must FAIL"

    # ── D21: fail-closed — enabled + counted row missing side → FAIL (never default a side) ──
    bal_missing = _fails({"balance_check": {"enabled": True},
                          "rows": [{"category": "grant-income", "side": "income", "kind": "cash", "years": {2026: 100000}},
                                   {"category": "salaries", "kind": "cash", "years": {2026: 100000}}]})
    assert bal_missing["balance"] is False, "counted row missing side must FAIL (fail-closed)"
    # non-counted row without side is fine (excluded from balance)
    bal_ncnt = _fails({"balance_check": {"enabled": True},
                       "rows": [{"category": "grant-income", "side": "income", "kind": "cash", "years": {2026: 100000}},
                                {"category": "salaries", "side": "expenditure", "kind": "cash", "years": {2026: 100000}},
                                {"category": "memo", "counts_toward_total": False, "kind": "cash", "years": {2026: 5}}]})
    assert bal_ncnt["balance"] is True, "non-counted sideless row excluded → still balances"
    # tolerance absorbs a small delta
    bal_tol = _fails({"balance_check": {"enabled": True, "tolerance": 100},
                      "rows": [{"category": "grant-income", "side": "income", "kind": "cash", "years": {2026: 100000}},
                               {"category": "salaries", "side": "expenditure", "kind": "cash", "years": {2026: 100050}}]})
    assert bal_tol["balance"] is True, "delta within tolerance PASSes"

    # ── D21: balance_check absent/disabled → no balance rule (ARC-style budgets unchanged) ──
    assert "balance" not in _fails(data), "balance_check absent → no balance rule (existing budgets unchanged)"
    assert "balance" not in _fails({"balance_check": {"enabled": False},
                                    "rows": [{"category": "x", "kind": "cash", "years": {2026: 1}}]}), \
        "balance_check disabled → no side required, no rule"

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
    try:
        results, tot, yo = run(data)
    except BudgetError as exc:
        print(f"BUDGET ERROR (fail-closed): {exc}", file=sys.stderr)
        return 2
    render(results, tot, yo)
    return 1 if any(not ok for _, ok, _ in results) else 0


if __name__ == "__main__":
    sys.exit(main())
