# /// script
# requires-python = ">=3.9"
# dependencies = ["pyyaml"]
# ///
"""预算构建器 (Stage 5: budget COSTING/itemisation) — 与 validate_budget 互补，FAIL-CLOSED.

validate_budget.py **校验**一份已建好的预算；本脚本从人给的输入**构建/itemise**它：
personnel 按 `人 × FTE × 年薪 × 年份 + on-costs` 逐项算，other_costs 逐年落，产出
(1) validate_budget 可直接消费的 `budget.yaml`（rows[] 同 schema），(2) 人可读的
itemised 表（喂 paste-ready Section-5 字段 / docx）。

设计公理（承接 CLAUDE.md "scripts never author or judge content"）：本脚本**只做算术**，
不选 rate/FTE。人没给的 rate/FTE/金额 = 字面 `null` = `[TO SET]`，脚本**绝不臆造/默认为 0**。
选率是判断（人的事）；从给定率求和是算术（脚本的事）——算钱本就该在这层，不该漏。

输入 `budget-plan.yaml`:
    currency: AUD
    target_total: 285000            # 可省；给了则核对 computed==target，差额只报不抹平
    personnel:
      - id: pers-ra                 # 稳定 id → 成为 budget row id（spine budget_line 引用它）
        role: "Research Associate/Engineer"
        person: inv-lead            # 可省 → entity-store investigators[].id（spine 就位）
        fte: 0.5
        annual_rate: 95000          # null == [TO SET]，绝不臆造
        years: [2027, 2028, 2029]
        on_cost_pct: 0.30           # null == [TO SET]；on-cost 单独成行（便于 row-cap 分别打）
        category: personnel         # 缺省 personnel
        funding_source: requested   # 缺省 requested
        org: uts
        kind: cash                  # 缺省 cash
        phase: p1                   # 可省，透传
    other_costs:
      - id: travel
        category: travel
        description: "AU↔VN economy travel, per diem, visa"
        years: {2027: 15000, 2028: 18000, 2029: 12000}   # 逐年；某年 null == [TO SET]
        # 或: total: 45000, spread: even                   # 备选：按 personnel 年份集均摊
        funding_source: requested
        org: uts
        kind: cash

用法:
    uv run build_budget.py budget-plan.yaml -o budget.yaml     # 建 + 写 + 打印 itemised 表
    uv run build_budget.py --self-test

退出码（对齐 validate_budget/build_manifest）:
    0 = 完整、无 [TO SET]、（若给 target）无缺口
    1 = rule-level：存在 [TO SET] 空值，或 target 缺口 —— fail-closed，落审
    2 = 结构错误：plan 畸形 / 负 fte,rate / 缺 personnel&other_costs
"""
import argparse
import sys

EPS = 1e-9


class PlanError(ValueError):
    """输入结构性错误 (畸形 plan / 负值)，fail-closed 硬错误 → exit 2。"""


def _num_or_none(v, name):
    if v is None:
        return None
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        raise PlanError(f"{name} 需为数值或 null，得到 {v!r}")
    if v < 0:
        raise PlanError(f"{name} 不可为负: {v}")
    return float(v)


def _passthrough(src, row, keys):
    for k in keys:
        if src.get(k) is not None:
            row[k] = src[k]


def _match_level(level, scales):
    """精确或子串匹配费率表的 level 名（'Level A' 命中 'Level A - Research Associate…'）。"""
    if level in scales:
        return level
    lo = str(level).lower().strip()
    cands = [k for k in scales if lo and (lo in k.lower() or k.lower().startswith(lo))]
    return cands[0] if len(cands) == 1 else (cands[0] if cands else None)


def resolve_rate_ref(ref, rates, years, pid):
    """rate_ref{level,step,step_progression} × 费率表 → (base_by_year, on_cost_default, note).

    逐年 step 递进（默认 true；§UTS 规则要求项目内每年进一档），越出最高档则封顶。
    表/level/step 缺失 → base 为 None（=[TO SET]，绝不臆造），并回一条 note。
    """
    if not rates:
        raise PlanError(f"{pid}: rate_ref 需要 --rates 费率表（rule 每年变，是 instance data）")
    scales = rates.get("scales") or {}
    lvl = _match_level(ref.get("level"), scales)
    if lvl is None:
        return {y: None for y in years}, None, f"level {ref.get('level')!r} 不在费率表"
    steps = scales[lvl]
    step_keys = list(steps)
    start = ref.get("step")
    if start not in steps:
        return {y: None for y in years}, None, f"step {start!r} 不在 {lvl}"
    idx0 = step_keys.index(start)
    prog = ref.get("step_progression", True)
    base_by_year = {}
    for i, y in enumerate(years):
        j = min(idx0 + i, len(step_keys) - 1) if prog else idx0
        base_by_year[y] = steps[step_keys[j]].get("base")
    return base_by_year, rates.get("on_cost_pct"), None


def build(plan, rates=None):
    """→ (currency, out_rows[], items[], blocked[], target, grand). out_rows = validate_budget schema.

    `rates` (optional): a rate-table dict (see import_uts_rates.py) letting a personnel row use
    `rate_ref:{level,step,step_progression}` instead of a hand-typed `annual_rate` — the base is
    looked up per year (with annual step progression). Rates change yearly = instance data; the
    table is supplied via --rates, never hardcoded here.
    """
    if not isinstance(plan, dict):
        raise PlanError("budget-plan 根节点须为映射")
    personnel = plan.get("personnel") or []
    other = plan.get("other_costs") or []
    if not personnel and not other:
        raise PlanError("budget-plan 至少需 personnel 或 other_costs 之一")
    currency = plan.get("currency", "AUD")
    target = _num_or_none(plan.get("target_total"), "target_total")

    out_rows, items, blocked = [], [], []

    # ── personnel: 人 × FTE × 率 × 年 + on-costs（on-cost 单独成行）──
    for p in personnel:
        pid = p.get("id")
        if not pid:
            raise PlanError("personnel 行缺 id")
        fte = _num_or_none(p.get("fte"), f"{pid}.fte")
        rate = _num_or_none(p.get("annual_rate"), f"{pid}.annual_rate")
        oncost = _num_or_none(p.get("on_cost_pct"), f"{pid}.on_cost_pct")
        years = p.get("years") or []
        if not isinstance(years, list) or not years:
            raise PlanError(f"{pid}.years 须为非空年份列表")

        # ── HDR stipend: base scholarship + top-up, itemised as TWO rows, NO on-cost (§2.8) ──
        stip = p.get("stipend")
        if stip is not None:
            base = _num_or_none(stip.get("base"), f"{pid}.stipend.base")
            topup = _num_or_none(stip.get("top_up"), f"{pid}.stipend.top_up")
            for suffix, amt, cat, lbl in ((("", base, "hdr-stipend", "base scholarship"),
                                           ("-topup", topup, "hdr-stipend-topup", "top-up"))):
                row = {"id": pid + suffix, "category": cat,
                       "funding_source": p.get("funding_source", "requested"),
                       "kind": p.get("kind", "cash"), "years": {y: amt for y in years},
                       "counts_toward_total": True}
                _passthrough(p, row, ("org", "phase", "funding_status"))
                out_rows.append(row)
                items.append({"section": "PERSONNEL", "id": pid + suffix,
                              "label": f"{p.get('role', pid)} — HDR {lbl}"
                                       f" ({'[TO SET]' if amt is None else format(amt, ',.0f')}/yr, no on-cost)",
                              "years": {y: amt for y in years}})
                if amt is None:
                    blocked.append(f"{pid}{suffix}: HDR {lbl} [TO SET]")
            continue

        # per-year base: from a rate_ref lookup (with step progression), else the flat annual_rate
        ref = p.get("rate_ref")
        ref_note = None
        if ref:
            base_by_year, oc_default, ref_note = resolve_rate_ref(ref, rates, years, pid)
            if oncost is None:
                oncost = _num_or_none(oc_default, f"{pid}.on_cost(table)")
        else:
            base_by_year = {y: rate for y in years}

        salary_years, oncost_years = {}, {}
        for y in years:
            b = base_by_year.get(y)
            sal = None if (fte is None or b is None) else round(fte * b, 2)   # round to cents: real
            salary_years[y] = sal                                            # rate tables carry many dp
            oncost_years[y] = None if (sal is None or oncost is None) else round(sal * oncost, 2)

        base = {"id": pid, "category": p.get("category", "personnel"),
                "funding_source": p.get("funding_source", "requested"),
                "kind": p.get("kind", "cash"), "years": salary_years,
                "counts_toward_total": True}
        _passthrough(p, base, ("org", "phase", "funding_status"))
        out_rows.append(base)

        fte_txt = f"{fte:g}" if fte is not None else "[TO SET]"
        if ref:
            bmin = min((v for v in base_by_year.values() if v is not None), default=None)
            bmax = max((v for v in base_by_year.values() if v is not None), default=None)
            src = f"{ref.get('level')}/{ref.get('step')}"
            src += "+prog" if ref.get("step_progression", True) else ""
            rate_txt = (f"{bmin:,.0f}→{bmax:,.0f}" if bmin != bmax else f"{bmin:,.0f}") if bmin is not None else "[TO SET]"
            label = f"{p.get('role', pid)} — {fte_txt} FTE × {rate_txt}/yr ({src})"
        else:
            rate_txt = f"{rate:,.0f}" if rate is not None else "[TO SET]"
            label = f"{p.get('role', pid)} — {fte_txt} FTE × {rate_txt}/yr × {years}"
        items.append({"section": "PERSONNEL", "id": pid, "label": label, "years": salary_years})
        if ref_note:
            blocked.append(f"{pid}: {ref_note}")
        if any(v is None for v in salary_years.values()):
            blocked.append(f"{pid}: salary [TO SET] (fte, rate, or rate_ref lookup unresolved)")

        # on-cost 行（率 0 或全 0 则跳过）
        if oncost is None or oncost > 0:
            oc_id = f"{pid}-oncost"
            oc = {"id": oc_id, "category": "on-costs",
                  "funding_source": p.get("funding_source", "requested"),
                  "kind": p.get("kind", "cash"), "years": oncost_years,
                  "counts_toward_total": True}
            _passthrough(p, oc, ("org", "phase", "funding_status"))
            out_rows.append(oc)
            oc_txt = f"{oncost * 100:g}%" if oncost is not None else "[TO SET]"
            items.append({"section": "PERSONNEL", "id": oc_id,
                          "label": f"{p.get('role', pid)} on-costs ({oc_txt})",
                          "years": oncost_years})
            if any(v is None for v in oncost_years.values()):
                blocked.append(f"{oc_id}: on-costs [TO SET] (on_cost_pct or salary is null)")

    # ── other_costs: 逐年，或 total+spread 按 personnel 年份集均摊 ──
    fy_set = []
    for it in items:
        for y in it["years"]:
            if y not in fy_set:
                fy_set.append(y)
    for o in other:
        oid = o.get("id")
        if not oid:
            raise PlanError("other_costs 行缺 id")
        years = o.get("years")
        if years is None and o.get("total") is not None:
            tot = _num_or_none(o.get("total"), f"{oid}.total")
            spread = o.get("spread", "even")
            if spread != "even":
                raise PlanError(f"{oid}.spread 仅支持 'even'（或直接给 years）")
            if not fy_set:
                raise PlanError(f"{oid} 用 total+spread 但无 personnel 年份集可摊；请直接给 years")
            per = tot / len(fy_set)
            years = {y: per for y in fy_set}
        elif isinstance(years, dict):
            years = {y: _num_or_none(v, f"{oid}.years[{y}]") for y, v in years.items()}
        else:
            raise PlanError(f"{oid} 需 years 映射，或 total(+spread)")
        row = {"id": oid, "category": o.get("category", "other"),
               "funding_source": o.get("funding_source", "requested"),
               "kind": o.get("kind", "cash"), "years": years,
               "counts_toward_total": o.get("counts_toward_total", True)}
        _passthrough(o, row, ("org", "phase", "funding_status"))
        out_rows.append(row)
        items.append({"section": "OTHER", "id": oid,
                      "label": f"{o.get('description', o.get('category', oid))}", "years": years})
        if any(v is None for v in years.values()):
            blocked.append(f"{oid}: amount [TO SET]")

    # ── 总额：Σ rows（缺省不计 None）；requested 计入 declared_totals ──
    def rtotal(r):
        return sum(v for v in r["years"].values() if v is not None)

    grand = sum(rtotal(r) for r in out_rows if r.get("counts_toward_total", True))
    requested = sum(rtotal(r) for r in out_rows
                    if r.get("counts_toward_total", True) and r.get("funding_source") == "requested")
    return currency, out_rows, items, blocked, target, grand, requested


def render_table(currency, items, out_rows, blocked, target, grand, requested):
    lines = [f"== Itemised budget ({currency}) =="]
    section = None
    for it in items:
        if it["section"] != section:
            section = it["section"]
            lines.append(section)
        cells = "  ".join(f"{y}:{'[TO SET]' if v is None else format(v, ',.0f')}"
                          for y, v in it["years"].items())
        sub = sum(v for v in it["years"].values() if v is not None)
        has_gap = any(v is None for v in it["years"].values())
        subtxt = f"subtotal {sub:,.0f}" + (" (+[TO SET])" if has_gap else "")
        lines.append(f"  {it['id']:<22} {it['label']}")
        lines.append(f"  {'':<22}   {cells}   {subtxt}")
    # by-year
    years = {}
    for r in out_rows:
        if not r.get("counts_toward_total", True):
            continue
        for y, v in r["years"].items():
            if v is not None:
                years[y] = years.get(y, 0.0) + v
    lines.append("== totals ==")
    lines.append("  by year: " + ", ".join(f"{y}:{v:,.0f}" for y, v in sorted(years.items(), key=lambda x: str(x[0]))))
    lines.append(f"  requested: {requested:,.0f}   grand(counted): {grand:,.0f}")
    if target is not None:
        gap = target - grand
        lines.append(f"  target: {target:,.0f}   GAP: {gap:+,.0f}"
                     + ("  (reconciled)" if abs(gap) <= 1.0 else "  — NOT reconciled (fix inputs, never fudge)"))
    if blocked:
        lines.append("== [TO SET] (fail-closed — supply before validate_budget) ==")
        lines.extend(f"  ! {b}" for b in blocked)
    return "\n".join(lines)


def emit_yaml(currency, out_rows, target, requested):
    doc = {"declared_totals": {"requested": round(requested, 2)}, "rows": out_rows}
    return doc


def run(plan, rates=None):
    currency, out_rows, items, blocked, target, grand, requested = build(plan, rates)
    gap_bad = target is not None and abs(target - grand) > 1.0    # dollar tolerance (grant budgets are whole $; sub-$ = rounding noise)
    return currency, out_rows, items, blocked, target, grand, requested, gap_bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan", nargs="?", help="budget-plan.yaml 路径")
    ap.add_argument("-o", "--out", help="写出的 budget.yaml 路径（省则只打印表）")
    ap.add_argument("--rates", help="费率表 yaml（import_uts_rates.py 产出）：让 personnel rate_ref 查表")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.plan:
        ap.error("需要 budget-plan.yaml 路径 (或 --self-test)")
    import yaml
    plan = yaml.safe_load(open(args.plan, encoding="utf-8")) or {}
    rates = yaml.safe_load(open(args.rates, encoding="utf-8")) if args.rates else None
    try:
        currency, out_rows, items, blocked, target, grand, requested, gap_bad = run(plan, rates)
    except PlanError as exc:
        print(f"PLAN ERROR (fail-closed): {exc}", file=sys.stderr)
        return 2
    print(render_table(currency, items, out_rows, blocked, target, grand, requested))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            yaml.safe_dump(emit_yaml(currency, out_rows, target, requested), fh,
                           sort_keys=False, allow_unicode=True, default_flow_style=False)
        print(f"\nwrote {args.out}")
    return 1 if (blocked or gap_bad) else 0


def self_test():
    # 完整算术：0.5 FTE × 100000 × [2027,2028] + 30% on-cost
    plan = {"currency": "AUD", "target_total": 130000,
            "personnel": [{"id": "ra", "role": "RA", "fte": 0.5, "annual_rate": 100000,
                           "years": [2027, 2028], "on_cost_pct": 0.30, "org": "uts"}]}
    currency, rows, items, blocked, target, grand, requested, gap_bad = run(plan)
    by_id = {r["id"]: r for r in rows}
    assert by_id["ra"]["years"] == {2027: 50000, 2028: 50000}, by_id["ra"]
    assert by_id["ra-oncost"]["years"] == {2027: 15000, 2028: 15000}, by_id["ra-oncost"]
    assert abs(grand - 130000) < EPS, grand
    assert not blocked and not gap_bad, (blocked, gap_bad)

    # null rate → salary+oncost [TO SET]，exit-worthy
    plan_null = {"personnel": [{"id": "ra", "fte": 1.0, "annual_rate": None,
                                "years": [2027], "on_cost_pct": 0.30}]}
    _, rows2, _, blocked2, _, grand2, _, _ = run(plan_null)
    assert {r["id"]: r["years"] for r in rows2}["ra"] == {2027: None}, "null rate → null amount, 不臆造"
    assert grand2 == 0.0, "None 不计入总额（非 0 填充）"
    assert any("ra:" in b for b in blocked2) and any("ra-oncost:" in b for b in blocked2), blocked2

    # target 缺口只报不抹平
    _, _, _, _, _, g3, _, gap3 = run({"target_total": 999999,
                                      "personnel": [{"id": "x", "fte": 1, "annual_rate": 1000,
                                                     "years": [2027], "on_cost_pct": 0}]})
    assert gap3 is True and abs(g3 - 1000) < EPS, (g3, gap3)

    # other_costs total+spread 均摊到 personnel 年份集
    plan_sp = {"personnel": [{"id": "p", "fte": 1, "annual_rate": 100, "years": [2027, 2028], "on_cost_pct": 0}],
               "other_costs": [{"id": "travel", "category": "travel", "total": 1000, "spread": "even"}]}
    _, rows_sp, _, _, _, _, _, _ = run(plan_sp)
    tr = {r["id"]: r for r in rows_sp}["travel"]
    assert tr["years"] == {2027: 500, 2028: 500}, tr

    # 负值 → PlanError（exit 2）
    for bad in ({"personnel": [{"id": "n", "fte": -1, "annual_rate": 1, "years": [2027], "on_cost_pct": 0}]},
                {"personnel": [{"id": "n", "fte": 1, "annual_rate": -5, "years": [2027], "on_cost_pct": 0}]}):
        try:
            run(bad)
        except PlanError:
            pass
        else:
            raise AssertionError("负 fte/rate 必须 PlanError")

    # 空 plan → PlanError
    try:
        run({"currency": "AUD"})
    except PlanError:
        pass
    else:
        raise AssertionError("既无 personnel 也无 other_costs 必须 PlanError")

    # rate_ref 查表 + 逐年 step 递进：Level A Step1→Step2→Step3 across 3 years, on_cost from table
    rates = {"on_cost_pct": 0.30, "scales": {
        "Level A - Research Associate": {
            "Step 1": {"base": 100000.0}, "Step 2": {"base": 110000.0}, "Step 3": {"base": 120000.0}}}}
    pr = {"personnel": [{"id": "ra", "fte": 1.0, "years": [2027, 2028, 2029],
                         "rate_ref": {"level": "Level A", "step": "Step 1", "step_progression": True}}]}
    _, rr, _, rb, _, rg, _, _ = run(pr, rates)
    by = {r["id"]: r for r in rr}
    assert by["ra"]["years"] == {2027: 100000.0, 2028: 110000.0, 2029: 120000.0}, by["ra"]  # step progression
    assert by["ra-oncost"]["years"] == {2027: 30000.0, 2028: 33000.0, 2029: 36000.0}, by["ra-oncost"]  # 30% from table
    assert abs(rg - (330000 + 99000)) < EPS, rg
    assert not rb, rb
    # no progression → flat Step 1 all years
    _, rr2, _, _, _, _, _, _ = run({"personnel": [{"id": "ra", "fte": 1.0, "years": [2027, 2028],
        "rate_ref": {"level": "Level A", "step": "Step 2", "step_progression": False}}]}, rates)
    assert {r["id"]: r for r in rr2}["ra"]["years"] == {2027: 110000.0, 2028: 110000.0}
    # unknown step → [TO SET] (base null), blocked, never invented
    _, _, _, rb3, _, _, _, _ = run({"personnel": [{"id": "ra", "fte": 1.0, "years": [2027],
        "rate_ref": {"level": "Level A", "step": "Step 9"}}]}, rates)
    assert any("Step 9" in b for b in rb3), rb3
    # rate_ref with no --rates table → PlanError
    try:
        run({"personnel": [{"id": "ra", "fte": 1.0, "years": [2027], "rate_ref": {"level": "X", "step": "Y"}}]})
    except PlanError:
        pass
    else:
        raise AssertionError("rate_ref 无 --rates 必须 PlanError")

    # HDR stipend: base + top-up, two rows, no on-cost
    stp = run({"personnel": [{"id": "phd", "role": "PhD", "years": [2027, 2028, 2029],
                              "stipend": {"base": 39000, "top_up": 5000}, "org": "uts"}]})
    srows = {r["id"]: r for r in stp[1]}
    assert srows["phd"]["years"] == {2027: 39000, 2028: 39000, 2029: 39000}, srows["phd"]
    assert srows["phd-topup"]["years"] == {2027: 5000, 2028: 5000, 2029: 5000}, srows["phd-topup"]
    assert "phd-oncost" not in srows and "phd-topup-oncost" not in srows, "stipends carry NO on-cost"
    assert abs(stp[5] - (39000 + 5000) * 3) < EPS, stp[5]        # grand = (base+topup)×3
    assert not stp[3], stp[3]                                    # complete → no [TO SET]
    # missing top_up → [TO SET], never invented
    _, sr2, _, sb2, _, _, _, _ = run({"personnel": [{"id": "phd", "years": [2027],
                                                     "stipend": {"base": 39000}}]})
    assert {r["id"]: r for r in sr2}["phd-topup"]["years"] == {2027: None}, sr2
    assert any("phd-topup" in b for b in sb2), sb2

    # 往返：emit 的 rows 能被 validate_budget 消费（结构对齐）
    doc = emit_yaml(*[currency] + [rows] + [target, requested])
    assert doc["rows"] and "declared_totals" in doc
    for r in doc["rows"]:
        assert {"id", "category", "funding_source", "kind", "years", "counts_toward_total"} <= set(r), r

    print("self-test OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
