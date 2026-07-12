# /// script
# requires-python = ">=3.9"
# dependencies = ["pyyaml"]
# ///
"""合作方 co-contribution 分配器 (Stage B2: matched-funding split) — FAIL-CLOSED.

Linkage / CRC-P / AEA 有 matched-funding 生死线（co-contribution : grant ≥ 目标比）。validate_budget
只**事后查**比例，此前无构建器。本脚本从**目标比 + 合作方**构建 per-partner 的现金/in-kind ×
逐年分摊，产出可直接进 entity-store contributions + budget co-contribution 行，并核对是否达标。

设计公理（同族 builder）：**只做算术，不臆造**。合作方既没 declared(cash/in_kind) 也没 share
→ [TO SET]。达不到目标比 → 报缺口（never fudge）。逐年分摊四舍五入到分。

两种合作方模式:
  - declared:  直接给 cash / in_kind（null = [TO SET]）
  - derive:    给 share（占 required 总额的比例）+ cash_fraction（其中现金占比）→ 反算 cash/in_kind

输入 `cocontribution-plan.yaml`:
    grant: 285000                 # 申请的公共资金
    target_ratio: 1.0             # 要求 co-contribution : grant（Linkage 常 1:1）
    currency: AUD
    years: [2027, 2028, 2029]     # 可省；给了则逐年均摊
    partners:
      - {id: pA, name: "Industry A", cash: 100000, in_kind: 50000}        # declared
      - {id: pB, name: "Industry B", share: 0.5, cash_fraction: 0.4}      # derive from share

用法:
    uv run build_cocontribution.py cocontribution-plan.yaml [-o cocontribution.yaml]
    uv run build_cocontribution.py --self-test

退出码: 0 = 完整且达标; 1 = [TO SET] 或未达目标比（缺口只报不抹平）; 2 = 结构错误。
"""
import argparse
import sys

EPS = 0.01   # dollar tolerance


class CoConError(ValueError):
    """结构错误 → exit 2。"""


def _n(v, name):
    if v is None:
        return None
    if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
        raise CoConError(f"{name} 须为非负数值或 null: {v!r}")
    return float(v)


def _spread(total, years):
    """把 total 逐年均摊，四舍五入到分，末年吸收余数保证求和精确。"""
    if total is None:
        return {y: None for y in years} if years else {"total": None}
    if not years:
        return {"total": round(total, 2)}
    per = round(total / len(years), 2)
    out = {y: per for y in years[:-1]}
    out[years[-1]] = round(total - per * (len(years) - 1), 2)
    return out


def compute(plan):
    if not isinstance(plan, dict):
        raise CoConError("cocontribution-plan 根节点须为映射")
    grant = _n(plan.get("grant"), "grant")
    ratio = _n(plan.get("target_ratio"), "target_ratio")
    years = plan.get("years") or []
    partners = plan.get("partners") or []
    if grant is None or ratio is None:
        raise CoConError("grant 与 target_ratio 必填")
    if not partners:
        raise CoConError("至少需 1 个 partner")
    required = grant * ratio

    rows, blocked = [], []
    declared_total = 0.0
    for p in partners:
        pid = p.get("id") or "<partner>"
        cash, in_kind = _n(p.get("cash"), f"{pid}.cash"), _n(p.get("in_kind"), f"{pid}.in_kind")
        share, cash_frac = _n(p.get("share"), f"{pid}.share"), _n(p.get("cash_fraction"), f"{pid}.cash_fraction")

        if share is not None and (cash is not None or in_kind is not None):
            raise CoConError(f"{pid}: share 与 declared(cash/in_kind) 二选一，不可并存")
        if share is not None:
            if cash_frac is None:
                raise CoConError(f"{pid}: 用 share 时需 cash_fraction")
            target = required * share
            cash = round(target * cash_frac, 2)
            in_kind = round(target * (1 - cash_frac), 2)

        if cash is None and in_kind is None:
            blocked.append(f"{pid}: cash/in_kind (or share) [TO SET]")
        c = cash or 0.0
        k = in_kind or 0.0
        declared_total += c + k
        rows.append({"id": pid, "name": p.get("name"),
                     "cash": cash, "in_kind": in_kind,
                     "cash_by_year": _spread(cash, years), "in_kind_by_year": _spread(in_kind, years)})

    achieved = declared_total / grant if grant else 0.0
    gap = required - declared_total
    met = gap <= EPS
    return {
        "grant": grant, "target_ratio": ratio, "required_co_contribution": round(required, 2),
        "declared_co_contribution": round(declared_total, 2),
        "ratio_achieved": round(achieved, 3), "shortfall": round(max(gap, 0.0), 2),
        "meets_target": met, "partners": rows,
    }, blocked


def render(res, blocked):
    L = ["== Co-contribution allocation =="]
    L.append(f"  grant {res['grant']:.0f} × target {res['target_ratio']} = required {res['required_co_contribution']:.0f}")
    for r in res["partners"]:
        cash = "[TO SET]" if r["cash"] is None else f"{r['cash']:.0f}"
        ik = "[TO SET]" if r["in_kind"] is None else f"{r['in_kind']:.0f}"
        L.append(f"  {r['id']:<12} {r.get('name') or '':<18} cash {cash}  in-kind {ik}")
    L.append(f"  declared {res['declared_co_contribution']:.0f}  ratio achieved {res['ratio_achieved']} "
             f"(target {res['target_ratio']})")
    if res["meets_target"]:
        L.append("  MEETS target ratio ✓")
    else:
        L.append(f"  SHORTFALL {res['shortfall']:.0f} — below target ratio (raise contributions or re-plan; never fudge)")
    if blocked:
        L.append("== [TO SET] (fail-closed) ==")
        L.extend(f"  ! {b}" for b in blocked)
    return "\n".join(L)


def run(plan):
    return compute(plan)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan", nargs="?")
    ap.add_argument("-o", "--out")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.plan:
        ap.error("需要 cocontribution-plan.yaml (或 --self-test)")
    import yaml
    plan = yaml.safe_load(open(args.plan, encoding="utf-8")) or {}
    try:
        res, blocked = run(plan)
    except CoConError as exc:
        print(f"CO-CONTRIBUTION ERROR (fail-closed): {exc}", file=sys.stderr)
        return 2
    print(render(res, blocked))
    if args.out:
        import yaml
        yaml.safe_dump(res, open(args.out, "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
        print(f"\nwrote {args.out}")
    return 1 if (blocked or not res["meets_target"]) else 0


def self_test():
    # declared meets 1:1
    res, b = run({"grant": 200000, "target_ratio": 1.0,
                  "partners": [{"id": "pA", "cash": 120000, "in_kind": 80000}]})
    assert res["required_co_contribution"] == 200000 and res["declared_co_contribution"] == 200000, res
    assert res["meets_target"] and res["ratio_achieved"] == 1.0 and not b, (res, b)

    # derive from share: required 100k, share 0.5 → 50k, cash_fraction 0.4 → cash 20k / in-kind 30k
    res2, _ = run({"grant": 100000, "target_ratio": 1.0,
                   "partners": [{"id": "pA", "share": 0.5, "cash_fraction": 0.4},
                                {"id": "pB", "share": 0.5, "cash_fraction": 1.0}]})
    pa = {p["id"]: p for p in res2["partners"]}["pA"]
    assert pa["cash"] == 20000 and pa["in_kind"] == 30000, pa
    assert res2["declared_co_contribution"] == 100000 and res2["meets_target"], res2

    # shortfall: required 200k, declared 150k → not met, shortfall 50k, exit-worthy
    res3, _ = run({"grant": 200000, "target_ratio": 1.0,
                   "partners": [{"id": "pA", "cash": 150000, "in_kind": 0}]})
    assert not res3["meets_target"] and res3["shortfall"] == 50000, res3

    # per-year spread sums exactly
    res4, _ = run({"grant": 90000, "target_ratio": 1.0, "years": [2027, 2028, 2029],
                   "partners": [{"id": "pA", "cash": 90000, "in_kind": 0}]})
    cy = {p["id"]: p for p in res4["partners"]}["pA"]["cash_by_year"]
    assert abs(sum(cy.values()) - 90000) < 1e-6, cy

    # missing cash/in_kind and no share → [TO SET]
    _, b5 = run({"grant": 100000, "target_ratio": 1.0, "partners": [{"id": "pA"}]})
    assert any("pA" in x for x in b5), b5

    # structural: share + declared both → error; share without cash_fraction → error
    for bad in ({"grant": 1, "target_ratio": 1, "partners": [{"id": "p", "share": 0.5, "cash": 1}]},
                {"grant": 1, "target_ratio": 1, "partners": [{"id": "p", "share": 0.5}]},
                {"grant": 1, "target_ratio": 1, "partners": []}):
        try:
            run(bad)
        except CoConError:
            pass
        else:
            raise AssertionError(f"应 CoConError: {bad}")

    print("self-test OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
