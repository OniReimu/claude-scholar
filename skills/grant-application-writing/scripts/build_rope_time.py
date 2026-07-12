# /// script
# requires-python = ">=3.9"
# dependencies = ["pyyaml"]
# ///
"""ROPE-time / 资格日期计算器 (Stage A0: eligibility date arithmetic) — FAIL-CLOSED.

narrative-award / ECR 类方案的**硬资格闸**是"PhD 至今 ≤ N 年"，且 career interruption 可
折抵。§1.1 明说"从记录算，别信记忆的数"，但 `validate_ir.check_computed_gates` 只 eval 人已
算好的数——没脚本做这个日期算术。本脚本补上：从 `phd_conferral` + `interruptions[]` 算出
**校正后 years-since-PhD** 和 **是否在窗内** 的布尔，直接喂 eligibility gate。

设计公理（同 build_budget/build_timeline）：**只做日期算术，不臆造**。`as_of`（资格基准日，
通常是 census/deadline）是显式**输入**，绝不用 `today()`（保证可复现 + 可测）。缺 `phd_conferral`
= `[TO SET]` → BLOCK。interruption 缺日期 = `[TO SET]`。

折抵口径（ARC ROPE 惯例）：一段 interruption 从 `from` 到 `to`，按 `fte_fraction`（工作占比，
缺省 0 = 全脱产）折抵 `duration × (1 - fte_fraction)`；总折抵从 raw 年数里扣，得 effective 年数。
effective ≤ window → 合格。（等价于把窗口按 interruption 外延。）

输入 `rope-plan.yaml`（可从 entity-store investigators[].rope_context 提升）:
    as_of: "2026-07-20"          # 资格基准日（scheme census/deadline）
    phd_conferral: "2018-03-15"  # PhD 授予日；null = [TO SET]
    window_years: 5              # 方案的 years-since-PhD 上限（DECRA=5）
    interruptions:
      - {from: "2020-01-01", to: "2021-06-30", fte_fraction: 0.5, reason: "part-time parental leave"}
      - {from: "2022-03-01", to: "2022-09-01", fte_fraction: 0.0, reason: "medical (full)"}

用法:
    uv run build_rope_time.py rope-plan.yaml [-o rope.yaml]
    uv run build_rope_time.py --self-test

退出码: 0 = 完整且合格; 1 = rule-level（[TO SET] 空值，或算出 NOT eligible）; 2 = 结构错误。
"""
import argparse
import sys
from datetime import date

DAYS_PER_YEAR = 365.25
BORDERLINE_MARGIN = 0.05     # ~18 days: within this of the window, the verdict is date-noise-sensitive


class RopeError(ValueError):
    """结构错误（畸形日期 / from>to / 负 window）→ exit 2。"""


def _pdate(v, name):
    if v is None:
        return None
    try:
        return date.fromisoformat(str(v))
    except ValueError:
        raise RopeError(f"{name} 非 ISO 日期 (YYYY-MM-DD): {v!r}")


def _years(d0, d1):
    return (d1 - d0).days / DAYS_PER_YEAR


def compute(plan):
    if not isinstance(plan, dict):
        raise RopeError("rope-plan 根节点须为映射")
    as_of = _pdate(plan.get("as_of"), "as_of")
    phd = _pdate(plan.get("phd_conferral"), "phd_conferral")
    window = plan.get("window_years")
    if window is not None and (not isinstance(window, (int, float)) or window < 0):
        raise RopeError(f"window_years 须为非负数值: {window!r}")

    blocked = []
    if as_of is None:
        raise RopeError("as_of 必填（资格基准日，如 scheme census/deadline）")
    if phd is None:
        blocked.append("phd_conferral: [TO SET]")

    breakdown = []
    total_credit = 0.0
    for i, itp in enumerate(plan.get("interruptions") or []):
        f = _pdate(itp.get("from"), f"interruptions[{i}].from")
        t = _pdate(itp.get("to"), f"interruptions[{i}].to")
        frac = itp.get("fte_fraction", 0.0)
        if not isinstance(frac, (int, float)) or not (0.0 <= frac <= 1.0):
            raise RopeError(f"interruptions[{i}].fte_fraction 须在 [0,1]: {frac!r}")
        if f is None or t is None:
            blocked.append(f"interruptions[{i}]: date [TO SET]")
            continue
        if t < f:
            raise RopeError(f"interruptions[{i}] to<from: {t}<{f}")
        dur = _years(f, t)
        credit = dur * (1.0 - frac)
        total_credit += credit
        breakdown.append({"from": str(f), "to": str(t), "fte_fraction": frac,
                          "duration_years": round(dur, 3), "credit_years": round(credit, 3),
                          "reason": itp.get("reason")})

    raw = None if phd is None else _years(phd, as_of)
    effective = None if raw is None else raw - total_credit
    # eligible ∈ True | False | "borderline" | None. Within BORDERLINE_MARGIN of the window the
    # verdict is date-noise-sensitive (365.25 approx, census-day convention) — do NOT hard-assert;
    # flag borderline and route to an exact day-count against the scheme's census-date rule.
    eligible = None
    if effective is not None and window is not None:
        gap = float(window) - effective          # >0 = inside window, <0 = over
        if abs(gap) <= BORDERLINE_MARGIN:
            eligible = "borderline"
        else:
            eligible = gap > 0

    return {"as_of": str(as_of),
            "phd_conferral": None if phd is None else str(phd),
            "raw_years_since_phd": None if raw is None else round(raw, 3),
            "total_interruption_credit_years": round(total_credit, 3),
            "effective_years_since_phd": None if effective is None else round(effective, 3),
            "window_years": window,
            "eligible": eligible,
            "interruptions": breakdown}, blocked


def render(res, blocked):
    L = ["== ROPE-time / eligibility ==",
         f"  as_of: {res['as_of']}   PhD conferral: {res['phd_conferral'] or '[TO SET]'}"]
    if res["raw_years_since_phd"] is not None:
        L.append(f"  raw years since PhD: {res['raw_years_since_phd']}")
    if res["interruptions"]:
        L.append("  interruptions (credit = duration × (1 − fte_fraction)):")
        for b in res["interruptions"]:
            L.append(f"    {b['from']}→{b['to']}  fte {b['fte_fraction']}  "
                     f"dur {b['duration_years']}y  credit {b['credit_years']}y"
                     + (f"  ({b['reason']})" if b['reason'] else ""))
    L.append(f"  total interruption credit: {res['total_interruption_credit_years']}y")
    if res["effective_years_since_phd"] is not None:
        L.append(f"  EFFECTIVE years since PhD: {res['effective_years_since_phd']}"
                 + (f"   window: ≤{res['window_years']}y" if res['window_years'] is not None else ""))
    if res["eligible"] == "borderline":
        L.append(f"  ELIGIBLE: BORDERLINE (within ~{BORDERLINE_MARGIN}y of the window) — "
                 f"verify the exact day-count against the scheme's census-date rule before relying on it")
    elif res["eligible"] is True:
        L.append("  ELIGIBLE: YES")
    elif res["eligible"] is False:
        L.append("  ELIGIBLE: NO — over the window")
    elif res["window_years"] is None:
        L.append("  (no window_years supplied — computed effective years only, no eligibility verdict)")
    if blocked:
        L.append("== [TO SET] (fail-closed) ==")
        L.extend(f"  ! {b}" for b in blocked)
    return "\n".join(L)


def run(plan):
    return compute(plan)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan", nargs="?", help="rope-plan.yaml")
    ap.add_argument("-o", "--out")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.plan:
        ap.error("需要 rope-plan.yaml (或 --self-test)")
    import yaml
    plan = yaml.safe_load(open(args.plan, encoding="utf-8")) or {}
    try:
        res, blocked = run(plan)
    except RopeError as exc:
        print(f"ROPE ERROR (fail-closed): {exc}", file=sys.stderr)
        return 2
    print(render(res, blocked))
    if args.out:
        import yaml
        yaml.safe_dump(res, open(args.out, "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
        print(f"\nwrote {args.out}")
    # exit 1 on any review trigger: missing data, over-window, or borderline. A complete run with
    # no window_years (no verdict requested) or a clean eligible=True → exit 0.
    return 1 if (blocked or res["eligible"] is False or res["eligible"] == "borderline") else 0


def self_test():
    # 5.0y raw, 1y part-time@0.5 → credit 0.5 → effective 4.5 ≤ 5 → eligible
    r, b = run({"as_of": "2025-01-01", "phd_conferral": "2020-01-01", "window_years": 5,
                "interruptions": [{"from": "2021-01-01", "to": "2022-01-01", "fte_fraction": 0.5}]})
    assert abs(r["raw_years_since_phd"] - 5.0) < 0.02, r
    assert abs(r["total_interruption_credit_years"] - 0.5) < 0.02, r
    assert abs(r["effective_years_since_phd"] - 4.5) < 0.02, r
    assert r["eligible"] is True, r
    assert not b

    # 7y raw, no interruption → effective 7 > 5 → NOT eligible
    r2, _ = run({"as_of": "2027-01-01", "phd_conferral": "2020-01-01", "window_years": 5})
    assert r2["eligible"] is False, r2

    # full interruption (fte 0) credits full duration: 6y raw − 1y full → effective ≈ 5.0 exactly on the
    # window boundary → BORDERLINE (date-noise: don't hard-assert within ~18 days of the cutoff)
    r3, _ = run({"as_of": "2026-01-01", "phd_conferral": "2020-01-01", "window_years": 5,
                 "interruptions": [{"from": "2021-01-01", "to": "2022-01-01", "fte_fraction": 0.0}]})
    assert r3["eligible"] == "borderline", r3
    # clearly inside after a full interruption: 6y raw − 2y full → effective ≈ 4.0 ≤ 5 → eligible
    r3b, _ = run({"as_of": "2026-01-01", "phd_conferral": "2020-01-01", "window_years": 5,
                  "interruptions": [{"from": "2021-01-01", "to": "2023-01-01", "fte_fraction": 0.0}]})
    assert r3b["eligible"] is True, r3b

    # missing phd → [TO SET] blocked, no eligibility verdict
    r4, b4 = run({"as_of": "2026-01-01", "window_years": 5})
    assert r4["eligible"] is None and any("phd_conferral" in x for x in b4), (r4, b4)

    # interruption missing a date → [TO SET]
    _, b5 = run({"as_of": "2026-01-01", "phd_conferral": "2020-01-01", "window_years": 5,
                 "interruptions": [{"from": "2021-01-01", "fte_fraction": 0.0}]})
    assert any("interruptions[0]" in x for x in b5), b5

    # structural: to<from → RopeError
    try:
        run({"as_of": "2026-01-01", "phd_conferral": "2020-01-01", "window_years": 5,
             "interruptions": [{"from": "2022-01-01", "to": "2021-01-01", "fte_fraction": 0}]})
    except RopeError:
        pass
    else:
        raise AssertionError("to<from 必须 RopeError")

    # structural: bad fte_fraction, bad date, missing as_of
    for bad in ({"as_of": "2026-01-01", "phd_conferral": "2020-01-01", "interruptions": [{"from": "2021-01-01", "to": "2022-01-01", "fte_fraction": 2}]},
                {"as_of": "not-a-date", "phd_conferral": "2020-01-01"},
                {"phd_conferral": "2020-01-01", "window_years": 5}):
        try:
            run(bad)
        except RopeError:
            pass
        else:
            raise AssertionError(f"应 RopeError: {bad}")

    print("self-test OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
