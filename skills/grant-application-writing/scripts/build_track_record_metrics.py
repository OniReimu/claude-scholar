# /// script
# requires-python = ">=3.9"
# dependencies = ["pyyaml"]
# ///
"""履历指标引擎 (Stage B: track-record metrics) — narrative-award 的评分实体，FAIL-CLOSED.

DECRA / NHMRC Investigator / ERC 判的是**算出来的**履历指标，但此前 outputs_context 全靠手写。
本脚本从 publications[]/funding[] 算：机会校正后的 **pubs/year**、**venue-tier 分布**、
**M-of-N 分母**（"N 篇里 M 篇在顶级 venue"）、以及 **career-best 排名候选**（辅助排序，最终选
由人定）。是 build_budget 的 narrative-award 姊妹。

设计公理（同族 builder）：**只做算术 + 查表排序，不臆造**。venue 不在 tier 表 → tier=[TO SET]
（绝不猜档次）；缺 year/venue → [TO SET]。venue-tier 排名**逐年会变=instance data**，由 --tiers
供，不硬编（同薪资费率表那套）。机会分母用 ROPE interruption 折抵（可接 build_rope_time 结果）。

输入 `track-record-plan.yaml`:
    window: {career_start: 2018, as_of: 2026, interruption_years: 1.2}  # 机会窗（interruption 折抵）
    publications:
      - {id: p1, year: 2020, venue: "NeurIPS", role: first, citations: 45}
      - {id: p2, year: 2021, venue: "ICML",    role: corresponding, citations: 30}
    funding:
      - {id: g1, year: 2021, amount: 300000, role: lead}
    career_best_n: 10
--tiers `tier-table.yaml`:
    tier_ranks: {"A*": 3, A: 2, B: 1, C: 0}
    role_ranks: {first: 3, corresponding: 3, co-first: 2, CI: 1, co-author: 0}
    venues: {NeurIPS: "A*", ICML: "A*", ICLR: "A*", AAAI: A}   # venue → tier（normalized 查表）
    top_tier: "A*"                                             # M-of-N 的分子档

用法:
    uv run build_track_record_metrics.py track-record-plan.yaml --tiers tier-table.yaml [-o metrics.yaml]
    uv run build_track_record_metrics.py --self-test

退出码: 0 = 完整; 1 = 存在 [TO SET]（未知 tier / 缺字段）; 2 = 结构错误（畸形 / 机会窗<=0）。
"""
import argparse
import re
import sys


class TRError(ValueError):
    """结构错误 → exit 2。"""


def _norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _int(v, name):
    if v is None:
        return None
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        raise TRError(f"{name} 需为数值或 null: {v!r}")
    return float(v)


def compute(plan, tiers):
    if not isinstance(plan, dict):
        raise TRError("track-record-plan 根节点须为映射")
    tiers = tiers or {}
    tier_ranks = tiers.get("tier_ranks") or {}
    role_ranks = tiers.get("role_ranks") or {}
    venue_map = {_norm(k): v for k, v in (tiers.get("venues") or {}).items()}
    top_tier = tiers.get("top_tier")

    w = plan.get("window") or {}
    cs, as_of = _int(w.get("career_start"), "career_start"), _int(w.get("as_of"), "as_of")
    interr = _int(w.get("interruption_years"), "interruption_years") or 0.0
    blocked = []
    eff_years = None
    if cs is None or as_of is None:
        blocked.append("window.career_start/as_of: [TO SET]")
    else:
        eff_years = (as_of - cs) - interr
        if eff_years <= 0:
            raise TRError(f"机会窗 effective years <= 0 (as_of {as_of} - start {cs} - interr {interr})")

    pubs = plan.get("publications") or []
    rows = []
    for p in pubs:
        pid = p.get("id") or "<pub>"
        venue, year = p.get("venue"), p.get("year")
        cites = _int(p.get("citations"), f"{pid}.citations")
        tier = venue_map.get(_norm(venue)) if venue else None
        if not venue or year is None:
            blocked.append(f"{pid}: venue/year [TO SET]")
        elif tier is None:
            blocked.append(f"{pid}: venue {venue!r} not in tier table [TO SET]")
        rows.append({"id": pid, "year": year, "venue": venue, "tier": tier,
                     "role": p.get("role"), "citations": cites,
                     "tier_rank": tier_ranks.get(tier, 0) if tier else -1,
                     "role_rank": role_ranks.get(p.get("role"), 0)})

    n = len(rows)
    pubs_per_year = (n / eff_years) if eff_years else None
    # tier distribution
    dist = {}
    for r in rows:
        k = r["tier"] or "[TO SET]"
        dist[k] = dist.get(k, 0) + 1
    # M-of-N against the declared top tier
    m_of_n = None
    if top_tier is not None and n:
        M = sum(1 for r in rows if r["tier"] == top_tier)
        m_of_n = {"M": M, "N": n, "tier": top_tier}
    # career-best ranking: lexicographic (tier, citations, role, recency) — transparent, no arbitrary weights
    ranked = sorted(rows, key=lambda r: (r["tier_rank"], r["citations"] or -1,
                                         r["role_rank"], r["year"] or -1), reverse=True)
    best_n = plan.get("career_best_n", 10)
    career_best = [r["id"] for r in ranked[:best_n]]

    # funding
    fund = plan.get("funding") or []
    f_total = sum(_int(g.get("amount"), f"{g.get('id')}.amount") or 0 for g in fund)
    f_as_lead = sum(1 for g in fund if str(g.get("role", "")).lower() in ("lead", "ci", "pi", "first"))

    return {
        "effective_years": round(eff_years, 2) if eff_years else None,
        "n_publications": n,
        "pubs_per_year": round(pubs_per_year, 2) if pubs_per_year else None,
        "tier_distribution": dist,
        "m_of_n": m_of_n,
        "career_best_candidates": career_best,     # ASSISTIVE ranking; final pick is human/agent
        "funding_total": f_total, "funding_count": len(fund), "funding_as_lead": f_as_lead,
        "ranked_detail": ranked,
    }, blocked


def render(res, blocked):
    L = ["== Track-record metrics =="]
    if res["effective_years"] is not None:
        L.append(f"  opportunity window (interruption-adjusted): {res['effective_years']} years")
    L.append(f"  publications: {res['n_publications']}" +
             (f"   rate: {res['pubs_per_year']}/yr (relative to opportunity)" if res['pubs_per_year'] else ""))
    L.append("  tier distribution: " + ", ".join(f"{k}:{v}" for k, v in res["tier_distribution"].items()))
    if res["m_of_n"]:
        mn = res["m_of_n"]
        L.append(f"  M-of-N: {mn['M']} of {mn['N']} in {mn['tier']}  (honest denominator for a 'top-venue' claim)")
    L.append(f"  funding: {res['funding_count']} grant(s), total {res['funding_total']:.0f}, "
             f"{res['funding_as_lead']} as lead/CI")
    L.append("  career-best CANDIDATES (assistive ranking — final selection is the applicant's):")
    for r in res["ranked_detail"][:res.get("_best_n", 10) if res.get("_best_n") else 10]:
        if r["id"] not in res["career_best_candidates"]:
            continue
        L.append(f"    {r['id']}: {r['venue']} ({r['tier'] or '[TO SET]'}) {r['year']} "
                 f"role={r['role']} cites={r['citations']}")
    if blocked:
        L.append("== [TO SET] (fail-closed) ==")
        L.extend(f"  ! {b}" for b in blocked)
    return "\n".join(L)


def run(plan, tiers=None):
    return compute(plan, tiers)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan", nargs="?")
    ap.add_argument("--tiers", help="venue tier-table yaml (instance data; rankings change)")
    ap.add_argument("-o", "--out")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.plan:
        ap.error("需要 track-record-plan.yaml (或 --self-test)")
    import yaml
    plan = yaml.safe_load(open(args.plan, encoding="utf-8")) or {}
    tiers = yaml.safe_load(open(args.tiers, encoding="utf-8")) if args.tiers else {}
    try:
        res, blocked = run(plan, tiers)
    except TRError as exc:
        print(f"TRACK-RECORD ERROR (fail-closed): {exc}", file=sys.stderr)
        return 2
    print(render(res, blocked))
    if args.out:
        import yaml
        res.pop("ranked_detail", None)
        yaml.safe_dump(res, open(args.out, "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
        print(f"\nwrote {args.out}")
    return 1 if blocked else 0


def self_test():
    tiers = {"tier_ranks": {"A*": 3, "A": 2, "B": 1},
             "role_ranks": {"first": 3, "corresponding": 3, "CI": 1},
             "venues": {"NeurIPS": "A*", "ICML": "A*", "AAAI": "A"},
             "top_tier": "A*"}
    plan = {"window": {"career_start": 2018, "as_of": 2026, "interruption_years": 1.0},
            "publications": [
                {"id": "p1", "year": 2020, "venue": "NeurIPS", "role": "first", "citations": 40},
                {"id": "p2", "year": 2021, "venue": "AAAI", "role": "CI", "citations": 10},
                {"id": "p3", "year": 2022, "venue": "ICML", "role": "corresponding", "citations": 55}],
            "funding": [{"id": "g1", "year": 2021, "amount": 300000, "role": "lead"}],
            "career_best_n": 2}
    res, b = run(plan, tiers)
    assert res["effective_years"] == 7.0, res                       # (2026-2018)-1
    assert res["n_publications"] == 3 and res["pubs_per_year"] == round(3/7.0, 2), res
    assert res["tier_distribution"] == {"A*": 2, "A": 1}, res
    assert res["m_of_n"] == {"M": 2, "N": 3, "tier": "A*"}, res
    # ranking: p3 (A*, 55 cites) > p1 (A*, 40) > p2 (A, 10); top-2 = p3, p1
    assert res["career_best_candidates"] == ["p3", "p1"], res["career_best_candidates"]
    assert res["funding_total"] == 300000 and res["funding_as_lead"] == 1, res
    assert not b, b

    # unknown venue → tier [TO SET], flagged, exits-worthy
    res2, b2 = run({"window": {"career_start": 2018, "as_of": 2026},
                    "publications": [{"id": "px", "year": 2023, "venue": "Obscure Workshop", "citations": 1}]}, tiers)
    assert res2["tier_distribution"] == {"[TO SET]": 1}, res2
    assert any("not in tier table" in x for x in b2), b2

    # missing window → [TO SET], no rate
    res3, b3 = run({"publications": [{"id": "p", "year": 2020, "venue": "NeurIPS"}]}, tiers)
    assert res3["pubs_per_year"] is None and any("window" in x for x in b3), (res3, b3)

    # structural: effective years <= 0
    try:
        run({"window": {"career_start": 2025, "as_of": 2026, "interruption_years": 3}}, tiers)
    except TRError:
        pass
    else:
        raise AssertionError("effective years <=0 必须 TRError")

    print("self-test OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
