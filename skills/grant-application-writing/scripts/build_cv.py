# /// script
# requires-python = ">=3.9"
# dependencies = ["pyyaml"]
# ///
"""长度自适应 CV 组装 (Stage B → Stage D: length-adaptive CV) — FAIL-CLOSED.

会议点 #64/65："各个地方的 CV 要求可能不一样，有的 2 页有的 6 页……有一个自己的 profile
文件夹，让 AI 根据要求去生成。"本脚本把 applicant-profile 的 bio/admin 骨架 + evidence-store
的研究记录（publications/funding/supervision/awards/service）**按目标长度选取、排序、组装**成一份
Markdown CV。

设计公理（同族 builder）：**只 SELECT + ORDER + FORMAT 真实数据，绝不臆造 prose**。缺 name/
position = `[TO SET]`（不编造）。裁剪永不静默——每个 section 报"显示 N / 共 M（裁 K）"，符合
"no silent caps"纪律：一份 2 页 CV 隐藏了 20 篇论文，读起来像"就这些"，必须说清裁了多少。

长度由 applicant-profile 的 `cv_config.length_profiles[<name>]`（每 section 的条数上限）驱动——
这是 **instance data**（每人/每方案不同），不硬编在脚本里。脚本只做"取上限条、按 key 排序"。

输入:
  applicant-profile.yaml   bio/admin 骨架 + cv_config（section_order + length_profiles）
  --evidence evidence-store.yaml   研究记录（publications/funding/supervision/awards/service）
  --profile-name 2pp | --pages 2   选一个 length profile（--pages N → 名为 "<N>pp" 的 profile）
  --tiers tier-table.yaml          可选：venue→tier，用于 publications 排序（缺则按年份 recency）

用法:
  uv run build_cv.py applicant-profile.yaml --evidence evidence-store.yaml --profile-name 2pp -o cv-2pp.md
  uv run build_cv.py --self-test

退出码: 0 = 组装完整; 1 = rule-level（缺 name/position 等 [TO SET]）; 2 = 结构错误（无 cv_config /
未知 length profile）。
"""
import argparse
import sys

# section → evidence-store 的 bucket 名（appointments/education 来自 profile 本身）
_EVIDENCE_SECTIONS = {"publications", "funding", "supervision", "awards", "service"}
_PROFILE_SECTIONS = {"appointments", "education"}
_KNOWN_SECTIONS = _EVIDENCE_SECTIONS | _PROFILE_SECTIONS
_DEFAULT_ORDER = ["appointments", "education", "publications", "funding",
                  "supervision", "awards", "service"]


class CVError(ValueError):
    """结构错误（无 cv_config / 未知 length profile / 畸形）→ exit 2。"""


def _year(item):
    """从一行里尽力取一个可比较的年份（int），取不到给 0（排到最后）。"""
    for k in ("year", "end_year"):
        v = item.get(k)
        if isinstance(v, int):
            return v
    for k in ("period", "validity_window"):
        blk = item.get(k)
        if isinstance(blk, dict):
            for kk in ("to", "from"):
                s = str(blk.get(kk) or "")
                if s[:4].isdigit():
                    return int(s[:4])
    return 0


def _tier_rank(item, tier_map, tier_order):
    """venue → tier → 排名分（越大越靠前）；无 tiers 表或未命中 → 0。"""
    if not tier_map:
        return 0
    tier = tier_map.get(item.get("venue"))
    if tier is None or tier not in tier_order:
        return 0
    return len(tier_order) - tier_order.index(tier)   # 表头 tier 分最高


def _sort_key(section, tier_map, tier_order):
    """每 section 的排序 key（降序）：publications 先 tier 后年份；funding 先金额后年份；
    其余按年份 recency。"""
    if section == "publications":
        return lambda it: (_tier_rank(it, tier_map, tier_order), _year(it))
    if section == "funding":
        return lambda it: (float((it.get("amount") or {}).get("value", 0) or 0), _year(it))
    return lambda it: (_year(it),)


def _fmt(section, it):
    """一行 → 一条 Markdown 列表项（用真实字段，缺字段就略过，不填充占位）。"""
    if section == "publications":
        return it.get("citation") or f"{it.get('venue', '?')} ({_year(it) or 'n.d.'})"
    if section == "funding":
        amt = (it.get("amount") or {})
        money = f"{amt.get('currency', '')}{amt.get('value')}".strip() if amt.get("value") is not None else ""
        bits = [it.get("title"), it.get("scheme"), it.get("role"), money]
        return " — ".join(b for b in bits if b)
    if section == "supervision":
        bits = [it.get("level"), it.get("role"), it.get("status"), str(_year(it) or "")]
        return " — ".join(b for b in bits if b)
    if section == "awards":
        bits = [it.get("title"), it.get("awarding_body"), str(_year(it) or "")]
        return " — ".join(b for b in bits if b)
    if section == "service":
        bits = [it.get("kind"), it.get("detail")]
        return " — ".join(b for b in bits if b)
    if section == "appointments":
        p = it.get("period") or {}
        span = f"{p.get('from', '?')}–{p.get('to') or 'present'}"
        bits = [it.get("title"), it.get("org"), span]
        return " — ".join(b for b in bits if b)
    if section == "education":
        bits = [it.get("degree"), it.get("institution"), str(it.get("year") or "")]
        return " — ".join(b for b in bits if b)
    return str(it)


_HEADINGS = {
    "appointments": "Appointments", "education": "Education",
    "publications": "Selected Publications", "funding": "Grants & Funding",
    "supervision": "Research Supervision", "awards": "Awards & Honours",
    "service": "Professional Service",
}


def compute(profile, evidence, profile_name, tiers=None):
    if not isinstance(profile, dict):
        raise CVError("applicant-profile 根节点须为映射")
    app = profile.get("applicant") or {}
    cfg = profile.get("cv_config") or {}
    profiles = cfg.get("length_profiles") or {}
    if not profiles:
        raise CVError("applicant-profile 无 cv_config.length_profiles（长度上限是 instance data，必须提供）")
    if profile_name not in profiles:
        raise CVError(f"未知 length profile {profile_name!r}；可用: {sorted(profiles)}")
    caps = profiles[profile_name] or {}
    order = [s for s in (cfg.get("section_order") or _DEFAULT_ORDER) if s in _KNOWN_SECTIONS]

    tier_map = (tiers or {}).get("venue_tier") or {}
    tier_order = (tiers or {}).get("tier_order") or []
    evidence = evidence or {}

    blocked = []
    name = app.get("name")
    title = app.get("title")
    if not name:
        blocked.append("applicant.name [TO SET]")
    if not title:
        blocked.append("applicant.title [TO SET]")

    sections = []
    for sec in order:
        src = app.get(sec) if sec in _PROFILE_SECTIONS else evidence.get(sec)
        items = [it for it in (src or []) if isinstance(it, dict)]
        if not items:
            sections.append({"section": sec, "shown": [], "total": 0, "trimmed": 0})
            continue
        items.sort(key=_sort_key(sec, tier_map, tier_order), reverse=True)
        cap = caps.get(sec)
        cap = int(cap) if isinstance(cap, (int, float)) else len(items)
        shown = items[:cap]
        sections.append({"section": sec, "shown": shown,
                         "total": len(items), "trimmed": max(0, len(items) - len(shown))})
    return {"name": name, "title": title, "orcid": app.get("orcid"),
            "profile_name": profile_name, "sections": sections, "blocked": blocked}


def render_md(cv):
    """组装成 Markdown CV。"""
    L = [f"# {cv['name'] or '[TO SET — applicant.name]'}"]
    sub = " · ".join(x for x in (cv["title"] or "[TO SET — applicant.title]",
                                 f"ORCID {cv['orcid']}" if cv.get("orcid") else None) if x)
    if sub:
        L.append(f"*{sub}*")
    L.append(f"\n<!-- CV length profile: {cv['profile_name']} -->")
    for s in cv["sections"]:
        if not s["shown"]:
            continue
        L.append(f"\n## {_HEADINGS.get(s['section'], s['section'].title())}")
        if s["trimmed"]:
            L.append(f"<!-- showing {len(s['shown'])} of {s['total']} "
                     f"({s['trimmed']} trimmed to fit {cv['profile_name']}) -->")
        for it in s["shown"]:
            L.append(f"- {_fmt(s['section'], it)}")
    return "\n".join(L)


def render_report(cv):
    """给人看的选取/裁剪汇报（no silent caps）。"""
    L = [f"== CV assembled — length profile: {cv['profile_name']} =="]
    for s in cv["sections"]:
        if s["total"] == 0:
            L.append(f"  {s['section']:<14} (none in store)")
        else:
            tag = f"  ⚠ {s['trimmed']} trimmed" if s["trimmed"] else ""
            L.append(f"  {s['section']:<14} showing {len(s['shown'])} of {s['total']}{tag}")
    if cv["blocked"]:
        L.append("== [TO SET] (fail-closed) ==")
        L.extend(f"  ! {b}" for b in cv["blocked"])
    return "\n".join(L)


def run(profile, evidence, profile_name, tiers=None):
    return compute(profile, evidence, profile_name, tiers)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profile", nargs="?", help="applicant-profile.yaml")
    ap.add_argument("--evidence", help="evidence-store.yaml（研究记录）")
    ap.add_argument("--profile-name", help="length profile 名（如 2pp / 6pp）")
    ap.add_argument("--pages", type=int, help="等价于 --profile-name '<N>pp'")
    ap.add_argument("--tiers", help="tier-table.yaml（venue_tier + tier_order）")
    ap.add_argument("-o", "--out", help="写出 CV 的 .md 路径")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.profile:
        ap.error("需要 applicant-profile.yaml (或 --self-test)")
    pname = args.profile_name or (f"{args.pages}pp" if args.pages else None)
    if not pname:
        ap.error("需要 --profile-name 或 --pages")
    import yaml
    profile = yaml.safe_load(open(args.profile, encoding="utf-8")) or {}
    evidence = yaml.safe_load(open(args.evidence, encoding="utf-8")) or {} if args.evidence else {}
    tiers = yaml.safe_load(open(args.tiers, encoding="utf-8")) or {} if args.tiers else {}
    try:
        cv = run(profile, evidence, pname, tiers)
    except CVError as exc:
        print(f"CV ERROR (fail-closed): {exc}", file=sys.stderr)
        return 2
    print(render_report(cv))
    md = render_md(cv)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(md + "\n")
        print(f"\nwrote {args.out}")
    else:
        print("\n" + md)
    return 1 if cv["blocked"] else 0


def self_test():
    profile = {
        "applicant": {
            "name": "Dr Jane Q. Researcher", "title": "Lecturer",
            "orcid": "0000-0000-0000-0000",
            "appointments": [
                {"title": "Lecturer", "org": "Example Univ", "period": {"from": "2024-01", "to": None}},
                {"title": "Postdoc", "org": "Other Univ", "period": {"from": "2020-01", "to": "2023-12"}}],
            "education": [{"degree": "PhD in CS", "institution": "Example Univ", "year": 2019}],
        },
        "cv_config": {
            "section_order": ["appointments", "education", "publications", "funding", "awards"],
            "length_profiles": {
                "2pp": {"publications": 2, "funding": 1, "awards": 1},
                "6pp": {"publications": 10, "funding": 10, "awards": 10},
            },
        },
    }
    evidence = {
        "publications": [
            {"id": "p1", "citation": "A. 2024. Top result. FlagVenue.", "year": 2024, "venue": "FlagVenue"},
            {"id": "p2", "citation": "A. 2020. Older result. MinorVenue.", "year": 2020, "venue": "MinorVenue"},
            {"id": "p3", "citation": "A. 2022. Mid result. MidVenue.", "year": 2022, "venue": "MidVenue"}],
        "funding": [
            {"id": "g1", "title": "Big Grant", "scheme": "ARC DP", "role": "CI",
             "amount": {"value": 500000, "currency": "AUD"}, "period": {"from": "2023", "to": "2026"}},
            {"id": "g2", "title": "Small Grant", "scheme": "Internal", "role": "PI",
             "amount": {"value": 20000, "currency": "AUD"}, "period": {"from": "2021", "to": "2022"}}],
        "awards": [{"id": "a1", "title": "Best Paper", "awarding_body": "FlagVenue", "year": 2024}],
    }
    tiers = {"tier_order": ["A*", "A", "B"], "venue_tier": {"FlagVenue": "A*", "MidVenue": "B"}}

    # 2pp: publications capped at 2, ordered tier-then-year → FlagVenue(A*) first, then MidVenue(2022) over MinorVenue(2020)
    cv = run(profile, evidence, "2pp", tiers)
    pubs = next(s for s in cv["sections"] if s["section"] == "publications")
    assert len(pubs["shown"]) == 2 and pubs["trimmed"] == 1, pubs
    assert pubs["shown"][0]["id"] == "p1", ("A* venue must rank first", pubs["shown"])
    assert pubs["shown"][1]["id"] == "p3", ("2022 must beat 2020 as second", pubs["shown"])
    assert not cv["blocked"], cv["blocked"]

    # funding capped at 1 → the 500k grant (amount desc) survives, 1 trimmed
    fund = next(s for s in cv["sections"] if s["section"] == "funding")
    assert len(fund["shown"]) == 1 and fund["shown"][0]["id"] == "g1" and fund["trimmed"] == 1, fund

    # render_md carries the trim marker + a heading, and lists exactly the shown items
    md = render_md(cv)
    assert "trimmed to fit 2pp" in md and "## Selected Publications" in md, md
    assert md.count("\n- ") == (2 + 1 + 2 + 1 + 1), ("2 pubs +1 fund +2 appts +1 edu +1 award", md)

    # 6pp: nothing trimmed (caps ≥ counts)
    cv6 = run(profile, evidence, "6pp", tiers)
    assert all(s["trimmed"] == 0 for s in cv6["sections"]), cv6["sections"]

    # no tiers → publications fall back to year recency (2024 > 2022 > 2020)
    cv_nt = run(profile, evidence, "6pp", None)
    pubs_nt = next(s for s in cv_nt["sections"] if s["section"] == "publications")
    assert [p["id"] for p in pubs_nt["shown"]] == ["p1", "p3", "p2"], pubs_nt["shown"]

    # missing name/title → [TO SET], exit-1 signal (blocked non-empty)
    noname = {"applicant": {}, "cv_config": profile["cv_config"]}
    cv_b = run(noname, evidence, "2pp", tiers)
    assert any("name" in b for b in cv_b["blocked"]) and any("title" in b for b in cv_b["blocked"]), cv_b["blocked"]

    # structural: no cv_config → CVError; unknown profile name → CVError
    for bad in ({"applicant": {"name": "x"}}, ):
        try:
            run(bad, evidence, "2pp", tiers)
        except CVError:
            pass
        else:
            raise AssertionError("无 cv_config 必须 CVError")
    try:
        run(profile, evidence, "99pp", tiers)
    except CVError:
        pass
    else:
        raise AssertionError("未知 length profile 必须 CVError")

    print("self-test OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
