# /// script
# requires-python = ">=3.9"
# dependencies = ["pyyaml"]
# ///
"""FTE 投入分配 + 超额检查 (Stage B2: effort allocation) — FAIL-CLOSED.

多 CI 方案（ARC DP/Linkage/FT、NHMRC Investigator、CRC-P）要给每个 CI 报 time commitment，
且一个人跨项目不能 >100% FTE。§2.7(c) 说"对每个 CI 的 fte 查 current_commitments，超额是
可行性 flag"——但没脚本做这个算术。本脚本补上：每个 investigator 的 **本项目 FTE**（直接给，
或从 tasks[].effort 汇总）+ **既有承诺** → **总占用**，超 capacity 即 over-subscription flag。

设计公理（同族 builder）：**只做加法 + 比较，不臆造**。缺 project_fte / current_commitments
= `[TO SET]` → BLOCK。可选从 project-plan `tasks[].effort{person: fte}` 汇总本项目 FTE（若给）。

输入 `effort-plan.yaml`（investigators 可从 entity-store 提升；tasks 可从 project-plan 提升）:
    capacity_default: 1.0
    investigators:
      - id: inv-lead
        project_fte: 0.3          # 本项目 FTE（省则从 tasks[].effort 汇总；两者都缺 = [TO SET]）
        current_commitments: 0.5  # 已在别处承诺的 FTE；null = [TO SET]
        capacity: 1.0             # 总可用 FTE（省则用 capacity_default）
    tasks:                        # 可选：从 spine 汇总本项目 FTE
      - {id: task-1, effort: {inv-lead: 0.1, inv-ci-2: 0.2}}

用法:
    uv run build_effort_allocation.py effort-plan.yaml [-o effort.yaml]
    uv run build_effort_allocation.py --self-test

退出码: 0 = 完整且无人超额; 1 = rule-level（[TO SET] 空值，或有人 over-subscribed）; 2 = 结构错误。
"""
import argparse
import sys


class EffortError(ValueError):
    """结构错误（负值 / 畸形）→ exit 2。"""


def _nn(v, name):
    if v is None:
        return None
    if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
        raise EffortError(f"{name} 须为非负数值或 null: {v!r}")
    return float(v)


def compute(plan):
    if not isinstance(plan, dict):
        raise EffortError("effort-plan 根节点须为映射")
    invs = plan.get("investigators") or []
    if not invs:
        raise EffortError("effort-plan 无 investigators")
    cap_default = _nn(plan.get("capacity_default", 1.0), "capacity_default")

    # 从 tasks[].effort 汇总本项目 FTE（若提供）
    task_fte = {}
    for t in (plan.get("tasks") or []):
        eff = t.get("effort") or {}
        if not isinstance(eff, dict):
            raise EffortError(f"task {t.get('id')} 的 effort 须为 {{person: fte}} 映射")
        for pid, f in eff.items():
            task_fte[pid] = task_fte.get(pid, 0.0) + (_nn(f, f"task {t.get('id')}.effort[{pid}]") or 0.0)

    rows, blocked = [], []
    for iv in invs:
        pid = iv.get("id")
        if not pid:
            raise EffortError("investigator 缺 id")
        proj = _nn(iv.get("project_fte"), f"{pid}.project_fte")
        if proj is None and pid in task_fte:
            proj = task_fte[pid]
        cur = _nn(iv.get("current_commitments"), f"{pid}.current_commitments")
        cap = _nn(iv.get("capacity"), f"{pid}.capacity")
        cap = cap if cap is not None else cap_default

        missing = []
        if proj is None:
            missing.append("project_fte")
        if cur is None:
            missing.append("current_commitments")
        total = None if (proj is None or cur is None) else proj + cur
        over = None if total is None else round(total - cap, 4)
        status = "unknown"
        if total is not None:
            status = "over-subscribed" if over > 1e-9 else "ok"
        rows.append({"id": pid, "project_fte": proj, "current_commitments": cur,
                     "capacity": cap, "total": None if total is None else round(total, 4),
                     "over_by": over, "status": status})
        if missing:
            blocked.append(f"{pid}: {', '.join(missing)} [TO SET]")
    over_any = [r["id"] for r in rows if r["status"] == "over-subscribed"]
    return rows, blocked, over_any


def render(rows, blocked, over_any):
    L = ["== FTE effort allocation =="]
    for r in rows:
        pf = "[TO SET]" if r["project_fte"] is None else f"{r['project_fte']:g}"
        cc = "[TO SET]" if r["current_commitments"] is None else f"{r['current_commitments']:g}"
        tot = "—" if r["total"] is None else f"{r['total']:g}"
        tag = "" if r["status"] != "over-subscribed" else f"  ⚠ OVER by {r['over_by']:g}"
        L.append(f"  {r['id']:<14} project {pf} + current {cc} = {tot} / cap {r['capacity']:g}"
                 f"  [{r['status']}]{tag}")
    if over_any:
        L.append(f"== OVER-SUBSCRIBED: {', '.join(over_any)} (feasibility flag — reduce FTE or re-plan) ==")
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
        ap.error("需要 effort-plan.yaml (或 --self-test)")
    import yaml
    plan = yaml.safe_load(open(args.plan, encoding="utf-8")) or {}
    try:
        rows, blocked, over_any = run(plan)
    except EffortError as exc:
        print(f"EFFORT ERROR (fail-closed): {exc}", file=sys.stderr)
        return 2
    print(render(rows, blocked, over_any))
    if args.out:
        import yaml
        yaml.safe_dump({"allocation": rows}, open(args.out, "w", encoding="utf-8"),
                       sort_keys=False, allow_unicode=True)
        print(f"\nwrote {args.out}")
    return 1 if (blocked or over_any) else 0


def self_test():
    # ok: 0.3 + 0.5 = 0.8 ≤ 1.0
    rows, b, ov = run({"investigators": [{"id": "a", "project_fte": 0.3, "current_commitments": 0.5}]})
    assert rows[0]["status"] == "ok" and not ov and not b, (rows, b, ov)

    # over: 0.6 + 0.6 = 1.2 > 1.0
    rows2, _, ov2 = run({"investigators": [{"id": "a", "project_fte": 0.6, "current_commitments": 0.6}]})
    assert rows2[0]["status"] == "over-subscribed" and abs(rows2[0]["over_by"] - 0.2) < 1e-9 and ov2 == ["a"], rows2

    # capacity override: cap 1.5 → 1.2 ok
    rows3, _, ov3 = run({"investigators": [{"id": "a", "project_fte": 0.6, "current_commitments": 0.6, "capacity": 1.5}]})
    assert rows3[0]["status"] == "ok" and not ov3, rows3

    # project_fte from tasks[].effort aggregation
    rows4, _, _ = run({"investigators": [{"id": "a", "current_commitments": 0.2}],
                       "tasks": [{"id": "t1", "effort": {"a": 0.1}}, {"id": "t2", "effort": {"a": 0.2}}]})
    assert abs(rows4[0]["project_fte"] - 0.3) < 1e-9, rows4

    # missing current_commitments → [TO SET]
    _, b5, _ = run({"investigators": [{"id": "a", "project_fte": 0.3}]})
    assert any("current_commitments" in x for x in b5), b5

    # missing both project_fte and no tasks → [TO SET]
    _, b6, _ = run({"investigators": [{"id": "a", "current_commitments": 0.3}]})
    assert any("project_fte" in x for x in b6), b6

    # structural: negative, no investigators
    try:
        run({"investigators": [{"id": "a", "project_fte": -0.1, "current_commitments": 0.2}]})
    except EffortError:
        pass
    else:
        raise AssertionError("负 fte 必须 EffortError")
    try:
        run({"capacity_default": 1.0})
    except EffortError:
        pass
    else:
        raise AssertionError("无 investigators 必须 EffortError")

    print("self-test OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
