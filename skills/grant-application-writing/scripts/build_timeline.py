# /// script
# requires-python = ">=3.9"
# dependencies = ["pyyaml"]
# ///
"""时间线构建器 (Stage B3: schedule COSTING 的姊妹) — 与 build_budget 同族，FAIL-CLOSED.

build_budget 从人给的率**算钱**；本脚本从 project-plan 的 traceability spine **算排期**：
`tasks[].years[]` + `depends_on[]` + `foundational` → (1) task×year 的 Gantt 网格，
(2) 依赖排序的里程碑列表（每个里程碑挂它的 output/validation）。产出喂 work-plan 字段 / docx。

设计公理（同 build_budget）：**只做机械推导，不臆造日期**。纯粹在既有 spine id 上解析——
`depends_on` 已给出顺序，`years[]` 已给出时段；脚本不发明月份、不猜时长。没给 `years` 的 task
= `[TO SET]`。依赖成环 / 悬垂 = 结构错误（exit 2）。给了 `start_year` 才把相对年映射到日历年。

输入：直接读 `project-plan.yaml`（spine：tasks / outputs / validations）。可选 `--start-year 2027`。

用法:
    uv run build_timeline.py project-plan.yaml [--start-year 2027] [-o milestones.yaml]
    uv run build_timeline.py --self-test

退出码（对齐 build_budget/validate_*）:
    0 = 完整、无 [TO SET]
    1 = rule-level：某 task 缺 years（[TO SET]）
    2 = 结构错误：依赖成环 / depends_on 悬垂 / 缺 tasks
"""
import argparse
import sys

class PlanError(ValueError):
    """结构错误（成环 / 悬垂 / 缺 tasks）→ exit 2。"""


def _toposort(tasks):
    """Kahn 拓扑排序；成环 → PlanError。返回 task id 的依赖序。"""
    ids = [t["id"] for t in tasks]
    idset = set(ids)
    deps = {}
    for t in tasks:
        for d in (t.get("depends_on") or []):
            if d not in idset:
                raise PlanError(f"task {t['id']} 的 depends_on 悬垂: {d!r}")
        deps[t["id"]] = list(t.get("depends_on") or [])
    order, ready = [], [i for i in ids if not deps[i]]
    # 稳定：按原顺序
    ready.sort(key=ids.index)
    seen = set(ready)
    while ready:
        n = ready.pop(0)
        order.append(n)
        for i in ids:
            if i in seen:
                continue
            if all(d in order for d in deps[i]):
                ready.append(i)
                seen.add(i)
        ready.sort(key=ids.index)
    if len(order) != len(ids):
        cyc = [i for i in ids if i not in order]
        raise PlanError(f"depends_on 成环，无法排期: {cyc}")
    return order


def build(plan, start_year=None):
    if not isinstance(plan, dict):
        raise PlanError("project-plan 根节点须为映射")
    tasks = plan.get("tasks") or []
    if not tasks:
        raise PlanError("project-plan 无 tasks（spine 缺失）")
    order = _toposort(tasks)
    tby = {t["id"]: t for t in tasks}
    outs_by_task = {}
    for o in (plan.get("outputs") or []):
        outs_by_task.setdefault(o.get("task"), []).append(o.get("id"))
    val_by_task = {v.get("task"): v.get("id") for v in (plan.get("validations") or [])}

    all_years = sorted({y for t in tasks for y in (t.get("years") or [])})
    blocked = []
    rows, milestones = [], []
    for tid in order:
        t = tby[tid]
        yrs = t.get("years") or []
        if not yrs:
            blocked.append(f"{tid}: no years ([TO SET])")
        span = {}
        for y in all_years:
            span[y] = "█" if y in yrs else " "
        cal = None
        if start_year is not None and yrs:
            cal = [start_year + (y - 1) for y in yrs]
        rows.append({"id": tid, "objective": t.get("objective"),
                     "foundational": bool(t.get("foundational")),
                     "depends_on": t.get("depends_on") or [], "years": yrs,
                     "calendar_years": cal, "span": span,
                     "statement": t.get("statement", "")})
        # 里程碑挂在该 task 的最后一年（完成点）
        if yrs:
            done_y = max(yrs)
            done_cal = (start_year + done_y - 1) if start_year is not None else None
            milestones.append({"task": tid, "year": done_y, "calendar_year": done_cal,
                               "outputs": outs_by_task.get(tid, []),
                               "validation": val_by_task.get(tid),
                               "deliverable": t.get("statement", "")})
    milestones.sort(key=lambda m: (m["year"], order.index(m["task"])))
    return all_years, rows, milestones, blocked


def render(all_years, rows, milestones, blocked, start_year):
    lines = ["== Gantt (task × project-year) =="]
    hdr = "  " + " ".join(f"Y{y}" for y in all_years)
    lines.append(f"  {'task':<10} {hdr}")
    for r in rows:
        bar = "  ".join(r["span"][y] for y in all_years)
        tag = "◆" if r["foundational"] else " "
        dep = f"  ⟵ {','.join(r['depends_on'])}" if r["depends_on"] else ""
        lines.append(f"  {r['id']:<10}{tag}  {bar}{dep}")
    lines.append("== Milestones (dependency-ordered) ==")
    for m in milestones:
        yr = f"Y{m['year']}" + (f" ({m['calendar_year']})" if m["calendar_year"] else "")
        outs = f"  outputs: {', '.join(m['outputs'])}" if m["outputs"] else ""
        val = f"  validation: {m['validation']}" if m["validation"] else ""
        lines.append(f"  [{yr}] {m['task']}: {m['deliverable'][:80]}{outs}{val}")
    if start_year:
        lines.append(f"  (project-year 1 = {start_year})")
    if blocked:
        lines.append("== [TO SET] ==")
        lines.extend(f"  ! {b}" for b in blocked)
    return "\n".join(lines)


def run(plan, start_year=None):
    all_years, rows, milestones, blocked = build(plan, start_year)
    return all_years, rows, milestones, blocked


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan", nargs="?", help="project-plan.yaml 路径")
    ap.add_argument("--start-year", type=int, default=None, help="相对年→日历年映射（project-year 1）")
    ap.add_argument("-o", "--out", help="写出 milestones.yaml")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.plan:
        ap.error("需要 project-plan.yaml 路径 (或 --self-test)")
    import yaml
    plan = yaml.safe_load(open(args.plan, encoding="utf-8")) or {}
    try:
        all_years, rows, milestones, blocked = run(plan, args.start_year)
    except PlanError as exc:
        print(f"PLAN ERROR (fail-closed): {exc}", file=sys.stderr)
        return 2
    print(render(all_years, rows, milestones, blocked, args.start_year))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            yaml.safe_dump({"gantt": rows, "milestones": milestones}, fh,
                           sort_keys=False, allow_unicode=True, default_flow_style=False)
        print(f"\nwrote {args.out}")
    return 1 if blocked else 0


def self_test():
    # 基本：3 task，依赖 t3←t1,t2；拓扑序 + 里程碑
    plan = {"tasks": [
        {"id": "t1", "objective": "o1", "foundational": True, "depends_on": [], "years": [1, 2], "statement": "found"},
        {"id": "t2", "objective": "o2", "foundational": True, "depends_on": [], "years": [1], "statement": "found2"},
        {"id": "t3", "objective": "o3", "depends_on": ["t1", "t2"], "years": [2, 3], "statement": "integrate"}],
        "outputs": [{"id": "out3", "task": "t3"}],
        "validations": [{"id": "val3", "task": "t3"}]}
    all_years, rows, ms, blocked = run(plan, start_year=2027)
    assert all_years == [1, 2, 3], all_years
    assert [r["id"] for r in rows].index("t3") > [r["id"] for r in rows].index("t1"), "t3 在 t1 之后"
    m3 = {m["task"]: m for m in ms}["t3"]
    assert m3["year"] == 3 and m3["calendar_year"] == 2029, m3
    assert m3["outputs"] == ["out3"] and m3["validation"] == "val3", m3
    assert not blocked

    # 缺 years → [TO SET] + blocked
    _, _, _, b2 = run({"tasks": [{"id": "t1", "depends_on": [], "statement": "x"}]})
    assert any("t1:" in b for b in b2), b2

    # 成环 → PlanError
    try:
        run({"tasks": [{"id": "a", "depends_on": ["b"], "years": [1]},
                       {"id": "b", "depends_on": ["a"], "years": [1]}]})
    except PlanError:
        pass
    else:
        raise AssertionError("成环必须 PlanError")

    # 悬垂 depends_on → PlanError
    try:
        run({"tasks": [{"id": "a", "depends_on": ["ghost"], "years": [1]}]})
    except PlanError:
        pass
    else:
        raise AssertionError("悬垂 depends_on 必须 PlanError")

    # 无 tasks → PlanError
    try:
        run({"outputs": []})
    except PlanError:
        pass
    else:
        raise AssertionError("无 tasks 必须 PlanError")

    print("self-test OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
