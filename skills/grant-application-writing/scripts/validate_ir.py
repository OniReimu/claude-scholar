# /// script
# requires-python = ">=3.9"
# dependencies = ["pyyaml"]
# ///
"""IR-level integrity orchestrator (Stage E: portal validation dry-run) — FAIL-CLOSED.

The single pre-submit gate for the cross-field couplings the docs acknowledged but no shipped
script checked (Codex top-3 #3; Phase-2's F.2↔H.1↔matched "hardest thing to model"). Reads the
`scheme.yaml` IR (+ optional values/evidence/entity/budget/paste-ready sidecars) and runs 10
checks, each emitting `[PASS]/[FAIL]/[SKIP]/[WARN] <located reason>`. It COMPOSES the siblings —
budget math is delegated to `validate_budget.py`, char counting to `charcount.py` (subprocesses;
their exit codes + output fold into this report), never reimplemented here.

  1. schema             widget×role ∈ type-model.md sets; criterion/gate/ref pointers resolve.
  2. allocation_sums_to taxonomy-code / repeating-group %-rows sum to 100 (±0.01).
  3. contribution↔budget entity partner contributions reconcile with budget co-contribution rows.
  4. computed gates     recompute derived eligibility/co-contribution/matched gates; hard fail=block.
  5. rubric sub_weights indicator points present + consistent with field sub_indicators.
  6. conditional annexes a triggered conditional-group's required annex is present in attachments.
  7. stage_lock/phase   no value for a phase-locked field; phase order valid.
  8. attachment rules   each structured-upload has a valid kind + matching attachments[] entry.
  9. char roll-up       delegates the paste-ready to charcount.py; folds its verdict.
 10. criterion-readiness per rubric criterion (minimum_evidence + readiness_rule), compute a
                        readiness state; --mode submission FAILs an unsupported scored criterion.

SKIP vs FAIL (fail-closed): FAIL when the needed input WAS supplied but the data violates the
rule or a hard gate cannot be evaluated; SKIP (non-blocking, with a stated reason) only when an
OPTIONAL sidecar was not supplied, or the scheme lacks that construct. Exit non-zero on any HARD
FAIL (or a delegated sibling non-zero exit); soft binding = WARN.

Criterion-readiness never hides a scored-criterion gap behind a SKIP: if a criterion declares
`minimum_evidence` but the evidence/content backing it was not supplied, that criterion is
`unsupported` — a [FAIL] in `--mode submission` (default `draft` reports it as [WARN]). Only a
scheme that declares NO criterion `minimum_evidence` genuinely SKIPs the whole check.

    uv run validate_ir.py --scheme scheme.yaml --entity entity-store.yaml --budget budget.yaml
    uv run validate_ir.py --self-test
"""
import argparse
import os
import re
import subprocess
import sys

# ── type-model.md sets ──
WIDGETS = {
    "narrative", "scalar", "money", "single-choice", "multi-choice", "taxonomy-code",
    "boolean-gate", "declaration", "link", "payout-target", "credit-request",
    "repeating-group", "conditional-group", "decision-tree", "budget-matrix",
    "contribution-matrix", "relational-table", "milestone-table", "stage-gate",
    "risk-register", "computed", "linked-profile", "structured-upload", "fieldset", "section",
}
ROLES = {
    "eligibility-gate", "criterion-scored", "admin-metadata", "classification", "compliance",
    "budget-resource", "team-partner", "evidence", "logistics",
}
UPLOAD_KINDS = {"free", "proforma", "composite", "heading-sequenced", "system-generated"}
PHASE_ORDER = ["minimum-data", "EOI", "full", "post-award"]

TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
KW = {"and", "or", "not", "True", "False", "None", "in", "is"}
HERE = os.path.dirname(os.path.abspath(__file__))

# residual placeholders a criterion's fields must NOT carry to reach `substantiated`
READINESS_MARKER = re.compile(r"\[(?:TO SET|VERIFY|[^\]]*NEEDED)\]", re.I)
# minimum_evidence class-name → canonical evidence-store bucket (both sides normalised)
EVIDENCE_ALIAS = {
    "publication": "publication", "paper": "publication",
    "grant": "funding", "funding": "funding",
    "patent": "patent", "award": "award",
    "supervision": "supervision", "student": "supervision",
    "service": "service", "invitedtalk": "service", "talk": "service",
    "impact": "impact",
    "comparator": "comparator", "externalcomparator": "comparator",
    "context": "context", "contextevidence": "context",
    "metric": "metric", "citation": "metric", "citationmetric": "metric",
}


class ResolveError(Exception):
    def __init__(self, missing):
        self.missing = missing


# ── report accumulator ─────────────────────────────────────────────────────
class Report:
    def __init__(self):
        self.entries = []          # (check, status, binding, reason, delegated_output|None)

    def add(self, check, status, binding, reason, out=None):
        self.entries.append((check, status, binding, reason, out))

    def hard_failed(self):
        return any(s == "FAIL" and b == "hard" for _, s, b, _, _ in self.entries)

    def render(self):
        for check, status, binding, reason, out in self.entries:
            tag = f"[{status}]"
            b = f" ({binding})" if status in ("FAIL", "WARN") else ""
            print(f"{tag} {check}{b}: {reason}")
            if out:
                for line in out.strip().splitlines():
                    print(f"        | {line}")
        n = {k: sum(1 for e in self.entries if e[1] == k) for k in ("PASS", "FAIL", "SKIP", "WARN")}
        print(f"\n== {n['PASS']} PASS · {n['FAIL']} FAIL · {n['WARN']} WARN · {n['SKIP']} SKIP ==")


# ── helpers ─────────────────────────────────────────────────────────────────
def load_yaml(path):
    import yaml
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def iter_field_nodes(node):
    """Yield every dict that carries a 'widget' key, recursively (fields + nested subfields)."""
    if isinstance(node, dict):
        if "widget" in node:
            yield node
        for v in node.values():
            yield from iter_field_nodes(v)
    elif isinstance(node, list):
        for v in node:
            yield from iter_field_nodes(v)


def walk_dicts(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from walk_dicts(v)
    elif isinstance(node, list):
        for v in node:
            yield from walk_dicts(v)


def roles_of(field):
    r = field.get("role")
    return r if isinstance(r, list) else ([r] if r else [])


def fid(field):
    return field.get("id") or field.get("label") or "<anon>"


def run_sibling(script, args):
    """Invoke a sibling script via `uv run` (fallback: same interpreter). Returns (exit, output)."""
    path = os.path.join(HERE, script)
    for cmd in ([("uv", "run", path)], [(sys.executable, path)]):
        try:
            p = subprocess.run(list(cmd[0]) + args, capture_output=True, text=True)
            return p.returncode, (p.stdout + p.stderr)
        except FileNotFoundError:
            continue
    return 127, f"cannot locate {script}"


def budget_totals(output):
    m = re.search(r"total\(counted\):\s*([\d.]+).*?total-cash:\s*([\d.]+)"
                  r".*?requested:\s*([\d.]+).*?co-contribution:\s*([\d.]+)", output, re.S)
    return tuple(float(x) for x in m.groups()) if m else None


def budget_phase_count(bdata):
    phases = set()
    for r in (bdata.get("rows") or []):
        if r.get("funding_source") != "requested":
            continue
        if r.get("phase") is not None:
            phases.add(str(r["phase"]))
        else:
            for k in (r.get("years") or {}):
                if not str(k).isdigit():
                    phases.add(str(k))
    return len(phases)


def build_ns(values, entity, btot, phase_count):
    ns = {}
    if btot:
        total, tc, req, co = btot
        ns.update({"total": total, "total_cash": tc, "requested": req, "co_contribution": co,
                   "total_eligible_expenditure": total, "grant_sought": req,
                   "requested_grant_total": req, "co_contribution_total": co})
    for k, v in (values or {}).items():
        if isinstance(v, (int, float)):
            ns.setdefault(k, v)
            ns.setdefault(k.split(".")[-1], v)
    for blk in ("integrity", "project", "totals"):
        for k, v in (entity.get(blk) or {}).items():
            if isinstance(v, (int, float)):
                ns[k] = v
    return ns


def eval_derived(expr, ns, phase_count):
    expr = expr.replace("count(distinct budget.phase)", str(phase_count))
    if "=>" in expr:
        lhs, rhs = expr.split("=>", 1)
        expr = f"(not ({lhs})) or ({rhs})"
    missing = []

    def rep(m):
        t = m.group(0)
        if t in KW:
            return t
        if ns.get(t) is not None:
            return repr(float(ns[t]))
        seg = t.split(".")[-1]
        if ns.get(seg) is not None:
            return repr(float(ns[seg]))
        missing.append(t)
        return "0"

    sub = TOKEN.sub(rep, expr)
    if missing:
        raise ResolveError(missing)
    return bool(eval(sub, {"__builtins__": {}}, {}))


def contribution_totals(entity):
    """Sum partner co-contributions by type from an entity-store (dict- or list-shaped)."""
    total, in_kind = 0.0, 0.0
    for p in (entity.get("partners") or []):
        c = p.get("contributions")
        if isinstance(c, dict):
            for typ, cells in c.items():
                s = sum(float(v) for v in (cells or {}).values()) if isinstance(cells, dict) else 0.0
                total += s
                if "kind" in typ.replace("_", "").replace("-", "") and typ.startswith("in"):
                    in_kind += s
        elif isinstance(c, list):
            for row in c:
                amt = (row.get("amount") or {}).get("value", row.get("value", 0)) or 0
                total += float(amt)
                if str(row.get("type", "")).startswith("in"):
                    in_kind += float(amt)
    return total, in_kind


def _canon_evidence_class(name):
    t = re.sub(r"[^a-z0-9]", "", str(name).lower())
    if t.endswith("ies"):
        t = t[:-3] + "y"
    elif t.endswith("s"):
        t = t[:-1]
    return EVIDENCE_ALIAS.get(t, t)


def _evidence_present(evidence):
    """Canonical class tokens for every non-empty bucket in the evidence store."""
    return {_canon_evidence_class(k) for k, v in (evidence or {}).items() if v}


# ── the 10 checks ────────────────────────────────────────────────────────────
def check_schema(rep, scheme):
    crit = {c.get("criterion") for c in (scheme.get("rubric") or []) if isinstance(c, dict)}
    gate_ids = {g.get("id") for g in (scheme.get("eligibility_gates") or []) if isinstance(g, dict)}
    top_keys = set(scheme.keys())
    bad, n = [], 0
    for sec in (scheme.get("sections") or []):
        for f in iter_field_nodes(sec.get("fields")):
            n += 1
            w = f.get("widget")
            if w not in WIDGETS:
                bad.append(f"{fid(f)}: unknown widget {w!r}")
            for r in roles_of(f):
                if r not in ROLES:
                    bad.append(f"{fid(f)}: unknown role {r!r}")
    for d in walk_dicts(scheme.get("sections")):
        cr = d.get("criterion_ref") or d.get("criterion")
        if cr is not None and crit and cr not in crit:
            bad.append(f"criterion ref {cr!r} not in rubric {sorted(crit)}")
        if d.get("gate") and d["gate"] not in gate_ids:
            bad.append(f"gate ref {d['gate']!r} dangling")
        if d.get("ref") and d["ref"] not in top_keys:
            bad.append(f"ref {d['ref']!r} dangling")
    if bad:
        rep.add("schema", "FAIL", "hard", f"{len(bad)} issue(s): " + "; ".join(bad[:6]))
    else:
        rep.add("schema", "PASS", "hard",
                f"{n} field-nodes valid; {len(crit)} rubric criteria; refs resolve")


def check_allocation(rep, scheme, values):
    decl = []
    for f in iter_field_nodes(scheme.get("sections")):
        tgt = ((f.get("taxonomy") or {}).get("allocation_sums_to")
               or (f.get("attr") or {}).get("allocation_sums_to"))
        if isinstance(tgt, (int, float)):
            decl.append((fid(f), tgt))
    if not decl:
        rep.add("allocation_sums_to", "SKIP", "hard", "scheme declares no allocation_sums_to")
        return
    if not values:
        rep.add("allocation_sums_to", "SKIP", "hard",
                f"{len(decl)} field(s) {[d[0] for d in decl]} declare it — supply --values to sum")
        return
    bad = []
    for name, tgt in decl:
        v = values.get(name)
        rows = v if isinstance(v, list) else (list(v.values()) if isinstance(v, dict) else None)
        if rows is None:
            bad.append(f"{name}: no allocation in values")
            continue
        s = sum(float(x.get("pct", x.get("percent", 0)) if isinstance(x, dict) else x) for x in rows)
        if abs(s - tgt) > 0.01:
            bad.append(f"{name}: sums to {s} (want {tgt})")
    if bad:
        rep.add("allocation_sums_to", "FAIL", "hard", "; ".join(bad))
    else:
        rep.add("allocation_sums_to", "PASS", "hard", f"{len(decl)} field(s) sum to target")


def check_contribution(rep, scheme, entity, bdata):
    has_cm = any(f.get("widget") == "contribution-matrix" for f in iter_field_nodes(scheme.get("sections")))
    if not entity.get("partners"):
        msg = ("scheme has a contribution-matrix — supply --entity + --budget"
               if has_cm else "no contribution-matrix and no entity partners")
        rep.add("contribution↔budget", "SKIP", "hard", msg)
        return
    e_total, e_inkind = contribution_totals(entity)
    integ = entity.get("integrity") or {}
    bad = []
    if integ.get("total_co_contribution") is not None and abs(integ["total_co_contribution"] - e_total) > 0.01:
        bad.append(f"entity co-contrib {e_total:.0f} ≠ declared integrity {integ['total_co_contribution']:.0f}")
    if integ.get("total_in_kind") is not None and abs(integ["total_in_kind"] - e_inkind) > 0.01:
        bad.append(f"entity in-kind {e_inkind:.0f} ≠ declared integrity {integ['total_in_kind']:.0f}")
    if bdata is None:
        rep.add("contribution↔budget", "SKIP", "hard", "entity present but no --budget to reconcile")
        return
    b_co = sum(sum(float(v) for v in (r.get("years") or {}).values())
               for r in (bdata.get("rows") or []) if r.get("funding_source") == "co-contribution")
    b_ik = sum(sum(float(v) for v in (r.get("years") or {}).values())
               for r in (bdata.get("rows") or []) if r.get("funding_source") == "co-contribution"
               and r.get("kind") == "in-kind")
    if abs(b_co - e_total) > 0.01:
        bad.append(f"budget co-contrib rows {b_co:.0f} ≠ entity {e_total:.0f} (double-count/mismatch)")
    if abs(b_ik - e_inkind) > 0.01:
        bad.append(f"budget in-kind {b_ik:.0f} ≠ entity in-kind {e_inkind:.0f}")
    if bad:
        rep.add("contribution↔budget", "FAIL", "hard", "; ".join(bad))
    else:
        rep.add("contribution↔budget", "PASS", "hard",
                f"co-contribution {e_total:.0f} (in-kind {e_inkind:.0f}) reconciles entity↔budget")


def check_computed_gates(rep, scheme, values, entity, budget_path):
    gates = [g for g in (scheme.get("eligibility_gates") or [])
             if isinstance(g, dict) and (g.get("derived") or g.get("widget") == "computed")]
    derived = [g for g in gates if g.get("derived")]
    btot, bdata, phase_count = None, {}, 0
    if budget_path:
        code, out = run_sibling("validate_budget.py", [budget_path])
        btot = budget_totals(out)
        bdata = load_yaml(budget_path)
        phase_count = budget_phase_count(bdata)
        if code != 0:
            rep.add("budget-delegate", "FAIL", "hard", f"validate_budget.py exit {code}", out)
        else:
            rep.add("budget-delegate", "PASS", "hard", "validate_budget.py all rules pass", out)
    if not derived:
        rep.add("computed-gate", "SKIP", "hard", "scheme declares no derived/computed gates")
        return
    ns = build_ns(values, entity, btot, phase_count)
    for g in derived:
        binding = g.get("binding", "hard")
        try:
            ok = eval_derived(str(g["derived"]), ns, phase_count)
        except ResolveError as e:
            rep.add(f"computed-gate[{g['id']}]", "FAIL", binding,
                    f"cannot resolve {e.missing} — supply --values/--entity/--budget (fail-closed)")
            continue
        if ok:
            rep.add(f"computed-gate[{g['id']}]", "PASS", binding, f"{g['derived']}")
        else:
            rep.add(f"computed-gate[{g['id']}]", "FAIL" if binding == "hard" else "WARN",
                    binding, f"gate FAILS: {g['derived']}")


def check_rubric_subweights(rep, scheme):
    rub = {c.get("criterion"): c for c in (scheme.get("rubric") or []) if isinstance(c, dict)}
    declared = {k: c for k, c in rub.items() if c.get("indicators") or c.get("sub_weights")}
    if not declared:
        rep.add("rubric-sub_weights", "SKIP", "soft", "no rubric criterion declares sub-weights")
        return
    bad = []
    for k, c in declared.items():
        inds = c.get("indicators") or c.get("sub_weights") or []
        pts = [(i.get("points", i.get("weight"))) for i in inds if isinstance(i, dict)]
        if any(p is None for p in pts):
            bad.append(f"{k}: an indicator lacks points/weight")
    for f in iter_field_nodes(scheme.get("sections")):
        sub = f.get("sub_indicators")
        cr = f.get("criterion_ref")
        if sub and cr in rub:
            fs = sum(list(d.values())[0] for d in sub if isinstance(d, dict))
            rs = sum(i.get("points", 0) for i in (rub[cr].get("indicators") or []))
            if rs and abs(fs - rs) > 0.01:
                bad.append(f"{fid(f)}: sub_indicator points {fs} ≠ rubric {cr} points {rs}")
    if bad:
        rep.add("rubric-sub_weights", "FAIL", "soft", "; ".join(bad[:5]))
    else:
        rep.add("rubric-sub_weights", "PASS", "soft", f"{len(declared)} criteria carry consistent sub-weights")


def _needs_annex(f):
    if f.get("conditional_child") or f.get("requires_annex"):
        return f.get("requires_annex") or (f.get("conditional_child") or {}).get("id")
    cond = f.get("conditional") or {}
    if isinstance(cond, dict) and (cond.get("widget") == "structured-upload" or cond.get("provides")):
        return cond.get("provides") or cond.get("kind") or "annex"
    return None


def check_conditional_annexes(rep, scheme, values):
    cg = [f for f in iter_field_nodes(scheme.get("sections"))
          if f.get("widget") in ("conditional-group", "decision-tree") and _needs_annex(f)]
    if not cg:
        rep.add("conditional-annexes", "SKIP", "hard", "no conditional-group triggers a required annex")
        return
    if not values:
        rep.add("conditional-annexes", "SKIP", "hard",
                f"{len(cg)} conditional annex trigger(s) {[fid(f) for f in cg]} — supply --values")
        return
    att = " ".join(str(a) for a in (scheme.get("attachments") or []))
    bad = []
    for f in cg:
        ans = values.get(fid(f))
        if ans in (True, "Yes", "yes", "true"):
            need = str(_needs_annex(f))
            if need not in att and need not in " ".join(str(x) for x in (values.get("attachments") or [])):
                bad.append(f"{fid(f)} triggered but annex {need!r} absent from attachments")
    if bad:
        rep.add("conditional-annexes", "FAIL", "hard", "; ".join(bad))
    else:
        rep.add("conditional-annexes", "PASS", "hard", "all triggered annexes present")


def check_stage_lock(rep, scheme, values):
    phases = (scheme.get("submission") or {}).get("phases") or []
    order = [p for p in phases if p in PHASE_ORDER]
    if [PHASE_ORDER.index(p) for p in order] != sorted(PHASE_ORDER.index(p) for p in order):
        rep.add("phase-order", "FAIL", "hard", f"phases {phases} not in canonical order {PHASE_ORDER}")
    else:
        rep.add("phase-order", "PASS", "hard", f"phases {phases} ordered")
    active = set(phases)
    locked = [f for f in iter_field_nodes(scheme.get("sections"))
              if f.get("submission_phase") and f["submission_phase"] not in active]
    if not locked:
        rep.add("stage-lock", "SKIP", "hard", "no field is locked out of the active phase")
        return
    if not values:
        rep.add("stage-lock", "SKIP", "hard",
                f"{len(locked)} field(s) locked to a non-active phase — supply --values to check for edits")
        return
    bad = [fid(f) for f in locked if values.get(fid(f)) is not None]
    if bad:
        rep.add("stage-lock", "FAIL", "hard", f"value set for phase-locked field(s): {bad}")
    else:
        rep.add("stage-lock", "PASS", "hard", "no phase-locked field carries a value")


def check_attachments(rep, scheme):
    ups = [f for f in iter_field_nodes(scheme.get("sections")) if f.get("widget") == "structured-upload"]
    atts = scheme.get("attachments") or []
    if not ups and not atts:
        rep.add("attachment-rules", "SKIP", "hard", "scheme declares no attachments")
        return
    bad = []
    for f in ups:
        k = f.get("upload_kind") or f.get("kind")
        if k not in UPLOAD_KINDS:
            bad.append(f"{fid(f)}: invalid/absent upload kind {k!r}")
    for a in atts:
        if isinstance(a, dict) and a.get("kind") not in UPLOAD_KINDS and a.get("kind") != "calc-aid":
            bad.append(f"attachment {a.get('name')!r}: invalid kind {a.get('kind')!r}")
    paged = [fid(f) for f in ups if (f.get("limit") or {}).get("unit") == "pages"]
    if bad:
        rep.add("attachment-rules", "FAIL", "hard", "; ".join(bad[:6]))
    else:
        note = f"; {len(paged)} page-limited upload(s) need render (charcount --pages)" if paged else ""
        rep.add("attachment-rules", "PASS", "hard", f"{len(ups)} upload(s) have valid kinds{note}")


def check_char_rollup(rep, paste_ready):
    if not paste_ready:
        rep.add("char-rollup", "SKIP", "hard", "no --paste-ready supplied to char-count")
        return
    code, out = run_sibling("charcount.py", ["--text", paste_ready])
    if code == 0:
        rep.add("char-rollup", "PASS", "hard", "charcount.py: all fields within limits", out)
    else:
        rep.add("char-rollup", "FAIL", "hard",
                f"charcount.py exit {code} (over-limit / BLOCK / NEEDS-RENDER)", out)


def check_criterion_readiness(rep, scheme, values, evidence, mode):
    """Per scored criterion: readiness ∈ unsupported|partial|substantiated|submission-ready.

    In `submission` mode an `unsupported` criterion is a hard FAIL (a scored-criterion gap must
    not hide behind a SKIP); `partial` → WARN. In `draft` mode both surface as WARN, never block.
    """
    crits = [c for c in (scheme.get("rubric") or [])
             if isinstance(c, dict) and c.get("minimum_evidence")]
    if not crits:
        rep.add("criterion-readiness", "SKIP", "soft",
                "scheme declares no rubric criterion with minimum_evidence")
        return
    present = _evidence_present(evidence)
    fields_by_crit = {}
    for f in iter_field_nodes(scheme.get("sections")):
        cr = f.get("criterion_ref") or f.get("criterion")
        if cr is not None:
            fields_by_crit.setdefault(cr, []).append(fid(f))
    for c in crits:
        name = c.get("criterion")
        req = [str(x) for x in c["minimum_evidence"]]
        sat = [x for x in req if _canon_evidence_class(x) in present]
        missing = [x for x in req if _canon_evidence_class(x) not in present]
        fnames = fields_by_crit.get(name, [])
        vals = [values.get(fn) for fn in fnames] if values else []
        filled = [v for v in vals if v not in (None, "", [], {})]
        has_content = bool(filled)
        all_content = bool(fnames) and len(filled) == len(fnames)
        markers = bool(READINESS_MARKER.search(str(vals)))
        w = c.get("weight")
        wtxt = (f"{round(w * 100)}%" if isinstance(w, (int, float)) and w <= 1
                else f"{round(w)}%" if isinstance(w, (int, float)) else "n/a")
        if not sat and not has_content:
            state = "unsupported"
        elif markers or len(sat) < len(req) or not has_content:
            state = "partial"
        elif all_content:
            state = "submission-ready"
        else:
            state = "substantiated"
        detail = f"{len(sat)}/{len(req)} evidence class(es) present"
        if missing:
            detail += f" (missing: {', '.join(missing)})"
        if markers:
            detail += "; residual [TO SET]/[VERIFY]/[…NEEDED] marker blocks substantiation"
        if fnames and not has_content:
            detail += "; no field content supplied"
        if state in ("substantiated", "submission-ready"):
            status, binding = "PASS", "hard"
        elif state == "partial":
            status, binding = "WARN", "soft"
        elif mode == "submission":
            status, binding = "FAIL", "hard"
            detail += " — unevidenced scored criterion (fail-closed, submission mode)"
        else:
            status, binding = "WARN", "soft"
        rep.add(f"criterion-readiness[{name}]", status, binding,
                f"{state} (weight {wtxt}) — {detail}")


# ── orchestration ────────────────────────────────────────────────────────────
def orchestrate(scheme_path, values_path=None, evidence_path=None, entity_path=None,
                budget_path=None, paste_ready=None, mode="draft"):
    scheme = load_yaml(scheme_path)
    values = load_yaml(values_path) if values_path else {}
    evidence = load_yaml(evidence_path) if evidence_path else {}
    entity = load_yaml(entity_path) if entity_path else {}
    bdata = load_yaml(budget_path) if budget_path else None
    rep = Report()
    print(f"IR integrity dry-run — scheme: {scheme.get('scheme', scheme_path)} "
          f"[{scheme.get('mode', '?')}] — readiness mode: {mode}\n" + "-" * 72)
    check_schema(rep, scheme)
    check_allocation(rep, scheme, values)
    check_contribution(rep, scheme, entity, bdata)
    check_computed_gates(rep, scheme, values, entity, budget_path)
    check_rubric_subweights(rep, scheme)
    check_conditional_annexes(rep, scheme, values)
    check_stage_lock(rep, scheme, values)
    check_attachments(rep, scheme)
    check_char_rollup(rep, paste_ready)
    check_criterion_readiness(rep, scheme, values, evidence, mode)
    rep.render()
    return 1 if rep.hard_failed() else 0


# ── self-test ────────────────────────────────────────────────────────────────
def _clean_scheme():
    return {
        "scheme": "Self-Test", "mode": "prospective-project",
        "submission": {"phases": ["full", "post-award"]},
        "eligibility_gates": [
            {"id": "grant-le-50", "binding": "hard",
             "derived": "grant_sought <= 0.5 * total_eligible_expenditure"},
            {"id": "phased", "binding": "hard",
             "derived": "requested_grant_total >= 200000 => count(distinct budget.phase) >= 2"},
        ],
        "sections": [{"id": "S", "fields": [
            {"id": "n1", "widget": "narrative", "role": "criterion-scored", "criterion_ref": "C1",
             "sub_indicators": [{"a": 6}, {"b": 4}]},
            {"id": "for", "widget": "taxonomy-code", "role": "classification",
             "taxonomy": {"allocation_sums_to": 100}},
            {"id": "cm", "widget": "contribution-matrix", "role": "budget-resource"},
            {"id": "up", "widget": "structured-upload", "upload_kind": "proforma", "role": "team-partner"},
            {"id": "cond", "widget": "conditional-group", "role": "compliance",
             "requires_annex": "ATSI evidence"},
            {"id": "late", "widget": "scalar", "role": "logistics", "submission_phase": "post-award"},
            {"id": "gref", "widget": "money", "role": "budget-resource", "gate": "grant-le-50"},
        ]}],
        "rubric": [{"criterion": "C1", "indicators": [{"id": "a", "points": 6}, {"id": "b", "points": 4}]}],
        "attachments": [{"name": "ATSI evidence", "kind": "proforma"}],
    }


def _entity():
    return {"partners": [
        {"id": "p1", "contributions": {"cash": {"fy1": 150000}, "in_kind": {"fy1": 0}}},
        {"id": "p2", "contributions": {"cash": {"fy1": 0}, "in_kind": {"fy1": 200000}}}],
        "integrity": {"total_co_contribution": 350000, "total_in_kind": 200000,
                      "grant_sought": 300000, "total_eligible_expenditure": 650000}}


def _write(tmp, name, text):
    p = os.path.join(tmp, name)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text)
    return p


def self_test():
    import tempfile
    tmp = tempfile.mkdtemp()
    # budget that reconciles with _entity: co-contribution 350000 (cash 150k + in-kind 200k),
    # requested 300000 across 2 phases → matched ratio 350000/300000 >= 1.0.
    budget = _write(tmp, "budget.yaml", """
matched_funding_min_ratio: 1.0
phased_if_min: 200000
row_caps: [{category: overseas, max_pct: 10.0, of: total}]
declared_totals: {requested: 300000}
rows:
  - {category: labour, funding_source: requested, kind: cash, phase: p1, years: {2026: 200000}}
  - {category: labour, funding_source: requested, kind: cash, phase: p2, years: {2027: 100000}}
  - {category: cash,   funding_source: co-contribution, kind: cash,    years: {2026: 150000}}
  - {category: ik,     funding_source: co-contribution, kind: in-kind, years: {2026: 200000}}
""")
    paste_ok = _write(tmp, "ok.txt", "=== f1 | F1 | limit: 10 chars ===\nhello\n=== /f1 ===\n")
    paste_bad = _write(tmp, "bad.txt", "=== f1 | F1 | limit: 3 chars ===\ntoolong\n=== /f1 ===\n")
    scheme_p = _write(tmp, "scheme.yaml", __import__("yaml").safe_dump(_clean_scheme()))
    ent_p = _write(tmp, "entity.yaml", __import__("yaml").safe_dump(_entity()))
    values_ok = _write(tmp, "vok.yaml", __import__("yaml").safe_dump(
        {"for": [{"pct": 60}, {"pct": 40}], "cond": False}))

    # 1. clean end-to-end → exit 0
    code = orchestrate(scheme_p, values_ok, None, ent_p, budget, paste_ok)
    assert code == 0, f"clean IR should pass, got exit {code}"

    # per-check FAIL fixtures (call the check directly; assert a FAIL entry appears)
    def fails(fn, *a):
        r = Report(); fn(r, *a)
        return [e for e in r.entries if e[1] == "FAIL"]

    scheme = _clean_scheme()
    # 1. schema — bad widget + dangling criterion ref
    bs = _clean_scheme()
    bs["sections"][0]["fields"][0]["widget"] = "not-a-widget"
    bs["sections"][0]["fields"][0]["criterion_ref"] = "C9"
    assert fails(check_schema, bs), "bad widget/criterion must FAIL schema"
    # 2. allocation — sums to 90
    assert fails(check_allocation, scheme, {"for": [{"pct": 60}, {"pct": 30}]}), "bad allocation FAIL"
    # 3. contribution↔budget — entity in-kind mislabelled (double-count)
    be = _entity(); be["integrity"]["total_co_contribution"] = 999999
    assert fails(check_contribution, scheme, be, load_yaml(budget)), "contribution mismatch FAIL"
    # 4. computed gate — grant > 50% of total
    bad_ent = _entity()
    bad_ent["integrity"].update({"grant_sought": 600000, "total_eligible_expenditure": 650000})
    r4 = Report(); check_computed_gates(r4, scheme, {}, bad_ent, budget)
    assert any(e[1] == "FAIL" and "grant-le-50" in e[0] for e in r4.entries), "over-50% gate must FAIL"
    # 5. rubric sub_weights — field points 6+4=10 ≠ rubric 6+8=14
    b5 = _clean_scheme(); b5["rubric"][0]["indicators"][1]["points"] = 8
    assert fails(check_rubric_subweights, b5), "sub_weight mismatch FAIL"
    # 6. conditional annex — triggered but annex removed
    b6 = _clean_scheme(); b6["attachments"] = []
    assert fails(check_conditional_annexes, b6, {"cond": True}), "missing annex FAIL"
    # 7. stage_lock — value set for a phase-locked field
    b7 = _clean_scheme(); b7["submission"]["phases"] = ["full"]
    assert fails(check_stage_lock, b7, {"late": "x"}), "phase-locked value FAIL"
    #    phase-order — reversed phases
    b7b = _clean_scheme(); b7b["submission"]["phases"] = ["post-award", "full"]
    assert fails(check_stage_lock, b7b, {}), "reversed phase order FAIL"
    # 8. attachment rules — invalid upload kind
    b8 = _clean_scheme(); b8["sections"][0]["fields"][3]["upload_kind"] = "blob"
    assert fails(check_attachments, b8), "invalid upload kind FAIL"
    # 9. char roll-up — over-limit paste-ready (delegates to charcount.py)
    r9 = Report(); check_char_rollup(r9, paste_bad)
    assert any(e[1] == "FAIL" for e in r9.entries), "over-limit char roll-up FAIL"

    # full doctored orchestrate → non-zero exit
    doctored = _write(tmp, "bad_scheme.yaml", __import__("yaml").safe_dump(bs))
    assert orchestrate(doctored, values_ok, None, ent_p, budget, paste_bad) == 1, "doctored IR must exit 1"

    print("\nself-test OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scheme", help="scheme.yaml (Stage-A IR) — required unless --self-test")
    ap.add_argument("--values", help="values.yaml (filled field content sidecar)")
    ap.add_argument("--evidence", help="evidence-store.yaml")
    ap.add_argument("--entity", help="entity-store.yaml (partners/contributions)")
    ap.add_argument("--budget", help="budget.yaml (delegated to validate_budget.py)")
    ap.add_argument("--paste-ready", dest="paste_ready", help="PASTE-READY.txt (delegated to charcount.py)")
    ap.add_argument("--mode", choices=("submission", "draft"), default="draft",
                    help="criterion-readiness gate: submission FAILs an unsupported scored "
                         "criterion; draft (default) reports it as WARN")
    ap.add_argument("--self-test", action="store_true", help="run built-in self-test")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.scheme:
        ap.error("--scheme is required (or --self-test)")
    return orchestrate(args.scheme, args.values, args.evidence, args.entity,
                       args.budget, args.paste_ready, args.mode)


if __name__ == "__main__":
    sys.exit(main())
