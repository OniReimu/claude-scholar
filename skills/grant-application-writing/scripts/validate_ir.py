# /// script
# requires-python = ">=3.9"
# dependencies = ["pyyaml"]
# ///
"""IR-level integrity orchestrator (Stage E: portal validation dry-run) — FAIL-CLOSED.

The single pre-submit gate for the cross-field couplings the docs acknowledged but no shipped
script checked (Codex top-3 #3; Phase-2's F.2↔H.1↔matched "hardest thing to model"). Reads the
`scheme.yaml` IR (+ optional values/evidence/entity/budget/paste-ready/plan sidecars) and runs 16
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
 11. partner-commitment  per partner, reconcile the letter_commitment figure against the
                        application contributions + budget line; --mode submission FAILs a
                        mismatch, a conditional-as-committed claim, or an unverified partner.
 12. process-dispatch   second dispatch axis — the scheme's assessment-PROCESS shape
                        (single-stage-review/staged/interview-gated/panel-routed/curated/
                        rolling) is a non-empty closed-vocab subset and consistent with the
                        phases/routing fields/rejoinder it implies; an unknown process tag, or
                        a missing process on a scheme that HAS a rubric, is a hard FAIL.

Checks 13–16 are `prospective-project`-mode project-substance passes reading the `--plan`
project-plan.yaml sidecar (aims/design, benefits, additionality, risks registers). Each is
gated on `mode == prospective-project` AND a supplied `--plan`; otherwise a labelled SKIP.
When the register IS present they are fail-closed — a present-but-empty field is never
green-washed. `--mode submission` FAILs, `--mode draft` WARNs (mirrors criterion-readiness).

 13. research-design-adequacy every aims[] id is covered by ≥1 design[].aim, has a non-empty
                        success_criterion, and each covering design has a non-empty answers_aim;
                        an uncovered/unmeasured aim FAILs submission (per aim).
 14. benefits-realisation every benefits[] row carries a non-empty owner + metric + timing; an
                        aspirational (unowned/unmeasured/untimed) benefit FAILs submission (per benefit).
 15. additionality-vfm  additionality.counterfactual is non-empty; reports the leverage ratio
                        (co_contribution/grant) and, when --budget is supplied, cross-checks
                        grant/co-contribution against budget totals (>1% mismatch or a missing
                        counterfactual FAILs submission).
 16. risk-triggers      every risk with impact==high carries a non-empty trigger + contingency +
                        owner; a triggerless high-impact risk FAILs submission (per risk).

SKIP vs FAIL (fail-closed): FAIL when the needed input WAS supplied but the data violates the
rule or a hard gate cannot be evaluated; SKIP (non-blocking, with a stated reason) only when an
OPTIONAL sidecar was not supplied, or the scheme lacks that construct. Exit non-zero on any HARD
FAIL (or a delegated sibling non-zero exit); soft binding = WARN.

Criterion-readiness never hides a scored-criterion gap behind a SKIP: if a criterion declares
`minimum_evidence` but the evidence/content backing it was not supplied, that criterion is
`unsupported` — a [FAIL] in `--mode submission` (default `draft` reports it as [WARN]). Only a
scheme that declares NO criterion `minimum_evidence` genuinely SKIPs the whole check.

    uv run validate_ir.py --scheme scheme.yaml --entity entity-store.yaml --budget budget.yaml
    uv run validate_ir.py --scheme scheme.yaml --plan project-plan.yaml --budget budget.yaml
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

# ── second dispatch axis: closed process-archetype vocabulary (extend only by falsifiability) ──
PROCESS_VOCAB = {
    "single-stage-review", "staged", "interview-gated", "panel-routed", "curated", "rolling",
}
STAGED_GATING_PHASES = {"EOI", "pre-proposal", "minimum-data"}
INTERVIEW_MARKERS = ("interview", "defense", "defence")

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


# ── the 16 checks ────────────────────────────────────────────────────────────
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


def _partner_declared(p):
    """The APPLICATION's declared (cash, in_kind) for a partner from `contributions`.

    Returns None for a type the partner does not declare at all (so a genuinely absent
    figure is never green-washed to 0); a declared type that sums to 0 stays 0.0.
    """
    cash = in_kind = None
    c = p.get("contributions")
    if isinstance(c, dict):
        for typ, cells in c.items():
            key = str(typ).replace("_", "").replace("-", "").lower()
            s = sum(float(v) for v in (cells or {}).values()) if isinstance(cells, dict) else float(cells or 0)
            if key.startswith("in") and "kind" in key:
                in_kind = (in_kind or 0.0) + s
            elif key == "cash":
                cash = (cash or 0.0) + s
    elif isinstance(c, list):
        for row in c:
            amt = (row.get("amount") or {}).get("value", row.get("value", 0)) or 0
            if str(row.get("type", "")).startswith("in"):
                in_kind = (in_kind or 0.0) + float(amt)
            else:
                cash = (cash or 0.0) + float(amt)
    return cash, in_kind


def _partner_budget(bdata, pid):
    """The (cash, in_kind) co-contribution BUDGET lines tagged to a partner id."""
    if not bdata:
        return None, None
    cash = in_kind = None
    for r in (bdata.get("rows") or []):
        if r.get("funding_source") != "co-contribution" or str(r.get("partner")) != str(pid):
            continue
        s = sum(float(v) for v in (r.get("years") or {}).values())
        if r.get("kind") == "in-kind":
            in_kind = (in_kind or 0.0) + s
        else:
            cash = (cash or 0.0) + s
    return cash, in_kind


def _letter_figures(p):
    """The LETTER's (cash, in_kind, conditional?) from a `letter_commitment` block."""
    lc = p.get("letter_commitment")
    if not isinstance(lc, dict):
        return None, None, False

    def one(blk):
        b = lc.get(blk)
        if isinstance(b, dict):
            v = b.get("value")
            return (float(v) if v is not None else None), bool(b.get("conditional"))
        return None, False

    cash, cash_cond = one("cash")
    in_kind, ik_cond = one("in_kind")
    return cash, in_kind, (cash_cond or ik_cond or bool(lc.get("conditional")))


def check_partner_commitment_reconciliation(rep, scheme, entity, bdata, mode):
    """Per partner, reconcile the three figures the IR can see: the application's declared
    `contributions`, the matching `contribution/budget-matrix` line, and the LETTER's
    `letter_commitment`. A numeric mismatch, a conditional commitment rendered as
    `status: committed`, or a cash/in-kind line with no letter_commitment AND no provenance
    (UNVERIFIED) is a hard FAIL in `submission` mode; in `draft` mode each surfaces as WARN.
    """
    partners = entity.get("partners") or []
    if not partners:
        rep.add("partner-commitment-reconciliation", "SKIP", "hard",
                "no entity partners to reconcile letter↔application↔budget")
        return
    fmt = lambda v: f"{v:.0f}" if v is not None else "—"
    for p in partners:
        pid = p.get("id") or p.get("name") or "<partner>"
        a_cash, a_ik = _partner_declared(p)
        b_cash, b_ik = _partner_budget(bdata, pid)
        l_cash, l_ik, conditional = _letter_figures(p)
        lc = p.get("letter_commitment") if isinstance(p.get("letter_commitment"), dict) else None
        status_claim = str(p.get("status") or (lc or {}).get("status") or "").lower()
        has_line = any(v is not None for v in (a_cash, a_ik, b_cash, b_ik))
        provenance = p.get("provenance") or (lc or {}).get("provenance")

        problems = []
        for label, vals in (("cash", (a_cash, b_cash, l_cash)), ("in_kind", (a_ik, b_ik, l_ik))):
            present = [v for v in vals if v is not None]
            if len(present) >= 2 and max(present) - min(present) > 0.01:
                srcs = ", ".join(f"{nm} {v:.0f}" for nm, v in
                                 zip(("application", "budget", "letter"), vals) if v is not None)
                problems.append(f"{label} mismatch ({srcs})")
        if conditional and status_claim == "committed":
            problems.append("letter marks the commitment conditional but partner status=committed")
        unverified = has_line and lc is None and not provenance

        if not problems and not unverified:
            if not has_line and l_cash is None and l_ik is None:
                rep.add(f"partner-commitment-reconciliation[{pid}]", "PASS", "hard",
                        "no cash/in-kind commitment declared — nothing to reconcile")
            else:
                rep.add(f"partner-commitment-reconciliation[{pid}]", "PASS", "hard",
                        f"reconciled — cash a/{fmt(a_cash)} b/{fmt(b_cash)} l/{fmt(l_cash)}, "
                        f"in_kind a/{fmt(a_ik)} b/{fmt(b_ik)} l/{fmt(l_ik)}")
            continue
        reason = ("; ".join(problems) if problems else
                  f"cash/in-kind line present but no letter_commitment and no provenance "
                  f"(cash a/{fmt(a_cash)} b/{fmt(b_cash)}, in_kind a/{fmt(a_ik)} b/{fmt(b_ik)}) — UNVERIFIED")
        if mode == "submission":
            rep.add(f"partner-commitment-reconciliation[{pid}]", "FAIL", "hard",
                    reason + " (fail-closed, submission mode)")
        else:
            rep.add(f"partner-commitment-reconciliation[{pid}]", "WARN", "soft", reason)


def check_process_dispatch(rep, scheme, mode):
    """Second dispatch axis: the scheme's assessment-PROCESS shape must be a declared subset
    of the closed archetype vocabulary and consistent with the stages/fields/rejoinder it
    implies. Fail-closed: an unknown process tag is ALWAYS a hard FAIL; a missing `process`
    on a scheme that carries a rubric is a parse gap (hard FAIL), not a silent default. A
    scheme with no rubric and no process has nothing to route → SKIP.

    Per recognised tag: `staged` requires a gating phase in `submission.phases` (FAIL in
    submission / WARN in draft); `panel-routed` requires a taxonomy-code/classification
    routing field (WARN if absent or set at low confidence — a routing gap); `interview-gated`
    expects a defense-prep deliverable (WARN if absent); a top-level `rejoinder.enabled`
    capability must ride on `single-stage-review` (WARN if not).
    """
    proc = scheme.get("process")
    tags = proc if isinstance(proc, list) else ([proc] if proc else [])
    has_rubric = bool(scheme.get("rubric"))

    if not tags:
        if has_rubric:
            rep.add("process-dispatch", "FAIL", "hard",
                    "scheme carries a rubric but declares no `process` — a missing assessment "
                    "process is a parse gap, not a default (fail-closed)")
        else:
            rep.add("process-dispatch", "SKIP", "hard",
                    "scheme declares no `process` and no rubric — nothing to route")
        return

    unknown = [t for t in tags if t not in PROCESS_VOCAB]
    known = [t for t in tags if t in PROCESS_VOCAB]
    if unknown:
        rep.add("process-dispatch", "FAIL", "hard",
                f"unknown process tag(s) {unknown} — not in closed vocab "
                f"{sorted(PROCESS_VOCAB)} (fail-closed)")
    if not known:
        return
    if not unknown:
        rep.add("process-dispatch", "PASS", "hard", f"process {known} ⊆ closed vocab")

    # staged ⇒ a gating EOI/pre-proposal/minimum-data phase must precede the full
    if "staged" in known:
        phases = (scheme.get("submission") or {}).get("phases") or []
        gating = sorted(set(phases) & STAGED_GATING_PHASES)
        if gating:
            rep.add("process-dispatch[staged]", "PASS", "hard",
                    f"staged process wires to gating phase {gating}")
        else:
            reason = (f"staged process but submission.phases {phases} contains no gating phase "
                      f"{sorted(STAGED_GATING_PHASES)} — no EOI/pre-proposal sub-pipeline to gate")
            if mode == "submission":
                rep.add("process-dispatch[staged]", "FAIL", "hard",
                        reason + " (fail-closed, submission mode)")
            else:
                rep.add("process-dispatch[staged]", "WARN", "soft", reason)

    # panel-routed ⇒ the routing taxonomy/classification field is gate-critical
    if "panel-routed" in known:
        routing = [f for f in iter_field_nodes(scheme.get("sections"))
                   if f.get("widget") == "taxonomy-code" or "classification" in roles_of(f)]
        low = [fid(f) for f in routing
               if str(f.get("confidence", "")).lower() in ("low", "none", "unset", "guess")
               or f.get("low_confidence")]
        if not routing:
            rep.add("process-dispatch[panel-routed]", "WARN", "soft",
                    "panel-routed process but no taxonomy-code/classification field present — "
                    "routing gap (a wrong/absent code silently routes to the wrong panel)")
        elif low:
            rep.add("process-dispatch[panel-routed]", "WARN", "soft",
                    f"panel-routed routing field(s) {low} set at low/unset confidence — verify "
                    f"the code routes to the intended panel before submission")
        else:
            rep.add("process-dispatch[panel-routed]", "PASS", "hard",
                    f"panel-routed process has routing field(s) {[fid(f) for f in routing]}")

    # rejoinder is a within-round right-of-reply capability — belongs to single-stage-review
    rej = scheme.get("rejoinder")
    if isinstance(rej, dict) and rej.get("enabled"):
        if "single-stage-review" in known:
            rep.add("process-dispatch[rejoinder]", "PASS", "hard",
                    "rejoinder.enabled rides on single-stage-review")
        else:
            rep.add("process-dispatch[rejoinder]", "WARN", "soft",
                    f"rejoinder.enabled but process {known} has no single-stage-review — a "
                    f"within-round right-of-reply belongs to a single-stage review")

    # interview-gated ⇒ a defense-prep deliverable (attachment or field) is expected
    if "interview-gated" in known:
        found = None
        for a in (scheme.get("attachments") or []):
            name = a.get("name") if isinstance(a, dict) else a
            if name and any(w in str(name).lower() for w in INTERVIEW_MARKERS):
                found = name
                break
        if not found:
            for f in iter_field_nodes(scheme.get("sections")):
                label = f"{f.get('id', '')} {f.get('label', '')}".lower()
                if any(w in label for w in INTERVIEW_MARKERS):
                    found = fid(f)
                    break
        if found:
            rep.add("process-dispatch[interview-gated]", "PASS", "hard",
                    f"interview-gated process has a defense-prep deliverable ({found})")
        else:
            rep.add("process-dispatch[interview-gated]", "WARN", "soft",
                    "interview-gated process but no interview/defense deliverable (attachment or "
                    "field) — a defense-prep artifact is expected (soft)")


# ── project-substance passes (checks 13–16, prospective-project mode + --plan) ──
def _nonempty(v):
    """Fail-closed truthiness: a present-but-empty field (None, "", "  ") is NOT satisfied."""
    if v is None:
        return False
    if isinstance(v, str):
        return bool(v.strip())
    return bool(v)


def _project_gate(rep, check, scheme, plan):
    """Gate a project-substance check on prospective-project mode + a supplied --plan.

    Returns True when the check should run; otherwise emits a labelled SKIP and returns False.
    """
    if scheme.get("mode") != "prospective-project":
        rep.add(check, "SKIP", "soft",
                f"scheme mode {scheme.get('mode', '?')!r} ≠ prospective-project — no project plan to assess")
        return False
    if not plan:
        rep.add(check, "SKIP", "soft",
                "prospective-project mode but no --plan project-plan.yaml supplied")
        return False
    return True


def _budget_source_total(bdata, source):
    return sum(sum(float(v) for v in (r.get("years") or {}).values())
               for r in (bdata.get("rows") or []) if r.get("funding_source") == source)


def _pct_mismatch(a, b):
    """True when a and b differ by more than 1% of the larger magnitude."""
    return abs(a - b) / max(abs(a), abs(b), 1.0) > 0.01


def check_research_design_adequacy(rep, scheme, plan, mode):
    """#4 deepens methods-feasibility (§2.3): the design must ANSWER each aim. Every aims[] id
    is covered by ≥1 design[].aim, every aim has a non-empty success_criterion, and every
    covering design has a non-empty answers_aim. In `submission` mode an uncovered / unmeasured
    aim is a hard FAIL (per aim); in `draft` mode it WARNs.
    """
    if not _project_gate(rep, "research-design-adequacy", scheme, plan):
        return
    aims = plan.get("aims") or []
    if not aims:
        rep.add("research-design-adequacy", "SKIP", "soft", "project-plan declares no aims register")
        return
    covered = {}
    for d in (plan.get("design") or []):
        if isinstance(d, dict) and d.get("aim") is not None:
            covered.setdefault(str(d["aim"]), []).append(d)
    for a in aims:
        aid = str(a.get("id") or a.get("statement") or "<aim>")
        rows = covered.get(aid, [])
        problems = []
        if not rows:
            problems.append("no design row covers this aim")
        if not _nonempty(a.get("success_criterion")):
            problems.append("empty success_criterion")
        if any(not _nonempty(d.get("answers_aim")) for d in rows):
            problems.append("a covering design has empty answers_aim")
        if not problems:
            rep.add(f"research-design-adequacy[{aid}]", "PASS", "hard",
                    f"answered by {len(rows)} design row(s) with success_criterion + answers_aim")
        elif mode == "submission":
            rep.add(f"research-design-adequacy[{aid}]", "FAIL", "hard",
                    "; ".join(problems) + " (fail-closed, submission mode)")
        else:
            rep.add(f"research-design-adequacy[{aid}]", "WARN", "soft", "; ".join(problems))


def check_benefits_realisation(rep, scheme, plan, mode):
    """#5 deepens impact-pathway (§2.5): each benefit must be realisable, measurable, and OWNED.
    Every benefits[] row carries a non-empty owner AND metric AND timing. A benefit missing any
    of the three is aspirational — a hard FAIL (per benefit) in `submission`, a WARN in `draft`.
    """
    if not _project_gate(rep, "benefits-realisation", scheme, plan):
        return
    benefits = plan.get("benefits") or []
    if not benefits:
        rep.add("benefits-realisation", "SKIP", "soft", "project-plan declares no benefits register")
        return
    for b in benefits:
        bid = str(b.get("id") or b.get("benefit") or "<benefit>")
        missing = [f for f in ("owner", "metric", "timing") if not _nonempty(b.get(f))]
        if not missing:
            rep.add(f"benefits-realisation[{bid}]", "PASS", "hard",
                    "realisable — owner + metric + timing all present")
        elif mode == "submission":
            rep.add(f"benefits-realisation[{bid}]", "FAIL", "hard",
                    f"aspirational — missing {', '.join(missing)} (fail-closed, submission mode)")
        else:
            rep.add(f"benefits-realisation[{bid}]", "WARN", "soft",
                    f"aspirational — missing {', '.join(missing)}")


def check_additionality_vfm(rep, scheme, plan, bdata, mode):
    """#12 additionality / value-for-money: a non-empty counterfactual (without THIS grant it
    wouldn't happen) plus a VfM leverage ratio (co_contribution/grant). When --budget is
    supplied, the leverage grant/co-contribution are cross-checked against the budget totals
    (>1% mismatch, like partner-commitment). A missing counterfactual or a budget mismatch is a
    hard FAIL in `submission`, a WARN in `draft`.
    """
    if not _project_gate(rep, "additionality-vfm", scheme, plan):
        return
    add = plan.get("additionality")
    if not isinstance(add, dict) or not add:
        rep.add("additionality-vfm", "SKIP", "soft", "project-plan declares no additionality register")
        return
    problems = []
    if not _nonempty(add.get("counterfactual")):
        problems.append("empty counterfactual (the without-this-grant argument)")
    lev = add.get("leverage") or {}
    grant, co = lev.get("grant"), lev.get("co_contribution")
    ratio = (f"; leverage ratio {co / grant:.2f} (co-contribution {co:.0f} / grant {grant:.0f})"
             if isinstance(grant, (int, float)) and isinstance(co, (int, float)) and grant
             else "; no leverage grant/co_contribution to compute VfM ratio")
    if bdata is not None:
        if isinstance(grant, (int, float)):
            b_req = _budget_source_total(bdata, "requested")
            if _pct_mismatch(grant, b_req):
                problems.append(f"leverage.grant {grant:.0f} ≠ budget requested {b_req:.0f} (>1%)")
        if isinstance(co, (int, float)):
            b_co = _budget_source_total(bdata, "co-contribution")
            if _pct_mismatch(co, b_co):
                problems.append(f"leverage.co_contribution {co:.0f} ≠ budget co-contribution {b_co:.0f} (>1%)")
    if not problems:
        rep.add("additionality-vfm", "PASS", "hard", "counterfactual present" + ratio)
    elif mode == "submission":
        rep.add("additionality-vfm", "FAIL", "hard",
                "; ".join(problems) + " (fail-closed, submission mode)")
    else:
        rep.add("additionality-vfm", "WARN", "soft", "; ".join(problems))


def check_risk_triggers(rep, scheme, plan, mode):
    """#14 deepens risk-mitigation (§2.4): upgrade risk from static coverage to a live
    trigger→contingency register. Every risk with impact==high carries a non-empty trigger AND
    contingency AND owner. A high-impact risk missing any of the three is a hard FAIL (per risk)
    in `submission`, a WARN in `draft`. A plan whose risks are all sub-high needs no trigger register.
    """
    if not _project_gate(rep, "risk-triggers", scheme, plan):
        return
    risks = plan.get("risks") or []
    if not risks:
        rep.add("risk-triggers", "SKIP", "soft", "project-plan declares no risks register")
        return
    high = [r for r in risks if str(r.get("impact", "")).lower() == "high"]
    if not high:
        rep.add("risk-triggers", "PASS", "hard",
                f"{len(risks)} risk(s) declared, none high-impact — no trigger register required")
        return
    for r in high:
        rid = str(r.get("id") or r.get("risk") or "<risk>")
        missing = [f for f in ("trigger", "contingency", "owner") if not _nonempty(r.get(f))]
        if not missing:
            rep.add(f"risk-triggers[{rid}]", "PASS", "hard",
                    "high-impact risk has trigger + contingency + owner")
        elif mode == "submission":
            rep.add(f"risk-triggers[{rid}]", "FAIL", "hard",
                    f"high-impact risk missing {', '.join(missing)} (fail-closed, submission mode)")
        else:
            rep.add(f"risk-triggers[{rid}]", "WARN", "soft",
                    f"high-impact risk missing {', '.join(missing)}")


# ── orchestration ────────────────────────────────────────────────────────────
def orchestrate(scheme_path, values_path=None, evidence_path=None, entity_path=None,
                budget_path=None, paste_ready=None, mode="draft", plan_path=None):
    scheme = load_yaml(scheme_path)
    values = load_yaml(values_path) if values_path else {}
    evidence = load_yaml(evidence_path) if evidence_path else {}
    entity = load_yaml(entity_path) if entity_path else {}
    bdata = load_yaml(budget_path) if budget_path else None
    plan = load_yaml(plan_path) if plan_path else {}
    rep = Report()
    print(f"IR integrity dry-run — scheme: {scheme.get('scheme', scheme_path)} "
          f"[{scheme.get('mode', '?')}] — readiness mode: {mode}\n" + "-" * 72)
    check_schema(rep, scheme)
    check_allocation(rep, scheme, values)
    check_contribution(rep, scheme, entity, bdata)
    check_partner_commitment_reconciliation(rep, scheme, entity, bdata, mode)
    check_computed_gates(rep, scheme, values, entity, budget_path)
    check_rubric_subweights(rep, scheme)
    check_conditional_annexes(rep, scheme, values)
    check_stage_lock(rep, scheme, values)
    check_attachments(rep, scheme)
    check_char_rollup(rep, paste_ready)
    check_criterion_readiness(rep, scheme, values, evidence, mode)
    check_process_dispatch(rep, scheme, mode)
    check_research_design_adequacy(rep, scheme, plan, mode)
    check_benefits_realisation(rep, scheme, plan, mode)
    check_additionality_vfm(rep, scheme, plan, bdata, mode)
    check_risk_triggers(rep, scheme, plan, mode)
    rep.render()
    return 1 if rep.hard_failed() else 0


# ── self-test ────────────────────────────────────────────────────────────────
def _clean_scheme():
    return {
        "scheme": "Self-Test", "mode": "prospective-project",
        "process": ["single-stage-review"],
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
        "rubric": [{"criterion": "C1", "weight": 0.40,
                    "minimum_evidence": ["publication"],
                    "readiness_rule": ">=1 backed claim per sub-indicator; no [TO SET] markers",
                    "indicators": [{"id": "a", "points": 6}, {"id": "b", "points": 4}]}],
        "attachments": [{"name": "ATSI evidence", "kind": "proforma"}],
    }


def _evidence():
    return {"publications": [{"id": "pub1", "title": "A backed result", "year": 2024}]}


def _entity():
    # letter_commitment + provenance so each partner reconciles under the submission gate;
    # figures match the declared contributions (fictional entities only).
    return {"partners": [
        {"id": "p1", "status": "committed", "provenance": "corpus/loi-p1.pdf",
         "contributions": {"cash": {"fy1": 150000}, "in_kind": {"fy1": 0}},
         "letter_commitment": {"cash": {"value": 150000, "conditional": False},
                               "in_kind": {"value": 0, "conditional": False}}},
        {"id": "p2", "status": "committed", "provenance": "corpus/loi-p2.pdf",
         "contributions": {"cash": {"fy1": 0}, "in_kind": {"fy1": 200000}},
         "letter_commitment": {"cash": {"value": 0, "conditional": False},
                               "in_kind": {"value": 200000, "conditional": False}}}],
        "integrity": {"total_co_contribution": 350000, "total_in_kind": 200000,
                      "grant_sought": 300000, "total_eligible_expenditure": 650000}}


def _plan():
    # A fully-complete project-plan (fictional entities only): every aim answered by a design
    # with a success_criterion + answers_aim; every benefit owned + measured + timed; a
    # counterfactual + leverage matching the _entity/budget fixture (grant 300k, co 350k);
    # every high-impact risk carries a trigger + contingency + owner.
    return {
        "aims": [{"id": "aim-1", "statement": "Establish the shared-ledger throughput bound",
                  "success_criterion": "sustained ≥10k tx/s on the reference cluster"}],
        "design": [{"aim": "aim-1", "methods": ["controlled load test"],
                    "controls": ["single-node baseline", "no-batching ablation"],
                    "validity": {"sample_size": 30, "power": 0.8,
                                 "threats": ["warm-cache bias mitigated by cold-start runs"]},
                    "answers_aim": "the controlled comparison isolates throughput from cache effects, "
                                   "producing the bound aim-1 claims"}],
        "benefits": [{"id": "ben-1", "benefit": "lower settlement latency for regional members",
                      "type": "outcome", "beneficiary": "ACME credit-union members",
                      "owner": "ACME Cooperative", "timing": "by Y3",
                      "metric": "median settlement time down 40%", "preconditions": ["member onboarding"]}],
        "additionality": {"counterfactual": "without this grant ACME will not build the shared "
                                            "ledger because no single member captures the return alone",
                          "not_business_as_usual": True,
                          "leverage": {"grant": 300000, "co_contribution": 350000, "currency": "AUD"}},
        "risks": [{"id": "risk-1", "risk": "data-sharing MOU delayed", "likelihood": "medium",
                   "impact": "high", "trigger": "if the MOU is unsigned by month 6",
                   "monitoring": "quarterly steering review",
                   "contingency": "switch to the synthetic cohort prepared in month 3",
                   "owner": "CI Rivera"}],
    }


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
    ev_p = _write(tmp, "evidence.yaml", __import__("yaml").safe_dump(_evidence()))
    values_ok = _write(tmp, "vok.yaml", __import__("yaml").safe_dump(
        {"for": [{"pct": 60}, {"pct": 40}], "cond": False}))
    # values_full also fills the scored narrative n1 → criterion C1 substantiation-ready
    values_full = _write(tmp, "vfull.yaml", __import__("yaml").safe_dump(
        {"for": [{"pct": 60}, {"pct": 40}], "cond": False,
         "n1": "Prior work establishes the baseline this project extends."}))

    # 1. clean end-to-end (draft) → exit 0, C1 substantiated from evidence + content
    code = orchestrate(scheme_p, values_full, ev_p, ent_p, budget, paste_ok)
    assert code == 0, f"clean IR should pass, got exit {code}"
    # 1b. clean in SUBMISSION mode → still exit 0 (every scored criterion substantiated)
    code = orchestrate(scheme_p, values_full, ev_p, ent_p, budget, paste_ok, mode="submission")
    assert code == 0, f"clean IR must pass in submission mode, got exit {code}"

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
    # 10. criterion-readiness — a scored criterion with NO backing evidence/content:
    #     submission mode FAILs it (the key fix), draft mode only WARNs; evidenced → PASS.
    def readiness(sch, vals, ev, mode):
        r = Report(); check_criterion_readiness(r, sch, vals, ev, mode); return r.entries
    sub = readiness(scheme, {}, {}, "submission")
    assert any(e[1] == "FAIL" and e[0].startswith("criterion-readiness[") for e in sub), \
        "unevidenced scored criterion must FAIL in submission mode"
    drf = readiness(scheme, {}, {}, "draft")
    assert (any(e[1] == "WARN" and e[0].startswith("criterion-readiness[") for e in drf)
            and not any(e[1] == "FAIL" for e in drf)), "same criterion only WARNs in draft mode"
    good = readiness(scheme, {"n1": "Prior work backs this."}, _evidence(), "submission")
    assert any(e[1] == "PASS" and e[0].startswith("criterion-readiness[") for e in good), \
        "evidenced + written scored criterion must PASS (substantiated)"

    # 11. partner-commitment-reconciliation — call the check directly (mirrors #10):
    #     a fully-reconciled partner PASSes; a mismatched partner (letter cash 100k vs
    #     budget line 120k) FAILs in submission, only WARNs in draft; an unverified partner
    #     (cash line, no letter_commitment, no provenance) FAILs submission, WARNs draft.
    def recon(partners, bdata, mode):
        r = Report()
        check_partner_commitment_reconciliation(r, scheme, {"partners": partners}, bdata, mode)
        return r.entries

    good_partner = {"id": "acme", "status": "committed", "provenance": "corpus/loi-acme.pdf",
                    "contributions": {"cash": {"fy1": 100000}, "in_kind": {"fy1": 60000}},
                    "letter_commitment": {"cash": {"value": 100000, "conditional": False},
                                          "in_kind": {"value": 60000, "conditional": False}}}
    good_budget = {"rows": [
        {"funding_source": "co-contribution", "kind": "cash", "partner": "acme", "years": {2026: 100000}},
        {"funding_source": "co-contribution", "kind": "in-kind", "partner": "acme", "years": {2026: 60000}}]}
    assert any(e[1] == "PASS" and e[0].startswith("partner-commitment-reconciliation[")
               for e in recon([good_partner], good_budget, "submission")), \
        "fully-reconciled partner must PASS"

    bad_partner = {"id": "beacon", "status": "committed", "provenance": "corpus/loi-beacon.pdf",
                   "contributions": {"cash": {"fy1": 100000}},
                   "letter_commitment": {"cash": {"value": 100000, "conditional": False}}}
    bad_budget = {"rows": [
        {"funding_source": "co-contribution", "kind": "cash", "partner": "beacon", "years": {2026: 120000}}]}
    assert any(e[1] == "FAIL" and e[0].startswith("partner-commitment-reconciliation[")
               for e in recon([bad_partner], bad_budget, "submission")), \
        "letter cash 100k vs budget 120k must FAIL in submission mode"
    drf_p = recon([bad_partner], bad_budget, "draft")
    assert (any(e[1] == "WARN" for e in drf_p) and not any(e[1] == "FAIL" for e in drf_p)), \
        "same mismatch only WARNs in draft mode"

    unverified = {"id": "ghost", "contributions": {"cash": {"fy1": 50000}}}
    assert any(e[1] == "FAIL" for e in recon([unverified], None, "submission")), \
        "unverified partner (cash line, no letter, no provenance) must FAIL submission (fail-closed)"
    assert not any(e[1] == "FAIL" for e in recon([unverified], None, "draft")), \
        "unverified partner only WARNs in draft mode"

    conditional_partner = {"id": "vega", "status": "committed", "provenance": "corpus/loi-vega.pdf",
                           "contributions": {"cash": {"fy1": 80000}},
                           "letter_commitment": {"cash": {"value": 80000, "conditional": True}}}
    assert any(e[1] == "FAIL" for e in recon([conditional_partner], None, "submission")), \
        "conditional commitment rendered as status=committed must FAIL submission"

    # 12. process-dispatch — call the check directly (mirrors #10/#11): a valid multi-tag
    #     scheme with a routing field + rejoinder PASSes; an unknown tag FAILs (fail-closed);
    #     a staged scheme with no gating (EOI) phase FAILs submission / only WARNs draft.
    def dispatch(sch, m):
        r = Report(); check_process_dispatch(r, sch, m); return r.entries

    valid = _clean_scheme()
    valid["process"] = ["single-stage-review", "panel-routed"]
    valid["rejoinder"] = {"enabled": True}   # _clean_scheme carries a taxonomy/classification field
    vd = dispatch(valid, "submission")
    assert (any(e[1] == "PASS" and e[0].startswith("process-dispatch") for e in vd)
            and not any(e[1] == "FAIL" for e in vd)), \
        "valid multi-tag process (routing field + rejoinder) must PASS with no FAIL"

    unknown_proc = _clean_scheme(); unknown_proc["process"] = ["peer-reviewww"]
    assert any(e[1] == "FAIL" and e[0] == "process-dispatch"
               for e in dispatch(unknown_proc, "draft")), \
        "unknown process tag must FAIL (fail-closed) even in draft mode"

    staged = _clean_scheme(); staged["process"] = ["staged"]
    staged["submission"] = {"phases": ["full", "post-award"]}   # no EOI/pre-proposal/minimum-data
    assert any(e[1] == "FAIL" and e[0].startswith("process-dispatch[staged]")
               for e in dispatch(staged, "submission")), \
        "staged with no gating phase must FAIL in submission mode"
    staged_drf = dispatch(staged, "draft")
    assert (any(e[1] == "WARN" and e[0].startswith("process-dispatch[staged]") for e in staged_drf)
            and not any(e[1] == "FAIL" for e in staged_drf)), \
        "same staged scheme only WARNs in draft mode"

    # full doctored orchestrate → non-zero exit
    doctored = _write(tmp, "bad_scheme.yaml", __import__("yaml").safe_dump(bs))
    assert orchestrate(doctored, values_ok, None, ent_p, budget, paste_bad) == 1, "doctored IR must exit 1"
    # readiness gate end-to-end: clean scheme, unevidenced C1 → submission FAILs, draft passes
    assert orchestrate(scheme_p, values_ok, None, ent_p, budget, paste_ok, mode="submission") == 1, \
        "unevidenced scored criterion must fail submission-mode orchestrate"
    assert orchestrate(scheme_p, values_ok, None, ent_p, budget, paste_ok, mode="draft") == 0, \
        "same IR passes (WARN only) in draft mode"

    print("\nself-test OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scheme", help="scheme.yaml (Stage-A IR) — required unless --self-test")
    ap.add_argument("--values", help="values.yaml (filled field content sidecar)")
    ap.add_argument("--evidence", help="evidence-store.yaml")
    ap.add_argument("--entity", help="entity-store.yaml (partners/contributions)")
    ap.add_argument("--budget", help="budget.yaml (delegated to validate_budget.py)")
    ap.add_argument("--plan", help="project-plan.yaml (aims/design, benefits, additionality, "
                                   "risks registers — prospective-project mode)")
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
                       args.budget, args.paste_ready, args.mode, args.plan)


if __name__ == "__main__":
    sys.exit(main())
