# /// script
# requires-python = ">=3.9"
# dependencies = ["pyyaml"]
# ///
"""IR-level integrity orchestrator (Stage E: portal validation dry-run) — FAIL-CLOSED.

The single pre-submit gate for the cross-field couplings the docs acknowledged but no shipped
script checked (Codex top-3 #3; Phase-2's F.2↔H.1↔matched "hardest thing to model"). Reads the
`scheme.yaml` IR (+ optional values/evidence/entity/budget/paste-ready/plan sidecars) and runs 20
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

Checks 17–18 are the ROPE / track-record PRESENTATION-layer passes. Each is gated on its store
block (institutional_support from `--entity`; outputs_context from `--evidence`, narrative-award
mode) and fail-closed — a present-but-unreconciled total or an unsourced superlative is never
green-washed. `--mode submission` FAILs, `--mode draft` WARNs (mirrors partner-commitment).

 17. institutional-support the host-institution statement's committed support reconciles:
                        sum(items[].value) == total.value (±1%), every committed VALUED item has
                        provenance, and (when --budget declares them) total.value matches the
                        budget's non-ARC/institutional contribution lines; else FAILs submission (per org).
 18. outputs-context-completeness (narrative-award) every career_best.ids entry sits in ≥1 named
                        cluster and every cluster primacy.claim carries an attributor; every
                        clusters[].outputs / career_best.ids entry resolves to exactly one
                        publications[].id (a dangling or duplicated id FAILs); an
                        uncontextualised output or an unsourced superlative FAILs submission
                        (per output / per cluster).

Check 19 is a `prospective-project`-mode pass gated on a traceability spine in the `--plan`
sidecar (objectives/tasks/outputs/validations); otherwise a labelled SKIP. When the spine IS
present it is fail-closed — every id must resolve. `--mode submission` FAILs, `--mode draft` WARNs.

 19. traceability-spine     referential integrity + four-way crosswalk over the --plan spine:
                        every objectives[].aim→aims[].id, tasks[].objective→objectives[].id,
                        tasks[].depends_on→a task id, subtasks[].output→outputs[].id,
                        outputs[].task→a task id, outputs[].benefit→benefits[].id (when declared),
                        validations[].task→a task id resolves; every task carries ≥1 person AND
                        ≥1 years (person→investigators[].id when --entity supplied); with --budget
                        each tasks[].budget_lines→a budget row id AND every non-institutional
                        budget row is referenced by ≥1 task; no id is duplicated. A broken edge /
                        dangling / duplicate / unstaffed / unfunded task FAILs submission (per edge).

Check 20 is the requirement-coverage join over the scheme's graded obligation model
(`requirements[]`, the CFP's must/should/desirable logic) and the `--plan` objectives/tasks/
outputs that carry `addresses: [req-ids]`; gated on requirements[] present (else a labelled SKIP).

 20. requirement-coverage  for each requirements[] entry whose `applies_if` predicate holds
                        (evaluated against supplied classification values — an UNKNOWN or
                        unparseable predicate is fail-closed → the requirement is treated as
                        applicable), resolve the plan nodes whose `addresses` contains the req id.
                        A lone req needs ≥1 addressing node; an `alternatives` group under
                        `quantifier: at_least_one` is met by ANY member, under `all` by every
                        member. A mandatory/expected requirement (or an unmet at_least_one/all
                        group) with NO addressing FAILs submission / WARNs draft (per req, naming
                        the id + text); a desirable/optional gap is an informational WARN that
                        never blocks. An adjunct `domain-review` WARN surfaces any criterion/claim
                        tagged `needs_domain_review` with no recorded sign-off — a
                        route-to-specialist flag (§4.6), never a silent pass.

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

# ── first dispatch axis (Stage A0): instrument / register / deliverable classification ──
# Three ORTHOGONAL facets set at intake, BEFORE mode/process, that route the pipeline:
#   instrument (award|grant) → which deliverables to build (an award funds no project → none);
#   register  (industrial|academic) → the plainness dial (industry-partnered vs pure-academic —
#             orthogonal to funder-family: ARC LP is ARC yet industrial, ARC DP/DECRA academic);
#   requires  → the closed set of deliverables the scheme actually demands. This DECOUPLES the
#             budget/plan machinery from `mode`: an ARC DECRA is mode=narrative-award yet
#             instrument=grant with a budget + work_plan, so it must run the project passes.
INSTRUMENT_VOCAB = {"award", "grant"}
REGISTER_VOCAB = {"industrial", "academic"}
DELIVERABLE_VOCAB = {"budget", "work_plan", "in_kind", "stipend", "co_contribution"}

TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
KW = {"and", "or", "not", "True", "False", "None", "in", "is"}
# requirements[].applies_if — a SAFE minimal `<field> ==|!= <value>` predicate (never eval())
APPLIES_IF = re.compile(r"^\s*([\w.]+)\s*(==|!=)\s*(.+?)\s*$")
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


# ── the 20 checks ────────────────────────────────────────────────────────────
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


def check_classification(rep, scheme):
    """#21 Stage-A0 dispatch classification — validate the `classification` block that routes the
    pipeline BEFORE drafting. Three ORTHOGONAL facets: `instrument` (award|grant — the deliverable
    axis), `register` (industrial|academic — the plainness dial, orthogonal to funder-family:
    ARC LP is ARC yet industrial), `funder_family`; plus a `requires` list naming the deliverables
    to build. Gate: run only when the scheme declares a `classification` block (legacy IRs SKIP —
    the pipeline falls back to `mode`). Fail-closed: an unknown instrument/register, a `requires`
    entry outside the closed deliverable vocabulary, or an AWARD that nonetheless requires a
    budget/plan (a contradiction — an award funds no project) is a hard FAIL. A GRANT that requires
    nothing to build is a soft WARN (unusual — confirm it truly has no budget/work_plan).
    """
    cls = scheme.get("classification")
    if not isinstance(cls, dict) or not cls:
        rep.add("classification", "SKIP", "soft",
                "scheme declares no A0 classification block — pipeline falls back to `mode`")
        return
    problems = []
    instrument = cls.get("instrument")
    if instrument not in INSTRUMENT_VOCAB:
        problems.append(f"instrument {instrument!r} not in {sorted(INSTRUMENT_VOCAB)}")
    register = cls.get("register")
    if register is not None and register not in REGISTER_VOCAB:
        problems.append(f"register {register!r} not in {sorted(REGISTER_VOCAB)}")
    req = cls.get("requires")
    if req is not None and not isinstance(req, list):
        problems.append(f"requires {req!r} is not a list")
        req = None
    unknown = [d for d in (req or []) if d not in DELIVERABLE_VOCAB]
    if unknown:
        problems.append(f"requires has unknown deliverable(s) {unknown} "
                        f"not in {sorted(DELIVERABLE_VOCAB)}")
    if problems:
        rep.add("classification", "FAIL", "hard", "; ".join(problems) + " (fail-closed)")
        return
    # consistency: an award funds no project → must require none; a grant normally builds ≥1
    if instrument == "award" and req:
        rep.add("classification", "FAIL", "hard",
                f"instrument=award but requires {req} — an award funds no project, so it needs no "
                "budget/work_plan/in_kind/stipend (fail-closed)")
        return
    if instrument == "grant" and not req:
        rep.add("classification", "WARN", "soft",
                "instrument=grant but requires is empty — a grant normally builds at least a "
                "budget or work_plan; confirm this scheme genuinely demands neither")
        return
    reg = f", register {register}" if register else ""
    fam = f", funder_family {cls.get('funder_family')}" if cls.get("funder_family") else ""
    rep.add("classification", "PASS", "hard",
            f"instrument {instrument}{reg}{fam} — requires {req or []}")


# ── project-substance passes (checks 13–16, gated on classification.requires[work_plan]) ──
def _nonempty(v):
    """Fail-closed truthiness: a present-but-empty field (None, "", "  ") is NOT satisfied."""
    if v is None:
        return False
    if isinstance(v, str):
        return bool(v.strip())
    return bool(v)


def _scheme_requires(scheme, deliverable):
    """True when the scheme's A0 classification requires a deliverable (budget/work_plan/…).

    Prefers the explicit `classification.requires` list (Stage-A0 dispatch, the INSTRUMENT axis);
    falls back to the legacy `mode == prospective-project` heuristic for schemes that predate the
    classification block, so existing IRs keep working unchanged. This is what decouples the
    budget/plan machinery from `mode`: an ARC DECRA is `mode: narrative-award` yet
    `instrument: grant` with a budget + work_plan, and must run the project-substance passes —
    the old `mode == prospective-project` gate wrongly SKIPped them.
    """
    cls = scheme.get("classification") or {}
    req = cls.get("requires")
    if isinstance(req, list):
        return deliverable in req
    return scheme.get("mode") == "prospective-project"


def _project_gate(rep, check, scheme, plan):
    """Gate a project-substance check on the scheme requiring a work_plan + a supplied --plan.

    Now gated on the A0 classification's `requires: [work_plan]` (with a legacy `mode` fallback),
    NOT on `mode == prospective-project` — so a narrative-award-MODE scheme that nonetheless funds
    a project (e.g. ARC DECRA) still has its plan assessed. Returns True when the check should run;
    otherwise emits a labelled SKIP and returns False.
    """
    if not _scheme_requires(scheme, "work_plan"):
        cls = scheme.get("classification") or {}
        why = (f"classification.instrument {cls.get('instrument', '?')!r} does not require a work_plan"
               if cls.get("requires") is not None
               else f"scheme mode {scheme.get('mode', '?')!r} ≠ prospective-project")
        rep.add(check, "SKIP", "soft", f"{why} — no project plan to assess")
        return False
    if not plan:
        rep.add(check, "SKIP", "soft",
                "scheme requires a work_plan but no --plan project-plan.yaml supplied")
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


# ── ROPE / track-record presentation-layer passes (checks 17–18) ──────────────
def _institutional_support_orgs(entity):
    """Yield (org_id, institutional_support dict) for every org carrying the block."""
    out = []
    for o in (entity.get("organizations") or []):
        if isinstance(o, dict) and isinstance(o.get("institutional_support"), dict):
            out.append((o.get("id") or o.get("name") or "<org>", o["institutional_support"]))
    return out


def _budget_institutional_total(bdata):
    """Sum of the budget's non-ARC / institutional (host co-investment) contribution rows, or
    None when the budget declares no such line (so absence never fabricates a mismatch)."""
    if not bdata:
        return None
    srcs = {"institutional", "institution", "host", "host-institution"}
    rows = [r for r in (bdata.get("rows") or []) if str(r.get("funding_source")) in srcs]
    if not rows:
        return None
    return sum(sum(float(v) for v in (r.get("years") or {}).values()) for r in rows)


def check_institutional_support_reconciliation(rep, scheme, entity, bdata, mode):
    """#17 host-institution statement (batch-2 partner-reconciliation analog): the stated
    `total.value` reconciles with sum(items[].value) within 1%; every `status: committed`
    item that carries a monetary value has non-empty `provenance` (a null-value in-kind line
    such as teaching-relief is backed by its `basis`, not a dollar figure); and when a budget
    is supplied AND declares institutional contribution rows, `total.value` reconciles with
    that non-ARC/institutional line total (>1%). A total mismatch, a stated total absent while
    items are present, or a committed valued item with no provenance is a hard FAIL in
    `submission` mode; in `draft` mode each surfaces as WARN. Gate: run only when an
    organizations[].institutional_support block is present, else a labelled SKIP.
    """
    blocks = _institutional_support_orgs(entity)
    if not blocks:
        rep.add("institutional-support-reconciliation", "SKIP", "hard",
                "no organizations[].institutional_support block to reconcile")
        return
    fmt = lambda v: f"{v:.0f}" if v is not None else "—"
    b_inst = _budget_institutional_total(bdata)
    for oid, sup in blocks:
        items = sup.get("items") or []
        total = sup.get("total") if isinstance(sup.get("total"), dict) else {}
        total_value = total.get("value")
        item_sum = sum(float(it.get("value")) for it in items
                       if isinstance(it, dict) and it.get("value") is not None)

        problems = []
        if total_value is None:
            problems.append(f"items present (sum {item_sum:.0f}) but no stated total.value to "
                            "reconcile — UNRECONCILED")
        elif _pct_mismatch(item_sum, float(total_value)):
            problems.append(f"sum(items) {item_sum:.0f} ≠ stated total {float(total_value):.0f} (>1%)")
        for it in items:
            if not isinstance(it, dict):
                continue
            if (str(it.get("status", "")).lower() == "committed"
                    and it.get("value") is not None and not _nonempty(it.get("provenance"))):
                problems.append(f"committed item {it.get('kind', '<item>')!r} "
                                f"(value {float(it['value']):.0f}) has no provenance")
        if b_inst is not None and total_value is not None and _pct_mismatch(float(total_value), b_inst):
            problems.append(f"stated total {float(total_value):.0f} ≠ budget institutional "
                            f"contribution {b_inst:.0f} (>1%)")

        if not problems:
            rep.add(f"institutional-support-reconciliation[{oid}]", "PASS", "hard",
                    f"reconciled — sum(items) {fmt(item_sum)} = stated total {fmt(total_value)}"
                    + (f", budget institutional {fmt(b_inst)}" if b_inst is not None else ""))
        elif mode == "submission":
            rep.add(f"institutional-support-reconciliation[{oid}]", "FAIL", "hard",
                    "; ".join(problems) + " (fail-closed, submission mode)")
        else:
            rep.add(f"institutional-support-reconciliation[{oid}]", "WARN", "soft",
                    "; ".join(problems))


def check_outputs_context_completeness(rep, scheme, evidence, mode):
    """#18 ROPE field-calibration completeness (narrative-award): every career-best output is
    placed in a named cluster, and every cluster primacy claim is sourced. Gate on scheme
    mode == narrative-award AND an evidence-store `outputs_context` block (else a labelled
    SKIP). Each `career_best.ids` entry must appear in ≥1 `clusters[].outputs`; each cluster
    carrying a non-empty `primacy.claim` must carry a non-empty `primacy.attributor` (else the
    superlative is unsourced). Every `clusters[].outputs` id and `career_best.ids` entry must
    resolve to EXACTLY ONE `publications[].id` in the evidence store — a dangling (no match) or
    duplicated (>1 match) id is fail-closed. A missing cluster placement (per output), an
    unsourced primacy (per cluster), or an unresolved output id is a hard FAIL in `submission`, a
    WARN in `draft`. Fail-closed.
    """
    if scheme.get("mode") != "narrative-award":
        rep.add("outputs-context-completeness", "SKIP", "soft",
                f"scheme mode {scheme.get('mode', '?')!r} ≠ narrative-award — no outputs-context to assess")
        return
    oc = evidence.get("outputs_context") if isinstance(evidence, dict) else None
    if not isinstance(oc, dict) or not oc:
        rep.add("outputs-context-completeness", "SKIP", "soft",
                "narrative-award mode but no evidence-store outputs_context block supplied")
        return
    clusters = oc.get("clusters") or []
    clustered = set()
    for c in clusters:
        if isinstance(c, dict):
            for o in (c.get("outputs") or []):
                clustered.add(str(o))
    best_ids = [str(i) for i in ((oc.get("career_best") or {}).get("ids") or [])]

    status, binding = ("FAIL", "hard") if mode == "submission" else ("WARN", "soft")
    tail = " (fail-closed, submission mode)" if mode == "submission" else ""
    any_entry = False
    # coverage: every career-best id appears in ≥1 named cluster
    for oid in best_ids:
        any_entry = True
        if oid in clustered:
            rep.add(f"outputs-context-completeness[{oid}]", "PASS", "hard",
                    "career-best output placed in ≥1 named cluster")
        else:
            rep.add(f"outputs-context-completeness[{oid}]", status, binding,
                    f"career-best output {oid!r} appears in no clusters[].outputs — "
                    "uncontextualised" + tail)
    # sourced primacy: a cluster stating a primacy.claim needs a primacy.attributor
    for c in clusters:
        if not isinstance(c, dict):
            continue
        prim = c.get("primacy") or {}
        thread = c.get("thread") or "<cluster>"
        if isinstance(prim, dict) and _nonempty(prim.get("claim")):
            any_entry = True
            if _nonempty(prim.get("attributor")):
                rep.add(f"outputs-context-completeness[{thread}]", "PASS", "hard",
                        "primacy claim carries an attributor (sourced)")
            else:
                rep.add(f"outputs-context-completeness[{thread}]", status, binding,
                        f"cluster {thread!r} states a primacy claim with no attributor — "
                        "unsourced superlative" + tail)
    # one-to-one resolution: every referenced output id → exactly one publications[].id
    pub_counts = {}
    for pub in (evidence.get("publications") or []):
        if isinstance(pub, dict) and pub.get("id") is not None:
            pub_counts[str(pub["id"])] = pub_counts.get(str(pub["id"]), 0) + 1
    for rid in sorted(clustered | set(best_ids)):
        any_entry = True
        n = pub_counts.get(rid, 0)
        if n == 1:
            rep.add(f"outputs-context-completeness[{rid}]", "PASS", "hard",
                    "output id resolves to exactly one publications[].id")
        elif n == 0:
            rep.add(f"outputs-context-completeness[{rid}]", status, binding,
                    f"output id {rid!r} resolves to no publications[].id — dangling reference" + tail)
        else:
            rep.add(f"outputs-context-completeness[{rid}]", status, binding,
                    f"output id {rid!r} resolves to {n} publications[].id entries — "
                    "duplicated/ambiguous" + tail)
    if not any_entry:
        rep.add("outputs-context-completeness", "SKIP", "soft",
                "outputs_context present but declares no career_best ids and no cluster primacy claims")


def check_traceability_spine(rep, scheme, plan, entity, bdata, mode):
    """#19 traceability spine & cross-field crosswalk (prospective-project mode). Gated on
    prospective-project mode + a spine present in --plan (objectives/tasks/outputs/validations);
    else a labelled SKIP. When the spine IS present it is fail-closed — every id must resolve.

    Referential integrity: every objectives[].aim → aims[].id; tasks[].objective →
    objectives[].id; tasks[].depends_on → a task id; subtasks[].output → outputs[].id;
    outputs[].task → a task id; outputs[].benefit → benefits[].id (when the output declares one);
    validations[].task → a task id. Crosswalk: every task carries ≥1 person AND ≥1 years; with
    --entity, each person → investigators[].id; with --budget, each tasks[].budget_lines → a
    budget row id AND (reverse) every non-institutional budget row is referenced by ≥1 task. No id
    is duplicated across the spine. Each broken edge / dangling / duplicate / unstaffed / unfunded
    is a hard FAIL (per edge, naming the ids) in `submission`, a WARN in `draft`.
    """
    if not _project_gate(rep, "traceability-spine", scheme, plan):
        return
    objectives = plan.get("objectives") or []
    tasks = plan.get("tasks") or []
    outputs = plan.get("outputs") or []
    validations = plan.get("validations") or []
    if not (objectives or tasks or outputs or validations):
        rep.add("traceability-spine", "SKIP", "soft",
                "prospective-project plan declares no traceability spine "
                "(objectives/tasks/outputs/validations)")
        return

    def ids_of(rows):
        return [str(r.get("id")) for r in rows if isinstance(r, dict) and r.get("id") is not None]

    aim_ids = set(ids_of(plan.get("aims") or []))
    ben_ids = set(ids_of(plan.get("benefits") or []))
    obj_ids = set(ids_of(objectives))
    task_ids = set(ids_of(tasks))
    out_ids = set(ids_of(outputs))
    inv_ids = set(ids_of(entity.get("investigators") or [])) if entity else set()

    problems = []

    # no duplicate ids across the spine (objectives/tasks/subtasks/outputs/validations)
    seen = {}
    for r in objectives + tasks + outputs + validations:
        if isinstance(r, dict) and r.get("id") is not None:
            seen[str(r["id"])] = seen.get(str(r["id"]), 0) + 1
    for t in tasks:
        for st in (t.get("subtasks") or []) if isinstance(t, dict) else []:
            if isinstance(st, dict) and st.get("id") is not None:
                seen[str(st["id"])] = seen.get(str(st["id"]), 0) + 1
    for i in sorted(k for k, c in seen.items() if c > 1):
        problems.append(f"duplicate id {i!r} appears {seen[i]}× across the spine")

    # objectives[].aim → aims[].id
    for o in objectives:
        if not isinstance(o, dict):
            continue
        aim = o.get("aim")
        if aim is not None and str(aim) not in aim_ids:
            problems.append(f"objective {o.get('id')!r} → aim {aim!r} resolves to no aims[].id")

    # tasks[].objective / depends_on / subtasks[].output; staffing + timing + person→investigators
    for t in tasks:
        if not isinstance(t, dict):
            continue
        tid = t.get("id")
        obj = t.get("objective")
        if obj is not None and str(obj) not in obj_ids:
            problems.append(f"task {tid!r} → objective {obj!r} resolves to no objectives[].id")
        for dep in (t.get("depends_on") or []):
            if str(dep) not in task_ids:
                problems.append(f"task {tid!r} depends_on {dep!r} resolves to no task id")
        for st in (t.get("subtasks") or []):
            if not isinstance(st, dict):
                continue
            out = st.get("output")
            if out is not None and str(out) not in out_ids:
                problems.append(f"subtask {st.get('id')!r} → output {out!r} resolves to no outputs[].id")
        persons = t.get("person") or []
        persons = persons if isinstance(persons, list) else [persons]
        years = t.get("years") or []
        years = years if isinstance(years, list) else [years]
        if not persons:
            problems.append(f"task {tid!r} has no person — unstaffed")
        if not years:
            problems.append(f"task {tid!r} has no years — untimed")
        if entity:
            for p in persons:
                if str(p) not in inv_ids:
                    problems.append(f"task {tid!r} person {p!r} resolves to no investigators[].id")

    # outputs[].task → a task id; outputs[].benefit → benefits[].id (when declared)
    for o in outputs:
        if not isinstance(o, dict):
            continue
        oid = o.get("id")
        tsk = o.get("task")
        if tsk is not None and str(tsk) not in task_ids:
            problems.append(f"output {oid!r} → task {tsk!r} resolves to no task id")
        ben = o.get("benefit")
        if ben is not None and str(ben) not in ben_ids:
            problems.append(f"output {oid!r} → benefit {ben!r} resolves to no benefits[].id")

    # validations[].task → a task id
    for v in validations:
        if not isinstance(v, dict):
            continue
        tsk = v.get("task")
        if tsk is not None and str(tsk) not in task_ids:
            problems.append(f"validation {v.get('id')!r} → task {tsk!r} resolves to no task id")

    # four-way crosswalk to budget (when --budget supplied): forward + reverse
    if bdata is not None:
        rows = [r for r in (bdata.get("rows") or []) if isinstance(r, dict)]
        inst = {"institutional", "institution", "host", "host-institution"}
        row_ids = {str(r.get("id")) for r in rows if r.get("id") is not None}
        referenced = set()
        for t in tasks:
            if not isinstance(t, dict):
                continue
            for bl in (t.get("budget_lines") or []):
                referenced.add(str(bl))
                if str(bl) not in row_ids:
                    problems.append(f"task {t.get('id')!r} budget_line {bl!r} resolves to no budget row id")
        for r in rows:
            if str(r.get("funding_source")) in inst:
                continue
            rid = r.get("id")
            if rid is None:
                problems.append(f"non-institutional budget row {r.get('category', '<row>')!r} "
                                "carries no id — cannot cross-reference to a task")
            elif str(rid) not in referenced:
                problems.append(f"budget row {str(rid)!r} is referenced by no task — unfunded/orphan line")

    if not problems:
        rep.add("traceability-spine", "PASS", "hard",
                f"spine resolves — {len(objectives)} objective(s), {len(tasks)} task(s), "
                f"{len(outputs)} output(s), {len(validations)} validation(s); "
                "every edge + crosswalk intact")
        return
    status, binding = ("FAIL", "hard") if mode == "submission" else ("WARN", "soft")
    tail = " (fail-closed, submission mode)" if mode == "submission" else ""
    for p in problems:
        rep.add("traceability-spine", status, binding, p + tail)


# ── requirement-coverage + domain-review (check 20 + GAP-3 light adjunct) ─────
def _classification_map(scheme, values, plan):
    """Collect classification field → value from a `classification` block (plan or values)
    and any scheme classification-role field carrying a value in --values. Keyed by name."""
    cm = {}
    for src in ((plan or {}).get("classification"), (values or {}).get("classification")):
        if isinstance(src, dict):
            for k, v in src.items():
                cm.setdefault(str(k), v)
    for f in iter_field_nodes((scheme or {}).get("sections")):
        if "classification" in roles_of(f) and values and fid(f) in values:
            cm.setdefault(str(fid(f)), values[fid(f)])
    return cm


def _eval_applies_if(pred, cmap):
    """SAFE minimal evaluator for a `<field> ==|!= <value>` predicate — NEVER eval(). Returns
    (applies: bool, note: str|None). Fail-closed: an omitted predicate always applies; an
    unparseable predicate or an unknown classification field is treated as APPLICABLE (a note
    records why), so a conditional obligation is never silently skipped."""
    if pred is None or (isinstance(pred, str) and not pred.strip()):
        return True, None
    m = APPLIES_IF.match(str(pred))
    if not m:
        return True, f"applies_if {pred!r} unparseable — treated as applicable (fail-closed)"
    field, op, rhs = m.group(1), m.group(2), m.group(3).strip().strip("\"'")
    key = field.split(".")[-1]
    if key in cmap:
        actual = cmap[key]
    elif field in cmap:
        actual = cmap[field]
    else:
        return True, f"classification {field!r} unknown — treated as applicable (fail-closed)"
    hit = str(actual).strip() == rhs
    return (hit if op == "==" else not hit), None


def check_requirement_coverage(rep, scheme, plan, values, mode):
    """#20 requirement-coverage: join the scheme's graded obligation model (`requirements[]`,
    parsed from the CFP) against the project-plan nodes (objectives/tasks/outputs) that carry
    `addresses: [req-ids]` — a scheme STATES the obligation, the plan MEETS it. For each
    requirement whose `applies_if` predicate holds (evaluated against supplied classification
    values; an unknown/unparseable predicate is fail-closed → the requirement is treated as
    applicable), the group is met when the addressing plan node(s) exist: a lone req needs ≥1
    addressing node; an `alternatives` group under `quantifier: at_least_one` needs ANY member
    addressed, under `all` needs EVERY member. A `mandatory`/`expected` requirement (or an unmet
    at_least_one/all group) with NO addressing is a hard FAIL in `submission`, a WARN in `draft`,
    naming the req id + text. A `desirable`/`optional` unmet requirement is an informational WARN
    that never blocks. Gate: run only when the scheme declares requirements[] (else a labelled
    SKIP). Absent a `strength`, a requirement defaults to mandatory (fail-closed).
    """
    reqs = scheme.get("requirements")
    if not reqs:
        rep.add("requirement-coverage", "SKIP", "soft",
                "scheme declares no requirements[] obligation model")
        return
    addressed = set()
    for coll in ("objectives", "tasks", "outputs"):
        for node in ((plan or {}).get(coll) or []):
            if isinstance(node, dict):
                for rid in (node.get("addresses") or []):
                    addressed.add(str(rid))
    cmap = _classification_map(scheme, values, plan)

    for r in reqs:
        if not isinstance(r, dict):
            continue
        rid = str(r.get("id") or "<req>")
        text = r.get("text") or ""
        strength = str(r.get("strength") or "mandatory").lower()
        applies, note = _eval_applies_if(r.get("applies_if"), cmap)
        if not applies:
            rep.add(f"requirement-coverage[{rid}]", "PASS", "hard",
                    f"applies_if {r.get('applies_if')!r} does not hold — requirement not applicable")
            continue
        alts = r.get("alternatives")
        if isinstance(alts, list) and alts:
            quant = str(r.get("quantifier") or "all").lower()
            members = [str(a) for a in alts]
            hit = [m for m in members if m in addressed]
            met = bool(hit) if quant == "at_least_one" else all(m in addressed for m in members)
            qdesc = f"{quant} of {members}"
            addr_desc = f"addressed: {hit}" if hit else "no alternative addressed"
        else:
            met = rid in addressed
            qdesc = "lone requirement"
            addr_desc = "addressed" if met else "no plan node addresses it"
        notesfx = f"; {note}" if note else ""
        if met:
            rep.add(f"requirement-coverage[{rid}]", "PASS", "hard",
                    f"{strength} requirement met ({qdesc}) — {addr_desc}{notesfx}")
            continue
        detail = f"{strength} requirement unaddressed ({qdesc}): {text!r} — {addr_desc}"
        if strength in ("desirable", "optional"):
            rep.add(f"requirement-coverage[{rid}]", "WARN", "soft",
                    detail + " (informational, non-blocking)" + notesfx)
        elif mode == "submission":
            rep.add(f"requirement-coverage[{rid}]", "FAIL", "hard",
                    detail + " (fail-closed, submission mode)" + notesfx)
        else:
            rep.add(f"requirement-coverage[{rid}]", "WARN", "soft", detail + notesfx)


def check_domain_review(rep, scheme, plan):
    """GAP-3 light (§4.6 domain-adequacy hook): a criterion (scheme rubric) or plan claim/node
    tagged `needs_domain_review` (a discipline the skill cannot itself adjudicate — e.g.
    security-proofs / threat-model / consensus-safety / statistical-validity / clinical-trial-
    design) with NO recorded sign-off surfaces as a WARN (never a silent pass). It is a
    FLAG-and-ROUTE discipline, not domain knowledge — the skill routes the claim to a specialist
    and never presents it as validated. Minimal: one WARN naming the unsigned tags; SKIP when
    nothing is tagged.
    """
    unsigned = []

    def scan(items, kind):
        for it in (items or []):
            if not isinstance(it, dict) or not it.get("needs_domain_review"):
                continue
            signed = (it.get("domain_signoff") or it.get("domain_sign_off") or it.get("signoff"))
            if not _nonempty(signed):
                label = it.get("id") or it.get("criterion") or it.get("claim") or f"<{kind}>"
                unsigned.append(f"{label} ({it['needs_domain_review']})")

    scan(scheme.get("rubric"), "criterion")
    for coll in ("aims", "objectives", "tasks", "outputs", "claims"):
        scan((plan or {}).get(coll), coll)
    if not unsigned:
        rep.add("domain-review", "SKIP", "soft", "no criterion/claim tagged needs_domain_review")
        return
    rep.add("domain-review", "WARN", "soft",
            "needs_domain_review with no recorded sign-off — route to a specialist, do NOT "
            f"present as validated: {'; '.join(unsigned)}")


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
    check_classification(rep, scheme)
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
    check_institutional_support_reconciliation(rep, scheme, entity, bdata, mode)
    check_outputs_context_completeness(rep, scheme, evidence, mode)
    check_traceability_spine(rep, scheme, plan, entity, bdata, mode)
    check_requirement_coverage(rep, scheme, plan, values, mode)
    check_domain_review(rep, scheme, plan)
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


def _spine_plan():
    # A fully-resolved traceability spine (fictional ids only) extending the batch-4 registers:
    # every objective→aim, task→objective, depends_on→task, subtask→output, output→task/benefit,
    # validation→task edge resolves; every task is staffed (person→investigators) + timed + funded
    # (budget_lines→budget rows); no id is duplicated. Mirrors project-plan.template.yaml.
    return {
        "aims": [{"id": "aim-1", "statement": "Establish the shared-ledger throughput bound",
                  "success_criterion": "sustained ≥10k tx/s on the reference cluster"}],
        "benefits": [{"id": "ben-1", "benefit": "lower settlement latency", "owner": "ACME Cooperative",
                      "metric": "median settlement time down 40%", "timing": "by Y3"}],
        "objectives": [{"id": "obj-1", "aim": "aim-1", "statement": "prove the throughput bound"},
                       {"id": "obj-2", "aim": "aim-1", "statement": "sustain it under 10x load"}],
        "tasks": [
            {"id": "task-1", "objective": "obj-1", "statement": "derive the bound",
             "foundational": True, "depends_on": [],
             "subtasks": [{"id": "st-1.1", "statement": "no closed-form bound → cannot size the "
                           "cluster → derive one → sizing becomes deterministic", "output": "out-1"}],
             "person": ["inv-lead"], "years": [1], "budget_lines": ["bl-1"], "validation": "val-1"},
            {"id": "task-2", "objective": "obj-2", "statement": "build the load-replay harness",
             "foundational": False, "depends_on": ["task-1"],
             "subtasks": [{"id": "st-2.1", "statement": "batch ingest lacks backpressure → tail "
                           "latency spikes → add adaptive batching → bounds p99", "output": "out-2"}],
             "person": ["inv-lead"], "years": [2, 3], "budget_lines": ["bl-2"], "validation": "val-2"}],
        "outputs": [{"id": "out-1", "task": "task-1", "kind": "theory", "benefit": "ben-1"},
                    {"id": "out-2", "task": "task-2", "kind": "demonstrator"}],
        "validations": [
            {"id": "val-1", "task": "task-1", "baseline": "single-node baseline",
             "stress": "cold-start burst", "mechanism_check": "does the bound hold",
             "metric": "≥10k tx/s", "comparator_class": "scholarly"},
            {"id": "val-2", "task": "task-2", "baseline": "batch-ingest architecture",
             "stress": "10x burst replay", "mechanism_check": "does batching engage backpressure",
             "metric": "p99 < 800 ms", "comparator_class": "standard"}],
    }


def _spine_entity():
    # entity-store investigators[] the spine's task.person ids resolve into (fictional only).
    return {"investigators": [{"id": "inv-lead", "name": "Lead CI"}]}


def _spine_budget():
    # budget rows carrying stable ids the spine's tasks[].budget_lines resolve into; each
    # non-institutional row referenced by ≥1 task (closes the four-way crosswalk).
    return {"rows": [
        {"id": "bl-1", "category": "labour", "funding_source": "requested", "kind": "cash",
         "years": {2026: 100000}},
        {"id": "bl-2", "category": "equipment", "funding_source": "requested", "kind": "cash",
         "years": {2027: 50000}}]}


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

    # 13–16. project-substance passes (prospective-project mode + --plan). Call each check
    #     directly (mirrors #10/#11/#12): a complete plan PASSes; a broken register (uncovered
    #     aim / ownerless benefit / missing counterfactual / triggerless high-impact risk) FAILs
    #     submission and only WARNs draft; a present-but-empty field is never green-washed; a
    #     non-project scheme (or absent plan) SKIPs the whole pass.
    scheme_pp = _clean_scheme()          # _clean_scheme already sets mode: prospective-project
    plan = _plan()
    bud = load_yaml(budget)

    def design_adq(sch, pl, m):
        r = Report(); check_research_design_adequacy(r, sch, pl, m); return r.entries

    def benefits(sch, pl, m):
        r = Report(); check_benefits_realisation(r, sch, pl, m); return r.entries

    def addl(sch, pl, bd, m):
        r = Report(); check_additionality_vfm(r, sch, pl, bd, m); return r.entries

    def risks(sch, pl, m):
        r = Report(); check_risk_triggers(r, sch, pl, m); return r.entries

    # complete plan → all four PASS with no FAIL, even in submission mode
    for entries in (design_adq(scheme_pp, plan, "submission"),
                    benefits(scheme_pp, plan, "submission"),
                    risks(scheme_pp, plan, "submission")):
        assert any(e[1] == "PASS" for e in entries) and not any(e[1] == "FAIL" for e in entries), \
            "complete plan must PASS its substance pass"
    addl_ok = addl(scheme_pp, plan, bud, "submission")
    assert any(e[1] == "PASS" for e in addl_ok) and not any(e[1] == "FAIL" for e in addl_ok), \
        "complete additionality + matching budget leverage must PASS"

    # non-project scheme (and absent plan) → labelled SKIP, never a FAIL
    non_pp = _clean_scheme(); non_pp["mode"] = "single-applicant"
    assert any(e[1] == "SKIP" for e in design_adq(non_pp, plan, "submission")), \
        "non-project mode must SKIP the design-adequacy pass"
    assert any(e[1] == "SKIP" for e in risks(scheme_pp, {}, "submission")), \
        "absent --plan must SKIP the risk-triggers pass"

    # 13. research-design-adequacy — uncovered aim (design row removed) FAILs submission / WARNs draft
    p13 = _plan(); p13["design"] = []
    assert any(e[1] == "FAIL" and e[0].startswith("research-design-adequacy[")
               for e in design_adq(scheme_pp, p13, "submission")), "uncovered aim must FAIL submission"
    d13 = design_adq(scheme_pp, p13, "draft")
    assert any(e[1] == "WARN" for e in d13) and not any(e[1] == "FAIL" for e in d13), \
        "uncovered aim only WARNs in draft mode"

    # 14. benefits-realisation — a present-but-empty owner is fail-closed (not green-washed)
    p14 = _plan(); p14["benefits"][0]["owner"] = ""
    assert any(e[1] == "FAIL" and e[0].startswith("benefits-realisation[")
               for e in benefits(scheme_pp, p14, "submission")), "ownerless benefit must FAIL submission"
    d14 = benefits(scheme_pp, p14, "draft")
    assert any(e[1] == "WARN" for e in d14) and not any(e[1] == "FAIL" for e in d14), \
        "ownerless benefit only WARNs in draft mode"

    # 15. additionality-vfm — missing counterfactual FAILs submission / WARNs draft
    p15 = _plan(); p15["additionality"]["counterfactual"] = None
    assert any(e[1] == "FAIL" and e[0] == "additionality-vfm"
               for e in addl(scheme_pp, p15, None, "submission")), "missing counterfactual must FAIL submission"
    d15 = addl(scheme_pp, p15, None, "draft")
    assert any(e[1] == "WARN" for e in d15) and not any(e[1] == "FAIL" for e in d15), \
        "missing counterfactual only WARNs in draft mode"
    #     leverage/budget mismatch (>1%) FAILs submission (like partner-commitment)
    p15b = _plan(); p15b["additionality"]["leverage"]["co_contribution"] = 999999
    assert any(e[1] == "FAIL" and e[0] == "additionality-vfm"
               for e in addl(scheme_pp, p15b, bud, "submission")), \
        "leverage co-contribution vs budget mismatch must FAIL submission"

    # 16. risk-triggers — a triggerless high-impact risk (trigger: null) FAILs submission / WARNs draft
    p16 = _plan(); p16["risks"][0]["trigger"] = None
    assert any(e[1] == "FAIL" and e[0].startswith("risk-triggers[")
               for e in risks(scheme_pp, p16, "submission")), "triggerless high-impact risk must FAIL submission"
    d16 = risks(scheme_pp, p16, "draft")
    assert any(e[1] == "WARN" for e in d16) and not any(e[1] == "FAIL" for e in d16), \
        "triggerless high-impact risk only WARNs in draft mode"

    # 17. institutional-support-reconciliation — call directly (mirrors #11): a reconciled
    #     host-institution statement PASSes; a total≠sum(items) mismatch, or a committed VALUED
    #     item with no provenance, FAILs submission / WARNs draft; no block → SKIP. Fictional
    #     orgs only (ACME University; a null-value teaching-relief line is exempt from provenance).
    def instrecon(orgs, bdata, m):
        r = Report()
        check_institutional_support_reconciliation(r, scheme, {"organizations": orgs}, bdata, m)
        return r.entries

    good_org = {"id": "org-lead", "name": "ACME University", "role": "lead",
                "institutional_support": {
                    "items": [
                        {"kind": "establishment-grant", "value": 120000, "currency": "AUD",
                         "status": "committed", "provenance": "corpus/host-statement.pdf"},
                        {"kind": "stipend-topup", "value": 30000, "currency": "AUD",
                         "status": "committed", "provenance": "corpus/host-statement.pdf"},
                        {"kind": "teaching-relief", "value": None,
                         "basis": "reduced load from salary savings", "status": "committed"}],
                    "total": {"value": 150000, "currency": "AUD"},
                    "statement_provenance": "corpus/host-statement.pdf"}}
    assert any(e[1] == "PASS" and e[0].startswith("institutional-support-reconciliation[")
               for e in instrecon([good_org], None, "submission")), \
        "reconciled institutional_support must PASS"

    bad_org = {"id": "org-lead", "name": "ACME University", "role": "lead",
               "institutional_support": {
                   "items": [
                       {"kind": "establishment-grant", "value": 120000, "status": "committed",
                        "provenance": "corpus/host-statement.pdf"},
                       {"kind": "stipend-topup", "value": 30000, "status": "committed",
                        "provenance": "corpus/host-statement.pdf"}],
                   "total": {"value": 200000, "currency": "AUD"}}}   # 150000 ≠ 200000
    assert any(e[1] == "FAIL" and e[0].startswith("institutional-support-reconciliation[")
               for e in instrecon([bad_org], None, "submission")), \
        "total ≠ sum(items) must FAIL submission"
    d17 = instrecon([bad_org], None, "draft")
    assert any(e[1] == "WARN" for e in d17) and not any(e[1] == "FAIL" for e in d17), \
        "same total mismatch only WARNs in draft mode"

    noprov_org = {"id": "org-lead", "institutional_support": {
        "items": [{"kind": "establishment-grant", "value": 150000, "status": "committed"}],
        "total": {"value": 150000}}}
    assert any(e[1] == "FAIL" for e in instrecon([noprov_org], None, "submission")), \
        "committed valued item with no provenance must FAIL submission (fail-closed)"
    assert any(e[1] == "SKIP" for e in instrecon([{"id": "org-lead", "name": "ACME University"}],
                                                 None, "submission")), \
        "org with no institutional_support block must SKIP"

    # 18. outputs-context-completeness — call directly (mirrors #11): a fully clustered +
    #     sourced-primacy outputs_context PASSes under narrative-award; an unclustered
    #     career-best id, or a cluster primacy.claim with no attributor, FAILs submission /
    #     WARNs draft. Non-narrative-award mode (or absent block) SKIPs. Fictional ids only.
    na_scheme = _clean_scheme(); na_scheme["mode"] = "narrative-award"

    def outctx(sch, ev, m):
        r = Report(); check_outputs_context_completeness(r, sch, ev, m); return r.entries

    good_oc = {
        "publications": [{"id": "C1", "title": "A backed result", "year": 2024},
                         {"id": "J1", "title": "A journal result", "year": 2023},
                         {"id": "J2", "title": "A second journal result", "year": 2024}],
        "outputs_context": {
            "clusters": [
                {"thread": "shared-ledger throughput", "outputs": ["C1", "J1"],
                 "primacy": {"claim": "first sub-second finality in <tightly-scoped area>",
                             "attributor": "J1"}},
                {"thread": "privacy-preserving audit", "outputs": ["J2"],
                 "primacy": {"claim": None, "attributor": None}}],
            "career_best": {"label_scheme": {"best": "[*]"}, "ids": ["C1", "J2"]}}}
    goc = outctx(na_scheme, good_oc, "submission")
    assert any(e[1] == "PASS" for e in goc) and not any(e[1] == "FAIL" for e in goc), \
        "fully clustered + sourced-primacy outputs_context must PASS"

    bad_oc = {"outputs_context": {
        "clusters": [{"thread": "shared-ledger throughput", "outputs": ["C1"],
                      "primacy": {"claim": None, "attributor": None}}],
        "career_best": {"ids": ["C1", "J9"]}}}   # J9 in no cluster
    assert any(e[1] == "FAIL" and e[0].startswith("outputs-context-completeness[")
               for e in outctx(na_scheme, bad_oc, "submission")), \
        "an unclustered career-best id must FAIL submission"
    d18 = outctx(na_scheme, bad_oc, "draft")
    assert any(e[1] == "WARN" for e in d18) and not any(e[1] == "FAIL" for e in d18), \
        "same unclustered id only WARNs in draft mode"

    unsourced_oc = {"outputs_context": {
        "clusters": [{"thread": "shared-ledger throughput", "outputs": ["C1"],
                      "primacy": {"claim": "milestone in <tightly-scoped area>", "attributor": None}}],
        "career_best": {"ids": ["C1"]}}}
    assert any(e[1] == "FAIL" and e[0].startswith("outputs-context-completeness[")
               for e in outctx(na_scheme, unsourced_oc, "submission")), \
        "a cluster primacy claim with no attributor must FAIL submission"
    du = outctx(na_scheme, unsourced_oc, "draft")
    assert any(e[1] == "WARN" for e in du) and not any(e[1] == "FAIL" for e in du), \
        "same unsourced primacy only WARNs in draft mode"

    assert any(e[1] == "SKIP" for e in outctx(scheme_pp, good_oc, "submission")), \
        "non-narrative-award mode must SKIP outputs-context"
    assert any(e[1] == "SKIP" for e in outctx(na_scheme, {}, "submission")), \
        "absent outputs_context block must SKIP"

    # Tier 2 (one-to-one resolution) — every clusters[].outputs / career_best.ids entry resolves
    #     to exactly one publications[].id: a resolving id PASSes; a career-best id backed by no
    #     publication is a dangling reference (FAIL submission / WARN draft). Fictional ids only.
    resolved_oc = {
        "publications": [{"id": "C1", "title": "A backed result", "year": 2024}],
        "outputs_context": {
            "clusters": [{"thread": "shared-ledger throughput", "outputs": ["C1"],
                          "primacy": {"claim": None, "attributor": None}}],
            "career_best": {"ids": ["C1"]}}}
    roc = outctx(na_scheme, resolved_oc, "submission")
    assert any(e[1] == "PASS" for e in roc) and not any(e[1] == "FAIL" for e in roc), \
        "a career-best id resolving to exactly one publication must PASS"

    dangling_oc = {
        "publications": [{"id": "C1", "title": "A backed result", "year": 2024}],
        "outputs_context": {
            "clusters": [{"thread": "shared-ledger throughput", "outputs": ["C1", "J7"],
                          "primacy": {"claim": None, "attributor": None}}],
            "career_best": {"ids": ["C1", "J7"]}}}   # J7 resolves to no publications[].id
    assert any(e[1] == "FAIL" and e[0].startswith("outputs-context-completeness[")
               for e in outctx(na_scheme, dangling_oc, "submission")), \
        "a career-best id resolving to no publication must FAIL submission"
    dr = outctx(na_scheme, dangling_oc, "draft")
    assert any(e[1] == "WARN" for e in dr) and not any(e[1] == "FAIL" for e in dr), \
        "same dangling id only WARNs in draft mode"

    # 19. traceability-spine — call directly (mirrors #11): a fully-resolved spine PASSes; a
    #     dangling task.objective, an unstaffed task (no person), and a budget row referenced by
    #     no task each FAIL submission / WARN draft; a plan with no spine block SKIPs. Fictional
    #     ids only (aim-1/obj-1/task-1/inv-lead/bl-1).
    def spine(sch, pl, ent, bd, m):
        r = Report(); check_traceability_spine(r, sch, pl, ent, bd, m); return r.entries

    sp, sp_ent, sp_bud = _spine_plan(), _spine_entity(), _spine_budget()
    ok_spine = spine(scheme_pp, sp, sp_ent, sp_bud, "submission")
    assert any(e[1] == "PASS" for e in ok_spine) and not any(e[1] == "FAIL" for e in ok_spine), \
        "fully-resolved spine must PASS submission"
    assert any(e[1] == "SKIP" for e in spine(scheme_pp, _plan(), sp_ent, sp_bud, "submission")), \
        "a plan with no spine block must SKIP traceability-spine"

    dang = _spine_plan(); dang["tasks"][1]["objective"] = "obj-404"
    assert any(e[1] == "FAIL" and e[0] == "traceability-spine"
               for e in spine(scheme_pp, dang, sp_ent, sp_bud, "submission")), \
        "a dangling task.objective must FAIL submission"
    d19a = spine(scheme_pp, dang, sp_ent, sp_bud, "draft")
    assert any(e[1] == "WARN" for e in d19a) and not any(e[1] == "FAIL" for e in d19a), \
        "same dangling task.objective only WARNs in draft mode"

    unstaffed = _spine_plan(); unstaffed["tasks"][1]["person"] = []
    assert any(e[1] == "FAIL" and e[0] == "traceability-spine"
               for e in spine(scheme_pp, unstaffed, sp_ent, sp_bud, "submission")), \
        "an unstaffed task (no person) must FAIL submission"
    d19b = spine(scheme_pp, unstaffed, sp_ent, sp_bud, "draft")
    assert any(e[1] == "WARN" for e in d19b) and not any(e[1] == "FAIL" for e in d19b), \
        "same unstaffed task only WARNs in draft mode"

    orphan_bud = _spine_budget()
    orphan_bud["rows"].append({"id": "bl-orphan", "category": "travel",
                               "funding_source": "requested", "kind": "cash", "years": {2027: 20000}})
    assert any(e[1] == "FAIL" and e[0] == "traceability-spine"
               for e in spine(scheme_pp, sp, sp_ent, orphan_bud, "submission")), \
        "a budget row referenced by no task must FAIL submission"
    d19c = spine(scheme_pp, sp, sp_ent, orphan_bud, "draft")
    assert any(e[1] == "WARN" for e in d19c) and not any(e[1] == "FAIL" for e in d19c), \
        "same orphan budget row only WARNs in draft mode"

    # 20. requirement-coverage — call directly (mirrors #19): join scheme.requirements[] against
    #     plan objectives/tasks/outputs carrying `addresses: [req-ids]`. A met mandatory PASSes;
    #     an unmet mandatory FAILs submission / only WARNs draft; an applies_if that does not hold
    #     is not applicable; an unknown classification is fail-closed (applies); an unmet desirable
    #     never FAILs; an at_least_one group met by ONE alternative PASSes; no requirements[] →
    #     SKIP. Fictional ids only (req-a1/req-a1b, workstream A/B, obj-1/task-1).
    def reqcov(sch, pl, vals, m):
        r = Report(); check_requirement_coverage(r, sch, pl, vals, m); return r.entries

    req_scheme = {"requirements": [
        {"id": "req-a1", "text": "the mandatory obligation", "strength": "mandatory",
         "applies_if": "classification.workstream == A"}]}
    met_plan = {"tasks": [{"id": "task-1", "addresses": ["req-a1"]}]}
    empty_plan = {"tasks": [{"id": "task-1"}]}
    clsA = {"classification": {"workstream": "A"}}
    clsB = {"classification": {"workstream": "B"}}

    ok = reqcov(req_scheme, met_plan, clsA, "submission")
    assert any(e[1] == "PASS" and e[0].startswith("requirement-coverage[") for e in ok) \
        and not any(e[1] == "FAIL" for e in ok), "addressed mandatory requirement must PASS"

    um = reqcov(req_scheme, empty_plan, clsA, "submission")
    assert any(e[1] == "FAIL" and e[0].startswith("requirement-coverage[") for e in um), \
        "unaddressed mandatory requirement must FAIL submission"
    umd = reqcov(req_scheme, empty_plan, clsA, "draft")
    assert any(e[1] == "WARN" for e in umd) and not any(e[1] == "FAIL" for e in umd), \
        "same unaddressed mandatory only WARNs in draft mode"

    assert not any(e[1] == "FAIL" for e in reqcov(req_scheme, empty_plan, clsB, "submission")), \
        "a requirement whose applies_if does not hold must not FAIL (not applicable)"
    assert any(e[1] == "FAIL" for e in reqcov(req_scheme, empty_plan, {}, "submission")), \
        "unknown classification is fail-closed (applies) → unmet mandatory FAILs submission"

    des_scheme = {"requirements": [{"id": "req-d1", "text": "a nice-to-have", "strength": "desirable"}]}
    assert not any(e[1] == "FAIL" for e in reqcov(des_scheme, {"tasks": []}, {}, "submission")), \
        "an unmet desirable requirement must not FAIL (informational only)"

    grp_scheme = {"requirements": [
        {"id": "req-a1", "text": "the group obligation", "strength": "mandatory",
         "quantifier": "at_least_one", "alternatives": ["req-a1", "req-a1b"]}]}
    grp_plan = {"objectives": [{"id": "obj-1", "addresses": ["req-a1b"]}]}
    gm = reqcov(grp_scheme, grp_plan, {}, "submission")
    assert any(e[1] == "PASS" for e in gm) and not any(e[1] == "FAIL" for e in gm), \
        "an at_least_one group met by one alternative must PASS"

    assert any(e[1] == "SKIP" for e in reqcov({}, met_plan, clsA, "submission")), \
        "a scheme with no requirements[] must SKIP requirement-coverage"

    # GAP-3 light — an unsigned needs_domain_review tag WARNs; a signed-off one does not; none → SKIP
    def domrev(sch, pl):
        r = Report(); check_domain_review(r, sch, pl); return r.entries

    dr_scheme = {"rubric": [{"criterion": "C1", "needs_domain_review": "security-proofs"}]}
    assert any(e[1] == "WARN" and e[0] == "domain-review" for e in domrev(dr_scheme, {})), \
        "an unsigned needs_domain_review tag must surface a WARN"
    signed = {"rubric": [{"criterion": "C1", "needs_domain_review": "security-proofs",
                          "domain_signoff": "Specialist reviewer, 2026"}]}
    assert not any(e[1] == "WARN" and e[0] == "domain-review" for e in domrev(signed, {})), \
        "a signed-off needs_domain_review tag must not WARN"
    assert any(e[1] == "SKIP" and e[0] == "domain-review" for e in domrev({}, {})), \
        "no needs_domain_review tags → SKIP domain-review"

    # requirement-coverage end-to-end: a scheme with a mandatory requirement + a plan whose nodes
    # do NOT address it fails submission-mode orchestrate but only WARNs (exit 0) in draft.
    req_scheme_p = _write(tmp, "req_scheme.yaml", __import__("yaml").safe_dump({
        "scheme": "Req-Test", "mode": "single-applicant",
        "requirements": [{"id": "req-a1", "text": "the mandatory obligation",
                          "strength": "mandatory"}]}))
    req_plan_unmet = _write(tmp, "req_plan_unmet.yaml", __import__("yaml").safe_dump(
        {"tasks": [{"id": "task-1"}]}))
    req_plan_met = _write(tmp, "req_plan_met.yaml", __import__("yaml").safe_dump(
        {"tasks": [{"id": "task-1", "addresses": ["req-a1"]}]}))
    assert orchestrate(req_scheme_p, mode="submission", plan_path=req_plan_unmet) == 1, \
        "unaddressed mandatory requirement must fail submission-mode orchestrate"
    assert orchestrate(req_scheme_p, mode="draft", plan_path=req_plan_unmet) == 0, \
        "same unaddressed requirement passes (WARN only) in draft-mode orchestrate"
    assert orchestrate(req_scheme_p, mode="submission", plan_path=req_plan_met) == 0, \
        "an addressed mandatory requirement passes submission-mode orchestrate"

    # project-substance end-to-end: complete plan passes submission; a broken plan (triggerless
    # high-impact risk) fails submission-mode orchestrate but only WARNs (exit 0) in draft.
    plan_ok_p = _write(tmp, "plan.yaml", __import__("yaml").safe_dump(_plan()))
    broken = _plan(); broken["risks"][0]["trigger"] = None
    plan_bad_p = _write(tmp, "plan_bad.yaml", __import__("yaml").safe_dump(broken))
    assert orchestrate(scheme_p, values_full, ev_p, ent_p, budget, paste_ok,
                       mode="submission", plan_path=plan_ok_p) == 0, \
        "complete plan must pass submission-mode orchestrate"
    assert orchestrate(scheme_p, values_full, ev_p, ent_p, budget, paste_ok,
                       mode="submission", plan_path=plan_bad_p) == 1, \
        "triggerless high-impact risk must fail submission-mode orchestrate"
    assert orchestrate(scheme_p, values_full, ev_p, ent_p, budget, paste_ok,
                       mode="draft", plan_path=plan_bad_p) == 0, \
        "same plan passes (WARN only) in draft-mode orchestrate"

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
