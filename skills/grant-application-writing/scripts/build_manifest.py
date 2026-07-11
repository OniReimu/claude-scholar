# /// script
# requires-python = ">=3.9"
# dependencies = ["pyyaml"]
# ///
"""Run-audit manifest builder (Stage F: pre-submission audit record) — FAIL-CLOSED.

Assembles a reproducible AUDIT MANIFEST of ONE application run: the inputs it ran on
(with content hashes), the artifacts it built, the full `validate_ir.py` verdict,
per-criterion readiness, open blockers, and a single fail-closed `ready_to_submit`
boolean. It COMPOSES `validate_ir.py` as a subprocess — it never re-implements the 16
checks and never authors or judges content. Purely mechanical: it hashes files, parses
`validate_ir`'s report lines, and derives one gate boolean.

  1. Hash inputs      each supplied --scheme/--values/--evidence/--entity/--budget/--plan/
                      --paste-ready file → {role, path, sha256, bytes}. A supplied-but-
                      missing file is a fail-closed error (exit 2), never a silent skip.
  2. Compose          run validate_ir.py with the SAME sidecars + --mode, capture stdout +
                      exit code, parse every `[STATUS] check[id] (binding): reason` line.
                      validation.hard_fail = (validate_ir exit != 0).
  3. Readiness        from the parsed `criterion-readiness[...]` lines, extract per-criterion
                      {criterion, state} (unsupported|partial|substantiated|submission-ready).
  4. Artifacts        from --artifacts (comma list) record {kind (by ext), path, exists,
                      sha256/bytes if present}. A planned-but-absent artifact is recorded
                      with exists: false — not a hard fail.
  5. Blockers         from optional --blockers YAML (list of {id, severity, status}) record
                      them + count OPEN HARD blockers. Absent file → blockers: [].
  6. ready_to_submit  DERIVED, fail-closed. true ONLY IF the readiness gate ran in submission
                      mode AND validate_ir exit == 0 AND zero open hard blockers AND every
                      criterion is substantiated/submission-ready. Any missing/unparseable
                      piece → false with a non-empty `ready_blockers`. NEVER defaults to true.

Manifest keys (fixed order): run, inputs, artifacts, validation, readiness, blockers,
ready_to_submit, ready_blockers.

Exit: 2 on an operational failure (missing input / validate_ir un-runnable); 1 when the
manifest is generated but the run is NOT ready to submit (drops into a review gate); 0 when
ready_to_submit is true.

    uv run build_manifest.py --scheme scheme.yaml --plan project-plan.yaml --mode submission
    uv run build_manifest.py --scheme scheme.yaml --evidence ev.yaml --values v.yaml \
        --artifacts application.docx,budget.svg --blockers blockers.yaml -o build-manifest.yaml
    uv run build_manifest.py --self-test
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
VALIDATE_IR = os.path.join(HERE, "validate_ir.py")
TOOL = "grant-application-writing build_manifest"

# validate_ir.py's Report.render() emits one line per check:
#   [STATUS] check[id] (binding): reason   — binding present only for FAIL/WARN.
LINE_RE = re.compile(
    r"^\[(?P<status>PASS|FAIL|WARN|SKIP)\]\s+"
    r"(?P<check>.+?)"
    r"(?:\s+\((?P<binding>hard|soft)\))?"
    r"\s*:\s*(?P<reason>.*)$"
)
READINESS_STATES = ("submission-ready", "substantiated", "partial", "unsupported")
READY_STATES = {"substantiated", "submission-ready"}

MANIFEST_KEYS = ["run", "inputs", "artifacts", "validation", "readiness", "blockers",
                 "ready_to_submit", "ready_blockers"]


class AuditError(Exception):
    """Operational failure (missing input / validate_ir un-runnable) — fail-closed, exit 2."""


# ── helpers ──────────────────────────────────────────────────────────────────
def load_yaml(path):
    import yaml
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_inputs(pairs):
    """pairs = [(role, path)]. A supplied-but-missing path is a fail-closed AuditError."""
    inputs = []
    for role, path in pairs:
        if not path:
            continue
        if not os.path.isfile(path):
            raise AuditError(f"{role} input not found: {path}")
        inputs.append({"role": role, "path": path,
                       "sha256": sha256_file(path), "bytes": os.path.getsize(path)})
    return inputs


def parse_validation(stdout):
    """Parse every `[STATUS] check[id] (binding): reason` line into a dict; ignore the rest."""
    entries = []
    for line in stdout.splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        check, cid = m.group("check").strip(), None
        if check.endswith("]") and "[" in check:
            i = check.index("[")
            check, cid = check[:i], check[i + 1:-1]
        entries.append({"check": check, "id": cid, "status": m.group("status"),
                        "binding": m.group("binding"), "reason": m.group("reason").strip()})
    return entries


def extract_readiness(entries):
    """Per-criterion {criterion, state} from the `criterion-readiness[<crit>]` lines."""
    out = []
    for e in entries:
        if e["check"] == "criterion-readiness" and e["id"] is not None:
            first = e["reason"].split()[0] if e["reason"] else ""
            out.append({"criterion": e["id"],
                        "state": first if first in READINESS_STATES else "unknown"})
    return out


def compose_validate_ir(args):
    """Run validate_ir.py with the passed-through sidecars + mode; parse its verdict."""
    cmd = [sys.executable, VALIDATE_IR, "--scheme", args.scheme, "--mode", args.mode]
    for flag, val in (("--values", args.values), ("--evidence", args.evidence),
                      ("--entity", args.entity), ("--budget", args.budget),
                      ("--plan", args.plan), ("--paste-ready", args.paste_ready)):
        if val:
            cmd += [flag, val]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as exc:
        raise AuditError(f"cannot run validate_ir.py: {exc}")
    entries = parse_validation(p.stdout)
    counts = {k: sum(1 for e in entries if e["status"] == k)
              for k in ("PASS", "FAIL", "WARN", "SKIP")}
    validation = {
        "mode": args.mode,
        "checks_seen": len(entries),
        "counts": counts,
        "hard_fail": p.returncode != 0,
        "exit_code": p.returncode,
        "findings": [e for e in entries if e["status"] != "PASS"],
    }
    return validation, entries


def build_artifacts(spec):
    arts = []
    for raw in (spec.split(",") if spec else []):
        path = raw.strip()
        if not path:
            continue
        ext = os.path.splitext(path)[1].lstrip(".").lower() or "unknown"
        exists = os.path.isfile(path)
        art = {"kind": ext, "path": path, "exists": exists}
        if exists:
            art["sha256"] = sha256_file(path)
            art["bytes"] = os.path.getsize(path)
        arts.append(art)
    return arts


def build_blockers(path):
    """Return (blockers[], open_hard_count, note|None). Absent file → ([], 0, note)."""
    if not path:
        return [], 0, "no blockers file supplied"
    if not os.path.isfile(path):
        raise AuditError(f"blockers file not found: {path}")
    data = load_yaml(path)
    raw = data.get("blockers", []) if isinstance(data, dict) else data
    blockers, open_hard = [], 0
    for b in (raw or []):
        if not isinstance(b, dict):
            continue
        blockers.append(b)
        # fail-closed: a blocker with no status is treated as open
        if str(b.get("severity", "")).lower() == "hard" and \
                str(b.get("status", "open")).lower() == "open":
            open_hard += 1
    return blockers, open_hard, None


def derive_ready(mode, validation, readiness, open_hard, scheme_has_rubric):
    """Fail-closed `ready_to_submit`: true only if every required piece holds; else false."""
    rb = []
    if mode != "submission":
        rb.append(f"readiness gate ran in {mode!r} mode — ready_to_submit requires submission mode")
    if validation["checks_seen"] == 0:
        rb.append("validate_ir produced no parseable check lines — unparseable verdict (fail-closed)")
    if validation["hard_fail"]:
        rb.append(f"validate_ir reported a hard FAIL (exit {validation['exit_code']})")
    if open_hard:
        rb.append(f"{open_hard} open hard blocker(s)")
    if scheme_has_rubric and not readiness:
        rb.append("scheme declares a rubric but no criterion-readiness lines were emitted — "
                  "criterion coverage unconfirmed (fail-closed)")
    for r in readiness:
        if r["state"] not in READY_STATES:
            rb.append(f"criterion {r['criterion']!r} is {r['state']} "
                      "(not substantiated/submission-ready)")
    return (not rb), rb


# ── assembly ─────────────────────────────────────────────────────────────────
def assemble(args):
    roles = [("scheme", args.scheme), ("values", args.values), ("evidence", args.evidence),
             ("entity", args.entity), ("budget", args.budget), ("plan", args.plan),
             ("paste-ready", args.paste_ready)]
    inputs = hash_inputs(roles)
    scheme = load_yaml(args.scheme)

    run = {
        "scheme": scheme.get("scheme", os.path.basename(args.scheme)),
        "scheme_version": scheme.get("scheme_version") or scheme.get("version"),
        "mode": scheme.get("mode"),
        "process": scheme.get("process"),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": TOOL,
    }
    validation, entries = compose_validate_ir(args)
    readiness = extract_readiness(entries)
    artifacts = build_artifacts(args.artifacts)
    blockers, open_hard, blockers_note = build_blockers(args.blockers)
    ready, ready_blockers = derive_ready(
        args.mode, validation, readiness, open_hard, bool(scheme.get("rubric")))

    manifest = {
        "run": run,
        "inputs": inputs,
        "artifacts": artifacts,
        "validation": validation,
        "readiness": readiness,
        "blockers": blockers,
        "ready_to_submit": ready,
        "ready_blockers": ready_blockers,
    }
    return manifest, blockers_note


def summarize(manifest, blockers_note):
    v, c = manifest["validation"], manifest["validation"]["counts"]
    lines = [
        f"scheme: {manifest['run']['scheme']} [{manifest['run']['mode']}] · "
        f"readiness gate: {v['mode']}",
        f"validation: {c['PASS']} PASS / {c['FAIL']} FAIL / {c['WARN']} WARN / "
        f"{c['SKIP']} SKIP · hard_fail={v['hard_fail']}",
        "readiness: " + (", ".join(f"{r['criterion']}={r['state']}"
                                    for r in manifest["readiness"]) or "(none)"),
    ]
    if blockers_note:
        lines.append(f"blockers: {blockers_note}")
    lines.append(f"ready_to_submit: {manifest['ready_to_submit']}")
    for b in manifest["ready_blockers"]:
        lines.append(f"  ready_blocker · {b}")
    return "\n".join(lines)


# ── self-test ────────────────────────────────────────────────────────────────
def _ns(**kw):
    base = dict(scheme=None, values=None, evidence=None, entity=None, budget=None,
                plan=None, paste_ready=None, blockers=None, artifacts=None,
                mode="submission", output=None)
    base.update(kw)
    return argparse.Namespace(**base)


def self_test():
    import copy
    import shutil
    import tempfile
    import yaml

    def W(tmp, name, data):
        p = os.path.join(tmp, name)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(data if isinstance(data, str) else yaml.safe_dump(data))
        return p

    # Minimal prospective-project scheme with a rubric criterion (fictional scheme only).
    scheme = {
        "scheme": "ACME-Prospective-Fund",
        "scheme_version": "2026.1",
        "mode": "prospective-project",
        "process": ["single-stage-review"],
        "submission": {"phases": ["full", "post-award"]},
        "sections": [{"id": "S", "fields": [
            {"id": "q1", "widget": "narrative", "role": "criterion-scored",
             "criterion_ref": "Quality"}]}],
        "rubric": [{"criterion": "Quality", "weight": 0.5,
                    "minimum_evidence": ["publication"],
                    "readiness_rule": ">=1 backed claim; no [TO SET] markers"}],
    }
    evidence = {"publications": [{"id": "pub1", "title": "A backed result", "year": 2025}]}
    values = {"q1": "Prior work establishes the baseline this project extends."}
    good_plan = {
        "aims": [{"id": "aim-1", "statement": "Bound the shared-ledger throughput",
                  "success_criterion": "sustained >=10k tx/s on the reference cluster"}],
        "design": [{"aim": "aim-1",
                    "answers_aim": "the controlled load test isolates throughput from cache"}],
        "benefits": [{"id": "ben-1", "benefit": "lower settlement latency",
                      "owner": "ACME Cooperative", "metric": "median down 40%", "timing": "by Y3"}],
        "additionality": {"counterfactual": "without this grant ACME will not build the ledger"},
        "risks": [{"id": "risk-1", "risk": "data-sharing MOU delayed", "impact": "high",
                   "trigger": "if the MOU is unsigned by month 6",
                   "contingency": "switch to the synthetic cohort", "owner": "CI Rivera"}],
    }

    tmp = tempfile.mkdtemp()
    try:
        s_p = W(tmp, "scheme.yaml", scheme)
        ev_p = W(tmp, "evidence.yaml", evidence)
        v_p = W(tmp, "values.yaml", values)
        good_p = W(tmp, "plan.yaml", good_plan)

        # 1. CLEAN submission run → all 8 keys in order, ready_to_submit a bool (True here).
        m, _ = assemble(_ns(scheme=s_p, evidence=ev_p, values=v_p, plan=good_p, mode="submission"))
        assert list(m.keys()) == MANIFEST_KEYS, list(m.keys())
        assert isinstance(m["ready_to_submit"], bool), m["ready_to_submit"]
        assert m["validation"]["hard_fail"] is False, m["validation"]
        assert m["blockers"] == [], m["blockers"]
        assert any(r["state"] in READY_STATES for r in m["readiness"]), m["readiness"]
        assert m["ready_to_submit"] is True, m["ready_blockers"]

        # 2. injected hard FAIL: high-impact risk missing its trigger, submission mode.
        bad = copy.deepcopy(good_plan)
        bad["risks"][0]["trigger"] = None
        bad_p = W(tmp, "plan_bad.yaml", bad)
        mf, _ = assemble(_ns(scheme=s_p, evidence=ev_p, values=v_p, plan=bad_p, mode="submission"))
        assert mf["validation"]["hard_fail"] is True, mf["validation"]
        assert mf["ready_to_submit"] is False, mf["ready_to_submit"]
        assert mf["ready_blockers"], "hard-fail run must carry non-empty ready_blockers"

        # 3. input-hash determinism: same file → same sha256.
        assert m["inputs"][0]["role"] == "scheme"
        assert m["inputs"][0]["sha256"] == sha256_file(s_p) == sha256_file(s_p), "sha256 must be stable"

        # a draft-mode run is never certified ready (submission-mode gate required).
        md, _ = assemble(_ns(scheme=s_p, evidence=ev_p, values=v_p, plan=good_p, mode="draft"))
        assert md["ready_to_submit"] is False and md["ready_blockers"], "draft mode is not ready"

        # parse round-trip: the report line grammar → {check, id, status, binding, reason}.
        e = parse_validation("[FAIL] risk-triggers[risk-1] (hard): missing trigger\n"
                             "[PASS] criterion-readiness[Quality]: submission-ready (weight 50%)\n"
                             "== 1 PASS · 1 FAIL ==")
        assert e[0] == {"check": "risk-triggers", "id": "risk-1", "status": "FAIL",
                        "binding": "hard", "reason": "missing trigger"}, e[0]
        assert extract_readiness(e) == [{"criterion": "Quality", "state": "submission-ready"}], e
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("self-test OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scheme", help="scheme.yaml (Stage-A IR) — required unless --self-test")
    ap.add_argument("--values", help="values.yaml (filled field content sidecar)")
    ap.add_argument("--evidence", help="evidence-store.yaml")
    ap.add_argument("--entity", help="entity-store.yaml (partners/contributions)")
    ap.add_argument("--budget", help="budget.yaml")
    ap.add_argument("--plan", help="project-plan.yaml (prospective-project mode)")
    ap.add_argument("--paste-ready", dest="paste_ready", help="PASTE-READY.txt")
    ap.add_argument("--blockers", help="blockers.yaml (list of {id, severity, status})")
    ap.add_argument("--artifacts", help="comma list of built artifact paths (a.docx,b.svg)")
    ap.add_argument("--mode", choices=("submission", "draft"), default="submission",
                    help="readiness gate passed through to validate_ir; the manifest is a "
                         "pre-submission audit, so submission (default) is the strict gate")
    ap.add_argument("-o", "--output", help="write manifest YAML here (else stdout)")
    ap.add_argument("--self-test", action="store_true", help="run built-in self-test")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.scheme:
        ap.error("--scheme is required (or --self-test)")

    import yaml
    try:
        manifest, blockers_note = assemble(args)
    except AuditError as exc:
        print(f"AUDIT ERROR: {exc}", file=sys.stderr)
        return 2
    text = yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True, default_flow_style=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    print(summarize(manifest, blockers_note), file=sys.stderr)
    return 0 if manifest["ready_to_submit"] else 1


if __name__ == "__main__":
    sys.exit(main())
