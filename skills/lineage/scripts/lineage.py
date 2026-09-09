#!/usr/bin/env python3
"""lineage.py - frontier view over the tree of ideas.

Parses docs/lineage.md into docs/lineage.json and serves a page that renders from that
JSON alone. Probes: results scan, generator static analysis, PBS output declarations,
git activity, and qstat on both clusters over ssh.

A probe that fails keeps serving its last good value, clearly marked stale, because an
empty result must never be indistinguishable from a clean one. Three states stay
distinct throughout: "no signal" (nothing to look at), "stalled" (looked, nothing
moved) and "live" (a running job, or failing that a recent commit - the page says
which).

    lineage.py collect                 write lineage.json, print a summary
    lineage.py --once                  collect, print any INCOMPLETE, exit non-zero on error
    lineage.py --watch [--interval 30] collect on a loop and serve on 127.0.0.1:8787
    lineage.py --emit DIR              index.html + lineage.json for static hosting
    lineage.py --init                  probe a new project and write a starter outline
    lineage.py --watch --notify CMD    run CMD when a leaf goes live -> stalled

Design: rfc_lineage.md. Build order: plan_lineage.md.
"""

import argparse
import datetime
import glob
import http.server
import json
import os
import re
import shlex
import socketserver
import subprocess
import sys
import threading
import time

OUTLINE_CANDIDATES = ("docs/lineage.md", "lineage.md", "LINEAGE.md", ".lineage.md")


def find_project_root(start=None):
    """The project is where the outline lives, not where this script lives — so the
    script can sit in a plugin and still be run from inside any project."""
    here = os.path.abspath(start or os.getcwd())
    d = here
    while True:
        for rel in OUTLINE_CANDIDATES:
            if os.path.exists(os.path.join(d, rel)):
                return d
        parent = os.path.dirname(d)
        if parent == d:
            return here                     # nothing found: --init will create one here
        d = parent


def find_outline(root=None):
    root = root or ROOT
    for rel in OUTLINE_CANDIDATES:
        p = os.path.join(root, rel)
        if os.path.exists(p):
            return p
    return os.path.join(root, OUTLINE_CANDIDATES[0])


ROOT = find_project_root()
OUTLINE = find_outline()
OUT_JSON = os.path.join(os.path.dirname(OUTLINE), "lineage.json")

# Everything project-specific lives in an optional block at the top of the outline.
# Omit it entirely and the tool degrades to git-only, which still renders the tree.
DEFAULTS = {
    "results": "",          # dir holding artifacts, e.g. results/
    "scripts": "",          # job scripts, e.g. run_*.pbs
    "generators": "",       # code that consumes artifacts, e.g. paper/scripts/*.py
    "prefix": r"v\d+[a-z]*",  # how a version prefix looks in this project
    "clusters": "",         # ssh-host:username, comma separated
    "training": r"saves/|--do_train|train_",
    "remote_results": "",   # host:/path/to/runtime/*/results — where a cluster writes
    "scheduler": "pbs",     # pbs | slurm | none
}
CONFIG = dict(DEFAULTS)


def parse_config(text):
    """Read the `<!-- lineage ... -->` block. Every key is optional."""
    cfg = dict(DEFAULTS)
    m = re.search(r"<!--\s*lineage\s*\n(.*?)-->", text, re.S)
    if not m:
        return cfg, False
    for line in m.group(1).split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        if k in cfg:
            cfg[k] = re.sub(r"\s+#.*$", "", v).strip()
    return cfg, True


def cfg_globs(key):
    return [g.strip() for g in CONFIG.get(key, "").split(",") if g.strip()]


def cfg_clusters():
    out = []
    for item in CONFIG.get("clusters", "").split(","):
        item = item.strip()
        if not item:
            continue
        host, _, user = item.partition(":")
        out.append({"name": host.strip(), "host": host.strip(), "user": user.strip()})
    return out

# A line whose pattern slot says `[decision]` closes by someone making a call, not by
# running anything. Keeping those apart from "no experiment attached yet" is the whole
# point: one costs GPU hours, the other costs ten minutes of attention.
DECISION_TOKENS = ("decision", "judgement", "judgment", "call")

SECTIONS = {
    "Open": "open",
    "Closed — cashed in": "cashed",
    "Closed — falsified": "falsified",
}


# --------------------------------------------------------------------------- parse

def _split_body(body):
    """Pull verdict, claim and match patterns off a node line, leaving the name."""
    status, verdict = None, None
    m = re.search(r"([✓✗])\s*(.*)$", body)
    if m:
        status = "cashed" if m.group(1) == "✓" else "falsified"
        verdict = m.group(2).strip(" —-")
        body = body[: m.start()].strip()

    claim = None
    for arrow in ("→", "->"):          # people type both; losing one silently is worse
        if arrow in body:              # than any ambiguity accepting it could cause
            body, claim = body.split(arrow, 1)
            claim = claim.strip().strip("`").strip()
            body = body.strip()
            break

    patterns, has_field = [], False
    m = re.search(r"`\[([^\]]*)\]`", body)
    if m:
        has_field = True
        raw = m.group(1).strip()
        if raw not in ("—", "-", ""):
            patterns = [p.strip() for p in raw.split(",") if p.strip()]
        body = (body[: m.start()] + body[m.end():]).strip()

    kind = "decision" if any(p.lower() in DECISION_TOKENS for p in patterns) else None
    patterns = [p for p in patterns if p.lower() not in DECISION_TOKENS]
    name = body.strip().strip("*").strip().rstrip("—- ").strip()
    return name, patterns, has_field, claim, status, verdict, kind


def parse_outline(path):
    """Return (nodes, incomplete). Narrow by design: only the forms actually in use."""
    nodes, incomplete = [], []
    rel = os.path.relpath(path, ROOT)
    section = None
    stack = []          # (depth, node_id)
    prev_depth = -1

    with open(path, encoding="utf-8") as fh:
        lines = fh.read().split("\n")

    for lineno, line in enumerate(lines, 1):
        if line.startswith("## "):
            section = line[3:].strip()
            stack, prev_depth = [], -1
            continue

        m = re.match(r"^( *)- (.+)$", line)
        if m and section in SECTIONS:
            indent = len(m.group(1))
            if indent % 2:
                incomplete.append({"kind": "outline", "where": "%s:%d" % (rel, lineno),
                                   "why": "indent %d is not a multiple of 2" % indent})
            depth = indent // 2
            if depth > prev_depth + 1:
                incomplete.append({"kind": "outline", "where": "%s:%d" % (rel, lineno),
                                   "why": "indent jumps from depth %d to %d"
                                          % (prev_depth, depth)})

            name, patterns, has_field, claim, status, verdict, kind = _split_body(m.group(2))
            lifecycle = SECTIONS[section]

            if lifecycle == "open" and status and depth == 0:
                incomplete.append({"kind": "outline", "where": "%s:%d" % (rel, lineno),
                                   "why": "'%s' is a top-level line carrying a verdict — move "
                                          "it under Closed" % name[:40]})
            if lifecycle != "open" and depth == 0 and not status:
                incomplete.append({"kind": "outline", "where": "%s:%d" % (rel, lineno),
                                   "why": "'%s' is under %s but has no ✓/✗ verdict"
                                          % (name[:40], section)})

            while stack and stack[-1][0] >= depth:
                stack.pop()
            node = {
                "id": "n%d" % lineno,
                "name": name,
                "depth": depth,
                "parent": stack[-1][1] if stack else None,
                "section": section,
                "lifecycle": status or lifecycle,
                "verdict": verdict,
                "patterns": patterns,
                "declared_pattern": has_field,
                "claim": claim,
                "kind": kind,
                "intent": None,
                "line": lineno,
                "signal": "none",       # step 2 fills this in
                "last_activity": None,
                "jobs": 0,
                "artifacts": None,
            }
            stack.append((depth, node["id"]))
            prev_depth = depth
            nodes.append(node)
            continue

        m = re.match(r"^ *> (.+)$", line)
        if m and nodes and section in SECTIONS:
            add = m.group(1).strip()
            b = re.match(r"!blocks:?\s*(.*)", add)
            if b:
                nodes[-1]["blocks"] = b.group(1).strip() or "blocks submission"
                continue
            nodes[-1]["intent"] = (nodes[-1]["intent"] + " " + add) if nodes[-1]["intent"] else add

    for n in nodes:
        if n["lifecycle"] != "open":
            n.pop("blocks", None)      # a closed line no longer blocks anything

    parents = {n["parent"] for n in nodes if n["parent"]}
    for n in nodes:
        n["is_leaf"] = n["id"] not in parents

    return nodes, incomplete


# ----------------------------------------------------------------------- probes
#
# Deliberately narrow: each probe parses only the forms actually in use today and
# reports INCOMPLETE on anything else. An unparsed form must never read as an empty
# result - "nothing found" and "could not look" are different answers.

def literal_prefix_re():
    """Built from the configured prefix shape, not hardcoded to this project's."""
    return re.compile(r"""["'](%s)(?:_[A-Za-z0-9_*.{}-]*)?["']"""
                      % (CONFIG.get("prefix") or DEFAULTS["prefix"]))
VARIABLE_PREFIX = re.compile(r"""prefix\s*=\s*(?!["'])[A-Za-z_]""")
DECLARED_OUTPUT = (
    r'^\s*PREFIX="([^"$]+)"',
    r'^\s*OUTPUT_ROOT="\$\{OUTPUT_ROOT:-[^}]*?/([A-Za-z0-9_]+)\}"',
    r'^#\s+(v\d+[A-Za-z0-9_]*)\s*->\s*results/',
)

FRESH_HOURS = 48.0


def _read(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _prefix_of(name):
    m = re.match("(%s)" % CONFIG.get("prefix") or DEFAULTS["prefix"], name)
    return m.group(1) if m else None


def _pfx_match(candidate, base):
    """v77 matches v77app but v2 must not match v22."""
    if candidate == base:
        return True
    if not candidate.startswith(base):
        return False
    return not candidate[len(base)].isdigit()


def expand_pattern(pat):
    """A match pattern -> (prefixes, repo paths). Three forms are recognised:
    a path or filename, a vNN-vMM range, and a bare prefix glob."""
    if "/" in pat or pat.endswith((".pbs", ".py", ".md", ".tex", ".json")):
        base = _prefix_of(re.sub(r"^run_", "", os.path.basename(pat)))
        return ({base} if base else set()), [pat.replace("**", "").rstrip("/") or "."]
    m = re.fullmatch(r"v(\d+)[a-z]*-v(\d+)[a-z]*", pat)
    if m:
        return {"v%d" % n for n in range(int(m.group(1)), int(m.group(2)) + 1)}, []
    base = _prefix_of(pat)
    if base:
        return {base}, []
    if "*" in pat:                      # a bare file glob, e.g. run_c4tier2_*
        return set(), [pat]
    return set(), []


def scan_results():
    rel = CONFIG.get("results", "").strip()
    if not rel:
        return None, "no results directory configured"
    d = os.path.join(ROOT, rel)
    if not os.path.isdir(d):
        return None, "%s not found" % rel
    counts, mtimes, total = {}, {}, 0
    for name in os.listdir(d):
        p = _prefix_of(name)
        if not p:
            continue
        counts[p] = counts.get(p, 0) + 1
        total += 1
        try:
            ts = os.path.getmtime(os.path.join(d, name))
        except OSError:
            continue
        if ts > mtimes.get(p, 0):
            mtimes[p] = ts
    return (counts, mtimes), "%d prefixes over %d matching files" % (len(counts), total)


def scan_generators():
    named, incomplete, seen, lit = {}, [], 0, literal_prefix_re()
    if not cfg_globs("generators"):
        return {}, [], "no generator globs configured - consumption not checked"
    for pattern in cfg_globs("generators"):
        for path in sorted(glob.glob(os.path.join(ROOT, pattern))):
            seen += 1
            rel = os.path.relpath(path, ROOT)
            txt = _read(path)
            for m in lit.finditer(txt):
                named.setdefault(m.group(1), set()).add(rel)
            for m in re.finditer(r"#\s*lineage:\s*consumes\s+([A-Za-z0-9_*]+)", txt):
                named.setdefault(m.group(1), set()).add(rel + " (declared)")
            m = VARIABLE_PREFIX.search(txt)
            if m:
                incomplete.append({"kind": "probe", "where": "%s:%d"
                                   % (rel, txt[: m.start()].count("\n") + 1),
                                   "why": "the prefix arrives in a variable"})
    return ({k: sorted(v) for k, v in named.items()}, incomplete,
            "%d scripts, %d prefixes named literally, %d unresolvable"
            % (seen, len(named), len(incomplete)))


def scan_scripts():
    scripted, declared, training = set(), {}, set()
    globs = cfg_globs("scripts")
    if not globs:
        return set(), {}, set(), "no job-script glob configured"
    hint = re.compile(CONFIG.get("training") or DEFAULTS["training"])
    for path in sorted(sum((glob.glob(os.path.join(ROOT, g)) for g in globs), [])):
        base = os.path.basename(path)
        if "smoke" in base:
            continue
        m = re.search("(%s)" % (CONFIG.get("prefix") or DEFAULTS["prefix"]), base)
        sp = m.group(1) if m else None
        if not sp:
            continue
        scripted.add(sp)
        txt = _read(path)
        for rx in DECLARED_OUTPUT:
            m = re.search(rx, txt, re.M)
            if m:
                declared.setdefault(sp, set()).add(m.group(1))
        if hint.search(txt):
            training.add(sp)
    return (scripted, {k: sorted(v) for k, v in declared.items()}, training,
            "%d non-smoke prefixes, %d declare an output target" % (len(scripted), len(declared)))


def git_last(paths, timeout=8):
    if not paths:
        return None
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%cI", "--"] + list(paths),
                             cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    except Exception:                                              # noqa: BLE001
        return None
    return out.stdout.strip() or None


def outline_written(name, timeout=8):
    """When this line last appeared or changed in the outline. Derived from git, so it
    stays honest without anyone maintaining a date by hand."""
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%cI", "-S", name, "--",
             os.path.relpath(OUTLINE, ROOT)],
            cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    except Exception:                                              # noqa: BLE001
        return None
    return out.stdout.strip() or None


def _hours_since(iso):
    if not iso:
        return None
    try:
        then = datetime.datetime.fromisoformat(iso)
    except ValueError:
        return None
    now = datetime.datetime.now(then.tzinfo)
    return (now - then).total_seconds() / 3600.0


# Cache of the last value each probe returned successfully. A probe that fails keeps
# serving its snapshot, clearly marked stale - an empty result must never be
# indistinguishable from a clean one.
_LAST_GOOD = {}


def probe(name, fn):
    """Run a probe; on failure fall back to its last good value and say so."""
    err = None
    try:
        val, detail = fn()
        if val is not None:
            _LAST_GOOD[name] = (val, time.time())
            return val, {"name": name, "ok": True, "stale": False, "detail": detail}
        err = detail
    except Exception as exc:                                       # noqa: BLE001
        err = repr(exc)
    if name in _LAST_GOOD:
        val, ts = _LAST_GOOD[name]
        mins = int((time.time() - ts) / 60)
        return val, {"name": name, "ok": False, "stale": True, "stale_minutes": mins,
                     "detail": "%s — still showing the snapshot taken %dm ago" % (err, mins)}
    return None, {"name": name, "ok": False, "stale": False, "detail": err}


SHORT_RUN_MINUTES = 15


def _hhmm(v):
    m = re.fullmatch(r"(\d+):(\d\d)", v or "")
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def _dhms(v):
    """Slurm walltimes: D-HH:MM:SS, HH:MM:SS, MM:SS, or UNLIMITED."""
    v = (v or "").strip()
    if not v or v.upper() in ("UNLIMITED", "PARTITION_LIMIT", "INVALID"):
        return None
    days = 0
    if "-" in v:
        d, _, v = v.partition("-")
        days = int(d) if d.isdigit() else 0
    bits = [b for b in v.split(":") if b.isdigit()]
    if not bits:
        return None
    bits = [int(b) for b in bits]
    while len(bits) < 3:
        bits.insert(0, 0)
    return days * 1440 + bits[0] * 60 + bits[1]


SLURM_STATE = {"RUNNING": "R", "PENDING": "Q", "SUSPENDED": "H", "REQUEUED": "Q",
               "COMPLETED": "C", "FAILED": "X", "TIMEOUT": "X", "CANCELLED": "X",
               "OUT_OF_MEMORY": "X", "NODE_FAIL": "X", "PREEMPTED": "X"}
DONE_STATES = ("F", "C", "X")     # F: PBS finished, outcome unknown · C: ok · X: failed


def _parse_sacct(text):
    """sacct -X -n -P -o JobIDRaw,JobName,State,Elapsed,Timelimit"""
    jobs, unparsed = [], 0
    for line in text.split("\n"):
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 5:
            unparsed += 1
            continue
        jid, name, state, elapsed, limit = parts[:5]
        st = SLURM_STATE.get(state.strip().split()[0].upper())
        if st is None:
            unparsed += 1
            continue
        jobs.append({"id": jid.strip(), "name": name.strip(), "state": st,
                     "elapsed_min": _dhms(elapsed), "asked_min": _dhms(limit)})
    return jobs, unparsed


def _parse_qstat(text):
    """Tolerant on purpose: PBS column layouts differ between the two clusters.
    Reads the state letter and, when present, the requested and elapsed walltime —
    a job that finished in minutes against a long request is worth surfacing; whether it
    failed is only knowable when the scheduler says so."""
    jobs, unparsed = [], 0
    for line in text.split("\n"):
        line = line.rstrip()
        if not re.match(r"^\d+\.\S+", line):
            continue
        parts = line.split()
        idx = next((i for i in range(len(parts) - 1, -1, -1)
                    if len(parts[i]) == 1 and parts[i].isalpha() and parts[i].isupper()), None)
        if idx is None:
            unparsed += 1
            continue
        elapsed = _hhmm(parts[idx + 1]) if idx + 1 < len(parts) else None
        asked = _hhmm(parts[idx - 1]) if idx >= 1 else None
        jobs.append({"id": parts[0], "name": parts[3] if len(parts) > 3 else "",
                     "state": parts[idx], "elapsed_min": elapsed, "asked_min": asked})
    return jobs, unparsed


def scan_scheduler(cluster, timeout=20):
    # Finished jobs must be included: that is how a job that died in eight minutes stays
    # visible instead of simply vanishing from the queue.
    kind = (CONFIG.get("scheduler") or "pbs").strip().lower()
    if kind == "none":
        return [], "scheduler disabled in config"
    if kind == "slurm":
        remote = ("sacct -u %s -X -n -P -S now-7days "
                  "-o JobIDRaw,JobName,State,Elapsed,Timelimit" % cluster["user"])
    else:
        remote = "qstat -x -u %s" % cluster["user"]
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
           "-o", "StrictHostKeyChecking=accept-new", cluster["host"], remote]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return None, "ssh not available on this machine"
    except subprocess.TimeoutExpired:
        return None, "ssh timed out after %ds" % timeout
    if p.returncode != 0:
        tail = [l for l in (p.stderr or "").strip().split("\n") if l.strip()]
        return None, (tail[-1][:120] if tail else "ssh exit %d" % p.returncode)
    jobs, unparsed = (_parse_sacct if kind == "slurm" else _parse_qstat)(p.stdout)
    jobs = jobs[-80:]
    running = [j for j in jobs if j["state"] in ("R", "Q", "H")]
    detail = "%d running, %d in recent history" % (len(running), len(jobs))
    if unparsed:
        detail += ", %d rows unparsed" % unparsed
    return jobs, detail


def short_runs(all_jobs):
    """Terminal jobs that ended far sooner than the walltime asked for; `certain` marks
    the ones the scheduler explicitly reports as failed."""
    out = []
    for cluster, jobs in all_jobs.items():
        for j in (jobs or [])[-15:]:      # only what happened recently enough to act on
            if j["state"] not in ("F", "X"):   # a COMPLETED job did not die, however brief
                continue
            e, a = j.get("elapsed_min"), j.get("asked_min")
            if e is not None and e <= SHORT_RUN_MINUTES and (a or 0) > 60:
                out.append({"cluster": cluster, "id": j["id"].split(".")[0],
                            "name": j.get("name", ""), "elapsed_min": e, "asked_min": a,
                            "certain": j["state"] == "X"})
    return out


def jobs_by_prefix(all_jobs):
    out = {}
    for cluster, jobs in all_jobs.items():
        for j in jobs or []:
            if j["state"] not in ("R", "Q", "H"):
                continue
            jm = re.search("(%s)" % (CONFIG.get("prefix") or DEFAULTS["prefix"]),
                           j.get("name", ""))
            base = jm.group(1) if jm else None
            if base:
                out.setdefault(base, []).append(dict(j, cluster=cluster))
    return out


EVENT_KEEP = 60
COVERAGE_DAYS = 7
COVERAGE_IGNORE = re.compile(r"(^|/)(\.|docs/lineage\.json|docs/lineage_snapshot\.html|"
                             r"lineage\.json|__pycache__|node_modules)")


def git_recent_paths(days=COVERAGE_DAYS, timeout=15):
    try:
        out = subprocess.run(
            ["git", "log", "--since=%d days ago" % days, "--name-only", "--format="],
            cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    except Exception:                                              # noqa: BLE001
        return None
    seen = {}
    here = os.path.basename(ROOT) + "/"
    for line in out.stdout.split("\n"):
        line = line.strip()
        if not line or COVERAGE_IGNORE.search(line):
            continue
        if line.startswith(here):
            line = line[len(here):]
        elif "/" in line and not os.path.exists(os.path.join(ROOT, line)):
            continue
        top = line.split("/")[0] if "/" in line else line
        seen[top] = seen.get(top, 0) + 1
    return seen


def git_uncommitted(timeout=10):
    """Work in progress that has not been committed yet — the strongest signal that
    somebody is at a keyboard right now, and one nothing else on this page can see."""
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                             capture_output=True, text=True, timeout=timeout)
    except Exception:                                              # noqa: BLE001
        return None
    seen, here = {}, os.path.basename(ROOT) + "/"
    for line in out.stdout.split("\n"):
        path = line[3:].strip().strip('"')
        if not path or COVERAGE_IGNORE.search(path):
            continue
        if path.startswith(here):
            path = path[len(here):]
        elif "/" in path and not os.path.exists(os.path.join(ROOT, path)):
            continue
        top = path.split("/")[0]
        seen[top] = seen.get(top, 0) + 1
    return seen


def scan_remote(timeout=60):
    """Count result files sitting on the cluster. A job can finish, write two hundred
    files there, and leave this machine looking as though nothing happened."""
    spec = CONFIG.get("remote_results", "").strip()
    if not spec or ":" not in spec:
        return {}, "no remote results path configured"
    host, root = spec.split(":", 1)
    glob_expr = root if "*" in root else root.rstrip("/") + "/*/results"
    script = ('for d in %s; do [ -d "$d" ] && '
              'echo "$(basename $(dirname $d)) $(ls "$d" 2>/dev/null | wc -l)"; done' % glob_expr)
    try:
        out = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                              host.strip(), script],
                             capture_output=True, text=True, timeout=timeout)
    except Exception as exc:                                       # noqa: BLE001
        return None, repr(exc)
    if out.returncode != 0:
        tail = [l for l in (out.stderr or "").strip().split("\n") if l.strip()]
        return None, (tail[-1][:110] if tail else "ssh exit %d" % out.returncode)
    got = {}
    for line in out.stdout.split("\n"):
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            got[parts[0]] = int(parts[1])
    return got, "%d result dirs on %s" % (len(got), host.strip())


def unsynced(remote, landed, running=None):
    """Remote directories holding files that have not made it here."""
    out = []
    for name, n in sorted(remote.items(), key=lambda kv: -kv[1]):
        base = _prefix_of(name)
        local = sum(v for k, v in landed.items() if base and _pfx_match(k, base))
        if n > 0 and local == 0:
            live = bool(base and (running or {}).get(base))
            out.append({"dir": name, "prefix": base or name, "remote": n,
                        "local": local, "in_flight": live})
    return out


GROUPS = ("running", "changed", "writeup", "stalled", "planned", "decision", "waiting")
MANUSCRIPT_EXT = (".tex", ".md", ".bib")


def _manuscript_only(n):
    """Every pattern on this line points at prose, not at a run. Nothing moving there
    means nobody has written it down yet - a different thing from an experiment that
    stopped, and the word 'stalled' was carrying both."""
    pats = n.get("patterns") or []
    return bool(pats) and all(p.endswith(MANUSCRIPT_EXT) for p in pats)


def group_of(n):
    """One definition of a line's stage, computed once and used everywhere."""
    if n.get("kind") == "decision":
        return "decision"
    if n.get("jobs"):
        return "running"
    if n.get("declared_unlanded") and not n.get("artifacts"):
        return "planned"
    if n["signal"] == "live":
        return "changed"
    if n["signal"] == "stalled":
        return "writeup" if _manuscript_only(n) else "stalled"
    return "waiting"


def closed_with_activity(nodes, mtimes, days=1):
    """A line declared finished, that is still moving. Its pattern claims the new work,
    so it shows up in neither the frontier nor the unattributed list - it falls between
    them. Either the line was not finished, or work started under it and nobody wrote
    that down."""
    out, cutoff = [], time.time() - days * 86400
    for n in nodes:
        if n["lifecycle"] == "open":
            continue
        why = []
        if n.get("jobs"):
            why.append("%d job running now" % n["jobs"])
        for b in n.get("matched_prefixes", []):
            ts = mtimes.get(b)
            if ts and ts >= cutoff:
                why.append("%s wrote files %.1fd ago" % (b, (time.time() - ts) / 86400))
        if why:
            out.append({"name": n["name"], "verdict": n.get("verdict") or "",
                        "why": "; ".join(why[:3])})
    return out


def coverage_gaps(nodes, mtimes, running, git_paths):
    """Work that is moving and belongs to no registered node. CANDIDATES for a new
    line - never presented as lines the tool inferred by itself (RFC 11.5)."""
    prefixes, paths = set(), set()
    for n in nodes:
        for pat in n["patterns"]:
            pf, pa = expand_pattern(pat)
            prefixes |= pf
            for x in pa:
                paths.add(x.split("/")[0].replace("*", ""))

    def claimed(base):
        return any(_pfx_match(base, b) or _pfx_match(b, base) for b in prefixes)

    gaps = {"jobs": [], "artifacts": [], "paths": [], "days": COVERAGE_DAYS}
    for base, jobs in (running or {}).items():
        if not claimed(base):
            gaps["jobs"].append({"prefix": base,
                                 "detail": ", ".join("%s %s" % (j["cluster"],
                                                                j["id"].split(".")[0])
                                                     for j in jobs[:3])})
    cutoff = time.time() - COVERAGE_DAYS * 86400
    for base, ts in sorted(mtimes.items(), key=lambda kv: -kv[1]):
        if ts >= cutoff and not claimed(base):
            gaps["artifacts"].append({
                "prefix": base,
                "detail": "newest file %.1fd ago" % ((time.time() - ts) / 86400)})
    for top, n in sorted((git_paths or {}).items(), key=lambda kv: -kv[1]):
        if top in paths:
            continue
        # a path anywhere containing a claimed prefix belongs to that node
        if any(re.search(re.escape(b) + r"(?![0-9])", top) for b in prefixes):
            continue
        gaps["paths"].append({"prefix": top,
                              "detail": "%d file change%s" % (n, "" if n == 1 else "s")})
    gaps["paths"] = gaps["paths"][:8]
    return gaps





def _identity(n):
    """Stable across edits to the outline - line numbers shift, names do not."""
    return n["name"]


def diff_events(prev, cur, trustworthy=True):
    """The only transition worth interrupting someone for is live -> stalled: a line's
    last job ended and nothing picked it up. That is the moment drift is born."""
    if not prev:
        return []
    now = cur["collected_at"]
    old = {_identity(n): n for n in prev.get("nodes", [])}
    events = []
    for n in cur["nodes"]:
        was = old.get(_identity(n))
        if was is None:
            if n["lifecycle"] == "open" and n.get("is_leaf"):
                events.append({"at": now, "kind": "born", "name": n["name"]})
            continue
        if was["lifecycle"] == "open" and n["lifecycle"] != "open":
            events.append({"at": now, "kind": "closed", "name": n["name"],
                           "detail": n["lifecycle"]})
        elif trustworthy and was.get("signal") == "live" and n.get("signal") == "stalled":
            events.append({"at": now, "kind": "stalled", "name": n["name"],
                           "detail": "last job ended, nothing picked it up"})
    for n in prev.get("nodes", []):
        if n["lifecycle"] == "open" and n.get("is_leaf") \
                and _identity(n) not in {_identity(x) for x in cur["nodes"]}:
            events.append({"at": now, "kind": "gone", "name": n["name"]})
    return events


def notify(events, command):
    """Hand each drift birth to whatever the owner already reads. No assumption about
    the channel: the command receives one line of text as its final argument."""
    sent = 0
    for e in events:
        if e["kind"] != "stalled":
            continue
        msg = "lineage: '%s' went stalled - %s" % (e["name"], e.get("detail", ""))
        if not command:
            sys.stderr.write(msg + "\n")
            continue
        try:
            subprocess.run(command + [msg], timeout=20, check=False)
            sent += 1
        except Exception as exc:                                   # noqa: BLE001
            sys.stderr.write("notify failed: %r\n" % (exc,))
    return sent


def classify_unlanded(scripted, landed, declared, training, running=None):
    """The four buckets from plan step 2, for prefixes with a script and no artifact."""
    out = {"elsewhere": [], "training": [], "running": [], "no_trace": []}
    for p in sorted(scripted - set(landed)):
        decl = declared.get(p, [])
        away = [d for d in decl if not d.startswith(p)]
        if away:
            out["elsewhere"].append({"prefix": p, "to": away})
        elif (running or {}).get(p):
            out["running"].append({"prefix": p, "to": [
                "%s %s" % (j["cluster"], j["id"].split(".")[0])
                for j in running[p][:3]]})
        elif p in training:
            out["training"].append({"prefix": p, "to": ["saves/unlearn (not local)"]})
        else:
            out["no_trace"].append({"prefix": p, "to": []})
    return out


def enrich(nodes, landed, named, running=None, incomplete=None,
           prev_by_name=None, sched_ok=True):
    """Attach evidence and activity to every node. Never infers absence from a
    failed lookup: a node with no pattern gets signal 'none', not 'stalled'."""
    for n in nodes:
        prefixes, paths = set(), []
        for pat in n["patterns"]:
            pf, pa = expand_pattern(pat)
            if not pf and not pa and incomplete is not None:
                incomplete.append({"kind": "outline",
                                   "where": "%s:%d" % (os.path.relpath(OUTLINE, ROOT), n["line"]),
                                   "why": "match pattern %r is not a prefix, range or path" % pat})
            prefixes |= pf
            paths += pa
        matched = sorted(u for u in landed if any(_pfx_match(u, b) for b in prefixes))
        n["matched_prefixes"] = matched
        n["artifacts"] = sum(landed[u] for u in matched) if matched else 0
        n["consumed_by"] = sorted({g for u in matched for g in named.get(u, [])})
        n["declared_unlanded"] = sorted(
            b for b in prefixes if not any(_pfx_match(u, b) for u in landed))

        jobs = [j for b in prefixes for j in (running or {}).get(b, [])]
        n["jobs"] = len(jobs)
        n["job_detail"] = []
        for j in jobs[:6]:
            bit = "%s %s" % (j["cluster"], j["id"].split(".")[0])
            e, a = j.get("elapsed_min"), j.get("asked_min")
            if e is not None and a:
                bit += "  %dh%02dm of %dh" % (e // 60, e % 60, a // 60)
            n["job_detail"].append(bit)

        gpaths = list(paths) + ["run_%s*.pbs" % b for b in sorted(prefixes)] \
                             + ["%s*.py" % b for b in sorted(prefixes)]
        n["last_activity"] = git_last(gpaths) if gpaths else None
        hrs = _hours_since(n["last_activity"])
        if jobs:
            signal, because = "live", "job"
        elif not prefixes and not paths:
            signal, because = "none", None
        elif hrs is not None and hrs <= FRESH_HOURS:
            signal, because = "live", "commit"
        else:
            signal, because = "stalled", None

        # A probe that could not look must never downgrade a node. Without this a single
        # ssh timeout manufactures a drift alert - the mirror image of a false green, and
        # a tool that cries wolf dies faster than one that stays quiet.
        if not sched_ok and signal == "stalled":
            was = (prev_by_name or {}).get(n["name"])
            if was and was.get("signal") == "live":
                signal, because = "live", was.get("live_because") or "job"
                n["carried_over"] = True

        n["signal"], n["live_because"] = signal, because

        # A line with nothing to probe still has one honest clock: how long it has sat
        # in the outline untouched. That is derived from git, not declared by anyone.
        if signal in ("none", "stalled") and n["lifecycle"] == "open" and n.get("is_leaf"):
            n["written_at"] = outline_written(n["name"])
            n["written_hours"] = _hours_since(n["written_at"])
        n["idle_hours"] = round(hrs, 1) if hrs is not None else None
    return nodes


# ------------------------------------------------------------------------- collect

def load_previous():
    try:
        with open(OUT_JSON, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:                                              # noqa: BLE001
        return None


def collect(previous=None):
    started = time.time()
    previous = previous if previous is not None else load_previous()
    collectors = []
    nodes, incomplete = [], []

    if not os.path.exists(OUTLINE):
        collectors.append({"name": "outline", "ok": False,
                           "detail": "%s not found" % os.path.relpath(OUTLINE, ROOT)})
    else:
        try:
            globals()["CONFIG"], had_cfg = parse_config(_read(OUTLINE))
            nodes, incomplete = parse_outline(OUTLINE)
            collectors.append({
                "name": "outline", "ok": True,
                "detail": "%d nodes, %d incomplete%s"
                          % (len(nodes), len(incomplete),
                             "" if had_cfg else " - no config block, git-only mode")})
        except Exception as exc:                                   # noqa: BLE001
            collectors.append({"name": "outline", "ok": False, "detail": repr(exc)})

    landed_pair, detail = scan_results()
    have_results = landed_pair is not None
    collectors.append({"name": "results",
                       "ok": True if have_results else (None if "configured" in detail else False),
                       "detail": detail})
    landed, mtimes = landed_pair if have_results else ({}, {})

    try:
        named, gen_incomplete, detail = scan_generators()
        collectors.append({"name": "generators", "ok": True, "detail": detail})
        incomplete += gen_incomplete
    except Exception as exc:                                       # noqa: BLE001
        named = {}
        collectors.append({"name": "generators", "ok": False, "detail": repr(exc)})

    try:
        scripted, declared, training, detail = scan_scripts()
        collectors.append({"name": "scripts", "ok": True, "detail": detail})
    except Exception as exc:                                       # noqa: BLE001
        scripted, declared, training = set(), {}, set()
        collectors.append({"name": "scripts", "ok": False, "detail": repr(exc)})

    git_ok = git_last(["."]) is not None
    collectors.append({"name": "git", "ok": git_ok,
                       "detail": "reads commit times" if git_ok
                                 else "no git history reachable"})
    sched, any_sched = {}, False
    for c in cfg_clusters():
        jobs, status = probe("qstat:" + c["name"], lambda c=c: scan_scheduler(c))
        status["name"] = "%s %s" % (c["name"],
            "sacct" if (CONFIG.get("scheduler") or "pbs").lower() == "slurm" else "qstat")
        status["is_scheduler"] = True
        collectors.append(status)
        sched[c["name"]] = jobs or []
        any_sched = any_sched or status["ok"] or status.get("stale", False)
    running = jobs_by_prefix(sched)
    died_early = short_runs(sched)

    sched_probes = [c for c in collectors if c.get("is_scheduler")]
    sched_clean = bool(sched_probes) and all(c["ok"] for c in sched_probes)
    prev_by_name = {n["name"]: n for n in (previous or {}).get("nodes", [])}

    if nodes:
        enrich(nodes, landed, named, running, incomplete, prev_by_name,
               sched_clean or not cfg_clusters())

    landed_set, named_set = (set(landed), set(named)) if have_results else (set(), set())
    prefix_index = {}
    for b in sorted(landed_set | named_set,
                    key=lambda x: (int(re.sub(r"\D", "", x) or 0), x)):
        prefix_index[b] = {"files": landed.get(b, 0),
                           "consumed": b in named_set,
                           "landed": b in landed_set,
                           "latest": mtimes.get(b)}
    evidence = {
        "consumed": sorted(landed_set & named_set),
        "landed_not_named": sorted(landed_set - named_set),
        "named_not_landed": sorted(named_set - landed_set),
    }
    buckets = (classify_unlanded(scripted, landed, declared, training, running)
               if have_results else {"elsewhere": [], "training": [], "running": [], "no_trace": []})
    remote, rstatus = probe("remote", scan_remote)
    rstatus["name"] = "remote results"
    collectors.append(rstatus)
    remote = remote or {}
    not_synced = unsynced(remote, landed, running)

    gaps = coverage_gaps(nodes, mtimes, running, git_recent_paths())
    gaps["uncommitted"] = [{"prefix": k, "detail": "%d uncommitted change%s"
                            % (v, "" if v == 1 else "s")}
                           for k, v in sorted((git_uncommitted() or {}).items(),
                                              key=lambda kv: -kv[1])[:8]]
    reopened = closed_with_activity(nodes, mtimes)

    openn = [n for n in nodes if n["lifecycle"] == "open"]
    leaves = [n for n in openn if n["is_leaf"]]
    for n in nodes:
        # every open line, not only the leaves: since a blocking parent can appear in
        # the frontier, one arriving there with no stage is a hole waiting to show
        n["group"] = group_of(n) if n["lifecycle"] == "open" else None
    counts = {
        "nodes": len(nodes),
        "open": len(openn),
        "open_leaves": len(leaves),
        "cashed": sum(1 for n in nodes if n["lifecycle"] == "cashed"),
        "falsified": sum(1 for n in nodes if n["lifecycle"] == "falsified"),
        "unmatched": sum(1 for n in leaves if not n["patterns"]),
        "live": sum(1 for n in leaves if n["signal"] == "live"),
        "stalled": sum(1 for n in leaves if n["signal"] == "stalled"),
        "no_signal": sum(1 for n in leaves if n["signal"] == "none"),
        "landed": len(landed_set),
        "named": len(named_set),
        "unlanded": sum(len(v) for v in buckets.values()),
        "by_group": {g: sum(1 for n in leaves if n["group"] == g) for g in GROUPS},
        "blocking": sum(1 for n in nodes if n.get("blocks") and n["lifecycle"] == "open"),
        "jobs": sum(1 for js in sched.values() for j in js
                    if j.get("state") in ("R", "Q", "H")),
    }

    snap_counts = {g: sum(1 for n in leaves if n["group"] == g) for g in GROUPS}
    snap = dict(snap_counts, at=datetime.datetime.now().astimezone().isoformat(timespec="minutes"),
                closed=sum(1 for n in nodes if n["lifecycle"] != "open"))
    hist = [h for h in ((previous or {}).get("history") or []) if h.get("at") != snap["at"]]
    hist = (hist + [snap])[-240:]

    events = diff_events(previous, {"collected_at":
        datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "nodes": nodes},
        trustworthy=sched_clean or not cfg_clusters())
    history = ((previous or {}).get("events") or []) + events
    history = history[-EVENT_KEEP:]

    return {
        "schema": 4,
        "events": history,
        "new_events": events,
        "project": os.path.basename(ROOT),
        "root": os.path.basename(ROOT),
        "source": os.path.relpath(OUTLINE, ROOT),
        "collected_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "elapsed_ms": int((time.time() - started) * 1000),
        "stage": ("step 3 - every probe wired" if any_sched else
                  "step 3 - schedulers unreachable from this machine, so job state is "
                  "MISSING rather than empty; 'live' below falls back to recent commits"),
        "schedulers_ok": any_sched,
        "schedulers_clean": sched_clean,
        "collectors": collectors,
        "incomplete": incomplete,
        "counts": counts,
        "history": hist,
        "evidence": evidence,
        "have_results": have_results,
        "prefixes": prefix_index,
        "buckets": buckets,
        "coverage": gaps,
        "reopened": reopened,
        "died_early": died_early,
        "unsynced": not_synced,
        "nodes": nodes,
    }


def write_json(data, path=None):
    path = path or OUT_JSON          # resolved at call time, so --project can retarget it
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    return path


def summarise(data, stream=sys.stdout):
    c = data["counts"]
    print("collected %s in %dms" % (data["collected_at"], data["elapsed_ms"]), file=stream)
    for col in data["collectors"]:
        mark = {True: "ok  ", False: "FAIL", None: "--  "}[col["ok"]]
        print("  %s %-11s %s" % (mark, col["name"], col["detail"]), file=stream)
    print("  %d nodes / %d open / %d open leaves (%d live, %d stalled, %d no signal)"
          % (c["nodes"], c["open"], c["open_leaves"], c["live"], c["stalled"],
             c.get("no_signal", 0)), file=stream)
    ev = data.get("evidence", {})
    if ev:
        print("  evidence: %d landed+consumed / %d landed never named / %d named never landed"
              % (len(ev["consumed"]), len(ev["landed_not_named"]),
                 len(ev["named_not_landed"])), file=stream)
        if ev["named_not_landed"]:
            print("    claims point at absent artifacts: %s"
                  % " ".join(ev["named_not_landed"]), file=stream)
    b = data.get("buckets", {})
    if b:
        print("  unlanded %d: %d elsewhere / %d training / %d still running / %d no trace"
              % (sum(len(v) for v in b.values()), len(b["elsewhere"]), len(b["training"]),
                 len(b["running"]), len(b["no_trace"])), file=stream)
        for e in b["elsewhere"]:
            print("    %s -> %s" % (e["prefix"], ",".join(e["to"])), file=stream)
    if c["unmatched"]:
        print("  %d open leaves have no match pattern (they will read 'no signal' forever)"
              % c["unmatched"], file=stream)
    fixable = [m for m in data["incomplete"] if m.get("kind") == "outline"]
    limits = [m for m in data["incomplete"] if m.get("kind") != "outline"]
    if limits:
        print("  %d generator(s) take the prefix as an argument, leaving %d prefixes unverifiable"
              % (len(limits),
              len(data.get("evidence", {}).get("landed_not_named", []))), file=stream)
    for u in data.get("unsynced", [])[:6]:
        if u.get("in_flight"):
            continue
        print("  ~ %s: %d files on the cluster, 0 here" % (u["dir"], u["remote"]), file=stream)
    for j in data.get("died_early", []):
        print("  !! %s %s (%s) finished in %dm against a %dh request"
              % (j["cluster"], j["id"], j["name"], j["elapsed_min"], j["asked_min"] // 60),
              file=stream)
    for r in data.get("reopened", []):
        print("  ! closed line still moving: %s — %s" % (r["name"][:46], r["why"]), file=stream)
    ev = data.get("new_events") or []
    for e in ev:
        mark = {"stalled": "!!", "born": "++", "closed": "--", "gone": "??"}.get(e["kind"], "  ")
        print("  %s %-8s %s%s" % (mark, e["kind"], e["name"][:52],
                                  (" - " + e["detail"]) if e.get("detail") else ""),
              file=stream)
    for m in fixable:
        print("FIX: %s: %s" % (m["where"], m["why"]), file=stream)
    for m in limits:
        print("LIMIT: %s: %s" % (m["where"], m["why"]), file=stream)


# ---------------------------------------------------------------------------- serve

class Handler(http.server.SimpleHTTPRequestHandler):
    state = {"data": None, "page": ""}

    def do_GET(self):                                              # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self._send(self.state["page"].encode("utf-8"), "text/html; charset=utf-8")
        if path == "/lineage.json":
            body = json.dumps(self.state["data"], ensure_ascii=False).encode("utf-8")
            return self._send(body, "application/json; charset=utf-8")
        self.send_error(404)
        return None

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
        return None

    def log_message(self, *args):                                  # noqa: A003
        pass


EMIT_DIR = None
NOTIFY_CMD = None


def serve(interval, port, bind="127.0.0.1"):
    Handler.state["page"] = SHELL % PAGE.replace("__INLINE_DATA__", "null")

    def loop():
        seen = os.path.getmtime(OUTLINE) if os.path.exists(OUTLINE) else 0
        while True:
            try:
                Handler.state["data"] = collect(Handler.state["data"])
                write_json(Handler.state["data"])
                notify(Handler.state["data"]["new_events"], NOTIFY_CMD)
                if EMIT_DIR:
                    emit(Handler.state["data"], EMIT_DIR)
            except Exception as exc:                               # noqa: BLE001
                sys.stderr.write("collect failed: %r\n" % (exc,))
            # Sleep in slices so an edit to the outline is picked up within seconds
            # instead of waiting out the whole interval.
            deadline = time.time() + interval
            while time.time() < deadline:
                time.sleep(min(2.0, max(0.1, deadline - time.time())))
                try:
                    m = os.path.getmtime(OUTLINE)
                except OSError:
                    continue
                if m > seen:
                    seen = m
                    break

    Handler.state["data"] = collect()
    write_json(Handler.state["data"])
    summarise(Handler.state["data"])
    threading.Thread(target=loop, daemon=True).start()

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((bind, port), Handler) as httpd:
        where = "127.0.0.1" if bind in ("127.0.0.1", "localhost") else bind
        print("\nserving http://%s:%d/  (collect every %ds, ctrl-c to stop)"
              % (where, port, interval))
        if bind not in ("127.0.0.1", "localhost"):
            print("bound to %s - anyone who can reach this interface can read the page" % bind)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


TEMPLATE = """# Lineage — %(project)s

<!-- lineage
results: %(results)s
scripts: %(scripts)s
generators: %(generators)s
prefix: %(prefix)s
clusters: %(clusters)s
-->

Structure only. **No state in this file** — no job ids, exit codes, scores or paths.
Those are collected into lineage.json and rot if hand-kept.

Indentation is derivation: a child forked from its parent. `[...]` is the match pattern
the collector uses — a prefix, a `vNN-vMM` range, or a path. `>` is what the line would
establish. `✓` cashed in, `✗` falsified; unmarked means open.

Every config key above is optional. Delete the ones that do not apply and the matching
probe simply reports that it is not configured — it never pretends to have looked.

## Open

- **%(example)s** `[%(exprefix)s]` → name the table or figure this feeds
  > one sentence: what this line would establish or overturn
  - a line that forked off this one `[—]`
    > `[—]` means no version trace: it still appears, marked "no signal"

## Closed — cashed in

- something finished `[—]` ✓ what it became

## Closed — falsified

- something that walked to a negative conclusion `[—]` ✗ what the conclusion was
"""


def init_outline(root=None):
    """Probe the project and write a starter outline. Never overwrites."""
    root = root or ROOT
    path = find_outline(root)
    if os.path.exists(path):
        return path, False

    results = next((d for d in ("results", "outputs", "runs", "out", "artifacts")
                    if os.path.isdir(os.path.join(root, d))), "")
    scripts = ""
    for g in ("run_*.pbs", "*.pbs", "*.sbatch", "scripts/*.sbatch", "jobs/*.sh"):
        if glob.glob(os.path.join(root, g)):
            scripts = g
            break
    generators = [g for g in ("paper/scripts/*.py", "paper_preprint_full/scripts/*.py",
                              "scripts/*.py", "figures/*.py", "analysis/*.py")
                  if glob.glob(os.path.join(root, g))]

    prefix, example = DEFAULTS["prefix"], ""
    if results:
        names = os.listdir(os.path.join(root, results))[:400]
        hits = [re.match(r"([a-z]+)(\d+)([a-z]*)", n) for n in names]
        hits = [h for h in hits if h]
        if names and len(hits) > len(names) * 0.3:
            letters = max({h.group(1) for h in hits}, key=lambda x: (len(x), x))
            prefix = "%s\\d+[a-z]*" % letters
            example = sorted({h.group(0) for h in hits})[0]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(TEMPLATE % {
            "project": os.path.basename(root),
            "results": results, "scripts": scripts,
            "generators": ", ".join(generators),
            "prefix": prefix, "clusters": "",
            "example": "your first line of work",
            "exprefix": example or "—"})
    return path, True


def emit(data, outdir):
    """A deployable pair: index.html (with the current data inlined as a fallback)
    and lineage.json (which the page polls for anything newer). Copy the directory
    anywhere static - the server needs no Python and no cluster credentials."""
    os.makedirs(outdir, exist_ok=True)
    page = PAGE.replace("__INLINE_DATA__", json.dumps(data, ensure_ascii=False))
    with open(os.path.join(outdir, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(SHELL % page)
    write_json(data, os.path.join(outdir, "lineage.json"))
    return outdir


# ------------------------------------------------------------------------------ cli

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", nargs="?", default=None, choices=["collect"])
    ap.add_argument("--once", action="store_true",
                    help="collect, report, exit non-zero if anything failed")
    ap.add_argument("--watch", action="store_true", help="collect on a loop and serve")
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--project", metavar="DIR",
                    help="project directory to read; defaults to the nearest one above cwd "
                         "that holds an outline")
    ap.add_argument("--bind", default="127.0.0.1",
                    help="interface to serve on; use a tailnet address to share with a team")
    ap.add_argument("--notify", metavar="CMD",
                    help="shell command to receive drift notices; the message is appended "
                         "as the final argument. Without it they go to stderr.")
    ap.add_argument("--init", action="store_true",
                    help="probe this project and write a starter outline")
    ap.add_argument("--emit", metavar="DIR",
                    help="write index.html + lineage.json to DIR for static hosting")
    args = ap.parse_args()

    if args.project:
        globals()["ROOT"] = os.path.abspath(args.project)
        globals()["OUTLINE"] = find_outline(ROOT)
        globals()["OUT_JSON"] = os.path.join(os.path.dirname(OUTLINE), "lineage.json")

    if args.init:
        path, made = init_outline()
        print(("wrote %s - edit it, then run: python3 lineage.py --watch" if made
               else "%s already exists, leaving it alone") % os.path.relpath(path, ROOT))
        return 0

    if args.watch:
        globals()["EMIT_DIR"] = args.emit
        globals()["NOTIFY_CMD"] = shlex.split(args.notify) if args.notify else None
        serve(args.interval, args.port, args.bind)
        return 0

    data = collect()
    write_json(data)
    summarise(data)
    notify(data["new_events"], shlex.split(args.notify) if args.notify else None)

    if args.emit:
        print("emitted  -> %s/{index.html,lineage.json}" % emit(data, args.emit))

    if args.once:
        broken = any(c["ok"] is False for c in data["collectors"]) or data["incomplete"]
        return 1 if broken else 0
    return 0


SHELL = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
         '<meta name="viewport" content="width=device-width, initial-scale=1">'
         '</head><body>%s</body></html>\n')

# Page body only - no doctype/html/head/body wrapper, so a snapshot can be published
# as-is by hosts that supply their own skeleton. serve() wraps it in SHELL.
PAGE = r"""<title>Project Lineage</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>
:root{
  --bg:#F5F6F8; --surface:#FFF; --surface-2:#EDEFF3;
  --ink:#161A22; --ink-2:#4A515D; --ink-3:#7C848F;
  --rule:#DCE0E7; --rule-2:#C7CDD6; --accent:#2C4A7C;
  --live:#0E7490; --stall:#B45309; --stall-bg:#FAEEDD;
  --alert:#A33A2A; --alert-bg:#F8E7E3; --done:#8A9099;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  --sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#12161C; --surface:#181D25; --surface-2:#1F252F;
  --ink:#E7EAEF; --ink-2:#A8B0BC; --ink-3:#737C8A;
  --rule:#2A313B; --rule-2:#3A424E; --accent:#7FA3D4;
  --live:#4CB8CC; --stall:#DD9A4B; --stall-bg:#362715;
  --alert:#E08573; --alert-bg:#38201B; --done:#767E8B;
}}
:root[data-theme="dark"]{
  --bg:#12161C; --surface:#181D25; --surface-2:#1F252F;
  --ink:#E7EAEF; --ink-2:#A8B0BC; --ink-3:#737C8A;
  --rule:#2A313B; --rule-2:#3A424E; --accent:#7FA3D4;
  --live:#4CB8CC; --stall:#DD9A4B; --stall-bg:#362715;
  --alert:#E08573; --alert-bg:#38201B; --done:#767E8B;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
     font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased}
img{max-width:100%}
[hidden]{display:none!important}
.wrap{max-width:1080px;margin:0 auto;padding:28px 24px 96px;
      display:flex;flex-direction:column;gap:30px}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}

.masthead{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;
          gap:8px 20px;padding-bottom:12px;border-bottom:2px solid var(--ink)}
.masthead h1{margin:0;font-size:21px;font-weight:700;letter-spacing:-.01em}
.masthead h1 span{font-weight:400;color:var(--ink-3);font-size:15px;margin-left:8px}
.plan-file{font-family:var(--mono);font-size:12px;color:var(--ink-2)}
.plan-file b{font-weight:500;color:var(--ink)}

.health{display:flex;flex-wrap:wrap;align-items:center;gap:4px 22px;padding:11px 16px;
        background:var(--surface);border:1px solid var(--rule);
        border-left:3px solid var(--ink-3);font-family:var(--mono);font-size:12.5px}
.health.aged{border-left-color:var(--stall)}
.health .src{color:var(--ink-2);white-space:nowrap}
.health .src b{font-weight:500}
.ok{color:var(--live)} .warn{color:var(--stall)} .na{color:var(--ink-3)}
.health-note{flex-basis:100%;color:var(--ink-2);font-size:12px;padding-top:6px;
             border-top:1px dashed var(--rule);margin-top:4px}
.health-note.bad{color:var(--alert)}

section{display:flex;flex-direction:column;gap:12px}
.sec-head{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:6px 16px}
.sec-head h2{margin:0;font-size:12px;font-weight:600;letter-spacing:.09em;
             text-transform:uppercase;color:var(--ink-2)}
.tally{font-family:var(--mono);font-size:12.5px;color:var(--ink-2);
       display:flex;gap:16px;flex-wrap:wrap}
.tally b{font-weight:600}

.leaves{overflow-x:auto}
.closed-row{width:100%;min-width:660px;display:grid;
  grid-template-columns:16px minmax(170px,1fr) minmax(0,1.7fr) 74px;gap:12px;
  align-items:baseline;padding:8px 10px 8px 0;background:none;border:0;
  border-left:3px solid var(--done);text-align:left;font:inherit;color:inherit;cursor:pointer}
.closed-row:hover{background:var(--surface)}
.closed-row:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
.cmark{color:var(--done);font-weight:600;text-align:center}
.cmark.falsi{color:var(--stall)}
.cname{font-weight:500}
.cverdict{color:var(--ink-2);font-size:12.5px;overflow:hidden;text-overflow:ellipsis;
          white-space:nowrap}
.cwhen{font-family:var(--mono);font-size:11.5px;color:var(--ink-3);text-align:right}
.leaf-group{min-width:720px;display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 12px;
  padding:13px 10px 4px 0;border-left:3px solid transparent;
  font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--ink-3)}
.leaf-group b{font-weight:600;color:var(--ink-2)}
.leaf-group span{text-transform:none;letter-spacing:0;font-size:11.5px;color:var(--ink-3)}
.leaf:first-of-type{border-top:0}
.leaf-cols{min-width:720px;display:grid;
  grid-template-columns:12px minmax(180px,1.4fr) minmax(96px,132px) 72px 78px minmax(104px,150px);
  align-items:end;gap:12px;padding:0 10px 5px 0;border-left:3px solid transparent;
  border-bottom:1px solid var(--rule-2);
  font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--ink-3);
  font-weight:600}
.leaf{border-bottom:1px solid var(--rule)}
.leaf-head{width:100%;min-width:720px;display:grid;
  grid-template-columns:12px minmax(180px,1.4fr) minmax(96px,132px) 72px 78px minmax(104px,150px);
  align-items:center;gap:12px;padding:9px 10px 9px 0;background:none;border:0;
  border-left:3px solid var(--done);text-align:left;font:inherit;color:inherit;cursor:pointer}
.leaf-head:hover{background:var(--surface)}
.leaf-head:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
.leaf[data-signal="live"]  .leaf-head{border-left-color:var(--live)}
.leaf[data-signal="stalled"] .leaf-head{border-left-color:var(--stall)}
.leaf[data-signal="blocking"] .leaf-head{border-left-color:var(--alert);
  background:linear-gradient(90deg,var(--alert-bg),transparent 340px)}
.blk{display:inline-block;width:7px;height:7px;background:var(--alert);margin-right:7px;
     transform:rotate(45deg);vertical-align:middle}
#g-blocking b{color:var(--alert)}
.caret{justify-self:center;color:var(--ink-3);font-size:10px;transition:transform .14s ease}
.leaf-head[aria-expanded="true"] .caret{transform:rotate(90deg)}
.leaf-name{font-weight:500}
.leaf-sub{display:block;font-family:var(--mono);font-size:11.5px;color:var(--ink-3);font-weight:400}
.cell{font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:12.5px;
      color:var(--ink-2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0}
.leaf-name{min-width:0;overflow:hidden}
.leaf-sub{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.legend{display:flex;flex-wrap:wrap;gap:6px 20px;font-size:11.5px;color:var(--ink-3);
        padding:2px 2px 4px}
.legend .sw{display:inline-block;width:9px;height:3px;margin-right:6px;vertical-align:middle}
.legend .sw.live{background:var(--live)}
.legend .sw.stall{background:var(--stall)}
.legend .sw.none{background:var(--done)}
.cell.dim{color:var(--ink-3)}
.target{color:var(--accent)} .target.none{color:var(--stall)}
.stalled-age{color:var(--stall);font-weight:500}
.running-job{color:var(--live);font-weight:600}

.leaf-body{padding:4px 16px 20px 30px;display:flex;flex-direction:column;gap:16px;font-size:13px}
.consequence{display:flex;gap:9px;padding:9px 12px;background:var(--alert-bg);
  border-left:3px solid var(--alert);color:var(--alert);font-size:12.5px;max-width:76ch}
.question{color:var(--ink);max-width:68ch;padding-left:11px;border-left:2px solid var(--rule-2)}
.question b{display:block;font-size:11px;letter-spacing:.08em;text-transform:uppercase;
            color:var(--ink-3);font-weight:600;margin-bottom:2px}
.block-label{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3);
             font-weight:600;margin-bottom:6px}
.anc{display:grid;grid-template-columns:18px minmax(0,1fr) 64px;gap:10px;align-items:baseline;
     padding:5px 0;border-bottom:1px dotted var(--rule)}
.anc:last-child{border-bottom:0}
.anc-arrow{color:var(--ink-3);font-family:var(--mono)}
.anc-node{font-size:13px;color:var(--ink)}
.anc-why{color:var(--ink-2);font-style:italic}
.anc-age{font-family:var(--mono);font-size:11.5px;color:var(--ink-3);text-align:right}
.versions{display:flex;flex-wrap:wrap;gap:5px;align-items:center}
.vchip{font-family:var(--mono);font-size:11.5px;padding:2px 7px;border:1px solid var(--rule-2);
       background:var(--surface);color:var(--ink-2);white-space:nowrap}
.vchip.none{border-style:dashed;color:var(--stall);border-color:var(--stall)}
.vchip.landed{border-color:var(--accent);color:var(--accent)}
.retry-note{font-family:var(--mono);font-size:11.5px;color:var(--stall);margin-left:4px}
.files{font-family:var(--mono);font-size:11.5px;color:var(--ink-3);word-break:break-all}

.tree{padding:10px 15px 14px;overflow-x:auto;border-top:1px solid var(--rule)}
.tnode{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:14px;align-items:baseline;
       padding:3px 0;border-bottom:1px dotted var(--rule)}
.tnode:last-child{border-bottom:0}
.tname{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tname .rail{color:var(--rule-2);font-family:var(--mono);white-space:pre}
.tnode[data-leaf="1"] .tname b{font-weight:600}
.tnode .tmeta{font-family:var(--mono);font-size:11.5px;color:var(--ink-3);white-space:nowrap}
.tnode .tmeta .lf{color:var(--stall)}
.tdot{display:inline-block;width:6px;height:6px;border-radius:50%;margin-right:7px;
      vertical-align:middle}
.tdot.run{background:var(--live)}
.tdot.chg{background:var(--live);opacity:.45}
.tdot.stl{background:var(--stall)}
.tdot.unk{background:transparent;border:1.5px solid var(--done)}

details.fold{border:1px solid var(--rule);background:var(--surface)}
details.fold>summary{padding:11px 15px;cursor:pointer;display:flex;flex-wrap:wrap;gap:8px 20px;
                     align-items:baseline;font-size:13px;list-style:none}
details.fold>summary::-webkit-details-marker{display:none}
details.fold>summary::before{content:"▸";color:var(--ink-3);font-size:10px}
details.fold[open]>summary::before{content:"▾"}
details.fold>summary b{font-weight:600}
details.fold>summary .n{font-family:var(--mono);color:var(--ink-2)}
.fold-body{padding:4px 15px 16px;border-top:1px solid var(--rule);font-size:12.5px;
           color:var(--ink-2);display:flex;flex-direction:column;gap:2px}
.frow{display:grid;grid-template-columns:minmax(0,1fr) 190px;gap:12px;padding:4px 0;
      border-bottom:1px dotted var(--rule)}
.frow:last-child{border-bottom:0}
.frow .to{font-family:var(--mono);font-size:12px;color:var(--accent)}
.frow .vd{font-family:var(--mono);font-size:12px;color:var(--ink-3)}

.incomplete{background:var(--stall-bg);border-left:3px solid var(--stall);padding:11px 15px;
            font-family:var(--mono);font-size:12px;color:var(--stall);
            display:flex;flex-direction:column;gap:4px}
.inc-why{font-family:var(--sans);font-size:12.5px;color:var(--ink-2);max-width:78ch;
         padding-bottom:5px;margin-bottom:3px;border-bottom:1px dashed var(--rule-2)}
.coverage{border-top:1px solid var(--rule-2);padding-top:14px;font-size:12.5px;color:var(--ink-2);
          display:flex;flex-direction:column;gap:7px}
.coverage .hd{font-weight:600;color:var(--ink)}
.stack{display:flex;height:26px;gap:1px;background:var(--rule);border:1px solid var(--rule)}
.seg{display:flex;align-items:center;justify-content:center;min-width:3px}
.seg span{font-family:var(--mono);font-size:11.5px;font-weight:600;color:#fff;
          mix-blend-mode:normal}
.b-ok{background:var(--accent)} .b-warn{background:var(--stall)} .b-bad{background:var(--alert)}
.stack-key{display:flex;flex-wrap:wrap;gap:4px 18px;font-size:11.5px;color:var(--ink-3);
           padding-top:2px}
.stack-key i{display:inline-block;width:9px;height:9px;margin-right:6px;vertical-align:-1px}
.stack-key .b-ok,.stack-key .b-warn,.stack-key .b-bad{border-radius:2px}
.trend{background:var(--surface);border:1px solid var(--rule);padding:8px 6px 2px;
       overflow-x:auto}
.trend svg{display:block;min-width:600px;width:100%;height:auto}
.buckets{display:grid;grid-template-columns:repeat(auto-fit,minmax(228px,1fr));gap:1px;
         background:var(--rule);border:1px solid var(--rule)}
.bucket{background:var(--surface);padding:13px 15px;display:flex;flex-direction:column;gap:6px}
.bucket.focus{background:var(--alert-bg)}
.bucket.caution{background:var(--stall-bg)}
.bucket-top{display:flex;align-items:baseline;justify-content:space-between;gap:10px}
.bucket-name{font-weight:600;font-size:13px}
.bucket-n{font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:19px;
          font-weight:600;color:var(--ink)}
.bucket.focus .bucket-n,.bucket.focus .bucket-name{color:var(--alert)}
.bucket.caution .bucket-n,.bucket.caution .bucket-name{color:var(--stall)}
.bucket-list{font-family:var(--mono);font-size:11.5px;color:var(--ink-2);line-height:1.7;
             word-break:break-word}
.bucket-why{font-size:11.5px;color:var(--ink-3)}
.summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(122px,1fr));gap:1px;
         background:var(--rule);border:1px solid var(--rule)}
.tile{background:var(--surface);padding:13px 15px 12px;display:flex;flex-direction:column;
      gap:1px;text-decoration:none;color:inherit;cursor:pointer;
      border-top:2px solid transparent}
.tile:hover{background:var(--surface-2)}
.tile:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
.tile b{font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:27px;
        font-weight:600;line-height:1.05;letter-spacing:-.02em}
.tile span{font-size:10.5px;letter-spacing:.07em;text-transform:uppercase;color:var(--ink-3)}
.t-run{border-top-color:var(--live)}      .t-run b{color:var(--live)}
.t-warn{border-top-color:var(--stall)}    .t-warn b{color:var(--stall)}
.t-bad{border-top-color:var(--alert)}     .t-bad b{color:var(--alert)}
.t-mute{border-top-color:var(--rule-2)}   .t-mute b{color:var(--ink-2)}
@keyframes flashbg{from{background:var(--stall-bg)}to{background:transparent}}
.flash{animation:flashbg 1.1s ease-out}
@media (prefers-reduced-motion:reduce){.flash{animation:none;outline:2px solid var(--stall)}}
.still{background:var(--stall-bg);border-left:3px solid var(--stall);padding:10px 15px;
        font-size:12.5px;color:var(--ink-2);max-width:none}
.still b{color:var(--stall)}
.chore{font-family:var(--mono);font-size:11.5px;color:var(--ink-3);padding:0 2px}
.bucket.quiet{background:var(--surface)}
.bucket.quiet .bucket-n,.bucket.quiet .bucket-name{color:var(--ink-3)}
.bucket.quiet .bucket-list{color:var(--ink-3);opacity:.75}
.stage{font-family:var(--mono);font-size:12px;color:var(--stall);background:var(--stall-bg);
       padding:8px 13px;border-left:3px solid var(--stall)}
@media (max-width:700px){.wrap{padding:20px 14px 72px}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
<div class="wrap" id="app"><div class="stage">loading…</div></div>

<script id="inline-data" type="application/json">__INLINE_DATA__</script>
<script>
var DATA = null, poll = null;

function esc(s){ return String(s == null ? "" : s)
  .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

function el(tag, cls, html){
  var n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html != null) n.innerHTML = html;
  return n;
}

function byId(nodes){ var m = {}; nodes.forEach(function(n){ m[n.id] = n; }); return m; }

function ancestry(node, map){
  var out = [], cur = node;
  while (cur.parent && map[cur.parent]) { cur = map[cur.parent]; out.push(cur); }
  return out;
}

function childrenOf(id, nodes){ return nodes.filter(function(n){ return n.parent === id; }); }

function ageOf(n){
  if (n.jobs) return n.jobs + "j";
  return stamp(n.last_activity || n.written_at);
}
var MON = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
function stamp(iso){
  if (!iso) return "—";
  var t = new Date(iso), now = new Date();
  if (t.toDateString() === now.toDateString())
    return ("0" + t.getHours()).slice(-2) + ":" + ("0" + t.getMinutes()).slice(-2);
  var y = new Date(now.getTime() - 864e5);
  if (t.toDateString() === y.toDateString()) return "yesterday";
  return t.getDate() + " " + MON[t.getMonth()];
}
function span(h){
  if (h == null) return "";
  return h < 48 ? Math.round(h) + "h ago" : (h < 720 ? Math.round(h / 24) + "d ago"
                                                     : Math.round(h / 720) + "mo ago");
}
function ageText(n){
  if (n.carried_over) return "held over";
  if (n.kind === "decision") return n.written_at ? stamp(n.written_at) : "new";
  if (n.signal === "none") return n.written_at ? stamp(n.written_at) : "new";
  return n.last_activity ? stamp(n.last_activity) : "new";
}
function ageTitle(n){
  if (n.jobs) return n.jobs + " job(s) in the queue";
  if (n.carried_over) return "the scheduler could not be reached; this is the previous reading";
  if (n.signal === "none")
    return n.written_at ? "written into the outline " + span(n.written_hours) +
                          ", untouched since"
                        : "written but not committed yet, so git has no date for it";
  return n.last_activity ? "last commit " + span(n.idle_hours)
                         : "written but not committed yet, so git has no date for it";
}
function ageClass(n){
  if (n.carried_over) return "stalled-age";
  if (n.signal === "none") return "dim";
  return n.signal === "stall" || n.signal === "stalled" ? "stalled-age" : "";
}
function ageLabel(iso){
  if (!iso) return "—";
  var ms = Date.now() - new Date(iso).getTime();
  var h = ms / 3.6e6;
  if (h < 1) return Math.max(1, Math.round(ms / 6e4)) + "m";
  if (h < 48) return Math.round(h) + "h";
  return Math.round(h / 24) + "d";
}

function render(d){
  var app = document.getElementById("app");
  app.innerHTML = "";
  var map = byId(d.nodes);
  var open = d.nodes.filter(function(n){ return n.lifecycle === "open"; });
  /* open leaves, plus any open line that blocks submission: a blocker on a
     parent covers every line under it, and must not be the one thing the
     banner counts and the list never shows */
  var leaves = open.filter(function(n){ return n.is_leaf || n.blocks; });

  /* masthead */
  var mast = el("div", "masthead");
  mast.appendChild(el("h1", null, esc(d.project || d.root) +
    " Lineage<span>" + esc(d.root) + "</span>"));
  document.title = (d.project || d.root) + " Lineage";
  mast.appendChild(el("div", "plan-file",
    "structure &nbsp;<b>" + esc(d.source) + "</b> &nbsp;·&nbsp; " +
    d.counts.nodes + " nodes"));
  app.appendChild(mast);

  /* collector health */
  var age = (Date.now() - new Date(d.collected_at).getTime()) / 1000;
  var health = el("div", "health" + (age > 120 ? " aged" : ""));
  health.appendChild(el("span", "src", "collected&nbsp; <b>" +
    esc(d.collected_at.replace("T", " ").slice(0, 19)) + "</b>" +
    (age > 120 ? " <span class='warn'>(" + Math.round(age / 60) + "m ago)</span>" : "")));
  d.collectors.forEach(function(c){
    var cls = c.ok === true ? "ok" : (c.stale ? "warn" : (c.ok === false ? "warn" : "na"));
    var word = c.ok === true ? "ok"
             : (c.stale ? "STALE " + c.stale_minutes + "m"
             : (c.ok === false ? "FAIL" : "—"));
    health.appendChild(el("span", "src",
      esc(c.name) + " <b class='" + cls + "'>" + word + "</b>"));
  });
  var bad = d.collectors.filter(function(c){ return c.ok === false; });
  var stale = bad.filter(function(c){ return c.stale; });
  health.appendChild(el("div", "health-note" + (bad.length ? " bad" : ""),
    bad.length
      ? "⚠ " + esc(bad.map(function(c){ return c.name + ": " + c.detail; }).join(" · ")) +
        (stale.length ? "  Figures below come from the collection " + stale[0].stale_minutes +
                        "m ago." : "") +
        (d.schedulers_clean === false
          ? "  Lines marked “held over” keep that earlier state, and no drift alert was raised."
          : "")
      : ""));
  app.appendChild(health);

  var died = (d.died_early || []).length;
  var stale2 = (d.unsynced || []).filter(function(u){ return !u.in_flight; });
  var moving = (d.reopened || []).length;
  var uncom = ((d.coverage || {}).uncommitted || []).length;
  var g = d.counts.by_group || {};

  var tiles = [
    ["bad",  d.counts.blocking || 0, "blocking submission", "#g-blocking"],
    ["run",  g.running || 0, "lines running",   "#g-running"],
    ["warn", g.stalled || 0, "stalled",          "#g-stalled"],
    ["bad",  died,           "died early",       "#died"],
    ["warn", stale2.length,  "waiting to sync",  "#sync"],
    ["warn", moving,         "closed but moving","#moving"],
    ["warn", g.decision || 0, "awaiting your call","#g-decision"],
    ["mute", g.waiting || 0, "no experiment yet","#g-waiting"],
    ["mute", uncom,          "uncommitted",      "#uncommitted"]
  ].filter(function(t, i){ return i === 1 || t[1] > 0; });

  var sum = el("div", "summary");
  tiles.forEach(function(t){
    var a = document.createElement("a");
    a.className = "tile t-" + t[0];
    a.href = t[3];
    a.innerHTML = "<b>" + t[1] + "</b><span>" + esc(t[2]) + "</span>";
    a.addEventListener("click", function(ev){
      ev.preventDefault();
      var target = document.querySelector(t[3]);
      if (!target) return;
      target.scrollIntoView({behavior: "smooth", block: "start"});
      target.classList.remove("flash"); void target.offsetWidth;
      target.classList.add("flash");
    });
    sum.appendChild(a);
  });
  app.appendChild(sum);

  var notes = [];
  var dark = d.collectors.filter(function(c){ return c.ok === null; });
  if (dark.length) notes.push(dark.length + " probe" + (dark.length > 1 ? "s" : "") +
    " not wired: " + dark.map(function(c){ return c.name; }).join(", "));
  if (d.counts.no_signal) notes.push(d.counts.no_signal + " of the " + d.counts.open_leaves +
    " lines have no match pattern in " + esc(d.source) +
    ", so their state is unverifiable — add one and they light up");
  if (notes.length) app.appendChild(el("div", "chore", notes.join(" · ")));


  var fixable = (d.incomplete || []).filter(function(m){ return m.kind === "outline"; });
  var limits  = (d.incomplete || []).filter(function(m){ return m.kind !== "outline"; });
  if (fixable.length){
    var inc = el("div", "incomplete");
    inc.appendChild(el("div", null, "<b>" + fixable.length +
      " thing" + (fixable.length > 1 ? "s" : "") + " to fix in " + esc(d.source) + "</b>"));
    inc.appendChild(el("div", "inc-why",
      "The outline itself is malformed here, so these nodes are read wrong or not at all."));
    fixable.forEach(function(m){
      inc.appendChild(el("div", null, esc(m.where) + "  —  " + esc(m.why)));
    });
    app.appendChild(inc);
  }

  /* a job that ended far sooner than it asked for */
  if ((d.died_early || []).length){
    var dw = el("div", "incomplete");
    dw.id = "died";
    var certain = d.died_early.filter(function(j){ return j.certain; }).length;
    dw.appendChild(el("div", null, "<b>" + d.died_early.length + " job" +
      (d.died_early.length > 1 ? "s" : "") + " ended in minutes, not hours</b>"));
    dw.appendChild(el("div", "inc-why",
      certain === d.died_early.length
        ? "The scheduler calls these failures. In the queue they look like any other finished job."
        : "In the queue these look like any other finished job."));
    d.died_early.forEach(function(j){
      dw.appendChild(el("div", null, esc(j.cluster + " " + j.id + "  " + j.name) +
        "  \u2014  ran " + j.elapsed_min + "m against a " + Math.round(j.asked_min / 60) +
        "h request" + (j.certain ? ", scheduler says failed" : "")));
    });
    app.appendChild(dw);
  }

  var waiting = (d.unsynced || []).filter(function(u){ return !u.in_flight; });
  if (waiting.length){
    var uw = el("div", "incomplete");
    uw.id = "sync";
    uw.appendChild(el("div", null, "<b>" + waiting.length +
      " result set" + (waiting.length > 1 ? "s are" : " is") + " still on the cluster</b>"));
    uw.appendChild(el("div", "inc-why",
      "Nothing has been copied back yet, so the lines they belong to look idle."));
    waiting.forEach(function(u){
      uw.appendChild(el("div", null, esc(u.dir) + "  —  " + u.remote + " files there, 0 here"));
    });
    app.appendChild(uw);
  }


  /* a line declared done that is still moving - it hides between the two lists */
  if ((d.reopened || []).length){
    var rw = el("div", "incomplete");
    rw.id = "moving";
    rw.appendChild(el("div", null, "<b>" + d.reopened.length +
      " closed line" + (d.reopened.length > 1 ? "s are" : " is") + " still moving</b>"));
    rw.appendChild(el("div", "inc-why",
      "Work is happening under a line you already closed. Either it is not finished, or " +
      "something new started there and was never written down."));
    d.reopened.forEach(function(r){
      rw.appendChild(el("div", null, esc(r.name) + "  —  " + esc(r.why)));
    });
    app.appendChild(rw);
  }

  /* frontier */
  var sec = el("section");
  sec.id = "frontier";
  var head = el("div", "sec-head");
  head.appendChild(el("h2", null, "Frontier · what is open now"));
  head.appendChild(el("div", "tally",
    "<span><b>" + leaves.length + "</b> open lines, grouped by what they are doing</span>"));
  sec.appendChild(head);
  var legend = el("div", "legend");
  legend.innerHTML =
    '<span>sorted by what is burning, then by how long it has sat</span>' +
    '<span>“no paper target” = the line never said what it would change in the paper</span>';
  sec.appendChild(legend);

  var GROUPS = [
    ["running",  "RUNNING",          "a job is in the queue right now"],
    ["changed",  "RECENTLY CHANGED", "no job, but code moved in the last 48h"],
    ["writeup",  "NOT WRITTEN UP YET", "these point at a manuscript file that has not been edited"],
    ["stalled",  "STALLED",          "it has a prefix and results, and nothing has moved"],
    ["planned",  "PLANNED",          "a prefix is written down; nothing has landed under it yet"],
    ["decision", "AWAITING YOUR CALL", "these close by a judgement, not by an experiment — " +
                                       "each one is minutes of your attention, not GPU time"],
    ["waiting",  "NO EXPERIMENT YET", "nothing is attached yet — give one a `[prefix]` when you start the run"]
  ];
  function groupOf(n){ return n.group || "waiting"; }
  var rank = {}; GROUPS.forEach(function(g, i){ rank[g[0]] = i; });
  leaves.sort(function(a, b){
    if (!!a.blocks !== !!b.blocks) return a.blocks ? -1 : 1;
    var d = rank[groupOf(a)] - rank[groupOf(b)];
    if (d) return d;
    if (groupOf(a) === "running") return (b.jobs || 0) - (a.jobs || 0);
    if (groupOf(a) === "waiting" || groupOf(a) === "planned")
      return (b.written_hours || 0) - (a.written_hours || 0);
    return (b.idle_hours || 0) - (a.idle_hours || 0);   // the oldest sits highest
  });

  var box = el("div", "leaves");
  var hdr = el("div", "leaf-cols");
  hdr.innerHTML = "<span></span><span>line</span><span>tracked as</span>" +
                  "<span>files</span><span>last moved</span><span>feeds</span>";
  box.appendChild(hdr);
  var lastGroup = null;
  leaves.forEach(function(n){
    var gk = n.blocks ? "blocking" : groupOf(n);
    if (gk !== lastGroup){
      lastGroup = gk;
      var meta = gk === "blocking"
        ? ["blocking", "BLOCKING SUBMISSION", "these stand between the paper and going out"]
        : GROUPS[rank[gk]];
      var count = leaves.filter(function(x){
        return (x.blocks ? "blocking" : groupOf(x)) === gk; }).length;
      var gh = el("div", "leaf-group");
      gh.id = "g-" + gk;
      gh.innerHTML = "<b>" + meta[1] + " · " + count + "</b><span>" + esc(meta[2]) + "</span>";
      box.appendChild(gh);
    }
    var leaf = el("div", "leaf");
    leaf.setAttribute("data-signal", n.blocks ? "blocking"
                                   : (n.signal === "none" ? "" : n.signal));

    var parent = n.parent && map[n.parent] ? map[n.parent].name : null;
    var btn = el("button", "leaf-head");
    btn.setAttribute("aria-expanded", "false");
    btn.innerHTML =
      '<span class="caret">▶</span>' +
      '<span class="leaf-name" title="' + esc(d.source + ":" + n.line) + '">' +
        (n.blocks ? '<i class="blk" title="' + esc(n.blocks) + '"></i>' : '') + esc(n.name) +
        (n.job_detail && n.job_detail.length
           ? '<span class="leaf-sub">' + esc(n.job_detail.join(" · ")) + '</span>'
           : !n.is_leaf ? '<span class="leaf-sub">covers every line under it</span>'
           : parent ? '<span class="leaf-sub">under ' + esc(parent) + '</span>' : '') +
      '</span>' +
      '<span class="cell' + (n.patterns.length ? '' : ' dim') + '" title="' +
        (n.patterns.length ? esc(n.patterns.join(" "))
                           : "no match pattern in the outline — nothing for a probe to look at") +
        '">' + (n.patterns.length ? esc(n.patterns.join(" ")) : "not tracked") + '</span>' +
      '<span class="cell' + (n.artifacts ? '' : ' dim') + '">' +
        (n.signal === "none" ? "—" : n.artifacts + " art") + '</span>' +
      '<span class="cell ' + ageClass(n) + '" title="' + esc(ageTitle(n)) + '">' +
        ageText(n) + '</span>' +
      '<span class="cell ' + (n.claim ? "target" : "target none") + '" title="' +
        (n.claim ? esc(n.claim) : "this line does not say what it would change in the paper") +
        '">' + (n.claim ? "→ " + esc(n.claim) : "no paper target") + '</span>';

    var body = el("div", "leaf-body");
    body.hidden = true;
    if (n.blocks){
      body.appendChild(el("div", "consequence",
        "<span>⚠</span><span><b>Blocks submission.</b> " + esc(n.blocks) + "</span>"));
    }
    if (n.intent){
      body.appendChild(el("div", "question",
        "<b>opened to establish</b>" + esc(n.intent)));
    }
    var chain = ancestry(n, map);
    if (chain.length){
      var wrap = el("div");
      wrap.appendChild(el("div", "block-label", "derived from"));
      chain.forEach(function(a){
        var row = el("div", "anc");
        row.innerHTML = '<span class="anc-arrow">↑</span>' +
          '<span><span class="anc-node">' + esc(a.name) + '</span>' +
          (a.intent ? '<span class="anc-why"> — ' + esc(a.intent) + '</span>' : '') + '</span>' +
          '<span class="anc-age">' + esc(ageOf(a)) + '</span>';
        wrap.appendChild(row);
      });
      body.appendChild(wrap);
    }
    var pw = el("div");
    var hit = n.matched_prefixes || [];
    var missing = n.declared_unlanded || [];
    pw.appendChild(el("div", "block-label",
      hit.length || missing.length ? "attempts" : "match patterns"));
    var chips = el("div", "versions");
    if (hit.length || missing.length){
      hit.forEach(function(b){
        var meta = (d.prefixes || {})[b] || {};
        var c = el("span", "vchip" + (meta.consumed ? " landed" : ""), esc(b) +
          (meta.files ? ' <span style="opacity:.6">' + meta.files + '</span>' : ''));
        c.title = meta.consumed ? "a generator file quotes this prefix"
                                : "artifacts exist; no generator quotes it";
        chips.appendChild(c);
      });
      missing.forEach(function(b){
        var c = el("span", "vchip none", esc(b));
        c.title = "declared in the outline, nothing landed under it";
        chips.appendChild(c);
      });
      if (hit.length > 3)
        chips.appendChild(el("span", "retry-note", "↻ " + hit.length + " attempts"));
    } else if (n.patterns.length){
      n.patterns.forEach(function(p){ chips.appendChild(el("span", "vchip", esc(p))); });
    } else {
      chips.appendChild(el("span", "vchip none",
        n.declared_pattern ? "declared: no version trace" : "none declared"));
    }
    pw.appendChild(chips);
    body.appendChild(pw);
    if (n.claim || n.verdict){
      body.appendChild(el("div", "files", esc(n.verdict || "")));
    }

    btn.addEventListener("click", function(){
      var isOpen = btn.getAttribute("aria-expanded") === "true";
      btn.setAttribute("aria-expanded", isOpen ? "false" : "true");
      body.hidden = isOpen;
    });
    leaf.appendChild(btn);
    leaf.appendChild(body);
    box.appendChild(leaf);
  });
  sec.appendChild(box);
  app.appendChild(sec);



  /* what closed lately - finishing things is half of what a person wants to see */
  var closedEv = (d.events || []).filter(function(e){ return e.kind === "closed"; });
  if (closedEv.length){
    var byName = {}; d.nodes.forEach(function(n){ byName[n.name] = n; });
    var cs2 = el("section");
    var ch = el("div", "sec-head");
    ch.appendChild(el("h2", null, "Recently closed · " + closedEv.length));
    ch.appendChild(el("div", "tally", "<span>✓ became a claim · ✗ walked to a negative " +
      "conclusion, which is paper material</span>"));
    cs2.appendChild(ch);
    var cb = el("div", "leaves");
    closedEv.slice().reverse().slice(0, 12).forEach(function(e){
      var n = byName[e.name] || {};
      var mark = n.lifecycle === "falsified" ? "✗" : "✓";
      // the conclusion lives in the ">" line; the verdict field is often just a target
      var said = n.intent || n.verdict || "";
      var wrap = el("div", "leaf");
      var btn = el("button", "closed-row");
      btn.setAttribute("aria-expanded", "false");
      btn.innerHTML =
        '<span class="cmark ' + (mark === "✗" ? "falsi" : "") + '">' + mark + '</span>' +
        '<span class="cname">' + esc(e.name) + '</span>' +
        '<span class="cverdict">' + esc(said || n.verdict || "—") + '</span>' +
        '<span class="cwhen">' + esc(stamp(e.at)) + '</span>';
      var body = el("div", "leaf-body");
      body.hidden = true;
      if (said) body.appendChild(el("div", "question",
        "<b>what it concluded</b>" + esc(said)));
      if (n.verdict) body.appendChild(el("div", "question",
        "<b>where it landed</b>" + esc(n.verdict)));
      var chain = n.id ? ancestry(n, map) : [];
      if (chain.length){
        var cw = el("div");
        cw.appendChild(el("div", "block-label", "derived from"));
        chain.forEach(function(a){
          var row = el("div", "anc");
          row.innerHTML = '<span class="anc-arrow">↑</span>' +
            '<span><span class="anc-node">' + esc(a.name) + '</span>' +
            (a.intent ? '<span class="anc-why"> — ' + esc(a.intent) + '</span>' : '') +
            '</span><span class="anc-age">' + esc(ageOf(a)) + '</span>';
          cw.appendChild(row);
        });
        body.appendChild(cw);
      }
      btn.addEventListener("click", function(){
        var open2 = btn.getAttribute("aria-expanded") === "true";
        btn.setAttribute("aria-expanded", open2 ? "false" : "true");
        body.hidden = open2;
      });
      wrap.appendChild(btn); wrap.appendChild(body);
      cb.appendChild(wrap);
    });
    cs2.appendChild(cb);
    app.appendChild(cs2);
  }

  /* full open tree */
  var t = el("section");

  var treeFold = el("details", "fold");
  treeFold.open = true;
  treeFold.innerHTML = "<summary><b>Derivation</b> <span class='n'>" + open.length +
    " open lines · indentation is what forked from what · leaves in bold</span></summary>";
  var tree = el("div", "tree");
  open.forEach(function(n){
    var row = el("div", "tnode");
    row.setAttribute("data-leaf", n.is_leaf ? "1" : "0");
    var rail = n.depth ? "  ".repeat(n.depth - 1) + "└─ " : "";
    var dot = n.is_leaf
      ? '<i class="tdot ' + (n.jobs ? "run" : (n.signal === "live" ? "chg"
        : (n.signal === "stalled" ? "stl" : "unk"))) + '"></i>' : '';
    row.innerHTML =
      '<span class="tname"><span class="rail">' + esc(rail) + '</span>' + dot +
        (n.is_leaf ? "<b>" + esc(n.name) + "</b>" : esc(n.name)) + '</span>' +
      '<span class="tmeta">' +
        (n.patterns.length ? esc(n.patterns.join(" ")) : "—") +
        (n.claim ? " → " + esc(n.claim) : "") +
        (n.is_leaf ? ' <span class="lf">leaf</span>' : "") + '</span>';
    tree.appendChild(row);
  });
  treeFold.appendChild(tree);
  t.appendChild(treeFold);
  app.appendChild(t);

  /* how the frontier has moved - only drawn once there is real history */
  var hist = (d.history || []).filter(function(h){ return h && h.at; });
  var hspanH = hist.length >= 2
    ? (new Date(hist[hist.length - 1].at) - new Date(hist[0].at)) / 3.6e6 : 0;
  if (hist.length >= 4 && hspanH >= 6){
    var ts = el("section");
    var th2 = el("div", "sec-head");
    th2.appendChild(el("h2", null, "Frontier over time"));
    th2.appendChild(el("div", "tally", "<span>" + hist.length +
      " collections, most recent on the right</span>"));
    ts.appendChild(th2);
    var W = 900, H = 130, PAD = 24;
    var keys = [["running", "var(--live)"], ["changed", "var(--live)"],
                ["stalled", "var(--stall)"], ["planned", "var(--accent)"],
                ["waiting", "var(--done)"]];
    var maxv = 1;
    hist.forEach(function(h){
      var t = keys.reduce(function(a, k){ return a + (h[k[0]] || 0); }, 0);
      if (t > maxv) maxv = t;
    });
    var step = hist.length > 1 ? (W - PAD * 2) / (hist.length - 1) : 0;
    var svg = ['<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" ' +
               'aria-label="stacked count of frontier lines at each collection">'];
    var base = new Array(hist.length).fill(H - PAD);
    keys.forEach(function(k){
      var top = [], i;
      for (i = 0; i < hist.length; i++){
        var v = (hist[i][k[0]] || 0) / maxv * (H - PAD * 2);
        top.push(base[i] - v);
      }
      var dpath = "M" + PAD + " " + base[0];
      for (i = 0; i < hist.length; i++) dpath += " L" + (PAD + i * step) + " " + top[i];
      for (i = hist.length - 1; i >= 0; i--) dpath += " L" + (PAD + i * step) + " " + base[i];
      svg.push('<path d="' + dpath + 'Z" fill="' + k[1] + '" fill-opacity="' +
               (k[0] === "changed" ? ".28" : k[0] === "waiting" ? ".22" : ".55") +
               '" stroke="none"/>');
      base = top;
    });
    svg.push('<line x1="' + PAD + '" y1="' + (H - PAD) + '" x2="' + (W - PAD) +
             '" y2="' + (H - PAD) + '" stroke="var(--rule-2)" stroke-width="1"/>');
    svg.push('<text x="' + PAD + '" y="' + (H - 8) + '" font-size="10" ' +
             'fill="var(--ink-3)" font-family="IBM Plex Mono, monospace">' +
             esc(hist[0].at.slice(5, 16).replace("T", " ")) + '</text>');
    svg.push('<text x="' + (W - PAD) + '" y="' + (H - 8) + '" font-size="10" text-anchor="end" ' +
             'fill="var(--ink-3)" font-family="IBM Plex Mono, monospace">' +
             esc(hist[hist.length - 1].at.slice(5, 16).replace("T", " ")) + '</text>');
    svg.push('<text x="' + PAD + '" y="16" font-size="10" fill="var(--ink-3)" ' +
             'font-family="IBM Plex Mono, monospace">' + maxv + ' lines</text></svg>');
    var box2 = el("div", "trend", svg.join(""));
    ts.appendChild(box2);
    app.appendChild(ts);
  }

  /* evidence */
  if (d.evidence && d.have_results !== false){
    var es = el("section");
    var eh = el("div", "sec-head");
    eh.appendChild(el("h2", null, "Evidence · artifacts against claims"));
    eh.appendChild(el("div", "tally", "<span>" + d.counts.landed +
      " prefixes landed · " + d.counts.named + " named in generator code</span>"));
    es.appendChild(eh);
    var ev3 = [["consumed", "read by the paper", "b-ok"],
               ["landed_not_named", "status unknown", "b-warn"],
               ["named_not_landed", "missing", "b-bad"]];
    var tot = ev3.reduce(function(a, x){ return a + (d.evidence[x[0]] || []).length; }, 0);
    if (tot){
      var bar = el("div", "stack");
      ev3.forEach(function(x){
        var n = (d.evidence[x[0]] || []).length;
        if (!n) return;
        var seg = el("div", "seg " + x[2]);
        seg.style.flexGrow = n;
        seg.title = n + " " + x[1];
        seg.innerHTML = n >= tot * 0.12 ? "<span>" + n + "</span>" : "";
        bar.appendChild(seg);
      });
      es.appendChild(bar);
      var key = el("div", "stack-key");
      key.innerHTML = ev3.map(function(x){
        return "<span><i class='" + x[2] + "'></i>" + esc(x[1]) + "</span>"; }).join("");
      es.appendChild(key);
    }
    var eg = el("div", "buckets");
    [["consumed", "Named in the paper's code", "",
      "these prefixes appear in the scripts that build the tables and figures"],
     ["named_not_landed", "Missing", "focus",
      "the paper's code asks for these files and they are not there"],
     ["landed_not_named", "Status unknown", "caution",
      limits.length
        ? "the files are here; nobody can tell yet whether the paper uses them"
        : "the files are here and no script mentions them"]
    ].forEach(function(sp){
      var list = d.evidence[sp[0]] || [];
      var b = el("div", "bucket" + (sp[2] ? " " + sp[2] : ""));
      b.innerHTML = '<div class="bucket-top"><span class="bucket-name">' + sp[1] +
        '</span><span class="bucket-n">' + list.length + '</span></div>' +
        '<div class="bucket-list">' + (list.length ? esc(list.join(" ")) : "—") + '</div>' +
        '<div class="bucket-why">' + esc(sp[3]) + '</div>';
      eg.appendChild(b);
    });
    es.appendChild(eg);
    app.appendChild(es);
  }

  /* awaiting classification */
  if (d.buckets && d.have_results !== false){
    var bs = el("section");
    var bh = el("div", "sec-head");
    bh.appendChild(el("h2", null, "Scripted but empty · " + d.counts.unlanded + " prefixes"));
    bh.appendChild(el("div", "tally",
      "<span>three of these four are normal; the last one is the one to read</span>"));
    bs.appendChild(bh);
    var bg = el("div", "buckets");
    [["elsewhere", "Filed under another name", "",
      "the script says so itself — the results exist, under a different prefix. Fine."],
     ["training", "Training runs", "quiet",
      "model weights on the cluster, not result files here"],
     ["running", "In the queue", "quiet",
      "a job is running; results have not landed yet"],
     ["no_trace", "Nothing came of these", "focus",
      "a script exists and nothing came out of it, anywhere"]
    ].forEach(function(sp){
      var rows = d.buckets[sp[0]] || [];
      var b = el("div", "bucket" + (sp[2] ? " " + sp[2] : ""));
      var names = rows.map(function(r){
        return sp[2] === "quiet" ? r.prefix
             : (r.to && r.to.length ? r.prefix + " → " + r.to.join(",") : r.prefix); });
      b.innerHTML = '<div class="bucket-top"><span class="bucket-name">' + sp[1] +
        '</span><span class="bucket-n">' + rows.length + '</span></div>' +
        '<div class="bucket-list">' + (names.length ? esc(names.join("  ")) : "—") + '</div>' +
        '<div class="bucket-why">' + esc(sp[3]) + '</div>';
      bg.appendChild(b);
    });
    bs.appendChild(bg);
    app.appendChild(bs);
  }

  /* closed */
  var cs = el("section");
  cs.appendChild(el("div", "sec-head", "<h2>Closed nodes</h2>"));
  [["cashed", "Cashed in", "became a claim"],
   ["falsified", "Falsified", "walked to a negative conclusion — this is paper material"]
  ].forEach(function(spec){
    var rows = d.nodes.filter(function(n){ return n.lifecycle === spec[0]; });
    if (!rows.length) return;
    var det = el("details", "fold");
    det.innerHTML = "<summary><b>" + spec[1] + "</b> <span class='n'>" +
      rows.length + " nodes — " + spec[2] + "</span></summary>";
    var fb = el("div", "fold-body");
    rows.forEach(function(n){
      var r = el("div", "frow");
      r.innerHTML = "<span>" + esc(n.name) +
        (n.patterns.length ? " <span class='mono' style='color:var(--ink-3)'>" +
          esc(n.patterns.join(" ")) + "</span>" : "") + "</span>" +
        (n.claim ? "<span class='to'>→ " + esc(n.claim) + "</span>"
                 : "<span class='vd'>" + esc(n.verdict || "") + "</span>");
      fb.appendChild(r);
    });
    det.appendChild(fb);
    cs.appendChild(det);
  });
  app.appendChild(cs);

  /* every change since collection began, folded away */
  var evAll = (d.events || []).slice();
  if (evAll.length){
    var kinds = {};
    evAll.forEach(function(e){ kinds[e.kind] = (kinds[e.kind] || 0) + 1; });
    var ord = ["stalled", "born", "closed", "gone"];
    var lbl = ord.filter(function(k){ return kinds[k]; })
                 .map(function(k){ return kinds[k] + " " + k; }).join(" · ");
    var det = el("details", "fold");
    det.innerHTML = "<summary><b>Transitions</b> <span class='n'>" + esc(lbl) +
      " — a leaf going stalled is the moment drift is born</span></summary>";
    var fb2 = el("div", "fold-body");
    evAll.slice().reverse().slice(0, 16).forEach(function(e){
      var r = el("div", "frow");
      var mk = {stalled: "\u25D0", born: "+", closed: "\u2713", gone: "\u00D7"}[e.kind] || "\u00B7";
      r.innerHTML = "<span><span class='mono' style='color:" +
        (e.kind === "stalled" ? "var(--stall)" : "var(--ink-3)") + "'>" + mk + " " +
        esc(e.kind) + "</span>  " + esc(e.name) + "</span>" +
        "<span class='vd'>" + esc(stamp(e.at)) + "</span>";
      fb2.appendChild(r);
    });
    det.appendChild(fb2);
    app.appendChild(det);
  }

  /* why some files read "status unknown" - tooling detail, folded */
  if (limits.length){
    var lf = el("details", "fold");
    lf.innerHTML = "<summary><b>Why some files show \u201Cstatus unknown\u201D</b> " +
      "<span class='n'>" + limits.length + " scripts pass the name in a variable, so the " +
      "reader cannot follow it</span></summary>";
    var lb = el("div", "fold-body");
    lb.appendChild(el("div", null, "<span style='color:var(--ink-2)'>The reader follows a " +
      "literal prefix but not one that arrives as an argument. Making any of these literal, " +
      "or naming the prefix in a comment beside it, would resolve those artifacts.</span>"));
    limits.forEach(function(m2){
      var r = el("div", "frow");
      r.innerHTML = "<span class='mono' style='font-size:12px'>" + esc(m2.where) + "</span>" +
                    "<span class='vd'>" + esc(m2.why) + "</span>";
      lb.appendChild(r);
    });
    lf.appendChild(lb);
    app.appendChild(lf);
  }

  /* work in progress that has not been committed */
  var uncommitted = (d.coverage || {}).uncommitted || [];
  if (uncommitted.length){
    var uc = el("div", "coverage");
    uc.id = "uncommitted";
    uc.appendChild(el("span", "hd", "Uncommitted work — " + uncommitted.length +
      " place" + (uncommitted.length > 1 ? "s" : "")));
    uc.appendChild(el("span", null,
      "Not committed yet. Anything concluded only here cannot be recovered by re-running."));
    var ul = el("div", null);
    ul.style.cssText = "font-family:var(--mono);font-size:12px;color:var(--ink-3);" +
      "display:flex;flex-direction:column;gap:2px";
    uncommitted.forEach(function(u){
      ul.appendChild(el("div", null, "   " + esc(u.prefix) + "  —  " + esc(u.detail)));
    });
    uc.appendChild(ul);
    app.appendChild(uc);
  }

  /* coverage */
  var cov = el("div", "coverage");
  cov.id = "coverage";
  var g = d.coverage || {jobs: [], artifacts: [], paths: [], days: 7};
  var nGap = g.jobs.length + g.artifacts.length + g.paths.length;
  cov.appendChild(el("span", "hd",
    "Unattributed activity — " + nGap + " over the last " + g.days + " days"));
  cov.appendChild(el("span", null,
    "Work that moved and belongs to none of the " + d.counts.nodes + " lines in " +
    "<span class='mono'>" + esc(d.source) + "</span>. " +
    "<b>If one of these matters, add it to the outline</b> and it shows up above from then on."));
  [["jobs", "a job is running that no line accounts for", true],
   ["artifacts", "new files landed under a prefix no line accounts for", false],
   ["paths", "code changed where no line is watching", false]
  ].forEach(function(sp){
    var rows = g[sp[0]] || [];
    if (!rows.length) return;
    var wrap = el("div", sp[2] ? "incomplete" : null);
    if (!sp[2]) wrap.style.cssText = "font-family:var(--mono);font-size:12px;" +
      "color:var(--ink-3);display:flex;flex-direction:column;gap:2px";
    wrap.appendChild(el("div", null, "<b>" + rows.length + "</b> " + esc(sp[1])));
    rows.forEach(function(r){
      wrap.appendChild(el("div", null, "   " + esc(r.prefix) + "  —  " + esc(r.detail)));
    });
    cov.appendChild(wrap);
  });
  cov.appendChild(el("span", null,
    "<span style='color:var(--ink-3)'>No view here judges whether a scientific conclusion is " +
    "correct. This tracks where work points and where it came from, not whether the thinking " +
    "is sound.</span>"));
  app.appendChild(cov);
}

function load(){
  fetch("lineage.json", {cache: "no-store"})
    .then(function(r){ return r.ok ? r.json() : Promise.reject(r.status); })
    .then(function(j){
      if (!DATA || j.collected_at !== DATA.collected_at){ DATA = j; render(j); }
    })
    .catch(function(){
      if (!DATA){
        var raw = document.getElementById("inline-data").textContent.trim();
        if (raw && raw !== "null"){
          DATA = JSON.parse(raw);
          render(DATA);
          if (poll) { clearInterval(poll); poll = null; }
          var note = el("div", "still",
            "Collected " + esc(String(DATA.collected_at).replace("T", " ").slice(0, 16)) +
            ". <b>This page changes when the agent next runs a collection</b> — after a job is " +
            "submitted or finishes, results sync, or the outline changes. Reload to see the " +
            "latest one.");
          var app = document.getElementById("app");
          app.insertBefore(note, app.children[1] || null);
        }
        else {
          document.getElementById("app").innerHTML =
            "<div class='incomplete'>No <span class='mono'>lineage.json</span> and no inlined " +
            "snapshot. Run <span class='mono'>python lineage.py --watch</span>.</div>";
        }
      }
    });
}

load();
poll = setInterval(load, 10000);
</script>
"""


if __name__ == "__main__":
    sys.exit(main())
