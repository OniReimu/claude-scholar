#!/usr/bin/env python3
"""Check that an experiment record's analysis plan is frozen before a real run.

Usage:
    prereg_check.py <path/to/EXP-YYYY-NNN>

Prints one OK line and exits 0 when design.md, protocol.md and analysis-plan.md
exist, analysis-plan.md is marked frozen, its decision rules are filled in, and (in
a git work tree) it is committed with no pending changes. Otherwise prints one
REFUSED line per problem and exits 1. Standard library only.
"""

import re
import subprocess
import sys
from pathlib import Path

REQUIRED = ("design.md", "protocol.md", "analysis-plan.md")

# "Filled" test: the template writes each blank as one of the placeholders below and
# keeps its instructions in HTML comments. After removing the comments, every claim
# block ("### ...") in the Decision rules section must hold its Supported and Refuted
# lines, and no template placeholder may remain. Matching the template's own
# placeholders leaves prose such as "<1 point" or "<https://...>" alone.
BLANKS = ("<the claim", "<metric>", "<higher or lower>", "<baseline>", "<N, default 3>",
          "<δ in", "<evaluation item>", "<count>", "<source of randomness>",
          "<extension rule", "<threshold in")


def rules_unfilled(rules):
    claims = re.split(r"^### ", rules, flags=re.M)[1:]
    return (not claims or any(b in rules for b in BLANKS)
            or any("**Supported:**" not in c or "**Refuted:**" not in c for c in claims))


def plan_problems(text):
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    out = []
    m = re.search(r"^\*\*Status:\*\*\s*(\S+)", text, flags=re.M)
    status = m.group(1) if m else "missing"
    if status.lower() != "frozen":
        out.append(f"analysis-plan.md Status is '{status}', not frozen")
    m = re.search(r"^## Decision rules[ \t]*$(.*?)(?=^## |\Z)", text, flags=re.M | re.S)
    rules = m.group(1) if m else ""
    if rules_unfilled(rules):
        out.append("analysis-plan.md Decision rules are unfilled "
                   "(a claim lacks its verdict lines, or template blanks are left)")
    return out


def git(cwd, *args):
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    except OSError:  # git not installed
        return None


def main(argv):
    if len(argv) != 2:
        sys.exit(__doc__)
    record = Path(argv[1]).resolve()
    missing = [f for f in REQUIRED if not (record / f).is_file()]
    if missing:
        for f in missing:
            print(f"REFUSED: {f} not found in {record}")
        return 1
    try:
        problems = plan_problems((record / "analysis-plan.md").read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as e:
        print(f"REFUSED: cannot read analysis-plan.md ({e})")
        return 1

    inside = git(record, "rev-parse", "--is-inside-work-tree")
    in_git = inside is not None and inside.returncode == 0
    commit = ""
    if in_git:
        if git(record, "ls-files", "--error-unmatch", "analysis-plan.md").returncode != 0:
            problems.append("analysis-plan.md is not tracked by git")
        elif git(record, "diff", "--quiet", "HEAD", "--", "analysis-plan.md").returncode != 0:
            problems.append("analysis-plan.md differs from HEAD (commit it; "
                            "record later changes in deviations.md)")
        else:
            commit = git(record, "log", "-1", "--format=%h", "--", "analysis-plan.md").stdout.strip()

    if problems:
        for p in problems:
            print(f"REFUSED: {p}")
        return 1
    where = f"committed at {commit}" if in_git else "not in a git work tree, commit check skipped"
    print(f"OK {record.name}: analysis plan frozen ({where})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
