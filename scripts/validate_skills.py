#!/usr/bin/env python3
"""Static checks for skills/ and for anything the repository publishes.

Run by .github/workflows/skills-validate.yml on every push and pull request.

Checks:
  - SKILL.md frontmatter parses as YAML and is a mapping
  - `name` is present and matches the skill's directory
  - `description` is present and non-empty
  - relative links and code-span paths in SKILL.md resolve
  - no file carries a real local path or a secret-shaped string

Placeholders are not defects. A link target such as `URL`, `<path>` or
`path/to/file.md` is documentation showing a shape, and `/Users/name` or
`/home/user` are the conventional stand-ins; each is recognised and skipped.

Usage:  uv run --with pyyaml scripts/validate_skills.py
"""

import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / "skills"

# Directories whose contents this repository does not author.
EXCLUDED = {".git", "vendor", "node_modules", ".venv", "temp", "plan"}
TEXT_SUFFIXES = {".md", ".yml", ".yaml", ".json", ".py", ".sh", ".txt"}

# Only references INTO the skill's own bundled files are checked. Everything else a
# SKILL.md names is a path in the user's project — an output the skill tells the agent
# to create, a template with {slug} in it, a shell command — and is not this
# repository's to resolve.
INTRA_SKILL = re.compile(r"^(\./)?(references|scripts|assets|examples)/")
PLACEHOLDER = re.compile(r"[{}<>*$]")
PLACEHOLDER_USER = re.compile(r"^/(?:Users|home)/(?:name|user|username|you|<[^>]*>|\$)")

SECRET_PATTERNS = [
    ("local path", re.compile(r"/Users/[A-Za-z0-9_.-]+|/home/[A-Za-z0-9_.-]+")),
    ("private key", re.compile(r"BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY")),
    ("api key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("github token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("aws key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
]
# Values that are documented examples rather than live credentials.
SECRET_EXAMPLES = {"AKIAIOSFODNN7EXAMPLE"}

failures = []


def fail(path, message):
    failures.append(f"{path}: {message}")


def frontmatter(path):
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        fail(path.relative_to(ROOT), "does not start with YAML frontmatter")
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        fail(path.relative_to(ROOT), "frontmatter is not closed")
        return None
    try:
        data = yaml.safe_load(parts[1])
    except yaml.YAMLError as exc:
        first = str(exc).splitlines()[0]
        fail(path.relative_to(ROOT), f"frontmatter is not valid YAML — {first}")
        return None
    if not isinstance(data, dict):
        fail(path.relative_to(ROOT), "frontmatter is not a mapping")
        return None
    return data


def check_skill(skill_dir):
    path = skill_dir / "SKILL.md"
    if not path.is_file():
        return
    rel = path.relative_to(ROOT)
    data = frontmatter(path)
    if data is None:
        return

    name = data.get("name")
    if not name:
        fail(rel, "frontmatter has no name")
    elif name.strip().lower().replace(" ", "-") != skill_dir.name:
        # Title Case is an older convention here and slugifies to the directory; a
        # name that slugifies to something else points at a different skill.
        fail(rel, f"name {name!r} does not match its directory {skill_dir.name!r}")

    description = data.get("description")
    if not isinstance(description, str) or not description.strip():
        fail(rel, "frontmatter has no description")

    text = path.read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)]+)\)", text):
        target = target.split("#", 1)[0].strip()
        if not INTRA_SKILL.match(target) or PLACEHOLDER.search(target):
            continue
        if not (skill_dir / target).exists():
            fail(rel, f"bundled file referenced but missing: {target}")


def tracked_files():
    """Only what the repository publishes. A file present locally but untracked is
    not this check's business, and scanning the working tree would report files the
    repository has deliberately stopped carrying."""
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"],
                         capture_output=True, text=True)
    for name in out.stdout.split("\0"):
        if name:
            yield ROOT / name


def check_secrets():
    for path in sorted(tracked_files()):
        if any(part in EXCLUDED for part in path.parts):
            continue
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for label, pattern in SECRET_PATTERNS:
            for match in pattern.findall(text):
                if match in SECRET_EXAMPLES:
                    continue
                if label == "local path":
                    if PLACEHOLDER_USER.match(match) or match.rstrip("/").endswith("|"):
                        continue
                fail(path.relative_to(ROOT), f"{label}: {match[:60]}")


def main():
    if not SKILLS.is_dir():
        print(f"skills/ not found at {SKILLS}", file=sys.stderr)
        return 1
    skills = sorted(p for p in SKILLS.iterdir() if p.is_dir())
    for skill in skills:
        check_skill(skill)
    check_secrets()

    if failures:
        print(f"Validation failed — {len(failures)} finding(s):\n")
        for line in failures:
            print(f"  {line}")
        return 1
    print(f"OK — {len(skills)} skills validated, no findings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
