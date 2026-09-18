#!/usr/bin/env python3
"""Check a repository's projects against the research-workspace contract.

A project is a directory holding PROJECT-WORKSPACE.md. In a monorepo that is each
subproject; in a single-project repository it is the root itself. Directories without
the file are reported as unmanaged and otherwise left alone — the contract forbids
inferring managed status from a directory's name.

Contract source: DELONG-L/Research-Workflow-Skills,
research-workspace-governance/references/project-workspace-contract.md

Usage:
    uv run --with pyyaml validate_workspace.py                 # every project
    uv run --with pyyaml validate_workspace.py <project> ...   # named projects only
    uv run --with pyyaml validate_workspace.py --root <path>   # repository to check

Exit status is non-zero when anything fails.
"""

import subprocess
import sys
from pathlib import Path

import yaml

def _default_root():
    """The enclosing Git working tree, falling back to the current directory."""
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                         capture_output=True, text=True)
    return Path(out.stdout.strip()) if out.returncode == 0 else Path.cwd()


ROOT = _default_root()

# The contract's standard topology is treated here as a CLOSED set: a managed
# project may hold these directories and no others. This is stricter than the
# upstream contract, which names the topology without forbidding additions. The
# stricter reading is deliberate — every directory outside the set is a place
# where a second copy of something can accumulate unnoticed.
CANON_DIRS = {"experiments", "src", "configs", "scripts", "data", "artifacts", "paper"}
CANON_FILES = {"PROJECT-WORKSPACE.md", "task_plan.md", "notes.md",
               ".research-workspace.yml", "pyproject.toml", "uv.lock",
               "README.md", "CLAUDE.md", "AGENTS.md", ".gitignore"}


def front_matter(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("missing opening YAML front matter")
    try:
        end = next(i for i, ln in enumerate(lines[1:], 1) if ln.strip() == "---")
    except StopIteration:
        raise ValueError("missing closing YAML front matter")
    data = yaml.safe_load("\n".join(lines[1:end]))
    if not isinstance(data, dict):
        raise ValueError("front matter must be a mapping")
    return data


def check_contract(data):
    """Port of the upstream validate_workspace_contract. Returns list of errors."""
    errs = []
    for key in ("workspace_kind", "contract_version", "project_id", "local"):
        if key not in data:
            errs.append(f"missing {key}")
    if data.get("workspace_kind") != "research":
        errs.append("workspace_kind must be research")
    if not isinstance(data.get("project_id"), str) or not data.get("project_id", "").strip():
        errs.append("project_id must be a non-empty string")

    local = data.get("local")
    if not isinstance(local, dict):
        errs.append("local must be a mapping")
    else:
        for key in ("experiment_root", "source_roots"):
            if key not in local:
                errs.append(f"local.{key} missing")
        if not isinstance(local.get("source_roots"), list) or not local.get("source_roots"):
            errs.append("local.source_roots must be a non-empty list")

    paper = data.get("paper")
    if paper is not None:
        if not isinstance(paper, dict):
            errs.append("paper must be null or a mapping")
        else:
            for key in ("repository", "link_file", "figure_manifest"):
                if key not in paper:
                    errs.append(f"paper.{key} missing")

    hpc, sync = data.get("hpc"), data.get("sync")
    if hpc is None and sync is not None:
        errs.append("sync requires hpc")
    if hpc is not None:
        if not isinstance(hpc, dict):
            errs.append("hpc must be null or a mapping")
        else:
            for key in ("host", "project_root"):
                if key not in hpc:
                    errs.append(f"hpc.{key} missing")
        if not isinstance(sync, dict):
            errs.append("configured hpc requires sync")
        else:
            for key in ("upload_allow", "upload_exclude", "return_allow"):
                if key not in sync:
                    errs.append(f"sync.{key} missing")
                elif not isinstance(sync[key], list):
                    errs.append(f"sync.{key} must be a list")
    return errs


def check_closed_set(project):
    """Report directories and files at the project root outside the canon, and
    non-record directories inside the experiment root."""
    extra_dirs, extra_files = [], []
    for entry in sorted(project.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_dir() and entry.name not in CANON_DIRS:
            extra_dirs.append(entry.name)
        elif entry.is_file() and entry.name not in CANON_FILES:
            extra_files.append(entry.name)

    stray_records = []
    exp = project / "experiments"
    if exp.is_dir():
        stray_records = sorted(e.name for e in exp.iterdir()
                               if e.is_dir() and not e.name.startswith((".", "EXP-")))
    return extra_dirs, extra_files, stray_records


def check_version_control(project, data):
    """source_roots are defined as version-controlled roots. The monorepo's root
    .gitignore ignores */experiments/, */scripts/, */results/, */cache/ and */pbs/,
    so a declared root can be a directory git never sees."""
    errs = []
    for root in (data.get("local") or {}).get("source_roots") or []:
        target = f"{project.name}/{root}/"
        ignored = subprocess.run(
            ["git", "-C", str(ROOT), "check-ignore", "-q", target]
        ).returncode == 0
        if ignored:
            errs.append(f"local.source_roots '{root}' is gitignored, so it is not a "
                        "version-controlled source root")
    return errs


def check_paths(project, data):
    """Audit invariants that need the filesystem. Returns list of errors."""
    errs = []
    local = data.get("local") or {}

    exp_root = local.get("experiment_root")
    # The record root may not exist until the first EXP record is written; that is
    # not a violation. Records living anywhere else is.
    for root in local.get("source_roots") or []:
        if not (project / root).exists():
            errs.append(f"local.source_roots '{root}' does not exist")

    # EXP-* records must live under the declared experiment root
    if isinstance(exp_root, str):
        declared = (project / exp_root).resolve()
        for rec in project.rglob("EXP-*"):
            if not rec.is_dir() or ".git" in rec.parts:
                continue
            if declared not in rec.resolve().parents:
                errs.append(f"experiment record outside declared root: {rec.relative_to(project)}")

    # Record-level figure manifests: provenance may live in the experiment record
    # when the paper is an ordinary directory rather than its own repository.
    if isinstance(exp_root, str):
        for manifest in (project / exp_root).glob("EXP-*/figure-manifest.yml"):
            errs += [f"{manifest.relative_to(project)}: {e}"
                     for e in check_manifest(project, manifest)]

    paper = data.get("paper")
    if isinstance(paper, dict):
        repo = project / str(paper.get("repository", ""))
        if not repo.is_dir():
            errs.append(f"paper.repository '{paper.get('repository')}' does not exist")
            return errs
        if not (repo / ".git").exists():
            errs.append(f"paper.repository '{paper.get('repository')}' is not an independent Git repository")
        link = repo / str(paper.get("link_file", ""))
        if not link.is_file():
            errs.append(f"paper link file missing: {link.relative_to(project)}")
        else:
            link_data = yaml.safe_load(link.read_text(encoding="utf-8")) or {}
            if link_data.get("workspace_kind") != "research-paper":
                errs.append(f"{link.relative_to(project)}: workspace_kind must be research-paper")
            parent = link_data.get("parent_contract", "")
            if not parent:
                errs.append(f"{link.relative_to(project)}: parent_contract missing")
            elif not (link.parent / parent).resolve().is_file():
                errs.append(f"{link.relative_to(project)}: parent_contract '{parent}' resolves to nothing")
        manifest = repo / str(paper.get("figure_manifest", ""))
        if not manifest.is_file():
            errs.append(f"figure manifest missing: {paper.get('figure_manifest')}")
        else:
            errs += check_manifest(repo, manifest)
    return errs


# A manifest may declare code_revision once for the whole release rather than per
# artifact, and an artifact with no executable run carries data_snapshot in place of
# experiment_id. Both are faithful to the contract, so neither is required per entry
# when the alternative is present.
REQUIRED_ARTIFACT_FIELDS = ("artifact_id", "paper_path", "generator", "status")
INHERITABLE_FIELDS = ("code_revision", "project_id")


def _looks_like_path(value):
    """A manifest may describe a manual process instead of naming a file. The
    contract allows that, so only values shaped like paths are checked on disk."""
    if " " in value.strip():
        return False
    return "/" in value or "." in value.rsplit("/", 1)[-1]


def _resolve(base, value):
    """Manifest paths may be semicolon-separated, and may address a member inside an
    archive as `archive.zip::inner/path`. Returns the paths that must exist."""
    if isinstance(value, (list, tuple)):
        value = "; ".join(str(v) for v in value)
    out = []
    for part in (v.strip() for v in str(value).split(";")):
        if not part or not _looks_like_path(part):
            continue
        out.append(base / part.split("::", 1)[0])
    return out


def check_manifest(repo, manifest):
    """Every promoted asset needs source, experiment/data, generator and revision
    links. Values that describe a manual process rather than name a file are
    accepted as-is; only path-shaped values are checked against the filesystem."""
    errs = []
    data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    declared = data.get("project_id")
    if declared and declared != repo.parent.name and declared != repo.parent.parent.name:
        errs.append(f"{manifest.name}: project_id '{declared}' matches neither the "
                    f"project directory nor its parent")

    artifacts = data.get("artifacts") or data.get("figures") or []
    if not isinstance(artifacts, list):
        return [f"{manifest.name}: artifact list must be a sequence"]
    for item in artifacts:
        if not isinstance(item, dict):
            continue
        name = item.get("artifact_id", "<unnamed>")
        for field in REQUIRED_ARTIFACT_FIELDS:
            if not item.get(field):
                errs.append(f"{manifest.name}[{name}]: missing {field}")
        if not item.get("code_revision") and not data.get("code_revision"):
            errs.append(f"{manifest.name}[{name}]: no code_revision, and none declared "
                        "for the manifest as a whole")
        if not item.get("run_id") and not item.get("data_snapshot"):
            errs.append(f"{manifest.name}[{name}]: needs run_id or data_snapshot")
        if not item.get("experiment_id") and not item.get("data_snapshot"):
            errs.append(f"{manifest.name}[{name}]: needs experiment_id, or "
                        "data_snapshot when no experiment produced it")
        for field in ("source_data", "generator"):
            value = item.get(field)
            if not value:
                continue
            for path in _resolve(manifest.parent, value):
                if not path.exists() and not (repo.parent / path.name).exists():
                    rel = str(path).replace(str(manifest.parent) + "/", "")
                    errs.append(f"{manifest.name}[{name}]: {field} '{rel}' does not exist")
    return errs


def paper_only_remote_on_primary():
    """The primary research repo must not carry a paper-only remote such as Overleaf."""
    out = subprocess.run(
        ["git", "-C", str(ROOT), "remote", "-v"], capture_output=True, text=True
    ).stdout
    return ["primary repository carries a paper-only remote: " + ln.split()[1]
            for ln in out.splitlines() if "overleaf.com" in ln]


def submodule_index_modes():
    """A registered submodule committed as ordinary blobs is double-tracked."""
    errs = []
    cfg = subprocess.run(
        ["git", "-C", str(ROOT), "config", "-f", ".gitmodules", "--get-regexp", r"^submodule\..*\.path$"],
        capture_output=True, text=True,
    ).stdout
    for line in cfg.splitlines():
        path = line.split(maxsplit=1)[1]
        entry = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-s", "--", path],
            capture_output=True, text=True,
        ).stdout.splitlines()
        if not entry:
            errs.append(f"submodule '{path}' is registered but absent from the index")
        elif not all(ln.startswith("160000") for ln in entry):
            errs.append(f"submodule '{path}' is double-tracked (committed as ordinary blobs, not a gitlink)")
    return errs


def orphan_links():
    """A .research-workspace.yml whose parent contract does not resolve."""
    errs = []
    for link in ROOT.glob("*/.research-workspace.yml"):
        data = yaml.safe_load(link.read_text(encoding="utf-8")) or {}
        parent = data.get("parent_contract", "")
        if not parent or not (link.parent / parent).resolve().is_file():
            rel = link.relative_to(ROOT)
            errs.append(f"{rel}: parent_contract '{parent}' resolves to nothing")
    return errs


def main():
    global ROOT
    argv = sys.argv[1:]
    if "--root" in argv:
        i = argv.index("--root")
        ROOT = Path(argv[i + 1]).resolve()
        del argv[i:i + 2]
    wanted = set(argv)

    candidates = [p for p in sorted(ROOT.iterdir()) if p.is_dir() and not p.name.startswith(".")]
    if (ROOT / "PROJECT-WORKSPACE.md").is_file():
        candidates.insert(0, ROOT)
    managed, unmanaged, failures = [], [], 0

    for project in candidates:
        if wanted and project.name not in wanted:
            continue
        contract = project / "PROJECT-WORKSPACE.md"
        if not contract.is_file():
            unmanaged.append(project.name)
            continue
        managed.append(project.name)
        try:
            data = front_matter(contract)
        except ValueError as exc:
            print(f"FAIL {project.name}: {exc}")
            failures += 1
            continue
        errs = check_contract(data) + check_paths(project, data) + check_version_control(project, data)
        if data.get("project_id") != project.name:
            errs.append(f"project_id '{data.get('project_id')}' does not match directory name")
        if errs:
            failures += 1
            print(f"FAIL {project.name}")
            for err in errs:
                print(f"     - {err}")
        else:
            print(f"OK   {project.name}")

    if managed:
        print("\nOutside the standard topology:")
        for name in managed:
            dirs, files, stray = check_closed_set(ROOT / name)
            if not (dirs or files or stray):
                print(f"  clean  {name}")
                continue
            failures += 1
            print(f"  {name}")
            if dirs:
                print(f"     {len(dirs):>3} extra dir(s):   " + ", ".join(dirs))
            if files:
                head = ", ".join(files[:6]) + (f" … +{len(files) - 6}" if len(files) > 6 else "")
                print(f"     {len(files):>3} extra file(s):  {head}")
            if stray:
                print(f"     {len(stray):>3} non-record dir(s) under experiments/: " + ", ".join(stray))

    repo_errs = paper_only_remote_on_primary() + submodule_index_modes() + orphan_links()
    if repo_errs:
        print("\nRepository-level audit:")
        for err in repo_errs:
            print(f"     - {err}")
        failures += len(repo_errs)

    print(f"\nmanaged: {len(managed)}   unmanaged: {len(unmanaged)}   failures: {failures}")
    if unmanaged and not wanted:
        print("unmanaged (ordinary mode, no contract): " + ", ".join(unmanaged))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
