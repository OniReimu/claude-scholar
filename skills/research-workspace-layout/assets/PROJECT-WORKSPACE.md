---
workspace_kind: research
contract_version: 1
project_id: "<project-id>"          # must match the project directory name
local:
  experiment_root: "experiments"
  source_roots:                     # declare only roots that exist AND are version-controlled
    - "src"
    - "configs"
    - "scripts"
paper: null                         # see below to enable
hpc: null
sync: null
---

# Project Workspace

## Scope

- Local project root: `.`
- Paper repository: not configured
- Remote execution mirror: not configured

## Local rules

- Program-level planning stays at this root in `task_plan.md` and `notes.md`.
- Experiment records go under `experiments/EXP-YYYY-NNN/`. `experiments/` holds
  records only — implementation code belongs in `src/`, job scripts in `scripts/`.
- `data/` holds manifests and pointers. Raw data lives outside the repository and is
  named in the owning record's `data-manifest.yml`.
- Nothing else at this root. A directory that does not fit one of the slots above is
  a sign that something needs a decision, not a new slot.

## Optional extensions

Replace `paper: null` only when the manuscript is a separate Git repository
(Overleaf, say):

```yaml
paper:
  repository: "paper"
  link_file: ".research-workspace.yml"
  figure_manifest: "figure-manifest.yml"
```

Replace `hpc: null` and `sync: null` only when a remote execution mirror is approved:

```yaml
hpc:
  host: "<ssh-alias>"
  project_root: "<remote-project-root>"
sync:
  upload_allow: ["src/", "configs/", "scripts/"]
  upload_exclude: [".git/", "paper/", "runs/", "results/", "checkpoints/"]
  return_allow: ["experiments/EXP-*/results/", "reports/"]
```

## Change notes

- Record intentional changes to repository boundaries, remote roots, sync rules or
  retention policy here.
