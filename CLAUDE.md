# Claude Scholar Configuration

## Project Overview

**Claude Scholar** — A personal Claude Code configuration system for academic research and software development.

**Mission**: Cover the complete academic research lifecycle (from ideation to publication) and software development workflows, while providing plugin development and project management capabilities.

---

## User Background

### Academic Profile
- **Degree**: Computer Science PhD
- **Target Venues**:
  - Top conferences: NeurIPS, ICML, ICLR, KDD
  - High-impact journals: Nature, Science, Cell, PNAS
- **Focus**: Academic writing quality, logical coherence, natural expression

### Technology Stack Preferences

**Python Ecosystem**:
- **Package Manager**: `uv` — modern Python package manager
- **Configuration**: Hydra + OmegaConf (composition, overrides, type safety)
- **Model Training**: Transformers Trainer

**Git Conventions**:
- **Commit Standard**: Conventional Commits
  ```
  Type: feat, fix, docs, style, refactor, perf, test, chore
  Scope: data, model, config, trainer, utils, workflow
  ```
- **Branch Strategy**: master/develop/feature/bugfix/hotfix/release
- **Merge Strategy**: Rebase to sync feature branches, merge with `--no-ff`

---

## Global Configuration

### Language Settings
- Respond in Chinese (中文)
- Keep technical terms in English (e.g., NeurIPS, RLHF, TDD, Git)
- Do not translate proper nouns or names

### Working Directory Conventions
- Plan documents: `/plan` folder
- Temporary files: `/temp` folder
- Auto-create folders if they do not exist

### Task Execution Principles
- For complex tasks, discuss approach first, then break down and implement
- Run example tests after implementation
- Back up before changes; do not break existing functionality
- Clean up temporary files promptly after completion

### Paper Writing Rules (Global)

Paper writing rules are defined in `policy/rules/` (28 rule cards), covering figure formatting, LaTeX math, experiment structure, citation verification, submission compliance, etc.
Rule specification and full registry: `policy/README.md`. Skills reference rules via `<!-- policy:RULE_ID -->` markers.
**Writing tasks must first read `policy/README.md` + relevant rule cards; `policy/rules/` is the single source of truth.**

### Working Style
- **Task Management**: Use TodoWrite to track progress; plan before executing complex tasks; prefer existing skills
- **Communication**: Ask proactively when uncertain; confirm before critical operations; follow hook-enforced workflows
- **Code Style**: Python follows PEP 8; comments in Chinese; naming in English

---

## Core Workflows

### Research Lifecycle (7 Stages)

```
Ideation → ML Development → Experiment Analysis → Paper Writing → Self-Review → Submission/Rebuttal → Post-Acceptance
```

| Stage | Core Tools | Commands |
|-------|-----------|----------|
| 1. Research Ideation | `research-ideation` skill + `literature-reviewer` agent | `/research-init` |
| 2. ML Project Development | `architecture-design` skill + `code-reviewer` agent | `/plan`, `/commit`, `/tdd` |
| 3. Experiment Analysis | `results-analysis` skill + `data-analyst` agent | `/analyze-results` |
| 4. Paper Writing | `ml-paper-writing` skill + `paper-miner` agent | - |
| 5. Paper Self-Review | `paper-self-review` skill | - |
| 6. Submission & Rebuttal | `review-response` skill + `rebuttal-writer` agent | `/rebuttal` |
| 7. Post-Acceptance | `post-acceptance` skill | `/presentation`, `/poster`, `/promote` |

### Supporting Workflows

- **Automation**: 5 Hooks auto-trigger at various session stages (skill evaluation, environment init, work summary, security checks)
- **Knowledge Extraction**: `paper-miner` and `kaggle-miner` agents continuously extract knowledge from papers and competitions
- **Skill Evolution**: `skill-development` → `skill-quality-reviewer` → `skill-improver` three-step improvement cycle

---

## Skills Catalog (35 skills)

### 🔬 Research & Analysis (5 skills)

- **research-ideation**: Research ideation startup (5W1H, literature review, gap analysis, research question formulation)
- **results-analysis**: Experiment results analysis (statistical tests, visualization, ablation studies)
- **citation-verification**: Citation verification (multi-layer: format → API → metadata → content)
- **daily-paper-generator**: Daily paper generator for research tracking
- **paper-figure-generator**: Academic paper figure generation (system overview, pipeline, architecture diagrams, etc., powered by AutoFigure-Edit, produces editable SVG)

### 📝 Paper Writing & Publication (7 skills)

- **ml-paper-writing**: ML/AI paper writing assistant
  - Conferences: NeurIPS, ICML, ICLR, ACL, AAAI, COLM
  - Journals: Nature, Science, Cell, PNAS
- **writing-anti-ai**: Remove AI writing patterns, bilingual support (EN/CN)
- **paper-self-review**: Paper self-review (multi-item quality checklist, including figure and math conformance)
- **review-response**: Systematic rebuttal writing
- **post-acceptance**: Post-acceptance processing (presentations, posters, promotion)
- **doc-coauthoring**: Document co-authoring workflow
- **latex-conference-template-organizer**: LaTeX conference template organization

### 💻 Development Workflow (6 skills)

- **daily-coding**: Daily coding checklist (minimal mode, auto-trigger)
- **git-workflow**: Git workflow standards (Conventional Commits, branch management)
- **code-review-excellence**: Code review best practices
- **bug-detective**: Debugging and error investigation (Python, Bash/Zsh, JavaScript/TypeScript)
- **architecture-design**: ML project code architecture and design patterns
- **verification-loop**: Verification loops and testing

### 🔌 Plugin Development (8 skills)

- **skill-development**: Skill development guide
- **skill-improver**: Skill improvement tool
- **skill-quality-reviewer**: Skill quality review
- **command-development**: Slash command development
- **command-name**: Plugin structure guide
- **agent-identifier**: Agent development configuration
- **hook-development**: Hook development and event handling
- **mcp-integration**: MCP server integration

### 🧪 Tools & Utilities (4 skills)

- **planning-with-files**: Planning and progress tracking with Markdown files
- **uv-package-manager**: uv package manager usage
- **webapp-testing**: Local web application testing
- **kaggle-learner**: Kaggle competition learning

### 🎨 Web Design (3 skills)

- **frontend-design**: Create distinctive, production-grade frontend interfaces, avoiding generic AI aesthetics
- **ui-ux-pro-max**: UI/UX design intelligence (50+ styles, 97 palettes, 57 font pairings, 9 tech stacks)
- **web-design-reviewer**: Website design visual inspection, identifying and fixing responsive, accessibility, and layout issues

### 🔧 System (2 skills)

- **using-claude-scholar**: Meta skill ensuring correct use of the Claude Scholar skill system, enforcing skill evaluation discipline
- **policy-rule-creator**: Policy Engine rule creation wizard (requirements → Rule Card → Registry → Integration Marker → Lint → validation)

---

## Commands (50+)

### Research Workflow Commands

| Command | Function |
|---------|----------|
| `/research-init` | Start research ideation workflow (5W1H, literature review, gap analysis) |
| `/analyze-results` | Analyze experiment results (statistical tests, visualization, ablation studies) |
| `/rebuttal` | Generate systematic rebuttal document |
| `/presentation` | Create conference presentation outline |
| `/poster` | Generate academic poster design plan |
| `/promote` | Generate promotional content (Twitter, LinkedIn, blog) |

### Development Workflow Commands

| Command | Function |
|---------|----------|
| `/plan` | Create implementation plan |
| `/commit` | Commit code (following Conventional Commits) |
| `/update-github` | Commit and push to GitHub |
| `/update-readme` | Update README documentation |
| `/code-review` | Code review |
| `/tdd` | Test-driven development workflow |
| `/build-fix` | Fix build errors |
| `/verify` | Verify changes |
| `/checkpoint` | Create checkpoint |
| `/refactor-clean` | Refactor and clean up |
| `/learn` | Extract reusable patterns from code |
| `/create_project` | Create new project |
| `/setup-pm` | Configure package manager (uv/pnpm) |
| `/update-memory` | Check and update CLAUDE.md memory |

### SuperClaude Command Set (`/sc`)

- `/sc agent` - Agent dispatch
- `/sc analyze` - Code analysis
- `/sc brainstorm` - Interactive brainstorming
- `/sc build` - Build project
- `/sc business-panel` - Business panel
- `/sc cleanup` - Code cleanup
- `/sc design` - System design
- `/sc document` - Generate documentation
- `/sc estimate` - Effort estimation
- `/sc explain` - Code explanation
- `/sc git` - Git operations
- `/sc help` - Help information
- `/sc implement` - Feature implementation
- `/sc improve` - Code improvement
- `/sc index` - Project index
- `/sc index-repo` - Repository index
- `/sc load` - Load context
- `/sc pm` - Package manager operations
- `/sc recommend` - Recommend solutions
- `/sc reflect` - Reflect and summarize
- `/sc research` - Technical research
- `/sc save` - Save context
- `/sc select-tool` - Tool selection
- `/sc spawn` - Spawn subtasks
- `/sc spec-panel` - Specification panel
- `/sc task` - Task management
- `/sc test` - Test execution
- `/sc troubleshoot` - Troubleshooting
- `/sc workflow` - Workflow management

---

## Agents (14)

### Research Workflow Agents

- **literature-reviewer** - Literature search, categorization, and trend analysis
- **data-analyst** - Automated data analysis and visualization
- **rebuttal-writer** - Systematic rebuttal writing with tone optimization
- **paper-miner** - Extract writing knowledge from successful papers

### Development Workflow Agents

- **architect** - System architecture design
- **build-error-resolver** - Build error resolution
- **bug-analyzer** - Deep code execution flow analysis and root cause investigation
- **code-reviewer** - Code review
- **dev-planner** - Development task planning and decomposition
- **refactor-cleaner** - Code refactoring and cleanup
- **tdd-guide** - TDD workflow guidance
- **kaggle-miner** - Kaggle engineering practice extraction

### Design & Content Agents

- **ui-sketcher** - UI blueprint design and interaction specification
- **story-generator** - User story and requirements generation

---

## Hooks (5)

Cross-platform Node.js hooks for automated workflow execution:

| Hook | Trigger | Function |
|------|---------|----------|
| `session-start.js` | Session start | Display Git status, TODOs, available commands |
| `skill-forced-eval.js` | Every user input | Force evaluation of all available skills |
| `session-summary.js` | Session end | Generate work log, detect CLAUDE.md updates |
| `stop-summary.js` | Session stop | Quick status check, temporary file detection |
| `security-guard.js` | File operations | Security validation (secret detection, dangerous command interception) |

---

## Rules

### Development & Operations Rules (4 Rules, `rules/` directory)

Global constraints, always active:

| Rule File | Purpose |
|-----------|---------|
| `coding-style.md` | ML project code standards: 200-400 line files, immutable configs, type hints, Factory & Registry patterns |
| `agents.md` | Agent orchestration: auto-invocation triggers, parallel execution, multi-perspective analysis |
| `security.md` | Security standards: secret management, sensitive file protection, pre-commit security checks |
| `experiment-reproducibility.md` | Experiment reproducibility: random seeds, config recording, environment logging, checkpoint management |

### Paper Writing Rules (28 Rules, `policy/rules/` directory)

Paper writing rules are managed by the Policy Engine, covering core/domain/venue three layers.
Rule specification and registry: `policy/README.md`.

---

## Naming Conventions

### Skill Naming
- Format: kebab-case (lowercase + hyphens)
- Form: Prefer gerund form (verb+ing)
- Examples: `scientific-writing`, `git-workflow`, `bug-detective`

### Tag Naming
- Format: Title Case
- Abbreviations in all caps: TDD, RLHF, NeurIPS, ICLR
- Examples: `[Writing, Research, Academic]`

### Description Conventions
- Person: Third person
- Content: Include purpose and usage scenarios
- Example: "Provides guidance for academic paper writing, covering top conference submission requirements"

---

## Runtime Compatibility

Claude Scholar supports two runtime environments:

### Claude Code (Full Support)

- **Installation**: Clone to `~/.claude` or install as a plugin
- **Hooks**: 5 automation hooks (session-start, skill-forced-eval, session-summary, stop-summary, security-guard)
- **Skills**: All 35 skills (including `using-claude-scholar` meta skill)
- **Commands**: 50+ slash commands
- **Agents**: 14 specialized agents

### Codex (Skills + AGENTS.md)

- **Installation**: `scripts/install-codex.sh` creates symlink to `~/.agents/skills/`
- **Hooks**: N/A (Codex does not support hooks)
- **Skills**: 27 universal skills + 6 Claude Code-specific skills (marked `platform: claude-code`, serve as reference documentation)
- **Commands**: N/A (Codex does not support slash commands; use skills and AGENTS.md directives instead)
- **AGENTS.md**: Encodes all hook behaviors as natural language instructions (skill evaluation, security awareness, session behavior)

### Tool Mapping

| Claude Code | Codex | Notes |
|------------|-------|-------|
| `TodoWrite` | `plan` tool | Codex built-in planning tool |
| `Skill` tool | Native skill discovery | Auto-discovered from `~/.agents/skills/` |
| `Task` subagent | `spawn_agent` | Codex natively supports sub-agents |
| `Edit` / `Write` | `apply_patch` | File editing |
| `Grep` / `Glob` | `rg` / `rg --files` | ripgrep |
| `EnterPlanMode` | `plan` tool | Complex task planning |

---

## Task Completion Summary

Proactively provide a brief summary after each task:

```
📋 Operation Review
1. [Main operations]
2. [Modified files]

📊 Current Status
• [Git/filesystem/runtime state]

💡 Next Steps
1. [Actionable suggestions]
```
