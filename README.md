# Claude Scholar

<div align="center">
  <img src="LOGO.jpeg" alt="Claude Scholar Logo" width="100%"/>
</div>

**Language**: [English](README.md) | [中文](README.zh-CN.md)

Personal Claude Code configuration repository, optimized for academic research and software development - a complete working environment.

## News

- **2026-03-02 (v1.4.1)**: Added Workflow Orchestrator — stateful, resumable research run coordination layer. 10-stage pipeline with persistent run state (`.claude/orchestrator/`), artifact fingerprinting (SHA256), auto-stale detection at session start, rollback with downstream cascade, stage gates (human approval + policy lint). Zero new commands — activates transparently via existing skills/agents/hooks.
- **2026-02-21**: Added first SoK policy pack: 4 semantic `SOK.*` rule cards, `security-sok-sp` profile, and entry-skill marker wiring. SoK remains profile-activated scope in v1 (no schema migration yet).
- **2026-02-19 (v1.3.0)**: Introduced the paper policy engine (`policy/`): rule-card based design in `policy/rules/` (single source of truth), layered scope (`core/domain/venue`), profile overlays in `policy/profiles/`, and executable validation/lint workflows via `policy/validate.sh` and `policy/lint.sh`. Synced Figure workflow policy (Figure 1 required; non-experimental figures default to AutoFigure-Edit).
- **2026-02-16 (v1.2.1)**: Added a global figure rule: no in-image titles for any generated visuals (AutoFigure-Edit conceptual diagrams, legacy image APIs, or Python experimental plots). Use captions in paper text/LaTeX instead.
- **2026-02-16**: Enforced `paper-figure-generator` execution priority: default `AutoFigure-Edit + OpenRouter` first, fallback to legacy Gemini/OpenAI flow only after failure; added troubleshooting note for outdated plugin cache prompts (`GOOGLE_API_KEY` / `OPENAI_API_KEY`).
- **2026-02-15**: Migrated `paper-figure-generator` to AutoFigure-Edit — generates editable SVG vector figures from method text descriptions; replaces Gemini/OpenAI raster generation; supports style transfer via reference images; uses OpenRouter + Roboflow (free SAM3 API)
- **2026-02-13**: Added `paper-figure-generator` skill; packaged project as Claude Code plugin (`.claude-plugin/plugin.json`); added `.env.example`; deep workflow integration across ml-paper-writing, results-analysis, post-acceptance, and using-claude-scholar; 34 skills total
- **2026-02-11**: Major update — added 10 new skills (research-ideation, results-analysis, citation-verification, review-response, paper-self-review, post-acceptance, daily-coding, frontend-design, ui-ux-pro-max, web-design-reviewer), 7 new agents, 8 research workflow commands, 2 new rules (security, experiment-reproducibility); restructured CLAUDE.md; 89 files changed
- **2026-01-26**: Rewrote all Hooks to cross-platform Node.js; completely rewrote README; expanded ML paper writing knowledge base; merged PR #1 (cross-platform support)

## Introduction

Claude Scholar is a personal configuration system for Claude Code CLI, providing rich skills, commands, agents, and hooks optimized for:
- **Academic Research** - Complete research lifecycle: idea generation → experimentation → results analysis → paper writing → review response → conference preparation
- **Software Development** - Git workflows, code review, test-driven development, ML project architecture
- **Plugin Development** - Skill, Command, Agent, Hook development guides with quality assessment
- **Project Management** - Planning documents, code standards, automated workflows with cross-platform hooks

## Quick Navigation

| Topic | Description |
|-------|-------------|
| 🚀 [Quick Start](#quick-start) | Get up and running in minutes |
| 📚 [Core Workflows](#core-workflows) | Paper writing, code organization, skill evolution |
| 🛠️ [What's Included](#whats-included) | Skills, commands, agents overview |
| 📖 [Installation Guide](#installation-options) | Full, minimal, or selective setup |
| 🔧 [Project Rules](#project-rules) | Coding rules + paper policy engine |

## Core Workflows

### Primary Workflows

Complete academic research lifecycle - 7 stages from idea to publication.

#### 1. Research Ideation

Systematic research startup with idea generation and literature review:

**Tools**: `research-ideation` skill + `literature-reviewer` agent

**Process**:
- **5W1H Brainstorming**: What, Why, Who, When, Where, How → structured thinking framework
- **Literature Review**: arXiv + Semantic Scholar integration → automated paper search and classification
- **Gap Analysis**: 5 types (Literature, Methodological, Application, Interdisciplinary, Temporal) → identify research opportunities
- **Research Question**: SMART principles → formulate specific, measurable questions

**Command**: `/research-init "topic"` → launches complete research startup workflow

#### 2. ML Project Development

Maintainable ML project structure for experiment code:

**Tools**: `architecture-design` skill + `code-reviewer` agent + `git-workflow` skill

**Process**:
- **Structure**: Factory & Registry patterns → config-driven models (only `cfg` parameter) → enforced by `rules/coding-style.md`
- **Code Style**: 200-400 line files → type hints required → `@dataclass(frozen=True)` for configs → max 3-level nesting
- **Debug** (`bug-detective`): Error pattern matching for Python/Bash/JS → stack trace analysis → anti-pattern identification
- **Git**: Conventional Commits (`feat/scope: message`) → branch strategy (master/develop/feature) → merge with `--no-ff`

**Commands**: `/plan`, `/commit`, `/code-review`, `/tdd`

#### 3. Experiment Analysis

Statistical analysis and visualization of experimental results:

**Tools**: `results-analysis` skill + `data-analyst` agent

**Process**:
- **Data Processing**: Automated cleaning and preprocessing of experiment logs
- **Statistical Testing**: t-test, ANOVA, Wilcoxon signed-rank → validate significance
- **Visualization**: matplotlib/seaborn integration → publication-ready figures (line plots, bar charts, heatmaps)
- **Ablation Studies**: Systematic component analysis → understand contribution of each part

**Command**: `/analyze-results <experiment_dir>` → generates analysis report with figures and statistics

#### 4. Paper Writing

Systematic paper writing from template to final draft:

**Tools**: `ml-paper-writing` skill + `paper-miner` agent + `latex-conference-template-organizer` skill

**Process**:
- **Template Preparation**: Download conference .zip → extract main files → remove sample content → clean Overleaf-ready structure
- **Citation Verification** (`citation-verification`): Multi-layer validation (Format → API → Information → Content) → prevents hallucinations
- **Systematic Writing**: Narrative framing → 5-sentence abstract formula → section-by-section drafting with feedback cycles
- **Anti-AI Processing** (`writing-anti-ai`): Remove inflated symbolism, promotional language, vague attributions → add human voice and rhythm → bilingual support (EN/CN)

**Venues**: NeurIPS, ICML, ICLR, ACL, AAAI, COLM, Nature, Science, Cell, PNAS

#### 5. Paper Self-Review

Quality assurance before submission:

**Tools**: `paper-self-review` skill

**Process**:
- **Structure Check**: Logical flow, section balance, narrative coherence
- **Logic Validation**: Argument soundness, claim-evidence alignment, assumption clarity
- **Citation Audit**: Reference accuracy, proper attribution, citation completeness
- **Figure Quality**: Visual clarity, caption completeness, color accessibility
- **Writing Polish**: Grammar, clarity, conciseness, academic tone
- **Compliance**: Page limits, formatting requirements, ethical disclosures

**Multi-item checklist** → systematic quality assessment (including figure/title and LaTeX math conformance)

#### 6. Submission & Rebuttal

Paper submission and review response:

**Tools**: `review-response` skill + `rebuttal-writer` agent

**Submission Process**:
- **Pre-submission**: Conference-specific checklists (NeurIPS 16-item, ICML Broader Impact, ICLR LLM disclosure)
- **Format Check**: Page limits, anonymization, supplementary materials
- **Final Review**: Proofread, check references, verify figures

**Rebuttal Process**:
- **Review Analysis**: Parse and classify comments (Major/Minor/Typo/Misunderstanding)
- **Response Strategy**: Accept/Defend/Clarify/Experiment → tailored approach per comment type
- **Rebuttal Writing**: Structured response with evidence and reasoning
- **Tone Management**: Professional, respectful, evidence-based language

**Command**: `/rebuttal <review_file>` → generates complete rebuttal document with experiment plan

#### 7. Post-Acceptance Processing

Conference preparation and research promotion:

**Tools**: `post-acceptance` skill

**Process**:
- **Presentation**: Slide creation guidance (15/20/30 min formats) → visual design principles → storytelling structure
- **Poster**: Academic poster templates (A0/A1 sizes) → layout optimization → visual hierarchy
- **Promotion**: Social media content (Twitter/X, LinkedIn) → blog posts → press releases → research summaries

**Commands**: `/presentation`, `/poster`, `/promote` → automated content generation

**Coverage**: 90% of academic research lifecycle (from idea to publication)

### Workflow Orchestrator

Claude Scholar includes a stateful **Workflow Orchestrator** that tracks progress across the research lifecycle as a single, resumable run. No new commands are needed -- the orchestrator activates transparently when relevant skills and agents are invoked.

**Key features:**
- **Single mode, resumable runs**: State persists in `.claude/orchestrator/` across sessions. Resume from where you left off.
- **10-stage pipeline**: intake -> literature -> proposal -> development -> experiments -> analysis -> writeup -> self_review -> rebuttal -> post_acceptance
- **Stage gates**: Human approval and policy lint checks at stage boundaries prevent premature progression.
- **Artifact fingerprinting**: SHA256 hashes detect file changes and mark affected stages as `stale`.
- **Experiments boundary**: The `experiments` stage enters `blocked` until the user provides a `data_path` with actual results. Rollback is always possible ("roll back to stage X").

**How it works:**
- Session start hook displays active run ID, current stage, and next action.
- Skills and agents automatically read/write run state per the [Run Card contract](orchestrator/run-card.md).
- Stage registry defined in `orchestrator/stages.json`; runtime library at `scripts/lib/orchestrator.js`.

See [docs/orchestrator.md](docs/orchestrator.md) for full documentation.

### Supporting Workflows

These workflows run in the background to enhance the primary workflows.

#### Automated Enforcement Workflow

Cross-platform hooks (Node.js) automate workflow enforcement:

```
Session Start → Skill Evaluation → Session End → Session Stop
```

- **skill-forced-eval** (`skill-forced-eval.js`): Before EVERY user prompt → dynamically scans all available skills (local + plugins) → forces evaluation of each skill → requires activation before implementation → ensures no relevant skill is missed
- **session-start** (`session-start.js`): Session begins → displays Git status, pending todos, available commands, package manager → shows project context at a glance
- **session-summary** (`session-summary.js`): Session ends → generates comprehensive work log → summarizes all changes made → provides smart recommendations for next steps
- **stop-summary** (`stop-summary.js`): Session stops → quick status check → detects temporary files → shows actionable cleanup suggestions

**Cross-platform**: All hooks use Node.js (not shell scripts) ensuring Windows/macOS/Linux compatibility.

#### Knowledge Extraction Workflow

Two specialized mining agents continuously extract knowledge to improve skills:

- **paper-miner** (agent): Analyze research papers (PDF/DOCX/arXiv links) → extracts writing patterns, structure insights, venue requirements, rebuttal strategies → updates `ml-paper-writing/references/knowledge/` with categorized entries (structure.md, writing-techniques.md, submission-guides.md, review-response.md)
- **kaggle-miner** (agent): Study winning Kaggle competition solutions → extract competition briefs, front-runner detailed technical analysis, code templates, best practices → update the `kaggle-learner` skill's knowledge base (`references/knowledge/[domain]/` directories, categorized by NLP/CV/Time Series/Tabular/Multimodal)

**Knowledge feedback loop**: Each paper or solution analyzed enriches the knowledge base, creating a self-improving system that evolves with your research.

#### Skill Evolution System

3-step continuous improvement cycle for maintaining and improving skills:

```
skill-development → skill-quality-reviewer → skill-improver
```

1. **Develop** (`skill-development`): Create skills with proper YAML frontmatter → clear descriptions with trigger phrases → progressive disclosure (lean SKILL.md, details in `references/`)
2. **Review** (`skill-quality-reviewer`): 4-dimension quality assessment → Description Quality (25%), Content Organization (30%), Writing Style (20%), Structural Integrity (25%) → generates improvement plan with prioritized fixes
3. **Improve** (`skill-improver`): Merges suggested changes → updates documentation → iterates on feedback → reads improvement plans and applies changes automatically

## File Structure

```
claude-scholar/
├── AGENTS.md            # Codex behavioral reference (kept in repo; no longer copied)
├── .codex/              # Codex-specific files
│   └── INSTALL.md               # Codex installation guide
│
├── hooks/               # Cross-platform JavaScript hooks (Claude Code only)
│   ├── session-start.js         # Session begin - shows Git status, todos, commands
│   ├── skill-forced-eval.js     # Force skill evaluation before each prompt
│   ├── session-summary.js       # Session end - generates work log with recommendations
│   ├── stop-summary.js          # Session stop - quick status check, temp file detection
│   └── security-guard.js        # Security validation for file operations
│
├── skills/              # 35 specialized skills (domain knowledge + workflows)
│   ├── ml-paper-writing/        # Full paper writing: NeurIPS, ICML, ICLR, ACL, AAAI, COLM
│   │   └── references/
│   │       └── knowledge/        # Extracted patterns from successful papers
│   │       ├── structure.md           # Paper organization patterns
│   │       ├── writing-techniques.md  # Sentence templates, transitions
│   │       ├── submission-guides.md   # Venue requirements (page limits, etc.)
│   │       └── review-response.md     # Rebuttal strategies
│   │
│   ├── research-ideation/        # Research startup: 5W1H, literature review, gap analysis
│   │   └── references/
│   │       ├── 5w1h-framework.md           # Systematic thinking tool
│   │       ├── gap-analysis-guide.md       # 5 types of research gaps
│   │       ├── literature-search-strategies.md
│   │       ├── research-question-formulation.md
│   │       ├── method-selection-guide.md
│   │       └── research-planning.md
│   │
│   ├── results-analysis/         # Experiment analysis: statistics, visualization, ablation
│   │   └── references/
│   │       ├── statistical-methods.md      # t-test, ANOVA, Wilcoxon
│   │       ├── visualization-best-practices.md  # matplotlib/seaborn
│   │       ├── results-writing-guide.md    # Writing results sections
│   │       └── common-pitfalls.md          # Common analysis mistakes
│   │
│   ├── review-response/          # Systematic rebuttal writing
│   │   └── references/
│   │       ├── review-classification.md    # Major/Minor/Typo/Misunderstanding
│   │       ├── response-strategies.md      # Accept/Defend/Clarify/Experiment
│   │       ├── rebuttal-templates.md       # Structured response templates
│   │       └── tone-guidelines.md          # Professional language
│   │
│   ├── paper-self-review/        # multi-item quality checklist
│   ├── post-acceptance/          # Conference preparation
│   │   └── references/
│   │       ├── presentation-templates/     # Slide creation (15/20/30 min)
│   │       ├── poster-templates/           # Academic poster design
│   │       ├── promotion-examples/         # Social media content
│   │       └── design-guidelines.md        # Visual design principles
│   │
│   ├── citation-verification/    # Multi-layer citation validation
│   ├── writing-anti-ai/         # Remove AI patterns: symbolism, promotional language
│   │   └── references/
│   │       ├── patterns-english.md    # English AI patterns to remove
│   │       └── patterns-chinese.md     # Chinese AI patterns to remove
│   │
│   ├── architecture-design/     # ML project patterns: Factory, Registry, Config-driven
│   ├── git-workflow/            # Git discipline: Conventional Commits, branching
│   ├── bug-detective/           # Debugging: Python, Bash, JS/TS error patterns
│   ├── code-review-excellence/  # Code review: security, performance, maintainability
│   ├── skill-development/       # Skill creation: YAML, progressive disclosure
│   ├── skill-quality-reviewer/  # Skill assessment: 4-dimension scoring
│   ├── skill-improver/          # Skill evolution: merge improvements
│   ├── kaggle-learner/          # Learn from Kaggle winning solutions
│   ├── doc-coauthoring/         # Document collaboration workflow
│   ├── latex-conference-template-organizer  # Template cleanup for Overleaf
│   └── ... (10+ more skills)
│
├── commands/            # 50+ slash commands (quick workflow execution)
│   ├── research-init.md         # Launch research startup workflow
│   ├── analyze-results.md       # Analyze experiment results
│   ├── rebuttal.md              # Generate systematic rebuttal document
│   ├── presentation.md          # Create conference presentation outline
│   ├── poster.md                # Generate academic poster design plan
│   ├── promote.md               # Generate promotion content
│   ├── plan.md                  # Implementation planning with agent delegation
│   ├── commit.md                # Conventional Commits: feat/fix/docs/refactor
│   ├── code-review.md           # Quality and security review workflow
│   ├── tdd.md                   # Test-driven development: Red-Green-Refactor
│   ├── build-fix.md             # Fix build errors automatically
│   ├── verify.md                # Run verification loops
│   ├── checkpoint.md            # Save verification state
│   ├── refactor-clean.md        # Remove dead code
│   ├── learn.md                 # Extract patterns from code
│   └── sc/                      # SuperClaude command suite (20+ commands)
│       ├── sc-agent.md           # Agent management
│       ├── sc-estimate.md       # Development time estimation
│       ├── sc-improve.md         # Code improvement
│       └── ...
│
├── agents/              # 14 specialized agents (focused task delegation)
│   ├── literature-reviewer.md   # Literature search and trend analysis
│   ├── data-analyst.md          # Automated data analysis and visualization
│   ├── rebuttal-writer.md       # Systematic rebuttal writing
│   ├── paper-miner.md           # Extract paper knowledge: structure, techniques
│   ├── architect.md             # System design: architecture decisions
│   ├── code-reviewer.md         # Review code: quality, security, best practices
│   ├── tdd-guide.md             # Guide TDD: test-first development
│   ├── kaggle-miner.md          # Extract engineering practices from Kaggle
│   ├── build-error-resolver.md  # Fix build errors: analyze and resolve
│   ├── refactor-cleaner.md      # Remove dead code: detect and cleanup
│   ├── bug-analyzer.md          # Deep code execution flow analysis and root cause investigation
│   ├── dev-planner.md           # Implementation planning and task breakdown
│   ├── ui-sketcher.md           # UI blueprint design and interaction specs
│   └── story-generator.md       # User story and requirement generation
│
├── rules/               # Global guidelines (always-follow constraints)
│   ├── coding-style.md          # ML project standards: file size, immutability, types
│   ├── agents.md                # Agent orchestration: when to delegate, parallel execution
│   ├── security.md              # Secrets management, sensitive file protection
│   └── experiment-reproducibility.md  # Random seeds, config recording, checkpoints
│
├── orchestrator/        # Workflow Orchestrator (stage registry + run card)
│   ├── stages.json              # Stage definitions (10 stages, artifacts, gates)
│   └── run-card.md              # Skills/agents integration contract
│
├── policy/              # Paper policy engine (rule cards + validation + lint)
│   ├── rules/                    # Canonical paper-writing rule cards (single source of truth)
│   ├── profiles/                 # Domain/venue overlays (severity/params tuning)
│   ├── validate.sh               # Rule-card integrity validation
│   ├── lint.sh                   # Machine-enforceable lint checks
│   └── README.md                 # Policy engine design and conventions
│
├── scripts/
│   ├── install-codex.sh         # Codex installer (macOS/Linux, symlink-based)
│   ├── install-codex-windows.ps1 # Codex installer (Windows, junction-based)
│   └── lib/                     # Shared script utilities
│
├── CLAUDE.md            # Global configuration: project overview, preferences, rules
│
└── README.md            # This file - overview, installation, features
```

## Feature Highlights

### Skills (33 total)

**Web Design:**
- `frontend-design` - Create distinctive, production-grade frontend interfaces
- `ui-ux-pro-max` - UI/UX design intelligence (50+ styles, 97 palettes, 9 stacks)
- `web-design-reviewer` - Visual inspection and design issue fixing

**Writing & Academic:**
- `ml-paper-writing` - Full paper writing guidance for top conferences/journals
- `writing-anti-ai` - Remove AI writing patterns (bilingual support)
- `doc-coauthoring` - Structured document collaboration workflow
- `latex-conference-template-organizer` - LaTeX template management
- `daily-paper-generator` - Automated daily paper generation for research tracking

**Research Workflow:**
- `research-ideation` - Research startup: 5W1H brainstorming, literature review, gap analysis
- `results-analysis` - Experiment analysis: statistical testing, visualization, ablation studies
- `review-response` - Systematic rebuttal writing with tone management
- `paper-self-review` - multi-item quality checklist for paper self-assessment (figures + LaTeX math conformance)
- `post-acceptance` - Conference preparation: presentations, posters, promotion
- `citation-verification` - Multi-layer citation validation to prevent hallucinations
- `paper-figure-generator` - Generate editable SVG academic figures (system overviews, pipelines, architectures) via AutoFigure-Edit

**Development:**
- `daily-coding` - Daily coding checklist (minimal, auto-triggered)
- `git-workflow` - Git best practices (Conventional Commits, branching)
- `code-review-excellence` - Code review guidelines
- `bug-detective` - Debugging for Python, Bash, JS/TS
- `architecture-design` - ML project design patterns
- `verification-loop` - Testing and validation

**Plugin Development:**
- `skill-development` - Skill creation guide
- `skill-improver` - Skill improvement tools
- `skill-quality-reviewer` - Quality assessment
- `command-development` - Slash command creation
- `agent-identifier` - Agent configuration
- `hook-development` - Hook development guide
- `mcp-integration` - MCP server integration

**Utilities:**
- `uv-package-manager` - Modern Python package management
- `planning-with-files` - Markdown-based planning
- `webapp-testing` - Local web application testing
- `kaggle-learner` - Learn from Kaggle solutions

### Commands (50+)

**Research Commands:**
| Command | Purpose |
|---------|---------|
| `/research-init` | Launch research startup workflow (5W1H, literature review, gap analysis) |
| `/analyze-results` | Analyze experiment results (statistics, visualization, ablation) |
| `/rebuttal` | Generate systematic rebuttal document from review comments |
| `/presentation` | Create conference presentation outline |
| `/poster` | Generate academic poster design plan |
| `/promote` | Generate promotion content (Twitter, LinkedIn, blog) |

**Development Commands:**
| Command | Purpose |
|---------|---------|
| `/plan` | Create implementation plans |
| `/commit` | Commit with Conventional Commits |
| `/code-review` | Perform code review |
| `/tdd` | Test-driven development workflow |
| `/build-fix` | Fix build errors |
| `/verify` | Verify changes |
| `/checkpoint` | Create checkpoints |
| `/refactor-clean` | Refactor and cleanup |
| `/learn` | Extract reusable patterns |
| `/sc` | SuperClaude command suite (20+ commands) |

### Agents (14 specialized)

**Research Agents:**
- **literature-reviewer** - Literature search, classification, and trend analysis
- **data-analyst** - Automated data analysis and visualization
- **rebuttal-writer** - Systematic rebuttal writing with tone optimization
- **paper-miner** - Extract paper writing knowledge from successful publications

**Development Agents:**
- **architect** - System architecture design
- **build-error-resolver** - Fix build errors
- **code-reviewer** - Review code quality
- **refactor-cleaner** - Remove dead code
- **tdd-guide** - Guide TDD workflow
- **kaggle-miner** - Extract Kaggle engineering practices
- **bug-analyzer** - Deep code execution flow analysis and root cause investigation
- **dev-planner** - Implementation planning and task breakdown

**Design & Content Agents:**
- **ui-sketcher** - UI blueprint design and interaction specs
- **story-generator** - User story and requirement generation

## Quick Start

### Multi-Runtime Support

Claude Scholar supports two runtimes:

| | Claude Code | Codex |
|---|------------|-------|
| **Skills** | 35 (full) | 27 universal + 6 reference |
| **Hooks** | 5 automated | N/A (using-claude-scholar skill replaces) |
| **Commands** | 50+ slash commands | N/A (use skills directly) |
| **Agents** | 14 specialized | 14 (via `spawn_agent`) |
| **Install** | Clone / Plugin | Symlink only (native skill discovery) |

### Installation Options

#### Claude Code Installation

Choose the installation method that fits your needs:

##### Option 1: Plugin Installation (Recommended)

Install via Claude Code plugin manager:

```bash
# Step 1: Add marketplace
claude plugin marketplace add OniReimu/claude-scholar

# Step 2: Install plugin
claude plugin install claude-scholar@claude-scholar
```

**Benefits**: Automatic component discovery, version tracking, easy updates via `claude plugin update`.

**Includes**: All 35 skills, 50+ commands, 14 agents, 5 hooks, and project rules.

##### Option 2: Full Installation (Git Clone)

Complete setup by cloning directly to `~/.claude`:

```bash
# Clone the repository
git clone https://github.com/OniReimu/claude-scholar.git ~/.claude

# Restart Claude Code CLI
```

**Includes**: All 35 skills, 50+ commands, 14 agents, 5 hooks, and project rules.

##### Option 3: Minimal Installation

Core hooks and essential skills only (faster load, less complexity):

```bash
# Clone repository
git clone https://github.com/OniReimu/claude-scholar.git /tmp/claude-scholar

# Copy only hooks and core skills
mkdir -p ~/.claude/hooks ~/.claude/skills
cp /tmp/claude-scholar/hooks/*.js ~/.claude/hooks/
cp -r /tmp/claude-scholar/skills/ml-paper-writing ~/.claude/skills/
cp -r /tmp/claude-scholar/skills/research-ideation ~/.claude/skills/
cp -r /tmp/claude-scholar/skills/results-analysis ~/.claude/skills/
cp -r /tmp/claude-scholar/skills/review-response ~/.claude/skills/
cp -r /tmp/claude-scholar/skills/writing-anti-ai ~/.claude/skills/
cp -r /tmp/claude-scholar/skills/git-workflow ~/.claude/skills/
cp -r /tmp/claude-scholar/skills/bug-detective ~/.claude/skills/

# Cleanup
rm -rf /tmp/claude-scholar
```

**Includes**: 5 hooks, 7 core skills (complete research workflow + essential development).

##### Option 4: Selective Installation

Pick and choose specific components:

```bash
# Clone repository
git clone https://github.com/OniReimu/claude-scholar.git /tmp/claude-scholar
cd /tmp/claude-scholar

# Copy what you need, for example:
# - Hooks only
cp hooks/*.js ~/.claude/hooks/

# - Specific skills
cp -r skills/latex-conference-template-organizer ~/.claude/skills/
cp -r skills/architecture-design ~/.claude/skills/

# - Specific agents
cp agents/paper-miner.md ~/.claude/agents/

# - Project rules
cp rules/coding-style.md ~/.claude/rules/
cp rules/agents.md ~/.claude/rules/
```

**Recommended for**: Advanced users who want custom configurations.

#### Codex Installation

```bash
# Clone the repository
git clone https://github.com/OniReimu/claude-scholar.git ~/claude-scholar

# Run the install script (creates symlinks, migrates legacy AGENTS.md)
chmod +x ~/claude-scholar/scripts/install-codex.sh
~/claude-scholar/scripts/install-codex.sh
```

**Windows (PowerShell):**
```powershell
git clone https://github.com/OniReimu/claude-scholar.git $HOME\claude-scholar
& "$HOME\claude-scholar\scripts\install-codex-windows.ps1"
```

**What it does:**
- Creates symlink: `~/.agents/skills/claude-scholar` → `skills/`
- Detects and migrates legacy `~/.codex/AGENTS.md`
- Updates via `git pull` — no re-install needed

See [.codex/INSTALL.md](.codex/INSTALL.md) for detailed Codex installation guide.

### Requirements

- Claude Code CLI or Codex CLI (v0.91+)
- Git
- (Optional) Node.js (for hooks)
- (Optional) uv, Python (for Python development)

### First Run

After installation, the hooks provide automated workflow assistance:

1. **Every prompt** triggers `skill-forced-eval` → ensures applicable skills are considered
2. **Session starts** with `session-start` → displays project context
3. **Sessions end** with `session-summary` → generates work log with recommendations
4. **Session stops** with `stop-summary` → provides status check

## Project Rules

### Paper Policy Engine

Defined in `policy/`:
- `policy/rules/` is the single source of truth for paper-writing constraints (figures, LaTeX, citations, experiments, submission).
- Rule-card design uses frontmatter metadata (`id`, `layer`, `artifacts`, `phases`, `check_kind`, `enforcement`) plus required sections (`Requirement`, `Rationale`, `Check`, `Examples`).
- Layering model: `core` (always on), `domain` (field-specific), `venue` (conference/journal specific); profile overlays live in `policy/profiles/*.md`.
- SoK in v1 is activated by profile (for example `policy/profiles/security-sok-sp.md`), currently with semantic `SOK.*` rules (`SOK.TAXONOMY_REQUIRED`, `SOK.METHODOLOGY_REPORTING`, `SOK.BIG_TABLE_REQUIRED`, `SOK.RESEARCH_AGENDA_REQUIRED`).
- Current limitation: `policy/lint.sh --profile` loads a single flat profile file (no inheritance/composition yet).
- Validation and enforcement workflow:
  - `bash policy/validate.sh` for structure/integration checks
  - `bash policy/lint.sh` for machine-enforceable checks
- Skills/commands reference rules via `<!-- policy:RULE_ID -->` markers.

### Coding Style

Enforced by `rules/coding-style.md`:
- **File Size**: 200-400 lines maximum
- **Immutability**: Use `@dataclass(frozen=True)` for configs
- **Type Hints**: Required for all functions
- **Patterns**: Factory & Registry for all modules
- **Config-Driven**: Models accept only `cfg` parameter

### Agent Orchestration

Defined in `rules/agents.md`:
- Available agent types and purposes
- Parallel task execution
- Multi-perspective analysis

### Security

Defined in `rules/security.md`:
- Secrets management (environment variables, `.env` files)
- Sensitive file protection (never commit tokens, keys, credentials)
- Pre-commit security checks via hooks

### Experiment Reproducibility

Defined in `rules/experiment-reproducibility.md`:
- Random seed management for reproducibility
- Configuration recording (Hydra auto-save)
- Environment recording and checkpoint management

## Contributing

This is a personal configuration, but you're welcome to:
- Fork and adapt for your own research
- Submit issues for bugs
- Suggest improvements via issues

## License

MIT License

## Acknowledgments

Built with Claude Code CLI and enhanced by the open-source community.

### References

This project is inspired by and builds upon excellent work from the community:

- **[everything-claude-code](https://github.com/anthropics/everything-claude-code)** - Comprehensive resource for Claude Code CLI
- **[AI-research-SKILLs](https://github.com/zechenzhangAGI/AI-research-SKILLs)** - Research-focused skills and configurations

These projects provided valuable insights and foundations for the research-oriented features in Claude Scholar.

---

**For data science, AI research, and academic writing.**

Repository: [https://github.com/OniReimu/claude-scholar](https://github.com/OniReimu/claude-scholar)
