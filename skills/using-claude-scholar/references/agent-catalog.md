# Agent Catalog (14 Agents)

## Research Agents

| Agent | Purpose |
|-------|---------|
| **literature-reviewer** | Literature search, classification, and trend analysis |
| **data-analyst** | Automated data analysis and visualization |
| **rebuttal-writer** | Systematic rebuttal writing with tone optimization |
| **paper-miner** | Extract writing knowledge from successful papers |
| **kaggle-miner** | Extract engineering practices from Kaggle solutions |

## Development Agents

| Agent | Purpose |
|-------|---------|
| **architect** | System architecture design |
| **build-error-resolver** | Fix build errors: analyze and resolve |
| **code-reviewer** | Review code quality, security, best practices |
| **refactor-cleaner** | Remove dead code: detect and cleanup |
| **tdd-guide** | Guide TDD: test-first development |
| **bug-analyzer** | Deep code execution flow analysis and root cause investigation |
| **dev-planner** | Implementation planning and task breakdown |

## Design & Content Agents

| Agent | Purpose |
|-------|---------|
| **ui-sketcher** | UI blueprint design and interaction specs |
| **story-generator** | User story and requirement generation |

## Auto-Invocation Rules

| Condition | Agent |
|-----------|-------|
| Code just written | `code-reviewer` |
| Build failure | `build-error-resolver` |
| Complex feature | `dev-planner` then `architect` |
| Bug report | `bug-analyzer` |
| New feature with tests | `tdd-guide` |

Use `spawn_agent` (Codex) or `Task` subagent (Claude Code) for parallel execution of independent agent tasks.
