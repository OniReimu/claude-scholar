# Tool Mapping: Claude Code → Codex

> **Runtime guard**: This mapping applies **only on Codex**. On Claude Code, use the native tools directly — do NOT substitute Codex equivalents.

When running on Codex and skills reference Claude Code-specific tools, use these equivalents:

| Claude Code Tool | Codex Equivalent | Notes |
|-----------------|------------------|-------|
| `TodoWrite` | `plan` tool | Use Codex built-in planning tool |
| `Skill` tool | Native skill discovery | Skills are auto-loaded from `~/.agents/skills/` |
| `Task` subagent | `spawn_agent` | Codex natively supports sub-agents for parallel execution |
| `Edit` / `Write` | `apply_patch` | Single file editing |
| `Grep` / `Glob` | `rg` / `rg --files` | Use ripgrep for search |
| `Read` | `cat` / `rg` | Use command-line tools to read files |
| `EnterPlanMode` | `plan` tool | Use for complex task planning |
