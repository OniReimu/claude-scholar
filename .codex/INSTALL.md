# Claude Scholar — Codex Installation Guide

## Prerequisites

- [Codex CLI](https://github.com/openai/codex) (v0.91+)
- Git
- (Optional) Node.js, uv, Python

## Quick Install

### macOS / Linux

```bash
# 1. Clone the repository
git clone https://github.com/Galaxy-Dawn/claude-scholar.git ~/claude-scholar

# 2. Run the install script
chmod +x ~/claude-scholar/scripts/install-codex.sh
~/claude-scholar/scripts/install-codex.sh
```

### Windows (PowerShell)

```powershell
# 1. Clone the repository
git clone https://github.com/Galaxy-Dawn/claude-scholar.git $HOME\claude-scholar

# 2. Run the install script (as Administrator)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
& "$HOME\claude-scholar\scripts\install-codex-windows.ps1"
```

## What the Install Script Does

1. Creates a **symlink** from `~/.agents/skills/claude-scholar` → your cloned repo's `skills/` directory
2. Copies `AGENTS.md` to `~/.codex/AGENTS.md` (global Codex instructions)

This means:
- **Zero maintenance updates**: `git pull` in `~/claude-scholar` updates everything instantly
- **No file duplication**: Skills are accessed via symlink, not copied
- **Easy uninstall**: Remove the symlink and AGENTS.md

## Manual Install

If you prefer manual setup:

```bash
# Create skills symlink
mkdir -p ~/.agents/skills
ln -s /path/to/claude-scholar/skills ~/.agents/skills/claude-scholar

# Copy AGENTS.md for global instructions
mkdir -p ~/.codex
cp /path/to/claude-scholar/AGENTS.md ~/.codex/AGENTS.md
```

## Verify Installation

```bash
# Check symlink exists
ls -la ~/.agents/skills/claude-scholar

# Check AGENTS.md
cat ~/.codex/AGENTS.md | head -5

# Start a Codex session and verify skills are discovered
codex
```

## How It Works

### Skill Discovery

Codex automatically discovers skills from `~/.agents/skills/`. Each skill directory must contain a `SKILL.md` with YAML frontmatter including a `description` field. Codex parses this description to determine when to activate the skill.

### AGENTS.md

Codex automatically injects `AGENTS.md` contents at the start of each conversation. This file contains:
- Skill evaluation protocol (equivalent to Claude Code's `skill-forced-eval.js` hook)
- Tool mapping table (Claude Code tools → Codex equivalents)
- Security rules (equivalent to `security-guard.js` hook)
- Session behavior (equivalent to `session-start.js` and `session-summary.js` hooks)
- Coding style and project rules

### What's Different from Claude Code

| Feature | Claude Code | Codex |
|---------|------------|-------|
| Skill activation | `skill-forced-eval.js` hook | AGENTS.md instructions + native discovery |
| Security checks | `security-guard.js` hook | AGENTS.md security rules |
| Session start | `session-start.js` hook | AGENTS.md session behavior |
| Session summary | `session-summary.js` hook | AGENTS.md task completion template |
| File editing | `Edit` / `Write` tools | `apply_patch` |
| Search | `Grep` / `Glob` tools | `rg` / `rg --files` |
| Sub-agents | `Task` tool | `spawn_agent` |
| Commands | Slash commands (`/commit`) | Not available (use skills directly) |
| Hooks | 5 lifecycle hooks | Not available (AGENTS.md replaces) |

## Uninstall

```bash
# macOS / Linux
rm ~/.agents/skills/claude-scholar
rm ~/.codex/AGENTS.md

# Windows (PowerShell)
Remove-Item "$HOME\.agents\skills\claude-scholar" -Force
Remove-Item "$HOME\.codex\AGENTS.md" -Force
```
