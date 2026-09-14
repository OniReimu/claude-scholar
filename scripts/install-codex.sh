#!/usr/bin/env bash
# Claude Scholar — Codex Installation Script
# Creates symlinks for native skill discovery
#
# Usage: ./scripts/install-codex.sh [--repo-path /path/to/claude-scholar]

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# Determine repo path
REPO_PATH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-path)
      REPO_PATH="$2"
      shift 2
      ;;
    *)
      error "Unknown option: $1. Usage: $0 [--repo-path /path/to/claude-scholar]"
      ;;
  esac
done

if [[ -z "$REPO_PATH" ]]; then
  # Auto-detect: script is at scripts/install-codex.sh, repo is parent
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_PATH="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

# Validate repo
if [[ ! -d "$REPO_PATH/skills" ]]; then
  error "skills/ directory not found in $REPO_PATH. Is this the claude-scholar repository?"
fi

echo ""
echo "=========================================="
echo "  Claude Scholar — Codex Installer"
echo "=========================================="
echo ""
info "Repository: $REPO_PATH"
echo ""

# --- Step 1: Create skills symlinks ---
# Two locations, because which one Codex actually reads is not something this
# script can test from here:
#   ~/.agents/skills   — the cross-agent convention this installer has always used
#   ~/.codex/skills    — where Codex keeps its own skills, and where its built-in
#                        ones live under .system/
# `.system/` holds six skills each with their own SKILL.md one level down, so a
# directory of skills is a shape Codex already recurses into; `claude-scholar/`
# has the same shape. Linking both costs one symlink and removes the guess.
SKILLS_TARGETS=("$HOME/.agents/skills/claude-scholar" "$HOME/.codex/skills/claude-scholar")

info "Creating skills symlinks..."

for SKILLS_TARGET in "${SKILLS_TARGETS[@]}"; do
  mkdir -p "$(dirname "$SKILLS_TARGET")"

  if [[ -L "$SKILLS_TARGET" ]]; then
    EXISTING_TARGET="$(readlink "$SKILLS_TARGET")"
    if [[ "$EXISTING_TARGET" == "$REPO_PATH/skills" ]]; then
      ok "Symlink already correct: $SKILLS_TARGET"
    else
      warn "Symlink exists but points to: $EXISTING_TARGET"
      warn "Updating to: $REPO_PATH/skills"
      rm "$SKILLS_TARGET"
      ln -s "$REPO_PATH/skills" "$SKILLS_TARGET"
      ok "Symlink updated: $SKILLS_TARGET"
    fi
  elif [[ -e "$SKILLS_TARGET" ]]; then
    error "$SKILLS_TARGET exists but is not a symlink. Remove it manually and re-run."
  else
    ln -s "$REPO_PATH/skills" "$SKILLS_TARGET"
    ok "Symlink created: $SKILLS_TARGET → $REPO_PATH/skills"
  fi
done

# --- Cleanup: remove legacy AGENTS.md if present ---
AGENTS_LEGACY="$HOME/.codex/AGENTS.md"
if [[ -f "$AGENTS_LEGACY" ]] && grep -q "Claude Scholar" "$AGENTS_LEGACY" 2>/dev/null; then
  rm "$AGENTS_LEGACY"
  ok "Removed legacy ~/.codex/AGENTS.md (no longer needed)."
fi

# --- Summary ---
echo ""
echo "=========================================="
echo "  Installation Complete"
echo "=========================================="
echo ""
for SKILLS_TARGET in "${SKILLS_TARGETS[@]}"; do
  ok "Skills symlink: $SKILLS_TARGET"
done
info "Codex will natively discover skills from the symlinked directories."
echo ""
info "To verify, start a Codex session:"
echo "  codex"
echo ""
info "To update later, just pull the repo:"
echo "  cd $REPO_PATH && git pull"
echo ""
info "To uninstall:"
for SKILLS_TARGET in "${SKILLS_TARGETS[@]}"; do
  echo "  rm $SKILLS_TARGET"
done
echo ""
