# Claude Scholar — Codex Installation Script for Windows
# Creates junction for native skill discovery
#
# Usage: .\scripts\install-codex-windows.ps1 [-RepoPath C:\path\to\claude-scholar]
# Note: Run as Administrator for junction creation

param(
    [string]$RepoPath = ""
)

$ErrorActionPreference = "Stop"

function Write-Info  { Write-Host "[INFO] $args" -ForegroundColor Blue }
function Write-Ok    { Write-Host "[OK] $args" -ForegroundColor Green }
function Write-Warn  { Write-Host "[WARN] $args" -ForegroundColor Yellow }
function Write-Err   { Write-Host "[ERROR] $args" -ForegroundColor Red; exit 1 }

# Determine repo path
if (-not $RepoPath) {
    $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $RepoPath = Split-Path -Parent $ScriptDir
}

# Validate repo
if (-not (Test-Path "$RepoPath\skills")) {
    Write-Err "skills\ directory not found in $RepoPath. Is this the claude-scholar repository?"
}

Write-Host ""
Write-Host "=========================================="
Write-Host "  Claude Scholar - Codex Installer"
Write-Host "=========================================="
Write-Host ""
Write-Info "Repository: $RepoPath"
Write-Host ""

# --- Step 1: Create skills junctions ---
# Two locations, matching install-codex.sh: ~/.agents/skills is the cross-agent
# convention this installer has always used, ~/.codex/skills is where Codex keeps
# its own skills (its built-ins sit under .system/, one SKILL.md per subdirectory,
# so a directory of skills is a shape it already recurses into). Linking both
# costs one junction and removes the guess about which one is read.
$SkillsTargets = @(
    "$HOME\.agents\skills\claude-scholar",
    "$HOME\.codex\skills\claude-scholar"
)

Write-Info "Creating skills junctions..."

foreach ($SkillsTarget in $SkillsTargets) {
    $SkillsParent = Split-Path -Parent $SkillsTarget
    if (-not (Test-Path $SkillsParent)) {
        New-Item -ItemType Directory -Path $SkillsParent -Force | Out-Null
    }

    if (Test-Path $SkillsTarget) {
        $item = Get-Item $SkillsTarget -Force
        if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
            Write-Warn "Junction already exists at $SkillsTarget"
            Write-Info "Removing and recreating..."
            Remove-Item $SkillsTarget -Force
        } else {
            Write-Err "$SkillsTarget exists but is not a junction. Remove it manually and re-run."
        }
    }

    # Create directory junction (works without admin on modern Windows)
    cmd /c mklink /J "$SkillsTarget" "$RepoPath\skills" | Out-Null
    Write-Ok "Junction created: $SkillsTarget -> $RepoPath\skills"
}

# --- Cleanup: remove legacy AGENTS.md if present ---
$AgentsLegacy = "$HOME\.codex\AGENTS.md"
if (Test-Path $AgentsLegacy) {
    $content = Get-Content $AgentsLegacy -Raw -ErrorAction SilentlyContinue
    if ($content -match "Claude Scholar") {
        Remove-Item $AgentsLegacy
        Write-Ok "Removed legacy $AgentsLegacy (no longer needed)."
    }
}

# --- Summary ---
Write-Host ""
Write-Host "=========================================="
Write-Host "  Installation Complete"
Write-Host "=========================================="
Write-Host ""
foreach ($SkillsTarget in $SkillsTargets) { Write-Ok "Skills junction: $SkillsTarget" }
Write-Info "Codex will natively discover skills from the junction directory."
Write-Host ""
Write-Info "To verify, start a Codex session:"
Write-Host "  codex"
Write-Host ""
Write-Info "To update later, just pull the repo:"
Write-Host "  cd $RepoPath; git pull"
Write-Host ""
Write-Info "To uninstall:"
foreach ($SkillsTarget in $SkillsTargets) { Write-Host "  Remove-Item '$SkillsTarget' -Force" }
Write-Host ""
