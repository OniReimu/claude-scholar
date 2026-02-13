# Claude Scholar — Codex Installation Script for Windows
# Creates junction for skill discovery and copies AGENTS.md
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
if (-not (Test-Path "$RepoPath\AGENTS.md")) {
    Write-Err "AGENTS.md not found in $RepoPath. Is this the claude-scholar repository?"
}
if (-not (Test-Path "$RepoPath\skills")) {
    Write-Err "skills\ directory not found in $RepoPath."
}

Write-Host ""
Write-Host "=========================================="
Write-Host "  Claude Scholar - Codex Installer"
Write-Host "=========================================="
Write-Host ""
Write-Info "Repository: $RepoPath"
Write-Host ""

# --- Step 1: Create skills junction ---
$SkillsTarget = "$HOME\.agents\skills\claude-scholar"

Write-Info "Step 1: Creating skills junction..."

$SkillsParent = "$HOME\.agents\skills"
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

# --- Step 2: Copy AGENTS.md ---
$CodexDir = "$HOME\.codex"
$AgentsTarget = "$CodexDir\AGENTS.md"

Write-Info "Step 2: Installing AGENTS.md..."

if (-not (Test-Path $CodexDir)) {
    New-Item -ItemType Directory -Path $CodexDir -Force | Out-Null
}

if (Test-Path $AgentsTarget) {
    $content = Get-Content $AgentsTarget -Raw -ErrorAction SilentlyContinue
    if ($content -match "Claude Scholar") {
        Write-Info "Updating Claude Scholar AGENTS.md..."
    } else {
        Write-Warn "Existing AGENTS.md is not from Claude Scholar."
        Write-Warn "Backing up to ${AgentsTarget}.bak"
        Copy-Item $AgentsTarget "${AgentsTarget}.bak"
    }
}

Copy-Item "$RepoPath\AGENTS.md" $AgentsTarget -Force
Write-Ok "AGENTS.md installed to $AgentsTarget"

# --- Summary ---
Write-Host ""
Write-Host "=========================================="
Write-Host "  Installation Complete"
Write-Host "=========================================="
Write-Host ""
Write-Ok "Skills junction: $SkillsTarget"
Write-Ok "AGENTS.md: $AgentsTarget"
Write-Host ""
Write-Info "To verify, start a Codex session:"
Write-Host "  codex"
Write-Host ""
Write-Info "To update later, just pull the repo:"
Write-Host "  cd $RepoPath; git pull"
Write-Host ""
Write-Info "To uninstall:"
Write-Host "  Remove-Item '$SkillsTarget' -Force"
Write-Host "  Remove-Item '$AgentsTarget' -Force"
Write-Host ""
