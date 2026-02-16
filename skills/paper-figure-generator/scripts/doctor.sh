#!/usr/bin/env bash
# 运行前环境检查：uv/venv/.env/API keys/SAM backend/cairosvg
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILL_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(cd "$SKILL_DIR/../.." && pwd)"

echo "AutoFigure-Edit doctor"
echo "  skill:   $SKILL_DIR"
echo "  project: $PROJECT_ROOT"
echo ""

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: 'uv' not found."
  echo "Install uv first (recommended):"
  echo "  macOS (Homebrew): brew install uv"
  echo "  pipx:             pipx install uv"
  exit 1
fi

VENV_PY="$SCRIPT_DIR/.venv/bin/python"
if [ ! -f "$VENV_PY" ]; then
  echo "Error: virtual environment not found:"
  echo "  $VENV_PY"
  echo "Run setup:"
  echo "  bash $SCRIPT_DIR/setup.sh"
  exit 1
fi

# Load .env from project root if present (do NOT print values)
if [ -f "$PROJECT_ROOT/.env" ]; then
  set -a
  source "$PROJECT_ROOT/.env"
  set +a
else
  echo "Warning: project root .env not found."
  echo "Create it from .env.example:"
  echo "  cp .env.example .env"
  echo ""
fi

PROVIDER="${AUTOFIGURE_PROVIDER:-openrouter}"
case "$PROVIDER" in
  openrouter)
    KEY_VAR="OPENROUTER_API_KEY"
    ;;
  bianxie)
    KEY_VAR="BIANXIE_API_KEY"
    ;;
  *)
    echo "Error: unknown provider: $PROVIDER"
    echo "Supported: openrouter, bianxie"
    exit 1
    ;;
esac

if [ -z "${!KEY_VAR:-}" ]; then
  echo "Error: $KEY_VAR not set."
  echo "Add it to project root .env, or export it in your shell."
  exit 1
fi

SAM_BACKEND="local"
if [ -n "${ROBOFLOW_API_KEY:-}" ]; then
  SAM_BACKEND="roboflow"
elif [ -n "${FAL_KEY:-}" ]; then
  SAM_BACKEND="fal"
fi

echo "Provider:"
echo "  provider=$PROVIDER"
echo "  api_key=$KEY_VAR (set)"
echo ""

echo "SAM3 backend:"
if [ "$SAM_BACKEND" = "local" ]; then
  echo "  local (Warning: requires local SAM3 installation)"
else
  echo "  $SAM_BACKEND (API key detected)"
fi
echo ""

if "$VENV_PY" -c "import cairosvg" >/dev/null 2>&1; then
  echo "SVG->PDF:"
  echo "  cairosvg: OK (in skill venv)"
else
  echo "SVG->PDF:"
  echo "  cairosvg: MISSING (in skill venv)"
  echo "  Install it:"
  echo "    uv pip install --python \"$VENV_PY\" cairosvg"
fi
echo ""

echo "Next:"
echo "  bash $SCRIPT_DIR/generate.sh --method_file figures/{slug}/method.txt --output_dir figures/{slug}"

