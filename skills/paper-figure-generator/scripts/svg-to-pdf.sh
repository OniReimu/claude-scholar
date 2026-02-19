#!/usr/bin/env bash
# Convert AutoFigure-Edit SVG output to PDF for LaTeX inclusion.
#
# Usage:
#   svg-to-pdf.sh --svg figures/{slug}/final.svg --pdf figures/{slug}/figure.pdf
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

SVG_PATH=""
PDF_PATH=""

args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  case "${args[i]}" in
    --svg)
      SVG_PATH="${args[i+1]:-}"
      ;;
    --pdf)
      PDF_PATH="${args[i+1]:-}"
      ;;
  esac
done

if [ -z "$SVG_PATH" ] || [ -z "$PDF_PATH" ]; then
  echo "Usage: $0 --svg <path> --pdf <path>"
  exit 1
fi

VENV_PY="$SCRIPT_DIR/.venv/bin/python"
if [ ! -f "$VENV_PY" ]; then
  echo "Error: virtual environment not found. Run setup first:"
  echo "  bash $SCRIPT_DIR/setup.sh"
  exit 1
fi

# macOS: ensure Homebrew Cairo can be found by cairocffi/cairosvg
if [ "$(uname)" = "Darwin" ] && [ -d "/opt/homebrew/lib" ]; then
  export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
fi

if ! "$VENV_PY" -c "import cairosvg" >/dev/null 2>&1; then
  echo "Error: cairosvg not available in skill venv (package missing or Cairo dylib unresolved)."
  echo "Install it:"
  echo "  uv pip install --python \"$VENV_PY\" cairosvg"
  echo "If you're on macOS with Homebrew, ensure /opt/homebrew/lib is present."
  exit 1
fi

mkdir -p "$(dirname "$PDF_PATH")"

"$VENV_PY" -c "import cairosvg; cairosvg.svg2pdf(url='$SVG_PATH', write_to='$PDF_PATH')"
echo "Wrote: $PDF_PATH"

echo ""
echo "[no-title-lint] Checking SVG/PDF for accidental in-figure title text..."
"$VENV_PY" "$SCRIPT_DIR/lint_no_title.py" --strict --path "$SVG_PATH"
"$VENV_PY" "$SCRIPT_DIR/lint_no_title.py" --strict --path "$PDF_PATH"
