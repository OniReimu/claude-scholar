#!/usr/bin/env bash
# Kept as the existing entry point. The checks now live in scripts/validate_skills.py,
# which CI runs; this forwards to it so the two cannot drift apart.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if command -v uv >/dev/null 2>&1; then
  exec uv run --with pyyaml python "$ROOT_DIR/scripts/validate_skills.py" "$@"
fi
exec python3 "$ROOT_DIR/scripts/validate_skills.py" "$@"
