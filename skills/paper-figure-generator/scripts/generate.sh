#!/usr/bin/env bash
# AutoFigure-Edit wrapper — 从方法文本生成可编辑 SVG 学术图表
# Usage: generate.sh --method_file <path> --output_dir <path> [options]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILL_DIR="$(dirname "$SCRIPT_DIR")"

# 从项目根目录加载 .env（从 skill 目录向上两级）
PROJECT_ROOT="$(cd "$SKILL_DIR/../.." && pwd)"
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# 解析关键参数（用于默认值/去重/run.json）
METHOD_FILE=""
OUTPUT_DIR=""
PROVIDER_ARG_PRESENT=0
API_KEY_ARG_PRESENT=0
SAM_BACKEND_ARG_PRESENT=0
PROVIDER=""
SAM_BACKEND=""

args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  case "${args[i]}" in
    --method_file)
      METHOD_FILE="${args[i+1]:-}"
      ;;
    --output_dir)
      OUTPUT_DIR="${args[i+1]:-}"
      ;;
    --provider)
      PROVIDER_ARG_PRESENT=1
      PROVIDER="${args[i+1]:-}"
      ;;
    --api_key)
      API_KEY_ARG_PRESENT=1
      ;;
    --sam_backend)
      SAM_BACKEND_ARG_PRESENT=1
      SAM_BACKEND="${args[i+1]:-}"
      ;;
  esac
done

if [ -z "$METHOD_FILE" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Error: --method_file and --output_dir are required."
  echo "Tip: run doctor:"
  echo "  bash $SCRIPT_DIR/doctor.sh"
  exit 1
fi

if [ "$PROVIDER_ARG_PRESENT" -eq 0 ]; then
  PROVIDER="${AUTOFIGURE_PROVIDER:-openrouter}"
fi

KEY_VAR=""
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

if [ "$API_KEY_ARG_PRESENT" -eq 0 ] && [ -z "${!KEY_VAR:-}" ]; then
  echo "Error: $KEY_VAR not set."
  echo "Add it to your project root .env file or export it."
  echo "Tip: run doctor:"
  echo "  bash $SCRIPT_DIR/doctor.sh"
  exit 1
fi

# 检查虚拟环境
PYTHON="${SCRIPT_DIR}/.venv/bin/python"
if [ ! -f "$PYTHON" ]; then
    echo "Error: Virtual environment not found. Run setup first:"
    echo "  bash ${SCRIPT_DIR}/setup.sh"
    exit 1
fi

# 自动选择 SAM3 backend
SAM_ARGS=()
if [ "$SAM_BACKEND_ARG_PRESENT" -eq 0 ]; then
  if [ -n "${ROBOFLOW_API_KEY:-}" ]; then
      SAM_ARGS+=(--sam_backend roboflow)
      SAM_BACKEND="roboflow"
  elif [ -n "${FAL_KEY:-}" ]; then
      SAM_ARGS+=(--sam_backend fal)
      SAM_BACKEND="fal"
  else
      SAM_BACKEND="local"
      echo "Warning: No SAM3 backend API key found (ROBOFLOW_API_KEY or FAL_KEY)."
      echo "Falling back to local SAM3 (requires local installation)."
  fi
fi

# macOS: Homebrew 动态库路径（Cairo 等）
if [ "$(uname)" = "Darwin" ] && [ -d "/opt/homebrew/lib" ]; then
    export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
fi

mkdir -p "$OUTPUT_DIR"

# 记录本次运行参数（不写入任何 secret value）
"$PYTHON" - <<'PY' "$OUTPUT_DIR" "$PROJECT_ROOT" "$PROVIDER" "$SAM_BACKEND" "$KEY_VAR" "$API_KEY_ARG_PRESENT" "$METHOD_FILE" "${args[@]}"
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

output_dir = Path(sys.argv[1])
project_root = Path(sys.argv[2])
provider = sys.argv[3]
sam_backend = sys.argv[4]
key_var = sys.argv[5]
api_key_arg_present = sys.argv[6] == "1"
method_file = sys.argv[7]
raw_args = sys.argv[8:]

def _git_rev(root: Path) -> str | None:
    try:
        import subprocess
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root)
            .decode()
            .strip()
        )
    except Exception:
        return None

run = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "project_root": str(project_root),
    "output_dir": str(output_dir),
    "method_file": str(Path(method_file)),
    "provider": provider,
    "sam_backend": sam_backend,
    "args": raw_args,
    "env": {
        "api_key_source": "cli" if api_key_arg_present else "env",
        "api_key_var": None if api_key_arg_present else key_var,
        # Never write secret values; only record presence.
        "has_OPENROUTER_API_KEY": bool(os.environ.get("OPENROUTER_API_KEY")),
        "has_BIANXIE_API_KEY": bool(os.environ.get("BIANXIE_API_KEY")),
        "has_ROBOFLOW_API_KEY": bool(os.environ.get("ROBOFLOW_API_KEY")),
        "has_FAL_KEY": bool(os.environ.get("FAL_KEY")),
    },
    "git_rev": _git_rev(project_root),
    "platform": {"system": platform.system(), "release": platform.release()},
}

(output_dir / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=True) + "\n")
PY

CMD=("$PYTHON" "$SCRIPT_DIR/autofigure2.py")
if [ "$PROVIDER_ARG_PRESENT" -eq 0 ]; then
  CMD+=(--provider "$PROVIDER")
fi
if [ "$API_KEY_ARG_PRESENT" -eq 0 ]; then
  CMD+=(--api_key "${!KEY_VAR}")
else
  echo "Warning: --api_key provided via CLI args; prefer using .env to avoid shell history leakage."
fi
CMD+=("${SAM_ARGS[@]}")
CMD+=("$@")

"${CMD[@]}"
