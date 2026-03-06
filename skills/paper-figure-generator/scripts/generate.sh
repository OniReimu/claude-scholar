#!/usr/bin/env bash
# AutoFigure-Edit wrapper — 从方法文本生成可编辑 SVG 学术图表
# Usage: generate.sh --method_file <path> --output_dir <path> [options]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILL_DIR="$(dirname "$SCRIPT_DIR")"
SYSTEM_PYTHON="$(command -v python3 || command -v python || true)"

# 解析关键参数（用于默认值/去重/run.json）
METHOD_FILE=""
OUTPUT_DIR=""
PROVIDER_ARG_PRESENT=0
API_KEY_ARG_PRESENT=0
SAM_BACKEND_ARG_PRESENT=0
USE_REFERENCE_IMAGE_ARG_PRESENT=0
REFERENCE_IMAGE_PATH_ARG_PRESENT=0
PROVIDER=""
SAM_BACKEND=""
REFERENCE_IMAGE_PATH_ARG=""

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
    --use_reference_image)
      USE_REFERENCE_IMAGE_ARG_PRESENT=1
      ;;
    --reference_image_path)
      REFERENCE_IMAGE_PATH_ARG_PRESENT=1
      REFERENCE_IMAGE_PATH_ARG="${args[i+1]:-}"
      ;;
    --reference_image_path=*)
      REFERENCE_IMAGE_PATH_ARG_PRESENT=1
      REFERENCE_IMAGE_PATH_ARG="${args[i]#*=}"
      ;;
  esac
done

if [ -z "$METHOD_FILE" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Error: --method_file and --output_dir are required."
  echo "Tip: run doctor:"
  echo "  bash $SCRIPT_DIR/doctor.sh"
  exit 1
fi

# 从实际 workspace 根目录加载 .env，而不是 skill 安装目录
WORKSPACE_ROOT="$PWD"
if [ -n "$SYSTEM_PYTHON" ]; then
  DETECTED_ROOT="$("$SYSTEM_PYTHON" "$SCRIPT_DIR/workspace_root.py" \
    --output-dir "$OUTPUT_DIR" \
    --method-file "$METHOD_FILE" \
    --workspace-cwd "$PWD" 2>/dev/null || true)"
  if [ -n "$DETECTED_ROOT" ]; then
    WORKSPACE_ROOT="$DETECTED_ROOT"
  fi
fi

if [ -f "$WORKSPACE_ROOT/.env" ]; then
    set -a
    source "$WORKSPACE_ROOT/.env"
    set +a
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
PYTHON="${AUTOFIGURE_PYTHON:-${SCRIPT_DIR}/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
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

# 默认参考图策略（用户未显式提供 reference_image_path 时启用）
FINAL_ARGS=("${args[@]}")
DEFAULT_REFERENCE_DIR="$SKILL_DIR/.autofigure-edit/img/reference"
DEFAULT_REFERENCE_PATH=""
for candidate in \
  "$DEFAULT_REFERENCE_DIR/sample3.png" \
  "$DEFAULT_REFERENCE_DIR/sample2.png"; do
  if [ -f "$candidate" ]; then
    DEFAULT_REFERENCE_PATH="$candidate"
    break
  fi
done

if [ "$REFERENCE_IMAGE_PATH_ARG_PRESENT" -eq 0 ]; then
  if [ -n "$DEFAULT_REFERENCE_PATH" ]; then
    if [ "$USE_REFERENCE_IMAGE_ARG_PRESENT" -eq 0 ]; then
      FINAL_ARGS+=(--use_reference_image)
    fi
    FINAL_ARGS+=(--reference_image_path "$DEFAULT_REFERENCE_PATH")
    echo "Reference style: default ($DEFAULT_REFERENCE_PATH)"
  elif [ "$USE_REFERENCE_IMAGE_ARG_PRESENT" -eq 1 ]; then
    echo "Error: --use_reference_image provided but no --reference_image_path."
    echo "Also no default reference image found in:"
    echo "  $DEFAULT_REFERENCE_DIR"
    exit 1
  else
    echo "Reference style: none (no default reference found)"
  fi
else
  echo "Reference style: user-provided ($REFERENCE_IMAGE_PATH_ARG)"
fi

# 记录本次运行参数（不写入任何 secret value）
API_KEY_SOURCE="env"
if [ "$API_KEY_ARG_PRESENT" -eq 1 ]; then
  API_KEY_SOURCE="cli"
fi

"$PYTHON" "$SCRIPT_DIR/write_run_metadata.py" \
  --output-dir "$OUTPUT_DIR" \
  --method-file "$METHOD_FILE" \
  --provider "$PROVIDER" \
  --sam-backend "$SAM_BACKEND" \
  --api-key-source "$API_KEY_SOURCE" \
  --api-key-var "$KEY_VAR" \
  --workspace-cwd "$WORKSPACE_ROOT" \
  -- "${FINAL_ARGS[@]}"

CMD=("$PYTHON" "$SCRIPT_DIR/autofigure2.py")
if [ "$PROVIDER_ARG_PRESENT" -eq 0 ]; then
  CMD+=(--provider "$PROVIDER")
fi
if [ "$API_KEY_ARG_PRESENT" -eq 0 ]; then
  CMD+=(--api_key "${!KEY_VAR}")
else
  echo "Warning: --api_key provided via CLI args; prefer using .env to avoid shell history leakage."
fi
if [ "${#SAM_ARGS[@]}" -gt 0 ]; then
  CMD+=("${SAM_ARGS[@]}")
fi
CMD+=("${FINAL_ARGS[@]}")

"${CMD[@]}"

echo ""
echo "[no-title-lint] Checking generated outputs for accidental in-figure title text..."
"$PYTHON" "$SCRIPT_DIR/lint_no_title.py" --strict --path "$OUTPUT_DIR"
