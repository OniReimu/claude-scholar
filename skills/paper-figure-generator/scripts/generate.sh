#!/usr/bin/env bash
# AutoFigure-Edit wrapper — 从方法文本生成可编辑 SVG 学术图表
# Usage: generate.sh --method_file <path> --output_dir <path> [options]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILL_DIR="$(dirname "$SCRIPT_DIR")"
AUTOFIGURE_DIR="${SKILL_DIR}/.autofigure-edit"

# 从项目根目录加载 .env（从 skill 目录向上两级）
PROJECT_ROOT="$(cd "$SKILL_DIR/../.." && pwd)"
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# 检查安装
if [ ! -d "$AUTOFIGURE_DIR" ]; then
    echo "Error: AutoFigure-Edit not found at $AUTOFIGURE_DIR"
    echo ""
    echo "Run setup first:"
    echo "  bash ${SCRIPT_DIR}/setup.sh"
    exit 1
fi

# 检查 API key
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "Error: OPENROUTER_API_KEY not set."
    echo "Add it to your project root .env file or export it."
    exit 1
fi

# 默认配置：OpenRouter + Roboflow（API 模式，无需本地安装 SAM3）
SAM_ARGS=()
if [ -n "${ROBOFLOW_API_KEY:-}" ]; then
    SAM_ARGS+=(--sam_backend roboflow)
elif [ -n "${FAL_KEY:-}" ]; then
    SAM_ARGS+=(--sam_backend fal)
else
    echo "Warning: No SAM3 backend API key found (ROBOFLOW_API_KEY or FAL_KEY)."
    echo "Falling back to local SAM3 (requires local installation)."
fi

# 使用 AutoFigure-Edit 虚拟环境中的 Python
PYTHON="${AUTOFIGURE_DIR}/.venv/bin/python"
if [ ! -f "$PYTHON" ]; then
    echo "Error: Virtual environment not found. Run setup first:"
    echo "  bash ${SCRIPT_DIR}/setup.sh"
    exit 1
fi

# macOS: Homebrew 动态库路径（Cairo 等）
if [ "$(uname)" = "Darwin" ] && [ -d "/opt/homebrew/lib" ]; then
    export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
fi

"$PYTHON" "$AUTOFIGURE_DIR/autofigure2.py" \
    --provider openrouter \
    --api_key "${OPENROUTER_API_KEY}" \
    "${SAM_ARGS[@]}" \
    "$@"
