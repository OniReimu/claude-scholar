#!/usr/bin/env bash
# 安装 AutoFigure-Edit Python 依赖（源码已 vendor 在 scripts/ 目录中）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

# 创建虚拟环境并安装依赖
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR ..."
    uv venv "$VENV_DIR"
fi

echo "Installing dependencies with uv..."
uv pip install --python "$VENV_DIR/bin/python" -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "Setup complete."
echo "  venv: $VENV_DIR"
echo "  script: $SCRIPT_DIR/autofigure2.py"
echo ""
echo "Required environment variables (add to project root .env):"
echo "  OPENROUTER_API_KEY=your-key"
echo "  ROBOFLOW_API_KEY=your-key"
