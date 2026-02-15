#!/usr/bin/env bash
# 一次性安装 AutoFigure-Edit 到 skill 目录（项目级别，不影响 home 目录）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILL_DIR="$(dirname "$SCRIPT_DIR")"
INSTALL_DIR="${SKILL_DIR}/.autofigure-edit"

if [ -d "$INSTALL_DIR" ]; then
    echo "AutoFigure-Edit already installed at $INSTALL_DIR"
    echo "Pulling latest changes..."
    cd "$INSTALL_DIR" && git pull
else
    echo "Cloning AutoFigure-Edit..."
    git clone https://github.com/ResearAI/AutoFigure-Edit.git "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"

# 创建虚拟环境并安装依赖
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv .venv
fi

echo "Installing dependencies with uv..."
uv pip install --python .venv/bin/python -r requirements.txt

echo ""
echo "Setup complete at: $INSTALL_DIR"
echo ""
echo "Required environment variables (add to project root .env):"
echo "  OPENROUTER_API_KEY=your-key"
echo "  ROBOFLOW_API_KEY=your-key"
