#!/usr/bin/env bash
# import-windows-config.sh - 在 Windows 上导入 Claude 配置
#
# 用法: ./import-windows-config.sh <path-to-backup.tar.gz>
#
# 此脚本将在 Windows 上导入 Mac 导出的配置，并自动修复平台兼容性问题

set -euo pipefail

echo "🚀 Claude 配置导入向导 (Windows)"
echo "=================================="
echo ""

# === 1. 环境检查 ===
echo "🔍 [1/7] 环境检查"

# 检查是否在 Git Bash 环境中
if [[ ! "$(uname -s)" =~ MINGW|MSYS|CYGWIN ]]; then
  echo "❌ 错误: 此脚本必须在 Git Bash 中运行"
  echo ""
  echo "请确保:"
  echo "   1. 已安装 Git for Windows"
  echo "   2. 使用 Git Bash（不是 CMD 或 PowerShell）"
  exit 1
fi
echo "   ✓ Git Bash 环境"

# 检查备份文件参数
BACKUP_FILE="${1:-}"

if [ -z "$BACKUP_FILE" ]; then
  echo "❌ 错误: 请提供备份文件路径"
  echo ""
  echo "用法: $0 <claude-mac-backup.tar.gz>"
  echo ""
  echo "示例: $0 ~/claude-mac-backup.tar.gz"
  echo "      $0 /c/Users/YourName/Downloads/claude-mac-backup.tar.gz"
  exit 1
fi

# 转换 Windows 路径格式
if [[ "$BACKUP_FILE" =~ ^[A-Za-z]: ]]; then
  # C:/path -> /c/path
  BACKUP_FILE="/$(echo "$BACKUP_FILE" | cut -c1 | tr 'A-Z' 'a-z')/${BACKUP_FILE:3}"
fi

if [ ! -f "$BACKUP_FILE" ]; then
  echo "❌ 错误: 备份文件不存在: $BACKUP_FILE"
  exit 1
fi
echo "   ✓ 备份文件: $BACKUP_FILE"

# 检查依赖
MISSING_DEPS=()
for cmd in jq tar gzip; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    MISSING_DEPS+=("$cmd")
  fi
done

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
  echo "❌ 缺少依赖: ${MISSING_DEPS[*]}"
  echo ""
  echo "安装方法 (使用 Chocolatey):"
  echo "   choco install jq"
  echo ""
  echo "或手动下载:"
  echo "   jq: https://stedolan.github.io/jq/download/"
  exit 1
fi
echo "   ✓ 依赖检查通过 (jq, tar, gzip)"

echo ""

# === 2. 解压备份 ===
echo "📦 [2/7] 解压备份文件"
TEMP_DIR=$(mktemp -d)
echo "   📁 临时目录: $TEMP_DIR"

cd "$TEMP_DIR"
if ! tar -xzf "$BACKUP_FILE"; then
  echo "❌ 解压失败，请检查备份文件是否完整"
  rm -rf "$TEMP_DIR"
  exit 1
fi

if [ ! -d "claude-migration" ]; then
  echo "❌ 备份文件格式不正确（找不到 claude-migration 目录）"
  rm -rf "$TEMP_DIR"
  exit 1
fi
cd claude-migration

echo "   ✓ 解压完成"
echo ""

# === 3. 显示迁移清单 ===
echo "📋 [3/7] 迁移清单"
echo "   配置文件:"
ls -1 *.json *.md 2>/dev/null | sed 's/^/      - /' || echo "      (无)"
echo "   Hooks:"
ls -1 hooks/*.sh 2>/dev/null | wc -l | xargs echo "      -" 个文件
echo "   Commands:"
find commands -name "*.md" 2>/dev/null | wc -l | xargs echo "      -" 个文件
echo "   Skills:"
find skills -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | xargs echo "      -" 个目录
echo ""

# === 4. 备份现有配置 ===
echo "📦 [4/7] 备份现有配置"
CLAUDE_DIR="$HOME/.claude"
EXISTING_BACKUP=""

if [ -d "$CLAUDE_DIR" ]; then
  EXISTING_BACKUP="$CLAUDE_DIR.backup.$(date +%Y%m%d_%H%M%S)"
  echo "   📁 现有配置已备份到: $EXISTING_BACKUP"
  cp -r "$CLAUDE_DIR" "$EXISTING_BACKUP"
else
  echo "   📁 ~/.claude 不存在，将创建新目录"
fi
echo ""

# === 5. 导入配置 ===
echo "📥 [5/7] 导入配置文件"

# 创建目录结构
mkdir -p "$CLAUDE_DIR/hooks/lib"
mkdir -p "$CLAUDE_DIR/commands"
mkdir -p "$CLAUDE_DIR/skills"
mkdir -p "$CLAUDE_DIR/migrations"

# 导入配置文件
echo "   复制配置文件..."
cp settings.json "$CLAUDE_DIR/" 2>/dev/null || echo "      ⚠️  settings.json 复制失败"
cp CLAUDE.md "$CLAUDE_DIR/" 2>/dev/null || echo "      ⚠️  CLAUDE.md 复制失败"
[ -f settings.local.json ] && cp settings.local.json "$CLAUDE_DIR/" || true

# 导入 Hooks
echo "   复制 Hooks..."
cp -r hooks/*.sh "$CLAUDE_DIR/hooks/" 2>/dev/null || true
if [ -d "hooks/lib" ]; then
  cp -r hooks/lib/* "$CLAUDE_DIR/hooks/lib/" 2>/dev/null || true
fi

# 导入 Commands
echo "   复制 Commands..."
cp -r commands/* "$CLAUDE_DIR/commands/" 2>/dev/null || true

# 导入 Skills
echo "   复制 Skills..."
cp -r skills/* "$CLAUDE_DIR/skills/" 2>/dev/null || true

# 保存迁移信息
cat > "$CLAUDE_DIR/migration-info.json" <<EOF
{
  "migration_date": "$(date -Iseconds)",
  "source_platform": "macOS",
  "source_hostname": "$(cat platform-info.json | jq -r '.source_hostname // "unknown"')",
  "export_time": "$(cat platform-info.json | jq -r '.export_time // "unknown"')"
}
EOF

echo "   ✓ 导入完成"
echo ""

# === 6. 修复平台兼容性 ===
echo "🔧 [6/7] 修复平台兼容性问题"

# 6.1 修复 shebang
echo "   修复 shebang..."
find "$CLAUDE_DIR/hooks" -name "*.sh" -type f 2>/dev/null | while read -r script; do
  sed -i 's|#!/bin/bash|#!/usr/bin/env bash|g' "$script" 2>/dev/null || true
done
echo "      ✓ shebang 已修复"

# 6.2 设置执行权限
echo "   设置执行权限..."
find "$CLAUDE_DIR/hooks" -name "*.sh" -type f -exec chmod +x {} \; 2>/dev/null || true
find "$CLAUDE_DIR/hooks/lib" -name "*.sh" -type f -exec chmod +x {} \; 2>/dev/null || true
echo "      ✓ 权限已设置"

# 6.3 创建平台兼容性库
echo "   创建平台兼容性库..."
cat > "$CLAUDE_DIR/hooks/lib/platform.sh" <<'PLATFORM_EOF'
#!/usr/bin/env bash
# platform.sh - 跨平台兼容性函数库
#
# 提供 macOS/Linux/Windows Git Bash 的跨平台兼容函数

# 跨平台获取文件修改时间（Unix 时间戳）
# 用法: stat_mtime <file>
# 返回: Unix 时间戳（秒）
stat_mtime() {
    local file="$1"

    # macOS/BSD stat
    if stat -f "%m" "$file" >/dev/null 2>&1; then
        stat -f "%m" "$file"
        return 0
    fi

    # Linux/GNU stat
    if stat -c "%Y" "$file" >/dev/null 2>&1; then
        stat -c "%Y" "$file"
        return 0
    fi

    # Windows Git Bash - 使用 date 命令
    if date -r "$file" +%s >/dev/null 2>&1; then
        date -r "$file" +%s
        return 0
    fi

    # 最后的后备方案
    echo "0"
    return 1
}

# 跨平台临时文件创建（mktemp 后备）
# 用法: mktemp_file [prefix]
# 返回: 临时文件路径
mktemp_file() {
    local prefix="${1:-tmp}"
    local tmpdir="${TMP:-/tmp}"

    # 尝试使用 mktemp
    if command mktemp >/dev/null 2>&1; then
        mktemp "${tmpdir}/${prefix}.XXXXXX"
        return 0
    fi

    # 后备方案：使用 $$ 和 RANDOM
    local random_suffix="${RANDOM:-$(date +%s%N)}"
    echo "${tmpdir}/${prefix}.$$.${random_suffix}"
    return 0
}

# 检测命令是否存在
# 用法: command_exists <command>
# 返回: 0=存在, 1=不存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}
PLATFORM_EOF

chmod +x "$CLAUDE_DIR/hooks/lib/platform.sh"
echo "      ✓ platform.sh 已创建"

# 6.4 修复 session-start.sh 中的 stat 调用
echo "   修复 stat 命令调用..."
if [ -f "$CLAUDE_DIR/hooks/session-start.sh" ]; then
  # 添加 platform.sh 的引用
  if ! grep -q "lib/platform.sh" "$CLAUDE_DIR/hooks/session-start.sh"; then
    sed -i '/source.*lib\/common.sh/a source "$(dirname "$0")\/lib\/platform.sh"' "$CLAUDE_DIR/hooks/session-start.sh" 2>/dev/null || true
  fi
  # 替换 stat 调用为 stat_mtime（使用简化的模式避免转义问题）
  sed -i 's/stat -f %m .* 2>.*dev.*null || stat -c %Y .* 2>.*dev.*null/stat_mtime "\$cache_file"/g' "$CLAUDE_DIR/hooks/session-start.sh" 2>/dev/null || true
  echo "      ✓ session-start.sh 已修复"
fi

# 6.5 更新 settings.json 中的 bash 路径
echo "   更新 settings.json 中的路径..."
if [ -f "$CLAUDE_DIR/settings.json" ]; then
  # 备份原文件
  cp "$CLAUDE_DIR/settings.json" "$CLAUDE_DIR/settings.json.import-backup"

  # 使用 jq 更新 hooks 命令路径
  # 使用双引号包裹 jq 过滤器，内部双引号转义
  jq ".hooks.SessionStart[0].hooks[0].command = \"bash ~/.claude/hooks/session-start.sh\" |
    .hooks.PreToolUse[0].hooks[0].command = \"bash ~/.claude/hooks/security-guard.sh\" |
    .hooks.UserPromptSubmit[0].hooks[0].command = \"bash ~/.claude/hooks/skill_forced_eval.sh\" |
    .hooks.SessionEnd[0].hooks[0].command = \"bash ~/.claude/hooks/session-summary.sh\" |
    .hooks.Stop[0].hooks[0].command = \"bash ~/.claude/hooks/stop-summary.sh\"
  " "$CLAUDE_DIR/settings.json" > "$CLAUDE_DIR/settings.json.tmp" 2>/dev/null || true

  if [ -f "$CLAUDE_DIR/settings.json.tmp" ]; then
    mv "$CLAUDE_DIR/settings.json.tmp" "$CLAUDE_DIR/settings.json"
    rm -f "$CLAUDE_DIR/settings.json.import-backup"
    echo "      ✓ settings.json 已更新"
  else
    echo "      ⚠️  settings.json 更新失败，恢复备份"
    mv "$CLAUDE_DIR/settings.json.import-backup" "$CLAUDE_DIR/settings.json"
  fi
fi

echo ""

# === 7. 清理和总结 ===
echo "🧹 [7/7] 清理临时文件"
rm -rf "$TEMP_DIR"
echo "   ✓ 临时文件已清理"
echo ""

# === 总结 ===
echo "=================================="
echo "✅ 导入完成!"
echo "=================================="
echo ""
echo "📋 已导入的配置:"
echo "   - 配置文件: settings.json, CLAUDE.md"
echo "   - Hooks: $(ls -1 "$CLAUDE_DIR/hooks"/*.sh 2>/dev/null | wc -l | tr -d ' ') 个"
echo "   - Commands: $(find "$CLAUDE_DIR/commands" -name "*.md" 2>/dev/null | wc -l | tr -d ' ') 个"
echo "   - Skills: $(find "$CLAUDE_DIR/skills" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ') 个"
echo ""

if [ -n "$EXISTING_BACKUP" ]; then
  echo "📦 原配置备份: $EXISTING_BACKUP"
fi

echo ""
echo "📋 后续步骤:"
echo ""
echo "   1. 设置环境变量（重要!）"
echo "      编辑 ~/.bashrc 添加:"
echo "      export HTTP_PROXY=http://127.0.0.1:7897"
echo "      export HTTPS_PROXY=http://127.0.0.1:7897"
echo "      export ANTHROPIC_AUTH_TOKEN=你的token"
echo "      export ANTHROPIC_BASE_URL=https://open.bigmodel.cn/api/anthropic"
echo "      ... (其他变量见 MANIFEST.md)"
echo ""
echo "   2. 重新加载 shell 配置:"
echo "      source ~/.bashrc"
echo ""
echo "   3. 运行验证脚本:"
echo "      bash ~/.claude/migrations/verify-config.sh"
echo ""
echo "   4. 重启 Claude Code"
echo ""
