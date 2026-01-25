#!/usr/bin/env bash
# verify-config.sh - 验证 Windows 上的 Claude 配置
#
# 用法: ./verify-config.sh
#
# 运行此脚本验证配置是否正确导入并修复

set -euo pipefail

echo "🧪 Claude 配置验证"
echo "=================="
echo ""

CLAUDE_DIR="$HOME/.claude"
ERRORS=0
WARNINGS=0

# 颜色定义（Windows Git Bash 支持 ANSI 颜色）
red='\033[0;31m'
green='\033[0;32m'
yellow='\033[0;33m'
blue='\033[0;34m'
nc='\033[0m' # No Color

check_pass() {
  echo -e "${green}✓${nc} $1"
}

check_fail() {
  echo -e "${red}✗${nc} $1"
  ((ERRORS++)) || true
}

check_warn() {
  echo -e "${yellow}⚠${nc} $1"
  ((WARNINGS++)) || true
}

check_info() {
  echo -e "${blue}ℹ${nc} $1"
}

# === 1. 环境检查 ===
echo "🔍 [1/9] 环境检查"
echo ""

# 检查平台
if [[ "$(uname -s)" =~ MINGW|MSYS|CYGWIN ]]; then
  check_pass "运行在 Git Bash 环境 ($(uname -s))"
else
  check_fail "未在 Git Bash 环境运行 ($(uname -s))"
fi

# 检查依赖工具
for cmd in jq git bash sed awk grep; do
  if command -v "$cmd" >/dev/null 2>&1; then
    check_pass "$cmd 已安装"
  else
    check_fail "$cmd 未安装"
  fi
done

# 检查 Claude Code
if command -v claude >/dev/null 2>&1; then
  VERSION=$(claude --version 2>/dev/null || echo "unknown")
  check_pass "Claude Code 已安装 (v$VERSION)"
else
  check_warn "Claude Code 未在 PATH 中找到（可能已安装但需要重启终端）"
fi

echo ""

# === 2. 目录结构检查 ===
echo "🔍 [2/9] 目录结构检查"
echo ""

[ -d "$CLAUDE_DIR" ] && check_pass "~/.claude 目录存在" || check_fail "~/.claude 目录不存在"
[ -d "$CLAUDE_DIR/hooks" ] && check_pass "hooks 目录存在" || check_fail "hooks 目录不存在"
[ -d "$CLAUDE_DIR/commands" ] && check_pass "commands 目录存在" || check_fail "commands 目录不存在"
[ -d "$CLAUDE_DIR/skills" ] && check_pass "skills 目录存在" || check_fail "skills 目录不存在"
[ -d "$CLAUDE_DIR/hooks/lib" ] && check_pass "hooks/lib 目录存在" || check_fail "hooks/lib 目录不存在"

echo ""

# === 3. 配置文件检查 ===
echo "🔍 [3/9] 配置文件检查"
echo ""

[ -f "$CLAUDE_DIR/settings.json" ] && check_pass "settings.json 存在" || check_fail "settings.json 缺失"
[ -f "$CLAUDE_DIR/CLAUDE.md" ] && check_pass "CLAUDE.md 存在" || check_fail "CLAUDE.md 缺失"
[ -f "$CLAUDE_DIR/migration-info.json" ] && check_pass "migration-info.json 存在" || check_info "migration-info.json 不存在（首次迁移）"

echo ""

# === 4. Hooks 检查 ===
echo "🔍 [4/9] Hooks 检查"
echo ""

REQUIRED_HOOKS=(
  "session-start.sh"
  "security-guard.sh"
  "skill_forced_eval.sh"
  "session-summary.sh"
  "stop-summary.sh"
)

for hook in "${REQUIRED_HOOKS[@]}"; do
  hook_path="$CLAUDE_DIR/hooks/$hook"
  if [ -f "$hook_path" ]; then
    # 检查可执行权限
    if [ -x "$hook_path" ]; then
      check_pass "$hook 存在且可执行"
    else
      check_warn "$hook 存在但不可执行"
    fi

    # 检查 shebang
    if head -n1 "$hook_path" 2>/dev/null | grep -q "#!/usr/bin/env bash"; then
      check_pass "  └─ shebang 正确 (#!/usr/bin/env bash)"
    elif head -n1 "$hook_path" 2>/dev/null | grep -q "#!/bin/bash"; then
      check_warn "  └─ shebang 可能不兼容 (#!/bin/bash)"
    fi

    # 检查语法
    if bash -n "$hook_path" 2>/dev/null; then
      check_pass "  └─ 语法正确"
    else
      check_fail "  └─ 语法错误"
    fi
  else
    check_fail "$hook 缺失"
  fi
done

# 检查共享库
if [ -f "$CLAUDE_DIR/hooks/lib/common.sh" ]; then
  check_pass "lib/common.sh 存在"
  bash -n "$CLAUDE_DIR/hooks/lib/common.sh" 2>/dev/null && check_pass "  └─ 语法正确" || check_fail "  └─ 语法错误"
else
  check_fail "lib/common.sh 缺失"
fi

if [ -f "$CLAUDE_DIR/hooks/lib/platform.sh" ]; then
  check_pass "lib/platform.sh 存在（跨平台兼容性）"
  bash -n "$CLAUDE_DIR/hooks/lib/platform.sh" 2>/dev/null && check_pass "  └─ 语法正确" || check_fail "  └─ 语法错误"
else
  check_warn "lib/platform.sh 不存在（平台兼容性可能有问题）"
fi

echo ""

# === 5. Commands 检查 ===
echo "🔍 [5/9] Commands 检查"
echo ""

CMD_COUNT=$(find "$CLAUDE_DIR/commands" -name "*.md" -type f 2>/dev/null | wc -l | tr -d ' ')
if [ "$CMD_COUNT" -gt 0 ]; then
  check_pass "找到 $CMD_COUNT 个命令文件"
  # 列出前几个
  find "$CLAUDE_DIR/commands" -name "*.md" -type f 2>/dev/null | head -3 | while read -r cmd; do
    check_info "  - $(basename "$cmd")"
  done
  if [ "$CMD_COUNT" -gt 3 ]; then
    check_info "  ... 还有 $((CMD_COUNT - 3)) 个"
  fi
else
  check_warn "没有找到命令文件"
fi

echo ""

# === 6. Skills 检查 ===
echo "🔍 [6/9] Skills 检查"
echo ""

SKILL_COUNT=$(find "$CLAUDE_DIR/skills" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
if [ "$SKILL_COUNT" -gt 0 ]; then
  check_pass "找到 $SKILL_COUNT 个技能目录"
  # 列出前几个
  find "$CLAUDE_DIR/skills" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -3 | while read -r skill; do
    check_info "  - $(basename "$skill")"
  done
  if [ "$SKILL_COUNT" -gt 3 ]; then
    check_info "  ... 还有 $((SKILL_COUNT - 3)) 个"
  fi
else
  check_warn "没有找到技能目录"
fi

echo ""

# === 7. settings.json 检查 ===
echo "🔍 [7/9] settings.json 检查"
echo ""

if [ -f "$CLAUDE_DIR/settings.json" ]; then
  # 语法检查
  if jq empty "$CLAUDE_DIR/settings.json" 2>/dev/null; then
    check_pass "settings.json JSON 语法正确"

    # 检查 hooks 配置
    HOOK_TYPES=$(jq '.hooks | length' "$CLAUDE_DIR/settings.json" 2>/dev/null || echo "0")
    check_pass "  └─ 配置了 $HOOK_TYPES 个 hook 类型"

    # 检查环境变量
    ENV_COUNT=$(jq '.env | length' "$CLAUDE_DIR/settings.json" 2>/dev/null || echo "0")
    check_pass "  └─ 配置了 $ENV_COUNT 个环境变量"

    # 检查 hooks 命令路径是否使用相对路径
    if jq -e '.hooks.SessionStart[0].hooks[0].command == "bash ~/.claude/hooks/session-start.sh"' "$CLAUDE_DIR/settings.json" >/dev/null 2>&1; then
      check_pass "  └─ hooks 使用 'bash' 相对路径（Windows 兼容）"
    elif jq -e '.hooks.SessionStart[0].hooks[0].command | startswith("/bin/bash")' "$CLAUDE_DIR/settings.json" >/dev/null 2>&1; then
      check_warn "  └─ hooks 使用 '/bin/bash' 绝对路径（在 Windows 上可能有问题）"
    fi
  else
    check_fail "settings.json JSON 语法错误"
  fi
else
  check_fail "settings.json 不存在，跳过检查"
fi

echo ""

# === 8. 环境变量检查 ===
echo "🔍 [8/9] 环境变量检查"
echo ""

check_env_var() {
  local var_name="$1"
  local expected_hint="$2"

  if [ -n "${!var_name:-}" ]; then
    check_pass "$var_name 已设置"
  else
    check_warn "$var_name 未设置 (预期: $expected_hint)"
  fi
}

# 从 settings.json 读取并检查环境变量
if [ -f "$CLAUDE_DIR/settings.json" ]; then
  jq -r '.env | to_entries[] | "\(.key)|\(.value)"' "$CLAUDE_DIR/settings.json" 2>/dev/null | while IFS='|' read -r key value; do
    if [ -n "${!key:-}" ]; then
      check_pass "$key 已设置"
    else
      check_warn "$key 未在 shell 环境中设置"
      check_info "    在 ~/.bashrc 中添加: export $key=$value"
    fi
  done
else
  check_info "settings.json 不存在，跳过环境变量检查"
fi

echo ""

# === 9. 平台兼容性检查 ===
echo "🔍 [9/9] 平台兼容性检查"
echo ""

# 检查 session-start.sh 中的 stat 处理
if [ -f "$CLAUDE_DIR/hooks/session-start.sh" ]; then
  if grep -q "stat_mtime" "$CLAUDE_DIR/hooks/session-start.sh" 2>/dev/null; then
    check_pass "session-start.sh 使用跨平台 stat_mtime 函数"
  elif grep -q "stat.*%m\|stat.*%Y" "$CLAUDE_DIR/hooks/session-start.sh" 2>/dev/null; then
    check_warn "session-start.sh 可能包含平台特定的 stat 调用"
  fi

  # 检查是否引用了 platform.sh
  if grep -q "lib/platform.sh" "$CLAUDE_DIR/hooks/session-start.sh" 2>/dev/null; then
    check_pass "session-start.sh 引用了 platform.sh"
  else
    check_warn "session-start.sh 可能未引用 platform.sh"
  fi
fi

echo ""

# === 功能测试 ===
echo "🧪 功能测试"
echo ""

# 测试 jq 功能
if echo '{}' | jq . >/dev/null 2>&1; then
  check_pass "jq 功能正常"
else
  check_fail "jq 功能异常"
fi

# 测试 Git 功能
if git --version >/dev/null 2>&1; then
  check_pass "Git 功能正常 ($(git --version | head -1))"
else
  check_fail "Git 功能异常"
fi

echo ""

# === 总结 ===
echo "================================"
echo "📊 验证结果摘要"
echo "================================"
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
  echo -e "${green}✓ 所有检查通过!${nc}"
  echo ""
  echo "🎉 配置验证成功！"
  echo ""
  echo "📋 下一步:"
  echo "   1. 确保环境变量已设置（见上方第 8 项检查）"
  echo "   2. 重新加载 shell: source ~/.bashrc"
  echo "   3. 重启 Claude Code"
  exit 0
elif [ $ERRORS -eq 0 ]; then
  echo -e "${yellow}⚠ 有 $WARNINGS 个警告${nc}"
  echo ""
  echo "配置基本可用，但建议解决警告问题以获得最佳体验。"
  echo ""
  echo "📋 常见警告处理:"
  echo "   - 环境变量未设置: 在 ~/.bashrc 中添加 export 语句"
  echo "   - shebang 警告: 通常不影响功能"
  echo "   - Claude Code 未找到: 重启终端或检查安装"
  exit 0
else
  echo -e "${red}✗ 发现 $ERRORS 个错误, $WARNINGS 个警告${nc}"
  echo ""
  echo "请解决错误后再使用 Claude Code。"
  echo ""
  echo "📋 常见错误处理:"
  echo "   - 语法错误: 检查脚本是否完整导入"
  echo "   - 文件缺失: 重新运行 import-windows-config.sh"
  echo "   - 依赖缺失: 安装 jq (choco install jq)"
  exit 1
fi
