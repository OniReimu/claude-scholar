#!/usr/bin/env bash
# export-mac-config.sh - 导出 Mac 配置到迁移包
#
# 用法: ./export-mac-config.sh
#
# 此脚本将 Mac 上的 Claude 配置打包成 tar.gz 文件，
# 可以传输到 Windows 后使用 import-windows-config.sh 导入

set -euo pipefail

BACKUP_DIR="$HOME/claude-migration"
BACKUP_FILE="$HOME/claude-mac-backup.tar.gz"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "📦 开始导出 Claude 配置..."
echo "================================"
echo ""

# 创建临时目录
rm -rf "$BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# 1. 复制配置文件
echo "📋 [1/6] 复制配置文件..."
cp ~/.claude/settings.json "$BACKUP_DIR/" 2>/dev/null || echo "   ⚠️  settings.json 不存在"
cp ~/.claude/CLAUDE.md "$BACKUP_DIR/" 2>/dev/null || echo "   ⚠️  CLAUDE.md 不存在"
if [ -f ~/.claude/settings.local.json ]; then
  cp ~/.claude/settings.local.json "$BACKUP_DIR/"
fi
echo "   ✓ 配置文件已复制"

# 2. 复制 Hooks
echo "📋 [2/6] 复制 Hooks..."
mkdir -p "$BACKUP_DIR/hooks"
cp -r ~/.claude/hooks/*.sh "$BACKUP_DIR/hooks/" 2>/dev/null || true
if [ -d ~/.claude/hooks/lib ]; then
  cp -r ~/.claude/hooks/lib "$BACKUP_DIR/hooks/"
fi
echo "   ✓ Hooks 已复制 ($(ls -1 "$BACKUP_DIR/hooks"/*.sh 2>/dev/null | wc -l | tr -d ' ') 个文件)"

# 2.5. 为 Windows 兼容性修改 Hooks
echo "📋 [2.5/6] 修改 Hooks 以兼容 Windows..."

# 修改 lib/common.sh - 添加 PATH 和跨平台函数
if [ -f "$BACKUP_DIR/hooks/lib/common.sh" ]; then
  # 在文件开头添加 PATH 设置
  sed -i '' '2a\
\
# 确保 PATH 包含用户 bin 目录（MSYS/Git Bash 环境）\
export PATH="/c/Users/10557/bin:$PATH"
' "$BACKUP_DIR/hooks/lib/common.sh"

  # 在文件末尾添加跨平台函数（如果还没有的话）
  if ! grep -q "mktemp_file" "$BACKUP_DIR/hooks/lib/common.sh"; then
    cat >> "$BACKUP_DIR/hooks/lib/common.sh" <<'EOF'

# 创建临时文件（跨平台兼容）
# 参数: $1 - 文件前缀
# 返回: 临时文件路径
mktemp_file() {
  local prefix="$1"
  local tmpdir="${TMPDIR:-/tmp}"

  # 尝试使用 mktemp
  if command -v mktemp >/dev/null 2>&1; then
    mktemp "${tmpdir}/${prefix}.XXXXXX"
  else
    # 回退方案：使用时间戳和随机数
    echo "${tmpdir}/${prefix}.$$-${RANDOM}"
  fi
}

# 获取文件修改时间（秒级时间戳）
# 参数: $1 - 文件路径
# 返回: Unix 时间戳
stat_mtime() {
  local file="$1"

  # macOS 使用 stat -f %m
  if stat -f %m "$file" >/dev/null 2>&1; then
    stat -f %m "$file"
  # Linux 使用 stat -c %Y
  elif stat -c %Y "$file" >/dev/null 2>&1; then
    stat -c %Y "$file"
  else
    # 回退：使用 find 的 -printf
    find "$file" -maxdepth 0 -printf '%T@\n' 2>/dev/null | cut -d. -f1
  fi
}
EOF
  fi
  echo "   ✓ lib/common.sh 已修改"
fi

# 修改 security-guard.sh - 添加 PATH
if [ -f "$BACKUP_DIR/hooks/security-guard.sh" ]; then
  sed -i '' '3a\
\
# 确保 PATH 包含用户 bin 目录（MSYS/Git Bash 环境）\
export PATH="/c/Users/10557/bin:$PATH"
' "$BACKUP_DIR/hooks/security-guard.sh"
  echo "   ✓ security-guard.sh 已修改"
fi

# 修改 skill_forced_eval.sh - 添加 PATH 和 common.sh source
if [ -f "$BACKUP_DIR/hooks/skill_forced_eval.sh" ]; then
  # 1. 添加 PATH（在 set -euo pipefail 之后）
  sed -i '' '/set -euo pipefail/a\
\
# 确保 PATH 包含用户 bin 目录（MSYS/Git Bash 环境）\
export PATH="/c/Users/10557/bin:$PATH"
' "$BACKUP_DIR/hooks/skill_forced_eval.sh"

  # 2. 修改注释
  sed -i '' 's/# 加载跨平台兼容函数/# 加载共享函数库/' "$BACKUP_DIR/hooks/skill_forced_eval.sh"

  # 3. 在 SCRIPT_DIR 行之后添加 common.sh source
  sed -i '' "/SCRIPT_DIR=/a\\
if [ -f \"\$SCRIPT_DIR/lib/common.sh\" ]; then\\
    source \"\$SCRIPT_DIR/lib/common.sh\"\\
fi
" "$BACKUP_DIR/hooks/skill_forced_eval.sh"

  echo "   ✓ skill_forced_eval.sh 已修改"
fi

echo "   ✓ Windows 兼容性修改完成"

# 3. 复制 Commands
echo "📋 [3/6] 复制 Commands..."
mkdir -p "$BACKUP_DIR/commands"
if [ -d ~/.claude/commands ]; then
  cp -r ~/.claude/commands/* "$BACKUP_DIR/commands/" 2>/dev/null || true
fi
echo "   ✓ Commands 已复制"

# 4. 复制 Skills
echo "📋 [4/6] 复制 Skills..."
mkdir -p "$BACKUP_DIR/skills"
if [ -d ~/.claude/skills ]; then
  cp -r ~/.claude/skills/* "$BACKUP_DIR/skills/" 2>/dev/null || true
fi
echo "   ✓ Skills 已复制"

# 5. 生成平台信息
echo "📋 [5/6] 生成平台信息..."
cat > "$BACKUP_DIR/platform-info.json" <<EOF
{
  "source_platform": "macOS",
  "source_hostname": "$(hostname)",
  "export_time": "$(date -Iseconds)",
  "bash_version": "$BASH_VERSION",
  "claude_home": "$HOME/.claude"
}
EOF
echo "   ✓ 平台信息已生成"

# 6. 生成迁移清单
echo "📋 [6/6] 生成迁移清单..."
cat > "$BACKUP_DIR/MANIFEST.md" <<'EOF'
# Claude 配置迁移清单

## 配置文件
- [ ] settings.json - 主配置（含 hooks、env、plugins）
- [ ] CLAUDE.md - 全局指令
- [ ] settings.local.json - 本地配置（如有）

## Hooks
- [ ] session-start.sh - 会话启动时显示项目状态
- [ ] security-guard.sh - PreToolUse 安全检查
- [ ] skill_forced_eval.sh - UserPromptSubmit 技能评估
- [ ] session-summary.sh - 会话结束时总结
- [ ] stop-summary.sh - 停止时总结
- [ ] lib/common.sh - 共享函数库

## Commands
- [ ] create_project.md - 项目创建命令

## Skills
- [ ] agent-identifier
- [ ] architecture-design
- [ ] bug-detective
- [ ] code-review-excellence
- [ ] command-development
- [ ] command-name
- [ ] doc-coauthoring
- [ ] git-workflow
- [ ] hook-development
- [ ] mcp-integration
- [ ] planning-with-files
- [ ] scientific-writing
- [ ] skill-development
- [ ] uv-package-manager

## 环境变量（需在 Windows 上手动配置）

从 settings.json 中提取的环境变量：

```bash
# 代理设置（根据你的 Windows 代理软件调整）
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897

# API 配置
export ANTHROPIC_AUTH_TOKEN=你的token
export ANTHROPIC_BASE_URL=https://open.bigmodel.cn/api/anthropic

# 超时设置
export API_TIMEOUT_MS=3000000

# 其他配置
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export ANTHROPIC_DEFAULT_HAIKU_MODEL=glm-4.5-air
export ANTHROPIC_DEFAULT_SONNET_MODEL=glm-4.7
export ANTHROPIC_DEFAULT_OPUS_MODEL=glm-4.7
```

### Windows 环境变量设置方法

**方法 1: 在 ~/.bashrc 中添加**（推荐）
```bash
echo 'export HTTP_PROXY=http://127.0.0.1:7897' >> ~/.bashrc
# ... 添加其他变量
source ~/.bashrc
```

**方法 2: Windows 系统环境变量**
1. 右键"此电脑" → 属性 → 高级系统设置
2. 环境变量 → 新建用户变量
3. 添加上述环境变量

## Windows 安装前检查

- [ ] Git for Windows 已安装（使用 Git Bash）
- [ ] jq 已安装（Chocolatey: `choco install jq`）
- [ ] Claude Code 已安装
- [ ] 备份文件已传输到 Windows

## 已知平台兼容性问题

### 1. PATH 环境变量
Windows Git Bash 中 hook 执行环境的 PATH 不包含 jq 等工具的安装路径。
**解决方案**: export 脚本已自动在所有 hook 文件中添加 `export PATH="/c/Users/10557/bin:$PATH"`。
**注意**: 如果你的 jq 安装路径不同，请手动修改此路径。

### 2. stat 命令
Mac 使用 BSD stat (`stat -f %m`)，Linux/Git Bash 使用 GNU stat (`stat -c %Y`)。
**解决方案**: export 脚本已自动在 `lib/common.sh` 中添加 `stat_mtime()` 跨平台函数。

### 3. mktemp 命令
某些 Windows 环境可能没有 mktemp 命令。
**解决方案**: export 脚本已自动在 `lib/common.sh` 中添加 `mktemp_file()` 跨平台函数。

### 4. bash 路径
Mac 上 `/bin/bash` 是标准路径，Windows Git Bash 可能在 `/usr/bin/bash`。
**解决方案**: import 脚本会将 settings.json 中的 `/bin/bash` 改为 `bash`。

### 5. shebang
使用 `#!/usr/bin/env bash` 而非 `#!/bin/bash` 以提高兼容性。
**解决方案**: import 脚本会自动修复。
EOF
echo "   ✓ 迁移清单已生成"

# 打包
echo ""
echo "📦 打包中..."
cd "$HOME"
tar -czf "$BACKUP_FILE" claude-migration/

# 获取文件大小
FILE_SIZE=$(ls -lh "$BACKUP_FILE" | awk '{print $5}')

echo ""
echo "================================"
echo "✅ 导出完成!"
echo "================================"
echo ""
echo "📁 备份文件: $BACKUP_FILE"
echo "📦 文件大小: $FILE_SIZE"
echo "📂 临时目录: $BACKUP_DIR"
echo ""
echo "📋 下一步操作:"
echo "   1. 将 $BACKUP_FILE 传输到 Windows"
echo "   2. 在 Windows Git Bash 中运行:"
echo "      chmod +x import-windows-config.sh"
echo "      ./import-windows-config.sh ~/claude-mac-backup.tar.gz"
echo ""
echo "💡 提示: 临时目录 $BACKUP_DIR 可以保留或手动删除"
