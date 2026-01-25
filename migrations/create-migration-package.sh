#!/usr/bin/env bash
# create-migration-package.sh - 创建完整的 Windows 迁移包
#
# 用法: ./create-migration-package.sh
#
# 此脚本将所有迁移工具打包成一个 tar.gz 文件，
# 方便传输到 Windows 使用

set -euo pipefail

MIGRATION_DIR="$HOME/.claude/migrations"
PACKAGE_DIR="$HOME/claude-windows-migration"
PACKAGE_FILE="$HOME/claude-windows-migration.tar.gz"

echo "📦 创建 Windows 迁移包"
echo "======================"
echo ""

# 检查迁移脚本是否存在
echo "🔍 检查迁移脚本..."
REQUIRED_SCRIPTS=(
  "export-mac-config.sh"
  "import-windows-config.sh"
  "verify-config.sh"
)

for script in "${REQUIRED_SCRIPTS[@]}"; do
  if [ -f "$MIGRATION_DIR/$script" ]; then
    echo "   ✓ $script"
  else
    echo "   ✗ $script 缺失"
    echo ""
    echo "请确保所有迁移脚本已创建"
    exit 1
  fi
done

echo ""

# 清理并创建目录
echo "📁 创建打包目录..."
rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR"

# 复制所有迁移脚本
echo "📋 复制迁移脚本..."
cp "$MIGRATION_DIR/export-mac-config.sh" "$PACKAGE_DIR/"
cp "$MIGRATION_DIR/import-windows-config.sh" "$PACKAGE_DIR/"
cp "$MIGRATION_DIR/verify-config.sh" "$PACKAGE_DIR/"
echo "   ✓ 迁移脚本已复制"

# 创建 README
echo "📝 创建 README.md..."
cat > "$PACKAGE_DIR/README.md" <<'EOF'
# Claude 配置迁移包 - Mac 到 Windows

## 📋 包含文件

| 文件 | 用途 | 运行平台 |
|------|------|---------|
| `export-mac-config.sh` | 导出 Mac 配置 | Mac |
| `import-windows-config.sh` | 导入到 Windows | Windows (Git Bash) |
| `verify-config.sh` | 验证配置正确性 | Windows (Git Bash) |

---

## 🚀 使用步骤

### 在 Mac 上

#### 1. 导出配置
打开终端，运行：

```bash
chmod +x export-mac-config.sh
./export-mac-config.sh
```

这将生成 `claude-mac-backup.tar.gz` 文件。

#### 2. 传输到 Windows
将以下文件传输到 Windows：
- `claude-mac-backup.tar.gz`（配置数据）
- `claude-windows-migration.tar.gz`（本工具包）

传输方式：U盘、网盘、邮件等。

---

### 在 Windows 上

#### 1. 安装依赖

**必需软件：**
- [Git for Windows](https://git-scm.com/download/win) - 提供 Git Bash
- [jq](https://stedolan.github.io/jq/) - JSON 处理工具

安装 jq（推荐使用 Chocolatey）：
```powershell
choco install jq
```

或手动下载：https://stedolan.github.io/jq/download/

#### 2. 解压工具包
在 Windows 上解压 `claude-windows-migration.tar.gz`

#### 3. 打开 Git Bash
右键文件夹 → "Git Bash Here"

#### 4. 导入配置
```bash
chmod +x import-windows-config.sh
./import-windows-config.sh /c/Users/YourName/Downloads/claude-mac-backup.tar.gz
```

**注意**: Windows 路径格式：
- `C:\Users\Name\file.tar.gz` → `/c/Users/Name/file.tar.gz`
- 或将文件放在 `~/` (home) 目录直接使用 `~/claude-mac-backup.tar.gz`

#### 5. 验证配置
```bash
chmod +x verify-config.sh
./verify-config.sh
```

#### 6. 设置环境变量
编辑 `~/.bashrc`，添加：

```bash
# Claude Code 环境变量
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897
export ANTHROPIC_AUTH_TOKEN=你的token
export ANTHROPIC_BASE_URL=https://open.bigmodel.cn/api/anthropic
export API_TIMEOUT_MS=3000000
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export ANTHROPIC_DEFAULT_HAIKU_MODEL=glm-4.5-air
export ANTHROPIC_DEFAULT_SONNET_MODEL=glm-4.7
export ANTHROPIC_DEFAULT_OPUS_MODEL=glm-4.7
```

然后重新加载：
```bash
source ~/.bashrc
```

#### 7. 重启 Claude Code

---

## ❓ 常见问题

### Q: bash: command not found: jq
**A**: jq 未安装或不在 PATH 中。
- 安装：`choco install jq`
- 验证：`jq --version`

### Q: 权限被拒绝 / Permission denied
**A**: 脚本没有执行权限。
```bash
chmod +x *.sh
```

### Q: stat 命令错误
**A**: 这是平台兼容性问题，`import-windows-config.sh` 会自动修复。
如果仍有问题，检查 `~/.claude/hooks/lib/platform.sh` 是否存在。

### Q: 环境变量未生效
**A**: 有两种设置方式：
1. 在 `~/.bashrc` 中添加（推荐，仅 Git Bash 生效）
2. Windows 系统环境变量（所有程序生效）

### Q: Claude Code 找不到命令/hook
**A**:
1. 运行 `verify-config.sh` 检查配置
2. 确保 Claude Code 已重启
3. 检查 `~/.claude/settings.json` 中的路径

### Q: 导入后 Hook 不工作
**A**: 检查：
1. `~/.claude/hooks/` 下的 `.sh` 文件是否有执行权限
2. `settings.json` 中的 hook 命令路径是否正确（应为 `bash ~/.claude/hooks/...`）

---

## 🔧 技术细节

### 平台兼容性处理

| 问题 | Mac | Windows | 解决方案 |
|------|-----|---------|---------|
| stat 命令 | `stat -f %m` (BSD) | `stat -c %Y` (GNU) | 创建 `stat_mtime()` 函数 |
| bash 路径 | `/bin/bash` | 可能不同 | 使用 `bash` (相对路径) |
| shebang | `#!/bin/bash` | 可能不兼容 | `#!/usr/bin/env bash` |
| 路径分隔符 | `/` | `/` (Git Bash 兼容) | 统一使用 `/` |

### 导入的配置

**配置文件：**
- `settings.json` - 主配置（hooks、环境变量、插件）
- `CLAUDE.md` - 全局指令

**Hooks：**
- `session-start.sh` - 会话启动时显示状态
- `security-guard.sh` - 命令安全检查
- `skill_forced_eval.sh` - 技能评估
- `session-summary.sh` - 会话总结
- `stop-summary.sh` - 停止总结
- `lib/common.sh` - 共享函数库
- `lib/platform.sh` - 平台兼容性库

**Commands：**
- 用户自定义的 slash 命令

**Skills：**
- 用户自定义的技能

---

## 📞 支持

遇到问题请检查：
1. Git Bash 版本是否最新
2. jq 是否正确安装（`jq --version`）
3. Claude Code 是否正确安装
4. 运行 `verify-config.sh` 查看详细错误信息

如需回滚，导入脚本会自动备份现有配置到：
`~/.claude.backup.YYYYMMDD_HHMMSS`
EOF

echo "   ✓ README.md 已创建"

# 创建环境变量设置脚本
echo "📝 创建环境变量设置脚本..."
cat > "$PACKAGE_DIR/setup-env.sh" <<'EOF'
#!/usr/bin/env bash
# setup-env.sh - 在 Windows 上设置 Claude 环境变量
#
# 用法: ./setup-env.sh
#
# 此脚本将帮助你在 ~/.bashrc 中设置 Claude Code 所需的环境变量

echo "📝 设置 Claude Code 环境变量"
echo "=============================="
echo ""

BASHRC="$HOME/.bashrc"

# 检查是否已配置
if grep -q "CLAUDE_CODE" "$BASHRC" 2>/dev/null; then
  echo "⚠️  环境变量似乎已在 ~/.bashrc 中配置"
  echo ""
  read -p "是否重新配置? (y/n): " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "取消操作"
    exit 0
  fi
  echo ""
fi

# 添加环境变量到 ~/.bashrc
cat >> "$BASHRC" <<'ENV_EOF'

# Claude Code 环境变量（由迁移脚本添加）
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897
export ANTHROPIC_AUTH_TOKEN=你的token
export ANTHROPIC_BASE_URL=https://open.bigmodel.cn/api/anthropic
export API_TIMEOUT_MS=3000000
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export ANTHROPIC_DEFAULT_HAIKU_MODEL=glm-4.5-air
export ANTHROPIC_DEFAULT_SONNET_MODEL=glm-4.7
export ANTHROPIC_DEFAULT_OPUS_MODEL=glm-4.7
ENV_EOF

echo "✅ 环境变量已添加到 ~/.bashrc"
echo ""
echo "📋 下一步操作:"
echo "   1. 编辑 ~/.bashrc，将 '你的token' 替换为实际的 ANTHROPIC_AUTH_TOKEN"
echo "   2. 重新加载配置:"
echo "      source ~/.bashrc"
echo "   3. 验证环境变量:"
echo "      echo \$ANTHROPIC_AUTH_TOKEN"
echo ""
echo "💡 提示: 也可以在 Windows 系统环境变量中设置（所有程序生效）"
EOF

chmod +x "$PACKAGE_DIR/setup-env.sh"
echo "   ✓ setup-env.sh 已创建"

# 创建快速开始指南
cat > "$PACKAGE_DIR/QUICKSTART.md" <<'EOF'
# 快速开始

## Mac 上（5 分钟）

```bash
# 1. 导出配置
chmod +x export-mac-config.sh
./export-mac-config.sh

# 2. 将生成的文件传输到 Windows:
#    - claude-mac-backup.tar.gz
#    - claude-windows-migration.tar.gz
```

## Windows 上（10 分钟）

```bash
# 1. 解压 claude-windows-migration.tar.gz

# 2. 打开 Git Bash，进入解压目录

# 3. 导入配置
chmod +x *.sh
./import-windows-config.sh ~/claude-mac-backup.tar.gz

# 4. 验证
./verify-config.sh

# 5. 设置环境变量
./setup-env.sh
# 然后编辑 ~/.bashrc 填入正确的 token

# 6. 重启 Claude Code
```

## 完成！

配置已迁移，Claude Code 应该可以正常使用了。
EOF

echo "   ✓ QUICKSTART.md 已创建"

# 设置所有脚本执行权限
chmod +x "$PACKAGE_DIR"/*.sh

# 打包
echo ""
echo "📦 打包中..."
cd "$HOME"
tar -czf "$PACKAGE_FILE" claude-windows-migration/

# 获取文件大小
if [ -f "$PACKAGE_FILE" ]; then
  FILE_SIZE=$(ls -lh "$PACKAGE_FILE" | awk '{print $5}')
  FILE_PATH=$(realpath "$PACKAGE_FILE" 2>/dev/null || echo "$PACKAGE_FILE")
else
  FILE_SIZE="未知"
  FILE_PATH="$PACKAGE_FILE"
fi

# 清理临时目录
rm -rf "$PACKAGE_DIR"

echo ""
echo "======================"
echo "✅ 迁移包创建完成!"
echo "======================"
echo ""
echo "📦 迁移包: $FILE_PATH"
echo "📦 文件大小: $FILE_SIZE"
echo ""
echo "📋 使用方法:"
echo ""
echo "   [Mac 上]"
echo "   1. 运行导出脚本:"
echo "      ~/.claude/migrations/export-mac-config.sh"
echo ""
echo "   2. 传输以下文件到 Windows:"
echo "      - ~/claude-mac-backup.tar.gz"
echo "      - $FILE_PATH"
echo ""
echo "   [Windows 上]"
echo "   3. 解压 $PACKAGE_FILE"
echo "   4. 阅读并按照 README.md 或 QUICKSTART.md 操作"
echo ""
