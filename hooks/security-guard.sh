#!/usr/bin/env bash
# PreToolUse Hook: 安全防护层
set -eo pipefail

input=$(cat)
tool_name=$(echo "$input" | jq -r '.tool_name')
cwd=$(echo "$input" | jq -r '.cwd')

# 默认允许
decision="allow"
reason=""
system_message=""

# === Bash 命令安全检查 ===
if [ "$tool_name" = "Bash" ]; then
  command=$(echo "$input" | jq -r '.tool_input.command // ""')

  # 危险命令黑名单（使用函数检查而非关联数组）
  is_dangerous() {
    local cmd="$1"
    # 删除操作
    echo "$cmd" | grep -qE "rm[[:space:]]-rf[[:space:]]+/" && return 0
    echo "$cmd" | grep -qE "rm[[:space:]]--no-preserve-root[[:space:]]+-rf[[:space:]]+/" && return 0
    echo "$cmd" | grep -qE "dd[[:space:]]+if=/dev/(zero|random)" && return 0
    echo "$cmd" | grep -qE ">(>?)?[[:space:]]*/dev/sd" && return 0
    echo "$cmd" | grep -qE ">(>?)?[[:space:]]*/dev/nvme" && return 0
    echo "$cmd" | grep -qE ">(>?)?[[:space:]]*/dev/vda" && return 0
    # 格式化
    echo "$cmd" | grep -qE "mkfs\." && return 0
    echo "$cmd" | grep -qE "format[[:space:]]" && return 0
    # 数据库删除
    echo "$cmd" | grep -qiE "DROP[[:space:]]+(DATABASE|TABLE)" && return 0
    echo "$cmd" | grep -qiE "DELETE[[:space:]]+FROM" && return 0
    echo "$cmd" | grep -qiE "TRUNCATE[[:space:]]+TABLE" && return 0
    # 系统配置
    echo "$cmd" | grep -qE "rm[[:space:]]+-rf?[[:space:]]+/(etc|usr|bin|sbin)" && return 0
    # 用户数据
    echo "$cmd" | grep -qE "rm[[:space:]]+-rf[[:space:]]+/home/" && return 0
    echo "$cmd" | grep -qE "rm[[:space:]]+-rf[[:space:]]+/Users/" && return 0
    return 1
  }

  # 警告模式检查
  check_warning() {
    local cmd="$1"
    echo "$cmd" | grep -qE "rm[[:space:]]+-[rf]" && echo "rm -" && return 0
    echo "$cmd" | grep -qE "\bmv[[:space:]]" && echo "mv" && return 0
    echo "$cmd" | grep -qE "\bcp[[:space:]]" && echo "cp" && return 0
    echo "$cmd" | grep -qE "chmod[[:space:]]+777" && echo "chmod 777" && return 0
    echo "$cmd" | grep -qE "chown[[:space:]]" && echo "chown" && return 0
    echo "$cmd" | grep -qE "(wget|curl)[[:space:]]" && echo "网络下载" && return 0
    echo "$cmd" | grep -qE "(pip|npm|yarn|brew|apt-get|yum)[[:space:]]+install" && echo "软件安装" && return 0
    echo "$cmd" | grep -qE "sudo[[:space:]]+(apt-get|yum)" && echo "sudo 安装" && return 0
    return 1
  }

  # 检查危险命令
  if is_dangerous "$command"; then
    decision="deny"
    reason="检测到危险命令"
  fi

  # 警告级别检查（敏感操作）
  if [ "$decision" = "allow" ]; then
    warning_pattern=$(check_warning "$command" || true)
    if [ -n "$warning_pattern" ]; then
      system_message="⚠️ 安全提醒: 正在执行敏感操作 ($warning_pattern)"
    fi
  fi

# === 文件写入安全检查 ===
elif [ "$tool_name" = "Write" ] || [ "$tool_name" = "Edit" ]; then
  file_path=$(echo "$input" | jq -r '.tool_input.file_path // ""')

  # 敏感路径黑名单
  sensitive_paths=(
    "/etc/"
    "/usr/bin/"
    "/usr/sbin/"
    "/bin/"
    "/sbin/"
    "/System/"
    "/dev/"
    "/proc/"
    "/sys/"
  )

  for path in "${sensitive_paths[@]}"; do
    if [[ "$file_path" == "$path"* ]]; then
      decision="deny"
      reason="禁止写入系统路径: $path"
      break
    fi
  done

  # 检查敏感文件
  sensitive_files=(
    ".env"
    ".env.local"
    ".env.production"
    "credentials.json"
    "key.pem"
    "key.json"
    "id_rsa"
    ".aws/credentials"
    ".npmrc"
  )

  if [ "$decision" = "allow" ]; then
    for file in "${sensitive_files[@]}"; do
      if [[ "$(basename "$file_path")" == "$file" ]]; then
        system_message="⚠️ 安全提醒: 正在修改敏感文件 ($file)"
        break
      fi
    done
  fi

  # 检查路径遍历攻击
  if [[ "$file_path" == *".."* ]]; then
    decision="deny"
    reason="检测到路径遍历攻击"
  fi

  # 检查绝对路径注入
  if [[ "$file_path" == *"~/"* ]] && [[ "$file_path" != "$cwd"* ]]; then
    system_message="⚠️ 路径提醒: 文件路径不在项目目录内"
  fi
fi

# === 构建输出 ===
if [ "$decision" = "deny" ]; then
  # 阻止执行
  echo "{\"hookSpecificOutput\":{\"permissionDecision\":\"deny\"},\"systemMessage\":\"🛑 安全拦截: $reason\\n\\n如需执行此操作，请手动在终端运行。\"}" >&2
  exit 2
else
  # 允许执行（可选警告消息）
  if [ -n "$system_message" ]; then
    echo "{\"continue\":true,\"systemMessage\":\"$system_message\"}"
  else
    echo "{\"continue\":true}"
  fi
  exit 0
fi
