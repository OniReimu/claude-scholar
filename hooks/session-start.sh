#!/usr/bin/env bash
# SessionStart Hook: 显示项目状态（使用共享库）
set -euo pipefail

# 加载共享函数
source "$(dirname "$0")/lib/common.sh"

# 加载跨平台兼容函数
if [ -f "$(dirname "$0")/lib/platform.sh" ]; then
    source "$(dirname "$0")/lib/platform.sh"
fi

input=$(cat)
cwd=$(echo "$input" | jq -r '.cwd')
project_name=$(basename "$cwd")

# 开始构建输出（使用真正的换行符）
output=$'🚀 '"$project_name"$' 会话已启动\n'
output+=$'▸ 时间: '"$(date '+%Y/%m/%d %H:%M:%S')"$'\n'
output+=$'▸ 目录: '"$cwd"$'\n\n'

# 使用共享函数获取 Git 状态
git_info=$(get_git_info "$cwd")
is_repo=$(echo "$git_info" | jq -r '.is_repo')

if [ "$is_repo" = "true" ]; then
  branch=$(echo "$git_info" | jq -r '.branch')
  output+=$'▸ Git 分支: '"$branch"$'\n\n'

  has_changes=$(echo "$git_info" | jq -r '.has_changes')
  if [ "$has_changes" = "true" ]; then
    count=$(echo "$git_info" | jq -r '.changes_count')
    output+=$'⚠️  未提交变更 ('"$count"$' 个文件):\n'

    # 显示变更列表
    changes=$(git -C "$cwd" status --porcelain 2>/dev/null || true)
    echo "$changes" | head -10 | while read -r line; do
      status=$(echo "$line" | cut -c1-2 | tr -d ' ')
      file=$(echo "$line" | cut -c4-)
      case "$status" in
        M) output+=$'  📝 '"$file"$'\n' ;;
        A) output+=$'  ➕ '"$file"$'\n' ;;
        D) output+=$'  ❌ '"$file"$'\n' ;;
        R) output+=$'  🔄 '"$file"$'\n' ;;
        ??) output+=$'  ❓ '"$file"$'\n' ;;
        *) output+=$'  • '"$file"$'\n' ;;
      esac
    done
    if [ "$count" -gt 10 ]; then
      output+=$'  ... (还有 '"$((count - 10))"$' 个文件)\n'
    fi
  else
    output+=$'✅ 工作区干净\n'
  fi
  output+=$'\n'
else
  output+=$'▸ Git: 非仓库\n\n'
fi

# 使用共享函数获取待办事项
output+=$'📋 待办事项:\n'
todo_info=$(get_todo_info "$cwd")
todo_found=$(echo "$todo_info" | jq -r '.found')

if [ "$todo_found" = "true" ]; then
  pending=$(echo "$todo_info" | jq -r '.pending')
  done=$(echo "$todo_info" | jq -r '.done')
  output+=$'  - '"$pending"$' 未完成 / '"$done"$' 已完成\n'

  # 显示前 5 个未完成事项
  cwd_esc=$(echo "$cwd" | sed 's/"/\\"/g')
  todo_file=$(echo "$todo_info" | jq -r '.file')

  # 查找实际的 todo 文件路径
  for possible in "$cwd/docs/todo.md" "$cwd/TODO.md" "$cwd/.claude/todos.md"; do
    if [ "$(basename "$possible")" = "$todo_file" ] && [ -f "$possible" ]; then
      pending_items=$(grep '^- \[ \]' "$possible" 2>/dev/null | head -5 || true)
      if [ -n "$pending_items" ]; then
        output+=$'\n  最近待办:\n'
        echo "$pending_items" | while read -r item; do
          content=$(echo "$item" | sed 's/^- \[ \] //' | cut -c1-60)
          output+=$'  - '"$content"$'\n'
        done
      fi
      break
    fi
  done
else
  output+=$'  未找到待办事项文件 (TODO.md, docs/todo.md 等)\n'
fi

output+=$'\n'

# 获取已启用的插件
output+=$'🔌 已启用插件:\n'

settings_file="$HOME/.claude/settings.json"
enabled_plugins_list=$(jq -r '.enabledPlugins | to_entries[] | select(.value == true) | .key' "$settings_file" 2>/dev/null || true)

if [ -n "$enabled_plugins_list" ]; then
  while read -r plugin; do
    # 格式化插件名称：superpowers@superpowers-marketplace -> superpowers
    plugin_name=$(echo "$plugin" | cut -d'@' -f1)
    # 获取版本信息（如果可用）
    plugin_version=$(jq -r ".plugins[\"$plugin\"][0].version // \"\"" "$HOME/.claude/plugins/installed_plugins.json" 2>/dev/null || echo "")
    if [ -n "$plugin_version" ]; then
      output+=$'  - '"$plugin_name"$' (v'"$plugin_version"$')\n'
    else
      output+=$'  - '"$plugin_name"$'\n'
    fi
  done < <(echo "$enabled_plugins_list")
else
  output+=$'  无\n'
fi

output+=$'\n'

# 动态获取快捷命令
output+=$'💡 可用命令:\n'

# 获取命令的函数（带缓存）
get_commands() {
  local settings_file="$HOME/.claude/settings.json"
  local cache_file="$HOME/.claude/cache/.commands_cache"
  local cache_max_age=3600  # 缓存有效期（秒）
  local tmp_seen=$(mktemp_file "claude-seen")  # 临时文件用于去重

  # 检查缓存是否有效
  local use_cache=false
  if [ -f "$cache_file" ]; then
    local cache_age=$(($(date +%s) - $(stat_mtime "$cache_file")))
    if [ "$cache_age" -lt "$cache_max_age" ]; then
      use_cache=true
    fi
  fi

  # 如果缓存有效，直接返回
  if [ "$use_cache" = "true" ]; then
    cat "$cache_file"
    rm -f "$tmp_seen"
    return
  fi

  # 创建缓存目录
  mkdir -p "$(dirname "$cache_file")"

  # 构建缓存内容
  local cache_content=""

  # 获取已启用的插件列表
  if [ -f "$settings_file" ]; then
    enabled_plugins=$(jq -r '.enabledPlugins | keys[]' "$settings_file" 2>/dev/null || true)

    for plugin in $enabled_plugins; do
      # 处理插件名称，转换为路径格式
      # superpowers@claude-plugins-official -> superpowers
      plugin_name=$(echo "$plugin" | cut -d'@' -f1)

      # 可能的命令目录路径（优先使用 cache 版本）
      command_dirs=(
        "$HOME/.claude/plugins/cache/claude-plugins-official/$plugin_name"
        "$HOME/.claude/plugins/marketplaces/claude-plugins-official/plugins/$plugin_name"
      )

      for plugin_base in "${command_dirs[@]}"; do
        # 首先检查直接的 commands 目录（如 superpowers/4.0.3/commands/）
        for possible_dir in "$plugin_base/commands/" "$plugin_base"/*/commands/; do
          if [ -d "$possible_dir" ]; then
            cmd_dir="$possible_dir"
            # 遍历命令文件
            for cmd_file in "$cmd_dir"*.md; do
              if [ -f "$cmd_file" ]; then
                cmd_name=$(basename "$cmd_file" .md)

                # 检查是否已处理过该命令（去重）
                if ! grep -qx "$cmd_name" "$tmp_seen" 2>/dev/null; then
                  echo "$cmd_name" >> "$tmp_seen"

                  # 尝试从 frontmatter 获取 description
                  desc=$(sed -n '/^---$/,/^---$/p' "$cmd_file" | grep '^description:' | sed 's/description: *"\(.*\)"$/\1/' | sed 's/description: *\(.*\)$/\1/' | head -1 || echo "")

                  if [ -z "$desc" ]; then
                    desc=$(grep -E '^#+' "$cmd_file" | head -1 | sed 's/^#+ *//' | cut -c1-50 || echo "")
                  fi

                  # 如果 description 还是空的，使用默认值
                  if [ -z "$desc" ]; then
                    desc="$plugin_name 命令"
                  fi

                  # 截断过长的描述（保留中文完整性）
                  desc=$(echo "$desc" | cut -c1-40)
                  if [ ${#desc} -ge 40 ]; then
                    desc="${desc}..."
                  fi

                  local line="| /$cmd_name | $desc |"
                  echo "$line"
                  cache_content+="$line"$'\n'
                fi
              fi
            done
            break  # 找到第一个有效目录后跳出，避免重复扫描
          fi
        done
      done
    done
  fi

  # 保存到缓存
  if [ -n "$cache_content" ]; then
    echo "$cache_content" > "$cache_file"
  fi

  rm -f "$tmp_seen"
}

# 获取命令列表
cmd_list=$(get_commands)

if [ -n "$cmd_list" ]; then
  # 将表格格式转换为列表格式（使用临时文件避免子shell问题）
  tmp_file=$(mktemp_file "claude-cmd")
  echo "$cmd_list" | while IFS='|' read -r _ cmd desc _; do
    # 去除首尾空格
    cmd=$(echo "$cmd" | xargs)
    desc=$(echo "$desc" | xargs)
    if [ -n "$cmd" ]; then
      echo '  '"$cmd"$'  '"$desc"
    fi
  done > "$tmp_file"
  output+=$(cat "$tmp_file")
  rm -f "$tmp_file"
  output+=$'\n'
else
  output+=$'未找到可用命令\n\n'
fi

# 输出 JSON（使用 jq 安全处理特殊字符）
jq -n --arg msg "$output" '{"continue":true,"systemMessage":$msg}'

exit 0
