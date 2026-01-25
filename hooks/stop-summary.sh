#!/usr/bin/env bash
# Stop Hook: 简化版 - 显示基础状态 + AI 总结提示
set -eo pipefail

# 加载共享函数
source "$(dirname "$0")/lib/common.sh"

# 读取输入
input=$(cat)
cwd=$(echo "$input" | jq -r '.cwd')
reason=$(echo "$input" | jq -r '.reason // "task_complete"')

# 构建简化消息
build_message() {
  local msg="\\n---\\n"
  msg+="✅ 会话结束\\n\\n"

  # 获取 Git 信息
  git_info=$(get_git_info "$cwd")
  is_repo=$(echo "$git_info" | jq -r '.is_repo')

  if [ "$is_repo" = "true" ]; then
    branch=$(echo "$git_info" | jq -r '.branch')
    has_changes=$(echo "$git_info" | jq -r '.has_changes')

    msg+="📁 Git 仓库\\n"
    msg+="  分支: $branch\\n"

    if [ "$has_changes" = "true" ]; then
      changes_details=$(get_changes_details "$cwd")
      added=$(echo "$changes_details" | jq -r '.added')
      modified=$(echo "$changes_details" | jq -r '.modified')
      deleted=$(echo "$changes_details" | jq -r '.deleted')
      total=$((added + modified + deleted))

      msg+="  变更: $total 个文件"
      [ "$added" != "0" ] && msg+=" (+$added)"
      [ "$modified" != "0" ] && msg+=" (~$modified)"
      [ "$deleted" != "0" ] && msg+=" (-$deleted)"
      msg+="\\n"
    else
      msg+="  状态: 干净\\n"
    fi
  else
    msg+="📁 非Git 仓库目录\\n"
  fi

  msg+="\\n"

  # 临时文件检测
  temp_info=$(detect_temp_files "$cwd")
  temp_count=$(echo "$temp_info" | jq -r '.count')

  if [ "$temp_count" != "0" ]; then
    msg+="🧹 临时文件: $temp_count 个\\n"
    # 列出所有临时文件路径
    while IFS= read -r file; do
      [ -z "$file" ] && continue
      msg+="  • $file\\n"
    done < <(echo "$temp_info" | jq -r '.files[]')
  fi

  msg+="---"

  echo "$msg"
}

# 构建并返回
system_message=$(build_message)
echo "{\"continue\":true,\"systemMessage\":\"$system_message\"}"

exit 0
