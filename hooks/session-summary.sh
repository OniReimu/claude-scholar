#!/usr/bin/env bash
# SessionEnd Hook: 合并版 - 工作日志 + 智能建议
set -eo pipefail

# 加载共享函数
source "$(dirname "$0")/lib/common.sh"

input=$(cat)
cwd=$(echo "$input" | jq -r '.cwd')
session_id=$(echo "$input" | jq -r '.session_id')
transcript_path=$(echo "$input" | jq -r '.transcript_path // ""')

# 创建工作日志目录
log_dir="$cwd/.claude/logs"
mkdir -p "$log_dir"
log_file="$log_dir/session-$(date +%Y%m%d)-${session_id:0:8}.md"

# 获取项目信息
project_name=$(basename "$cwd")

# 构建输出（同时保存文件和显示建议）
{
  echo "# 📝 工作日志 - $project_name"
  echo ""
  echo "**会话 ID**: $session_id"
  echo "**时间**: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "**目录**: $cwd"
  echo ""

  # Git 变更统计
  echo "## 📊 本次会话变更"
  git_info=$(get_git_info "$cwd")
  is_repo=$(echo "$git_info" | jq -r '.is_repo')

  if [ "$is_repo" = "true" ]; then
    branch=$(echo "$git_info" | jq -r '.branch')
    echo "**分支**: $branch"
    echo ""
    echo '```'
    git -C "$cwd" status --short 2>/dev/null || echo "无变更"
    echo '```'

    # 变更统计
    changes_details=$(get_changes_details "$cwd")
    added=$(echo "$changes_details" | jq -r '.added')
    modified=$(echo "$changes_details" | jq -r '.modified')
    deleted=$(echo "$changes_details" | jq -r '.deleted')

    echo ""
    echo "| 类型 | 数量 |"
    echo "|------|------|"
    echo "| 新增 | $added |"
    echo "| 修改 | $modified |"
    echo "| 删除 | $deleted |"
  else
    echo "非 Git 仓库"
  fi
  echo ""

  # 智能建议（原 Stop 功能）
  has_changes=$(echo "$git_info" | jq -r '.has_changes')
  if [ "$has_changes" = "true" ]; then
    echo "## 💡 建议操作"
    echo ""

    # 按文件类型分析
    type_analysis=$(analyze_changes_by_type "$cwd")
    test_files=$(echo "$type_analysis" | jq -r '.test_files')
    docs_files=$(echo "$type_analysis" | jq -r '.docs_files')
    sql_files=$(echo "$type_analysis" | jq -r '.sql_files')
    config_files=$(echo "$type_analysis" | jq -r '.config_files')
    service_files=$(echo "$type_analysis" | jq -r '.service_files')

    # 生成建议
    [ "$modified" != "0" ] || [ "$added" != "0" ] && echo "- 使用代码审查工具检查修改" || true
    [ "$test_files" != "0" ] && echo "- 有测试文件变更，记得运行测试套件" || true
    [ "$docs_files" != "0" ] && echo "- 文档已更新，确保与代码同步" || true
    [ "$sql_files" != "0" ] && echo "- SQL 文件有变更，确保更新所有数据库脚本" || true
    [ "$service_files" != "0" ] && echo "- 新增了 Service/Controller，记得更新 API 文档" || true
    [ "$config_files" != "0" ] && echo "- 配置文件已修改，检查是否需要更新环境变量" || true
    echo ""
  fi

  # 读取 transcript 提取关键操作（如果可用）
  if [ -n "$transcript_path" ] && [ -f "$transcript_path" ]; then
    echo "## 🔧 主要操作"
    echo ""
    grep -o "Tool used: [A-Z][a-z]*" "$transcript_path" 2>/dev/null | \
      sort | uniq -c | sort -rn | head -10 | \
      awk '{print "| " $2 " | " $1 " 次 |"}' || echo "| 无操作记录 |"
    echo ""
  fi

  # 下次继续建议
  echo "## 🎯 下次继续"
  echo ""

  # Git 提交建议
  if [ "$has_changes" = "true" ]; then
    echo "- ⚠️ 有未提交变更，建议先提交代码："
    echo '  ```bash'
    echo '  git add . && git commit -m "feat: xxx"'
    echo '  ```'
  else
    echo "- ✅ 工作区干净，可以开始新任务"
  fi

  # 待办事项提醒
  todo_info=$(get_todo_info "$cwd")
  todo_found=$(echo "$todo_info" | jq -r '.found')
  if [ "$todo_found" = "true" ]; then
    pending=$(echo "$todo_info" | jq -r '.pending')
    todo_file=$(echo "$todo_info" | jq -r '.file')
    echo "- 更新待办事项: $todo_file ($pending 个未完成)"
  fi

  echo "- 查看上下文快照: \`cat .claude/session-context-*.md\`"
  echo ""

  # 清理临时文件（仅 macOS）
  if [ "$(uname)" = "Darwin" ]; then
    nul_count=$(find "$cwd" -name "nul" -type f 2>/dev/null | wc -l | tr -d ' ' || true)
    if [ -n "$nul_count" ] && [ "$nul_count" != "0" ]; then
      echo "- 清理临时文件: 发现 $nul_count 个 nul 文件"
    fi
  fi

  echo ""

} > "$log_file"

# 构建显示给用户的消息
display_msg="\\n---\\n"
display_msg+="✅ **会话结束** | 工作日志已保存\\n\\n"
display_msg+="**本次变更**: "

if [ "$is_repo" = "true" ]; then
  if [ "$has_changes" = "true" ]; then
    count=$(echo "$git_info" | jq -r '.changes_count')
    display_msg+="$count 个文件\\n\\n"
    display_msg+="**建议操作**:\\n"
    display_msg+="- 查看日志: \`cat .claude/logs/$(basename "$log_file")\`\\n"
    display_msg+="- 提交代码: \\`git add . && git commit -m \"feat: xxx\"\\`\\n"
  else
    display_msg+="无\\n\\n工作区干净 ✅\\n"
  fi
else
  display_msg+="非 Git 仓库\\n"
fi

display_msg+="\\n---"

echo "{\"continue\":true,\"systemMessage\":\"$display_msg\"}"

exit 0
