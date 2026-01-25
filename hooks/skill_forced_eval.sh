#!/usr/bin/env bash
set -euo pipefail

# Hook triggered on UserPromptSubmit event
# Forces skill evaluation before AI starts thinking

# 检查 jq 是否可用
if ! command -v jq >/dev/null 2>&1; then
    # jq 不可用，跳过技能评估
    echo '{"continue": true}'
    exit 0
fi

# 加载跨平台兼容函数
SCRIPT_DIR="$(dirname "$0")"
if [ -f "$SCRIPT_DIR/lib/platform.sh" ]; then
    source "$SCRIPT_DIR/lib/platform.sh"
fi

# Read JSON from stdin and extract user_prompt
input=$(cat)
user_prompt=$(echo "$input" | jq -r '.user_prompt // ""')

# Check if the input is a slash command (escape hatch)
if [[ "$user_prompt" =~ ^/ ]]; then
    # Skip skill evaluation for slash commands
    exit 0
fi

# 动态收集技能列表
collect_skills() {
    local tmp_file=$(mktemp_file "claude-skills")

    # 1. 收集本地技能 (~/.claude/skills/)
    local skills_dir="$HOME/.claude/skills"
    if [ -d "$skills_dir" ]; then
        for skill_dir in "$skills_dir"/*/; do
            if [ -d "$skill_dir" ]; then
                basename "$skill_dir" >> "$tmp_file"
            fi
        done
    fi

    # 2. 收集所有插件技能
    local plugins_cache="$HOME/.claude/plugins/cache"
    if [ -d "$plugins_cache" ]; then
        # 遍历所有 marketplace
        for marketplace_dir in "$plugins_cache"/*/; do
            if [ -d "$marketplace_dir" ]; then
                # 遍历每个插件
                for plugin_dir in "$marketplace_dir"/*/; do
                    if [ -d "$plugin_dir" ]; then
                        local plugin_name=$(basename "$plugin_dir")
                        # 跳过非插件目录（如 .DS_Store）
                        case "$plugin_name" in .*) continue;; esac

                        # 找到最新版本的插件目录
                        local latest_version=$(ls -t "$plugin_dir" 2>/dev/null | head -1)
                        if [ -n "$latest_version" ]; then
                            local skills_dir="$plugin_dir/$latest_version/skills"
                            if [ -d "$skills_dir" ]; then
                                for skill_dir in "$skills_dir"/*/; do
                                    if [ -d "$skill_dir" ]; then
                                        local skill_name=$(basename "$skill_dir")
                                        echo "$plugin_name:$skill_name" >> "$tmp_file"
                                    fi
                                done
                            fi
                        fi
                    fi
                done
            fi
        done
    fi

    # 去重并格式化输出
    local skills=""
    if [ -s "$tmp_file" ]; then
        while IFS= read -r skill; do
            skills+="- $skill"$'\n'
        done < <(sort -u "$tmp_file")
    fi
    rm -f "$tmp_file"

    echo "$skills"
}

# 生成技能列表
SKILL_LIST=$(collect_skills)

# Otherwise, inject the skill evaluation instruction
cat <<EOF
## 指令：强制技能激活流程（必须执行）

### 步骤 1 - 评估技能
针对以下每个技能，陈述：[技能名] - 是/否 - [理由]

可用技能列表：
$SKILL_LIST
### 步骤 2 - 激活
如果任何技能为"是" → 立即使用 Skill(技能) 工具激活
如果所有技能为"否" → 说明"不需要技能"并继续

### 步骤 3 - 实现
只有在步骤 2 完成后，才能开始实现。

**关键规划**：
1.你必须在步骤2调用Skill()工具，不要跳过直接实现；
2.首先评估步骤1的所有技能，不要跳过任何一个技能；
3.多个技能相关时，全部激活；
4.判断仅包含是或否：是 = 明确相关且必需，否 = 不相关或非必需，去掉"可能"选项；
5.只有完成上述步骤之后才开始实现。
EOF

# 成功退出
exit 0