---
name: create_project
description: 创建新项目，基于模板初始化、配置 uv 和 Git
arguments:
  - name: project_name
    description: 项目名称
    required: true
  - name: path
    description: 项目路径（默认为 ~/Code/）
    required: false
---

# 创建新项目

此命令基于模板创建新项目，包含以下步骤：
1. 复制模板文件
2. 替换项目名称
3. 初始化 uv 项目
4. 配置 Git 仓库和分支策略
5. 创建初始 tag
6. 初始化 GitHub 远程仓库（可选）

```bash
# 解析参数
PROJECT_NAME="{{project_name}}"
PROJECT_PATH="${path:-$HOME/Code}"
FULL_PATH="$PROJECT_PATH/$PROJECT_NAME"

TEMPLATE_PATH="$HOME/Code/template"
INITIAL_TAG="v0.1.0"

echo "🚀 创建新项目: $PROJECT_NAME"
echo "📁 路径: $FULL_PATH"
echo ""

# 检查模板目录
if [ ! -d "$TEMPLATE_PATH" ]; then
  echo "❌ 错误: 模板目录不存在: $TEMPLATE_PATH"
  exit 1
fi

# 检查目标目录是否已存在
if [ -d "$FULL_PATH" ]; then
  echo "❌ 错误: 目录已存在: $FULL_PATH"
  exit 1
fi

# 1. 创建项目目录
echo "📂 创建项目目录..."
mkdir -p "$FULL_PATH"

# 2. 复制模板文件（排除 .git、.idea、.DS_Store 等）
echo "📋 复制模板文件..."
rsync -av --exclude='.git' \
          --exclude='.idea' \
          --exclude='.DS_Store' \
          --exclude='__pycache__' \
          --exclude='*.pyc' \
          --exclude='outputs/*' \
          --exclude='data/*' \
          "$TEMPLATE_PATH/" "$FULL_PATH/"

# 3. 替换项目名称
echo "✏️  替换项目名称..."
cd "$FULL_PATH"

# 替换 README.md 第一行（如果是示例标题）
if [ -f "README.md" ]; then
  # 检查第一行是否是 # 开头的标题
  FIRST_LINE=$(head -n 1 README.md)
  if [[ "$FIRST_LINE" == "#"* ]]; then
    # 替换第一行为项目名称
    echo "# $PROJECT_NAME" > README.md.new
    tail -n +2 README.md >> README.md.new
    mv README.md.new README.md
    echo "   ✓ 更新 README.md 标题"
  fi
fi

# 替换 pyproject.toml 中的项目名称（如果存在）
if [ -f "pyproject.toml ]; then
  sed -i.bak "s/name = \".*\"/name = \"$PROJECT_NAME\"/" pyproject.toml
  rm -f pyproject.toml.bak
  echo "   ✓ 更新 pyproject.toml 项目名"
fi

# 4. 初始化 uv 项目
echo "🔧 初始化 uv 项目..."
uv init --no-readme  # README 已从模板复制

# 5. 初始化 Git 仓库（默认在 master 分支）
echo "🔧 初始化 Git 仓库..."
git init

# 6. 初始提交在 master
echo "📝 创建初始提交..."
git add .
git commit -m "chore: 初始化项目

基于模板创建项目结构
- 配置项目结构
- 初始化 uv 依赖管理
- 设置 Git 工作流 (master/develop)
- 创建初始版本 $INITIAL_TAG"

# 7. 创建初始 tag（在 master 上）
echo "🏷️  创建初始标签: $INITIAL_TAG"
git tag -a "$INITIAL_TAG" -m "release: $INITIAL_TAG 初始版本

项目初始化完成"

# 8. 创建 develop 分支
echo "🌿 创建 develop 分支..."
git checkout -b develop

# 9. 询问是否创建 GitHub 仓库
echo ""
echo "✅ 项目创建完成！"
echo ""
echo "📍 项目位置: $FULL_PATH"
echo "🏷️  初始版本: $INITIAL_TAG"
echo ""
echo "📌 下一步操作:"
echo "   cd $FULL_PATH"
echo ""

# 询问是否创建 GitHub 远程仓库
read -p "是否创建 GitHub 远程仓库？(y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  # 检查 gh CLI 是否安装
  if ! command -v gh &> /dev/null; then
    echo "⚠️  GitHub CLI (gh) 未安装，跳过远程仓库创建"
    echo "   安装: brew install gh"
  else
    echo "🌐 创建 GitHub 远程仓库..."
    cd "$FULL_PATH"

    # 使用 gh CLI 创建仓库
    gh repo create "$PROJECT_NAME" --private --source=. --remote=origin

    # 推送分支和标签（先切换回 master）
    echo "📤 推送分支和标签到远程..."
    git checkout master
    git push -u origin master
    git push origin "$INITIAL_TAG"
    git push -u origin develop
    git checkout develop

    echo ""
    echo "✅ GitHub 仓库创建完成！"

    # 获取仓库 URL
    REPO_URL=$(git config --get remote.origin.url)
    if [[ "$REPO_URL" == "git@github.com"* ]]; then
      # SSH URL
      REPO_URL="https://github.com/$(git config --get user.name)/$PROJECT_NAME"
    fi
    echo "   👉 $REPO_URL"
  fi
else
  echo "⏭️  跳过 GitHub 仓库创建"
  echo "   稍后可手动执行:"
  echo "   cd $FULL_PATH && gh repo create $PROJECT_NAME --private --source=. --remote=origin"
fi

echo ""
echo "🎉 项目初始化完成！"
echo ""
echo "📋 Git 工作流说明:"
echo "   - master: 主分支（生产环境）- 禁止直接推送"
echo "   - develop: 开发分支"
echo "   - feature/xxx: 功能分支（从 develop 创建）"
echo "   - bugfix/xxx: Bug 修复分支（从 develop 创建）"
echo ""
echo "📚 常用命令:"
echo "   git checkout develop                    # 切换到开发分支"
echo "   git checkout -b feature/xxx             # 创建功能分支"
echo "   git checkout develop && git merge --no-ff feature/xxx  # 合并功能分支"
echo "   git tag -a v1.0.0 -m \"release: v1.0.0\" # 创建版本标签"
echo ""
echo "📦 下一步:"
echo "   cd $FULL_PATH"
echo "   uv sync                                 # 安装依赖"
echo ""
