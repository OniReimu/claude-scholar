# Task Plan: Create /create_project Command

## Goal
创建一个 `/create_project` 命令，用于快速初始化新项目，包含模板复制、uv 初始化、GitHub 配置

## Phases
- [ ] Phase 1: 理解需求和设计命令参数
- [ ] Phase 2: 探索模板结构和需要复制的文件
- [ ] Phase 3: 设计命令流程和 YAML frontmatter
- [ ] Phase 4: 创建命令文件
- [ ] Phase 5: 测试和完善

## Key Questions
1. 哪些文件/目录需要复制？
2. GitHub 配置的具体步骤是什么？
3. 是否需要交互式确认？

## Decisions Made
- 默认路径：`~/Code/`
- 使用 rsync 复制模板，排除 .git、.idea、__pycache__、outputs、data
- Git 分支策略：master (主分支) + develop (开发分支)
- 使用 gh CLI 创建 GitHub 仓库

## Status
**Currently in Phase 5** - 命令已完善，添加自动替换和 tag 功能
