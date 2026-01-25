---
name: git-workflow
description: This skill should be used when the user asks to "create git commit", "manage branches", "follow git workflow", "use Conventional Commits", "handle merge conflicts", or asks about git branching strategies, version control best practices, pull request workflows. Provides comprehensive Git workflow guidance for team collaboration.
version: 1.2.0
---

# Git 规范

本文档定义项目的 Git 使用规范，包括提交消息格式、分支管理策略、工作流程、合并策略等内容。遵循这些规范可以提高协作效率、便于追溯、支持自动化、减少冲突。

## Commit Message 规范

项目采用 **Conventional Commits** 规范：

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type 类型

| 类型 | 说明 | 示例 |
| :--- | :--- | :--- |
| `feat` | 新功能 | `feat(user): 添加用户导出功能` |
| `fix` | 修复Bug | `fix(login): 修复验证码不刷新问题` |
| `docs` | 文档更新 | `docs(api): 更新接口文档` |
| `refactor` | 重构 | `refactor(utils): 重构工具函数` |
| `perf` | 性能优化 | `perf(list): 优化列表性能` |
| `test` | 测试相关 | `test(user): 添加单元测试` |
| `chore` | 其他修改 | `chore: 更新依赖版本` |

### Subject 规范

- 使用动词开头：添加、修复、更新、移除、优化
- 不超过50个字符
- 不以句号结尾

更多详细规范和示例，参见 `references/commit-conventions.md`。

## 分支管理策略

### 分支类型

| 分支类型 | 命名规范 | 说明 | 生命周期 |
| :--- | :--- | :--- | :--- |
| master | `master` | 主分支，可发布状态 | 永久 |
| develop | `develop` | 开发分支，集成最新代码 | 永久 |
| feature | `feature/功能名` | 功能分支 | 开发完成后删除 |
| bugfix | `bugfix/问题描述` | Bug修复分支 | 修复完成后删除 |
| hotfix | `hotfix/问题描述` | 紧急修复分支 | 修复完成后删除 |
| release | `release/版本号` | 发布分支 | 发布完成后删除 |

### 分支命名示例

```
feature/user-management          # 用户管理功能
feature/123-add-export          # 关联Issue的功能
bugfix/login-error              # 登录错误修复
hotfix/security-vulnerability   # 安全漏洞修复
release/v1.0.0                  # 版本发布
```

### 分支保护规则

**master 分支：**
- 禁止直接推送
- 必须通过 Pull Request 合并
- 必须通过 CI 检查
- 必须至少一人 Code Review

**develop 分支：**
- 限制直接推送
- 建议通过 Pull Request 合并
- 必须通过 CI 检查

详细的分支策略和工作流程，参见 `references/branching-strategies.md`。

## 工作流程

### 日常开发流程

```bash
# 1. 同步最新代码
git checkout develop
git pull origin develop

# 2. 创建功能分支
git checkout -b feature/user-management

# 3. 开发并提交
git add .
git commit -m "feat(user): 添加用户列表页面"

# 4. 推送到远程
git push -u origin feature/user-management

# 5. 创建 Pull Request 并请求 Code Review

# 6. 合并到 develop（通过 PR）

# 7. 删除功能分支
git branch -d feature/user-management
git push origin -d feature/user-management
```

### 紧急修复流程

```bash
# 1. 从 master 创建修复分支
git checkout master
git pull origin master
git checkout -b hotfix/critical-bug

# 2. 修复并提交
git add .
git commit -m "fix(auth): 修复认证绕过漏洞"

# 3. 合并到 master
git checkout master
git merge --no-ff hotfix/critical-bug
git tag -a v1.0.1 -m "hotfix: 修复认证绕过漏洞"
git push origin master --tags

# 4. 同步到 develop
git checkout develop
git merge --no-ff hotfix/critical-bug
git push origin develop
```

### 版本发布流程

```bash
# 1. 创建发布分支
git checkout develop
git checkout -b release/v1.0.0

# 2. 更新版本号和文档

# 3. 提交版本更新
git add .
git commit -m "chore(release): 准备发布 v1.0.0"

# 4. 合并到 master
git checkout master
git merge --no-ff release/v1.0.0
git tag -a v1.0.0 -m "release: v1.0.0 正式版本"
git push origin master --tags

# 5. 同步到 develop
git checkout develop
git merge --no-ff release/v1.0.0
git push origin develop
```

## 合并策略

### Merge vs Rebase

| 特性 | Merge | Rebase |
| :--- | :--- | :--- |
| 历史记录 | 保留完整历史 | 线性历史 |
| 适用场景 | 公共分支 | 私有分支 |
| 推荐用法 | 合并到主分支 | 同步上游代码 |

### 使用建议

- **功能分支同步 develop**：使用 `rebase`
- **功能分支合并到 develop**：使用 `merge --no-ff`
- **develop 合并到 master**：使用 `merge --no-ff`

```bash
# ✅ 推荐: 功能分支同步 develop
git checkout feature/user-management
git rebase develop

# ✅ 推荐: 合并功能分支到 develop
git checkout develop
git merge --no-ff feature/user-management

# ❌ 不推荐: 在公共分支上 rebase
git checkout develop
git rebase feature/xxx  # 危险操作
```

**项目约定**：合并功能分支时使用 `--no-ff`，保留分支历史信息。

详细的合并策略和技巧，参见 `references/merge-strategies.md`。

## 冲突处理

### 识别冲突

```
<<<<<<< HEAD
// 当前分支的代码
const name = '张三'
=======
// 要合并的分支的代码
const name = '李四'
>>>>>>> feature/user-management
```

### 解决冲突

```bash
# 1. 查看冲突文件
git status

# 2. 手动编辑文件，解决冲突

# 3. 标记已解决
git add <file>

# 4. 完成合并
git commit  # merge 冲突
# 或
git rebase --continue  # rebase 冲突
```

### 冲突处理策略

```bash
# 保留当前分支版本
git checkout --ours <file>

# 保留传入分支版本
git checkout --theirs <file>

# 放弃合并
git merge --abort
git rebase --abort
```

### 预防冲突

1. **及时同步代码** - 每天开始工作前拉取最新代码
2. **小步提交** - 频繁提交小的改动
3. **功能模块化** - 不同功能在不同文件中实现
4. **沟通协作** - 避免同时修改同一文件

详细的冲突处理和高级技巧，参见 `references/conflict-resolution.md`。

## .gitignore 规范

### 基本规则

```
# 忽略所有 .log 文件
*.log

# 忽略目录
node_modules/

# 忽略根目录下的目录
/temp/

# 忽略所有目录下的文件
**/.env

# 不忽略特定文件
!.gitkeep
```

### 通用 .gitignore

```
node_modules/
dist/
build/
.idea/
.vscode/
.env
.env.local
logs/
*.log
.DS_Store
Thumbs.db
```

详细的 .gitignore 模式和项目特定配置，参见 `references/gitignore-guide.md`。

## 标签管理

采用 **语义化版本**（Semantic Versioning）：

```
主版本号.次版本号.修订号[-预发布标识]
MAJOR.MINOR.PATCH[-PRERELEASE]
```

### 版本变化说明

- **主版本号**：不兼容的 API 修改（v1.0.0 → v2.0.0）
- **次版本号**：向下兼容的功能新增（v1.0.0 → v1.1.0）
- **修订号**：向下兼容的问题修正（v1.0.0 → v1.0.1）

### 标签操作

```bash
# 创建附注标签（推荐）
git tag -a v1.0.0 -m "release: v1.0.0 正式版本"

# 推送标签
git push origin v1.0.0
git push origin --tags

# 查看标签
git tag
git show v1.0.0

# 删除标签
git tag -d v1.0.0
git push origin :refs/tags/v1.0.0
```

## 多人协作规范

### Pull Request 规范

创建 PR 时应包含：

```markdown
## 变更说明
<!-- 描述本次变更的内容和目的 -->

## 变更类型
- [ ] 新功能 (feat)
- [ ] Bug 修复 (fix)
- [ ] 代码重构 (refactor)

## 测试方式
<!-- 描述如何测试 -->

## 关联 Issue
Closes #xxx

## 检查清单
- [ ] 代码已自测
- [ ] 文档已更新
```

### Code Review 规范

审查要点：
- **代码质量**：清晰易读、命名规范、无重复代码
- **逻辑正确性**：业务逻辑正确、边界条件处理
- **安全性**：无安全漏洞、敏感信息保护
- **性能**：无明显性能问题、资源正确释放

详细的协作规范和最佳实践，参见 `references/collaboration.md`。

## 常见问题

### 修改最后一次提交

```bash
# 修改提交内容（未推送）
git add forgotten-file.ts
git commit --amend --no-edit

# 修改提交消息
git commit --amend -m "新的提交消息"
```

### 推送被拒绝

```bash
# 先拉取再推送
git pull origin master
git push origin master

# 使用 rebase 保持历史清晰
git pull --rebase origin master
git push origin master
```

### 回滚到之前版本

```bash
# 重置到指定提交（丢弃之后的提交）
git reset --hard abc123

# 创建反向提交（推荐，保留历史）
git revert abc123
```

### 暂存当前工作

```bash
git stash save "工作进行中"
git stash list
git stash pop
```

### 查看文件修改历史

```bash
git log -- <file>             # 提交历史
git log -p -- <file>          # 详细内容
git blame <file>              # 每行修改人
```

## 最佳实践总结

### 提交规范

✅ **推荐**：
- 使用 Conventional Commits 规范
- 提交消息清晰描述改动
- 一次提交只做一件事
- 提交前进行代码检查

❌ **禁止**：
- 提交消息模糊不清
- 一次提交多个不相关改动
- 提交敏感信息（密码、密钥）
- 直接在主分支开发

遵循这些规范可以提高协作效率、便于追溯、支持自动化、减少冲突。

### 分支管理

✅ **推荐**：
- 使用 feature 分支开发
- 定期同步主分支代码
- 功能完成后及时删除分支
- 使用 `--no-ff` 合并保留历史

❌ **禁止**：
- 在主分支直接开发
- 长期不合并的功能分支
- 分支命名不规范
- 在公共分支上 rebase

### 代码审查

✅ **推荐**：
- 所有代码通过 Pull Request
- 至少一人审核通过才能合并
- 提供建设性反馈

❌ **禁止**：
- 未经审查直接合并
- 自己审查自己的代码

## Additional Resources

### Reference Files

For detailed guidance on specific topics:

- **`references/commit-conventions.md`** - Commit message detailed conventions and examples
- **`references/branching-strategies.md`** - Comprehensive branch management strategies
- **`references/merge-strategies.md`** - Merge, rebase, and conflict resolution strategies
- **`references/conflict-resolution.md`** - Detailed conflict handling and prevention
- **`references/advanced-usage.md`** - Git performance optimization, security, submodules, and advanced techniques
- **`references/collaboration.md`** - Pull request and code review guidelines
- **`references/gitignore-guide.md`** - .gitignore patterns and project-specific configurations

### Example Files

Working examples in `examples/`:
- **`examples/commit-messages.txt`** - Good commit message examples
- **`examples/workflow-commands.sh`** - Common workflow command snippets

## Summary

本文档定义了项目的 Git 规范：

1. **Commit Message** - 采用 Conventional Commits 规范
2. **分支管理** - master/develop/feature/bugfix/hotfix/release 分支策略
3. **工作流程** - 日常开发、紧急修复、版本发布的标准流程
4. **合并策略** - 功能分支用 rebase 同步，用 merge --no-ff 合并
5. **标签管理** - 语义化版本，附注标签
6. **冲突处理** - 及时同步、小步提交、沟通协作

遵循这些规范可以提高协作效率、保证代码质量、简化版本管理。
