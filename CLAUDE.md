# Claude 配置

## 全局设置
- 用中文进行回答
- 不要将英文的特定名词或名称翻译成中文
- 对于复杂的任务首先提出问题和我交流意见，之后思考如何拆解任务，再进行实施，实施后需要进行示例测试
- 实施过程中不要影响现有的功能，做好备份，全部完成之后将临时文件及时删除
- 制定计划时产生的markdown存放在工作目录下的/plan文件夹，临时文件存放在/temp文件夹里面，文件夹不存在就新建
## 用户背景

### 学术背景
- 计算机科学 PhD
- 需要投稿顶会（NeurIPS, ICML, ICLR, KDD）和高影响期刊（Nature, Science, Cell, PNAS）
- 关注学术写作质量、逻辑连贯性和自然表达

## 可用 Skills

### 开发相关
- **git-workflow**: Git 工作流规范（Conventional Commits, 分支管理策略）
- **bug-detective**: 调试和错误排查（Python, Bash/Zsh, JavaScript/TypeScript）
- **code-review-excellence**: 代码审查最佳实践
- **architecture-design**: ML 项目代码框架和设计模式
- **uv-package-manager**: uv 包管理器使用

### 写作相关
- **scientific-writing**: 学术论文写作辅助
  - 顶会投稿（NeurIPS, ICML, ICLR, KDD）
  - 高影响期刊（Nature, Science, Cell, PNAS）
  - 逻辑分析、反 AI 化写作、审稿人视角润色

### Claude Code 插件开发
- **skill-development**: Skill 开发指南
- **command-development**: Slash 命令开发
- **agent-identifier**: Agent 开发
- **hook-development**: Hook 开发
- **mcp-integration**: MCP 服务器集成
- **command-name**: 插件结构和组织

### 其他
- **planning-with-files**: 使用 Markdown 文件进行规划和进度跟踪
- **doc-coauthoring**: 文档协作工作流

## 技术栈偏好

### Python 生态
- **包管理**: 使用 `uv` 进行依赖管理和虚拟环境
- **配置管理**: 使用 Hydra + OmegaConf 进行配置管理
  - 支持配置组合和覆盖
- **模型训练**: 使用 Transformers 的 Trainer 进行模型训练

### Git 规范
- **提交规范**: 遵循 Conventional Commits
  - Type: feat, fix, docs, style, refactor, perf, test, chore
  - Scope: data, model, config, trainer, utils, workflow
  - 详细规范见 `git-workflow` skill
- **分支策略**: master/develop/feature/bugfix/hotfix/release
- **合并策略**: 功能分支用 rebase 同步，用 merge --no-ff 合并

## 工作风格

### 任务管理
- 使用 TodoWrite 工具跟踪任务进度
- 复杂任务先规划再执行
- 优先使用已有的 skills 和工具

### 沟通方式
- 不确定时主动询问细节
- 重要操作前先确认再执行
- 遵循 hook 强制流程（当激活时）

### 代码风格
- Python 代码遵循 PEP 8
- 注释使用中文
- 函数和变量命名使用英文

### 任务完成后的行为
**每次任务完成或用户准备离开时，必须主动提供总结，包含以下内容：**

1. **📋 操作回顾**
   - 列出执行的主要操作
   - 修改了哪些文件
   - 使用了哪些工具/命令

2. **📊 当前状态**
   - Git 状态（如果在仓库中）：分支、变更文件数
   - 文件系统状态：临时文件、新增文件
   - 运行状态（如适用）：服务、进程等

3. **💡 下一步建议**
   - 基于实际操作的针对性建议
   - 考虑：是否需要提交代码、清理临时文件、测试验证等
   - 不要使用通用的"记得备份"建议，除非确实需要

**总结格式示例：**
```
---
📋 本次操作回顾
1. XXX
2. XXX

📊 当前状态
• XXX

💡 下一步建议
1. XXX
2. XXX
---
```
