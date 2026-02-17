# Policy Engine

论文写作规则注册中心。规则只定义一次，技能只引用不复制。

## 权威定义优先级

```
policy/rules/ (单一真相源) > CLAUDE.md/AGENTS.md (历史权威) > skills/*/SKILL.md (历史副本)
```

## 目录结构

- `rules/` — 规则卡片（1 rule = 1 file）
- `profiles/` — 领域+会议组合（加载规则集 + 覆盖参数）
- `README.md` — Rule Card 规范、Phase 词汇表、Rule ID Registry

## 使用方式

LLM 直读 `policy/` 目录，无需运行时解析脚本。技能文件中通过 HTML 注释标记（如 `<!-- policy:FIG.NO_IN_FIGURE_TITLE -->`）引用规则。
