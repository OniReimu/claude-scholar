# Policy Engine

论文写作规则注册中心。规则只定义一次，技能只引用不复制。

## ⚠️ MANDATORY: Author Style Guide

**所有论文写作任务开始前，必须先读 `policy/style-guide.md`。**
这是作者的个人写作风格指纹（基于 Pre-GPT 时期论文分析），与 `policy/rules/` 同级权威。
跳过 style-guide 写出的文字不是作者的风格。

## 权威定义优先级

```
policy/style-guide.md (整体性写作风格) ≡ policy/rules/ (单条可判定规则) > CLAUDE.md/AGENTS.md (指引入口) > skills/*/SKILL.md (上下文引用)
```

**`style-guide.md` 和 `rules/` 是同级权威，二者缺一不可。**

区分标准：
- **`style-guide.md`** — 整体性写作风格身份：偏好动词、句式模板、段落组织、叙事逻辑。写作时整体浸入，无法拆成单条 pass/fail
- **`rules/`** — 单条可判定的规则（1 rule = 1 file）：每条有明确的 pass/fail 判定标准，可用 regex 或 LLM 逐条检查

## 目录结构

- **`style-guide.md`** — ⚠️ 整体性写作风格身份（写作时浸入，不可拆分为单条判定）
- `rules/` — 单条可判定的规则卡片（每条有 pass/fail 标准，1 rule = 1 file）
- `profiles/` — 领域+会议组合（加载规则集 + 覆盖参数）
- `README.md` — Rule Card 规范、Phase 词汇表、Rule ID Registry

## 使用方式

LLM 直读 `policy/` 目录，无需运行时解析脚本。

- **写作前**: 必须加载 `style-guide.md`（整体风格浸入）
- **写作中**: 同时遵循 `rules/` 中的逐条规则
- **写作后**: 用 `rules/` 做 self-review 逐条检查

技能文件通过以下标记引用：
- 规则引用：`<!-- policy:RULE_ID -->`（如 `<!-- policy:FIG.NO_IN_FIGURE_TITLE -->`）
- 风格引用：`<!-- style:author-voice -->`（指向 `style-guide.md`）

<claude-mem-context>

</claude-mem-context>