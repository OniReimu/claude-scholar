---
name: using-claude-scholar
description: |
  This skill should be used when starting any conversation or before responding to any user message, to enforce mandatory skill evaluation (Yes/No) and activation before implementation.
version: 1.0.0
tags: [Meta, System, Skills]
---

# Using Claude Scholar

You are equipped with **Claude Scholar**, a comprehensive skill system for academic research and software development. This meta-skill ensures you use the system correctly.

## Policy Rules

> 论文写作规则统一定义在 `policy/rules/`（单一真相源）。
> 各 skill 通过 HTML 注释标记引用规则。**冲突时以 `policy/rules/` 为准。**
> 领域+会议组合见 `policy/profiles/`（如 `security-neurips.md`）。

| Rule ID | 摘要 |
|---------|------|
| `FIG.NO_IN_FIGURE_TITLE` | 图内不加标题 |
| `FIG.FONT_GE_24PT` | 图表字号 ≥ 24pt |
| `FIG.ONE_FILE_ONE_FIGURE` | 1 文件 = 1 图 |
| `FIG.VECTOR_FORMAT_REQUIRED` | 数据图用矢量格式 |
| `FIG.COLORBLIND_SAFE_PALETTE` | 色盲安全配色 |
| `FIG.SELF_CONTAINED_CAPTION` | Caption三要素 |
| `LATEX.EQ.DISPLAY_STYLE` | Display 公式用 equation 环境 |
| `LATEX.VAR.LONG_TOKEN_USE_TEXT` | 长变量名用 \text{} |
| `LATEX.NOTATION_CONSISTENCY` | 符号全文一致 |
| `LATEX.CMARK_XMARK_PMARK_MACROS` | 定性比较表统一 cmark/xmark/pmark 宏 |
| `REF.CROSS_REFERENCE_STYLE` | 交叉引用用 \ref 命令 |
| `PAPER.CONCLUSION_SINGLE_PARAGRAPH` | Conclusion 单段落 |
| `PAPER.SECTION_HEADINGS_MAX_6` | 顶级section≤6 |
| `EXP.TAKEAWAY_BOX` | 实验结果附 takeaway box |
| `EXP.ERROR_BARS_REQUIRED` | 实验需误差线 |
| `EXP.ABLATION_IN_RESULTS` | 消融实验在Results |
| `EXP.RESULTS_SUBSECTION_STRUCTURE` | 实验小节结构 |
| `TABLE.BOOKTABS_FORMAT` | 使用 booktabs 格式 |
| `TABLE.DIRECTION_INDICATORS` | 表头方向指示符 |
| `CITE.VERIFY_VIA_API` | 引文API验证 |
| `BIBTEX.CONSISTENT_CITATION_KEY_FORMAT` | BibTeX key格式统一 |
| `REPRO.RANDOM_SEED_DOCUMENTATION` | 随机种子文档 |
| `REPRO.COMPUTE_RESOURCES_DOCUMENTED` | 计算资源文档 |
| `PROSE.INTENSIFIERS_ELIMINATION` | 删除空洞强调词 |
| `PROSE.EM_DASH_RESTRICTION` | 限制em-dash |
| `SUBMIT.SECTION_NUMBERING_CONSISTENCY` | Section编号一致 |
| `SUBMIT.PAGE_LIMIT_STRICT` | 严格页数限制 |
| `ETHICS.LIMITATIONS_SECTION_MANDATORY` | 必须Limitations节 |
| `ANON.DOUBLE_BLIND_ANONYMIZATION` | 双盲匿名检查 |

## The #1 Rule

**Before responding to any user message, you MUST evaluate all available skills.**

This is not optional. This is not a suggestion. This is a hard requirement.

## Skill Evaluation Procedure

### Step 1: Scan

For every user message, scan the available skills list.

**Source of truth (priority order):**
1. **Runtime-provided list** (e.g., a hook/system instruction already printed an "Available skills:" list)
2. **This repo's plugin skills** (`skills/` directory)
3. **User-installed skills** (if the runtime exposes them)

If the runtime already provided a skills list, do NOT treat the hardcoded list below as exhaustive.

**Research & Analysis:**
- `research-ideation` — Research startup: 5W1H brainstorming, literature review, gap analysis
- `results-analysis` — Experiment analysis: statistical testing, visualization, ablation studies
- `citation-verification` — Multi-layer citation validation (Format → API → Info → Content)
- `daily-paper-generator` — Automated daily paper generation for research tracking
- `paper-figure-generator` — **AUTO-ACTIVATE during paper writing**: Generate editable SVG figures via AutoFigure-Edit when writing Figure 1, system model, method overview, or any section describing a system/pipeline/architecture. Also triggers on explicit requests for diagrams.

**Paper Writing & Publication:**
- `ml-paper-writing` — Full paper writing for NeurIPS, ICML, ICLR, ACL, AAAI, COLM, Nature, Science, Cell, PNAS
- `writing-anti-ai` — Remove AI writing patterns, add human voice (bilingual EN/CN)
- `paper-self-review` — multi-item quality checklist before submission (including figure/title and LaTeX math conformance)
- `review-response` — Systematic rebuttal writing with tone management
- `post-acceptance` — Conference preparation: presentations, posters, promotion
- `doc-coauthoring` — Document collaboration workflow
- `latex-conference-template-organizer` — LaTeX template cleanup for Overleaf

**Development:**
- `daily-coding` — Daily coding checklist (auto-triggers on any code modification)
- `git-workflow` — Git best practices: Conventional Commits, branching strategy
- `code-review-excellence` — Code review: security, performance, maintainability
- `bug-detective` — Debugging for Python, Bash, JS/TS error patterns
- `architecture-design` — ML project patterns: Factory, Registry, config-driven
- `verification-loop` — Testing and validation cycles

**Plugin Development:**
- `skill-development` — Skill creation guide
- `skill-quality-reviewer` — 4-dimension quality assessment
- `skill-improver` — Skill optimization and improvement
- `command-development` — Slash command creation
- `command-name` — Plugin structure and naming
- `agent-identifier` — Agent development configuration
- `hook-development` — Hook event handling
- `mcp-integration` — MCP server integration

**Tools & Utilities:**
- `planning-with-files` — Markdown-based planning and progress tracking
- `uv-package-manager` — Modern Python package management with uv
- `webapp-testing` — Local web application testing
- `kaggle-learner` — Learn from Kaggle competition solutions

**Web Design:**
- `frontend-design` — Production-grade frontend interfaces
- `ui-ux-pro-max` — UI/UX design intelligence (50+ styles, 97 palettes)
- `web-design-reviewer` — Visual inspection and design issue fixing
- `using-claude-scholar` — Meta-skill: enforce skill evaluation + activation discipline

### Step 2: Decide

For each skill, make a **Yes/No decision**. There is no "maybe".

- If the answer is Yes (even 1% likely), activate the skill
- If multiple skills apply, activate all of them

**Required output (must appear in your response, before implementation):**
- `[skill] - Yes/No - one-line reason`

### Step 3: Activate

Read the relevant SKILL.md file(s) before proceeding with your response.

**Fallback rules (do not get stuck):**
- If a skill is missing/unreadable: mark it as "No" with reason "unavailable", then proceed with next-best relevant skill(s).
- If multiple skills conflict: follow process/discipline skills first, then domain skills; state the chosen precedence in one line.

## Red Flags Table

If you catch yourself thinking any of these, **STOP** and re-evaluate:

| Thought | Reality |
|---------|---------|
| "No skill applies here" | Re-read the list. `daily-coding` alone covers most coding tasks. |
| "I can handle this without skills" | Skills contain project-specific conventions beyond your training data. |
| "This is too simple" | Even simple tasks benefit from checklists. |
| "I already know this" | Your training data doesn't include this project's specific patterns. |
| "The user didn't ask for a skill" | Skill evaluation is automatic, not user-initiated. |

## Available Agents

When a task requires delegation, use these specialized agents:

**Research:** literature-reviewer, data-analyst, rebuttal-writer, paper-miner, kaggle-miner
**Development:** architect, build-error-resolver, code-reviewer, refactor-cleaner, tdd-guide, bug-analyzer, dev-planner
**Design:** ui-sketcher, story-generator
