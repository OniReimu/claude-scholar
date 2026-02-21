---
name: using-claude-scholar
description: |
  This skill should be used when working on academic papers, research
  projects, or scientific writing — enforces skill evaluation and
  policy rule compliance for the claude-scholar framework. Also
  provides tool mapping, session behavior, security rules, and user
  preferences for multi-runtime environments (Claude Code + Codex).
version: 2.0.0
tags: [Meta, System, Skills, Academic]
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
| `EXP.FABRICATED_RESULTS_CAPTION_DISCLOSURE` | 非实跑结果 caption 强制披露 |
| `EXP.RESULTS_STATUS_DECLARATION_REQUIRED` | 非实跑结果小节状态声明 |
| `SOK.TAXONOMY_REQUIRED` | SoK 必须给出 taxonomy |
| `SOK.METHODOLOGY_REPORTING` | SoK 报告文献筛选方法 |
| `SOK.BIG_TABLE_REQUIRED` | SoK 必须有综合对比大表 |
| `SOK.RESEARCH_AGENDA_REQUIRED` | SoK 必须给出研究议程 |
| `TABLE.BOOKTABS_FORMAT` | 使用 booktabs 格式 |
| `TABLE.DIRECTION_INDICATORS` | 表头方向指示符 |
| `CITE.VERIFY_VIA_API` | 引文API验证 |
| `BIBTEX.CONSISTENT_CITATION_KEY_FORMAT` | BibTeX key格式统一 |
| `REPRO.RANDOM_SEED_DOCUMENTATION` | 随机种子文档 |
| `REPRO.COMPUTE_RESOURCES_DOCUMENTED` | 计算资源文档 |
| `PROSE.CRYPTO_CONSTRUCTION_TEMPLATE` | 密码化构造写法（security-crypto） |
| `PROSE.INTENSIFIERS_ELIMINATION` | 删除空洞强调词 |
| `PROSE.EM_DASH_RESTRICTION` | 限制em-dash |
| `SUBMIT.SECTION_NUMBERING_CONSISTENCY` | Section编号一致 |
| `SUBMIT.PAGE_LIMIT_STRICT` | 严格页数限制 |
| `ETHICS.LIMITATIONS_SECTION_MANDATORY` | 必须Limitations节 |
| `ANON.DOUBLE_BLIND_ANONYMIZATION` | 双盲匿名检查 |

When a SoK profile is active, also enforce:
- explicit taxonomy in Background/Related Work <!-- policy:SOK.TAXONOMY_REQUIRED -->
- documented survey methodology (search + screening) <!-- policy:SOK.METHODOLOGY_REPORTING -->
- taxonomy-aligned big comparison table <!-- policy:SOK.BIG_TABLE_REQUIRED -->
- concrete research agenda in Conclusion/Discussion <!-- policy:SOK.RESEARCH_AGENDA_REQUIRED -->

When experiments include placeholder/non-executed results, also enforce:
- red uppercase fabricated disclosure in each affected figure/table caption <!-- policy:EXP.FABRICATED_RESULTS_CAPTION_DISCLOSURE -->
- subsection-level `% [FABRICATED] ...` status declaration comment <!-- policy:EXP.RESULTS_STATUS_DECLARATION_REQUIRED -->

---

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

**Full skill catalog**: See `references/skill-catalog.md` for the complete 35-skill list with trigger conditions, organized by category (Research & Analysis, Paper Writing & Publication, Development, Plugin Development, Tools & Utilities, Web Design, System).

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

---

## Available Agents

When a task requires delegation, use specialized agents.

**Full agent catalog**: See `references/agent-catalog.md` for the complete 14-agent list with auto-invocation rules.

**Quick reference:**
- **Research:** literature-reviewer, data-analyst, rebuttal-writer, paper-miner, kaggle-miner
- **Development:** architect, build-error-resolver, code-reviewer, refactor-cleaner, tdd-guide, bug-analyzer, dev-planner
- **Design:** ui-sketcher, story-generator

---

## Tool Mapping (Claude Code → Codex)

> **Runtime guard**: This section applies **only on Codex**. On Claude Code, use the native tools directly (`Edit`, `Write`, `Grep`, `Glob`, `Task`, etc.) — do NOT substitute Codex equivalents.

When running on Codex and skills reference Claude Code-specific tools, use the Codex equivalents.

**Full mapping table**: See `references/tool-mapping.md`.

**Quick reference:**
| Claude Code | Codex |
|------------|-------|
| `TodoWrite` | `plan` tool |
| `Skill` tool | Native skill discovery |
| `Task` subagent | `spawn_agent` |
| `Edit` / `Write` | `apply_patch` |
| `Grep` / `Glob` | `rg` / `rg --files` |
| `EnterPlanMode` | `plan` tool |

---

## Session Behavior

### Session Start

At the beginning of each conversation, perform these steps:

1. **Git Status**: Run `git status` and `git log --oneline -5` to understand the current project state
2. **Check TODOs**: Look for TODO.md, docs/todo.md, or plan/ directory files
3. **Report Context**: Briefly summarize branch, pending changes, and any active tasks

### Task Completion Summary

After completing each task, provide a structured summary:

```
--- Task Summary ---
1. [Main operations performed]
2. [Files modified]

Current Status:
- [Git/filesystem/runtime state]

Next Steps:
1. [Actionable suggestions]
```

---

## Security Awareness

### Dangerous Command Blacklist

NEVER execute these commands without explicit user confirmation:

- `rm -rf /` or any recursive deletion of system directories
- `dd` writing to block devices
- `mkfs.*` or `format` commands
- `DROP DATABASE/TABLE`, `DELETE FROM`, `TRUNCATE TABLE`
- Recursive deletion of `/etc/`, `/usr/`, `/bin/`, `/home/`, `/Users/`

### Warning Commands (proceed with caution)

- `rm`, `mv` (verify target paths first)
- `chmod`, `chown` (verify permissions scope)
- `wget`, `curl` piped to shell
- `npm install -g`, `pip install` outside virtual env

### Sensitive File Protection

NEVER read, write, or commit:

| Pattern | Reason |
|---------|--------|
| `.env`, `.env.*` | Environment secrets |
| `*.pem`, `*.key` | Private keys |
| `credentials.json` | Service account credentials |
| `*_secret*`, `*_token*` | Named secret files |
| `settings.json` | API tokens |

### Code Security

- No hardcoded passwords or API keys
- No `eval()` / `exec()` with user input
- Use parameterized SQL queries, never string concatenation
- No disabled SSL verification without justification

---

## Rules Summary

### Coding Style (ML Projects)

- **File size**: 200-400 lines maximum, split when exceeding 400
- **Immutability**: Use `@dataclass(frozen=True)` for configurations
- **Type hints**: Required for all function signatures
- **Patterns**: Factory & Registry for all ML modules
- **Config-driven**: Models accept only `cfg` parameter
- **Nesting**: Maximum 3 levels deep
- **Imports**: Standard library → third-party → local

### Experiment Reproducibility

- Always set random seeds (Python: `random`, `numpy`, `torch`, `torch.cuda`)
- Record configurations via Hydra auto-save
- Track environment info (Python version, torch version, CUDA, GPU)
- Checkpoint naming: `best_model.pt`, `checkpoint_epoch_N.pt`, `checkpoint_latest.pt`
- Record dataset hash or version tag

### Paper Writing Rules (Policy Engine)

Paper writing rules (37 rule cards) are defined in `policy/rules/`.
See `policy/README.md` for the full Rule ID Registry and rule card specification.
Skills reference rules via HTML comment markers. In case of conflict, `policy/rules/` is the single source of truth.
SoK requirements are activated by selecting SoK profiles (for example `policy/profiles/security-sok-sp.md`).
**Writing tasks must first read `policy/README.md` + relevant rule cards.**

---

## User Preferences

- **Language**: Respond in Chinese (中文); keep technical terms in English (NeurIPS, RLHF, TDD, Git)
- **Git**: Follow Conventional Commits (`feat/fix/docs/style/refactor/perf/test/chore`)
- **Branch strategy**: master/develop/feature/bugfix/hotfix/release
- **Merge strategy**: Rebase to sync feature branches, merge with `--no-ff`
- **Package manager**: `uv` for Python projects
- **Config stack**: Hydra + OmegaConf
- **Training**: Transformers Trainer
- **Working directories**: Plans in `/plan`, temporary files in `/temp` (auto-create if missing)
- **Comments**: Chinese in code, English for naming
