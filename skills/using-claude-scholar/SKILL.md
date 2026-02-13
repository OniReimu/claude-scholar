---
name: using-claude-scholar
description: |
  Meta-skill that ensures proper use of the Claude Scholar skill system. This skill ALWAYS applies — it governs how you evaluate and activate other skills.

  ✅ This skill is ALWAYS active. It triggers on every user message.

  Purpose: Enforce the discipline of checking all available skills before responding.
version: 1.0.0
tags: [Meta, System, Skills]
---

# Using Claude Scholar

You are equipped with **Claude Scholar**, a comprehensive skill system for academic research and software development. This meta-skill ensures you use the system correctly.

## The #1 Rule

**Before responding to any user message, you MUST evaluate all available skills.**

This is not optional. This is not a suggestion. This is a hard requirement.

## Skill Evaluation Procedure

### Step 1: Scan

For every user message, scan the complete list of available skills:

**Research & Analysis:**
- `research-ideation` — Research startup: 5W1H brainstorming, literature review, gap analysis
- `results-analysis` — Experiment analysis: statistical testing, visualization, ablation studies
- `citation-verification` — Multi-layer citation validation (Format → API → Info → Content)
- `daily-paper-generator` — Automated daily paper generation for research tracking
- `paper-figure-generator` — Generate conceptual academic figures (system overviews, pipelines, architectures, threat models, comparisons) via Gemini/OpenAI

**Paper Writing & Publication:**
- `ml-paper-writing` — Full paper writing for NeurIPS, ICML, ICLR, ACL, AAAI, COLM, Nature, Science, Cell, PNAS
- `writing-anti-ai` — Remove AI writing patterns, add human voice (bilingual EN/CN)
- `paper-self-review` — 6-item quality checklist before submission
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

### Step 2: Decide

For each skill, make a **Yes/No decision**. There is no "maybe".

- If the answer is Yes (even 1% likely), activate the skill
- If multiple skills apply, activate all of them

### Step 3: Activate

Read the relevant SKILL.md file(s) before proceeding with your response.

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
