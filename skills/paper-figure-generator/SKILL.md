---
name: paper-figure-generator
description: |
  This skill should be used when the user asks to "generate a paper figure",
  "create system overview diagram", "make architecture figure", "draw pipeline",
  "create figure for paper", "design threat model figure", "make comparison figure",
  or needs a conceptual/illustrative academic diagram for a research paper.
  Generates publication-quality academic figures using Google Gemini or OpenAI image APIs.
version: 0.1.0
tags: [Research, Paper Writing, Figure Generation, Academic]
---

# Paper Figure Generator

Generate publication-quality conceptual figures for academic papers using AI image generation APIs (Google Gemini, OpenAI). Covers common figure types: system overviews, pipelines, threat models, comparisons, and architecture diagrams.

## 5-Step Workflow

### Step 1: Analyze — Extract Structured Content

Read the user's description of their method, system, or concept. Extract:

- **Components**: Named modules, blocks, or entities (3-8 recommended)
- **Relationships**: Data flow, connections, interactions between components
- **Groupings**: Which components belong together logically
- **Input/Output**: What enters and exits the system
- **Annotations**: Key formulas, dimensions, labels

Present the extracted structure to the user for confirmation before proceeding.

### Step 2: Select — Recommend Layout and Style

Based on the content analysis, recommend a **layout × style** combination from the options below. Use `AskUserQuestion` to let the user confirm or adjust.

**Layouts** (see `references/layouts.md` for details):

| Layout | Best For |
|--------|----------|
| `system-overview` | Figure 1, high-level method overview |
| `pipeline` | Multi-stage processing with data flow |
| `threat-model` | Security papers: attacker/defender/entities |
| `comparison` | Side-by-side: ours vs baseline |
| `architecture` | Detailed neural network / system architecture |

**Styles** (see `references/styles.md` for details):

| Style | Aesthetic |
|-------|-----------|
| `modern-gradient` | Gradient modules, rounded corners, soft shadows. NeurIPS/ICML 2024-2025 style |
| `clean-minimal` | Flat, solid colors, high contrast. Nature/Science style |
| `technical-blueprint` | Blueprint/schematic, engineering aesthetic |

Default recommendation: `system-overview` + `modern-gradient` for most papers.

### Step 3: Structure — Create Content File

Create `figures/{topic-slug}/content.md` with the structured content:

```markdown
# {Topic} - Figure Content

## Title
{Figure title}

## Components
1. {Name} - {Description}
2. ...

## Connections
- {Source} → {Target}: "{label}"
- ...

## Groupings
- "{Group Name}": {Component1}, {Component2}
- ...

## Annotations
- {annotation description}
- ...

## Emphasis
- {Component}: {reason for visual prominence}
```

### Step 4: Compose — Assemble Generation Prompt

Read the prompt template from `references/base-prompt-template.md`. Assemble the final prompt by concatenating:

1. **Base context** (fixed academic requirements)
2. **Layout fragment** from `references/layouts.md` (matching selected layout)
3. **Style fragment** from `references/styles.md` (matching selected style)
4. **Content section** (from Step 3's content.md, formatted per template)
5. **Aspect ratio** specification (default: 16:9)

Save the assembled prompt as `figures/{topic-slug}/prompt.md`.

### Step 5: Generate — Run Image Generation

Execute the generation script:

```bash
npx -y bun ${CLAUDE_PLUGIN_ROOT}/skills/paper-figure-generator/scripts/main.ts \
  --promptfiles figures/{topic-slug}/prompt.md \
  --output figures/{topic-slug}/figure.png \
  --ar 16:9
```

**CLI options:**
- `--promptfiles <path>` — Path to assembled prompt file
- `--output <path>` — Output image path (default: output.png)
- `--provider <google|openai>` — Override auto-detected provider
- `--model <name>` — Override default model
- `--ar <ratio>` — Aspect ratio: 16:9, 4:3, 1:1, 3:2 (default: 16:9)
- `--quality <normal|high>` — Image quality (default: normal)
- `--ref <path>` — Reference image for style guidance

**Provider configuration**: See `references/provider-setup.md` for API key setup.

After generation, display the output path and ask if the user wants to:
- Regenerate with adjustments
- Try a different style or layout
- Refine the prompt

## Important Notes

- This skill generates **conceptual/illustrative diagrams**, not data-driven plots or charts
- For data visualization (bar charts, line plots, heatmaps), use the `results-analysis` skill instead
- The generation script requires `bun` runtime (installed via `npx -y bun`)
- At least one API key (GOOGLE_API_KEY or OPENAI_API_KEY) must be configured in `.env`
- Generated figures are saved locally; embed them in LaTeX with `\includegraphics`
