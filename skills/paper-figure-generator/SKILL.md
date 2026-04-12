---
name: paper-figure-generator
description: |
  This skill should be used when the user asks to "generate a paper figure",
  "create system overview diagram", "make architecture figure", "draw pipeline",
  "create figure for paper", "design threat model figure", "make comparison figure",
  or needs a conceptual/illustrative academic diagram for a research paper.
  Also AUTO-ACTIVATES during paper writing workflows: when writing Figure 1,
  system model sections, method overview sections, or any paper section that
  describes a system architecture, pipeline, or workflow that should be visualized.
  If the user is writing a paper and discusses a method/system/pipeline, proactively
  suggest generating a figure for it.
  Generates editable SVG academic figures using AutoFigure-Edit from method text descriptions.
version: 0.2.2
tags: [Research, Paper Writing, Figure Generation, Academic, SVG]
---

# Paper Figure Generator

Generate publication-quality conceptual figures for academic papers using [AutoFigure-Edit](https://github.com/ResearAI/AutoFigure-Edit). Produces **editable SVG vector graphics** from method text descriptions. Covers common figure types: system overviews, pipelines, threat models, comparisons, and architecture diagrams.

## Policy Rules

> 本 skill 执行以下论文写作规则。权威定义在 `policy/rules/`。
> 行内出现处以 HTML 注释标记引用。**冲突时以 `policy/rules/` 为准。**

| Rule ID | 摘要 |
|---------|------|
| `FIG.NO_IN_FIGURE_TITLE` | 图内不加标题 |
| `FIG.SYSTEM_OVERVIEW_ASPECT_RATIO_GE_2TO1` | system overview/pipeline/architecture 图宽高比不小于 2:1 |

**Key advantages over generic image APIs:**
- Output is SVG (editable), not just raster PNG
- Designed specifically for scientific paper figures
- Supports style transfer via reference images
- Includes icon detection, segmentation, and SVG template generation

## Execution Priority (Mandatory)

1. **Default path (first priority):** always run AutoFigure-Edit via `scripts/generate.sh` with OpenRouter (`OPENROUTER_API_KEY`).
2. **Do not ask Google/OpenAI first:** never start by asking the user to choose Gemini/OpenAI provider before attempting the AutoFigure-Edit path.
3. **Fallback condition:** only fallback to legacy Gemini/OpenAI flow if AutoFigure-Edit generation actually fails and the user explicitly requests fallback.
4. **Outdated-skill detection:** if the agent shows a prompt like “needs `GOOGLE_API_KEY` or `OPENAI_API_KEY`”, treat it as an outdated plugin cache and continue with this skill's AutoFigure-Edit command path.
5. **No title in generated image:** never add a top title/heading text inside the generated figure; use paper caption instead. <!-- policy:FIG.NO_IN_FIGURE_TITLE -->

## Multi-Agent Workflow

**All subagents MUST use `model: opus` (Claude Opus 4.6).** Haiku is insufficient for complex protocol/architecture understanding.

### Step 1: Analyze — Planner Subagent

Spawn a **Planner subagent** (Opus, high thinking) to read the user's paper sections (method, system model, architecture) and extract a structured method description. The subagent should focus on:

- **Components**: Named modules, blocks, or entities (3-8 recommended)
- **Relationships**: Data flow, connections, interactions between components
- **Groupings**: Which components belong together logically
- **Input/Output**: What enters and exits the system
- **Annotations**: Key formulas, dimensions, labels

Present the extracted structure to the user for confirmation before proceeding.

**Recommended artifact (for repeatability):**
- Create a brief at `figures/{topic-slug}/brief.md` using `references/figure-brief.md`.
- Confirm the brief with the user before writing `method.txt`.

**Layout guidance** (see `references/layouts.md`): Choose the most appropriate layout type based on the content:

| Layout | Best For |
|--------|----------|
| `system-overview` | Figure 1, high-level method overview |
| `pipeline` | Multi-stage processing with data flow |
| `threat-model` | Security papers: attacker/defender/entities |
| `comparison` | Side-by-side: ours vs baseline |
| `architecture` | Detailed neural network / system architecture |

### Step 1.5: Critic — Review Method Description

Spawn a **Critic subagent** (Opus, high thinking) to review the Planner's output. The Critic checks:

- **Protocol correctness**: Are message flows accurate? (e.g., unicast vs broadcast, who aggregates)
- **Completeness**: Are all components, certificates, and phases represented?
- **Design logic**: Are there conflicting visual dimensions? (e.g., two time axes)
- **Consistency**: Do colors, labels, and annotations match across the diagram?

If the Critic finds issues, it produces a corrected `method.txt`. If no issues, proceed with the Planner's version.

### Step 2: Prepare — Write Method Text and Select Style

Create `figures/{topic-slug}/method.txt` with the method description from Step 1. Write it as clear, structured prose describing the system — AutoFigure-Edit generates figures directly from this text.

Hard constraint for `method.txt`: do not request an in-figure title (top heading text). Keep only component labels, arrows, and annotations. <!-- policy:FIG.NO_IN_FIGURE_TITLE -->

**Style transfer** (default enabled): `generate.sh` will automatically use built-in style references in `skills/paper-figure-generator/.autofigure-edit/img/reference/` (`sample3.png` primary, `sample2.png` secondary) when `--reference_image_path` is not provided. If the user wants a specific style, pass `--reference_image_path` explicitly. See `references/styles.md`.

### Step 3: Setup — Verify Dependencies

Check if the Python virtual environment is ready:

```bash
ls skills/paper-figure-generator/scripts/.venv/bin/python
```

Run a quick environment check (recommended):

```bash
bash skills/paper-figure-generator/scripts/doctor.sh
```

If not installed, run setup (one-time, installs Python dependencies only — source code is already in the repo):

```bash
# 1) 安装依赖并创建 venv
bash skills/paper-figure-generator/scripts/setup.sh

# 2) 登录 HuggingFace（部分模型需要认证访问，交互式输入 access token）
skills/paper-figure-generator/scripts/.venv/bin/hf auth login
```

### Step 4: Generate — Run AutoFigure-Edit

Execute the generation script:

```bash
AUTOFIGURE_PROVIDER=openrouter \
bash skills/paper-figure-generator/scripts/generate.sh \
  --method_file figures/{topic-slug}/method.txt \
  --output_dir figures/{topic-slug} \
  --optimize_iterations 0
```

> `--optimize_iterations 0` skips the API-based Step 4.6 optimization. Claude Code handles SVG refinement in Step 4.6 below, using session tokens instead of API keys.

**CLI options:**
- `--method_file <path>` — Path to method text file (required)
- `--output_dir <path>` — Output directory (required)
- `--use_reference_image --reference_image_path <path>` — Enable style transfer with reference image
- `--image_model <name>` — Override image generation model
- `--svg_model <name>` — Override SVG generation model (note: `google/gemini-3-pro-preview` may 404; use `google/gemini-2.5-pro` as fallback)
- `--sam_backend <local|fal|roboflow>` — Override SAM3 backend (default: auto-detected from env)
- `--sam_prompt <prompts>` — Comma-separated SAM3 detection prompts (default: `icon,robot,animal,person`). **Use Stylist recommendations — see Step 4.5.**
- `--optimize_iterations <n>` — SVG refinement iterations (0 to disable)
- `--merge_threshold <n>` — Region merging threshold (0 to disable)

**Default reference behavior:**
- If `--reference_image_path` is omitted, wrapper auto-adds reference style from:
  - `skills/paper-figure-generator/.autofigure-edit/img/reference/sample3.png` (primary)
  - fallback: `skills/paper-figure-generator/.autofigure-edit/img/reference/sample2.png`

**Output files:**
- `figures/{topic-slug}/figure.png` — Raster preview
- `figures/{topic-slug}/final.svg` — Editable SVG vector graphic
- `figures/{topic-slug}/icons/` — Extracted icon assets

After generation, display the output paths and proceed to Step 4.6 for optimization.

If Step 4 fails:
- First diagnose with `bash skills/paper-figure-generator/scripts/doctor.sh`
- Keep AutoFigure-Edit as default; only switch to legacy Gemini/OpenAI flow when the user explicitly asks for fallback

### Step 4.5: Stylist — SAM Parameter Selection (Auto)

**If Step 4 completes but SAM3 detects 0 icons**, the default prompts (`icon,robot,animal,person`) don't match the figure style. Run the Stylist to fix this.

**Stylist subagent** (Opus, high thinking) reads `figures/{topic-slug}/figure.png` and recommends:

1. **`sam_prompt`** — detection prompts matching the figure's visual elements:
   | Figure Style | Recommended Prompts |
   |-------------|-------------------|
   | System overview (servers, databases, users) | `icon,robot,person` (default) |
   | Message sequence / swimlane diagram | `rectangle,box` |
   | Pipeline / flow diagram | `rectangle,box,arrow` |
   | Architecture with icons | `icon,rectangle,box` |

2. **`min_score`** — confidence threshold to filter noise:
   - Figures with many small elements: `0.5` (permissive)
   - Figures with few large blocks: `0.6–0.7` (strict, avoids background detection)

3. **`merge_threshold`** — overlap merge threshold:
   - Dense diagrams: `0.7` (aggressive merge)
   - Sparse diagrams: `0.9` (default, minimal merge)

Then re-run SAM3 + remaining steps with Stylist's parameters:

```bash
# Re-run from Step 2 onward with Stylist-recommended params
AUTOFIGURE_PROVIDER=openrouter \
bash skills/paper-figure-generator/scripts/generate.sh \
  --method_file figures/{topic-slug}/method.txt \
  --output_dir figures/{topic-slug} \
  --sam_prompt "{stylist_recommended_prompts}" \
  --svg_model "google/gemini-2.5-pro" \
  --optimize_iterations 0
```

**Skip condition:** If Step 4 already detected icons successfully (>0), skip Stylist and proceed to Step 4.6.

### Step 4.6: Optimize — Claude Code SVG Refinement (No API Key Required)

**This step uses Claude Code's own session to refine the SVG — no OpenRouter/Bianxie API tokens consumed.**

Always run `generate.sh` with `--optimize_iterations 0` in Step 4 so that API-based optimization is skipped. Claude Code performs the refinement here instead.

**Scope boundary:** Step 4.6 ONLY refines an existing `final.svg` produced by Step 4. It does NOT replace the Step 4 pipeline (image generation, SAM3 icon detection, SVG template creation, icon replacement). If Step 4 fails for any reason (SAM3 API down, OpenRouter error, etc.), do NOT attempt to hand-write or generate SVG from scratch — report the Step 4 failure and stop.

**Target file:** `figures/{topic-slug}/final.svg` — this is the file Step 5 converts to PDF. Edit this file directly.

**Prerequisite check:** Before starting, verify both files exist:
- `figures/{topic-slug}/final.svg` — if missing, Step 4 failed; run `doctor.sh` and report the error.
- `figures/{topic-slug}/figure.png` — if missing, skip visual comparison; only perform structural SVG review (text alignment, spacing, layout consistency from SVG source alone).

**Iteration loop (max 2 rounds by default):**

For each iteration (up to 2):

- **(a) Render current SVG to PNG:**
  ```bash
  DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}" \
  skills/paper-figure-generator/scripts/.venv/bin/python -c "
  from cairosvg import svg2png
  svg2png(url='figures/{topic-slug}/final.svg', write_to='figures/{topic-slug}/final_preview.png', scale=2)
  "
  ```
  If this fails (cairosvg not installed or SVG parse error), run `bash skills/paper-figure-generator/scripts/doctor.sh` and report the issue. Do not proceed without a preview PNG.

  Known cairosvg limitations: CSS filters (`drop-shadow`, `blur`), `<foreignObject>`, and some gradient types may not render correctly. If the preview shows missing shadows or broken regions that look fine in the SVG source, these are cairosvg rendering gaps — do not attempt to fix them.

- **(b) Read and compare:**
  - Read `figures/{topic-slug}/figure.png` (original target) and `figures/{topic-slug}/final_preview.png` (current rendering) using the Read tool for visual comparison.
  - **Do NOT read `final.svg` raw** — it contains large base64-encoded icon images that waste tokens. Instead, use Grep to extract structural elements:
    ```bash
    grep -n '<\(text\|rect\|line\|path\|circle\|ellipse\|polygon\|g \|g>\|image\|marker\|defs\|use\|svg\)' figures/{topic-slug}/final.svg | grep -v 'base64'
    ```
    This shows layout-relevant tags with line numbers for targeted editing.

- **(c) Analyze differences** across two aspects:
  - **POSITION**: icon images, text elements, arrows, lines/borders
  - **STYLE**: sizes/proportions, font sizes/colors/weights, arrow styles, line styles

- **(d) Edit the SVG** using the Edit tool to fix identified issues. Preserve:
  - All `<image>` elements with `id` like `icon_AF01`, `icon_AF02` (replaced icons with embedded base64 data) — do not modify the `href` attribute or base64 payload
  - No in-figure title text <!-- policy:FIG.NO_IN_FIGURE_TITLE -->

- **(e) Re-render and verify** — run the render command from (a) again, then Read both `final_preview.png` and `figure.png` side-by-side to confirm improvement.

**After 2 iterations:** stop and ask the user:
- "SVG 已经过 2 轮优化，是否需要继续？" — if yes, do 1 more round; if no, proceed to Step 5.

**Skip condition:** If no actionable differences are found in Position or Style analysis (step c), skip further iterations and proceed directly to Step 5.

### Step 5: Finalize — Convert SVG to PDF for LaTeX

Convert the SVG to PDF for LaTeX inclusion:

```bash
# Option 1: 使用 skill 自带脚本（推荐）
bash skills/paper-figure-generator/scripts/svg-to-pdf.sh \
  --svg figures/{topic-slug}/final.svg \
  --pdf figures/{topic-slug}/figure.pdf

# Option 2: 项目 venv 中安装 cairosvg
uv pip install cairosvg
uv run python -c "import cairosvg; cairosvg.svg2pdf(url='figures/{topic-slug}/final.svg', write_to='figures/{topic-slug}/figure.pdf')"

# Option 3: Inkscape CLI
inkscape figures/{topic-slug}/final.svg --export-type=pdf --export-filename=figures/{topic-slug}/figure.pdf
```

Embed in LaTeX:

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/{topic-slug}/figure.pdf}
  \caption{System overview of the proposed method.}
  \label{fig:system-overview}
\end{figure}
```

## Important Notes

- This skill generates **conceptual/illustrative diagrams**, not data-driven plots or charts
- Do not add title text inside the image canvas; keep title/description in caption or surrounding paper text <!-- policy:FIG.NO_IN_FIGURE_TITLE -->
- For `system-overview` / `pipeline` / `architecture` layouts, keep canvas aspect ratio `width:height >= 2:1` (e.g., 2.1:1, 3:1) <!-- policy:FIG.SYSTEM_OVERVIEW_ASPECT_RATIO_GE_2TO1 -->
- For data visualization (bar charts, line plots, heatmaps), use the `results-analysis` skill instead
- AutoFigure-Edit source code (`autofigure2.py`) is vendored in `scripts/`; only `.venv/` is gitignored
- Requires LLM provider key (default `OPENROUTER_API_KEY`; optional `BIANXIE_API_KEY`) for Step 4 generation and a SAM3 backend key (`ROBOFLOW_API_KEY` recommended) in `.env`. Step 4.6 SVG refinement uses Claude Code session tokens instead — no additional API key needed
- Some models require HuggingFace authentication; run `hf auth login` in the project venv (see Step 3)
- Output SVG can be further edited with any SVG editor (Inkscape, Illustrator, AutoFigure-Edit's built-in editor)
