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
version: 0.2.0
tags: [Research, Paper Writing, Figure Generation, Academic, SVG]
---

# Paper Figure Generator

Generate publication-quality conceptual figures for academic papers using [AutoFigure-Edit](https://github.com/ResearAI/AutoFigure-Edit). Produces **editable SVG vector graphics** from method text descriptions. Covers common figure types: system overviews, pipelines, threat models, comparisons, and architecture diagrams.

**Key advantages over generic image APIs:**
- Output is SVG (editable), not just raster PNG
- Designed specifically for scientific paper figures
- Supports style transfer via reference images
- Includes icon detection, segmentation, and SVG template generation

## 5-Step Workflow

### Step 1: Analyze — Extract Method Description

Read the user's paper sections (method, system model, architecture) and extract a clear textual description of the system or method. Focus on:

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

### Step 2: Prepare — Write Method Text and Select Style

Create `figures/{topic-slug}/method.txt` with the method description from Step 1. Write it as clear, structured prose describing the system — AutoFigure-Edit generates figures directly from this text.

**Style transfer** (optional): If the user provides a reference image or wants a specific visual style, note the path for the `--reference_image_path` flag. See `references/styles.md` for guidance on selecting reference images.

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
bash skills/paper-figure-generator/scripts/setup.sh
```

### Step 4: Generate — Run AutoFigure-Edit

Execute the generation script:

```bash
bash skills/paper-figure-generator/scripts/generate.sh \
  --method_file figures/{topic-slug}/method.txt \
  --output_dir figures/{topic-slug}
```

**CLI options:**
- `--method_file <path>` — Path to method text file (required)
- `--output_dir <path>` — Output directory (required)
- `--use_reference_image --reference_image_path <path>` — Enable style transfer with reference image
- `--image_model <name>` — Override image generation model
- `--svg_model <name>` — Override SVG generation model
- `--sam_backend <local|fal|roboflow>` — Override SAM3 backend (default: auto-detected from env)
- `--optimize_iterations <n>` — SVG refinement iterations (0 to disable)
- `--merge_threshold <n>` — Region merging threshold (0 to disable)

**Output files:**
- `figures/{topic-slug}/figure.png` — Raster preview
- `figures/{topic-slug}/final.svg` — Editable SVG vector graphic
- `figures/{topic-slug}/icons/` — Extracted icon assets

After generation, display the output paths and ask if the user wants to:
- Regenerate with adjustments (modify method.txt or add reference image)
- Try a different style via reference image
- Manually edit the SVG

### Step 5: Finalize — Convert SVG to PDF for LaTeX

Convert the SVG to PDF for LaTeX inclusion:

```bash
# Option 1: 使用 skill 自带脚本（推荐）
bash skills/paper-figure-generator/scripts/svg-to-pdf.sh \
  --svg figures/{slug}/final.svg \
  --pdf figures/{slug}/figure.pdf

# Option 2: 项目 venv 中安装 cairosvg
uv pip install cairosvg
uv run python -c "import cairosvg; cairosvg.svg2pdf(url='figures/{slug}/final.svg', write_to='figures/{slug}/figure.pdf')"

# Option 3: Inkscape CLI
inkscape figures/{slug}/final.svg --export-type=pdf --export-filename=figures/{slug}/figure.pdf
```

Embed in LaTeX:

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/{slug}/figure.pdf}
  \caption{System overview of the proposed method.}
  \label{fig:system-overview}
\end{figure}
```

## Important Notes

- This skill generates **conceptual/illustrative diagrams**, not data-driven plots or charts
- For data visualization (bar charts, line plots, heatmaps), use the `results-analysis` skill instead
- AutoFigure-Edit source code (`autofigure2.py`) is vendored in `scripts/`; only `.venv/` is gitignored
- Requires LLM provider key (default `OPENROUTER_API_KEY`; optional `BIANXIE_API_KEY`) and a SAM3 backend key (`ROBOFLOW_API_KEY` recommended) in `.env`
- Output SVG can be further edited with any SVG editor (Inkscape, Illustrator, AutoFigure-Edit's built-in editor)
