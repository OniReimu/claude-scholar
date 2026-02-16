---
description: Generate editable SVG conceptual figures for academic papers via AutoFigure-Edit (paper-figure-generator). Produces method-driven diagrams (Figure 1, system overview, pipeline, architecture) and converts SVG to PDF for LaTeX.
---

# Paper Figure Command

Use this command to generate **editable SVG** conceptual figures (system overview, pipeline, architecture, threat model, comparison) for academic papers via AutoFigure-Edit.

## When to Use

Use `/paper-figure` when you want:
- Figure 1 / method overview diagram
- System Model / workflow diagram
- Threat model / comparison figure
- An editable SVG output (not just PNG)

## Workflow (Recommended)

1. Create a brief at `figures/{slug}/brief.md` (template: `skills/paper-figure-generator/references/figure-brief.md`)
2. Write `figures/{slug}/method.txt` from the brief
3. Run environment check:
   - `bash skills/paper-figure-generator/scripts/doctor.sh`
4. Generate:
   - `bash skills/paper-figure-generator/scripts/generate.sh --method_file figures/{slug}/method.txt --output_dir figures/{slug}`
5. Convert SVG to PDF for LaTeX:
   - `bash skills/paper-figure-generator/scripts/svg-to-pdf.sh --svg figures/{slug}/final.svg --pdf figures/{slug}/figure.pdf`

## Outputs

- `figures/{slug}/figure.png` (preview)
- `figures/{slug}/final.svg` (editable SVG)
- `figures/{slug}/figure.pdf` (for LaTeX)
- `figures/{slug}/run.json` (run metadata, no secrets)

