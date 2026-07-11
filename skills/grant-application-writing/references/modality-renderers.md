# Modality Registry (Stage D)

> The output modality is **dictated by the input form's modality** — you emit the form
> back in the shape it arrived. Each modality is a contract of three operations:
>
> - **`extract`** — blank form → field skeleton (feeds Stage A's `scheme.yaml`).
> - **`validate`** — does a filled draft satisfy the form's own constraints (limits, required, format)?
> - **`render`** — filled IR → native output the applicant submits.
>
> **Honesty rule (load-bearing).** A modality without a *working* renderer is declared
> **unsupported** and degraded honestly — never faked as an official fill. "I filled your
> PDF" when the bytes are actually a paste-ready sidecar is the one failure this stage exists
> to prevent. Every degrade path emits output **explicitly marked non-official**.

## Registry index

| modality | renderer | status |
|----------|----------|--------|
| `web-portal` | PASTE-READY.txt (+ browser-assisted live-fill) | **first-class** |
| `docx` | `scripts/render_docx.py` (python-docx write-back) | supported |
| `pdf-acroform` | `scripts/render_pdf.py` (pypdf field fill) | supported |
| `pdf-xfa` | — | degrade → PASTE-READY + companion .docx |
| `pdf-flat` / `pdf-scanned` | — | degrade → PASTE-READY + companion .docx |
| `latex` / `overleaf` | — | not-yet-implemented (extract/validate only) |
| `xlsx` / `csv` budget workbook | — | not-yet-implemented |
| `sf424-xml` (Grants.gov) | — | not-yet-implemented |
| `google-forms` / `ms-forms` | browser-assisted only | partial |
| `smartsimple` / `infoready` / `submittable` | browser-assisted only | partial |
| `zip` submission package | assembler over member renderers | planned |

---

## Renderer input contract — `scheme.yaml` (structure) + `values.yaml` (content)

Every renderer takes **two** inputs, never one: it loads **structure + limits** from
`scheme.yaml` (`sections[].fields[]` — the widget, role, `limit`, required flag) and the
**filled content** from a sidecar **`values.yaml`** (`{field-id: value}` — see
`templates/values.template.yaml`). Back-compat: `values.yaml` *is* a flat `{field-id: value}`
map, the same shape the scripts already accept as a bare IR.

Resolution is **fail-closed** (see the honesty rule): a value whose `field-id` is not in
`scheme.yaml`, or a **required** `scheme.yaml` field with **no** value in `values.yaml`, is a
**resolution failure** — the renderer exits non-zero and does **not** emit an "official" partial
(`render_pdf.py` / `render_docx.py`), unless an explicit `--allow-partial` marks the output
non-official.

---

## 0. Structural artefacts inside heading-sequenced uploads (figures / tables / timelines)

A funded project description (`structured-upload heading-sequenced`, e.g. an ARC C1) carries
part of its score in **structural artefacts**, not prose: an **architecture figure**, a
**"today's technique vs our innovation" comparison table**, and a **phase×year Gantt**. These
must be *rendered*, not left as ASCII/text mock-ups — an assessor scores a real diagram, not
layout intent (the honesty rule applies: a described figure is not a rendered figure).
- **Architecture / concept / pipeline figures** → route to the repo skills **`fireworks-tech-graph`** (flowcharts, architecture, sequence/state diagrams) or **`paper-figure-generator`** (system-overview / pipeline / architecture, editable SVG). Draft the figure *spec* here (nodes, flows, caption); hand it to those skills to render.
- **Comparison table & Gantt** → real tables (markdown → the upload's `.docx`/`.pdf`), with a live phase×year grid; not ASCII art.
- Draft may carry a **placeholder** figure block, but it is flagged `[FIGURE — RENDER via fireworks-tech-graph]` (per `method-passes.md` §1.8 invented-specifics marking), never shipped as the final artefact.

---

## 1. web-portal → PASTE-READY.txt  (first-class, most-used)

The dominant path: Symplectic Elements, ARC RMS, NHMRC Sapphire, Submittable-style portals.
There is no file to write back into, so the deliverable is a **paste-ready text file**, one
block per field, so the applicant copies each block into its box with zero editing.

- **extract** — read the portal's field DOM/labels into the field skeleton. For static HTML,
  parse the page. For **JS-validated / authenticated / dynamic-conditional / session-timeout**
  portals, use **browser-assisted extraction** (drive a live session, expand conditional
  branches, reveal hidden-required fields, read calculated budget pages that only render
  client-side). Record each field's `maxlength`, required flag, and any inline validator.
- **validate** — run a **dry-run validation log** against the extracted constraints *before*
  the applicant pastes: per-field char/word count vs limit, required-not-empty, option value
  is still in the (round-scoped) enum, budget totals match. In browser-assisted mode, optionally
  type into the live form and read back the portal's own validation messages, then **clear the
  fields** — dry-run means observe, never submit.
- **render** — emit `PASTE-READY.txt`, **ordered by field id**, one block per field, in the
  canonical block grammar (identical to `templates/PASTE-READY.template.txt`; the SAME grammar
  `charcount.py` parses and `render_pdf.py` produces):

```
=== rope | Research Opportunity and Performance Evidence (ROPE) | limit: 2000 chars ===
<the exact text to paste, already fitted to the limit>
=== /rope ===
```

The header carries **`<field-id> | <LABEL> | limit: <N|null> <unit>`** (`unit` ∈
chars|words|pages; `limit` may be literal `null` = UNVERIFIED). The **terminator
`=== /<field-id> ===` closes each block** so the last box cannot absorb trailing text — a missing
terminator is a hard error. The **body is everything strictly between the header line and its
matching terminator**. `charcount.py` counts the body and reports OVER / BLOCK (null limit) /
NEEDS-RENDER (`pages`) at the top of the file — never silently truncate.

**Optional live-fill.** When the portal permits and the applicant asks, browser-assisted mode
can type each validated block into the real form and pause before the final submit for human
review. Live-fill is opt-in and always preceded by the dry-run log; the applicant owns the
submit click.

---

## 2. .docx → official-template write-back

When the form *is* a Word template, fill it in place and hand back a filled copy of the
official document — never a re-typed lookalike.

- **extract** — walk the template for content-controls (Word SDT / AcroForm-like fields) and
  for **mandated headings** (`heading-sequenced` uploads); map each to a field id.
- **validate** — every required control filled; text under each heading within its page/word
  limit; template styles untouched.
- **render** — `scripts/render_docx.py` (python-docx). Fill content-controls by tag; where the
  template uses headings rather than controls, insert body text **under the mandated heading**
  in the fixed order. **Preserve template formatting** (styles, fonts, section breaks, headers/
  footers) — write into the existing structure, do not rebuild the document.
- **type-specific content controls (fail-closed).** A checkbox / dropdown / date SDT must be
  written **as that control type** — never plain text stuffed into a checkbox and called done. If
  a value cannot be written as the correct control type it is marked **UNRESOLVED** and the
  renderer **exits non-zero**. Match by **tag first, then alias**; report ambiguous/colliding
  matches. Multi-paragraph answers split into paragraphs/runs preserving the template style.

## 3. .pdf → route by sub-type

`scripts/render_pdf.py` dispatches on the PDF sub-type — a PDF is never one thing:

- **`acroform`** — real fillable form fields. **extract** field names/types via pypdf;
  **validate** required + max-length + field type; **render** fills fields with pypdf and sets
  `NeedAppearances` so viewers show the values. Supported.
- **`xfa`** — LiveCycle/XFA forms are a *different beast* from AcroForm (an XML dynamic layer,
  not flat widgets); they **often cannot be filled cleanly** with open tooling and may break if
  edited. Detect the `/XFA` entry; if present and not trivially AcroForm-backed, **degrade**
  (§4) rather than emit a corrupt file.
- **`flat-text` / `scanned`** — no form fields at all. **extract** via text layer, or OCR for
  scanned pages, carrying an **OCR-confidence** score per field; low-confidence extraction is
  flagged for human confirmation, never trusted silently. These cannot be filled in place →
  degrade (§4).

## 4. flat / scanned .pdf → HONEST DEGRADE

A flat or scanned PDF (and an unfillable XFA) has **no place to put text** — overlaying an
image is not an official submission. So do not pretend:

- **render** — emit `PASTE-READY.txt` (§1 block format) **plus** a companion `.docx` that mirrors
  the form's headings, both **explicitly marked non-official** in a banner at the top of each
  file (e.g. `*** UNOFFICIAL — content for transcription into the official PDF; not a valid
  submission file ***`).
- Report clearly that the official artifact must be produced by the applicant (or the portal),
  and that this skill supplies the *content*, not the *filed document*.

This is the concrete instance of the honesty rule: better a labelled sidecar than a fake fill.

## 5. Other modalities to register

Registered so Stage A can classify them; extract/validate implemented where cheap, `render`
honestly marked **not-yet-implemented** until a real application demands it.

- **LaTeX / Overleaf** — extract: parse `\newcommand`/macro-driven fields or template comments;
  validate: compile + length checks; render: **not-yet-implemented** (write macros / section
  bodies into the `.tex` tree). Degrade → PASTE-READY per section.
- **Excel / CSV budget workbook** — extract: read sheet/named-range structure; validate: row
  caps, cross-field sums, matched-funding ratios (a `budget-matrix`/`computed` job). Render:
  **not-yet-implemented** (openpyxl write-back). Note: **CRC-P's mandatory financial workbook**
  is exactly this — a formula-bearing `.xlsx` that must be returned filled, not re-typed.
- **Grants.gov / SF424 XML** — extract: parse the SF424 form-set schema; validate: against the
  XSD; render: **not-yet-implemented** (emit compliant submission XML). Degrade → PASTE-READY.
- **Google Forms / Microsoft Forms** — no file artifact; **browser-assisted only** (partial):
  extract fields from the live form, validate, live-fill under human control.
- **SmartSimple / InfoReady / Submittable** — hosted portals; treat as `web-portal`
  (browser-assisted, partial) with per-platform field-discovery quirks.
- **ZIP submission package** — a container, not a leaf modality. extract: manifest of required
  members + their own modalities; validate: all members present, each passes its own renderer's
  validate; render (**planned**): run the per-member renderers, then assemble the archive to the
  mandated filename pattern.

## 6. Modality-detection routine

Given an input file or URL, classify and dispatch:

```
detect(input):
  if input is URL or portal handle:
      → web-portal            (SmartSimple/InfoReady/Submittable/Google/MS Forms all route here)
  elif ext == .docx / .dotx:
      → docx
  elif ext == .pdf:
      probe = pdf_probe(bytes)
      if probe.has_xfa and not probe.acroform_backed:  → pdf-xfa      → HONEST DEGRADE (§4)
      elif probe.has_acroform_fields:                  → pdf-acroform
      elif probe.has_text_layer:                       → pdf-flat     → HONEST DEGRADE (§4)
      else:                                            → pdf-scanned  → OCR + HONEST DEGRADE (§4)
  elif ext in {.tex}:            → latex          (render not-yet-implemented → degrade)
  elif ext in {.xlsx, .xls, .csv}: → xlsx/csv budget workbook (render nyi → degrade)
  elif ext == .xml and schema ~ SF424: → sf424-xml (render nyi → degrade)
  elif ext == .zip:             → zip package    (recurse detect() over members)
  else:                         → UNSUPPORTED    → declare + HONEST DEGRADE, never fake
```

Rule of dispatch: pick the renderer whose `status` is *supported* for that modality; if the only
available path is *degrade* or *not-yet-implemented*, say so plainly and emit the marked-non-
official sidecar. Detection never silently coerces an input into a renderer it doesn't fit —
the same falsifiability discipline as the type model.
