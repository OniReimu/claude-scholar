# /// script
# requires-python = ">=3.10"
# dependencies = ["python-docx", "pypdf", "pyyaml"]
# ///
"""Stage-D pdf renderer: route by PDF sub-type, fill honestly or degrade honestly.

  pdf-acroform            -> fill form fields with pypdf, save an OFFICIAL filled PDF.
  pdf-xfa / flat / scanned -> DO NOT fake an official fill. Emit PASTE-READY.txt + a
                              companion .docx and print a clear non-official warning:
                              these must be entered via the portal / typed manually.

The honesty rule (SPEC §7): a modality without a working renderer is declared
unsupported and downgraded to paste-ready — never faked as an official fill.

    uv run render_pdf.py filled.yaml form.pdf -o out_dir/

IR format: same as render_docx.py (flat mapping | {fields:{...}} | [{id,text}]).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def load_ir(path: Path) -> dict[str, str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "fields" in raw and isinstance(raw["fields"], dict):
        raw = raw["fields"]
    out: dict[str, str] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            out[str(k)] = "" if v is None else str(v)
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and "id" in item:
                out[str(item["id"])] = str(item.get("text", item.get("value", "")))
    else:
        sys.exit("error: IR must be a mapping or a list of {id,text} objects")
    return out


def detect_subtype(reader, path: Path) -> str:
    try:
        if reader.xfa:
            return "pdf-xfa"
    except Exception:
        pass
    try:
        if reader.get_fields():
            return "pdf-acroform"
    except Exception:
        pass
    try:
        chars = sum(len(p.extract_text() or "") for p in reader.pages)
        if chars / max(1, len(reader.pages)) < 40:
            return "pdf-scanned"
    except Exception:
        pass
    return "pdf-flat"


def fill_acroform(reader, ir: dict[str, str], out_pdf: Path) -> tuple[int, list[str]]:
    from pypdf import PdfWriter

    fields = reader.get_fields() or {}
    by_norm = {norm(name): name for name in fields}
    mapping: dict[str, str] = {}
    unresolved: list[str] = []
    for key, val in ir.items():
        target = fields.get(key) and key or by_norm.get(norm(key))
        if target:
            mapping[target] = val
        else:
            unresolved.append(key)

    writer = PdfWriter()
    writer.append(reader)
    try:  # nudge viewers to render values that lack a pre-baked appearance stream
        writer.set_need_appearances_writer(True)
    except Exception:
        pass
    for page in writer.pages:
        try:
            writer.update_page_form_field_values(page, mapping, auto_regenerate=False)
        except Exception:
            # older/newer pypdf signatures differ; fall back without the kwarg
            writer.update_page_form_field_values(page, mapping)
    with out_pdf.open("wb") as fh:
        writer.write(fh)
    return len(mapping), unresolved


def write_paste_ready(ir: dict[str, str], out_txt: Path, subtype: str) -> None:
    lines = [
        "=" * 70,
        "PASTE-READY — NON-OFFICIAL. Portal / manual entry REQUIRED.",
        f"Source PDF sub-type: {subtype}. This tool cannot produce an official filled PDF",
        "for this sub-type, so values are laid out for copy-paste into the real form.",
        "=" * 70,
        "",
    ]
    for k, v in ir.items():
        n = len(v)
        lines.append(f"### {k}   [{n} chars]")
        lines.append(v)
        lines.append("")
    out_txt.write_text("\n".join(lines), encoding="utf-8")


def write_companion_docx(ir: dict[str, str], out_docx: Path, subtype: str) -> bool:
    try:
        from docx import Document
    except ImportError:
        return False
    doc = Document()
    doc.add_heading("NON-OFFICIAL companion — manual/portal entry required", level=0)
    p = doc.add_paragraph()
    p.add_run(f"Source PDF sub-type: {subtype}. ").bold = True
    p.add_run("This .docx is a readable copy of the field values, not a submission artifact. "
              "Enter these into the official form via its portal or by typing.")
    for k, v in ir.items():
        doc.add_heading(f"{k}  ({len(v)} chars)", level=2)
        doc.add_paragraph(v)
    doc.save(str(out_docx))
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ir", type=Path, help="filled IR YAML (field-id -> text)")
    ap.add_argument("pdf", type=Path, help="the .pdf form")
    ap.add_argument("-o", "--out", type=Path, required=True, help="output directory (created if missing)")
    args = ap.parse_args()

    for p in (args.ir, args.pdf):
        if not p.exists():
            sys.exit(f"error: file not found: {p}")
    if args.pdf.suffix.lower() != ".pdf":
        sys.exit("error: form must be a .pdf")

    try:
        from pypdf import PdfReader
    except ImportError:
        sys.exit("error: pypdf not available. Run via `uv run` so PEP 723 deps install.")

    ir = load_ir(args.ir)
    if not ir:
        sys.exit("error: IR contained no fields")

    args.out.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(str(args.pdf))
    subtype = detect_subtype(reader, args.pdf)
    stem = args.pdf.stem

    if subtype == "pdf-acroform":
        out_pdf = args.out / f"{stem}-filled.pdf"
        try:
            n, unresolved = fill_acroform(reader, ir, out_pdf)
        except Exception as exc:
            print(f"warning: AcroForm fill failed ({exc}); degrading to paste-ready.", file=sys.stderr)
            subtype = "pdf-flat"
        else:
            print(f"OFFICIAL fill: wrote {out_pdf}")
            print(f"  filled {n}/{len(ir)} fields into the AcroForm.")
            if unresolved:
                print(f"  UNRESOLVED ({len(unresolved)}) — no AcroForm field matched these IR keys:")
                for k in unresolved:
                    print(f"    - {k}")
                print("  Not written. Check the field names (run extract_form.py to list them).")
            return

    # xfa / flat / scanned -> honest degrade
    out_txt = args.out / f"{stem}-PASTE-READY.txt"
    out_docx = args.out / f"{stem}-companion.docx"
    write_paste_ready(ir, out_txt, subtype)
    docx_ok = write_companion_docx(ir, out_docx, subtype)

    print("=" * 70)
    print(f"NON-OFFICIAL OUTPUT — sub-type '{subtype}' has no working native filler.")
    print("Portal / manual entry is REQUIRED. Nothing here is a submittable PDF.")
    print("=" * 70)
    reason = {
        "pdf-xfa": "XFA dynamic form: pypdf cannot reliably fill the XFA packet.",
        "pdf-flat": "Flat PDF: no form fields to fill.",
        "pdf-scanned": "Scanned/image PDF: no text layer; would need OCR + overlay.",
    }.get(subtype, "Unsupported sub-type.")
    print(f"reason: {reason}")
    print(f"  wrote {out_txt}  (paste-ready, per-field char counts)")
    print(f"  wrote {out_docx}" if docx_ok else "  (companion .docx skipped: python-docx unavailable)")
    sys.exit(3)


if __name__ == "__main__":
    main()
