#!/usr/bin/env python3
"""
轻量 no-title lint：检测 AutoFigure-Edit 输出里“顶部大字标题”的可能性。

注意：这是启发式检查（heuristic），只做提示；默认不会让流程失败。
如需严格模式可使用 --strict（用于 CI/强约束场景）。
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET


@dataclass(frozen=True)
class Finding:
    path: Path
    text: str
    reason: str
    x: float | None = None
    y: float | None = None
    font_size: float | None = None


_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _parse_number(value: str | None) -> float | None:
    if not value:
        return None
    match = _NUM_RE.search(value)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _parse_number_list(value: str | None) -> list[float]:
    if not value:
        return []
    out: list[float] = []
    for token in re.split(r"[ ,]+", value.strip()):
        if not token:
            continue
        num = _parse_number(token)
        if num is not None:
            out.append(num)
    return out


def _parse_style(style: str | None) -> dict[str, str]:
    if not style:
        return {}
    out: dict[str, str] = {}
    for chunk in style.split(";"):
        if ":" not in chunk:
            continue
        key, val = chunk.split(":", 1)
        key = key.strip().lower()
        val = val.strip()
        if key:
            out[key] = val
    return out


def _iter_text_elements(root: ET.Element) -> Iterable[ET.Element]:
    # SVG 常见 namespace：{http://www.w3.org/2000/svg}
    for el in root.iter():
        tag = el.tag
        if isinstance(tag, str) and tag.endswith("text"):
            yield el


def _get_svg_canvas(root: ET.Element) -> tuple[float | None, float | None, float | None]:
    """Return (min_y, height, width) in SVG user units if possible."""
    view_box = root.attrib.get("viewBox")
    if view_box:
        nums = _parse_number_list(view_box)
        if len(nums) >= 4:
            min_x, min_y, width, height = nums[:4]
            return (min_y, height, width)

    # Fallback: try height/width attrs (may include units like px)
    height = _parse_number(root.attrib.get("height"))
    width = _parse_number(root.attrib.get("width"))
    return (0.0, height, width)


def _get_font_size(el: ET.Element) -> float | None:
    direct = _parse_number(el.attrib.get("font-size"))
    if direct is not None:
        return direct
    style = _parse_style(el.attrib.get("style"))
    return _parse_number(style.get("font-size"))


def _get_anchor(el: ET.Element) -> str | None:
    anchor = el.attrib.get("text-anchor")
    if anchor:
        return anchor.strip().lower()
    style = _parse_style(el.attrib.get("style"))
    a2 = style.get("text-anchor")
    return a2.strip().lower() if a2 else None


def _get_xy(el: ET.Element) -> tuple[float | None, float | None]:
    xs = _parse_number_list(el.attrib.get("x"))
    ys = _parse_number_list(el.attrib.get("y"))
    x = xs[0] if xs else None
    y = ys[0] if ys else None

    # Some SVGs put y on <tspan> rather than <text>
    if y is None:
        for child in el.iter():
            if child is el:
                continue
            if isinstance(child.tag, str) and child.tag.endswith("tspan"):
                ys2 = _parse_number_list(child.attrib.get("y"))
                if ys2:
                    y = ys2[0]
                    break
    return (x, y)


def _get_text(el: ET.Element) -> str:
    return "".join(el.itertext()).strip()


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    v = sorted(values)
    mid = len(v) // 2
    if len(v) % 2 == 1:
        return v[mid]
    return (v[mid - 1] + v[mid]) / 2.0


def lint_svg(svg_path: Path, *, top_fraction: float = 0.15) -> list[Finding]:
    try:
        root = ET.fromstring(svg_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 - lint tool should never crash pipeline
        return [
            Finding(
                path=svg_path,
                text="",
                reason=f"failed to parse SVG: {e}",
            )
        ]

    min_y, height, width = _get_svg_canvas(root)
    font_sizes: list[float] = []
    text_nodes: list[tuple[str, float | None, float | None, float | None, str | None]] = []

    for el in _iter_text_elements(root):
        text = _get_text(el)
        if not text:
            continue
        font_size = _get_font_size(el)
        x, y = _get_xy(el)
        anchor = _get_anchor(el)
        if font_size is not None:
            font_sizes.append(font_size)
        text_nodes.append((text, font_size, x, y, anchor))

    med = _median(font_sizes)
    large_threshold = max(24.0, (med or 0.0) * 1.6)

    if height is not None and min_y is not None:
        top_y_max = min_y + height * top_fraction
    else:
        # Best-effort fallback if height can't be determined.
        top_y_max = 100.0

    findings: list[Finding] = []
    for text, font_size, x, y, anchor in text_nodes:
        if y is None or font_size is None:
            continue
        if y > top_y_max:
            continue
        if font_size < large_threshold:
            continue
        if len(text) < 6:
            continue

        center_hint = ""
        if width is not None and x is not None and 0.35 * width <= x <= 0.65 * width:
            center_hint = " near-center"
        if anchor == "middle":
            center_hint = (center_hint + " anchor=middle").strip()

        findings.append(
            Finding(
                path=svg_path,
                text=text,
                reason=f"possible in-figure title: large top text ({center_hint})".strip(),
                x=x,
                y=y,
                font_size=font_size,
            )
        )

    return findings


def _have_cmd(name: str) -> bool:
    return shutil.which(name) is not None


def lint_pdf(pdf_path: Path) -> list[Finding]:
    if not _have_cmd("pdftotext"):
        return []
    try:
        proc = subprocess.run(
            ["pdftotext", "-f", "1", "-l", "1", "-layout", str(pdf_path), "-"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []

    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        return []

    # Heuristic: longest early line with multiple words is often a title.
    first_lines = lines[:15]
    candidate = max(first_lines, key=lambda s: len(re.sub(r"\s+", "", s)), default="")
    normalized_len = len(re.sub(r"\s+", "", candidate))
    # Ignore obvious multi-column label rows (e.g., "Verifier      Server")
    # which are common in diagram bodies and are not top titles.
    if re.search(r"\s{3,}", candidate):
        return []

    if normalized_len >= 14 and " " in candidate and len(candidate) >= 12:
        return [
            Finding(
                path=pdf_path,
                text=candidate,
                reason="possible title-like line from pdftotext (heuristic)",
            )
        ]
    return []


def lint_png(png_path: Path, *, top_fraction: float = 0.15) -> list[Finding]:
    # Optional: OCR the top region if tesseract + PIL are available.
    if not _have_cmd("tesseract"):
        return []
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return []

    try:
        img = Image.open(png_path)
        w, h = img.size
        crop_h = max(1, int(h * top_fraction))
        top = img.crop((0, 0, w, crop_h))

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td) / "top.png"
            top.save(tmp_path, format="PNG")

            proc = subprocess.run(
                ["tesseract", str(tmp_path), "stdout", "-l", "eng", "--psm", "6"],
                check=False,
                capture_output=True,
                text=True,
            )

        if proc.returncode != 0:
            return []

        text = next((ln.strip() for ln in proc.stdout.splitlines() if ln.strip()), "")
        if len(text) >= 14 and " " in text:
            return [
                Finding(
                    path=png_path,
                    text=text,
                    reason="possible title-like text from OCR (heuristic)",
                )
            ]
    except Exception:
        return []
    return []


def lint_path(path: Path) -> list[Finding]:
    if path.is_dir():
        findings: list[Finding] = []
        svg = path / "final.svg"
        if svg.exists():
            findings.extend(lint_svg(svg))
        pdf = path / "figure.pdf"
        if pdf.exists():
            findings.extend(lint_pdf(pdf))
        png = path / "figure.png"
        if png.exists():
            findings.extend(lint_png(png))
        return findings

    if path.suffix.lower() == ".svg":
        return lint_svg(path)
    if path.suffix.lower() == ".pdf":
        return lint_pdf(path)
    if path.suffix.lower() == ".png":
        return lint_png(path)
    return []


def _format_finding(f: Finding) -> str:
    loc = []
    if f.font_size is not None:
        loc.append(f"font={f.font_size:g}")
    if f.x is not None:
        loc.append(f"x={f.x:g}")
    if f.y is not None:
        loc.append(f"y={f.y:g}")
    loc_s = (" " + " ".join(loc)) if loc else ""
    snippet = f.text.replace("\n", " ")
    if len(snippet) > 120:
        snippet = snippet[:117] + "..."
    return f"- {f.path.name}: {f.reason}{loc_s} | text=\"{snippet}\""


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="No-title lint for AutoFigure-Edit outputs (warn on probable in-figure title text)."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("."),
        help="SVG/PDF/PNG file or output directory (expects final.svg/figure.pdf/figure.png).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any findings are detected (default: warn only).",
    )
    args = parser.parse_args(argv)

    findings = lint_path(args.path)

    # Print capability notes (non-fatal)
    if args.path.is_dir():
        notes: list[str] = []
        if not _have_cmd("pdftotext"):
            notes.append("pdftotext not found -> PDF check skipped")
        if not _have_cmd("tesseract"):
            notes.append("tesseract not found -> PNG OCR check skipped")
        if notes:
            print("[no-title-lint] Note: " + "; ".join(notes))

    if not findings:
        print("[no-title-lint] OK: no obvious in-figure title detected.")
        return 0

    print("[no-title-lint] WARNING: possible in-figure title detected (heuristic).")
    for f in findings:
        print(_format_finding(f))
    print(
        "[no-title-lint] Fix: remove any top heading text inside the figure; use LaTeX caption instead."
    )

    return 2 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
