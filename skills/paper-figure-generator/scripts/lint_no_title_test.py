import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestNoTitleLint(unittest.TestCase):
    def _write_svg(self, root: Path, name: str, svg: str) -> Path:
        path = root / name
        path.write_text(svg, encoding="utf-8")
        return path

    # --- SVG tests ---

    def test_detects_large_top_text(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            svg_path = self._write_svg(
                d,
                "final.svg",
                """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600" viewBox="0 0 800 600">
  <text x="400" y="40" text-anchor="middle" font-size="36">My Great Method</text>
  <text x="50" y="120" font-size="14">Module A</text>
  <text x="50" y="160" font-size="14">Module B</text>
</svg>
""",
            )

            import lint_no_title as lnt

            findings = lnt.lint_svg(svg_path)
            self.assertTrue(any("My Great Method" in f.text for f in findings))

    def test_ignores_small_top_labels(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            svg_path = self._write_svg(
                d,
                "final.svg",
                """<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600" viewBox="0 0 800 600">
  <text x="400" y="40" text-anchor="middle" font-size="14">Overview</text>
  <text x="50" y="120" font-size="14">Module A</text>
</svg>
""",
            )

            import lint_no_title as lnt

            findings = lnt.lint_svg(svg_path)
            self.assertEqual(findings, [])

    # --- --strict exit code tests ---

    def test_strict_returns_nonzero_on_findings(self) -> None:
        """--strict 模式下有 finding 应返回 exit code 2"""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self._write_svg(
                d,
                "final.svg",
                """<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600" viewBox="0 0 800 600">
  <text x="400" y="40" text-anchor="middle" font-size="36">Big Title Here Now</text>
  <text x="50" y="120" font-size="14">Module A</text>
  <text x="50" y="160" font-size="14">Module B</text>
</svg>
""",
            )

            import lint_no_title as lnt

            rc = lnt.main(["--strict", "--path", str(d / "final.svg")])
            self.assertEqual(rc, 2)

    def test_strict_returns_zero_on_clean(self) -> None:
        """--strict 模式下无 finding 应返回 exit code 0"""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self._write_svg(
                d,
                "final.svg",
                """<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600" viewBox="0 0 800 600">
  <text x="50" y="120" font-size="14">Module A</text>
  <text x="50" y="160" font-size="14">Module B</text>
</svg>
""",
            )

            import lint_no_title as lnt

            rc = lnt.main(["--strict", "--path", str(d / "final.svg")])
            self.assertEqual(rc, 0)

    def test_nonstrict_returns_zero_even_with_findings(self) -> None:
        """非 strict 模式下即使有 finding 也应返回 exit code 0"""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self._write_svg(
                d,
                "final.svg",
                """<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600" viewBox="0 0 800 600">
  <text x="400" y="40" text-anchor="middle" font-size="36">Big Title Here Now</text>
  <text x="50" y="120" font-size="14">Module A</text>
  <text x="50" y="160" font-size="14">Module B</text>
</svg>
""",
            )

            import lint_no_title as lnt

            rc = lnt.main(["--path", str(d / "final.svg")])
            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()

