import tempfile
import unittest
from pathlib import Path


class TestNoTitleLint(unittest.TestCase):
    def _write_svg(self, root: Path, name: str, svg: str) -> Path:
        path = root / name
        path.write_text(svg, encoding="utf-8")
        return path

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


if __name__ == "__main__":
    unittest.main()

