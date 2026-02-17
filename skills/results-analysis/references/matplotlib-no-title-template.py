"""
Matplotlib template for paper figures (no in-figure title).

规则（强制）：
- 不要使用 `plt.title()` / `ax.set_title()` / `fig.suptitle()`
- Figure 的“标题/描述”放在 LaTeX caption 里，而不是画在图里
- 图内只保留：axis labels、legend（必要时）、少量 annotations
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


def apply_paper_style() -> None:
    # 创建 2x 目标尺寸的图，由 LaTeX 缩放到 column width
    # 7" 宽 → LaTeX 缩放至 3.4" (single column) 时，24pt 字变为 ~12pt，仍可读
    # 规则：font ≥ 24pt in source（figures4papers 规范）
    mpl.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "font.size": 24,
            "axes.labelsize": 26,
            "axes.titlesize": 24,
            "legend.fontsize": 22,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "lines.linewidth": 3.0,
            "lines.markersize": 10,
            "axes.linewidth": 1.5,
            "grid.linewidth": 1.0,
            "grid.alpha": 0.35,
            # Vector-friendly: embed TrueType fonts in PDF
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_demo(out_path: Path) -> None:
    # 示例：折线图（无 title，仅 axis labels + legend）
    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = [1, 2, 3, 4, 5]
    y_a = [62.1, 66.4, 68.9, 70.3, 71.0]
    y_b = [60.8, 64.2, 66.7, 68.8, 69.5]

    ax.plot(x, y_a, marker="o", label="Method A")
    ax.plot(x, y_b, marker="s", label="Method B")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")

    ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout(pad=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    apply_paper_style()
    plot_demo(Path("figures/demo/figure.pdf"))


if __name__ == "__main__":
    main()

