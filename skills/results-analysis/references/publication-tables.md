# Publication-Quality Tables (data → LaTeX)

Scope: turn experiment results (CSV / Excel / DataFrame) into submission-ready LaTeX tables.
Fills the gap the figure skills don't cover — `scientific-figure-making` / `paper-figure-generator`
/ `fireworks-tech-graph` do plots and diagrams, **not tables**. Tables and plots are complementary:
use this for the central results table, ablation table, and the comparison matrix.

## Principles (all enforced by policy)

- **booktabs only** — `\toprule` / `\midrule` / `\bottomrule`; no vertical rules, no `\hline`. `<!-- policy:TABLE.BOOKTABS_FORMAT -->`
- **Direction indicators** in headers (`$\uparrow$` / `$\downarrow$`) so the reader knows which way is better. `<!-- policy:TABLE.DIRECTION_INDICATORS -->`
- **Caption says WHAT only**; push config/setup into `threeparttable` `\tablenotes`; result interpretation goes to text / takeaway box (mirrors the experiment-artifact caption convention). `<!-- policy:FIG.SELF_CONTAINED_CAPTION -->` `<!-- policy:EXP.TAKEAWAY_BOX -->`
- **Variation on every quantitative claim** — report `mean ± std` or `mean (95% CI)`; never a bare point estimate for a stochastic result. `<!-- policy:EXP.ERROR_BARS_REQUIRED -->`
- **Bold the best** per column; consistent significant figures; align decimals (siunitx `S` columns).
- Numbers in the table must match the source artifact exactly (no manual retyping drift). `<!-- policy:CITE.CLAIM_SUPPORT_REQUIRED -->`

## Reproducible generation (CSV/Excel → LaTeX)

Keep a config-driven `results → table` script (same reproducibility discipline as the figure workflow),
so a re-run regenerates the table from raw results rather than hand-edited LaTeX.

```python
import pandas as pd

df = pd.read_csv("results.csv")            # or pd.read_excel("results.xlsx", sheet_name=...)
# 1) format: mean±std strings; 2) bold per-column best; 3) fixed sig figs
def fmt(m, s):                              # mean, std
    return f"{m:.1f}$\\pm${s:.1f}"
best = df.groupby("metric")["mean"].transform("max")  # or min for ↓ metrics
# build a display frame, then:
latex = disp.to_latex(index=False, escape=False,
                      column_format="l" + "S[table-format=2.1]" * k)
```

Prefer pandas `to_latex(..., escape=False)` + siunitx `S` columns for decimal alignment, or
`pgfplotstable` / `csvsimple` to typeset live from CSV. Always diff the rendered numbers against
the raw results file before submission.

## threeparttable skeleton

```latex
\begin{table}[t]\centering
\begin{threeparttable}
\caption{Main results on \textsc{Bench}.}            % WHAT only
\begin{tabular}{l S[table-format=2.1] S[table-format=2.1]}
\toprule
Method & {Acc (\%) $\uparrow$} & {Latency (ms) $\downarrow$} \\
\midrule
Ours   & \bfseries 91.2 & \bfseries 12.3 \\
Base   & 87.4           & 15.1 \\
\bottomrule
\end{tabular}
\begin{tablenotes}\footnotesize
\item Mean over 5 seeds; $\pm$ is std. Config / hyperparameters: \dots
\end{tablenotes}
\end{threeparttable}
\end{table}
```

## When to use

Building any quantitative table for §Experiments (main results / ablation / comparison matrix).
For qualitative ✓/✗/◐ feature matrices (§Background comparison), see `ml-paper-writing`
`references/latex-style-guide.md` — those carry properties, not numbers.
