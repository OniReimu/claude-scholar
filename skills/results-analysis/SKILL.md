---
name: results-analysis
description: This skill should be used when the user asks to "analyze experimental results", "generate results section", "statistical analysis of experiments", "compare model performance", "create results visualization", or mentions connecting experimental data to paper writing. Provides comprehensive guidance for analyzing ML/AI experimental results and generating paper-ready content.
tags: [Research, Analysis, Statistics, Visualization, Paper Writing]
version: 0.1.1
---

# Results Analysis for ML/AI Research

A systematic experimental results analysis workflow connecting experimental data to paper writing.

## Core Features

This skill provides three core capabilities:

1. **Experimental Data Analysis** - Read and analyze experimental data in various formats
2. **Statistical Validation** - Perform statistical significance tests and performance comparisons
3. **Paper Content Generation** - Generate text and visualizations for the Results section

## When to Use

Use this skill when you need to:
- Analyze experimental results (CSV, JSON, TensorBoard logs)
- Generate the Results section of a paper
- Compare performance across multiple models
- Perform statistical significance tests
- Create publication-quality **data-driven** visualizations (bar charts, line plots, heatmaps, scatter plots)
- Validate the reliability of experimental results

> **Note:** This skill handles **data-driven** plots via matplotlib/seaborn. For **conceptual diagrams** (system overviews, pipeline figures, architecture diagrams, threat models), use the `paper-figure-generator` skill instead (generates editable SVG via AutoFigure-Edit).

## Workflow

### Standard Analysis Pipeline

```
Data Loading → Data Validation → Statistical Analysis → Visualization → Writing → Quality Check
```

### Step 1: Data Loading and Validation

**Supported Data Formats:**
- CSV files - Tabular data
- JSON files - Structured results
- TensorBoard logs - Training curves
- Python pickle - Complex objects

**Data Validation Checks:**
- Completeness check - Missing values, outliers
- Consistency check - Data format, units
- Reproducibility check - Random seeds, version info

Select appropriate tools for data loading and preliminary validation based on data format.

### Step 2: Statistical Analysis

**MANDATORY: Pre-checks before choosing a test** (see `references/statistical-methods.md`):
1. **Normality test**: Run Shapiro-Wilk test (n < 50) or Kolmogorov-Smirnov test (n ≥ 50), inspect Q-Q plots
2. **Variance homogeneity**: Run Levene's test (robust) or Bartlett's test (if normal)
3. **Select test based on results**:
   - Data is normal + equal variance → parametric tests (t-test, ANOVA)
   - Data is NOT normal or unequal variance → non-parametric tests (Wilcoxon, Mann-Whitney U, Kruskal-Wallis)

**Basic Statistics:**
- Mean ± Standard Deviation (report variance across runs)
- Standard Error (for confidence interval estimation)
- 95% Confidence Interval

**Significance Tests** (choose based on pre-check results):

| Scenario | Normal Data | Non-Normal Data |
|----------|-------------|-----------------|
| 2 groups | t-test (paired/independent) | Wilcoxon signed-rank / Mann-Whitney U |
| 3+ groups | One-way ANOVA | Kruskal-Wallis |
| Multiple comparisons | Bonferroni / Tukey HSD | Dunn's test |

**Key Principles:**
- Report complete statistical information (mean ± std, n runs, seeds used)
- Specify the test method and significance level (α = 0.05 by default)
- Report p-values AND effect sizes (Cohen's d, η²)
- Apply multiple comparison correction when testing multiple hypotheses
- Ensure reproducibility: document random seeds per `rules/experiment-reproducibility.md`

See `references/statistical-methods.md` for the complete statistical methods guide.

### Step 3: Model Performance Comparison

**Comparison Dimensions:**
- Accuracy/Performance metrics
- Training time/Inference speed
- Model complexity/Parameter count
- Robustness/Generalization ability

**Comparison Methods:**
- Baseline comparison - Compare with existing methods
- Ablation study - Validate component contributions
- Cross-dataset validation - Test generalization

Systematically compare performance across different methods, ensuring fair comparison.

### Step 4: Visualization

**Publication-Quality Visualization Requirements:**
- Vector format (PDF/EPS)
- Colorblind-friendly palette
- Clear labels and legends
- Appropriate error bars
- No in-figure title text (`plt.title` / `set_title` / `suptitle` forbidden)
- Readable in black-and-white print

**Visualization Selection Guide** — match data characteristics to the right figure type:

| Data Characteristic | Best Visualization | When to Use |
|--------------------|-------------------|-------------|
| Trend / convergence over epochs | **Line plot** | Training curves, learning rate schedules, performance over time |
| Performance comparison across methods | **Bar chart** | Ablation studies, comparing 3-8 methods on 1-3 metrics |
| Distribution / outliers across runs | **Box plot** or **violin plot** | Showing variance, comparing distributions across groups |
| Multi-objective tradeoff | **Pareto front** or **scatter matrix** | Accuracy vs latency, accuracy vs cost, multi-dimensional tradeoffs |
| Component contribution | **Waterfall chart** or **stacked bar** | Ablation showing cumulative contribution of each module |
| Fairness / group differences | **Grouped box plot** with CI error bars | Comparing performance across demographic groups |
| Feature importance / attention | **Heatmap** | Attention weights, correlation matrices, confusion matrices |
| High-dimensional embeddings | **t-SNE / UMAP scatter** | Cluster visualization, representation quality analysis |
| Sensitivity to hyperparameter | **Line plot with shaded CI** | Sweeping one hyperparameter while showing uncertainty |

**When to use figures vs tables:**
- **Figures (Python plots)**: Data is sparse, need to show trends/distributions/relationships, fewer than ~20 data points per comparison, spatial encoding adds meaning
- **Tables (`booktabs` + `\resizebox`)**: Dense numerical results, many metrics (5+) AND/OR many baselines (5+), readers need exact numbers, double-column (`table*`) for large comparison matrices

**Figure quality reference**: Follow [figures4papers](https://github.com/ChenLiu-1996/figures4papers) for publication-ready Python plotting — consistent style, proper font sizes, colorblind-safe palettes (Okabe-Ito or Paul Tol), no chart junk. Always save as PDF vector format.

**CRITICAL — Font size and line width** (common mistake: too small after scaling):

```python
# 在每个绘图脚本开头设置，确保缩放到论文列宽后仍可读
plt.rcParams.update({
    'font.size': 28,           # 全局默认
    'axes.labelsize': 30,      # x/y 轴标签
    'xtick.labelsize': 26,     # 刻度标签
    'ytick.labelsize': 26,
    'legend.fontsize': 26,     # 图例
    'lines.linewidth': 3.0,    # 线宽
    'lines.markersize': 10,    # 标记点
    'axes.linewidth': 2.0,     # 坐标轴线宽
})
```

**所有文字必须 ≥ 24pt**（源文件中的 matplotlib pt 值，非打印尺寸）。

**Accessibility requirements:**
- Use **colorblind-safe palettes**: Okabe-Ito (8 colors) or Paul Tol (up to 12 colors)
- Verify **grayscale readability** (8% of men have color vision deficiency)
- Differentiate lines by **style** (solid/dashed/dotted), not just color
- Save as **PDF vector format**: `plt.savefig('fig.pdf', bbox_inches='tight')`
- **1 file = 1 figure**: Do NOT use `plt.subplots()` to combine multiple plots. Each plot is a separate file. Composite layouts are handled in LaTeX via `\subfigure`.
- If multiple plots share a legend, save the legend as a separate image file
- Source font size ≥ 24pt, line width ≥ 2.5pt
- Put title semantics in caption/text, not inside the figure canvas

See `references/visualization-best-practices.md` for additional details.

### Step 5: Writing the Results Section

**Results Section Structure:**

```markdown
## Results

### Overview of Main Findings
[1-2 paragraphs summarizing core results]

### Experimental Setup
[Brief description of experimental configuration; details in appendix]

### Performance Comparison
[Comparison with baseline methods, including tables and figures]

### Ablation Study
[Validate contributions of each component]

### Statistical Significance
[Report statistical test results]

### Qualitative Analysis
[Case studies, visualization examples]
```

**Writing Principles:**
- Clearly state the hypothesis each experiment validates
- Guide readers to observe key phenomena: "Figure X shows..."
- Report complete statistical information
- Honestly report limitations

**Key Sentence Patterns** (see `references/results-writing-guide.md` for the full list):

| Context | Pattern |
|---------|---------|
| Introduce experiment | "To validate [hypothesis], we conducted [experiment]" |
| Describe results | "Our method achieves [value], outperforming [baseline]'s [value]" |
| Statistical significance | "This difference is statistically significant (p < 0.01)" |
| Ablation | "Removing [component] decreases performance by [value], indicating [conclusion]" |
| Figure reference | "Figure X shows [phenomenon]. We observe that [key observation]" |

See `references/results-writing-guide.md` for the complete writing guide.

### Step 6: Quality Check

**Checklist:**
- [ ] All values include error bars/confidence intervals
- [ ] Statistical test methods are specified
- [ ] Figures are clear and readable (including black-and-white print)
- [ ] No in-figure title text is used
- [ ] Hyperparameter search ranges are reported
- [ ] Computational resources are specified (GPU type, time)
- [ ] Random seed settings are specified (per `rules/experiment-reproducibility.md`)
- [ ] Config file saved alongside results (Hydra / OmegaConf snapshot)
- [ ] Environment recorded (Python version, GPU driver, key library versions)
- [ ] Results are reproducible (code/data available)

## Common Mistakes and Pitfalls

### Statistical Errors

❌ **Wrong approach:**
- Reporting only the best results (cherry-picking)
- Confusing standard deviation and standard error
- Not reporting statistical significance
- Not correcting for multiple comparisons

✅ **Correct approach:**
- Report all experimental results
- Clearly specify whether standard deviation or standard error is used
- Perform appropriate statistical tests
- Use Bonferroni or similar correction methods

### Visualization Errors

❌ **Wrong approach:**
- Using non-colorblind-friendly palettes
- Y-axis not starting from 0 (exaggerating differences)
- Missing error bars
- Adding chart titles inside figure canvas
- Overly complex figures

✅ **Correct approach:**
- Use Okabe-Ito or Paul Tol palettes
- Set reasonable axis ranges
- Include error bars and confidence intervals
- Keep titles in caption/paper text, not in figure canvas
- Keep figures clean and clear

### Writing Errors

❌ **Wrong approach:**
- Over-interpreting results
- Not describing experimental setup
- Hiding negative results
- Missing statistical information

✅ **Correct approach:**
- Objectively describe observed phenomena
- Provide sufficient experimental details
- Honestly report all results
- Report complete statistical information

See `references/common-pitfalls.md` for the complete error patterns and fixes.

## Integration with Paper Writing

### Collaboration with ml-paper-writing Skill

This skill focuses on experimental results analysis and works in tandem with the `ml-paper-writing` skill:

**results-analysis handles:**
- Data analysis and statistical tests
- Visualization generation
- Results interpretation

**ml-paper-writing handles:**
- Complete paper structure
- Citation management
- Conference format requirements

**Workflow Integration:**
```
Experiments complete → results-analysis analyzes
    ↓
Generate analysis report and visualizations
    ↓
ml-paper-writing integrates into paper
    ↓
Complete Results section
```

### Output Format

After analysis, the following are generated:

1. **Analysis Report** (`analysis-report.md`)
   - Statistical summary
   - Key findings
   - Suggested figures

2. **Visualization Files** (`figures/`)
   - PDF format figures
   - Standalone figure captions

3. **Results Draft** (`results-draft.md`)
   - Text ready for direct use in the paper
   - Includes figure references

## Examples and Templates

### Example Files

Refer to the `examples/` directory for complete examples:

- **`example-analysis-report.md`** - Complete analysis report example
- **`example-results-section.md`** - Paper Results section example

### Workflow Overview

The complete analysis pipeline includes:

1. **Data Loading** - Read results from experiment output files
2. **Statistical Analysis** - Compute basic statistics and perform significance tests
3. **Visualization** - Create publication-quality figures
4. **Report Generation** - Integrate analysis results and visualizations

See the guides in the `references/` directory for detailed methods and best practices.

## Reference Resources

### Detailed Guides

- **`references/statistical-methods.md`** - Complete statistical methods guide
- **`references/results-writing-guide.md`** - Results section writing standards
- **`references/visualization-best-practices.md`** - Visualization best practices
- **`references/common-pitfalls.md`** - Common errors and fixes

### External Resources

- [Nature Statistics Checklist](https://www.nature.com/documents/nr-reporting-summary-flat.pdf)
- [Science Reproducibility Guidelines](https://www.science.org/content/page/science-journals-editorial-policies)
- [NeurIPS Paper Checklist](https://neurips.cc/Conferences/2025/PaperInformation/PaperChecklist)

## Best Practices Summary

### Data Analysis

✅ **Recommended:**
- Run experiments multiple times (at least 3-5 runs)
- Report complete statistical information
- Use appropriate statistical tests
- Check data completeness

❌ **Prohibited:**
- Cherry-picking best results
- Ignoring statistical significance
- Hiding negative results
- Not reporting experimental setup

### Visualization

✅ **Recommended:**
- Use vector format
- Colorblind-friendly palettes
- Include error bars
- Clear labels

❌ **Prohibited:**
- Raster formats (PNG/JPG)
- Misleading axis scales
- Overly complex figures
- Missing legends

### Writing

✅ **Recommended:**
- Objectively describe results
- Provide sufficient detail
- Honestly report limitations
- Guide reader attention

❌ **Prohibited:**
- Over-interpretation
- Hiding details
- Exaggerating effects
- Vague descriptions

## Summary

This skill provides a systematic experimental results analysis workflow:

1. **Data Loading and Validation** - Ensure data quality
2. **Statistical Analysis** - Perform appropriate statistical tests
3. **Model Comparison** - Systematic performance comparison
4. **Visualization** - Publication-quality figures
5. **Writing** - Results section content
6. **Quality Check** - Ensure reproducibility

Following these principles produces high-quality, reproducible experimental results analysis that meets top conference standards.
