---
name: results-analysis
description: This skill should be used when the user asks to "analyze experimental results", "generate results section", "statistical analysis of experiments", "compare model performance", "create results visualization", or mentions connecting experimental data to paper writing. Provides comprehensive guidance for analyzing ML/AI experimental results and generating paper-ready content.
tags: [Research, Analysis, Statistics, Visualization, Paper Writing]
version: 0.1.1
---

# Results Analysis for ML/AI Research

A systematic experimental results analysis workflow connecting experimental data to paper writing.

## Policy Rules

> 本 skill 执行以下论文写作规则。权威定义在 `policy/rules/`。
> 行内出现处以 HTML 注释标记引用。**冲突时以 `policy/rules/` 为准。**

| Rule ID | 摘要 |
|---------|------|
| `FIG.NO_IN_FIGURE_TITLE` | 图内不加标题 |
| `FIG.FONT_GE_24PT` | 图表字号 ≥ 24pt |
| `FIG.ONE_FILE_ONE_FIGURE` | 1 文件 = 1 图 |
| `FIG.VECTOR_FORMAT_REQUIRED` | 数据图用矢量格式 |
| `FIG.COLORBLIND_SAFE_PALETTE` | 色盲安全配色 |
| `FIG.SELF_CONTAINED_CAPTION` | Caption三要素 |
| `TABLE.BOOKTABS_FORMAT` | 使用 booktabs 格式 |
| `TABLE.DIRECTION_INDICATORS` | 表头方向指示符 |
| `EXP.ERROR_BARS_REQUIRED` | 实验需误差线 |
| `EXP.TAKEAWAY_BOX` | 实验结果Takeaway |
| `EXP.ABLATION_IN_RESULTS` | 消融实验在Results |
| `EXP.RESULTS_SUBSECTION_STRUCTURE` | 实验小节结构 |
| `EXP.FABRICATED_RESULTS_CAPTION_DISCLOSURE` | 非实跑结果 caption 强制披露 |
| `EXP.RESULTS_STATUS_DECLARATION_REQUIRED` | 非实跑结果小节状态声明 |

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

Execution mode gate:
- If user already specifies mode at task start, use it directly.
- Otherwise, ask once: `ACTUAL_RUN` or `FABRICATED_PLACEHOLDER`.
- `ACTUAL_RUN`: requires raw result files before analysis.
- `FABRICATED_PLACEHOLDER`: workflow stops at code implementation for this round (no real execution claims); allow draft placeholders only with explicit fabricated disclosures. <!-- policy:EXP.RESULTS_STATUS_DECLARATION_REQUIRED --> <!-- policy:EXP.FABRICATED_RESULTS_CAPTION_DISCLOSURE -->

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
- Mode consistency check - If `ACTUAL_RUN`, raw result files must exist; if only placeholder data exists, switch to `FABRICATED_PLACEHOLDER` and disclose status

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

> **Implementation delegate**: 绘图代码实现使用 `scientific-figure-making` skill（[figures4papers](https://github.com/ChenLiu-1996/figures4papers)，via `vendor/figures4papers` submodule）。该 skill 提供 `apply_publication_style()`、`make_grouped_bar()`、`make_trend()`、`make_heatmap()` 等 helper 函数和语义化配色系统。
> 字号、配色、导出格式等细节由 `scientific-figure-making` 的 `FigureStyle` 和 `finalize_figure()` 全权处理。

#### 4a. Plotting Implementation（delegate 给 scientific-figure-making）

生成绘图代码时，**优先使用 `scientific-figure-making` skill 的 API**：

1. **Read** `skills/scientific-figure-making/references/api.md` — 获取函数签名、PALETTE 定义、FigureStyle 配置
2. **Read** `skills/scientific-figure-making/references/common-patterns.md` — 获取 layout patterns（ultra-wide 画布、dedicated legend panel、print-safe bars 等）
3. **使用 helper 函数**生成代码：
   - `apply_publication_style(style)` — 设置 rcParams（字号、线宽、配色由 FigureStyle 管理）
   - `make_grouped_bar()` / `make_trend()` / `make_heatmap()` / `make_scatter()` — 生成图表
   - `finalize_figure(fig, out_path, formats=['pdf'], dpi=300)` — 导出（格式由此函数管理）

#### 4b. 硬约束（仅 2 条 active policy rules）

| Policy Rule | 要求 |
|-------------|------|
| `FIG.NO_IN_FIGURE_TITLE` | 禁止 `plt.title()` / `ax.set_title()` / `fig.suptitle()`，标题只放 LaTeX caption |
| `FIG.ONE_FILE_ONE_FIGURE` | 1 文件 = 1 图。禁止 `plt.subplots(n, m)` 拼多个独立数据图。复合布局用 LaTeX `\subfigure`。Dedicated legend panel（`ax.set_axis_off()` + legend）视为同一语义单元，允许 |

<!-- policy:FIG.NO_IN_FIGURE_TITLE --> <!-- policy:FIG.ONE_FILE_ONE_FIGURE -->

> **其余 fig-* rules**（FONT_GE_24PT、VECTOR_FORMAT_REQUIRED、COLORBLIND_SAFE_PALETTE、SELF_CONTAINED_CAPTION、SYSTEM_OVERVIEW_ASPECT_RATIO）已退役为 `severity: info`，由 `scientific-figure-making` 的 API 和 design conventions 接管。

#### 4c. Visualization Selection Guide

匹配数据特征选择最合适的图表类型：

| Data Characteristic | Best Visualization | When to Use |
|--------------------|-------------------|-------------|
| Trend / convergence over epochs | **Line plot** (`make_trend`) | Training curves, learning rate schedules, performance over time |
| Performance comparison across methods | **Bar chart** (`make_grouped_bar`) | Ablation studies, comparing 3-8 methods on 1-3 metrics |
| Distribution / outliers across runs | **Box plot** or **violin plot** | Showing variance, comparing distributions across groups |
| Multi-objective tradeoff | **Pareto front** or **scatter matrix** (`make_scatter`) | Accuracy vs latency, accuracy vs cost |
| Component contribution | **Waterfall chart** or **stacked bar** | Ablation showing cumulative contribution of each module |
| Fairness / group differences | **Grouped box plot** with CI error bars | Comparing performance across demographic groups |
| Feature importance / attention | **Heatmap** (`make_heatmap`) | Attention weights, correlation matrices, confusion matrices |
| High-dimensional embeddings | **t-SNE / UMAP scatter** (`make_scatter`) | Cluster visualization, representation quality analysis |
| Sensitivity to hyperparameter | **Line plot with shaded CI** (`make_trend`) | Sweeping one hyperparameter while showing uncertainty |

**When to use figures vs tables:**
- **Figures (Python plots)**: Data is sparse, need to show trends/distributions/relationships, fewer than ~20 data points per comparison, spatial encoding adds meaning
- **Tables (`booktabs` + `\resizebox`)**: Dense numerical results, many metrics (5+) AND/OR many baselines (5+), readers need exact numbers, double-column (`table*`) for large comparison matrices <!-- policy:TABLE.BOOKTABS_FORMAT --> <!-- policy:TABLE.DIRECTION_INDICATORS -->

#### 4d. Layout Patterns（from scientific-figure-making）

参考 `skills/scientific-figure-making/references/common-patterns.md`，核心 patterns：

1. **Ultra-wide 画布** — `figsize=(45, 12)`, 宽高比 3-4:1，避免标签拥挤
2. **Dedicated legend panel** — 独立 subplot 放图例，数据区域保持干净
3. **No x-tick labels** — 用 legend 代替 x 轴标签（当 x 轴是 method/condition 时）
4. **Dynamic Y-axis** — `data.min() - margin` to `data.max() + margin`，突出差异
5. **Hatching + edge** — `edgecolor='black'` + hatch patterns，保证灰度可读
6. **Semantic color** — 蓝=ours, 绿=improvement, 红=baseline, 灰=neutral

See `references/visualization-best-practices.md` and `skills/scientific-figure-making/references/design-theory.md` for additional details.

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

### Ablation Study <!-- policy:EXP.ABLATION_IN_RESULTS -->
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
- [ ] All values include error bars/confidence intervals <!-- policy:EXP.ERROR_BARS_REQUIRED -->
- [ ] Statistical test methods are specified
- [ ] Figures are clear and readable (including black-and-white print)
- [ ] No in-figure title text is used <!-- policy:FIG.NO_IN_FIGURE_TITLE -->
- [ ] If any result is fabricated/synthetic/dummy, caption contains red uppercase disclosure <!-- policy:EXP.FABRICATED_RESULTS_CAPTION_DISCLOSURE -->
- [ ] If a subsection contains fabricated results, add `% [FABRICATED] ...` status declaration comment <!-- policy:EXP.RESULTS_STATUS_DECLARATION_REQUIRED -->
- [ ] Hyperparameter search ranges are reported
- [ ] Computational resources are specified (GPU type, time) <!-- policy:REPRO.COMPUTE_RESOURCES_DOCUMENTED -->
- [ ] Random seed settings are specified (per `rules/experiment-reproducibility.md`) <!-- policy:REPRO.RANDOM_SEED_DOCUMENTATION -->
- [ ] Config file saved alongside results (Hydra / OmegaConf snapshot)
- [ ] Environment recorded (Python version, GPU driver, key library versions)
- [ ] Results are reproducible (code/data available)

## Orchestrator Integration

This skill owns stage: **`analysis`**.

When invoked within an active research run (see `orchestrator/run-card.md`):

1. **Stage start**: Mark `analysis` → `in_progress`; verify `experiments` stage is `done` (data_path exists + fingerprinted).
2. **Analysis**: Execute the standard analysis pipeline; enforce experiment status disclosure rules.
3. **Stage end**: Prefer `fingerprintStageArtifacts({ cwd, run, stageId: 'analysis' })` so contract-declared files are tracked deterministically; persist `tracked_files` + `fingerprints`, then request human approval.
4. **Gate**: Run experiment status disclosure check before marking `done`.

**Expected artifacts** (files):
- `analysis-output/analysis-report.md`
- `analysis-output/results-draft.md`
- `analysis-output/visualization-specs.md`

**Gate execution and persistence**:

The `analysis` gate enforces experiment status disclosure rules (`EXP.FABRICATED_RESULTS_CAPTION_DISCLOSURE` and `EXP.RESULTS_STATUS_DECLARATION_REQUIRED`). After verification, persist results into run state:

```
gate_results.analysis = {
  last_run: "<ISO timestamp>",
  passed: true|false,
  summary: "<disclosure check outcome: all results verified as ACTUAL_RUN | fabricated results detected and properly disclosed | FAIL: undisclosed fabricated results>"
}
```

The stage may only be marked `done` if `gate_results.analysis.passed === true` **and** the user approves the analysis. If fabricated results are detected without proper disclosure, the gate fails and the user must either add disclosure markers or confirm the data is from actual runs.

If no active run exists, initialize one with `initRun()`.

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
- Using non-colorblind-friendly palettes <!-- policy:FIG.COLORBLIND_SAFE_PALETTE -->
- Y-axis not starting from 0 (exaggerating differences)
- Missing error bars <!-- policy:EXP.ERROR_BARS_REQUIRED -->
- Adding chart titles inside figure canvas <!-- policy:FIG.NO_IN_FIGURE_TITLE -->
- Overly complex figures

✅ **Correct approach:**
- Use Okabe-Ito or Paul Tol palettes <!-- policy:FIG.COLORBLIND_SAFE_PALETTE -->
- Set reasonable axis ranges
- Include error bars and confidence intervals <!-- policy:EXP.ERROR_BARS_REQUIRED -->
- Keep titles in caption/paper text, not in figure canvas <!-- policy:FIG.NO_IN_FIGURE_TITLE -->
- Keep figures clean and clear

### Writing Errors

❌ **Wrong approach:**
- Over-interpreting results
- Not describing experimental setup
- Hiding negative results
- Missing statistical information
- Presenting fabricated placeholders as if they were actual execution outputs

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
