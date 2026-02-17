---
id: FIG.ONE_FILE_ONE_FIGURE
slug: fig-one-file-one-figure
severity: error
locked: true
layer: core
artifacts: [figure]
phases: [writing-experiments, self-review, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: []
---

## Requirement

每个 Python 绘图脚本只生成一个图表，保存为一个文件。禁止使用 `plt.subplots(n, m)` 将多个子图组合成复合图。复合布局（并排对比、多条件网格）在 LaTeX 中通过 `\subfigure` 实现，不在 Python 中实现。

## Rationale

单文件单图便于 LaTeX `\includegraphics` 引用、版本管理和独立替换。复合布局由 LaTeX `\subfigure` 控制，允许论文排版阶段灵活调整子图排列、间距和 caption，无需重新运行 Python。如果多个子图共享图例，将图例单独保存为一个图像文件。

## Check

- **LLM 检查**: 审查脚本中是否使用 `plt.subplots(n, m)`（n*m > 1）将多个图合并为一个文件
- **要点**: 每个输出文件对应一个图语义单元（一个 plot、一个 shared legend、一个 colorbar 等）。一个脚本可以多次 `savefig()` 输出多个文件，但每个文件只含一个语义单元
- **允许**: `fig, ax = plt.subplots()` 创建单个 axes 是合法的（等价于 `plt.figure()`）

## Examples

### Pass

```python
# plot_accuracy.py — 只生成 accuracy 图
fig, ax = plt.subplots()
ax.plot(epochs, accuracy)
fig.savefig("accuracy.pdf")

# plot_loss.py — 只生成 loss 图
fig, ax = plt.subplots()
ax.plot(epochs, loss)
fig.savefig("loss.pdf")

# 在 LaTeX 中组合:
# \subfigure[Accuracy]{\includegraphics{accuracy.pdf}}
# \subfigure[Loss]{\includegraphics{loss.pdf}}
```

### Fail（Python subplots 复合图）

```python
# 违规：在 Python 中用 subplots 组合多个图
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(x, y1)
axes[1].plot(x, y2)
axes[2].plot(x, y3)
fig.savefig("ablation_all.pdf")
# 应拆分为 3 个独立脚本/文件，在 LaTeX 中用 \subfigure 组合
```

### Fail（覆盖同一文件）

```python
fig1, ax1 = plt.subplots()
ax1.plot(epochs, accuracy)
fig1.savefig("results.pdf")  # accuracy 图

fig2, ax2 = plt.subplots()
ax2.bar(methods, scores)
fig2.savefig("results.pdf")  # 覆盖了上面的文件！
```
