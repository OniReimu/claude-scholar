---
id: FIG.NO_IN_FIGURE_TITLE
slug: fig-no-in-figure-title
severity: error
locked: true
layer: core
artifacts: [figure]
phases: [ideation, writing-system-model, writing-methods, writing-experiments, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: lint_script
params: {}
conflicts_with: []
---

## Requirement

生成任何图表时，禁止在图像画布内添加标题（title）。标题语义放在论文 caption 或正文中。

对 matplotlib/seaborn，禁止使用 `plt.title()`、`ax.set_title()`、`fig.suptitle()`。

## Rationale

图内标题在论文排版中与 LaTeX caption 重复，浪费版面空间。顶会 reviewer 期望标题仅出现在 `\caption{}` 中。图内标题还会在缩放时影响可读性。

## Check

- **lint 脚本**: `skills/paper-figure-generator/scripts/lint_no_title.py` 检测 SVG/PDF/PNG 产物中的画布内大字号标题文字（启发式：顶部居中、大字号文本）
- **代码审查 regex**: `(plt\.title|ax\.set_title|fig\.suptitle)\s*\(` 检测 Python 代码中的标题调用
- **LLM 检查**: 审查生成的图像是否包含画布内文字标题

## Examples

### Pass

```python
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
fig.savefig("loss_curve.pdf")
# caption 在 LaTeX 中: \caption{Training loss over epochs.}
```

### Fail

```python
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title("Training Loss Over Epochs")  # 违规：图内标题
fig.savefig("loss_curve.pdf")
```
