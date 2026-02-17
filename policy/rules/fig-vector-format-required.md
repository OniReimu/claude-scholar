---
id: FIG.VECTOR_FORMAT_REQUIRED
slug: fig-vector-format-required
severity: error
locked: false
layer: core
artifacts: [figure, code]
phases: [writing-experiments, self-review, camera-ready]
domains: [core]
venues: [all]
check_kind: regex
enforcement: doc
params: {}
conflicts_with: []
lint_patterns:
  - pattern: "\\.savefig\\(['\"].*\\.(png|jpg|jpeg)['\"]"
    mode: match
lint_targets: "**/*.py"
---

## Requirement

所有从数据生成的图表（bar chart、line plot、scatter plot）必须保存为矢量格式（PDF 或 EPS），禁止使用光栅格式（PNG、JPG）。例外：照片和截图可使用 PNG（>=600 DPI）。

## Rationale

矢量格式在缩放和打印时不损失质量，PDF/EPS 是 LaTeX 投稿的标准格式。光栅格式在高分辨率打印时模糊。

## Check

- **Regex 检查**: Python 脚本中 `.savefig()` 是否输出 `.png`/`.jpg`/`.jpeg` 格式
- **LLM 审查**: 图像文件是否为矢量格式

## Examples

### Pass

```python
# 矢量格式输出
fig.savefig("accuracy.pdf")
fig.savefig("loss.eps")

# 例外：截图使用高分辨率 PNG
fig.savefig("screenshot.png", dpi=600)  # 仅限照片/截图
```

### Fail

```python
# 数据图表使用光栅格式
fig.savefig("accuracy.png")
fig.savefig("results.jpg")
fig.savefig("ablation.jpeg")
```
