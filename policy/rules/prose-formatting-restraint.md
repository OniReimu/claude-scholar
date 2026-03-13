---
id: PROSE.FORMATTING_RESTRAINT
slug: prose-formatting-restraint
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-background, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_style
enforcement: doc
params: {}
conflicts_with: []
---

## Requirement

正文中保持格式克制：

1. **Bold**（`\textbf{}`）：仅用于首次定义的核心概念或术语，不用于强调普通词语
2. **Italic**（`\textit{}`）：仅用于术语引入或外来语，不用于情绪强调
3. **Bullet/numbered list**：正文段落用连贯散文，不在段落中间插入 itemize。Contribution section 和 algorithm 描述中的 enumerate 除外
4. 保持论文模板的原生格式，不额外添加装饰性排版

## Rationale

过度格式化是 AI 生成内容的常见特征（尤其是 bold 滥用和内联列表）。经典工程论文的排版是克制的，信息通过文字逻辑而非视觉格式传达。

## Check

- **LLM 检查**:
  - 每页 `\textbf{}` 出现次数是否超过 3 次（Contribution/Definition 除外）
  - 正文段落中是否嵌入了 `itemize` 环境（应改为散文或独立成段）
  - 是否有仅用于情绪强调的 bold/italic（如 `\textbf{significantly}`）

## Examples

### Pass

```latex
We define the \textbf{unlearning completeness} as the degree to which
a model forgets the target data. The system is evaluated through
simulations on three benchmark datasets.
```

### Fail

```latex
Our method achieves \textbf{significant} improvements across
\textbf{all} metrics. The key advantages are:
\begin{itemize}
  \item \textbf{Higher accuracy}
  \item \textbf{Lower latency}
  \item \textbf{Better scalability}
\end{itemize}
```
