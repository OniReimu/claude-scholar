---
id: PROSE.GERUND_FRAGMENT_LITANY
slug: prose-gerund-fragment-litany
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

禁止使用独立的分词片段（gerund fragment）堆叠来举例或展开论点。每个句子必须是完整的、有主语和谓语的句子。

## Rationale

分词片段堆叠（"Fixing bugs. Writing features. Shipping code."）是 AI 生成文本的结构性 tic。这些片段没有语法主语，不构成完整句子，在学术写作中不可接受。人类不会自然地用这种方式写作。

## Check

- **LLM 检查**: 检测连续 2+ 个以动名词/现在分词开头的独立片段（句号结尾但无主语）
- **排除**: itemize/enumerate 环境中的列表项

## Examples

### Pass

```latex
The framework supports multiple operations including bug detection,
feature extraction, and model deployment.
```

### Fail

```latex
The framework supports multiple operations. Detecting bugs in
production models. Extracting features from raw data. Deploying
models to edge devices. Monitoring performance in real time.
```
