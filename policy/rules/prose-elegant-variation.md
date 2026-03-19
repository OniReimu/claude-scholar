---
id: PROSE.ELEGANT_VARIATION
slug: prose-elegant-variation
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
constraint_type: guidance
autofix: none
---

## Requirement

同一技术概念在全文中使用**同一术语**，不为"避免重复"而换用同义词。

常见违规示例：
- model / framework / architecture / system 混用指代同一个东西
- method / approach / technique / scheme 混用
- dataset / data / corpus / benchmark 混用指代同一数据集
- training / learning / optimization 混用指代同一过程

在 Introduction 中首次引入术语时明确定义，之后全文一致使用。

## Rationale

学术写作中术语一致性高于文学修辞。在同一篇论文中换用同义词会让读者困惑——"framework" 和 "architecture" 是同一个东西吗？还是两个不同的组件？这也是 AI 生成文本的常见特征（elegant variation）。

## Check

- **LLM 检查**:
  1. 全文中指代核心贡献的术语是否一致
  2. 是否存在同义词交替使用的模式
  3. 每个技术名词是否在首次出现时定义，后续一致使用

## Examples

### Pass

```latex
% 全文一致使用 "framework"
We propose a federated unlearning framework. The framework consists
of three components. We evaluate the framework on four datasets.
```

### Fail

```latex
% 混用 framework/system/architecture
We propose a federated unlearning framework. The system consists
of three modules. We evaluate the architecture on four datasets.
```
