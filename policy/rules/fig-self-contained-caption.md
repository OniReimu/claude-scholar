---
id: FIG.SELF_CONTAINED_CAPTION
slug: fig-self-contained-caption
severity: warn
locked: false
layer: core
artifacts: [figure, text]
phases: [writing-experiments, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: []
---

## Requirement

图表 caption 必须自包含（self-contained），无需阅读正文即可理解。Caption 应包含：图表展示什么（what）、如何生成/实验设置（how）、关键发现（key finding）。

## Rationale

很多读者先浏览图表再读正文。Self-contained caption 确保图表独立可理解，提升论文可读性。

## Check

- **LLM 检查**: caption 是否仅引用"如上文所述"而缺少具体描述
- **三要素检查**: 是否包含 what/how/finding 三要素

## Examples

### Pass

```latex
\caption{Accuracy comparison on CIFAR-10. We train ResNet-50
with SGD (lr=0.1) for 200 epochs. Our method (blue) outperforms
all baselines by 2.3\% on average.}
% what: Accuracy comparison on CIFAR-10
% how: ResNet-50, SGD, lr=0.1, 200 epochs
% finding: outperforms all baselines by 2.3%
```

### Fail

```latex
% 过于简短，缺少 what/how/finding
\caption{Results of our experiments.}

% 依赖正文，无法独立理解
\caption{See text for details.}

% 只有 what，缺少 how 和 finding
\caption{Training curves on ImageNet.}
```
