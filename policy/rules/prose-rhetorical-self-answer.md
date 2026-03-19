---
id: PROSE.RHETORICAL_SELF_ANSWER
slug: prose-rhetorical-self-answer
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-background, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: regex
enforcement: lint_script
params: {}
conflicts_with: []
constraint_type: guardrail
autofix: none
lint_patterns:
  - pattern: "\\?\\s+(A |An |The |It |This |That |One |Not |Yes|No|Devastating|Significant|Simple)"
    mode: match
lint_targets: "**/*.tex"
---

## Requirement

避免 "The X? A Y." 式的自问自答修辞。AI 自己提出一个问题然后立刻用短句回答来制造戏剧效果。学术论文应直接陈述结论，不用修辞性问答。

## Rationale

自问自答是博客和演讲的修辞手法，不适合学术论文的正式语域。在学术写作中，论点应通过论证和数据建立，不通过修辞技巧。这也是 AI 生成文本的常见模式。

## Check

- **regex 搜索**: 问号后紧接短句回答的模式
- **检查范围**: `.tex` 文件正文区域
- **排除**: Research Questions 章节中的正式 RQ 声明

## Examples

### Pass

```latex
The attack success rate reaches 97.3\%, which demonstrates that the
defense mechanism is ineffective against adversarial perturbations.
```

### Fail

```latex
The result? A 97.3\% attack success rate. The implication? The entire
defense mechanism is fundamentally flawed.
```
