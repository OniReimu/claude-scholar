---
id: PROSE.NEGATIVE_PARALLELISM
slug: prose-negative-parallelism
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
lint_patterns:
  - pattern: "\\b[Ii]t'?s not .{5,60}it'?s\\b"
    mode: match
  - pattern: "\\bnot just .{5,60}but\\b"
    mode: match
lint_targets: "**/*.tex"
---

## Requirement

避免 "It's not X — it's Y" 和 "not just X, but Y" 的否定式排比句式。这是 AI 写作中最常见的假深刻修辞。

在学术论文中，直接陈述观点，不需要先否定一个稻草人来制造转折。

## Rationale

否定式排比是 LLM 生成文本的最强信号之一。Pre-LLM 时代人类极少在正式写作中密集使用这种句式。一篇文章出现一次可以，反复出现就是 AI 痕迹。

## Check

- **regex 搜索**: 匹配 "It's not ... it's" 和 "not just ... but" 模式
- **检查范围**: `.tex` 文件正文区域
- **注意**: "not only ... but also" 在学术写作中偶尔使用是可接受的，但每篇论文不超过 2 次

## Examples

### Pass

```latex
The proposed method reduces both training time and memory consumption.
```

### Fail

```latex
It's not just about reducing training time -- it's about fundamentally
rethinking how we approach model efficiency.
```
