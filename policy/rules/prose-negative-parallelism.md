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
constraint_type: guardrail
autofix: assisted
lint_patterns:
  - pattern: "(?i)\\bit(?:'s| is) not .{5,60}it(?:'s| is)\\b"
    mode: match
  - pattern: "(?i)\\bnot just .{5,60}but\\b"
    mode: match
coverage_note: "not only ... but also is deliberately not patterned — it is frequency-managed (at most twice per paper), so a single instance is not a violation. Count it by hand during self-review."
lint_targets: "**/*.tex"
---

## Requirement

避免 "It's not X — it's Y" 和 "not just X, but Y" 的否定式排比句式。这是 AI 写作中最常见的假深刻修辞。

在学术论文中，直接陈述观点，不需要先否定一个稻草人来制造转折。

## Rationale

否定式排比是 LLM 生成文本的最强信号之一。Pre-LLM 时代人类极少在正式写作中密集使用这种句式。一篇文章出现一次可以，反复出现就是 AI 痕迹。

## Check

- **regex 搜索**: 匹配 "It's not ... it's" 和 "not just ... but" 模式。**契约式与非契约式都要匹配**——学术散文里缩写本身就被 `PROSE.INFORMAL_VOCABULARY` 排除，所以论文里这个句式几乎总是写成 "It is not X, it is Y"，只匹配 "It's" 等于在真实稿件上永远不触发
- **检查范围**: `.tex` 文件正文区域
- **注意**: "not only ... but also" **刻意不进 regex**——它是频次管理（每篇 ≤2 次可接受），单次命中不是违规，所以逐处报警只会制造噪音。数数归判断层

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
