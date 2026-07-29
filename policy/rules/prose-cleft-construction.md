---
id: PROSE.CLEFT_CONSTRUCTION
slug: prose-cleft-construction
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
constraint_type: guardrait
autofix: none
lint_patterns:
  - pattern: "\\b(which|that) is what\\b"
    mode: match
  - pattern: "\\bWhat [a-z][a-z ]{2,28} (is|does|makes|made|sets|gives)\\b"
    mode: match
  - pattern: "\\bIt is [a-z][^.]{2,35} that (is|was|are|were|has|have|had|makes|made|sets|gives|governs|drives|determines)\\b"
    mode: match
lint_targets: "**/*.tex"
---

## Requirement

用主谓宾直接陈述，不用分裂句（cleft）把成分前置来制造强调。禁止 `What X is is Y` / `That is what X does` / `which is what makes Y` / `It is X that does Y` 这类结构，改写为 `X is Y` / `X does Y`。

## Rationale

分裂句是把一个简单陈述拆成"框架 + 焦点"两段，读者必须先解析框架才拿得到信息。在口语和演讲里这个延迟制造悬念，在学术论文里它只是让读者多走一步。

这是 AI 生成文本最稳定的句法指纹之一：模型倾向用分裂句来"显得在强调"，而人类作者通常直接把重要的词放在主语位置。它也常常和悬空指代同时出现（`That is what...` 里的 `That` 往往没有明确先行词）。

单条出现不致命，密度是信号。一份 6000 字符的文档里出现 6 处分裂句，读起来就是通篇在"作强调"而没有实质递进。

## Check

- **regex 搜索**：`which/that is what`、`What X is/does/makes` 前置、`It is X that VERB`
- **检查范围**：`.tex` 正文；rebuttal / response 的 markdown 由 `writing-anti-ai` 工作流人工过一遍，linter 的 `lint_targets` 只吃单一扩展名
- **已知误报**：`It is now run to that specification` 这类"介词 + that"不是分裂句，pattern 3 通过要求 `that` 后接动词来规避，仍需人工确认
- **不适用**：直接引语内部（引审稿人或他人原文时保持原样）

## Examples

### Pass

```latex
The measured correlation sets the effective vote count.
Adaptive-K depends on that sensitivity.
The estimator has been validated for decades, which is why a measured $c$ can be trusted.
```

### Fail

```latex
That is what sets the effective vote count.
That sensitivity is what Adaptive-K needs.
The estimator has been validated for decades, which is what makes a measured $c$ trustworthy.
What the intervention establishes is that $\hat{c}$ is not fixed by the question set.
```
