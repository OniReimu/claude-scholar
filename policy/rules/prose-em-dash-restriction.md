---
id: PROSE.EM_DASH_RESTRICTION
slug: prose-em-dash-restriction
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-background, writing-methods, writing-experiments, writing-conclusion, self-review, revision, camera-ready]
domains: [core]
venues: [all]
check_kind: regex
enforcement: lint_script
params: {max_per_paragraph: 0}
conflicts_with: [PROSE.SEMICOLON_RESTRICTION]
constraint_type: guardrail
autofix: assisted
lint_patterns:
  - pattern: "---"
    mode: match
lint_targets: "**/*.tex"
---

## Requirement

禁止使用 em-dash（`---`）。一个都不允许。

**替换标点不算修法——结构必须变。** em-dash 挂着的那截尾巴，换成逗号之后原样还在，而本卡存在的理由正是这种"想起来再补一句"的挂载方式。判据：改完之后，原来被破折号挂着的内容是否已经进入某个语法主结构？

按尾巴的性质选：

| 尾巴是什么 | 改法 |
|---|---|
| 一个完整命题 | (1) **拆成新句子** |
| 对前面名词的限定 | (2) **关系从句**（`, which ...`）——它把尾巴接进从句结构，不是并排挂着 |
| 真正的旁白、可以删掉而不损失论证 | (4) **括号**，或直接删 |

**不推荐逗号插入语**：`X --- a 12.5$\times$ reduction` 改成 `X, a 12.5$\times$ reduction` 只换了标点，同位语仍然是尾巴。要么写成 `X. This is a 12.5$\times$ reduction ...`，要么并进主句。

Em-dash 是强烈的 AI 写作信号。

## Rationale

Em-dash 过度使用是 LLM 生成文本的典型特征。人类学术写作极少在一段内使用多个 em-dash。

## Check

- **regex 匹配**: `.tex` 文件正文区域中出现 `---` 即违规
- **注意区分**: LaTeX em-dash `---` 和 YAML frontmatter `---`，仅检查正文区域

## Examples

### Pass

```latex
% 全段无 em-dash
Our method builds on prior work and outperforms all baselines, including recent
state-of-the-art approaches, by a large margin.

% 用逗号插入语替代 em-dash
Our method, the first to combine both techniques, outperforms all baselines.
```

### Fail

```latex
Our method---which builds on prior work---outperforms baselines---including
recent state-of-the-art approaches---by a large margin.
```
