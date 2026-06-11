---
id: PROSE.TRAILING_AFTERTHOUGHT
slug: prose-trailing-afterthought
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
  - pattern: ",\\s*(as|if|where|when|once|though|albeit|yet|so|thus|too)\\s+\\w+\\.\\s"
    mode: match
lint_targets: "**/*.tex"
---

## Requirement

禁止在句末用逗号甩一个超短片段（trailing afterthought），如 `..., as editable.`、`..., if needed.`、`..., where applicable.`。这种"想起来补一句"的尾巴是 AI 味的填充，应该并入主句或删除。

把信息写进主句的语法结构里，不要用逗号挂一个零碎短语补全语义。

## Rationale

句末逗号 + 短片段是 LLM 生成文本的高频节奏 tic——主句说完后再轻飘飘补一个限定。它和 `PROSE.SUPERFICIAL_ING_SUFFIX` 是同一类毛病的不同变体：后者抓 `, highlighting...` 的 -ing 尾巴，本规则抓非 -ing 的短尾巴（`, as X.` / `, if X.`）。人类正式写作要么把限定写进主句，要么另起一句。

## Check

- **regex 搜索**: 匹配「逗号 + 连接词（as/if/where/when/once/though/albeit/yet/so/thus/too）+ 1 词 + 句号」
- **LLM 补充**: regex 无法穷尽所有甩尾形式（如 `, in principle.`、`, for now.`），self-review 时一并检查句末是否挂着无主谓的零碎短语
- **排除**: 完整的从句（有主谓，如 `, which we discuss in Section 4.`）；`, e.g., X.` / `, i.e., X.` 这类规范缩写引导
- **检查范围**: `.tex` 文件正文区域

## Examples

### Pass

```latex
The framework exports each record as an editable artifact.
% 把 "editable" 并入主句，而不是甩在句末
```

### Fail

```latex
The framework exports the processing records (Art.~30) and consent
withdrawals (Art.~7(3))~\cite{gdpr2016}, as editable.
% 句末逗号甩一个 "as editable." 短尾巴
```
