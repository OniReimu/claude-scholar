---
id: PROSE.VAGUE_QUANTIFIERS
slug: prose-vague-quantifiers
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
  - pattern: "\\b(some|many|several|a number of|a large amount of|a great deal of|lots of|a lot of|numerous|plenty of|a wide range of)\\b"
    mode: match
lint_targets: "**/*.tex"
---

## Requirement

禁止使用无数据支撑的模糊量词。用具体数字或引用替代。

| 禁用 | 替代 |
|------|------|
| some researchers | Zhang et al.~\cite{} and Li et al.~\cite{} |
| many studies | over 30 studies (surveyed in~\cite{}) |
| several baselines | five baselines |
| a number of | 具体数字 |
| a large amount of | 具体数字 + 单位 |
| a wide range of | 具体范围 |

## Rationale

模糊量词在学术写作中削弱精确度。审稿人会质疑 "many" 到底是多少。用数据说话是技术论文的核心原则。

## Check

- **regex 搜索**: 匹配禁用量词列表
- **排除合法用法**: 引用了具体数据源的量词（如 "several studies~\cite{a,b,c}"）可接受
- **检查范围**: `.tex` 文件正文区域

## Examples

### Pass

```latex
We compare against five state-of-the-art baselines~\cite{a,b,c,d,e}.
Over 30 studies have investigated federated learning privacy (surveyed in~\cite{survey}).
```

### Fail

```latex
Many studies have investigated this problem.
We compare against several baselines.
```
