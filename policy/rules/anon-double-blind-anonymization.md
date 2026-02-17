---
id: ANON.DOUBLE_BLIND_ANONYMIZATION
slug: anon-double-blind-anonymization
severity: error
locked: true
layer: venue
artifacts: [text, bibtex]
phases: [self-review, camera-ready]
domains: [core]
venues: [neurips, icml, iclr, acl, aaai, colm]
check_kind: manual
enforcement: doc
params: {}
conflicts_with: []
---

## Requirement

双盲投稿必须完全匿名化：

1. **删除作者姓名和单位** — `\author{}` 使用匿名模板
2. **匿名化自引** — 使用 "[Anonymous, 2024]" 替代自身引用
3. **删除含识别信息的致谢** — 致谢中不可包含资助编号、实验室名称等
4. **匿名化 GitHub/代码链接** — 使用 "Anonymous repository" 替代真实 URL

## Rationale

双盲审稿是所有顶会的强制要求。违反匿名化 = desk rejection，无论论文质量。

## Check

- **人工审查**: 检查 `\author{}` 是否为匿名模板
- **自引检查**: 自引是否匿名化，不包含作者真实姓名
- **链接检查**: GitHub URL 是否匿名，不包含真实用户名
- **致谢检查**: 致谢是否删除或匿名化

## Examples

### Pass

```latex
\author{Anonymous}

% 自引匿名化
As shown in prior work~\citep{anonymous_2024_method}, the approach is effective.

% 代码链接匿名化
Code is available at \url{https://anonymous.4open.science/r/XXXX}.
```

### Fail

```latex
\author{John Smith \\ MIT}

% 自引暴露身份
As shown in our previous work~\citep{smith_2023_deep}, which we developed at MIT...

% GitHub URL 包含真实用户名
Code is available at \url{https://github.com/johnsmith/our-method}.
```
