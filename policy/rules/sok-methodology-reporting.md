---
id: SOK.METHODOLOGY_REPORTING
slug: sok-methodology-reporting
severity: warn
locked: false
layer: domain
artifacts: [text]
phases: [writing-methods]
domains: [security, se, is]
venues: [all]
check_kind: llm_semantic
enforcement: doc
params: {}
conflicts_with: []
---

## Requirement

SoK 论文应报告文献收集与筛选方法（methodology reporting），建议至少说明：

1. 数据源（如 arXiv、DBLP、ACM DL、IEEE Xplore）
2. 检索时间窗口与关键词策略
3. 纳入/排除标准（inclusion/exclusion criteria）
4. 筛选流程（初筛、复筛、去重）

## Rationale

SoK 结论的可信度取决于语料收集过程是否可审计。方法披露不足会降低可复核性与结论稳健性。**本规则仅在 SoK profile 激活时生效**。

## Check

- **LLM 语义检查**:
  - 是否有“如何收集文献”的可执行描述
  - 是否包含筛选标准与时间范围
  - 是否避免“只说做了 review，但没有流程细节”
- **允许简化**: 篇幅受限时可给 condensed version，并在附录补充细节

## Examples

### Pass

```latex
\paragraph{Survey Methodology.}
We queried arXiv, DBLP, and ACM DL for 2019--2025 using
("machine unlearning" OR "certified removal") AND ("federated" OR "distributed").
After deduplication, 312 papers remained. We excluded non-peer-reviewed position
papers and retained 94 papers after full-text screening.
```

### Fail

```latex
We conducted a comprehensive literature survey and selected representative papers.
% 缺少数据源、时间窗口、检索策略、筛选标准与流程
```
